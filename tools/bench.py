#!/usr/bin/env python3
"""Benchmark the python reference pipeline against the plugin, on the same pages.

Two pipelines are timed on one image list:

* ``nmanga``    - ``nmanga.autolevel`` on Pillow images, with numpy and scipy,
                  which is what ``IMPLEMENTATIONS.md`` describes as the source of
                  the behaviour
* ``vapoursynth`` - ``imgseqs.Read(..., mismatch=True)`` through
                  ``resize.Bicubic``, ``PeakStats``, ``Levels``, ``Posterize``
                  and ``Deblur``

Both run in their own process so peak resident memory is comparable. The plugin
side runs with a 512 MiB frame cache, which is what a caller would set for a
manga volume and what `--cache` changes. Pass ``--write`` to refresh the results
block of ``docs/BENCH.md``.

``sandbox/`` is a private working tree and is not committed.

    uv run --extra golden --extra dev-tests tools/bench.py
    uv run --extra golden --extra dev-tests tools/bench.py --write docs/BENCH.md
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_NMANGA_PATH = REPO_ROOT.parent / "nao-manga-rls"
FIXTURES = REPO_ROOT / "tests" / "fixtures"

# The default `upper_limit`, `peak_percentage` and `skip_white` the nmanga
# orchestrator action uses, so both pipelines run the same operation.
UPPER_LIMIT = 60
PEAK_PERCENTAGE = 0.25
SHADE_THRESHOLD = 0.01
POSTERIZE_BITS = 4


# The deconvolution `Deblur` runs by default, so both pipelines sharpen the same
# way.
DEBLUR_METHOD = 0
DEBLUR_RADIUS = 0.8
DEBLUR_STRENGTH = 0.65
DEBLUR_ITERATIONS = 6
DEBLUR_THRESHOLD = 2.0
DEBLUR_OVERSHOOT = 0.0

#: The strength `method=1` defaults to, which is not the deconvolution's.
DEBLUR_UNSHARP_STRENGTH = 0.85

#: The deblur workflows, which `deblur_parameters` resolves.
DEBLUR_WORKFLOWS = ("deblur", "deblur-unsharp")


def deblur_parameters(workflow: str) -> tuple[int, float]:
    """The method and the strength one deblur workflow runs."""
    if workflow == "deblur-unsharp":
        return 1, DEBLUR_UNSHARP_STRENGTH
    return DEBLUR_METHOD, DEBLUR_STRENGTH


IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".jxl", ".tif", ".tiff", ".bmp")

# Frame cache the plugin side runs with, in MiB. The core default is far larger
# than a caller needs; this is enough to hold a whole manga volume of SD to
# HD-ish pages as GRAY8, which is what a real script would set.
DEFAULT_CACHE_MB = 512

#: Suites under `sandbox/`, and the workflows that make sense for each.
SUITES = {
    "levels": (REPO_ROOT / "sandbox" / "level-check", ("levels",)),
    "webp": (REPO_ROOT / "sandbox" / "level-webp-check", ("levels",)),
    "posterize": (
        REPO_ROOT / "sandbox" / "posterize-check",
        ("shades", "posterize", "deblur", "deblur-unsharp"),
    ),
}


# ---------------------------------------------------------------------------
# measurement
# ---------------------------------------------------------------------------


def peak_rss_bytes() -> int:
    """Highest resident set the process has reached."""
    if sys.platform == "win32":
        import ctypes
        from ctypes import wintypes

        class Counters(ctypes.Structure):
            _fields_ = [
                ("cb", wintypes.DWORD),
                ("PageFaultCount", wintypes.DWORD),
                ("PeakWorkingSetSize", ctypes.c_size_t),
                ("WorkingSetSize", ctypes.c_size_t),
                ("QuotaPeakPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPagedPoolUsage", ctypes.c_size_t),
                ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t),
                ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
                ("PagefileUsage", ctypes.c_size_t),
                ("PeakPagefileUsage", ctypes.c_size_t),
            ]

        # argtypes is not optional. Without it ctypes truncates the frame
        # handle to an int on 64 bit and the call fails without raising.
        kernel32 = ctypes.WinDLL("kernel32", use_last_error=True)
        query = getattr(kernel32, "K32GetProcessMemoryInfo", None)
        if query is None:
            query = ctypes.WinDLL("psapi", use_last_error=True).GetProcessMemoryInfo
        query.argtypes = [wintypes.HANDLE, ctypes.POINTER(Counters), wintypes.DWORD]
        query.restype = wintypes.BOOL

        counters = Counters()
        counters.cb = ctypes.sizeof(counters)
        if not query(kernel32.GetCurrentProcess(), ctypes.byref(counters), counters.cb):
            return 0
        return int(counters.PeakWorkingSetSize)

    import resource

    return int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss) * 1024


class Stages:
    """Wall seconds spent in each stage of one pipeline."""

    def __init__(self) -> None:
        self.seconds: dict[str, float] = {}
        self.levels: list[tuple[str, int, int]] = []

    def add(self, name: str, elapsed: float) -> None:
        self.seconds[name] = self.seconds.get(name, 0.0) + elapsed

    def total(self) -> float:
        return sum(self.seconds.values())

    def as_json(self, pages: int, extra: dict) -> dict:
        return {
            "pages": pages,
            "seconds": self.seconds,
            "total": self.total(),
            "peak_rss": peak_rss_bytes(),
            "levels": self.levels,
            **extra,
        }


def image_files(directory: Path, limit: int | None) -> list[Path]:
    files = sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES)
    return files[:limit] if limit else files


# ---------------------------------------------------------------------------
# reading the plugin's own timings
# ---------------------------------------------------------------------------

#: One line the plugin writes per frame under debug=1, for example
#: `[nimages][debug] PeakStats frame 3: black=12 white=245 histogram=2.101 ms
#: peaks=0.081 ms copy=2.130 ms total=4.315 ms`. The detail between the colon
#: and `total=` is whatever that filter chose to report, so only the
#: `name=value ms` pairs are read out of it.
DEBUG_LINE = re.compile(r"\[nimages\]\[debug\] (?P<function>\w+) frame \d+: (?P<body>.*)$")
DEBUG_STAGE = re.compile(r"(\w+)=([0-9.]+) ms")

#: `vapoursynth-imageseqs` writes one of these per frame under debug=1, and its
#: nested `(open=... read=...)` group is already part of `decode=`, so only the
#: outer `total=` can be summed. Its create line has no frame number and is
#: skipped by the pattern.
IMGSEQS_TOTAL = re.compile(
    r"\[imgseqs\]\[debug\] frame \d+ '.*?' \(\w+\): .* total=(?P<total>[0-9.]+) ms"
)

#: Which report column a filter's stages belong to.
ANALYZERS = ("PeakStats", "PeakGrayShades")

#: The stage columns every result row carries, in report order.
STAGE_COLUMNS = ("decode", "resize", "analyze", "apply")


class DebugLog:
    """Collects both plugins' debug lines and adds up the stages they hold.

    `decode` comes from `vapoursynth-imageseqs`, which times its own frame build
    including the container read and the colour conversion. `analyze` and `apply`
    come from this plugin. Whatever is left of the wall time after those three is
    `resize`, which covers `resize.Bicubic` and the frame plumbing around it.
    """

    def __init__(self) -> None:
        self.lines: list[str] = []

    def __call__(self, message_type: int, message: str) -> None:
        del message_type
        text = message.strip()
        if text.startswith(("[nimages][debug]", "[imgseqs][debug]")):
            self.lines.append(text)

    def take_stages(self) -> dict[str, float]:
        """Seconds per report column for the lines collected so far."""
        columns = {"decode": 0.0, "analyze": 0.0, "apply": 0.0}
        for line in self.lines:
            source = IMGSEQS_TOTAL.search(line)
            if source is not None:
                columns["decode"] += float(source["total"]) / 1000.0
                continue

            match = DEBUG_LINE.search(line)
            if match is None:
                continue
            column = "analyze" if match["function"] in ANALYZERS else "apply"
            for name, value in DEBUG_STAGE.findall(match["body"]):
                if name == "total":
                    continue
                columns[column] += float(value) / 1000.0
        self.lines.clear()
        return columns


log = DebugLog()


# ---------------------------------------------------------------------------
# the python reference pipeline
# ---------------------------------------------------------------------------


def run_nmanga(images: list[Path], workflow: str, nmanga_path: Path, warmup: int) -> dict:
    sys.path.insert(0, str(nmanga_path))
    from nmanga.autolevel import (
        analyze_gray_shades,
        apply_levels,
        find_local_peak,
        gamma_correction,
        posterize_image_by_bits,
    )
    from nmanga.deblur import deblur_deconv, deblur_edge_sharp
    from PIL import Image

    def process(path: Path) -> tuple[dict[str, float], int, int]:
        """One page, with the wall time of each stage it runs."""
        stage: dict[str, float] = {}

        start = time.perf_counter()
        image = Image.open(path)
        # `find_local_peak` and every other entry point convert for themselves,
        # so this only forces the decode that the first use would do anyway.
        image.load()
        gray = image.convert("L")
        stage["decode"] = time.perf_counter() - start

        black = 0
        white = 0
        if workflow == "levels":
            start = time.perf_counter()
            black, white, _ = find_local_peak(
                image,
                upper_limit=UPPER_LIMIT,
                peak_percentage=PEAK_PERCENTAGE,
                peak_prominence=None,
                skip_white_check=True,
            )
            stage["analyze"] = time.perf_counter() - start

            start = time.perf_counter()
            adjusted = apply_levels(
                gray,
                black_point=black,
                white_point=255,
                gamma=gamma_correction(black),
            )
            stage["apply"] = time.perf_counter() - start
            adjusted.close()
        elif workflow == "shades":
            start = time.perf_counter()
            shades = analyze_gray_shades(gray, threshold=SHADE_THRESHOLD)
            stage["analyze"] = time.perf_counter() - start
            black = len(shades)
        elif workflow == "posterize":
            start = time.perf_counter()
            posterized = posterize_image_by_bits(gray, POSTERIZE_BITS)
            stage["apply"] = time.perf_counter() - start
            posterized.close()
        elif workflow in DEBLUR_WORKFLOWS:
            method, strength = deblur_parameters(workflow)
            start = time.perf_counter()
            if method == 0:
                sharpened = deblur_deconv(
                    gray,
                    radius=DEBLUR_RADIUS,
                    strength=strength,
                    iterations=DEBLUR_ITERATIONS,
                    threshold=DEBLUR_THRESHOLD,
                    overshoot=DEBLUR_OVERSHOOT,
                )
            else:
                sharpened = deblur_edge_sharp(
                    gray,
                    radius=DEBLUR_RADIUS,
                    strength=strength,
                    threshold=DEBLUR_THRESHOLD,
                    overshoot=DEBLUR_OVERSHOOT,
                )
            stage["apply"] = time.perf_counter() - start
            sharpened.close()
        else:
            raise SystemExit(f"unknown workflow {workflow}")

        gray.close()
        image.close()
        return stage, black, white

    # numpy, scipy and Pillow all pay a first-call cost of about a second, which
    # would otherwise land on the first few pages.
    for path in images[:warmup]:
        process(path)

    stages = Stages()
    for path in images:
        stage, black, white = process(path)
        for name, elapsed in stage.items():
            stages.add(name, elapsed)
        if workflow not in DEBLUR_WORKFLOWS:
            stages.levels.append((path.name, black, white))

    return stages.as_json(len(images), {"pipeline": "nmanga", "workflow": workflow})


# ---------------------------------------------------------------------------
# the plugin pipeline
# ---------------------------------------------------------------------------


def to_gray8(source, vs, core):
    """Normalises a sequence to a constant `GRAY8` the way a caller should.

    An RGB source, which is what `vapoursynth-imageseqs` hands out for a JPEG or
    a PNG, converts straight to Gray with the `470bg` luma the reference uses.

    A YUV source, which is what it hands out for a lossy WebP, goes through RGB
    first. Taking the luma plane directly is cheaper but wrong for this
    comparison: the decoder's Y is limited range, so a page whose luma sits at 32
    arrives as 43, and the reference is analysing the full range 32 that Pillow
    computed from the RGB. The round trip matches the reference to within 0.01 of
    a code value.

    The odd edge comes off first, because zimg cannot convert an odd sized 4:2:0
    frame. That and the per-frame form below are the workaround
    `vapoursynth-imageseqs` documents; it uses `FrameEval` because a clip whose
    format varies cannot be judged at the node, and a clip whose format does not
    can be trimmed once instead.
    """
    if source.format.color_family == vs.RGB:
        return core.resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")
    if source.format.color_family == vs.YUV:
        return trim_and_convert(source, vs, core, source, source.width, source.height)

    # The clip varies, so the node reports an Undefined format and cannot say
    # whether its frames are RGB. One probe answers it, and a single resize is
    # both enough and cheaper than the per-frame form when the answer is yes.
    probe = source.get_frame(0)
    if probe.format.color_family == vs.RGB:
        return core.resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    # A sequence mixing RGB and YUV frames has to be converted one frame at a
    # time. What comes back has to be one format, so every branch returns Gray8
    # and a final resize declares it.
    plain_gray = core.resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    def convert(n=0, **_):
        frame = source.get_frame(n)
        if frame.format.color_family == vs.RGB:
            return plain_gray
        return trim_and_convert(source, vs, core, source, frame.width, frame.height)

    gray = core.std.FrameEval(source, convert)
    return core.resize.Bicubic(gray, format=vs.GRAY8)


def trim_and_convert(source, vs, core, clip, width: int, height: int):
    """Converts one YUV clip or sub-clip to `GRAY8` through RGB, trimming an odd
    edge off first and putting it back as black."""
    right = width % 2 if clip.format.subsampling_w else 0
    bottom = height % 2 if clip.format.subsampling_h else 0
    region = clip
    if right or bottom:
        region = core.std.CropAbs(clip, width=width - right, height=height - bottom)

    # No matrix or range arguments: the frame properties carry both, and they are
    # what make the round trip match the reference.
    rgb = core.resize.Bicubic(region, format=vs.RGB24)
    gray = core.resize.Bicubic(rgb, format=vs.GRAY8, matrix_s="470bg", range_s="full")
    if right or bottom:
        gray = core.std.AddBorders(gray, right=right, bottom=bottom)
    return gray


def run_vapoursynth(images: list[Path], workflow: str, cache_mb: int, warmup: int) -> dict:
    import vapoursynth as vs

    core = vs.core
    core.max_cache_size = cache_mb
    core.add_log_handler(log)

    # `mismatch=True` is what lets one clip hold pages of different sizes, which
    # is why the filters have to read every dimension from the frame.
    source = core.imgseqs.Read(
        files=[str(path) for path in images], mismatch=True, prefetch=0, debug=1
    )
    gray = to_gray8(source, vs, core)

    # Every filter reports its own stage times when `debug=1`. VapourSynth does
    # not keep an intermediate frame between two external requests, so pulling
    # `head` and then `tail` would analyse every page twice and count it twice.
    # Pulling only the last node and reading the stages from its own clock is
    # exact: the total is this process's wall time, and the stages are the
    # filters' own.
    if workflow == "levels":
        head = core.nimages.PeakStats(
            gray,
            upper_limit=UPPER_LIMIT,
            peak_percentage=PEAK_PERCENTAGE,
            skip_white=1,
            debug=1,
        )
        final = core.nimages.Levels(head, use_props=True, peak_offset=0, auto_gamma=True, debug=1)
    elif workflow == "shades":
        final = core.nimages.PeakGrayShades(gray, threshold=SHADE_THRESHOLD, debug=1)
    elif workflow == "posterize":
        final = core.nimages.Posterize(gray, bits=POSTERIZE_BITS, debug=1)
    elif workflow in DEBLUR_WORKFLOWS:
        method, strength = deblur_parameters(workflow)
        final = core.nimages.Deblur(
            gray,
            method=method,
            radius=DEBLUR_RADIUS,
            strength=strength,
            iterations=DEBLUR_ITERATIONS,
            threshold=DEBLUR_THRESHOLD,
            overshoot=DEBLUR_OVERSHOOT,
            debug=1,
        )
    else:
        raise SystemExit(f"unknown workflow {workflow}")

    for index in range(min(warmup, len(images))):
        final.get_frame(index)

    stages = Stages()
    for index in range(len(images)):
        log.lines.clear()

        start = time.perf_counter()
        frame = final.get_frame(index)
        elapsed = time.perf_counter() - start

        # The filters report their own stages and imgseqs reports the decode,
        # so what is left of the wall time is the resize and the frame plumbing.
        reported = log.take_stages()
        stages.add("resize", max(elapsed - sum(reported.values()), 0.0))
        for name, seconds in reported.items():
            stages.add(name, seconds)

        black = 0
        white = 0
        if workflow == "levels":
            black = int(frame.props["NImagesBlackLevel"])  # pyright: ignore[reportArgumentType]
            white = int(frame.props["NImagesWhiteLevel"])  # pyright: ignore[reportArgumentType]
        elif workflow == "shades":
            black = len(frame.props["NImagesGrayShades"])  # pyright: ignore[reportArgumentType]
        if workflow not in DEBLUR_WORKFLOWS:
            stages.levels.append((images[index].name, black, white))

    return stages.as_json(
        len(images),
        {
            "pipeline": "vapoursynth",
            "workflow": workflow,
            "threads": core.num_threads,
            "cache_mb": core.max_cache_size,
        },
    )


# ---------------------------------------------------------------------------
# orchestration
# ---------------------------------------------------------------------------


def run_in_worker(
    pipeline: str,
    workflow: str,
    suite: str,
    limit: int | None,
    cache_mb: int,
    warmup: int,
    nmanga_path: Path,
) -> dict:
    directory, _ = SUITES[suite]
    images = image_files(directory, limit)
    if not images:
        raise SystemExit(f"no images under {directory}")
    if pipeline == "nmanga":
        return run_nmanga(images, workflow, nmanga_path, warmup)
    return run_vapoursynth(images, workflow, cache_mb, warmup)


def worker(args: argparse.Namespace) -> int:
    # `--suite` and `--workflow` accumulate on the parent, which passes exactly
    # one of each down to a worker.
    suite = args.suite[-1]
    workflow = args.workflow[-1]
    result = run_in_worker(
        args.pipeline, workflow, suite, args.limit, args.cache, args.warmup, args.nmanga_path
    )
    Path(args.json).write_text(json.dumps(result, indent=2), encoding="utf-8")
    return 0


def spawn(args: argparse.Namespace, pipeline: str, workflow: str, suite: str, log: Path) -> dict:
    """Runs one pipeline in its own process and returns its measurement."""
    out = log.with_suffix(f".{suite}.{workflow}.{pipeline}.json")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker",
        "--pipeline",
        pipeline,
        "--workflow",
        workflow,
        "--suite",
        suite,
        "--json",
        str(out),
        "--nmanga-path",
        str(args.nmanga_path),
    ]
    if args.limit:
        command += ["--limit", str(args.limit)]
    if args.cache is not None:
        command += ["--cache", str(args.cache)]
    command += ["--warmup", str(args.warmup)]

    print(f"  running {pipeline} / {workflow} / {suite} ...", flush=True)
    started = time.perf_counter()
    # stdout and stderr go to a file rather than a pipe, which keeps the child
    # independent of this process's stdio handling.
    with log.open("w", encoding="utf-8") as handle:
        completed = subprocess.run(command, stdout=handle, stderr=subprocess.STDOUT, check=False)
    if completed.returncode != 0 or not out.is_file():
        tail = log.read_text(encoding="utf-8", errors="replace")[-2000:]
        raise SystemExit(f"{pipeline}/{workflow}/{suite} failed:\n{tail}")

    result = json.loads(out.read_text(encoding="utf-8"))
    result["wall"] = time.perf_counter() - started
    return result


def seconds(value: float) -> str:
    return f"{value:.2f}"


def mib(value: int) -> str:
    return f"{value / (1024 * 1024):.0f}"


def report(rows: list[dict]) -> str:
    """Renders the measured rows as the results block of docs/BENCH.md."""
    lines: list[str] = []
    grouped: dict[tuple[str, str], dict[str, dict]] = {}
    for row in rows:
        grouped.setdefault((row["suite"], row["workflow"]), {})[row["pipeline"]] = row

    lines.append(
        "| pages | workflow | pipeline | "
        + " | ".join(STAGE_COLUMNS)
        + " | total | per page | peak rss |"
    )
    lines.append(
        "| ---: | --- | --- | " + " | ".join("---:" for _ in STAGE_COLUMNS) + " | ---: | ---: | ---: |"
    )
    for (suite, workflow), pipelines in grouped.items():
        order = ["nmanga", "vapoursynth"]
        for pipeline in order:
            row = pipelines.get(pipeline)
            if row is None:
                continue
            stage = row["seconds"]
            cells = " | ".join(f"{seconds(stage.get(name, 0.0))} s" for name in STAGE_COLUMNS)
            lines.append(
                f"| {row['pages']} | {workflow} ({suite}) | {pipeline} | {cells} | "
                f"**{seconds(row['total'])} s** | "
                f"{row['total'] / max(row['pages'], 1) * 1000:.1f} ms | {mib(row['peak_rss'])} MiB |"
            )

    lines.append("")
    lines.append("| pages | workflow | pipeline | speedup | memory ratio | levels agreed |")
    lines.append("| ---: | --- | --- | ---: | ---: | --- |")
    for (suite, workflow), pipelines in grouped.items():
        reference = pipelines.get("nmanga")
        candidate = pipelines.get("vapoursynth")
        if reference is None or candidate is None:
            continue
        speedup = reference["total"] / candidate["total"] if candidate["total"] else float("inf")
        ratio = candidate["peak_rss"] / reference["peak_rss"] if reference["peak_rss"] else float("inf")
        agreed, total = compare_levels(reference, candidate)
        lines.append(
            f"| {reference['pages']} | {workflow} ({suite}) | vapoursynth vs nmanga | "
            f"{speedup:.2f}x | {ratio:.2f}x | {agreed}/{total} |"
        )

    return "\n".join(lines)


def compare_levels(reference: dict, candidate: dict) -> tuple[int, int]:
    """How many pages both pipelines gave the same black and white level."""
    by_name = {name: (black, white) for name, black, white in candidate["levels"]}
    agreed = 0
    total = 0
    for name, black, white in reference["levels"]:
        if name not in by_name:
            continue
        total += 1
        if by_name[name] == (black, white):
            agreed += 1
    return agreed, total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--suite", choices=sorted(SUITES), action="append")
    parser.add_argument("--workflow", action="append")
    parser.add_argument("--limit", type=int)
    parser.add_argument(
        "--cache",
        type=int,
        default=DEFAULT_CACHE_MB,
        help=f"VapourSynth frame cache in MiB (default {DEFAULT_CACHE_MB})",
    )
    parser.add_argument("--warmup", type=int, default=1, help="pages to process before timing")
    parser.add_argument("--nmanga-path", type=Path, default=DEFAULT_NMANGA_PATH)
    parser.add_argument("--json", help="worker only: where to write the measurement")
    parser.add_argument("--worker", action="store_true")
    parser.add_argument("--pipeline", choices=["nmanga", "vapoursynth"])
    parser.add_argument("--write", type=Path, help="refresh the results block of this file")
    parser.add_argument("--keep-logs", action="store_true")
    args = parser.parse_args()

    if args.worker:
        return worker(args)

    logs = REPO_ROOT / "target" / "bench"
    logs.mkdir(parents=True, exist_ok=True)

    rows: list[dict] = []
    for suite in args.suite or sorted(SUITES):
        _, workflows = SUITES[suite]
        for workflow in args.workflow or workflows:
            for pipeline in ("nmanga", "vapoursynth"):
                result = spawn(args, pipeline, workflow, suite, logs / "bench.log")
                result["suite"] = suite
                rows.append(result)

    block = report(rows)
    print()
    print(block)

    if args.write:
        write_block(args.write, block)
        print(f"\nupdated the results block of {args.write}")

    if not args.keep_logs:
        for path in logs.glob("*.json"):
            path.unlink()
        (logs / "bench.log").unlink(missing_ok=True)

    return 0


def write_block(path: Path, block: str) -> None:
    """Replaces everything between the two bench markers."""
    start = "<!-- bench:start -->"
    end = "<!-- bench:end -->"
    text = path.read_text(encoding="utf-8")
    if start not in text or end not in text:
        raise SystemExit(f"{path} has no {start} / {end} markers")
    before, rest = text.split(start, 1)
    _, after = rest.split(end, 1)
    path.write_text(f"{before}{start}\n{block}\n{end}{after}", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
