#!/usr/bin/env python3
"""Benchmark the python reference pipeline against the plugin, on the same pages.

Two pipelines are timed on one image list:

* ``nmanga``    - ``nmanga.autolevel`` on Pillow images, with numpy and scipy,
                  which is what ``IMPLEMENTATIONS.md`` describes as the source of
                  the behaviour
* ``vapoursynth`` - ``imgseqs.Read(..., mismatch=True)`` through
                  ``resize.Bicubic``, ``PeakStats``, ``Levels`` and ``Posterize``

Both run in their own process so peak resident memory is comparable. Pass
``--write`` to refresh the results block of ``docs/BENCH.md``.

``sandbox/`` is a private working tree and is not committed.

    uv run --extra golden --extra dev-tests tools/bench.py
    uv run --extra golden --extra dev-tests tools/bench.py --write docs/BENCH.md
"""

from __future__ import annotations

import argparse
import json
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

IMAGE_SUFFIXES = (".jpg", ".jpeg", ".png", ".webp", ".avif", ".jxl", ".tif", ".tiff", ".bmp")

#: Suites under `sandbox/`, and the workflows that make sense for each.
SUITES = {
    "levels": (REPO_ROOT / "sandbox" / "level-check", ("levels",)),
    "posterize": (REPO_ROOT / "sandbox" / "posterize-check", ("shades", "posterize")),
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
        stages.levels.append((path.name, black, white))

    return stages.as_json(len(images), {"pipeline": "nmanga", "workflow": workflow})


# ---------------------------------------------------------------------------
# the plugin pipeline
# ---------------------------------------------------------------------------


def run_vapoursynth(images: list[Path], workflow: str, cache_mb: int | None, warmup: int) -> dict:
    import vapoursynth as vs

    core = vs.core
    if cache_mb is not None:
        core.max_cache_size = cache_mb

    # `mismatch=True` is what lets one clip hold pages of different sizes, which
    # is why the filters have to read every dimension from the frame.
    source = core.imgseqs.Read(files=[str(path) for path in images], mismatch=True, prefetch=0)
    # The documented normalisation: bring the sequence to a constant format.
    # Dimensions still vary, so the clip stays variable sized.
    gray = core.resize.Bicubic(source, format=vs.GRAY8, matrix_s="470bg", range_s="full")

    if workflow == "levels":
        head = core.nimages.PeakStats(
            gray,
            upper_limit=UPPER_LIMIT,
            peak_percentage=PEAK_PERCENTAGE,
            skip_white=1,
        )
        tail = core.nimages.Levels(head, use_props=True, peak_offset=0, auto_gamma=True)
    elif workflow == "shades":
        head = core.nimages.PeakGrayShades(gray, threshold=SHADE_THRESHOLD)
        tail = None
    elif workflow == "posterize":
        head = core.nimages.Posterize(gray, bits=POSTERIZE_BITS)
        tail = None
    else:
        raise SystemExit(f"unknown workflow {workflow}")

    def process(index: int) -> tuple[dict[str, float], int, int]:
        """One page, with the wall time of each stage it runs."""
        stage: dict[str, float] = {}

        start = time.perf_counter()
        gray.get_frame(index)
        stage["decode"] = time.perf_counter() - start

        start = time.perf_counter()
        frame = head.get_frame(index)
        stage["analyze" if workflow != "posterize" else "apply"] = time.perf_counter() - start

        black = 0
        white = 0
        if workflow == "levels":
            black = int(frame.props["NImagesBlackLevel"])
            white = int(frame.props["NImagesWhiteLevel"])
        elif workflow == "shades":
            black = len(frame.props["NImagesGrayShades"])
        frame = None

        if tail is not None:
            start = time.perf_counter()
            tail.get_frame(index)
            stage["apply"] = time.perf_counter() - start

        return stage, black, white

    for index in range(min(warmup, len(images))):
        process(index)

    stages = Stages()
    for index in range(len(images)):
        stage, black, white = process(index)
        for name, elapsed in stage.items():
            stages.add(name, elapsed)
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
    cache_mb: int | None,
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

    lines.append("| pages | workflow | pipeline | decode | analyze | apply | total | per page | peak rss |")
    lines.append("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (suite, workflow), pipelines in grouped.items():
        order = ["nmanga", "vapoursynth"]
        for pipeline in order:
            row = pipelines.get(pipeline)
            if row is None:
                continue
            stage = row["seconds"]
            lines.append(
                "| {pages} | {workflow} ({suite}) | {pipeline} | {decode} s | {analyze} s | "
                "{apply} s | **{total} s** | {per_page} ms | {rss} MiB |".format(
                    pages=row["pages"],
                    workflow=workflow,
                    suite=suite,
                    pipeline=pipeline,
                    decode=seconds(stage.get("decode", 0.0)),
                    analyze=seconds(stage.get("analyze", 0.0)),
                    apply=seconds(stage.get("apply", 0.0)),
                    total=seconds(row["total"]),
                    per_page=f"{row['total'] / max(row['pages'], 1) * 1000:.1f}",
                    rss=mib(row["peak_rss"]),
                )
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
    parser.add_argument("--cache", type=int, help="VapourSynth frame cache in MiB")
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
