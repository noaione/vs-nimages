#!/usr/bin/env python3
"""Hash every Deblur case's output planes, so two builds compare byte for byte.

`nimages.Deblur` is a parallel filter with a strict spatial dependency, so a
change that keeps the algorithm can still move one byte. This script runs a
fixed case table through the installed plugin, hashes the output plane bytes and
prints one line per case, which makes two builds diffable with `diff`. It is the
harness for the device port: once `Deblur` gains `gpu:int:opt`, `--device`
compares `gpu=1` through `std.GPUDownload` against `gpu=0` on the same source.

The table covers both methods, `GRAY8`/`GRAY16`/`GRAYS`, an odd width, a 1 px
frame, a frame whose row padding is not zero, a variable-dimension clip, and one
non-default value of every argument. Pixels come from a numpy page written into
`std.ModifyFrame`, so no image file or writer is needed, and every case is
deterministic.

Run it with the development interpreter after the plugin is installed::

    uv sync --locked --extra dev --extra dev-tests
    .venv\\Scripts\\python.exe tools\\deblur_hash.py --write
    .venv\\Scripts\\python.exe tools\\deblur_hash.py --check
    .venv\\Scripts\\python.exe tools\\deblur_hash.py --device

`--write` records the table and the source revision under
`target/bench-images/deblur-hash.json`, which the tree ignores. `--check`
recomputes the table, fails on any case whose hash or arguments moved, and
reports a revision difference as a note because the pixels are what it compares.
`--device` exits nonzero when a device frame differs from its CPU equivalent
beyond what the format allows (8 bit exact, deeper integer one LSB, float the
fixture's 1/255), and exits 0 with a `device: skipped` line when the core has no
usable device or the filter has no `gpu` argument yet.

numpy is the only dependency beyond the plugin, so the dev-tests extra is enough.
"""

from __future__ import annotations

import argparse
import ctypes
import hashlib
import json
import platform
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parent.parent
#: The core every case runs on. The plugin comes from the venv the interpreter is in.
CORE = vs.core
#: The recorded table. `target/` is git-ignored, so this never reaches the tree.
RECORD = REPO_ROOT / "target" / "bench-images" / "deblur-hash.json"
#: Frames per case. Three catch a workspace handed between frames out of order.
FRAMES = 3
#: Byte written into the unused end of a row, so an over-read cannot see zero.
PADDING_BYTE = 0xA5
#: The value that tells a variable-dimension clip apart from a fixed-size one.
VARIABLE_SIZES = ((32, 16), (17, 9), (1, 1))


# ---------------------------------------------------------------------------
# Source clips
# ---------------------------------------------------------------------------


def page(width: int, height: int, frame: int) -> np.ndarray:
    """A deterministic 0..255 page with edges, a thin line and flat areas.

    The structures are placed relative to the frame's own size, so the 1 px case
    still goes through every branch instead of special-casing it.
    """
    row = np.arange(height, dtype=np.float64)[:, None]
    column = np.arange(width, dtype=np.float64)[None, :]
    codes = 40.0 + 80.0 * (row / max(height - 1, 1)) + 60.0 * (column / max(width - 1, 1))

    # A hard vertical edge with a thin line beside it.
    edge = max(width // 3, 1)
    codes[:, edge : edge + max(width // 16, 2)] = 10.0

    # A bright horizontal band, which the unsharp mask pushes against.
    middle = height // 2
    codes[middle : middle + max(height // 8, 2), :] = 245.0

    # A checker that survives the kernels' three tap window.
    codes[((column // 4) + (row // 4)) % 2 == 0] += 15.0

    # One dark line and one saturated strip.
    line = max(height // 4, 1)
    codes[line - 1 : line + 1, :] = 0.0
    codes[:, max(width - width // 5, 1) :] = 255.0

    # Every frame differs, so a workspace reused across frames shows up.
    codes += float((frame * 17) % 41)
    return codes


def samples(codes: np.ndarray, format_: vs.VideoFormat) -> np.ndarray:
    """The page as one plane's samples."""
    if format_.sample_type == vs.SampleType.FLOAT:
        return (codes / 255.0).astype(np.float32)
    maximum = (1 << format_.bits_per_sample) - 1
    values = np.rint(codes * (maximum / 255.0)).clip(0.0, float(maximum))
    if format_.bits_per_sample <= 8:
        return values.astype(np.uint8)
    return values.astype(np.uint16)


def source(width: int, height: int, format_: vs.VideoFormat, length: int = FRAMES) -> vs.VideoNode:
    """A clip whose frames hold `page` in the given format."""
    blank = CORE.std.BlankClip(width=width, height=height, format=format_, length=length)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        for plane in range(out.format.num_planes):
            np.asarray(out[plane])[...] = samples(page(out.width, out.height, n), out.format)
        return out

    return CORE.std.ModifyFrame(blank, blank, fill)


def padded(width: int, height: int, format_: vs.VideoFormat, length: int = FRAMES) -> vs.VideoNode:
    """A cropped frame whose row padding holds a sentinel instead of zero.

    VapourSynth aligns each row, so a 129 px frame gets a longer stride and bytes
    past the pixels that a kernel reading `stride` instead of `width` would see.
    `std.Crop` allocates a fresh frame; the modify callback then fills the unused
    end of every row, so the case is guaranteed non-zero padding rather than
    whatever the allocator left there.
    """
    cropped = CORE.std.Crop(source(width + 31, height, format_, length), right=31)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        del n
        out = f.copy()
        for plane in range(out.format.num_planes):
            stride = out.get_stride(plane)
            row_bytes = out.width * out.format.bytes_per_sample
            if stride <= row_bytes:
                raise AssertionError(
                    f"the padded case needs stride > row bytes, got {stride} for {row_bytes}"
                )
            pointer = out.get_write_ptr(plane).value
            for row in range(out.height):
                start = row * stride + row_bytes
                ctypes.memset(pointer + start, PADDING_BYTE, stride - row_bytes)
        return out

    return CORE.std.ModifyFrame(cropped, cropped, fill)


def variable() -> vs.VideoNode:
    """A clip whose frames report their own dimensions."""
    clips = [source(width, height, vs.GRAY8, length=2) for width, height in VARIABLE_SIZES]
    return CORE.std.Splice(clips, mismatch=True)


# ---------------------------------------------------------------------------
# Cases
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Case:
    """One source clip and the `Deblur` arguments applied to it."""

    name: str
    arguments: dict[str, Any]
    build: Callable[[], vs.VideoNode]


def cases() -> list[Case]:
    """The fixed case table, grouped by method and then by what the case shows."""
    return [
        Case("method0/gray8/defaults", {"method": 0}, lambda: source(96, 64, vs.GRAY8)),
        Case("method1/gray8/defaults", {"method": 1}, lambda: source(96, 64, vs.GRAY8)),
        Case("method0/gray16/defaults", {"method": 0}, lambda: source(96, 64, vs.GRAY16)),
        Case("method1/gray16/defaults", {"method": 1}, lambda: source(96, 64, vs.GRAY16)),
        Case("method0/grayf/defaults", {"method": 0}, lambda: source(96, 64, vs.GRAYS)),
        Case("method1/grayf/defaults", {"method": 1}, lambda: source(96, 64, vs.GRAYS)),
        Case("method0/gray8/odd-129x97", {"method": 0}, lambda: source(129, 97, vs.GRAY8)),
        Case("method1/gray8/odd-129x97", {"method": 1}, lambda: source(129, 97, vs.GRAY8)),
        Case("method0/gray8/one-pixel", {"method": 0}, lambda: source(1, 1, vs.GRAY8)),
        Case("method1/grayf/one-pixel", {"method": 1}, lambda: source(1, 1, vs.GRAYS)),
        Case("method0/gray8/padded-stride", {"method": 0}, lambda: padded(129, 97, vs.GRAY8)),
        Case("method1/gray8/padded-stride", {"method": 1}, lambda: padded(129, 97, vs.GRAY8)),
        Case("method0/gray8/variable-dims", {"method": 0}, variable),
        Case("method1/gray8/variable-dims", {"method": 1}, variable),
        Case(
            "method0/gray8/radius-max",
            {"method": 0, "radius": 16.0},
            lambda: source(64, 48, vs.GRAY8),
        ),
        Case(
            "method1/gray8/radius",
            {"method": 1, "radius": 1.5},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method0/gray8/strength",
            {"method": 0, "strength": 0.42},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method1/gray8/strength",
            {"method": 1, "strength": 1.1},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method0/gray8/threshold",
            {"method": 0, "threshold": 8.0},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method1/gray8/threshold",
            {"method": 1, "threshold": 0.0},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method0/gray8/overshoot",
            {"method": 0, "overshoot": 3.0},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method1/gray8/overshoot",
            {"method": 1, "overshoot": 5.5},
            lambda: source(96, 64, vs.GRAY8),
        ),
        Case(
            "method0/gray16/combined",
            {"method": 0, "radius": 2.0, "strength": 0.9, "iterations": 3, "threshold": 1.0, "overshoot": 1.0},
            lambda: source(96, 64, vs.GRAY16),
        ),
        Case(
            "method1/grayf/combined",
            {"method": 1, "radius": 2.5, "strength": 0.7, "threshold": 5.0, "overshoot": 2.0},
            lambda: source(96, 64, vs.GRAYS),
        ),
    ]


# ---------------------------------------------------------------------------
# Hashing
# ---------------------------------------------------------------------------


def run_case(case: Case) -> dict[str, Any]:
    """Runs one case and returns its format, per-frame geometry and hashes."""
    node = CORE.nimages.Deblur(case.build(), **case.arguments)
    digest = hashlib.sha256()
    digest.update(b"deblur-hash-v1\0")
    digest.update(case.name.encode("ascii"))

    frames: list[dict[str, Any]] = []
    format_name = ""
    for n in range(node.num_frames):
        with node.get_frame(n) as frame:
            format_name = frame.format.name
            frame_digest = hashlib.sha256()
            frame_digest.update(struct.pack("<ii", frame.width, frame.height))
            digest.update(struct.pack("<ii", frame.width, frame.height))
            for plane in range(frame.format.num_planes):
                data = np.asarray(frame[plane]).tobytes()
                frame_digest.update(struct.pack("<Q", len(data)))
                frame_digest.update(data)
                digest.update(struct.pack("<Q", len(data)))
                digest.update(data)
            frames.append(
                {
                    "frame": n,
                    "width": frame.width,
                    "height": frame.height,
                    "sha256": frame_digest.hexdigest(),
                }
            )

    return {
        "name": case.name,
        "arguments": case.arguments,
        "format": format_name,
        "frames": frames,
        "sha256": digest.hexdigest(),
    }


def compare_nodes(host: vs.VideoNode, device: vs.VideoNode) -> tuple[int, float]:
    """Compares two nodes frame by frame, returning (bad planes, worst sample gap).

    8 bit planes are compared byte for byte. Deeper integer planes allow one
    LSB: the device writes through an f32 multiply where the host uses f64. Float
    planes allow the frozen fixture's 1/255, which a 1 ULP code difference is far
    inside. A plane over its bound counts as a failure.
    """
    bad = 0
    worst = 0.0
    for n in range(host.num_frames):
        with host.get_frame(n) as left, device.get_frame(n) as right:
            if (left.width, left.height) != (right.width, right.height):
                bad += 1
                continue
            for plane in range(left.format.num_planes):
                a = np.asarray(left[plane])
                b = np.asarray(right[plane])
                if a.tobytes() == b.tobytes():
                    continue
                if a.dtype == np.uint8:
                    bad += 1
                    continue
                diff = np.abs(a.astype(np.float64) - b.astype(np.float64))
                worst = max(worst, float(diff.max()) if diff.size else 0.0)
                limit = 1.0 if a.dtype == np.uint16 else 1.0 / 255.0
                if diff.size and float(diff.max()) > limit:
                    bad += 1
    return bad, worst


# ---------------------------------------------------------------------------
# Provenance
# ---------------------------------------------------------------------------


def digest_file(path: Path) -> str:
    hasher = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def source_revision() -> dict[str, Any]:
    changes = git("status", "--porcelain")
    return {
        "head": git("rev-parse", "HEAD"),
        "subject": git("log", "-1", "--format=%s"),
        "dirty": bool(changes),
        "changes": changes.splitlines() if changes else [],
    }


def plugin_files() -> list[dict[str, Any]]:
    """The installed plugin files, so a stale build is visible in the table."""
    root = Path(vs.__file__).resolve().parent / "plugins" / "nimages"
    if not root.is_dir():
        return []
    return [
        {
            "name": path.name,
            "bytes": path.stat().st_size,
            "sha256": digest_file(path),
        }
        for path in sorted(root.iterdir())
        if path.is_file()
    ]


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "vapoursynth": str(vs.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }


def record(results: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "generator": "tools/deblur_hash.py",
        "version": 1,
        "source_revision": source_revision(),
        "plugin": plugin_files(),
        "environment": environment(),
        "cases": sorted(results, key=lambda result: result["name"]),
    }


# ---------------------------------------------------------------------------
# Output and modes
# ---------------------------------------------------------------------------


def print_table(results: list[dict[str, Any]]) -> None:
    """One fixed-width line per case, sorted by name, so two builds diff cleanly."""
    print(f"{'case':<36} {'format':<8} {'geometry':<10} {'frames':>6}  sha256")
    for result in sorted(results, key=lambda result: result["name"]):
        geometries = {(frame["width"], frame["height"]) for frame in result["frames"]}
        if len(geometries) == 1:
            width, height = next(iter(geometries))
            geometry = f"{width}x{height}"
        else:
            geometry = "varies"
        print(
            f"{result['name']:<36} {result['format'] or 'undefined':<8} "
            f"{geometry:<10} {len(result['frames']):>6}  {result['sha256']}"
        )


def shown_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def write_table(path: Path, results: list[dict[str, Any]]) -> int:
    payload = record(results)
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    print(f"wrote {shown_path(path)} ({len(text.encode('utf-8')):,} bytes)")
    return 0


def note_provenance(recorded: dict[str, Any], current: dict[str, Any]) -> None:
    """Prints what moved without failing: the cases are what `--check` compares."""
    for key in ("source_revision", "plugin", "environment"):
        if recorded.get(key) != current.get(key):
            print(f"note: {key} differs from the recorded run; the pixels are unchanged")


def check_table(path: Path, results: list[dict[str, Any]]) -> int:
    if not path.is_file():
        print(f"no recorded table at {shown_path(path)}; run --write first", file=sys.stderr)
        return 1

    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = {case["name"]: case for case in recorded.get("cases", [])}
    actual = {result["name"]: result for result in results}

    problems: list[str] = []
    for name in sorted(set(expected) | set(actual)):
        if name not in expected:
            problems.append(f"case {name} is not in the recorded table")
            continue
        if name not in actual:
            problems.append(f"recorded case {name} is no longer produced")
            continue
        want, got = expected[name], actual[name]
        if want.get("sha256") != got["sha256"]:
            problems.append(f"{name}: recorded {want.get('sha256')} actual {got['sha256']}")
        elif want.get("arguments") != got["arguments"]:
            problems.append(f"{name}: arguments changed from {want.get('arguments')} to {got['arguments']}")
        elif want.get("format") != got["format"]:
            problems.append(f"{name}: format changed from {want.get('format')} to {got['format']}")

    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        print(f"{len(problems)} case(s) differ from {shown_path(path)}", file=sys.stderr)
        return 1

    print(f"{len(actual)} cases match {shown_path(path)}")
    note_provenance(recorded, record(results))
    return 0


def describe_device(device: dict[str, Any]) -> str:
    api = ".".join(str(part) for part in device.get("api_version", ()))
    memory = device.get("device_memory", 0) / (1024 * 1024)
    state = "usable" if device.get("usable") else f"refused: {device.get('reason') or 'no reason given'}"
    return f"device {device['index']}: {device['name']} ({device.get('type')}, Vulkan {api}, {memory:.0f} MiB) {state}"


def compare_device(table: list[Case]) -> int:
    """Compares `gpu=1` against `gpu=0` on every case, or skips with a reason."""
    try:
        devices = list(CORE.vulkan_devices)
    except Exception as error:  # the core itself may be built without Vulkan
        print(f"device: skipped, the core has no Vulkan device list ({error})")
        return 0

    for device in devices:
        print(describe_device(device))
    usable = [device for device in devices if device.get("usable")]
    if not usable:
        print("device: skipped, no usable Vulkan 1.4 device")
        return 0

    chosen = usable[0]
    try:
        CORE.set_vulkan_device(chosen["index"])
    except Exception as error:
        print(f"device: error, could not pin device {chosen['index']}: {error}")
        return 1
    print(f"device: pinned {chosen['index']} {chosen['name']}")

    # `gpu` arrives with the device port. Until then the filter refuses the name
    # at creation, which is a skip rather than a failure of the CPU harness.
    # A synthetic frame, so the probe asks whether the argument exists, not
    # whether this build implements the case's format.
    probe = CORE.std.BlankClip(width=8, height=8, format=vs.GRAY8, length=1)
    try:
        CORE.nimages.Deblur(probe, method=1, gpu=1)
    except Exception as error:
        message = str(error)
        if "does not take argument" in message:
            print("device: skipped, Deblur has no gpu argument yet")
            return 0
        print(f"device: error, Deblur(gpu=1) refused at creation: {message}")
        return 1

    failures = 0
    skipped = 0
    for case in table:
        clip = case.build()
        try:
            host = CORE.nimages.Deblur(clip, gpu=0, **case.arguments)
            device = CORE.std.GPUDownload(CORE.nimages.Deblur(clip, gpu=1, **case.arguments))
        except Exception as error:
            print(f"{case.name:<36} skip ({error})")
            skipped += 1
            continue
        differences, worst = compare_nodes(host, device)
        if differences:
            verdict = f"MISMATCH ({differences})"
        elif worst > 0.0:
            verdict = f"ok (max {worst:g})"
        else:
            verdict = "ok"
        print(f"{case.name:<36} {verdict}")
        failures += bool(differences)
    if failures:
        print(f"{failures} case(s) differ between gpu=0 and gpu=1", file=sys.stderr)
        return 1
    note = f" ({skipped} skipped)" if skipped else ""
    print(f"gpu=1 matches gpu=0 on every implemented case{note}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="record the table under target/")
    mode.add_argument("--check", action="store_true", help="fail when a recorded hash differs")
    mode.add_argument(
        "--device", action="store_true", help="compare gpu=1 against gpu=0 when Deblur has the argument"
    )
    parser.add_argument("--output", type=Path, default=RECORD, help="recorded table path")
    args = parser.parse_args(argv)

    table = sorted(cases(), key=lambda case: case.name)
    if args.device:
        return compare_device(table)

    results = [run_case(case) for case in table]
    print_table(results)
    if args.write:
        return write_table(args.output, results)
    if args.check:
        return check_table(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
