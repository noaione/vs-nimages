#!/usr/bin/env python3
"""Compare `Levels(gpu=1)` and `Posterize(gpu=1)` against their host paths.

Both filters are one integer table lookup per sample, so the device path has to
be byte-identical, not close: any difference is a bug in the table upload, the
plane walk or the packed word a sample is written through. This harness builds a
fixed case table, pulls every frame through both engines and compares the plane
bytes exactly.

`--write` records a hash table for the host path, so a later build can prove the
CPU side did not move either; `--check` compares against it. The device
comparison is the gate for the port:

    .venv\\Scripts\\python.exe tools\\map_hash.py --device
    .venv\\Scripts\\python.exe tools\\map_hash.py --write
    .venv\\Scripts\\python.exe tools\\map_hash.py --check

Pixels come from a numpy page written into `std.ModifyFrame`, so no image file
or writer is needed, and every case is deterministic. Planes after the first
carry a different page, so a kernel that mapped plane 0 into every plane would
show up as a difference rather than as a pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import struct
import subprocess
import sys
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = vs.core
RECORD = REPO_ROOT / "target" / "bench-images" / "map-hash.json"
FRAMES = 3


def page(width: int, height: int, frame: int) -> np.ndarray:
    """A deterministic 0..255 page with a ramp, an edge and a flat band."""
    row = np.arange(height, dtype=np.float64)[:, None]
    column = np.arange(width, dtype=np.float64)[None, :]
    codes = 16.0 + 220.0 * ((row * 3.0 + column * 7.0) % 97.0) / 96.0
    codes[:, max(width // 3, 1) :] = 12.0
    codes[height // 2 : max(height // 2, 1) + max(height // 8, 1), :] = 240.0
    codes += float((frame * 13) % 37)
    return codes


def plane_page(width: int, height: int, plane: int, frame: int) -> np.ndarray:
    """One plane's page; planes after the first differ, so a mix-up shows up."""
    codes = page(width, height, frame)
    if plane == 0:
        return codes
    return np.mod(codes * (1.0 + 0.23 * plane) + 41.0 * plane, 256.0)


def samples(codes: np.ndarray, format_: vs.VideoFormat) -> np.ndarray:
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
            view = np.asarray(out[plane])
            height, width = view.shape
            view[...] = samples(plane_page(width, height, plane, n), out.format)
        return out

    return CORE.std.ModifyFrame(blank, blank, fill)


@dataclass(frozen=True)
class Case:
    """One clip and the arguments applied to it."""

    name: str
    build: Callable[[], vs.VideoNode]
    arguments: dict[str, Any]
    #: When the device engine must refuse the case, the text the error carries.
    refused: str = field(default="")

    def node(self, device: bool) -> vs.VideoNode:
        node = self.build()
        if self.name.startswith("levels/"):
            return CORE.nimages.Levels(node, gpu=1 if device else 0, **self.arguments)
        return CORE.nimages.Posterize(node, gpu=1 if device else 0, **self.arguments)


def gray(width: int, height: int) -> vs.VideoNode:
    return source(width, height, vs.GRAY8)


def cases() -> list[Case]:
    """The fixed case table, the maps first and then the two refusals."""
    levels = {
        "defaults": {"black": 16, "white": 238, "gamma": 1.2},
        "gamma": {"black": 8, "white": 250, "gamma": 2.2},
        "identity": {"black": 0, "white": 255, "gamma": 1.0},
        "peak-offset": {"black": 16, "white": 238, "gamma": 1.2, "peak_offset": 6},
        "auto-gamma": {"black": 10, "white": 240, "auto_gamma": 1},
    }
    table: list[Case] = []
    for name, arguments in levels.items():
        table.append(Case(f"levels/gray8/{name}", lambda: gray(97, 65), dict(arguments)))
    for format_, label in ((vs.GRAY16, "gray16"), (vs.RGB24, "rgb24"), (vs.YUV420P8, "yuv420p8")):
        table.append(
            Case(f"levels/{label}/defaults", lambda f=format_: source(96, 64, f), dict(levels["defaults"]))
        )
    for name, arguments in (("gamma", levels["gamma"]), ("auto-gamma", levels["auto-gamma"])):
        table.append(
            Case(f"levels/gray16/{name}", lambda: source(129, 97, vs.GRAY16), dict(arguments))
        )
    table += [
        Case("levels/rgb24/odd-97x65", lambda: source(97, 65, vs.RGB24), dict(levels["defaults"])),
        Case("levels/yuv420p8/odd-130x66", lambda: source(130, 66, vs.YUV420P8), dict(levels["defaults"])),
        Case("levels/yuv444p16/defaults", lambda: source(96, 64, vs.YUV444P16), dict(levels["defaults"])),
        Case("levels/gray8/one-pixel", lambda: source(1, 1, vs.GRAY8), dict(levels["defaults"])),
        Case("levels/gray8/2x2", lambda: source(2, 2, vs.GRAY8), dict(levels["defaults"])),
        Case(
            "levels/gray8/use-props",
            lambda: CORE.nimages.PeakStats(gray(97, 65), upper_limit=60),
            {"use_props": 1},
        ),
        Case(
            "levels/gray16/use-props",
            lambda: CORE.nimages.PeakStats(source(129, 97, vs.GRAY16), upper_limit=60),
            {"use_props": 1},
        ),
        Case("levels/grayf/refused", lambda: source(96, 64, vs.GRAYS), dict(levels["defaults"]), "float"),
    ]

    posterize = [(4, "bits4"), (7, "bits7")]
    for bits, label in posterize:
        table.append(Case(f"posterize/gray8/{label}", lambda: gray(97, 65), {"bits": bits}))
    for bits, label in ((4, "bits4"), (12, "bits12")):
        table.append(
            Case(f"posterize/gray16/{label}", lambda: source(129, 97, vs.GRAY16), {"bits": bits})
        )
    table += [
        Case("posterize/rgb24/bits5", lambda: source(96, 64, vs.RGB24), {"bits": 5}),
        Case("posterize/yuv420p8/bits4", lambda: source(130, 66, vs.YUV420P8), {"bits": 4}),
        Case("posterize/yuv444p16/bits6", lambda: source(96, 64, vs.YUV444P16), {"bits": 6}),
        Case("posterize/gray8/identity", lambda: gray(97, 65), {"bits": 8}),
        Case("posterize/gray8/one-pixel", lambda: source(1, 1, vs.GRAY8), {"bits": 3}),
        Case(
            "posterize/gray8/use-props",
            lambda: CORE.nimages.PeakGrayShades(gray(97, 65), threshold=0.01),
            {"use_props": 1},
        ),
        Case("posterize/gray8/lloyd-refused", lambda: gray(97, 65), {"bits": 4, "method": 1}, "method=1"),
    ]
    return table


def run_case(case: Case) -> dict[str, Any]:
    """Runs one case on the host path and returns its format and hashes."""
    node = case.node(False)
    digest = hashlib.sha256()
    digest.update(b"map-hash-v1\0")
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


def compare_nodes(host: vs.VideoNode, device: vs.VideoNode) -> int:
    """Compares two nodes exactly, returning the number of differing planes."""
    bad = 0
    for n in range(host.num_frames):
        with host.get_frame(n) as left, device.get_frame(n) as right:
            if (left.width, left.height) != (right.width, right.height):
                bad += 1
                continue
            for plane in range(left.format.num_planes):
                a = np.asarray(left[plane])
                b = np.asarray(right[plane])
                if a.shape != b.shape or a.tobytes() != b.tobytes():
                    bad += 1
    return bad


def describe_device(device: dict[str, Any]) -> str:
    api = ".".join(str(part) for part in device.get("api_version", ()))
    memory = device.get("device_memory", 0) / (1024 * 1024)
    state = "usable" if device.get("usable") else "refused: " + (device.get("reason") or "no reason")
    return f"device {device['index']}: {device['name']} (Vulkan {api}, {memory:.0f} MiB) {state}"


def compare_device(table: list[Case]) -> int:
    """Compares `gpu=1` against `gpu=0` on every case, or skips with a reason."""
    try:
        devices = list(CORE.vulkan_devices)
    except Exception as error:
        print(f"device: skipped, the core has no Vulkan device list ({error})")
        return 0
    for device in devices:
        print(describe_device(device))
    usable = [device for device in devices if device.get("usable")]
    if not usable:
        print("device: skipped, no usable Vulkan 1.4 device")
        return 0
    CORE.set_vulkan_device(usable[0]["index"])
    print(f"device: pinned {usable[0]['index']} {usable[0]['name']}")

    failures = 0
    skipped = 0
    for case in table:
        try:
            host = case.node(False)
        except Exception as error:
            print(f"{case.name:<34} skip ({error})")
            skipped += 1
            continue
        if case.refused:
            try:
                case.node(True)
            except Exception as error:
                if case.refused in str(error):
                    print(f"{case.name:<34} refused as intended")
                else:
                    print(f"{case.name:<34} refused with the wrong message: {error}")
                    failures += 1
            else:
                print(f"{case.name:<34} accepted, which it must not")
                failures += 1
            continue
        try:
            device = CORE.std.GPUDownload(case.node(True))
        except Exception as error:
            print(f"{case.name:<34} skip ({error})")
            skipped += 1
            continue
        differences = compare_nodes(host, device)
        print(f"{case.name:<34} {'MISMATCH (' + str(differences) + ')' if differences else 'identical'}")
        failures += bool(differences)

    if failures:
        print(f"{failures} case(s) differ between gpu=0 and gpu=1", file=sys.stderr)
        return 1
    note = f" ({skipped} skipped)" if skipped else ""
    print(f"gpu=1 is byte-identical to gpu=0 on every case{note}")
    return 0


def print_table(results: list[dict[str, Any]]) -> None:
    print(f"{'case':<34} {'format':<10} {'frames':>6}  sha256")
    for result in sorted(results, key=lambda entry: entry["name"]):
        print(
            f"{result['name']:<34} {result['format'] or 'undefined':<10}"
            f" {len(result['frames']):>6}  {result['sha256']}"
        )


def shown_path(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except ValueError:
        return str(path)


def environment() -> dict[str, Any]:
    return {
        "python": sys.version.split()[0],
        "vapoursynth": str(vs.__version__),
        "numpy": np.__version__,
        "platform": platform.platform(),
        "byteorder": sys.byteorder,
    }


def git(*arguments: str) -> str | None:
    try:
        return subprocess.check_output(
            ["git", *arguments], cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def write_table(path: Path, results: list[dict[str, Any]]) -> int:
    payload = {
        "generator": "tools/map_hash.py",
        "version": 1,
        "source_revision": {
            "head": git("rev-parse", "HEAD"),
            "subject": git("log", "-1", "--format=%s"),
            "dirty": bool(git("status", "--porcelain")),
        },
        "environment": environment(),
        "cases": sorted(results, key=lambda entry: entry["name"]),
    }
    text = json.dumps(payload, indent=2, ensure_ascii=False) + "\n"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(text.encode("utf-8"))
    print(f"wrote {shown_path(path)} ({len(text.encode('utf-8')):,} bytes)")
    return 0


def check_table(path: Path, results: list[dict[str, Any]]) -> int:
    if not path.is_file():
        print(f"no recorded table at {shown_path(path)}; run --write first", file=sys.stderr)
        return 1
    recorded = json.loads(path.read_text(encoding="utf-8"))
    expected = {case["name"]: case for case in recorded.get("cases", [])}
    actual = {result["name"]: result for result in results}
    problems = [
        f"{name}: no longer produced"
        for name in sorted(set(expected) - set(actual))
    ]
    for name in sorted(set(actual) - set(expected)):
        problems.append(f"{name}: not in the recorded table")
    for name in sorted(set(expected) & set(actual)):
        want, got = expected[name], actual[name]
        if want.get("sha256") != got["sha256"]:
            problems.append(f"{name}: recorded {want.get('sha256')} actual {got['sha256']}")
        elif want.get("arguments") != got["arguments"]:
            problems.append(f"{name}: arguments changed")
    if problems:
        for problem in problems:
            print(problem, file=sys.stderr)
        return 1
    print(f"{len(actual)} cases match {shown_path(path)}")
    if recorded.get("environment") != environment():
        print("note: the interpreter or numpy version differs; the pixels are unchanged")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="record the host table under target/")
    mode.add_argument("--check", action="store_true", help="fail when a recorded hash differs")
    mode.add_argument("--device", action="store_true", help="compare gpu=1 against gpu=0")
    parser.add_argument("--output", type=Path, default=RECORD, help="recorded table path")
    args = parser.parse_args(argv)

    table = sorted(cases(), key=lambda case: case.name)
    if args.device:
        return compare_device(table)

    results = []
    for case in table:
        try:
            results.append(run_case(case))
        except Exception as error:  # the two refused cases have no host output either
            print(f"{case.name:<34} skip ({error})")
    print_table(results)
    if args.write:
        return write_table(args.output, results)
    if args.check:
        return check_table(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
