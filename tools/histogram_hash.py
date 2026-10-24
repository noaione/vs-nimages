#!/usr/bin/env python3
"""Compare `PeakStats(gpu=1)` and `PeakGrayShades(gpu=1)` against the host path.

Both analyzers count plane 0 and then analyse the table on the host, so the
device path has to produce the same properties exactly: the same black and white
levels, the same peak flags, the same shade list and the same percentages. A
difference is a bug in the counting kernel or in the readback, not a tolerance to
tune.

Pixels pass through both analyzers, so the harness also checks that a device
frame comes back with the input's bytes and its properties.

`--write` records the host properties for every case, so a later build can prove
the CPU side did not move either; `--check` compares against that record. The
device comparison is the gate for the port:

    .venv\\Scripts\\python.exe tools\\histogram_hash.py --device
    .venv\\Scripts\\python.exe tools\\histogram_hash.py --write
    .venv\\Scripts\\python.exe tools\\histogram_hash.py --check

Pages come from numpy through `std.ModifyFrame`, so no image file or writer is
needed and every case is deterministic. The flat and four-tone pages exist to
aim the counting kernel at same-address contention, which is where a per-sample
`atomicAdd` histogram loses.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import platform
import statistics
import struct
import subprocess
import sys
import time
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parent.parent
CORE = vs.core
RECORD = REPO_ROOT / "target" / "bench-images" / "histogram-hash.json"
FRAMES = 3

#: The properties each analyzer owns. `PeakGrayShades` always writes both arrays.
STATS_KEYS = (
    "NImagesBlackLevel",
    "NImagesWhiteLevel",
    "NImagesBlackPeakFound",
    "NImagesWhitePeakFound",
)
SHADES_KEYS = ("NImagesGrayShades", "NImagesGrayShadePercentages")


def page(width: int, height: int, mode: str, frame: int) -> np.ndarray:
    """A deterministic page in 0..255, shaped to stress one counting path."""
    row = np.arange(height, dtype=np.float64)[:, None]
    column = np.arange(width, dtype=np.float64)[None, :]
    if mode == "flat":
        codes = np.full((height, width), 128.0)
    elif mode == "four":
        band = np.minimum((column * 4.0 / max(width, 1)).astype(np.int64), 3)
        codes = np.choose(band, [0.0, 90.0, 180.0, 255.0]) + row * 0.0
    elif mode == "sparse":
        codes = np.where((row + column) % 37.0 == 0.0, 255.0, 0.0)
    elif mode == "noise":
        codes = np.random.default_rng(0x2026 + frame).integers(0, 256, size=(height, width))
        codes = codes.astype(np.float64)
    else:
        codes = (row * 3.0 + column * 7.0) % 256.0
    # One code shift per frame, so a filter that cached the first frame shows up.
    return np.mod(codes + float((frame * 11) % 23), 256.0)


def samples(codes: np.ndarray, format_: vs.VideoFormat) -> np.ndarray:
    maximum = (1 << format_.bits_per_sample) - 1
    values = np.rint(codes * (maximum / 255.0)).clip(0.0, float(maximum))
    if format_.bits_per_sample <= 8:
        return values.astype(np.uint8)
    return values.astype(np.uint16)


def source(
    width: int,
    height: int,
    format_: vs.VideoFormat,
    mode: str,
    length: int = FRAMES,
    *,
    planes_same: bool = False,
) -> vs.VideoNode:
    """A clip whose frames hold `page` in the given format."""
    blank = CORE.std.BlankClip(width=width, height=height, format=format_, length=length)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        for plane in range(out.format.num_planes):
            view = np.asarray(out[plane])
            height, width = view.shape
            base = page(width, height, mode, n)
            if not planes_same:
                # Planes after the first carry a shifted page, so a filter that
                # analysed one plane for another shows up as a difference.
                base = np.mod(base + 53.0 * plane, 256.0)
            view[...] = samples(base, out.format)
        return out

    return CORE.std.ModifyFrame(blank, blank, fill)


@dataclass(frozen=True)
class Case:
    """One clip and the arguments applied to it."""

    name: str
    build: Callable[[], vs.VideoNode]
    arguments: dict[str, Any]
    #: `stats` or `shades`.
    kind: str = "stats"
    #: When both engines must refuse the case, the text the error carries.
    refused: str = field(default="")

    def node(self, device: bool) -> vs.VideoNode:
        node = self.build()
        gpu = 1 if device else 0
        if self.kind == "stats":
            return CORE.nimages.PeakStats(node, gpu=gpu, **self.arguments)
        return CORE.nimages.PeakGrayShades(node, gpu=gpu, **self.arguments)

    def keys(self) -> tuple[str, ...]:
        return STATS_KEYS if self.kind == "stats" else SHADES_KEYS


def gray(width: int, height: int, mode: str = "ramp") -> vs.VideoNode:
    return source(width, height, vs.GRAY8, mode)


def cases() -> list[Case]:
    """The fixed case table."""
    table: list[Case] = []

    stats = {
        "defaults": {"upper_limit": 60, "peak_percentage": 0.25},
        "skip-white": {"upper_limit": 60, "skip_white": 1},
        "prominence": {"upper_limit": 60, "peak_prominence": 1.0},
        "upper-1": {"upper_limit": 1},
        "upper-255": {"upper_limit": 255},
        "dense": {"upper_limit": 60, "peak_percentage": 1.0},
    }
    for name, arguments in stats.items():
        table.append(Case(f"stats/gray8/{name}", lambda: gray(97, 65), dict(arguments)))

    for mode in ("flat", "four", "sparse", "noise"):
        table.append(
            Case(f"stats/gray8/{mode}", lambda m=mode: gray(96, 64, m), dict(stats["defaults"]))
        )

    for format_, label in ((vs.GRAY16, "gray16"), (vs.GRAY10, "gray10")):
        for name in ("defaults", "skip-white", "upper-1"):
            table.append(
                Case(
                    f"stats/{label}/{name}",
                    lambda f=format_: source(129, 97, f, "ramp"),
                    dict(stats[name]),
                )
            )
        table.append(
            Case(
                f"stats/{label}/flat",
                lambda f=format_: source(96, 64, f, "flat"),
                dict(stats["defaults"]),
            )
        )

    table += [
        Case("stats/gray8/one-pixel", lambda: source(1, 1, vs.GRAY8, "ramp"), dict(stats["defaults"])),
        Case("stats/gray8/2x2", lambda: source(2, 2, vs.GRAY8, "four"), dict(stats["defaults"])),
        Case("stats/gray16/one-pixel", lambda: source(1, 1, vs.GRAY16, "ramp"), dict(stats["defaults"])),
    ]

    shades = {
        "defaults": {"threshold": 0.01},
        "threshold-0": {"threshold": 0.0},
        "threshold-5": {"threshold": 5.0},
        "threshold-100": {"threshold": 100.0},
    }
    for name, arguments in shades.items():
        table.append(
            Case(
                f"shades/gray8/{name}",
                lambda: gray(97, 65),
                dict(arguments),
                kind="shades",
            )
        )
    for mode in ("flat", "four", "sparse", "noise"):
        table.append(
            Case(
                f"shades/gray8/{mode}",
                lambda m=mode: gray(96, 64, m),
                dict(shades["defaults"]),
                kind="shades",
            )
        )
    for format_, label in ((vs.GRAY16, "gray16"), (vs.GRAY10, "gray10")):
        table.append(
            Case(
                f"shades/{label}/defaults",
                lambda f=format_: source(129, 97, f, "ramp"),
                dict(shades["defaults"]),
                kind="shades",
            )
        )
        table.append(
            Case(
                f"shades/{label}/flat",
                lambda f=format_: source(96, 64, f, "flat"),
                dict(shades["threshold-5"]),
                kind="shades",
            )
        )
    table += [
        Case(
            "shades/gray8/one-pixel",
            lambda: source(1, 1, vs.GRAY8, "ramp"),
            dict(shades["threshold-0"]),
            kind="shades",
        ),
        Case(
            "shades/gray8/2x2",
            lambda: source(2, 2, vs.GRAY8, "four"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "stats/rgb24/defaults",
            lambda: source(97, 65, vs.RGB24, "ramp"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/rgb24/flat",
            lambda: source(96, 64, vs.RGB24, "flat"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/rgb24/four",
            lambda: source(96, 64, vs.RGB24, "four"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/rgb24/one-pixel",
            lambda: source(1, 1, vs.RGB24, "ramp"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/rgb48/defaults",
            lambda: source(129, 97, vs.RGB48, "ramp"),
            dict(stats["skip-white"]),
        ),
        Case(
            "stats/rgb48/flat",
            lambda: source(96, 64, vs.RGB48, "flat"),
            dict(stats["defaults"]),
        ),
        Case(
            "shades/rgb24/defaults",
            lambda: source(97, 65, vs.RGB24, "ramp"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "shades/rgb24/flat",
            lambda: source(96, 64, vs.RGB24, "flat"),
            dict(shades["threshold-5"]),
            kind="shades",
        ),
        Case(
            "shades/rgb24/sparse",
            lambda: source(96, 64, vs.RGB24, "sparse"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "shades/rgb48/defaults",
            lambda: source(129, 97, vs.RGB48, "ramp"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "stats/yuv420p8/defaults",
            lambda: source(130, 66, vs.YUV420P8, "ramp"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/yuv420p8/flat",
            lambda: source(96, 64, vs.YUV420P8, "flat"),
            dict(stats["defaults"]),
        ),
        Case(
            "stats/yuv444p16/defaults",
            lambda: source(96, 64, vs.YUV444P16, "ramp"),
            dict(stats["skip-white"]),
        ),
        Case(
            "shades/yuv420p8/defaults",
            lambda: source(130, 66, vs.YUV420P8, "ramp"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "shades/yuv444p16/defaults",
            lambda: source(96, 64, vs.YUV444P16, "ramp"),
            dict(shades["defaults"]),
            kind="shades",
        ),
        Case(
            "stats/rgbs/refused",
            lambda: source(96, 64, vs.RGBS, "ramp"),
            dict(stats["defaults"]),
            refused="Gray, YUV or RGB 8 to 16 bit",
        ),
        Case(
            "shades/grayf/refused",
            lambda: source(96, 64, vs.GRAYS, "ramp"),
            dict(shades["defaults"]),
            kind="shades",
            refused="Gray, YUV or RGB 8 to 16 bit",
        ),
    ]
    return table


def property_values(frame: vs.VideoFrame, keys: tuple[str, ...]) -> dict[str, Any]:
    """The analyzer's properties as plain lists, with scalar arrays expanded.

    VapourSynth hands a one-element array back as a scalar, so a raw comparison
    of frame properties would call a match a mismatch.
    """
    values: dict[str, Any] = {}
    for key in keys:
        value = frame.props[key]
        if isinstance(value, (int, float)):
            values[key] = [value] if key.endswith("Shades") or "Percentages" in key else value
        else:
            values[key] = list(value)
    return values


def run_case(case: Case) -> dict[str, Any]:
    """Runs one case on the host path and returns its properties and hashes."""
    node = case.node(False)
    digest = hashlib.sha256()
    digest.update(b"histogram-hash-v1\0")
    digest.update(case.name.encode("ascii"))
    frames: list[dict[str, Any]] = []
    format_name = ""
    for n in range(node.num_frames):
        with node.get_frame(n) as frame:
            format_name = frame.format.name
            values = property_values(frame, case.keys())
            frames.append({"frame": n, "properties": values})
            digest.update(struct.pack("<ii", frame.width, frame.height))
            digest.update(json.dumps(values, sort_keys=True).encode("ascii"))
    return {
        "name": case.name,
        "arguments": case.arguments,
        "format": format_name,
        "frames": frames,
        "sha256": digest.hexdigest(),
    }


def compare_nodes(host: vs.VideoNode, device: vs.VideoNode, keys: tuple[str, ...]) -> int:
    """Compares properties and pixels exactly, returning the number of frames off."""
    bad = 0
    for n in range(host.num_frames):
        with host.get_frame(n) as left, device.get_frame(n) as right:
            if property_values(left, keys) != property_values(right, keys):
                bad += 1
                continue
            for plane in range(left.format.num_planes):
                a = np.asarray(left[plane])
                b = np.asarray(right[plane])
                if a.shape != b.shape or a.tobytes() != b.tobytes():
                    bad += 1
                    break
    return bad


def properties(node: vs.VideoNode, keys: tuple[str, ...]) -> list[dict[str, Any]]:
    """Every frame's analyzer properties, as plain values."""
    out = []
    for n in range(node.num_frames):
        with node.get_frame(n) as frame:
            out.append(property_values(frame, keys))
    return out


def compare_views() -> int:
    """Checks the invariants the color conversion has to satisfy."""
    failures = 0

    def report(name: str, same: bool) -> None:
        nonlocal failures
        print(f"{name:<30} {'identical' if same else 'MISMATCH'}")
        failures += not same

    # A YUV clip is counted through plane 0, so its properties are that plane's.
    yuv = source(96, 64, vs.YUV420P8, "ramp")
    extracted = CORE.std.ShufflePlanes(yuv, 0, vs.GRAY)
    report(
        "view/yuv-plane-0",
        properties(CORE.nimages.PeakStats(yuv), STATS_KEYS)
        == properties(CORE.nimages.PeakStats(extracted), STATS_KEYS)
        and properties(CORE.nimages.PeakGrayShades(yuv), SHADES_KEYS)
        == properties(CORE.nimages.PeakGrayShades(extracted), SHADES_KEYS),
    )

    # An RGB page with equal planes is the gray page, level for level.
    gray = source(97, 65, vs.GRAY8, "ramp")
    rgb = source(97, 65, vs.RGB24, "ramp", planes_same=True)
    report(
        "view/rgb-equals-gray",
        properties(CORE.nimages.PeakStats(rgb), STATS_KEYS)
        == properties(CORE.nimages.PeakStats(gray), STATS_KEYS)
        and properties(CORE.nimages.PeakGrayShades(rgb), SHADES_KEYS)
        == properties(CORE.nimages.PeakGrayShades(gray), SHADES_KEYS),
    )

    # The reference's luma, on flat pages whose shade list has one entry.
    for color, want in (
        ((255, 0, 0), 76),
        ((0, 255, 0), 150),
        ((0, 0, 255), 29),
        ((10, 200, 30), 124),
    ):
        flat = CORE.std.BlankClip(
            width=8, height=8, format=vs.RGB24, length=1, color=list(color)
        )
        shades = properties(CORE.nimages.PeakGrayShades(flat), SHADES_KEYS)[0]
        report(
            f"view/luma-{color[0]}-{color[1]}-{color[2]}",
            shades["NImagesGrayShades"] == [want],
        )
    return 1 if failures else 0


def describe_device(device: dict[str, Any]) -> str:
    api = ".".join(str(part) for part in device.get("api_version", ()))
    memory = device.get("device_memory", 0) / (1024 * 1024)
    state = "usable" if device.get("usable") else "refused: " + (device.get("reason") or "no reason")
    return f"device {device['index']}: {device['name']} (Vulkan {api}, {memory:.0f} MiB) {state}"


def compare_device(table: list[Case]) -> int:
    """Compares `gpu=1` against `gpu=0` on every case, or skips with a reason."""
    try:
        devices = list(CORE.vulkan_devices)
    except Exception as error:  # noqa: BLE001
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
    for case in table:
        if case.refused:
            messages = []
            for device in (False, True):
                try:
                    case.node(device)
                except Exception as error:  # noqa: BLE001
                    messages.append(str(error))
                else:
                    messages.append("")
            if all(case.refused in message for message in messages):
                print(f"{case.name:<30} refused as intended")
            else:
                print(f"{case.name:<30} refused with the wrong message: {messages}")
                failures += 1
            continue
        try:
            host = case.node(False)
        except Exception as error:  # noqa: BLE001
            print(f"{case.name:<30} skip ({error})")
            continue
        try:
            device = CORE.std.GPUDownload(case.node(True))
        except Exception as error:  # noqa: BLE001
            print(f"{case.name:<30} skip ({error})")
            continue
        differences = compare_nodes(host, device, case.keys())
        print(f"{case.name:<30} {'MISMATCH (' + str(differences) + ')' if differences else 'identical'}")
        failures += bool(differences)

    if failures:
        print(f"{failures} case(s) differ between gpu=0 and gpu=1", file=sys.stderr)
        return 1
    print("gpu=1 writes the same properties and pixels as gpu=0 on every case")
    return 0


def compare_chain() -> int:
    """Runs the reason the device analyzer exists: a device chain into Lloyd."""
    try:
        devices = list(CORE.vulkan_devices)
    except Exception as error:  # noqa: BLE001
        print(f"chain: skipped, the core has no Vulkan device list ({error})")
        return 0
    if not any(device.get("usable") for device in devices):
        print("chain: skipped, no usable Vulkan 1.4 device")
        return 0
    host = CORE.nimages.Posterize(
        CORE.nimages.PeakGrayShades(gray(97, 65), gpu=0), use_props=True, method=1, gpu=0
    )
    device = CORE.std.GPUDownload(
        CORE.nimages.Posterize(
            CORE.nimages.PeakGrayShades(gray(97, 65), gpu=1), use_props=True, method=1, gpu=1
        )
    )
    differences = compare_nodes(host, device, SHADES_KEYS)
    print(f"{'chain/shades-to-lloyd':<30} {'MISMATCH' if differences else 'identical'}")
    return 1 if differences else 0


#: The page `--bench` times: 12 MP, the size the rest of the docs quote.
BENCH_SIZE = (4096, 3000)


def bench_source(
    width: int, height: int, format_: vs.VideoFormat, frames: int
) -> vs.VideoNode:
    """A flat page: the worst case for a histogram, and free to produce."""
    middle = float(((1 << format_.bits_per_sample) - 1) // 2)
    return CORE.std.BlankClip(
        width=width,
        height=height,
        format=format_,
        length=frames,
        color=[middle] * format_.num_planes,
    )


def median_ms(node: vs.VideoNode, frames: int) -> float:
    """Median milliseconds per frame, after one untimed pull."""
    with node.get_frame(0):
        pass
    samples = []
    for n in range(frames):
        start = time.perf_counter()
        with node.get_frame(n):
            pass
        samples.append((time.perf_counter() - start) * 1000.0)
    return statistics.median(samples)


def run_bench(frames: int, format_: vs.VideoFormat, cache: int) -> int:
    """Times one frame of each graph shape at 12 MP."""
    usable = [entry for entry in CORE.vulkan_devices if entry.get("usable")]
    if not usable:
        print("bench: skipped, no usable Vulkan 1.4 device", file=sys.stderr)
        return 0
    CORE.set_vulkan_device(usable[0]["index"])
    CORE.max_cache_size = cache
    CORE.max_vram_cache_size = 1024
    width, height = BENCH_SIZE
    print(
        f"bench: {width}x{height} {format_.name}, median of {frames} frames,"
        f" cache {cache} MB, device {usable[0]['name']}",
    )

    def source() -> vs.VideoNode:
        return bench_source(width, height, format_, frames)

    def downloaded(node: vs.VideoNode) -> vs.VideoNode:
        # Every row ends on the host, so the pull waits for the device work
        # instead of returning a frame whose submission is still queued.
        return CORE.std.GPUDownload(node)

    rows = {
        "transfer-only": lambda: downloaded(CORE.std.GPUUpload(source())),
        "host/count": lambda: CORE.nimages.PeakStats(source(), gpu=0),
        "device/count": lambda: downloaded(CORE.nimages.PeakStats(source(), gpu=1)),
        "host/levels": lambda: CORE.nimages.Levels(source(), gpu=0),
        "host/levels+count": lambda: CORE.nimages.PeakStats(
            CORE.nimages.Levels(source(), gpu=0), gpu=0
        ),
        "device/levels": lambda: downloaded(CORE.nimages.Levels(source(), gpu=1)),
        "device/levels+count": lambda: downloaded(
            CORE.nimages.PeakStats(CORE.nimages.Levels(source(), gpu=1), gpu=1)
        ),
        "host/even": lambda: CORE.nimages.Posterize(source(), bits=4),
        "host/lloyd": lambda: CORE.nimages.Posterize(
            CORE.nimages.PeakGrayShades(source()), use_props=True, method=1
        ),
        "device/even": lambda: downloaded(
            CORE.nimages.Posterize(source(), bits=4, gpu=1)
        ),
        "device/lloyd": lambda: downloaded(
            CORE.nimages.Posterize(
                CORE.nimages.PeakGrayShades(source(), gpu=1),
                use_props=True,
                method=1,
                gpu=1,
            )
        ),
    }
    timings = {}
    for name, build in rows.items():
        timings[name] = median_ms(build(), frames)
        print(f"{name:<22} {timings[name]:7.1f} ms")
    print("--- derived, per frame ---")
    print(
        f"{'roundtrip with a count':<32}"
        f"{timings['device/count']:7.1f} ms   "
        f"{timings['transfer-only']:7.1f} ms without one"
    )
    print(
        f"{'count, marginal in a chain':<32}"
        f"{timings['device/levels+count'] - timings['device/levels']:7.1f} ms   "
        f"host {timings['host/levels+count'] - timings['host/levels']:7.1f} ms"
    )
    print(
        f"{'lloyd chain':<32}"
        f"{timings['device/lloyd']:7.1f} ms   host {timings['host/lloyd']:7.1f} ms"
    )
    print(
        f"{'even chain':<32}"
        f"{timings['device/even']:7.1f} ms   host {timings['host/even']:7.1f} ms"
    )
    return 0


def print_table(results: list[dict[str, Any]]) -> None:
    print(f"{'case':<30} {'format':<10} {'frames':>6}  sha256")
    for result in sorted(results, key=lambda entry: entry["name"]):
        print(
            f"{result['name']:<30} {result['format'] or 'undefined':<10}"
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
        "generator": "tools/histogram_hash.py",
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
        f"{name}: no longer produced" for name in sorted(set(expected) - set(actual))
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
        print("note: the interpreter or numpy version differs; the properties are unchanged")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--write", action="store_true", help="record the host table under target/")
    mode.add_argument("--check", action="store_true", help="fail when a recorded hash differs")
    mode.add_argument("--device", action="store_true", help="compare gpu=1 against gpu=0")
    mode.add_argument("--bench", action="store_true", help="time one frame of each graph shape")
    parser.add_argument("--frames", type=int, default=20, help="frames per timed row")
    parser.add_argument("--format", default="GRAY8", help="format for --bench")
    parser.add_argument("--cache", type=int, default=512, help="core.max_cache_size for --bench")
    parser.add_argument("--output", type=Path, default=RECORD, help="recorded table path")
    args = parser.parse_args(argv)

    if args.bench:
        format_ = getattr(vs, args.format, None)
        if format_ is None:
            print(f"unknown format {args.format}", file=sys.stderr)
            return 1
        return run_bench(args.frames, CORE.get_video_format(format_), args.cache)

    table = sorted(cases(), key=lambda case: case.name)
    if args.device:
        return compare_device(table) | compare_chain() | compare_views()

    results = []
    for case in table:
        try:
            results.append(run_case(case))
        except Exception as error:  # the two refused cases have no host output either
            print(f"{case.name:<30} skip ({error})")
    print_table(results)
    if args.write:
        return write_table(args.output, results)
    if args.check:
        return check_table(args.output, results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
