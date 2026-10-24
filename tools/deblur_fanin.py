#!/usr/bin/env python3
"""Measure `Deblur`'s aggregate throughput through a temporal fan-in.

latency is one frame's cost; a volume encode asks for frame after frame, so the
number that decides adoption is frames per second with several frames of the same
node outstanding. `std.AverageFrames` with `n` weights pulls `n` consecutive
frames of its input at once, which gives the filter exactly that shape.

a fresh node per repeat is what makes the run honest: the frame cache would
otherwise answer for the filter and the median would measure the cache.

The protocol is plan 16's. Run it with the development interpreter::

    .venv\\Scripts\\python.exe tools\\deblur_fanin.py
    .venv\\Scripts\\python.exe tools\\deblur_fanin.py --device --sweep
    .venv\\Scripts\\python.exe tools\\deblur_fanin.py --page 24 --format RGB24

`vram` is the peak the run added to the device, read from NVML above the idle
baseline, so it is what the pipeline held rather than what the desktop held. It
prints `-` when NVML is unavailable, which changes nothing else. A run that
exhausts the device aborts the host process, because VapourSynth treats the
allocation failure as fatal; `--vram MB` sets `core.max_vram_cache_size`, which
is the knob that keeps the core from admitting more frames than fit.
"""

from __future__ import annotations

import argparse
import ctypes
import statistics
import sys
import threading
import time
from dataclasses import dataclass

import vapoursynth as vs

#: The two page sizes `docs/BENCH.md` reports.
PAGES = {"12": (2902, 4128), "24": (5806, 4128)}
#: Odd weights only: `std.AverageFrames` centres the current frame.
FAN_INS = (1, 3, 5, 7, 9, 11, 13)
#: Frames per pull, which is also the fan-in `docs/BENCH.md` quotes.
DEFAULT_FAN_IN = 13
#: The cache a volume encode would set, in MB.
DEFAULT_CACHE = 512


@dataclass(frozen=True)
class Result:
    """One engine's numbers for one geometry."""

    seconds: float
    peak_vram: int
    fan_in: int

    @property
    def rate(self) -> float:
        return self.fan_in / self.seconds


class DeviceMemory:
    """Device use in bytes from NVML, or `None` where it is not installed."""

    class Info(ctypes.Structure):
        _fields_ = [
            ("total", ctypes.c_ulonglong),
            ("free", ctypes.c_ulonglong),
            ("used", ctypes.c_ulonglong),
        ]

    def __init__(self) -> None:
        self.lib = None
        for name in ("nvml.dll", "libnvidia-ml.so.1", "libnvidia-ml.so"):
            try:
                self.lib = ctypes.CDLL(name)
            except OSError:
                continue
            break
        if self.lib is None or self.lib.nvmlInit_v2() != 0:
            self.lib = None
            return
        self.handle = ctypes.c_void_p()
        if self.lib.nvmlDeviceGetHandleByIndex_v2(0, ctypes.byref(self.handle)) != 0:
            self.lib = None

    def used(self) -> int | None:
        if self.lib is None:
            return None
        info = self.Info()
        if self.lib.nvmlDeviceGetMemoryInfo(self.handle, ctypes.byref(info)) != 0:
            return None
        return info.used


def build(
    core: vs.Core,
    width: int,
    height: int,
    format_: vs.VideoFormat,
    method: int,
    fan_in: int,
    device: bool,
) -> vs.VideoNode:
    source = core.std.BlankClip(width=width, height=height, format=format_, length=fan_in)
    node = core.nimages.Deblur(source, method=method, gpu=1 if device else 0)
    if device:
        node = core.std.GPUDownload(node)
    return core.std.AverageFrames(node, weights=[1.0] * fan_in)


def measure(
    core: vs.Core,
    memory: DeviceMemory,
    baseline: int,
    width: int,
    height: int,
    format_: vs.VideoFormat,
    method: int,
    fan_in: int,
    device: bool,
    repeats: int,
) -> Result:
    elapsed = []
    peak = 0
    stop = threading.Event()

    def sample() -> None:
        nonlocal peak
        while not stop.is_set():
            seen = memory.used()
            if seen is not None:
                peak = max(peak, seen - baseline)
            time.sleep(0.02)

    sampler = threading.Thread(target=sample, daemon=True)
    sampler.start()
    for _ in range(repeats):
        node = build(core, width, height, format_, method, fan_in, device)
        start = time.perf_counter()
        node.get_frame(0)
        elapsed.append(time.perf_counter() - start)
        del node
    stop.set()
    sampler.join()

    return Result(
        seconds=statistics.median(elapsed),
        peak_vram=peak,
        fan_in=fan_in,
    )


def print_row(page: str, format_name: str, method: int, cpu: Result, gpu: Result | None) -> None:
    label = f"{page} MP {format_name} method={method}"
    cpu_text = f"cpu {cpu.rate:6.1f} frames/s"
    if gpu is None:
        print(f"{label:<28} {cpu_text}")
        return
    vram = "-" if gpu.peak_vram == 0 else f"{gpu.peak_vram / (1024 * 1024):.0f} MB"
    print(
        f"{label:<28} {cpu_text}   gpu {gpu.rate:6.1f} frames/s"
        f"  {gpu.rate / cpu.rate:5.2f}x  vram {vram}"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("--page", choices=sorted(PAGES), default="12")
    parser.add_argument("--format", default="GRAY8")
    parser.add_argument("--method", type=int, default=1)
    parser.add_argument(
        "--fanin",
        default=str(DEFAULT_FAN_IN),
        help=f"comma separated frame counts, odd only; default {DEFAULT_FAN_IN}",
    )
    parser.add_argument("--sweep", action="store_true", help="run every fan-in in FAN_INS")
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--cache", type=int, default=DEFAULT_CACHE)
    parser.add_argument(
        "--vram", type=int, default=0, help="core.max_vram_cache_size, in MB"
    )
    parser.add_argument("--device", action="store_true", help="run the device engine too")
    args = parser.parse_args(argv)

    core = vs.core
    core.max_cache_size = args.cache
    if args.vram:
        core.max_vram_cache_size = args.vram
    device = False
    if args.device:
        usable = [entry for entry in core.vulkan_devices if entry.get("usable")]
        if not usable:
            print("device: skipped, no usable Vulkan 1.4 device", file=sys.stderr)
            return 0
        core.set_vulkan_device(usable[0]["index"])
        device = True
        version = ".".join(map(str, usable[0]["api_version"]))
        print(f"device: {usable[0]['name']} (Vulkan {version})")

    format_ = getattr(vs, args.format, None)
    if format_ is None:
        print(f"unknown format {args.format}", file=sys.stderr)
        return 1
    width, height = PAGES[args.page]
    fan_ins = (
        FAN_INS
        if args.sweep
        else tuple(int(value) for value in args.fanin.split(",") if value)
    )
    for fan_in in fan_ins:
        if fan_in < 1 or fan_in % 2 == 0:
            print(f"fan-in {fan_in} must be odd and positive", file=sys.stderr)
            return 1

    memory = DeviceMemory()
    if device and memory.lib is None:
        print("vram: unavailable, NVML refused or missing", file=sys.stderr)
    # Read once, before any device work, so the peak is what the graph held
    # rather than what a warm allocator left behind.
    baseline = memory.used() or 0

    for fan_in in fan_ins:
        # One untimed pull each, so a first-touch cost is not the median.
        build(core, width, height, format_, args.method, fan_in, False).get_frame(0)
        cpu = measure(
            core,
            memory,
            baseline,
            width,
            height,
            format_,
            args.method,
            fan_in,
            False,
            args.repeats,
        )
        gpu = None
        if device:
            build(core, width, height, format_, args.method, fan_in, True).get_frame(0)
            gpu = measure(
                core,
                memory,
                baseline,
                width,
                height,
                format_,
                args.method,
                fan_in,
                True,
                args.repeats,
            )
        print_row(args.page, format_.name, args.method, cpu, gpu)
        print(
            f"{'':<28} fan-in {fan_in}, median of {args.repeats},"
            f" cache {args.cache} MB",
            flush=True,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
