#!/usr/bin/env python3
"""Measure how many frames a filter keeps in flight.

`get_frame_async` submits several requests before any result is consumed. A
filter that answers `Initial` with `request_frame_filter` and returns no frame
lets the core overlap the next input with the current output; one that waits for
its input at the request site serialises the graph.

The probe reports milliseconds per frame at several in-flight windows and the
speedup against one frame. It also compares one asynchronous frame with the same
frame pulled synchronously, so a faster wrong answer cannot pass as a win.

Run it with the development interpreter::

    python tools/async_probe.py
    python tools/async_probe.py --device
    python tools/async_probe.py --device --size 1920x1080 --in-flight 1 2 4 8

A host without a usable device prints the creation error and skips the device
rows, matching `tools/gpu_probe.py` and the validator. `--require-device` makes
that skip a failure for a job that is supposed to have one.
"""

from __future__ import annotations

import argparse
import statistics
import sys
import time

import numpy as np
import vapoursynth as vs

CORE = vs.core
#: The filters with a `gpu` argument, in a stable order.
FILTERS = ("Deblur", "Levels", "PeakGrayShades", "PeakStats", "Posterize")
#: A filter's output can be read on the host or left on the device.
ENGINES = (("host", 0), ("device", 1))


def build(core: vs.Core, name: str, source: vs.VideoNode, gpu: int) -> vs.VideoNode:
    """One graph row, ending on the host when `gpu` is set."""
    if name == "Deblur":
        node = core.nimages.Deblur(source, gpu=gpu)
    elif name == "Levels":
        node = core.nimages.Levels(source, black=16, white=235, gpu=gpu)
    elif name == "PeakGrayShades":
        node = core.nimages.PeakGrayShades(source, gpu=gpu)
    elif name == "PeakStats":
        node = core.nimages.PeakStats(source, gpu=gpu)
    elif name == "Posterize":
        node = core.nimages.Posterize(source, bits=4, gpu=gpu)
    else:
        raise ValueError(f"unknown filter {name}")
    if gpu:
        node = core.std.GPUDownload(node)
    return node


def source(core: vs.Core, width: int, height: int, format_: vs.VideoFormat, frames: int):
    """A host clip long enough for every in-flight window."""
    return core.std.BlankClip(width=width, height=height, format=format_, length=frames)


def one_pull(node: vs.VideoNode, in_flight: int) -> float:
    """Seconds per frame with `in_flight` requests outstanding."""
    start = 1
    started = time.perf_counter()
    futures = [node.get_frame_async(start + i) for i in range(in_flight)]
    for future in futures:
        with future.result():
            pass
    return (time.perf_counter() - started) / in_flight


def measure(build_node, in_flight: int, repeats: int) -> float:
    """Median seconds per frame over fresh nodes and distinct frame indexes."""
    samples = []
    for _ in range(repeats):
        node = build_node()
        # The warm-up frame is not one of the timed indexes, so the cache cannot
        # answer a request the filter was supposed to compute.
        with node.get_frame(0):
            pass
        samples.append(one_pull(node, in_flight))
    return statistics.median(samples)


def same_frame(left: vs.VideoFrame, right: vs.VideoFrame) -> bool:
    """Every plane and every property, so a swapped frame cannot pass."""
    if left.format.num_planes != right.format.num_planes:
        return False
    for plane in range(left.format.num_planes):
        if np.asarray(left[plane]).tobytes() != np.asarray(right[plane]).tobytes():
            return False
    return dict(left.props) == dict(right.props)


def exactness(core: vs.Core, name: str, source: vs.VideoNode, gpu: int) -> tuple[bool, str]:
    """One async frame against the same frame pulled synchronously."""
    requested = build(core, name, source, gpu)
    expected = build(core, name, source, gpu)
    future = requested.get_frame_async(1)
    with expected.get_frame(1) as reference, future.result() as actual:
        if not same_frame(actual, reference):
            return False, f"{name} gpu={gpu}"
    return True, ""


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", action="store_true", help="also measure gpu=1 rows")
    parser.add_argument(
        "--require-device",
        action="store_true",
        help="fail when no usable device answers instead of skipping",
    )
    parser.add_argument("--format", default="GRAY8", help="VapourSynth format name")
    parser.add_argument("--size", default="1280x720", help="frame size as WxH")
    parser.add_argument(
        "--in-flight",
        nargs="+",
        type=int,
        default=(1, 2, 4, 8),
        help="requests outstanding per row",
    )
    parser.add_argument("--repeats", type=int, default=3, help="fresh-node samples per cell")
    parser.add_argument("--cache", type=int, default=512, help="core.max_cache_size in MB")
    args = parser.parse_args(argv)

    if args.format not in dir(vs):
        print(f"unknown format {args.format}", file=sys.stderr)
        return 1
    try:
        width, height = (int(part) for part in args.size.lower().split("x", 1))
    except ValueError:
        print(f"bad --size {args.size!r}, expected WxH", file=sys.stderr)
        return 1
    if (
        width <= 0
        or height <= 0
        or min(args.in_flight) <= 0
        or args.repeats <= 0
    ):
        print("size, in-flight and repeats values must be positive", file=sys.stderr)
        return 1

    CORE.max_cache_size = args.cache
    format_ = CORE.get_video_format(getattr(vs, args.format))
    frames = max(args.in_flight) + 2
    clip = source(CORE, width, height, format_, frames)

    engines = [ENGINES[0]]
    if args.device:
        try:
            with build(CORE, "Levels", clip, 1).get_frame(0):
                pass
        except vs.Error as error:
            message = " ".join(str(error).split())
            if args.require_device:
                print(f"device: failed, {message}", file=sys.stderr)
                return 1
            print(f"device: skipped, {message}", file=sys.stderr)
        else:
            engines = list(ENGINES)

    print(f"{'filter':<14} {'engine':<8} {'in-flight':>9} {'ms/frame':>10} {'speedup':>8}")
    failed = False
    for name in FILTERS:
        for engine, gpu in engines:
            baseline = measure(lambda: build(CORE, name, clip, gpu), 1, args.repeats)
            for in_flight in args.in_flight:
                seconds = measure(lambda: build(CORE, name, clip, gpu), in_flight, args.repeats)
                speedup = baseline / seconds
                print(
                    f"{name:<14} {engine:<8} {in_flight:>9} "
                    f"{seconds * 1000:>10.2f} {speedup:>8.2f}x"
                )
            ok, detail = exactness(CORE, name, clip, gpu)
            if not ok:
                print(f"mismatch: {detail}", file=sys.stderr)
                failed = True

    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
