#!/usr/bin/env python3
"""Probe the Vulkan device the installed core can run `nimages` kernels on.

Prints every device the core reports, with its `usable` flag and the reason a
device is refused, pins the first usable one with `core.set_vulkan_device`, and
proves a `std.GPUUpload` -> `std.GPUDownload` round trip returns the input bytes.
This is the device gate `tests/check-nimages.py` will copy: a host with no usable
device skips with exit 0, so a missing GPU cannot hide as a failure and a broken
device path cannot hide as a skip.

Run it with the development interpreter::

    .venv\\Scripts\\python.exe tools\\gpu_probe.py

Exit status is 0 when either no device is usable (printing
`gpu: skipped, <reason>`) or a usable device survives the round trip. It is 1
when a device the core called usable cannot be pinned or moves one byte.
"""

from __future__ import annotations

import sys

import numpy as np
import vapoursynth as vs

CORE = vs.core
#: An odd width and height, so a row-padding bug cannot pass the round trip.
WIDTH = 37
HEIGHT = 23
#: Two frames, so the probe covers a per-frame transfer as well as the first.
FRAMES = 2


def sample_source(width: int = WIDTH, height: int = HEIGHT, frames: int = FRAMES) -> vs.VideoNode:
    """A deterministic `GRAY8` clip whose bytes are not all one value."""
    blank = CORE.std.BlankClip(width=width, height=height, format=vs.GRAY8, length=frames)

    def fill(n: int, f: vs.VideoFrame) -> vs.VideoFrame:
        out = f.copy()
        row = np.arange(out.height, dtype=np.uint32)[:, None]
        column = np.arange(out.width, dtype=np.uint32)[None, :]
        np.asarray(out[0])[...] = ((column * 7 + row * 13 + n * 29) % 251).astype(np.uint8)
        return out

    return CORE.std.ModifyFrame(blank, blank, fill)


def list_devices() -> tuple[list[dict[str, object]] | None, str]:
    """The core's device list, or `(None, reason)` when it has no Vulkan support."""
    try:
        return list(CORE.vulkan_devices), ""
    except Exception as error:  # a core built without Vulkan has no such property
        return None, f"the core has no Vulkan device list ({error})"


def describe(device: dict[str, object]) -> str:
    api = ".".join(str(part) for part in device.get("api_version", ()))
    memory = int(device.get("device_memory", 0)) / (1024 * 1024)
    state = "usable" if device.get("usable") else f"refused: {device.get('reason') or 'no reason given'}"
    return (
        f"{device['index']:>3}  {str(device.get('type', 'unknown')):<10}  {api:<9}  "
        f"{memory:>7.0f} MiB  {device['name']} ({state})"
    )


def round_trip() -> tuple[bool, str]:
    """Uploads a host clip and downloads it again, comparing every plane's bytes."""
    source = sample_source()
    uploaded = CORE.std.GPUUpload(source)
    downloaded = CORE.std.GPUDownload(uploaded)

    for n in range(source.num_frames):
        with uploaded.get_frame(n) as resident:
            if not resident.gpu_resident:
                return False, f"frame {n} is not device resident after std.GPUUpload"
        with source.get_frame(n) as host, downloaded.get_frame(n) as back:
            if back.gpu_resident:
                return False, f"frame {n} is still device resident after std.GPUDownload"
            if (host.width, host.height) != (back.width, back.height):
                return False, f"frame {n} changed geometry across the round trip"
            for plane in range(host.format.num_planes):
                if np.asarray(host[plane]).tobytes() != np.asarray(back[plane]).tobytes():
                    return False, f"frame {n} plane {plane} changed across the round trip"

    return True, f"{source.width}x{source.height}, {source.num_frames} frames"


def main() -> int:
    CORE.num_threads = 1

    devices, reason = list_devices()
    if devices is None:
        print(f"gpu: skipped, {reason}")
        return 0

    print(f"{'idx':>3}  {'type':<10}  {'vulkan':<9}  {'memory':>11}  name")
    for device in devices:
        print(describe(device))

    usable = [device for device in devices if device.get("usable")]
    if not usable:
        print("gpu: skipped, no usable Vulkan 1.4 device")
        return 0

    chosen = usable[0]
    try:
        CORE.set_vulkan_device(int(chosen["index"]))
    except Exception as error:
        print(f"gpu: error, could not pin device {chosen['index']}: {error}")
        return 1

    info = CORE.vulkan_device_info
    mib = 1024 * 1024
    print(
        f"gpu: pinned {chosen['index']} {chosen['name']} "
        f"({int(info['device_memory']) // mib} MiB, budget {int(info['budget']) // mib} MiB, "
        f"unified_memory {info['unified_memory']})"
    )

    ok, detail = round_trip()
    if not ok:
        print(f"gpu: error, {detail}")
        return 1
    print(f"gpu: ok, std.GPUUpload -> std.GPUDownload returned {detail} byte for byte")
    return 0


if __name__ == "__main__":
    sys.exit(main())
