#!/usr/bin/env python3
"""Integration checks for the built `nimages` plugin.

Run it against an installed plugin:

    .venv\\Scripts\\python.exe tests\\check-nimages.py

It replays the committed golden vectors from `tests/fixtures/` through the real
filters, so a passing run means the plugin agrees with the reference python
implementation and with the Rust unit tests. It needs numpy and VapourSynth and
nothing else.

`uv sync --extra dev --extra dev-tests` builds and installs the plugin first.
"""

from __future__ import annotations

import json
import math
import sys
from concurrent.futures import ThreadPoolExecutor
from functools import partial
from pathlib import Path

import numpy as np

import vapoursynth as vs

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"

PLUGIN_ID = "xyz.n4o.nimages"
NAMESPACE = "nimages"
FUNCTIONS = ("Levels", "PeakGrayShades", "PeakStats", "Posterize")

# Histogram cases above this many pixels are left to the Rust tests; building a
# frame that wide is not worth the time here.
MAX_SYNTHETIC_PIXELS = 1 << 16

core = vs.core

failures: list[str] = []
checks = 0


# ---------------------------------------------------------------------------
# harness
# ---------------------------------------------------------------------------


def check(condition: bool, message: str) -> None:
    global checks
    checks += 1
    if not condition:
        failures.append(message)


def same(actual: object, expected: object, message: str) -> None:
    check(actual == expected, f"{message}\n    got      {actual!r}\n    expected {expected!r}")


def section(title: str) -> None:
    print(f"\n== {title}")


def load(name: str) -> dict:
    return json.loads((FIXTURES / name).read_text(encoding="utf-8"))


def expect_error(what: str, call, *, contains: str = "") -> None:
    """Asserts that `call` raises, and that the message names the problem."""
    global checks
    checks += 1
    try:
        result = call()
    except Exception as error:
        text = str(error)
        if contains and contains not in text:
            failures.append(f"{what}: message does not mention {contains!r}: {text}")
        return
    failures.append(f"{what}: expected an error, got {result!r}")


def as_list(value: object) -> list:
    """Normalises a property array.

    VapourSynth hands back a bare scalar for a one element array and `None` for
    a property that was never set, so every array read goes through here.
    """
    if value is None:
        return []
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


# ---------------------------------------------------------------------------
# frame construction
# ---------------------------------------------------------------------------


def fill(n: int, f, *, array: np.ndarray):
    del n
    out = f.copy()
    np.asarray(out[0])[:] = array
    return out


def clip_of(array: np.ndarray, *, length: int = 1):
    """Builds a `GRAY8` clip whose every frame holds `array`."""
    height, width = array.shape
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAY8, length=length)
    return core.std.ModifyFrame(blank, blank, partial(fill, array=array))


def clip_of_frames(arrays: list[np.ndarray]):
    """Builds a `GRAY8` clip whose frame `n` holds `arrays[n]`."""
    height, width = arrays[0].shape
    blank = core.std.BlankClip(width=width, height=height, format=vs.GRAY8, length=len(arrays))
    lookup = arrays

    def pick(n: int, f):
        return fill(n, f, array=lookup[n])

    return core.std.ModifyFrame(blank, blank, pick)


def array_of(clip, n: int = 0) -> np.ndarray:
    with clip.get_frame(n) as frame:
        return np.asarray(frame[0]).copy()


def props_of(clip, n: int = 0) -> dict:
    with clip.get_frame(n) as frame:
        return dict(frame.props)


def frame_from_histogram(hist: list[int]) -> np.ndarray | None:
    """A one-row image whose histogram is exactly `hist`, or `None` if empty."""
    pixels = np.repeat(np.arange(256, dtype=np.uint8), np.asarray(hist, dtype=np.int64))
    if pixels.size == 0:
        # An empty histogram cannot be a frame, so only the Rust tests cover it.
        return None
    return pixels.reshape(1, pixels.size)


def frame_from_info(info: dict) -> np.ndarray:
    data = (FIXTURES / info["file"]).read_bytes()
    check(len(data) == info["bytes"], f"{info['file']}: unexpected size on disk")
    return np.frombuffer(data, dtype=np.uint8).reshape(info["height"], info["width"]).copy()


def ramp() -> np.ndarray:
    return np.arange(256, dtype=np.uint8).reshape(1, 256)


# ---------------------------------------------------------------------------
# sections
# ---------------------------------------------------------------------------


def check_registration() -> None:
    section("registration")
    plugins = [plugin for plugin in core.plugins() if str(plugin.identifier) == PLUGIN_ID]
    check(len(plugins) == 1, f"expected one plugin with identifier {PLUGIN_ID}")
    if not plugins:
        return
    plugin = plugins[0]

    same(str(plugin.namespace), NAMESPACE, "namespace")
    names = sorted(function.name for function in plugin.functions())
    same(names, list(FUNCTIONS), "registered functions")

    for name in FUNCTIONS:
        exported = getattr(core.nimages, name)
        same(exported.return_signature, "clip:vnode;", f"{name}: return signature")
        check(exported.signature.startswith("clip:vnode;"), f"{name}: takes a clip first")


def check_frame_peaks(frames: dict, fixtures: dict) -> None:
    section("peak statistics over the frame fixtures")
    for case in fixtures["peak_cases"]:
        info = frames[case["frame"]]
        clip = clip_of(frame_from_info(info))
        arguments = {
            "upper_limit": case["upper_limit"],
            "peak_percentage": case["peak_percentage"],
            "skip_white": int(case["skip_white"]),
        }
        if case["peak_prominence"] is not None:
            arguments["peak_prominence"] = case["peak_prominence"]

        props = props_of(core.nimages.PeakStats(clip, **arguments))
        want = case["expect"]
        for name, key in (
            ("black", "NImagesBlackLevel"),
            ("white", "NImagesWhiteLevel"),
            ("black_found", "NImagesBlackPeakFound"),
            ("white_found", "NImagesWhitePeakFound"),
        ):
            same(props[key], int(want[name]), f"{case['name']}: {key}")


def check_histogram_peaks(fixtures: dict) -> None:
    section("peak statistics over the histogram fixtures")
    tested = 0
    for case in fixtures["cases"]:
        total = case["total_pixels"]
        if total > MAX_SYNTHETIC_PIXELS:
            continue
        page = frame_from_histogram(fixtures["histograms"][case["histogram"]])
        if page is None:
            continue
        clip = clip_of(page)
        arguments = {
            "upper_limit": case["upper_limit"],
            "peak_percentage": case["peak_percentage"],
            "skip_white": int(case["skip_white"]),
        }
        if case["peak_prominence"] is not None:
            arguments["peak_prominence"] = case["peak_prominence"]

        props = props_of(core.nimages.PeakStats(clip, **arguments))
        want = case["expect"]
        tested += 1
        for name, key in (
            ("black", "NImagesBlackLevel"),
            ("white", "NImagesWhiteLevel"),
            ("black_found", "NImagesBlackPeakFound"),
            ("white_found", "NImagesWhitePeakFound"),
        ):
            same(props[key], int(want[name]), f"{case['name']}: {key}")
    print(f"   {tested} histogram cases materialised")


def check_frame_shades(frames: dict, fixtures: dict) -> None:
    section("gray shades over the frame fixtures")
    for case in fixtures["shade_cases"]:
        clip = clip_of(frame_from_info(frames[case["frame"]]))
        props = props_of(core.nimages.PeakGrayShades(clip, threshold=case["threshold"]))
        same(
            as_list(props["NImagesGrayShades"]),
            [entry["shade"] for entry in case["expect"]],
            f"{case['name']}: shades",
        )
        same(
            as_list(props["NImagesGrayShadePercentages"]),
            [entry["percentage"] for entry in case["expect"]],
            f"{case['name']}: percentages",
        )


def check_histogram_shades(fixtures: dict) -> None:
    section("gray shades over the histogram fixtures")
    tested = 0
    for case in fixtures["cases"]:
        if case["total_pixels"] > MAX_SYNTHETIC_PIXELS:
            continue
        page = frame_from_histogram(fixtures["histograms"][case["histogram"]])
        if page is None:
            continue
        clip = clip_of(page)
        props = props_of(core.nimages.PeakGrayShades(clip, threshold=case["threshold"]))
        tested += 1
        same(
            as_list(props["NImagesGrayShades"]),
            [entry["shade"] for entry in case["expect"]],
            f"{case['name']}: shades",
        )
        same(len(as_list(props["NImagesGrayShades"])), len(as_list(props["NImagesGrayShadePercentages"])),
             f"{case['name']}: the two arrays have the same length")
    print(f"   {tested} histogram cases materialised")


def check_levels(fixtures: dict) -> None:
    section("level tables")
    source = clip_of(ramp())
    for case in fixtures["cases"]:
        table = array_of(
            core.nimages.Levels(
                source, black=int(case["black"]), white=int(case["white"]), gamma=case["gamma"]
            )
        ).ravel()
        same(table.tolist(), case["expect"], f"black={case['black']} white={case['white']} gamma={case['gamma']}")


def check_posterize(fixtures: dict) -> None:
    section("posterize tables")
    source = clip_of(ramp())
    for case in fixtures["cases"]:
        table = array_of(core.nimages.Posterize(source, bits=case["bits"])).ravel()
        same(table.tolist(), case["expect"], f"bits={case['bits']}")
        same(sorted(set(table.tolist())), case["levels"], f"bits={case['bits']}: distinct levels")


def check_levels_from_properties() -> None:
    section("levels driven by PeakStats properties")
    # A page with a black peak at 12 and a white peak at 245.
    page = np.full((32, 48), 128, dtype=np.uint8)
    page[:8, :] = 12
    page[-8:, :] = 245
    clip = clip_of(page)

    stats = core.nimages.PeakStats(clip, upper_limit=60, peak_percentage=0.25)
    props = props_of(stats)
    same(props["NImagesBlackLevel"], 12, "detected black level")
    same(props["NImagesWhiteLevel"], 245, "detected white level")

    automatic = array_of(core.nimages.Levels(stats, use_props=True, auto_gamma=True))
    black_normalized = float(props["NImagesBlackLevel"]) / 255
    gamma = round(
        1.0 / (math.log(0.5) / math.log((0.5 - black_normalized) / (1.0 - black_normalized))), 2
    )
    expected = array_of(core.nimages.Levels(clip, black=12, white=245, gamma=gamma))
    same(automatic.tolist(), expected.tolist(), "use_props matches the same constant parameters")

    # peak_offset moves the black point in code values, not percentage points.
    offset = array_of(core.nimages.Levels(stats, use_props=True, peak_offset=1, auto_gamma=False))
    shifted = array_of(core.nimages.Levels(clip, black=13, white=245, gamma=1.0))
    same(offset.tolist(), shifted.tolist(), "peak_offset is a code value")

    # A frame with no peak at all still levels, from the defaults.
    flat = clip_of(np.full((16, 16), 200, dtype=np.uint8))
    flat_stats = core.nimages.PeakStats(flat, skip_white=1)
    same(props_of(flat_stats)["NImagesBlackPeakFound"], 0, "a flat frame has no black peak")
    array_of(core.nimages.Levels(flat_stats, use_props=True))  # must not raise


def check_property_preservation() -> None:
    section("property preservation")
    page = np.full((16, 16), 100, dtype=np.uint8)
    clip = core.std.SetFrameProps(
        clip_of(page), ImgSeqPath="I:/pages/001.png", ImgSeqIndex=7, NImagesCustom=3
    )

    stats = core.nimages.PeakStats(clip)
    shades = core.nimages.PeakGrayShades(clip)
    leveled = core.nimages.Levels(clip, black=10, white=200, gamma=1.0)
    posterized = core.nimages.Posterize(clip, bits=4)

    for label, node in (
        ("PeakStats", stats),
        ("PeakGrayShades", shades),
        ("Levels", leveled),
        ("Posterize", posterized),
    ):
        props = props_of(node)
        same(props.get("ImgSeqPath"), "I:/pages/001.png", f"{label}: ImgSeqPath")
        same(props.get("ImgSeqIndex"), 7, f"{label}: ImgSeqIndex")
        same(props.get("NImagesCustom"), 3, f"{label}: an unrelated property")


def check_analysis_leaves_pixels_alone() -> None:
    section("analysis leaves pixels alone")
    rng = np.random.default_rng(11)
    page = rng.integers(0, 256, (37, 13), dtype=np.uint8)
    clip = clip_of(page)
    for label, node in (
        ("PeakStats", core.nimages.PeakStats(clip)),
        ("PeakGrayShades", core.nimages.PeakGrayShades(clip)),
    ):
        same(array_of(node).tolist(), page.tolist(), f"{label} changed pixels")


def check_geometry() -> None:
    section("geometry and stride padding")
    rng = np.random.default_rng(3)
    shapes = [(1, 1), (1, 7), (7, 1), (13, 7), (17, 33), (64, 64), (33, 5)]
    for width, height in shapes:
        page = rng.integers(0, 256, (height, width), dtype=np.uint8)
        clip = clip_of(page)
        posterized = array_of(core.nimages.Posterize(clip, bits=1))
        expected = np.where(page < 128, 0, 255).astype(np.uint8)
        same(posterized.tolist(), expected.tolist(), f"{width}x{height}: posterize pixels")

        stats = props_of(core.nimages.PeakStats(clip))
        check("NImagesBlackLevel" in stats, f"{width}x{height}: PeakStats wrote its properties")

    # A one pixel frame cannot clear a 0.01 percent threshold, so both arrays
    # must be present and empty.
    single = clip_of(np.array([[128]], dtype=np.uint8))
    props = props_of(core.nimages.PeakGrayShades(single, threshold=0.01))
    same(as_list(props["NImagesGrayShades"]), [], "1x1: shades are an empty array")
    same(as_list(props["NImagesGrayShadePercentages"]), [], "1x1: percentages are an empty array")

    # A zero threshold accepts every present shade.
    wide = clip_of(np.array([[10, 20, 30, 40]], dtype=np.uint8))
    props = props_of(core.nimages.PeakGrayShades(wide, threshold=0.0))
    same(as_list(props["NImagesGrayShades"]), [10, 20, 30, 40], "threshold 0 keeps every shade")


def check_per_frame_results() -> None:
    section("each frame is analyzed on its own")
    # Frame n holds half a page at 200 and half at 5 + 10n, so the black peak
    # moves per frame while the clip's dimensions stay constant.
    pages = []
    for n in range(4):
        page = np.full((20, 20), 200, dtype=np.uint8)
        page[:10, :] = 5 + 10 * n
        pages.append(page)
    clip = clip_of_frames(pages)
    stats = core.nimages.PeakStats(clip, upper_limit=60)

    for n, expected in enumerate((5, 15, 25, 35)):
        props = props_of(stats, n)
        same(props["NImagesBlackLevel"], expected, f"frame {n}: black level")

    # Levels(use_props=True) must follow the same per-frame values.
    leveled = core.nimages.Levels(stats, use_props=True, auto_gamma=False)
    for n in range(4):
        frame = array_of(leveled, n)
        same(int(frame[0, 0]), 0, f"frame {n}: the black half maps to 0")
        same(int(frame[-1, 0]), 255, f"frame {n}: the white half maps to 255")


def check_request_patterns() -> None:
    section("repeated, out-of-order and concurrent requests")
    rng = np.random.default_rng(5)
    pages = [rng.integers(0, 256, (24, 19), dtype=np.uint8) for _ in range(4)]
    clip = clip_of_frames(pages)
    source = core.nimages.Posterize(clip, bits=2)

    reference = [array_of(source, n) for n in range(4)]

    for n in (3, 0, 2, 1, 2, 0, 3, 3):
        same(array_of(source, n).tolist(), reference[n].tolist(), f"request {n} again")

    stats = core.nimages.PeakStats(clip)
    expected_black = [props_of(stats, n)["NImagesBlackLevel"] for n in range(4)]

    def grab(n: int) -> int:
        return props_of(stats, n)["NImagesBlackLevel"]

    with ThreadPoolExecutor(max_workers=8) as pool:
        concurrent = list(pool.map(grab, [3, 1, 0, 2, 2, 0, 1, 3]))
    same(concurrent, [expected_black[n] for n in (3, 1, 0, 2, 2, 0, 1, 3)], "concurrent requests")


def check_errors() -> None:
    section("invalid arguments")
    gray = clip_of(np.full((8, 8), 100, dtype=np.uint8))
    rgb = core.std.BlankClip(width=8, height=8, format=vs.RGB24, length=1)
    gray16 = core.std.BlankClip(width=8, height=8, format=vs.GRAY16, length=1)

    expect_error("PeakStats on RGB", lambda: core.nimages.PeakStats(rgb), contains="Gray 8 bit")
    expect_error("Posterize on GRAY16", lambda: core.nimages.Posterize(gray16, bits=4), contains="Gray 8 bit")
    expect_error("PeakStats without a clip", lambda: core.nimages.PeakStats())
    expect_error("PeakStats upper_limit 0", lambda: core.nimages.PeakStats(gray, upper_limit=0))
    expect_error("PeakStats upper_limit 256", lambda: core.nimages.PeakStats(gray, upper_limit=256))
    expect_error(
        "PeakStats peak_percentage 200",
        lambda: core.nimages.PeakStats(gray, peak_percentage=200.0),
        contains="between 0 and 100",
    )
    expect_error(
        "PeakStats peak_prominence -1",
        lambda: core.nimages.PeakStats(gray, peak_prominence=-1.0),
        contains="between 0 and 100",
    )
    expect_error("PeakGrayShades threshold -1", lambda: core.nimages.PeakGrayShades(gray, threshold=-1.0))
    expect_error("Posterize without bits", lambda: core.nimages.Posterize(gray), contains="bits is required")
    expect_error("Posterize bits 0", lambda: core.nimages.Posterize(gray, bits=0), contains="between 1 and 8")
    expect_error("Posterize bits 9", lambda: core.nimages.Posterize(gray, bits=9), contains="between 1 and 8")
    expect_error(
        "Levels reversed endpoints",
        lambda: core.nimages.Levels(gray, black=200, white=100),
        contains="lower than white level",
    )
    expect_error(
        "Levels equal endpoints",
        lambda: core.nimages.Levels(gray, black=100, white=100),
        contains="lower than white level",
    )
    expect_error(
        "Levels auto_gamma above the domain",
        lambda: core.nimages.Levels(gray, black=200, white=255, auto_gamma=1),
        contains="below 128",
    )
    expect_error(
        "Levels peak_offset below zero",
        lambda: core.nimages.Levels(gray, black=0, white=255, peak_offset=-1),
        contains="between 0 and 255",
    )
    expect_error(
        "Levels gamma 0",
        lambda: core.nimages.Levels(gray, black=0, white=255, gamma=0.0),
        contains="gamma",
    )

    # use_props without a PeakStats upstream fails at frame time, not at create.
    plain = core.nimages.Levels(gray, use_props=True)
    expect_error("Levels use_props without the properties", lambda: plain.get_frame(0), contains="NImagesBlackLevel")


def check_determinism() -> None:
    section("determinism")
    rng = np.random.default_rng(13)
    page = rng.integers(0, 256, (40, 29), dtype=np.uint8)
    clip = clip_of(page, length=2)
    chain = core.nimages.Posterize(
        core.nimages.Levels(core.nimages.PeakStats(clip), use_props=True, auto_gamma=True), bits=4
    )
    first = array_of(chain, 0)
    second = array_of(chain, 0)
    same(first.tolist(), second.tolist(), "the same frame twice")
    same(array_of(chain, 1).tolist(), first.tolist(), "both frames hold the same page")


def main() -> int:
    frames = load("frames.json")
    peaks = load("peaks.json")
    shades = load("shades.json")
    levels = load("levels.json")
    posterize = load("posterize.json")

    check_registration()
    check_frame_peaks(frames["frames"], frames)
    check_histogram_peaks(peaks)
    check_frame_shades(frames["frames"], frames)
    check_histogram_shades(shades)
    check_levels(levels)
    check_posterize(posterize)
    check_levels_from_properties()
    check_property_preservation()
    check_analysis_leaves_pixels_alone()
    check_geometry()
    check_per_frame_results()
    check_request_patterns()
    check_errors()
    check_determinism()

    print()
    if failures:
        print(f"FAILED: {len(failures)} of {checks} checks", file=sys.stderr)
        for failure in failures[:40]:
            print(f"  - {failure}", file=sys.stderr)
        if len(failures) > 40:
            print(f"  ... and {len(failures) - 40} more", file=sys.stderr)
        return 1

    print(f"ok: {checks} checks passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
