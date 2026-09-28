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
import shutil
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
    expect_error(
        "PeakGrayShades on YUV",
        lambda: core.nimages.PeakGrayShades(core.std.BlankClip(width=8, height=8, format=vs.YUV420P8)),
        contains="Gray 8 bit",
    )
    expect_error(
        "Posterize bits above GRAY16 depth",
        lambda: core.nimages.Posterize(gray16, bits=17),
        contains="between 1 and 16",
    )
    expect_error(
        "Levels on float RGB",
        lambda: core.nimages.Levels(core.std.BlankClip(width=8, height=8, format=vs.RGBS)),
        contains="8 to 16 bit integer",
    )
    expect_error("PeakStats without a clip", lambda: core.nimages.PeakStats())  # type: ignore
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
    expect_error("Posterize without bits", lambda: core.nimages.Posterize(gray), contains="bits is required")  # type: ignore
    expect_error("Posterize bits 0", lambda: core.nimages.Posterize(gray, bits=0), contains="between 1 and 8")
    expect_error(
        "Posterize bits 9",
        lambda: core.nimages.Posterize(gray, bits=9),
        contains="between 1 and 8",
    )
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
        contains="half the sample range",
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


def posterize_table(fixtures: dict, bits: int) -> np.ndarray:
    for case in fixtures["cases"]:
        if case["bits"] == bits:
            return np.asarray(case["expect"], dtype=np.uint8)
    raise AssertionError(f"no posterize fixture for bits={bits}")


def check_color_families(posterize: dict) -> None:
    section("color families the mapping filters take")
    table = posterize_table(posterize, 3)
    families = {
        "GRAY8": vs.GRAY8,
        "RGB24": vs.RGB24,
        "YUV420P8": vs.YUV420P8,
        "YUV422P8": vs.YUV422P8,
        "YUV444P8": vs.YUV444P8,
    }
    rng = np.random.default_rng(17)

    for name, fmt in families.items():
        blank = core.std.BlankClip(width=16, height=8, format=fmt, length=1)
        shapes = [
            np.asarray(blank.get_frame(0)[plane]).shape
            for plane in range(blank.format.num_planes)
        ]
        # Fixed planes, so asking for frame 0 twice yields the same samples.
        planes = [rng.integers(0, 256, shape, dtype=np.uint8) for shape in shapes]
        source = clip_of_planes(planes, fmt, width=16, height=8)

        identity = core.nimages.Levels(source, black=0, white=255, gamma=1.0)
        for plane, values in enumerate(planes):
            same(
                np.asarray(identity.get_frame(0)[plane]).tolist(),
                values.tolist(),
                f"{name}: identity levels on plane {plane}",
            )

        posterized = core.nimages.Posterize(source, bits=3)
        for plane, values in enumerate(planes):
            same(
                np.asarray(posterized.get_frame(0)[plane]).tolist(),
                table[values].tolist(),
                f"{name}: posterize on plane {plane}",
            )

    # An RGB clip is three channels, so each must come back holding only what its
    # own values map to. Bleeding between channels shows up as a mismatch here.
    shaped = clip_of_rgb((9, 3), base=10, step=60)
    before = [np.asarray(shaped.get_frame(0)[plane]).copy() for plane in range(3)]
    same(int(before[0][0, 0]), 10, "red starts at 10")
    same(int(before[1][0, 0]), 70, "green starts at 70")
    same(int(before[2][0, 0]), 130, "blue starts at 130")

    posterized = core.nimages.Posterize(shaped, bits=3).get_frame(0)
    for plane in range(3):
        same(
            np.asarray(posterized[plane]).tolist(),
            table[before[plane]].tolist(),
            f"plane {plane} keeps its own channel",
        )


def clip_of_planes(planes: list[np.ndarray], format_, *, width: int, height: int):
    """Builds a one frame clip holding exactly `planes`."""
    blank = core.std.BlankClip(width=width, height=height, format=format_, length=1)

    def fill(n: int, f):
        del n
        out = f.copy()
        for plane, values in enumerate(planes):
            np.asarray(out[plane])[:] = values
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def round_ratio_ties_even(numerator: int, denominator: int) -> int:
    quotient, remainder = divmod(numerator, denominator)
    return quotient + int(
        remainder * 2 > denominator
        or (remainder * 2 == denominator and quotient % 2 != 0)
    )


def expected_posterize_u16(values: np.ndarray, bits: int) -> np.ndarray:
    maximum = 65_535
    levels = (1 << bits) - 1
    mapped = [
        round_ratio_ties_even(
            round_ratio_ties_even(int(value) * levels, maximum) * maximum,
            levels,
        )
        for value in values.flat
    ]
    return np.asarray(mapped, dtype=np.uint16).reshape(values.shape)


def check_high_depth_integer_mapping() -> None:
    section("higher-depth integer mapping")
    formats = {
        "GRAY16": vs.GRAY16,
        "RGB48": vs.RGB48,
        "YUV420P16": vs.YUV420P16,
    }
    values = np.asarray([0, 1_024, 30_500, 60_000, 65_535], dtype=np.uint16)

    for name, fmt in formats.items():
        blank = core.std.BlankClip(width=5, height=3, format=fmt, length=1)
        shapes = [
            np.asarray(blank.get_frame(0)[plane]).shape
            for plane in range(blank.format.num_planes)
        ]
        planes = [
            np.resize(np.roll(values, plane), shape).astype(np.uint16, copy=True)
            for plane, shape in enumerate(shapes)
        ]
        source = clip_of_planes(planes, fmt, width=5, height=3)

        identity = core.nimages.Levels(source, gamma=1.0)
        with identity.get_frame(0) as frame:
            for plane, expected in enumerate(planes):
                same(
                    np.asarray(frame[plane]).tolist(),
                    expected.tolist(),
                    f"{name}: native-range default Levels plane {plane}",
                )

        leveled = core.nimages.Levels(source, black=1_024, white=60_000, gamma=1.0)
        with leveled.get_frame(0) as frame:
            for plane, input_values in enumerate(planes):
                expected = np.where(
                    input_values < 1_024,
                    0,
                    np.where(
                        input_values > 60_000,
                        65_535,
                        np.rint((input_values.astype(np.float64) - 1_024)
                                * (65_535 / (60_000 - 1_024))),
                    ),
                ).astype(np.uint16)
                same(
                    np.asarray(frame[plane]).tolist(),
                    expected.tolist(),
                    f"{name}: Levels plane {plane}",
                )

        posterized = core.nimages.Posterize(source, bits=5)
        with posterized.get_frame(0) as frame:
            for plane, input_values in enumerate(planes):
                same(
                    np.asarray(frame[plane]).tolist(),
                    expected_posterize_u16(input_values, 5).tolist(),
                    f"{name}: Posterize plane {plane}",
                )


def clip_of_rgb(shape: tuple[int, int], *, base: int, step: int):
    width, height = shape
    blank = core.std.BlankClip(width=width, height=height, format=vs.RGB24, length=1)

    def fill(n: int, f):
        del n
        out = f.copy()
        for plane in range(3):
            row = np.arange(width, dtype=np.uint8) + base + plane * step
            np.asarray(out[plane])[:] = np.tile(row, (height, 1))
        return out

    return core.std.ModifyFrame(blank, blank, fill)


def check_dynamic_dimensions(posterize: dict) -> None:
    section("dynamic dimensions")
    table = posterize_table(posterize, 3)
    scratch = REPO_ROOT / "target" / "check-nimages"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(23)
    sizes = [(7, 5), (13, 3), (9, 9), (1, 1), (64, 2)]
    pages = []
    files = []
    for index, (width, height) in enumerate(sizes):
        page = rng.integers(0, 256, (height, width), dtype=np.uint8)
        path = scratch / f"page{index}.pgm"
        path.write_bytes(b"P5\n%d %d\n255\n" % (width, height) + page.tobytes())
        files.append(str(path))
        pages.append(page)

    clip = core.imgseqs.Read(files=files, mismatch=True, prefetch=0)
    same((clip.width, clip.height), (0, 0), "the source reports variable dimensions")

    nodes = (
        ("PeakStats", core.nimages.PeakStats(clip)),
        ("PeakGrayShades", core.nimages.PeakGrayShades(clip)),
        ("Levels", core.nimages.Levels(clip, black=10, white=200, gamma=1.0)),
        ("Posterize", core.nimages.Posterize(clip, bits=3)),
    )
    for label, node in nodes:
        for n, (width, height) in enumerate(sizes):
            with node.get_frame(n) as frame:
                check(
                    (frame.width, frame.height) == (width, height),
                    f"{label} frame {n}: got {frame.width}x{frame.height}, want {width}x{height}",
                )
                if label == "Posterize":
                    same(
                        np.asarray(frame[0]).tolist(),
                        table[pages[n]].tolist(),
                        f"{label} frame {n} pixels",
                    )

    shutil.rmtree(scratch, ignore_errors=True)


def check_debug_logging() -> None:
    section("debug logging")
    captured: list[str] = []
    handle = core.add_log_handler(lambda message_type, message: captured.append(message.strip()))

    def ours() -> list[str]:
        return [line for line in captured if line.startswith("[nimages][debug]")]

    try:
        page = np.full((9, 9), 100, dtype=np.uint8)
        clip = clip_of(page, length=2)

        # nothing is written unless asked for
        captured.clear()
        core.nimages.Posterize(clip, bits=4).get_frame(0)
        same(ours(), [], "silent without debug")

        # each filter reports its settings once, on the first frame it is asked
        # for, because VapourSynth drops a message logged from `create`
        stats = core.nimages.PeakStats(clip, debug=1)
        shades = core.nimages.PeakGrayShades(clip, debug=1)
        levels = core.nimages.Levels(stats, use_props=True, auto_gamma=True, debug=1)
        posterize = core.nimages.Posterize(levels, bits=4, debug=1)
        fixed = core.nimages.Levels(clip, black=10, white=200, gamma=1.0, debug=1)

        settings = (
            ("PeakStats", stats, "upper_limit=60"),
            ("PeakGrayShades", shades, "threshold=0.01"),
            ("Levels", levels, "use_props=true"),
            ("Posterize", posterize, "bits=4 colors=16"),
            ("Levels-constant", fixed, "resolved black=10 white=200"),
        )
        for label, node, argument in settings:
            captured.clear()
            with node.get_frame(1):
                pass
            function = label.split("-")[0]
            check(
                any(f"[nimages][debug] {function}: " in line and argument in line for line in ours()),
                f"{label}: a settings line naming {argument}",
            )
            check(
                any(
                    f"[nimages][debug] {function}: " in line and "Gray 8 bit 9x9" in line
                    for line in ours()
                ),
                f"{label}: a settings line naming the input",
            )
            check(
                any(f"{function} frame 1:" in line and "total=" in line for line in ours()),
                f"{label}: a per-frame timing line for frame 1",
            )

        # the settings line is written once, not once per frame
        captured.clear()
        with stats.get_frame(0):
            pass
        same(
            [line for line in ours() if line.startswith("[nimages][debug] PeakStats: ")],
            [],
            "PeakStats: settings are not repeated on a later frame",
        )
        # the lines carry the frame number the caller asked for
        captured.clear()
        with posterize.get_frame(0):
            pass
        check(
            any("Posterize frame 0:" in line for line in ours()),
            "Posterize: names frame 0",
        )
        check(
            not any("Posterize frame 1:" in line for line in ours()),
            "Posterize: does not name a frame that was not asked for",
        )
    finally:
        core.remove_log_handler(handle)


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
    check_color_families(posterize)
    check_high_depth_integer_mapping()
    check_dynamic_dimensions(posterize)
    check_debug_logging()
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
