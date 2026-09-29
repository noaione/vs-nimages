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


def lloyd_reference(counts: np.ndarray, colors: int) -> np.ndarray:
    """The `lloyd.py` solver over one histogram, as a full code-value table.

    This is an independent description of the reference the plugin's Lloyd path
    is checked against: 40 passes move every interior level to the mean of its
    bucket, both ends stay pinned, and the last assignment maps each code value
    to its nearest level.
    """
    maximum = counts.size - 1
    hist = counts.astype(np.float64)
    x = np.arange(counts.size)
    levels = np.linspace(0, maximum, colors)
    for _ in range(40):
        edges = np.concatenate([[-1.0], (levels[:-1] + levels[1:]) / 2, [maximum + 1.0]])
        bucket = np.digitize(x, edges) - 1
        for j in range(1, colors - 1):
            mask = bucket == j
            if hist[mask].sum() > 0:
                levels[j] = (hist[mask] * x[mask]).sum() / hist[mask].sum()
    edges = np.concatenate([[-1.0], (levels[:-1] + levels[1:]) / 2, [maximum + 1.0]])
    return np.round(levels[np.digitize(x, edges) - 1]).astype(np.uint16)


def check_lloyd_posterize() -> None:
    section("Lloyd-Max levels from the frame histogram")
    # Four clusters, one per bucket at four levels, so every interior level has
    # mass and the pinned ends stay on the code-value extremes.
    page = np.zeros((32, 64), dtype=np.uint8)
    page[:8, :] = 40
    page[8:16, :] = 90
    page[16:24, :] = 200
    page[24:, :] = 250
    clip = clip_of(page)
    counts = np.bincount(page.ravel(), minlength=256)

    for bits in (1, 2, 3, 4, 6):
        expected = lloyd_reference(counts, 1 << bits)
        mapped = array_of(core.nimages.Posterize(clip, bits=bits, method=1))
        same(mapped.tolist(), expected[page].tolist(), f"GRAY8 bits={bits}")

    # The default still spaces the levels evenly, so the two methods differ here.
    even = array_of(core.nimages.Posterize(clip, bits=3))
    lloyd = array_of(core.nimages.Posterize(clip, bits=3, method=1))
    check(even.tolist() != lloyd.tolist(), "method=1 must not repeat the even levels")

    # `bits` equal to the sample depth is the identity for both methods.
    identity = array_of(core.nimages.Posterize(clip, bits=8, method=1))
    same(identity.tolist(), page.tolist(), "GRAY8 bits=8 method=1 is the identity")

    # Two clusters leave a bucket empty, and it keeps the even level it started
    # on, which is what the reference does.
    sparse = np.zeros((8, 8), dtype=np.uint8)
    sparse[:, :4] = 40
    sparse[:, 4:] = 200
    sparse_clip = clip_of(sparse)
    sparse_counts = np.bincount(sparse.ravel(), minlength=256)
    expected = lloyd_reference(sparse_counts, 4)
    same(
        array_of(core.nimages.Posterize(sparse_clip, bits=2, method=1)).tolist(),
        expected[sparse].tolist(),
        "GRAY8 bits=2 on a two-cluster page",
    )

    # The same solver on a 16-bit page, whose histogram has one bin per code value.
    values = np.zeros((16, 16), dtype=np.uint16)
    values[:4, :] = 1_000
    values[4:8, :] = 9_000
    values[8:12, :] = 40_000
    values[12:, :] = 60_000
    wide = clip_of_planes([values], vs.GRAY16, width=16, height=16)
    wide_counts = np.bincount(values.ravel(), minlength=65_536)
    expected = lloyd_reference(wide_counts, 16)
    with core.nimages.Posterize(wide, bits=4, method=1).get_frame(0) as frame:
        same(np.asarray(frame[0]).tolist(), expected[values].tolist(), "GRAY16 bits=4")

    # `use_props` still picks the level count from the shades, and Lloyd picks
    # where the levels sit.
    shades = core.nimages.PeakGrayShades(clip)
    shade_count = len(as_list(props_of(shades)["NImagesGrayShades"]))
    bits = 1 if shade_count <= 2 else math.ceil(math.log2(shade_count))
    expected = lloyd_reference(counts, 1 << bits)
    same(
        array_of(core.nimages.Posterize(shades, use_props=True, method=1)).tolist(),
        expected[page].tolist(),
        "use_props picks the count and Lloyd the levels",
    )

    expect_error(
        "Posterize method 2",
        lambda: core.nimages.Posterize(clip, bits=2, method=2),
        contains="method must be 0 (even levels) or 1 (Lloyd-Max)",
    )


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

    expect_error(
        "PeakStats on RGB", lambda: core.nimages.PeakStats(rgb), contains="Gray 8 to 16 bit"
    )
    expect_error(
        "PeakGrayShades on YUV",
        lambda: core.nimages.PeakGrayShades(core.std.BlankClip(width=8, height=8, format=vs.YUV420P8)),
        contains="Gray 8 to 16 bit",
    )
    expect_error(
        "Posterize bits above GRAY16 depth",
        lambda: core.nimages.Posterize(gray16, bits=17),
        contains="between 1 and 16",
    )
    float_rgb = core.std.BlankClip(width=8, height=8, format=vs.RGBS)
    expect_error(
        "Levels use_props on float RGB",
        lambda: core.nimages.Levels(float_rgb, use_props=True),
        contains="do not support use_props=True",
    )
    expect_error(
        "Levels auto_gamma on float RGB",
        lambda: core.nimages.Levels(float_rgb, auto_gamma=1),
        contains="auto_gamma=True",
    )
    expect_error(
        "Levels peak_offset on float RGB",
        lambda: core.nimages.Levels(float_rgb, peak_offset=1),
        contains="nonzero peak_offset",
    )
    expect_error(
        "Levels fractional black on an integer clip",
        lambda: core.nimages.Levels(gray, black=10.5),
        contains="whole number of code values",
    )
    expect_error(
        "Levels negative black on an integer clip",
        lambda: core.nimages.Levels(gray, black=-1, white=200),
        contains="between 0 and 255",
    )
    expect_error(
        "Levels white above the sample maximum",
        lambda: core.nimages.Levels(gray, black=0, white=256),
        contains="between 0 and 255",
    )
    expect_error(
        "Levels infinite white",
        lambda: core.nimages.Levels(gray, black=0, white=float("inf")),
        contains="finite number",
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
        contains="half of the sample range",
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

    # 4:2:0 subsamples chroma, and VapourSynth refuses an odd width or height
    # for it, so every format in the map is built at an even size.
    width, height = 6, 4
    for name, fmt in formats.items():
        blank = core.std.BlankClip(width=width, height=height, format=fmt, length=1)
        shapes = [
            np.asarray(blank.get_frame(0)[plane]).shape
            for plane in range(blank.format.num_planes)
        ]
        planes = [
            np.resize(np.roll(values, plane), shape).astype(np.uint16, copy=True)
            for plane, shape in enumerate(shapes)
        ]
        source = clip_of_planes(planes, fmt, width=width, height=height)

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
                        np.rint(
                            (input_values.astype(np.float64) - 1_024)
                            * (65_535 / (60_000 - 1_024))
                        ),
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


def check_high_depth_analysis() -> None:
    section("higher-depth gray analysis")
    values = np.full((7, 5), 32_768, dtype=np.uint16)
    values[:3, :] = 15_420
    values[3:5, :] = 50_115
    source = clip_of_planes([values], vs.GRAY16, width=5, height=7)

    stats = core.nimages.PeakStats(source, upper_limit=60, peak_percentage=1.0)
    props = props_of(stats)
    same(props["NImagesBlackLevel"], 15_420, "native black peak above 255")
    same(props["NImagesWhiteLevel"], 50_115, "native white peak above 255")
    same(array_of(stats).tolist(), values.tolist(), "16-bit analysis leaves pixels alone")

    leveled = core.nimages.Levels(stats, use_props=True, auto_gamma=False)
    adjusted = array_of(leveled)
    same(int(adjusted[0, 0]), 0, "native use_props maps the black level to zero")
    same(int(adjusted[3, 0]), 65_535, "native use_props maps the white level to maximum")
    check(0 < int(adjusted[6, 0]) < 65_535, "native use_props maps midtones between endpoints")

    shades = props_of(core.nimages.PeakGrayShades(source, threshold=20.0))
    same(
        as_list(shades["NImagesGrayShades"]),
        [15_420, 32_768, 50_115],
        "native shade values",
    )
    same(
        as_list(shades["NImagesGrayShadePercentages"]),
        [15 / 35 * 100, 10 / 35 * 100, 10 / 35 * 100],
        "native shade percentages follow count order",
    )


def check_float_levels() -> None:
    section("GrayS and RGBS levels")
    values = np.asarray(
        [[-np.inf, -0.5, 0.0, 0.25, 0.5], [0.75, 1.0, 1.5, np.inf, np.nan]],
        dtype=np.float32,
    )
    source = clip_of_planes([values], vs.GRAYS, width=5, height=2)
    output = array_of(core.nimages.Levels(source, black=63.75, white=191.25))
    expected = np.asarray(
        [[0.0, 0.0, 0.0, 0.0, 0.5], [1.0, 1.0, 1.0, 1.0, np.nan]],
        dtype=np.float32,
    )
    same(np.isnan(output).tolist(), np.isnan(expected).tolist(), "NaN samples stay NaN")
    same(
        np.nan_to_num(output, nan=-99).tolist(),
        np.nan_to_num(expected, nan=-99).tolist(),
        "float samples are clamped and mapped per sample",
    )

    rgb_planes = [
        np.asarray([[0.0, 0.5, 1.0]], dtype=np.float32),
        np.asarray([[0.25, 0.5, 0.75]], dtype=np.float32),
        np.asarray([[1.0, 0.5, 0.0]], dtype=np.float32),
    ]
    rgb = clip_of_planes(rgb_planes, vs.RGBS, width=3, height=1)
    adjusted = core.nimages.Levels(rgb, black=63.75, white=191.25, gamma=2.0)
    with adjusted.get_frame(0) as frame:
        expected_planes = [
            np.asarray([[0.0, np.sqrt(0.5), 1.0]], dtype=np.float32),
            np.asarray([[0.0, np.sqrt(0.5), 1.0]], dtype=np.float32),
            np.asarray([[1.0, np.sqrt(0.5), 0.0]], dtype=np.float32),
        ]
        for plane, expected_plane in enumerate(expected_planes):
            check(
                np.allclose(np.asarray(frame[plane]), expected_plane, rtol=0, atol=1e-7),
                f"RGBS plane {plane} is mapped independently",
            )


def check_unified_level_endpoints() -> None:
    section("one endpoint argument set for both domains")
    # Omitted endpoints keep their defaults: 0 and the sample maximum on an integer
    # clip, and 0 and 255 (1.0 once scaled) on a float one.
    ramp8 = np.arange(256, dtype=np.uint8).reshape(1, 256)
    ramp16 = (np.arange(256, dtype=np.uint16) * 257).reshape(1, 256)
    for name, values, fmt in (
        ("GRAY8", ramp8, vs.GRAY8),
        ("GRAY16", ramp16, vs.GRAY16),
    ):
        source = clip_of_planes([values], fmt, width=256, height=1)
        same(
            array_of(core.nimages.Levels(source)).tolist(),
            values.tolist(),
            f"{name}: the default endpoints are the identity",
        )

    floats = np.linspace(0.0, 1.0, 256, dtype=np.float32).reshape(1, 256)
    gray_s = clip_of_planes([floats], vs.GRAYS, width=256, height=1)
    same(
        array_of(core.nimages.Levels(gray_s)).tolist(),
        floats.tolist(),
        "GRAYS: the default endpoints are the identity",
    )
    rgb_planes = [floats * 0.5, 1.0 - floats, floats * 0.25]
    rgb_s = clip_of_planes(rgb_planes, vs.RGBS, width=256, height=1)
    with core.nimages.Levels(rgb_s).get_frame(0) as frame:
        for plane, values in enumerate(rgb_planes):
            same(
                np.asarray(frame[plane]).tolist(),
                values.tolist(),
                f"RGBS plane {plane}: the default endpoints are the identity",
            )

    # The same numbers are code values on an integer clip and 8-bit float units on a
    # float one, which is the point of the unified arguments.
    leveled8 = array_of(core.nimages.Levels(clip_of(ramp8), black=51, white=204)).ravel()
    same(int(leveled8[51]), 0, "integer endpoints are code values: black maps to 0")
    same(int(leveled8[204]), 255, "integer endpoints are code values: white maps to 255")
    same(int(leveled8[2]), 0, "values below the black point clamp to 0")
    same(int(leveled8[250]), 255, "values above the white point clamp to 255")

    exact = clip_of_planes(
        [np.asarray([[0.25, 0.75]], dtype=np.float32)], vs.GRAYS, width=2, height=1
    )
    same(
        array_of(core.nimages.Levels(exact, black=63.75, white=191.25)).tolist(),
        [[0.0, 1.0]],
        "float endpoints are 8-bit units: 63.75 is 0.25 and 191.25 is 0.75",
    )

    values = np.asarray(
        [[-1.0, -0.25, 0.0, 0.1, 0.5, 0.75, 0.94, 1.0, 1.5, 2.0, np.nan]], dtype=np.float32
    )
    source = clip_of_planes([values], vs.GRAYS, width=11, height=1)
    for black, white in (
        (0.0, 255.0),
        (0.0, 1.0),
        (0.0, 245.0),
        (12.75, 239.7),
        (-25.5, 280.5),
        (63.75, 191.25),
    ):
        output = array_of(core.nimages.Levels(source, black=black, white=white))
        scaled = (values.astype(np.float64) - black / 255.0) / (white / 255.0 - black / 255.0)
        expected = np.clip(scaled, 0.0, 1.0).astype(np.float32)
        same(
            np.isnan(output).tolist(),
            np.isnan(expected).tolist(),
            f"black={black} white={white}: NaN stays NaN",
        )
        same(
            np.nan_to_num(output, nan=-99).tolist(),
            np.nan_to_num(expected, nan=-99).tolist(),
            f"black={black} white={white}: endpoints are scaled by 255",
        )


def check_mixed_depth_levels() -> None:
    section("a clip whose depth varies between frames")
    # `imgseqs` reports `Undefined` for a sequence that mixes depths, so the
    # endpoints resolve against each frame instead of at creation.
    scratch = REPO_ROOT / "target" / "check-nimages"
    shutil.rmtree(scratch, ignore_errors=True)
    scratch.mkdir(parents=True, exist_ok=True)

    page8 = np.arange(256, dtype=np.uint8).reshape(1, 256)
    page16 = (np.arange(256, dtype=np.uint16) * 257).reshape(1, 256)
    paths = [scratch / "depth8.pgm", scratch / "depth16.pgm"]
    paths[0].write_bytes(b"P5\n256 1\n255\n" + page8.tobytes())
    # A 16-bit PGM stores big-endian words.
    paths[1].write_bytes(b"P5\n256 1\n65535\n" + page16.astype(">u2").tobytes())

    clip = core.imgseqs.Read(files=[str(path) for path in paths], mismatch=True, prefetch=0)
    check(clip.format.color_family == vs.ColorFamily.UNDEFINED, "the source reports no fixed format")

    leveled = core.nimages.Levels(clip, black=10, white=200, gamma=1.0)
    fixed8 = core.nimages.Levels(clip_of(page8), black=10, white=200, gamma=1.0)
    fixed16 = core.nimages.Levels(
        clip_of_planes([page16], vs.GRAY16, width=256, height=1),
        black=10,
        white=200,
        gamma=1.0,
    )
    with leveled.get_frame(0) as frame:
        same(frame.format.bits_per_sample, 8, "frame 0 is 8 bit")
    with leveled.get_frame(1) as frame:
        same(frame.format.bits_per_sample, 16, "frame 1 is 16 bit")
    same(
        array_of(leveled, 0).tolist(),
        array_of(fixed8, 0).tolist(),
        "the 8 bit frame matches the same curve on a fixed 8 bit clip",
    )
    same(
        array_of(leveled, 1).tolist(),
        array_of(fixed16, 0).tolist(),
        "the 16 bit frame matches the same curve on a fixed 16 bit clip",
    )

    shutil.rmtree(scratch, ignore_errors=True)


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
    check_lloyd_posterize()
    check_levels_from_properties()
    check_property_preservation()
    check_analysis_leaves_pixels_alone()
    check_geometry()
    check_per_frame_results()
    check_request_patterns()
    check_errors()
    check_color_families(posterize)
    check_high_depth_integer_mapping()
    check_high_depth_analysis()
    check_float_levels()
    check_unified_level_endpoints()
    check_mixed_depth_levels()
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
