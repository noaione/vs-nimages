#!/usr/bin/env python3
"""Generate the golden vectors used by the ``vs-nimages`` test suite.

The fixtures in ``tests/fixtures`` are committed so that the Rust and Python
tests need neither this script's dependencies nor the ``nao-manga-rls``
checkout. This script is the only thing that needs them.

Ground truth comes from the reference implementation in
``nao-manga-rls/nmanga/autolevel.py``:

* ``find_local_peak``      -> black and white peak levels for ``peaks.json``
* ``analyze_gray_shades``  -> shade/percentage pairs for ``shades.json``
* ``apply_levels``         -> 256-entry tables for ``levels.json``
* ``posterize_image_by_bits`` -> 256-entry tables for ``posterize.json``

``nmanga`` does not expose the ``*PeakFound`` flags or an automatic-gamma
helper, so those expected values come from the reference implementation below.
Every case that can be built as an image is validated against ``nmanga`` first
and generation aborts on any disagreement, so only the flags and the gamma
values rest on the reference alone.

Usage::

    python tools/golden.py --nmanga-path ../nao-manga-rls
    python tools/golden.py --check          # fail if fixtures are stale
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import scipy
from PIL import Image

REPO_ROOT = Path(__file__).resolve().parent.parent
FIXTURES = REPO_ROOT / "tests" / "fixtures"
FRAMES = FIXTURES / "frames"

DEFAULT_NMANGA_PATH = REPO_ROOT.parent / "nao-manga-rls"
SEED = 20240607

#: Name of the provenance file, which `--check` reports on but does not compare
#: as data: it records the interpreter and library versions a run used.
MANIFEST = "manifest.json"

# Pixel counts above this are kept as pure histograms: they still exercise the
# algorithm and the u64 arithmetic, but no image is synthesized to validate
# them against nmanga.
MAX_IMAGE_PIXELS = 1 << 22

# ---------------------------------------------------------------------------
# Reference implementation
#
# A clean-room port target for the Rust code. Validated against
# scipy.signal.find_peaks over 20k random arrays (local maxima) and 41 699
# (array, peak) pairs (prominence), and against nmanga.find_local_peak over
# 640 image/parameter combinations.
# ---------------------------------------------------------------------------


def local_maxima(x: np.ndarray) -> list[int]:
    """Plateau-aware local maxima, matching scipy's ``_local_maxima_1d``.

    A plateau spanning ``i..j-1`` yields one peak at ``(i + j - 1) // 2``.
    Array boundaries are never peaks.
    """
    peaks: list[int] = []
    size = x.size
    i = 1
    while i < size - 1:
        if int(x[i - 1]) < int(x[i]):
            j = i + 1
            while j < size and int(x[j]) == int(x[i]):
                j += 1
            if j == size:
                break
            if int(x[j]) < int(x[i]):
                peaks.append((i + j - 1) // 2)
            i = j
        else:
            i += 1
    return peaks


def prominence(x: np.ndarray, peak: int) -> int:
    """SciPy-compatible prominence.

    Walk each side while the sample is not strictly higher than the peak,
    tracking the minimum; the base is the higher of the two side minima.
    """
    height = int(x[peak])
    left_min = height
    j = peak - 1
    while j >= 0 and int(x[j]) <= height:
        left_min = min(left_min, int(x[j]))
        j -= 1
    right_min = height
    j = peak + 1
    while j < x.size and int(x[j]) <= height:
        right_min = min(right_min, int(x[j]))
        j += 1
    return height - max(left_min, right_min)


def _select_peak(
    padded: np.ndarray,
    total_pixels: int,
    peak_percentage: float | None,
    peak_prominence: float | None,
) -> tuple[int, bool]:
    """Returns ``(roi_index, found)`` for one padded region of interest."""
    min_pixels = math.ceil(total_pixels * (peak_percentage / 100.0)) if peak_percentage is not None else 0
    min_prominence = math.ceil(total_pixels * (peak_prominence / 100.0)) if peak_prominence is not None else 0

    def best(height: int, prom: int, use_prominence: bool) -> tuple[int, bool]:
        candidates: list[tuple[int, int]] = []
        for peak in local_maxima(padded):
            if int(padded[peak]) < height:
                continue
            if use_prominence and prominence(padded, peak) < prom:
                continue
            candidates.append((peak, int(padded[peak])))
        if not candidates:
            return -1, False
        tallest = max(count for _, count in candidates)
        # First (lowest-bin) candidate on a tie, matching numpy.argmax.
        return next(peak for peak, count in candidates if count == tallest) - 1, True

    index, found = best(min_pixels, min_prominence, peak_prominence is not None)
    if not found:
        index, found = best(0, 0, False)
    return index, found


def reference_peaks(
    hist: np.ndarray,
    total_pixels: int,
    *,
    upper_limit: int = 60,
    peak_percentage: float | None = 0.25,
    peak_prominence: float | None = None,
    skip_white: bool = False,
) -> tuple[int, int, bool, bool]:
    """Reference ``find_local_peak`` plus the two ``*PeakFound`` flags."""
    black_roi = hist[0 : upper_limit + 1]
    black, black_found = _select_peak(
        np.concatenate(([0], black_roi, [0])), total_pixels, peak_percentage, peak_prominence
    )
    if not black_found:
        black = 0

    if skip_white:
        return black, 255, black_found, False

    white_roi = hist[255 - upper_limit : 256]
    white_index, white_found = _select_peak(
        np.concatenate(([0], white_roi, [0])), total_pixels, peak_percentage, peak_prominence
    )
    white = 255 - upper_limit + white_index if white_found else 255
    return black, white, black_found, white_found


def reference_gamma(black_level: int) -> float:
    """``nmanga.gamma_correction``, which is the same expression."""
    black_point = black_level / 255
    internal = math.log(0.5) / math.log((0.5 - black_point) / (1.0 - black_point))
    return round(1 / internal, 2)


def reference_shades(hist: np.ndarray, total_pixels: int, threshold: float) -> list[dict[str, float]]:
    """Reference ``analyze_gray_shades`` over the fixed ``0..=255`` binning.

    ``nmanga`` omits ``range=(0, 256)`` when calling ``np.histogram``, so NumPy
    derives the bin edges from the data and the reported "shade" is a bin index
    over that adaptive range rather than a gray value. ``find_local_peak`` in the
    same module passes the explicit range. See ``docs/FINDINGS.md`` §5.
    """
    pixel_threshold = math.ceil(total_pixels * (threshold / 100.0))
    included = [(shade, int(hist[shade])) for shade in range(256) if int(hist[shade]) > pixel_threshold]
    # Stable descending sort by count keeps ascending shade order on ties.
    included.sort(key=lambda item: item[1], reverse=True)
    return [{"shade": shade, "percentage": (count / total_pixels) * 100.0} for shade, count in included]


def nmanga_shades(nmanga: Nmanga, image: Image.Image, threshold: float) -> list[dict[str, float]]:
    """``nmanga.analyze_gray_shades`` normalised to plain Python values."""
    return [
        {"shade": int(entry["shade"]), "percentage": float(entry["percentage"])}
        for entry in nmanga.analyze_gray_shades(image, threshold=threshold)
    ]


# ---------------------------------------------------------------------------
# nmanga loading
# ---------------------------------------------------------------------------


class Nmanga:
    """The reference functions, imported from the sibling checkout."""

    def __init__(self, path: Path) -> None:
        module_path = path / "nmanga" / "autolevel.py"
        if not module_path.is_file() or not (path / "nmanga" / "deblur.py").is_file():
            raise SystemExit(f"cannot find nmanga under {path}; pass --nmanga-path")
        sys.path.insert(0, str(path))
        try:
            import nmanga.autolevel  # ruff: ignore[unused-import]
            import nmanga.deblur  # ruff: ignore[unused-import]
        except ImportError as error:  # pragma: no cover - depends on the checkout
            raise SystemExit(
                f"cannot import nmanga from {path}: {error}\n"
                "run this script with the interpreter that has numpy, scipy and Pillow"
            ) from error
        self.module = sys.modules["nmanga.autolevel"]
        self.deblur = sys.modules["nmanga.deblur"]

    def __getattr__(self, name: str) -> Any:
        return getattr(self.module, name)


# ---------------------------------------------------------------------------
# Fixture building blocks
# ---------------------------------------------------------------------------


def ramp() -> Image.Image:
    return Image.fromarray(np.arange(256, dtype=np.uint8).reshape(1, 256), mode="L")


def shapes(hist: np.ndarray) -> Image.Image:
    """Builds an ``L`` image whose histogram is exactly ``hist``.

    The width is the largest divisor of the pixel count that is at most 512, so
    no padding pixel is ever added and the histogram round-trips exactly.
    """
    total = int(hist.sum())
    if total <= 0:
        raise ValueError("cannot build an image from an empty histogram")
    if total > MAX_IMAGE_PIXELS:
        raise ValueError(f"{total} pixels is too many to materialise")
    width = total
    for candidate in range(min(total, 512), 0, -1):
        if total % candidate == 0:
            width = candidate
            break
    values = np.repeat(np.arange(256, dtype=np.uint8), hist)
    return Image.fromarray(values.reshape(total // width, width), mode="L")


def try_shapes(hist: np.ndarray) -> Image.Image | None:
    """``shapes`` for cases whose pixel count is small enough to materialise."""
    try:
        return shapes(hist)
    except ValueError:
        return None


def hist_of(image: Image.Image) -> np.ndarray:
    pixels = np.asarray(image, dtype=np.uint8)
    return np.bincount(pixels.ravel(), minlength=256).astype(np.int64)


def sparse(**bins: int) -> np.ndarray:
    hist = np.zeros(256, dtype=np.int64)
    for name, count in bins.items():
        hist[int(name.removeprefix("b"))] = count
    return hist


def filled(hist: np.ndarray, pixels: int, shade: int = 200) -> np.ndarray:
    """Adds the pixels not yet accounted for to `shade`, so the sum is exact."""
    hist = hist.copy()
    remaining = pixels - int(hist.sum())
    if remaining < 0:
        raise AssertionError(f"counts already exceed {pixels} pixels")
    hist[shade] += remaining
    return hist


# ---------------------------------------------------------------------------
# Peak cases
# ---------------------------------------------------------------------------


def peak_parameters() -> list[dict[str, Any]]:
    """The parameter sets every peak histogram case is evaluated under."""
    return [
        {"label": "default", "upper_limit": 60, "peak_percentage": 0.25, "peak_prominence": None, "skip_white": False},
        {
            "label": "no-height",
            "upper_limit": 60,
            "peak_percentage": None,
            "peak_prominence": None,
            "skip_white": False,
        },
        {
            "label": "prominence",
            "upper_limit": 60,
            "peak_percentage": 0.25,
            "peak_prominence": 0.1,
            "skip_white": False,
        },
        {
            "label": "skip-white",
            "upper_limit": 60,
            "peak_percentage": 0.25,
            "peak_prominence": None,
            "skip_white": True,
        },
        {"label": "narrow", "upper_limit": 1, "peak_percentage": 0.25, "peak_prominence": None, "skip_white": False},
        {"label": "wide", "upper_limit": 255, "peak_percentage": 0.25, "peak_prominence": None, "skip_white": False},
        {"label": "zero-pct", "upper_limit": 60, "peak_percentage": 0.0, "peak_prominence": None, "skip_white": False},
        {
            "label": "full-pct",
            "upper_limit": 60,
            "peak_percentage": 100.0,
            "peak_prominence": None,
            "skip_white": False,
        },
    ]


def peak_histograms() -> list[tuple[str, np.ndarray, int]]:
    """``(name, histogram, total_pixels)`` triples covering IMPLEMENTATIONS §13.1."""
    total = 64 * 64
    cases: list[tuple[str, np.ndarray, int]] = []

    def add(name: str, hist: np.ndarray, pixels: int | None = None) -> None:
        if (hist < 0).any():
            raise AssertionError(f"{name}: a histogram count is negative")
        resolved = int(hist.sum()) if pixels is None else pixels
        if int(hist.sum()) > resolved:
            raise AssertionError(f"{name}: counts exceed the {resolved} pixel total")
        cases.append((name, hist, resolved))

    # Uniform images.
    add("all-black", sparse(b0=total))
    add("all-white", sparse(b255=total))
    add("uniform-mid-gray", sparse(b128=total))
    add("uniform-gray-1", sparse(b1=total))
    add("uniform-gray-60", sparse(b60=total))
    add("uniform-gray-254", sparse(b254=total))

    # Peaks exactly on the region boundaries.
    add("peak-at-0", sparse(b0=total))
    add("peak-at-upper-limit", filled(sparse(b60=2000, b10=1000), total))
    add("peak-at-255-minus-limit", filled(sparse(b195=2000, b10=1000), total))
    add("peak-at-255", filled(sparse(b255=2000, b10=1000), total))
    add("peak-at-0-and-255", filled(sparse(b0=1000, b255=1000), total, shade=128))

    # Selection.
    add("equal-height-peaks", filled(sparse(b10=900, b20=900), total))
    add("first-is-not-tallest", filled(sparse(b10=400, b20=2000), total))
    add("tallest-is-lowest-on-tie", sparse(b5=700, b40=700, b100=700))
    add("black-and-white-both-found", filled(sparse(b10=2000, b245=2000), total, shade=128))
    add("white-peak-is-tallest", filled(sparse(b10=400, b250=2000), total, shade=128))

    # Plateaus, odd and even.
    add("plateau-odd-3", filled(sparse(b10=600, b11=900, b12=600), total))
    add("plateau-even-2", filled(sparse(b10=800, b11=800), total))
    add("plateau-even-4", filled(sparse(b10=800, b11=800, b12=800, b13=800), total))
    add("plateau-even-6", filled(sparse(b10=600, b11=600, b12=600, b13=600, b14=600, b15=600), total))
    add("plateau-at-region-edge", filled(sparse(b0=800, b1=800), total))
    add("plateau-at-region-end", filled(sparse(b59=800, b60=800), total))

    # Height threshold boundaries: the default 0.25% of 4096 is ceil(10.24) = 11.
    add("height-exactly-at-threshold", filled(sparse(b10=11), total))
    add("height-one-below-threshold", filled(sparse(b10=10), total))
    add("height-one-above-threshold", filled(sparse(b10=12), total))
    add("height-below-threshold-fallback", filled(sparse(b10=5, b30=3), total))
    add("no-qualifying-peak-fallback-finds-tallest", filled(sparse(b10=9, b20=10), total))

    # Prominence. 0.1% of a 4096 pixel frame is ceil(4.096) = 5.
    add("prominence-pass", filled(sparse(b10=2000, b9=100, b11=100), total))

    # A high massif makes the tallest peak barely stand out, so it fails a
    # prominence threshold while a shorter isolated peak passes. This is the one
    # construction where the prominence argument changes which peak wins rather
    # than only shrinking the candidate set.
    massif = sparse(**{f"b{shade}": 2999 for shade in range(0, 21)})
    massif[10] = 3000
    massif[40] = 2500
    add("prominence-changes-the-winner", massif)

    # Every peak fails prominence, so the first pass is empty and the fallback
    # runs without any thresholds.
    massif_only = sparse(**{f"b{shade}": 2999 for shade in range(0, 61)})
    massif_only[10] = 3000
    add("prominence-empties-the-pass", massif_only)

    # Nothing to find at all.
    add("flat-below-height-only", sparse(b10=1, b11=1, b12=1))
    add("empty-histogram", np.zeros(256, dtype=np.int64), total)
    add("single-bin-below-everything", sparse(b0=1))

    # Larger scale so the counts are not near the thresholds.
    big = 1_000_000
    add("large-counts", filled(sparse(b12=400_000, b60=1000, b245=300_000), big, shade=128), big)

    # u64 arithmetic: counts beyond 32 bits.
    huge = 1 << 40
    add("u64-counts", sparse(b12=huge, b200=huge, b0=huge // 2, b255=huge // 2))
    add("u64-single-bin", sparse(b12=(1 << 48)))

    return cases


def build_peak_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    histograms: dict[str, list[int]] = {}
    cases: list[dict[str, Any]] = []
    reference_only = 0

    for name, hist, total in peak_histograms():
        histograms[name] = hist.tolist()
        image = try_shapes(hist)
        if image is not None and not np.array_equal(hist_of(image), hist):
            raise AssertionError(f"{name}: synthesized image histogram does not match")

        for params in peak_parameters():
            black, white, black_found, white_found = reference_peaks(
                hist,
                total,
                upper_limit=params["upper_limit"],
                peak_percentage=params["peak_percentage"],
                peak_prominence=params["peak_prominence"],
                skip_white=params["skip_white"],
            )
            case_name = f"{name}/{params['label']}"

            if image is None:
                reference_only += 1
            else:
                want = nmanga.find_local_peak(
                    image,
                    upper_limit=params["upper_limit"],
                    peak_percentage=params["peak_percentage"],
                    peak_prominence=params["peak_prominence"],
                    skip_white_check=params["skip_white"],
                )
                if (want[0], want[1]) != (black, white):
                    raise AssertionError(f"{case_name}: nmanga={want[:2]} reference={(black, white)}")

            cases.append({
                "name": case_name,
                "histogram": name,
                "total_pixels": total,
                "upper_limit": params["upper_limit"],
                "peak_percentage": params["peak_percentage"],
                "peak_prominence": params["peak_prominence"],
                "skip_white": params["skip_white"],
                "expect": {
                    "black": black,
                    "white": white,
                    "black_found": black_found,
                    "white_found": white_found,
                },
            })

    return {
        "histograms": histograms,
        "reference_only_cases": reference_only,
        "cases": cases,
    }


# ---------------------------------------------------------------------------
# Gray-shade cases
# ---------------------------------------------------------------------------


def shade_histograms() -> list[tuple[str, np.ndarray]]:
    total = 64 * 64
    cases: list[tuple[str, np.ndarray]] = []

    def add(name: str, hist: np.ndarray) -> None:
        if (hist < 0).any():
            raise AssertionError(f"{name}: a histogram count is negative")
        cases.append((name, hist))

    add("all-black", sparse(b0=total))
    add("all-white", sparse(b255=total))
    add("uniform-mid-gray", sparse(b128=total))
    add("two-shades", sparse(b0=total // 4, b255=total - total // 4))
    add("two-adjacent-shades", sparse(b10=total // 2, b11=total - total // 2))
    # Every shade from 10 to 137 present; the reference would report bin indices
    # over the observed 10..137 range instead of these values.
    narrow = np.zeros(256, dtype=np.int64)
    narrow[10:138] = 32
    add("narrow-range-spread", narrow)
    add("no-significant-shades", sparse(b10=1, b20=1, b30=1))
    add(
        "all-256-shades-equal",
        np.full(256, total // 256, dtype=np.int64),
    )
    add(
        "equal-counts-keep-ascending-order",
        sparse(b200=500, b10=500, b100=500, b5=500, b255=500),
    )
    add("descending-by-count", sparse(b10=3000, b20=800, b30=100, b40=total - 3900))
    # 0.01% of 4096 is ceil(0.4096) = 1, so a count of 1 is excluded and 2 is kept.
    add("exactly-at-threshold", sparse(b10=1, b20=2, b30=total - 3))
    add("default-threshold-boundary", sparse(b10=1, b11=total - 1))

    big = 1 << 32
    add("large-counts", sparse(b10=big, b200=big, b0=1))

    return cases


def shade_thresholds() -> list[float]:
    return [0.01, 0.0, 1.0, 50.0, 100.0]


def frame_shade_thresholds() -> list[float]:
    """A trimmed set: the frame cases are the expensive ones to serialise."""
    return [0.01, 0.0]


def build_shade_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    histograms: dict[str, list[int]] = {}
    cases: list[dict[str, Any]] = []
    diverging = 0

    for name, hist in shade_histograms():
        histograms[name] = hist.tolist()
        total = int(hist.sum())
        image = try_shapes(hist)
        if image is not None and not np.array_equal(hist_of(image), hist):
            raise AssertionError(f"{name}: synthesized image histogram does not match")

        for threshold in shade_thresholds():
            expected = reference_shades(hist, total, threshold)
            case: dict[str, Any] = {
                "name": f"{name}@{threshold:g}",
                "histogram": name,
                "total_pixels": total,
                "threshold": threshold,
                "expect": expected,
            }
            if image is None:
                # Too many pixels to materialise; the reference alone covers it.
                case["parity"] = "reference-only"
            else:
                from_nmanga = nmanga_shades(nmanga, image, threshold)
                if from_nmanga == expected:
                    case["parity"] = "nmanga"
                else:
                    # nmanga bins adaptively, so its shade indices differ. Kept
                    # in the fixture so the divergence stays visible and tested.
                    case["parity"] = "diverges"
                    case["nmanga_expect"] = from_nmanga
                    diverging += 1
            cases.append(case)

    return {"histograms": histograms, "diverging_cases": diverging, "cases": cases}


# ---------------------------------------------------------------------------
# Levels cases
# ---------------------------------------------------------------------------


def levels_parameters() -> list[dict[str, Any]]:
    return [
        {"black": 0.0, "white": 255.0, "gamma": 1.0},
        {"black": 10.0, "white": 245.0, "gamma": 1.0},
        {"black": 10.0, "white": 245.0, "gamma": 1.37},
        {"black": 10.0, "white": 245.0, "gamma": 0.73},
        {"black": 37.0, "white": 231.0, "gamma": 0.73},
        {"black": 12.0, "white": 245.0, "gamma": 1.18},
        {"black": 12.0, "white": 245.0, "gamma": 1.07},
        {"black": 1.0, "white": 254.0, "gamma": 1.0},
        {"black": 60.0, "white": 255.0, "gamma": 1.53},
        {"black": 5.0, "white": 250.0, "gamma": 1.05},
        {"black": 0.0, "white": 255.0, "gamma": 0.5},
        {"black": 0.0, "white": 255.0, "gamma": 2.0},
        {"black": 127.0, "white": 255.0, "gamma": 8.0},
    ]


def build_level_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    source = ramp()

    for params in levels_parameters():
        table = (
            np.asarray(nmanga.apply_levels(source, params["black"], params["white"], params["gamma"])).ravel().tolist()
        )
        assert len(table) == 256
        cases.append({**params, "expect": [int(value) for value in table]})

    gamma_cases = [
        {"black_level": black, "expect": reference_gamma(black)} for black in (0, 1, 5, 12, 37, 60, 100, 127)
    ]
    for case in gamma_cases:
        if case["expect"] != nmanga.gamma_correction(case["black_level"]):
            raise AssertionError(f"gamma_correction({case['black_level']}) disagrees with nmanga")

    # A black point at or above 128 makes the reference raise; the plugin must
    # reject it instead. Recorded so the validation test has a contract.
    invalid_gamma_levels = [128, 130, 200, 254, 255]

    return {
        "cases": cases,
        "gamma_cases": gamma_cases,
        "invalid_gamma_levels": invalid_gamma_levels,
    }


# ---------------------------------------------------------------------------
# Posterize cases
# ---------------------------------------------------------------------------


def build_posterize_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    cases: list[dict[str, Any]] = []
    source = ramp()

    for bits in range(1, 9):
        table = np.asarray(nmanga.posterize_image_by_bits(source, bits)).ravel().tolist()
        assert len(table) == 256
        colors = 2**bits
        levels = sorted(set(int(value) for value in table))
        if len(levels) != colors:
            raise AssertionError(f"bits={bits}: expected {colors} levels, got {len(levels)}")
        cases.append({
            "bits": bits,
            "colors": colors,
            "levels": levels,
            "expect": [int(value) for value in table],
        })

    return {"cases": cases}


# ---------------------------------------------------------------------------
# Deblur cases
# ---------------------------------------------------------------------------

#: Side of the synthetic page every deblur case runs on.
DEBLUR_SIZE = 64
#: Where the deblur inputs and expectations are written.
DEBLUR_DIR = FIXTURES / "deblur"

#: The tolerance each case is frozen at, keyed by how one sample is stored. One
#: 8 bit code value is 257 samples at 16 bits, and a float sample is measured in
#: the reference's own `[0, 1]` units, where one 8 bit step is `1 / 255`.
#:
#: The exact-match fraction is the 8 bit rule: the reference is float64 and the
#: plugin is float32, so at 16 bits a handful of samples land on the other side
#: of a rounding boundary and only the bounds hold.
DEBLUR_TOLERANCE = {
    "u8": {"max_abs_diff": 1.0, "mean_abs_diff": 0.05, "zero_fraction": 0.999},
    "u16": {"max_abs_diff": 257.0, "mean_abs_diff": 1.0, "zero_fraction": 0.99},
    "f32": {"max_abs_diff": 1.0 / 255.0, "mean_abs_diff": 1.0e-5, "zero_fraction": 0.0},
}


def deblur_page() -> np.ndarray:
    """The 64x64 page every deblur case starts from, in 0..255."""
    plane = np.zeros((DEBLUR_SIZE, DEBLUR_SIZE), dtype=np.float64)

    # A soft ramp, which is the shape the deconvolution is there to undo.
    plane[:, 0:16] = np.linspace(60.0, 200.0, 16)
    # A hard step, a one pixel line beside it, and a low contrast step.
    plane[:, 16:32] = 40.0
    plane[:, 24:32] = 220.0
    plane[8:24, 28] = 255.0
    plane[40:56, 20:22] = 48.0
    # Flat mid gray holding a black square and a white square.
    plane[:, 32:48] = 128.0
    plane[16:32, 34:40] = 0.0
    plane[40:52, 42:46] = 255.0
    # A soft dark to light edge, with a striped texture across it.
    plane[:, 48:64] = np.linspace(20.0, 240.0, 16)
    plane[::4, 48:64] = 0.0
    return plane


def deblur_gray8() -> np.ndarray:
    return np.rint(deblur_page()).clip(0, 255).astype(np.uint8)


def deblur_rgb8() -> np.ndarray:
    """The same page in colour, so the equal-offset rule and its gamut limit
    have channel differences and clipped pixels to act on."""
    page = deblur_page()
    rgb = np.stack([page, page * 0.88 + 8.0, page * 0.70 + 26.0], axis=-1)
    return np.rint(np.clip(rgb, 0.0, 255.0)).astype(np.uint8)


def deblur_chroma(width: int, height: int) -> tuple[np.ndarray, np.ndarray]:
    """Two distinct chroma planes, so "chroma is untouched" means something."""
    row = np.broadcast_to(np.arange(height, dtype=np.float64).reshape(-1, 1), (height, width))
    column = np.broadcast_to(np.arange(width, dtype=np.float64).reshape(1, -1), (height, width))
    return (90.0 + column, 160.0 - row * 0.5)


def deblur_smoothstep(values: np.ndarray) -> np.ndarray:
    values = np.clip(values, 0, 1)
    return values * values * (3 - 2 * values)


def deblur_edge_mask(y: np.ndarray, threshold: float) -> np.ndarray:
    """`nmanga.deblur.edge_mask`, over `[0, 1]` values."""
    ndimage = scipy.ndimage
    base = ndimage.gaussian_filter(y, 0.5, mode="reflect")
    gradient = np.hypot(
        ndimage.sobel(base, axis=0, mode="reflect"),
        ndimage.sobel(base, axis=1, mode="reflect"),
    )
    magnitude = gradient * (255.0 / 8.0)
    mask = deblur_smoothstep((magnitude - threshold) / max(3 * threshold, 1e-6))
    return ndimage.gaussian_filter(mask, 0.45, mode="reflect")


def deblur_luma_stage(
    y: np.ndarray,
    *,
    method: int,
    radius: float,
    strength: float,
    iterations: int,
    threshold: float,
    overshoot: float,
) -> np.ndarray:
    """`nmanga.deblur`, over the luma plane in `[0, 1]`."""
    ndimage = scipy.ndimage
    if method == 0:
        # Positive pedestal: it keeps the multiplicative update away from
        # zero-locking at pure black.
        pedestal = 1 / 255
        observed = y + pedestal
        estimate = observed.copy()
        for _ in range(iterations):
            blurred = ndimage.gaussian_filter(estimate, radius, mode="reflect", truncate=4)
            ratio = observed / np.maximum(blurred, 1e-7)
            estimate = estimate * ndimage.gaussian_filter(ratio, radius, mode="reflect", truncate=4)
        candidate = y + strength * (estimate - pedestal - y)
    else:
        candidate = y + strength * (y - ndimage.gaussian_filter(y, radius, mode="reflect", truncate=4))

    low = ndimage.minimum_filter(y, size=3, mode="reflect") - overshoot / 255
    high = ndimage.maximum_filter(y, size=3, mode="reflect") + overshoot / 255
    candidate = np.clip(candidate, low, high)
    return np.clip(y + deblur_edge_mask(y, threshold) * (candidate - y), 0, 1)


def deblur_luma_expect(plane: np.ndarray, max_value: float, params: dict[str, Any]) -> np.ndarray:
    """The restored plane for a format whose luma is a plane of its own."""
    y = plane.astype(np.float64) / max_value
    return deblur_luma_stage(y, **params)


def deblur_rgb_expect(rgb: np.ndarray, params: dict[str, Any]) -> np.ndarray:
    """`nmanga.deblur`: one equal offset per pixel, limited to the gamut the
    pixel has left."""
    page = rgb.astype(np.float64) / 255.0
    y = page @ np.array([0.2126, 0.7152, 0.0722])
    delta = deblur_luma_stage(y, **params) - y
    delta = np.clip(delta, -page.min(axis=2), 1 - page.max(axis=2))
    return np.rint(np.clip(page + delta[..., None], 0, 1) * 255).astype(np.uint8)


def deblur_parameters(method: int, **overrides: Any) -> dict[str, Any]:
    """The defaults of whichever method is in effect, plus any override."""
    return {
        "method": method,
        "radius": 0.8,
        "strength": 0.65 if method == 0 else 0.85,
        "iterations": 6,
        "threshold": 2.0,
        "overshoot": 0.0,
        **overrides,
    }


def deblur_case_list() -> list[dict[str, Any]]:
    """Every case, before anything is written."""
    gray8 = deblur_gray8()
    rgb8 = deblur_rgb8()
    gray16 = gray8.astype(np.uint16) * np.uint16(257)
    gray32 = gray8.astype(np.float32) / np.float32(255.0)
    cases: list[dict[str, Any]] = []

    def restored(plane: np.ndarray, sample: str, max_value: float, params: dict[str, Any]) -> np.ndarray:
        """The restored luma, rounded back onto the sample's own grid."""
        values = deblur_luma_expect(plane, max_value, params)
        if sample == "u8":
            return np.rint(values * 255.0).clip(0, 255).astype(np.uint8)
        if sample == "u16":
            return np.rint(values * 65535.0).clip(0, 65535).astype(np.uint16)
        return values.astype(np.float32)

    def gray(name: str, vapour: str, plane: np.ndarray, sample: str, max_value: float, params: dict[str, Any]) -> None:
        cases.append({
            "name": name,
            "vapour": vapour,
            "format": "Gray",
            "sample": sample,
            "planes": [plane],
            "expect": [restored(plane, sample, max_value, params)],
            "params": params,
        })

    def yuv(name: str, vapour: str, sub_w: int, sub_h: int, params: dict[str, Any]) -> None:
        u, v = deblur_chroma(DEBLUR_SIZE >> sub_w, DEBLUR_SIZE >> sub_h)
        u = np.rint(u).clip(0, 255).astype(np.uint8)
        v = np.rint(v).clip(0, 255).astype(np.uint8)
        cases.append({
            "name": name,
            "vapour": vapour,
            "format": "YUV",
            "sample": "u8",
            "planes": [gray8, u, v],
            "expect": [restored(gray8, "u8", 255.0, params), u, v],
            "params": params,
        })

    def rgb(name: str, params: dict[str, Any]) -> None:
        page = deblur_rgb_expect(rgb8, params)
        cases.append({
            "name": name,
            "vapour": "RGB24",
            "format": "RGB",
            "sample": "u8",
            "planes": [np.ascontiguousarray(rgb8[:, :, channel]) for channel in range(3)],
            "expect": [np.ascontiguousarray(page[:, :, channel]) for channel in range(3)],
            "params": params,
        })

    gray("gray8/method0", "GRAY8", gray8, "u8", 255.0, deblur_parameters(0))
    gray("gray8/method1", "GRAY8", gray8, "u8", 255.0, deblur_parameters(1))
    gray(
        "gray8/wide",
        "GRAY8",
        gray8,
        "u8",
        255.0,
        deblur_parameters(0, radius=2.0, strength=0.9, iterations=3, threshold=1.0, overshoot=1.0),
    )
    gray("gray8/threshold0", "GRAY8", gray8, "u8", 255.0, deblur_parameters(1, threshold=0.0))
    gray("gray16/method0", "GRAY16", gray16, "u16", 65535.0, deblur_parameters(0))
    gray("gray16/method1", "GRAY16", gray16, "u16", 65535.0, deblur_parameters(1))
    gray("gray32/method0", "GRAYS", gray32, "f32", 1.0, deblur_parameters(0))
    gray("gray32/method1", "GRAYS", gray32, "f32", 1.0, deblur_parameters(1))
    rgb("rgb24/method0", deblur_parameters(0))
    rgb("rgb24/method1", deblur_parameters(1))
    yuv("yuv444p8/method0", "YUV444P8", 0, 0, deblur_parameters(0))
    yuv("yuv422p8/method0", "YUV422P8", 1, 0, deblur_parameters(0))
    yuv("yuv420p8/method0", "YUV420P8", 1, 1, deblur_parameters(0))
    yuv("yuv420p8/overshoot", "YUV420P8", 1, 1, deblur_parameters(0, overshoot=2.0))
    return cases


def write_deblur_plane(name: str, plane: np.ndarray) -> dict[str, Any]:
    path = DEBLUR_DIR / f"{name}.bin"
    payload = np.ascontiguousarray(plane).tobytes()
    path.write_bytes(payload)
    return {
        "file": f"deblur/{path.name}",
        "width": int(plane.shape[1]),
        "height": int(plane.shape[0]),
        "bytes": len(payload),
        "sha256": hashlib.sha256(payload).hexdigest(),
    }


def cross_check_deblur(nmanga: Nmanga) -> None:
    """Refuse to write the cases when the port and nmanga disagree.

    The colour path is the one nmanga exposes, and it shares its luma stage
    with every other case, so agreeing on it is what keeps the expectations
    attached to the reference rather than to this file.
    """
    rgb8 = deblur_rgb8()
    alpha = np.full((DEBLUR_SIZE, DEBLUR_SIZE, 1), 255, dtype=np.uint8)
    image = Image.fromarray(np.concatenate([rgb8, alpha], axis=-1), "RGBA")

    for method in (0, 1):
        params = deblur_parameters(method)
        shared = {
            "radius": params["radius"],
            "strength": params["strength"],
            "threshold": params["threshold"],
            "overshoot": params["overshoot"],
        }
        if method == 0:
            want = np.asarray(nmanga.deblur.deblur_deconv(image, iterations=params["iterations"], **shared))
        else:
            want = np.asarray(nmanga.deblur.deblur_edge_sharp(image, **shared))
        got = np.concatenate([deblur_rgb_expect(rgb8, params), alpha], axis=-1)
        if not np.array_equal(want, got):
            differing = int((want != got).sum())
            raise AssertionError(f"deblur method {method}: the port and nmanga differ on {differing} samples")


def build_deblur_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    cross_check_deblur(nmanga)
    DEBLUR_DIR.mkdir(parents=True, exist_ok=True)
    cases: list[dict[str, Any]] = []

    for case in deblur_case_list():
        slug = case["name"].replace("/", "-")
        planes = [write_deblur_plane(f"{slug}.in{index}", plane) for index, plane in enumerate(case["planes"])]
        expect = [write_deblur_plane(f"{slug}.out{index}", plane) for index, plane in enumerate(case["expect"])]
        cases.append({
            "name": case["name"],
            "vapour": case["vapour"],
            "format": case["format"],
            "sample": case["sample"],
            "width": int(case["planes"][0].shape[1]),
            "height": int(case["planes"][0].shape[0]),
            "planes": planes,
            "expect": expect,
            "tolerance": DEBLUR_TOLERANCE[case["sample"]],
            **case["params"],
        })

    return {"page": DEBLUR_SIZE, "cases": cases}

# ---------------------------------------------------------------------------
# Frame fixtures
# ---------------------------------------------------------------------------


def frame_images() -> list[tuple[str, Image.Image]]:
    rng = np.random.default_rng(SEED)
    frames: list[tuple[str, Image.Image]] = []

    frames.append(("tiny-1x1", Image.new("L", (1, 1), 0)))
    frames.append(("tiny-1x1-mid", Image.new("L", (1, 1), 128)))
    frames.append(("odd-13x7", Image.fromarray(rng.integers(0, 256, (7, 13), dtype=np.uint8), "L")))
    frames.append(("odd-17x33", Image.fromarray(rng.integers(0, 256, (33, 17), dtype=np.uint8), "L")))
    frames.append(("ramp-256x1", ramp()))
    frames.append(("flat-64x64-mid", Image.new("L", (64, 64), 128)))
    frames.append(("flat-64x64-black", Image.new("L", (64, 64), 0)))
    frames.append(("flat-64x64-white", Image.new("L", (64, 64), 255)))

    halftone = rng.integers(0, 256, (64, 64), dtype=np.uint8)
    halftone = np.where(halftone < 128, rng.integers(0, 24, halftone.shape), rng.integers(232, 256, halftone.shape))
    frames.append(("halftone-64x64", Image.fromarray(halftone.astype(np.uint8), "L")))

    frames.append(("lowkey-48x32", Image.fromarray(rng.integers(0, 70, (32, 48), dtype=np.uint8), "L")))
    frames.append(("highkey-48x32", Image.fromarray(rng.integers(190, 256, (32, 48), dtype=np.uint8), "L")))
    frames.append(("wide-256x8", Image.fromarray(rng.integers(0, 256, (8, 256), dtype=np.uint8), "L")))
    frames.append(("tall-8x256", Image.fromarray(rng.integers(0, 256, (256, 8), dtype=np.uint8), "L")))

    return frames


def build_frame_fixtures(nmanga: Nmanga) -> dict[str, Any]:
    FRAMES.mkdir(parents=True, exist_ok=True)
    index: dict[str, Any] = {}

    for name, image in frame_images():
        path = FRAMES / f"{name}.bin"
        payload = np.asarray(image, dtype=np.uint8).tobytes()
        path.write_bytes(payload)
        index[name] = {
            "width": image.width,
            "height": image.height,
            "file": f"frames/{name}.bin",
            "bytes": len(payload),
            "sha256": hashlib.sha256(payload).hexdigest(),
            "hist": hist_of(image).tolist(),
        }

    # Peak expectations for the real frames, so the integration tests have
    # ground truth that covers actual stride-padded reads.
    peak_cases: list[dict[str, Any]] = []
    for name, image in frame_images():
        hist = hist_of(image)
        total = image.width * image.height
        for params in peak_parameters():
            black, white, black_found, white_found = reference_peaks(
                hist,
                total,
                upper_limit=params["upper_limit"],
                peak_percentage=params["peak_percentage"],
                peak_prominence=params["peak_prominence"],
                skip_white=params["skip_white"],
            )
            want = nmanga.find_local_peak(
                image,
                upper_limit=params["upper_limit"],
                peak_percentage=params["peak_percentage"],
                peak_prominence=params["peak_prominence"],
                skip_white_check=params["skip_white"],
            )
            if (want[0], want[1]) != (black, white):
                raise AssertionError(f"frame {name}/{params['label']}: nmanga={want[:2]} reference={(black, white)}")
            peak_cases.append({
                "name": f"{name}/{params['label']}",
                "frame": name,
                "upper_limit": params["upper_limit"],
                "peak_percentage": params["peak_percentage"],
                "peak_prominence": params["peak_prominence"],
                "skip_white": params["skip_white"],
                "expect": {
                    "black": black,
                    "white": white,
                    "black_found": black_found,
                    "white_found": white_found,
                },
            })

    shade_cases: list[dict[str, Any]] = []
    for name, image in frame_images():
        hist = hist_of(image)
        total = image.width * image.height
        for threshold in frame_shade_thresholds():
            expected = reference_shades(hist, total, threshold)
            from_nmanga = nmanga_shades(nmanga, image, threshold)
            case: dict[str, Any] = {
                "name": f"{name}@{threshold:g}",
                "frame": name,
                "threshold": threshold,
                "expect": expected,
            }
            if from_nmanga == expected:
                case["parity"] = "nmanga"
            else:
                case["parity"] = "diverges"
                case["nmanga_expect"] = from_nmanga
            shade_cases.append(case)

    return {"frames": index, "peak_cases": peak_cases, "shade_cases": shade_cases}


# ---------------------------------------------------------------------------
# Self-check against scipy
# ---------------------------------------------------------------------------


def self_check() -> None:
    from scipy.signal import find_peaks

    rng = np.random.default_rng(SEED)
    maxima_checked = 0
    prom_checked = 0

    for _ in range(4000):
        size = int(rng.integers(3, 16))
        values = rng.integers(0, 5, size=size).astype(np.int64)
        want = find_peaks(values)[0].tolist()
        got = local_maxima(values)
        if want != got:
            raise AssertionError(f"local maxima mismatch: {values.tolist()} scipy={want} reference={got}")
        maxima_checked += 1

        peaks, props = find_peaks(values, prominence=0)
        for peak, prom in zip(peaks.tolist(), props["prominences"].tolist()):
            if prominence(values, peak) != int(prom):
                raise AssertionError(f"prominence mismatch: {values.tolist()} peak={peak}")
            prom_checked += 1

    print(f"self-check: {maxima_checked} arrays, {prom_checked} prominences match scipy")


# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------


def write(path: Path, payload: Any) -> bytes:
    text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
    data = text.encode("utf-8")
    path.write_bytes(data)
    return data


def manifest_matches(manifest: dict[str, Any]) -> bool:
    """Whether the committed manifest records this run's provenance."""
    path = FIXTURES / MANIFEST
    if not path.is_file():
        return False
    try:
        committed = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return committed.get("versions") == manifest["versions"]


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--nmanga-path", type=Path, default=DEFAULT_NMANGA_PATH)
    parser.add_argument("--check", action="store_true", help="verify the committed fixtures are up to date")
    parser.add_argument("--skip-self-check", action="store_true")
    args = parser.parse_args()

    if not args.skip_self_check:
        self_check()

    nmanga = Nmanga(args.nmanga_path.expanduser().resolve())

    FIXTURES.mkdir(parents=True, exist_ok=True)
    outputs: dict[str, Any] = {}
    outputs["frames.json"] = build_frame_fixtures(nmanga)
    outputs["peaks.json"] = build_peak_fixtures(nmanga)
    outputs["shades.json"] = build_shade_fixtures(nmanga)
    outputs["levels.json"] = build_level_fixtures(nmanga)
    outputs["posterize.json"] = build_posterize_fixtures(nmanga)
    outputs["deblur.json"] = build_deblur_fixtures(nmanga)

    manifest = {
        "generator": "tools/golden.py",
        "reference": "nao-manga-rls/nmanga/autolevel.py and nmanga/deblur.py",
        "seed": SEED,
        "versions": {
            "python": sys.version.split()[0],
            "numpy": np.__version__,
            "scipy": scipy.__version__,
            "pillow": Image.__version__,
        },
        "files": sorted(outputs),
    }
    outputs[MANIFEST] = manifest

    if args.check:
        # The manifest records the interpreter and library versions a run used,
        # so it differs between machines even when the data is current. Compare
        # the data files, then report the manifest separately.
        stale = []
        for name, payload in sorted(outputs.items()):
            if name == MANIFEST:
                continue
            path = FIXTURES / name
            text = json.dumps(payload, indent=2, sort_keys=False, ensure_ascii=False) + "\n"
            if not path.is_file() or path.read_text(encoding="utf-8") != text:
                stale.append(name)
        if stale:
            print(f"stale fixtures: {', '.join(stale)}", file=sys.stderr)
            return 1

        print("fixtures are up to date")
        if not manifest_matches(manifest):
            print(
                f"note: {MANIFEST} records a different interpreter or library "
                "version than this run; the data is unaffected"
            )
        return 0

    for name, payload in sorted(outputs.items()):
        data = write(FIXTURES / name, payload)
        print(f"wrote {name} ({len(data):,} bytes)")

    print(f"peak cases: {len(outputs['peaks.json']['cases'])}")
    print(f"shade cases: {len(outputs['shades.json']['cases'])}")
    print(f"levels cases: {len(outputs['levels.json']['cases'])}")
    print(f"posterize cases: {len(outputs['posterize.json']['cases'])}")
    print(f"deblur cases: {len(outputs['deblur.json']['cases'])}")
    print(f"frames: {len(outputs['frames.json']['frames'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
