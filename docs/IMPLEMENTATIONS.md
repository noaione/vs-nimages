# vapoursynth-nimages Implementation Plan

Status: proposed  
Project: `vapoursynth-nimages`  
Description: A collection of analyzer and tooling to manipulate images  
Reverse-domain namespace / plugin identifier: `xyz.n4o.nimages`  
Callable plugin namespace: `nimages`  
Preferred implementation: Rust  
Initial target: VapourSynth API 4, CPU-resident `GRAY8` clips

## 1. Summary

Implement the image-analysis and pixel-processing operations currently found in
[`nmanga/autolevel.py`](../nmanga/autolevel.py) as one native VapourSynth plugin
named `vapoursynth-nimages`, described as "A collection of analyzer and tooling
to manipulate images". The plugin should expose independent filters for peak
statistics, gray-shade analysis, levels, and posterization.

This is feasible in either Rust or C++. Rust is the preferred implementation:
the algorithms are small, frame-local, and do not require Python, Pillow,
NumPy, or SciPy after they have been ported. The `vapoursynth-rs` project
supports VapourSynth API 4 filter creation, frame access, properties, and
plugin export. None of the proposed filters require the API 4.3 GPU additions.

For `GRAY8` input, levels and posterization should use 256-entry lookup tables.
Peak and gray-shade analysis should use the same reusable 256-bin histogram
implementation and attach their results as frame properties. Because they are
independent filters, each scans its own requested input frame. A future version
may add 9-16-bit integer and 32-bit float support.

## 2. Goals

- Port `find_local_peak` without a SciPy runtime dependency.
- Port `analyze_gray_shades` without a NumPy runtime dependency.
- Port `apply_levels` with clearly defined ImageMagick-compatible semantics.
- Port `posterize_image_by_bits` without dithering.
- Process each image-sequence frame independently and safely in parallel.
- Preserve source frame properties such as `ImgSeqPath`.
- Make per-frame statistics available to Python through frame properties.
- Match the existing Python behavior where it is intentional, while documenting
  and correcting existing validation and unit-conversion bugs.
- Test the native results against Python and ImageMagick golden references.

## 3. Non-goals for the first version

- Loading image files; `vapoursynth-imageseqs` already provides this.
- Reimplementing general colorspace conversion.
- GPU/Vulkan processing.
- Reproducing `find_local_peak_legacy` and SciPy's CWT implementation.
- Supporting arbitrary YUV, subsampled, alpha, or variable sample-type input.
- Maintaining floating-point HDRI precision in the first `GRAY8` release.

## 4. Existing behavior to preserve

### 4.1 Peak detection

The current `find_local_peak` implementation:

1. Converts non-`L` Pillow images to grayscale.
2. Builds a 256-bin histogram over values 0 through 255.
3. Searches the inclusive black region `0..=upper_limit`.
4. Pads the region with one zero bin at each end so shade 0 and the upper
   boundary can be detected as peaks.
5. Applies a minimum height derived from `peak_percentage`.
6. Optionally applies a minimum prominence derived from `peak_prominence`.
7. Selects the tallest qualifying peak, not the first peak.
8. Falls back to the tallest unthresholded local peak when no peak qualifies.
9. Repeats the process for the inclusive white region
   `(255 - upper_limit)..=255`, unless white detection is disabled.
10. Defaults to black 0 and white 255 when a peak is unavailable.

Both percentage parameters represent percentages, not fractions. For example,
`0.25` means 0.25 percent of all pixels:

```text
minimum_count = ceil(total_pixels * percentage / 100)
```

### 4.2 Gray-shade analysis

The current `analyze_gray_shades` implementation:

1. Converts non-`L` Pillow images to grayscale.
2. Builds a 256-bin histogram.
3. Calculates `pixel_threshold = ceil(total_pixels * threshold / 100)`.
4. Includes a shade only when its count is strictly greater than the pixel
   threshold.
5. Reports each included shade and its percentage of the complete image.
6. Sorts results by descending percentage, retaining ascending shade order
   when percentages are equal.

The default `threshold=0.01` means 0.01 percent, not a fraction of one. When no
shade exceeds the threshold, the result is an empty list.

### 4.3 Levels

For an input value `x`, black point `b`, white point `w`, output maximum `Q`,
and gamma `g`, the intended transfer curve is:

```text
x < b  -> 0
x > w  -> Q
else   -> round(Q * ((x - b) / (w - b)) ** (1 / g))
```

For the current Pillow path, `Q` is 255. The same mapping is applied to every
processed band.

The existing automatic gamma calculation must be ported separately and tested
as part of the compatibility suite:

```text
black_normalized = black_level / 255
internal_gamma = log(0.5) / log((0.5 - black_normalized) /
                                (1.0 - black_normalized))
level_gamma = round(1.0 / internal_gamma, 2)
```

### 4.4 Posterization

The current grayscale posterization mapping uses `colors = 2 ** bits`:

```text
round(x * (colors - 1) / 255) * 255 / (colors - 1)
```

Pillow then calls `quantize(colors=colors, dither=NONE)` and converts the
result back to `L`. That quantization should normally be redundant after every
pixel has already been mapped to one of exactly `colors` gray values. It must
be proven redundant with golden tests before it is omitted from the native
implementation.

## 5. Proposed public plugin interface

Configure the plugin with:

```text
project/package name: vapoursynth-nimages
plugin identifier:    xyz.n4o.nimages
plugin namespace:     nimages
display name:         vapoursynth-nimages
description:          A collection of analyzer and tooling to manipulate images
```

VapourSynth distinguishes its globally unique reverse-domain plugin identifier
from its short callable namespace. The requested `xyz.n4o.nimages` value is the
plugin identifier used for lookup and collision prevention. The callable
namespace is `nimages`, producing calls such as `core.nimages.PeakStats(...)`.
Using the dotted identifier as the callable namespace would prevent ordinary
Python attribute syntax.

Exact registration syntax may be adjusted to match the Rust binding, but the
Python surface should remain equivalent to the following design.

### 5.1 `PeakStats`

```python
stats = core.nimages.PeakStats(
    clip,
    upper_limit=60,
    peak_percentage=0.25,
    peak_prominence=0.1,
    skip_white=False,
)
```

Input and output pixels are unchanged. The returned frames contain:

| Property | Type | Meaning |
|---|---:|---|
| `NImagesBlackLevel` | integer | Selected black peak, in source sample units |
| `NImagesWhiteLevel` | integer | Selected white peak, in source sample units |
| `NImagesBlackPeakFound` | integer boolean | Whether a black peak was found |
| `NImagesWhitePeakFound` | integer boolean | Whether a white peak was found |

Example property access:

```python
with stats.get_frame(n) as frame:
    black = frame.props["NImagesBlackLevel"]
    white = frame.props["NImagesWhiteLevel"]
```

Frame properties are required because VapourSynth evaluates frames lazily. A
filter-construction call cannot return statistics that have not yet been
calculated for every requested frame.

### 5.2 `PeakGrayShades`

```python
shades = core.nimages.PeakGrayShades(
    clip,
    threshold=0.01,
)
```

Input and output pixels are unchanged. The returned frames contain parallel
property arrays:

| Property | Type | Meaning |
|---|---:|---|
| `NImagesGrayShades` | integer array | Significant shade values, sorted by percentage |
| `NImagesGrayShadePercentages` | float array | Percentage for each corresponding shade |

The two parallel arrays must always have identical lengths and indexes. For an
empty analysis, store empty arrays if supported by the API/binding; otherwise
omit both properties together. The chosen empty-array behavior must be
documented and tested.

Example property access:

```python
with shades.get_frame(n) as frame:
    shade_values = list(frame.props.get("NImagesGrayShades", []))
    percentages = list(frame.props.get("NImagesGrayShadePercentages", []))
```

### 5.3 `Levels`

Support two modes:

```python
# Constant parameters, with the LUT constructed once.
leveled = core.nimages.Levels(
    clip,
    black=12,
    white=245,
    gamma=1.18,
)

# Per-frame parameters produced by PeakStats.
leveled = core.nimages.Levels(
    stats,
    use_props=True,
    peak_offset=0,
    auto_gamma=True,
)
```

When `use_props=True`, the filter reads `NImagesBlackLevel` and
`NImagesWhiteLevel` from the current input frame. A 256-entry LUT can be built
per frame; this cost is negligible relative to scanning and writing the image.

### 5.4 `Posterize`

```python
posterized = core.nimages.Posterize(clip, bits=4)
```

The first release accepts `GRAY8`, requires `bits` in `1..=8`, disables
dithering, and returns `GRAY8`.

The behavior for future RGB input must be explicit. To match the Python
function exactly, an RGB input would first be converted to gray and the output
would be grayscale. Applying posterization independently to RGB planes is a
different operation and should use a different option or filter name.

Peak analysis and levels deliberately remain separate. Users should compose
`PeakStats` and `Levels(use_props=True)` when automatic per-frame levels are
needed. This keeps analysis observable and reusable, avoids a second API for
the same policy, and lets callers inspect or override the detected properties
between the two filters.

## 6. Input format policy

### 6.1 First release

- CPU-resident frames only.
- Constant sample type and bit depth.
- `GRAY8` only.
- Dimensions must be read from each frame rather than assumed from global video
  information, so differing page dimensions can be supported where the Rust
  binding and source clip allow them.
- Iterate only over `width` valid samples in each row; never include stride
  padding in histograms or output.

Normalize an image-sequence clip before invoking the plugin. For example:

```python
import vapoursynth as vs

core = vs.core
src = core.imgseqs.Read(files=files, mismatch=True)

gray = core.resize.Bicubic(
    src,
    format=vs.GRAY8,
    matrix_s="470bg",
    range_s="full",
)
```

BT.470BG uses nominal `0.299 R + 0.587 G + 0.114 B` luma coefficients and is a
reasonable match for Pillow-style grayscale conversion. Golden tests must
still verify whether its rounding matches Pillow closely enough for peak
selection near bin boundaries.

### 6.2 Later extensions

- Integer `GRAY9` through `GRAY16`.
- `RGB24` input.
- RGB levels applied independently to each plane.
- Internal RGB-to-gray conversion for peak statistics and grayscale
  posterization.
- `GRAYS` and `RGBS` float processing for HDRI-like levels.
- Dynamic/variable-format clip handling.

For higher-depth peak detection, choose and document one of these policies:

1. Quantize input values into the same 256 analysis bins and continue returning
   8-bit-equivalent black and white values.
2. Use one bin per native integer sample and return native levels.

The second option is more precise; the first is more compatible with the
existing Python algorithm.

## 7. Peak detection algorithm

### 7.1 Histogram

- Use `[u64; 256]` for `GRAY8` counts.
- Count pixels row-by-row, respecting stride.
- Calculate `total_pixels` using a checked or sufficiently wide integer.
- Derive threshold counts with `ceil`, matching Python.

### 7.2 Candidate peaks

Reproduce the relevant behavior of `scipy.signal.find_peaks`:

- A non-plateau peak must be higher than both neighboring samples.
- A flat plateau is one peak if both outer neighbors are lower.
- Use the center of a plateau; for an even-length plateau, round the center
  downward as SciPy does.
- Treat a virtual zero bin as the neighbor outside each ROI boundary.
- Do not treat an all-zero ROI as a peak.

### 7.3 Prominence

Prominence can be implemented directly without a signal-processing library:

1. From the peak, scan left until reaching the boundary or a sample higher
   than the peak.
2. Record the minimum sample on that side.
3. Repeat to the right.
4. The reference base height is the higher of the two side minima.
5. Prominence is `peak_height - reference_base_height`.

The ROI contains at most 256 samples, so a straightforward scan for each peak
is preferable to a more complex optimization.

### 7.4 Selection and fallback

- Filter candidates by minimum height and optional minimum prominence.
- Choose the candidate with the greatest height.
- Preserve the lowest-bin candidate on equal heights to match the first-index
  behavior of `numpy.argmax`.
- If no candidate qualifies, repeat peak selection without the height or
  prominence thresholds.
- If no fallback candidate exists, return black 0 or white 255 and set the
  corresponding `PeakFound` property to false.

### 7.5 Gray-shade analysis

`PeakGrayShades` reuses the same `[u64; 256]` histogram implementation:

1. Validate that `threshold` is finite and non-negative.
2. Calculate `pixel_threshold = ceil(total_pixels * threshold / 100)`.
3. Visit shades in ascending numerical order.
4. Retain shades whose count is strictly greater than `pixel_threshold`.
5. Calculate `percentage = count / total_pixels * 100` as `f64`.
6. Sort by descending count/percentage, retaining ascending shade order for
   equal counts.
7. Write shade and percentage as parallel frame-property arrays. Pixel counts
   remain an internal implementation detail used for filtering and sorting.

Sorting by integer count avoids floating-point ordering ambiguity because every
percentage has the same `total_pixels` denominator. Percentages should still be
stored at full `f64` precision rather than rounded for display.

## 8. Levels and HDRI compatibility

### 8.1 Same curve, different precision

The Python LUT and ImageMagick `-level` use the same mathematical transfer
curve. ImageMagick's MagickCore implementation calculates:

```text
QuantumRange * pow(
    (pixel - black_point) / (white_point - black_point),
    1 / gamma,
)
```

and clamps the leveled image to its supported output range.

The implementations are not always byte-identical:

- `apply_levels` samples the curve at the 256 possible 8-bit inputs and rounds
  immediately to 8-bit.
- Q16-HDRI calculates using floating-point samples over an internal quantum
  range normally equal to 65535, then quantizes when exporting to 8-bit.
- Python `round()` uses ties-to-even.
- ImageMagick's integer export effectively uses half-up rounding.

An exhaustive `GRAY8` ramp comparison performed with ImageMagick 7.1.2-24
Q16-HDRI produced:

```text
black=10, white=245, gamma=1.00  -> 0 differing values
black=10, white=245, gamma=1.37  -> 1 differing value, by 1
black=37, white=231, gamma=0.73  -> 1 differing value, by 1
```

Therefore:

- A 256-entry LUT is appropriate for `GRAY8` and is practically equivalent to
  a single ImageMagick HDRI level operation followed by 8-bit output.
- Exact byte parity is not guaranteed because a small number of results may
  differ by one code value.
- A 256-entry LUT is not HDRI-equivalent for 16-bit, float, or multi-operation
  processing because it discards intermediate precision.

To provide genuine HDRI-like behavior later, accept `GRAYS`/`RGBS`, calculate
the formula per floating-point sample, and clamp only where required by the
defined filter contract. A 65536-entry LUT is also viable for 16-bit integer
input, but not for float input.

### 8.2 Colorspace and channel behavior

Neither the existing Pillow LUT nor ImageMagick `-level` inherently converts
the stored channel values to linear light before applying the curve. HDRI
describes storage and calculation precision, not automatic colorspace
linearization.

The native filter must define its channel behavior explicitly:

- `GRAY8`: process plane 0.
- Future RGB: process R, G, and B independently.
- Alpha: leave unchanged unless a future explicit `planes`/`process_alpha`
  option requests otherwise.

This differs from blindly multiplying a Pillow point table by the number of
bands, which can modify alpha as well.

## 9. Existing issues to fix or define

### 9.1 Percentage validation

The current checks use an impossible conjunction:

```python
value <= 0 and value >= 100
```

The plugin should reject values outside the documented range or use `or`, not
silently retain invalid input. Recommended contract:

```text
0 < peak_percentage <= 100
0 < peak_prominence <= 100
```

Allow absence of either option to disable that threshold.

### 9.2 Level-domain validation

Validate at filter construction when parameters are constant, and per frame
when they come from properties:

- `black < white`
- `gamma > 0`
- finite floating-point parameters
- adjusted black point remains below the white point

Automatic gamma additionally requires a black point for which its logarithm
expression is defined. The default `upper_limit=60` is safe, but callers must
not be able to trigger a NaN or domain error with an excessively large black
level.

### 9.3 Rounding policy

Choose one named compatibility mode:

- `pillow`: ties-to-even and exact 8-bit LUT compatibility.
- `imagemagick`: half-up output quantization.

Alternatively, define Pillow compatibility as the only first-version contract.
The choice must be covered by exhaustive tests over all 256 inputs.

### 9.4 CLI `peak_offset` units

The ImageMagick CLI path currently converts the black level to a percentage and
then adds `peak_offset` directly. If `peak_offset` is intended to represent an
8-bit code-value offset, adding 1 currently means one percentage point, or
about 2.55 code values.

Calculate the percentage after applying the offset:

```text
black_percent = (black_level + peak_offset) / 255 * 100
```

Do not use:

```text
black_level / 255 * 100 + peak_offset
```

### 9.5 Posterization rounding

Rust's `f64::round()` rounds midpoint values away from zero, unlike Python's
ties-to-even `round()`. The Rust algorithm should use an explicit ties-to-even
helper or an integer formulation proven to produce the same mapping.

## 10. Rust project layout

Recommended standalone plugin layout:

```text
vapoursynth-nimages/
  Cargo.toml
  README.md
  src/
    lib.rs
    histogram.rs
    peaks.rs
    gray_shades.rs
    levels.rs
    posterize.rs
    filters/
      mod.rs
      peak_stats.rs
      peak_gray_shades.rs
      levels.rs
      posterize.rs
  tests/
    fixtures/
```

Build the library as `cdylib`. Keep VapourSynth-specific and unsafe FFI code at
the registration/filter boundary. Histogram, peak detection, LUT construction,
and parameter validation should be safe, pure Rust functions.

Possible core types:

```rust
struct PeakOptions {
    upper_limit: u8,
    peak_percentage: Option<f64>,
    peak_prominence: Option<f64>,
    skip_white: bool,
}

struct PeakResult {
    black: u8,
    white: u8,
    black_found: bool,
    white_found: bool,
}

fn histogram_u8(
    data: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> [u64; 256];

fn find_local_peaks(
    histogram: &[u64; 256],
    total_pixels: u64,
    options: &PeakOptions,
) -> PeakResult;

struct GrayShade {
    shade: u8,
    count: u64,
    percentage: f64,
}

fn peak_gray_shades(
    histogram: &[u64; 256],
    total_pixels: u64,
    threshold: f64,
) -> Vec<GrayShade>;

fn levels_lut(black: f64, white: f64, gamma: f64) -> [u8; 256];

fn posterize_lut(bits: u8) -> [u8; 256];
```

## 11. VapourSynth filter lifecycle

Each filter should use the standard API 4 lifecycle:

1. Parse and validate arguments in the create function.
2. Retain the input node and immutable options in instance data.
3. Declare a strict spatial dependency on the input node.
4. On `arInitial`, request input frame `n` and return no frame.
5. On `arAllFramesReady`, retrieve input frame `n`.
6. For `PeakStats` and `PeakGrayShades`, make a writable frame/property copy,
   attach properties, and preserve the source pixel buffers where the API
   permits sharing.
7. For pixel-processing filters, allocate an output frame at the current
   frame's dimensions, copy input properties, and fill the output rows.
8. Release all acquired frame/node references according to the binding's
   ownership model.
9. Report frame errors through the VapourSynth frame context; never unwind a
   Rust panic across the C ABI.
10. Register as a fully parallel filter because no mutable global state or
    cross-frame ordering is needed.

Static levels and posterization LUTs should live in immutable filter instance
data and be shared by concurrent frame calls. Per-frame LUTs should be local to
the frame evaluation.

## 12. Implementation milestones

### Milestone 1: Freeze behavior

- Add Python golden-vector generation for the current functions.
- Decide whether the plugin follows Pillow or ImageMagick rounding.
- Decide how empty frame-property arrays are represented.
- Define invalid-input and no-peak behavior.
- Capture representative real manga pages for non-public local validation.

### Milestone 2: Pure Rust algorithms

- Implement histogram generation.
- Implement plateau-aware local maxima.
- Implement prominence and fallback selection.
- Implement gray-shade thresholding, sorting, and percentages.
- Implement levels and automatic gamma.
- Implement posterization.
- Unit-test without VapourSynth.

### Milestone 3: Minimal plugin

- Set up `cdylib` output and plugin registration.
- Implement `PeakStats`, `PeakGrayShades`, `Levels`, and `Posterize` for
  `GRAY8`.
- Preserve all incoming frame properties.
- Verify loading with `core.std.LoadPlugin` or normal plugin discovery.

### Milestone 4: Python integration

- Add a small Python wrapper in `nmanga` that composes `PeakStats` and
  `Levels(use_props=True)`.
- Integrate the wrapper with `vapoursynth-imageseqs`.
- Compare output and properties with the existing Pillow pipeline.

### Milestone 5: Distribution

- Build `.dll`, `.so`, and `.dylib` artifacts.
- Test Windows x86-64, Linux x86-64, and macOS arm64/x86-64 as required.
- Document manual installation and namespace discovery.
- Add automated release packaging and checksums.
- Consider vsrepo packaging after the API and output behavior stabilize.

### Milestone 6: Higher precision and color

- Add integer depths above 8-bit.
- Add float HDRI-like levels.
- Add RGB input and explicit plane behavior.
- Benchmark scalar LUT application before considering SIMD.

## 13. Test plan

### 13.1 Peak fixtures

- All black.
- All white.
- Uniform mid-gray.
- Peaks at bins 0, `upper_limit`, `255-upper_limit`, and 255.
- Equal-height peaks.
- Multiple peaks where the first is not the tallest.
- Odd- and even-length plateaus.
- A peak exactly at the minimum height threshold.
- Peaks immediately below and above the threshold.
- Prominence pass and fail cases.
- No qualifying peak with a successful fallback.
- No peak at all.
- White checking disabled.
- Very large synthetic counts to exercise `u64` arithmetic.

### 13.2 Gray-shade fixtures

- Default threshold of 0.01 percent.
- Threshold zero.
- A shade exactly at the calculated pixel threshold, which must be excluded.
- A shade one pixel above the threshold, which must be included.
- No significant shades.
- All 256 shades significant.
- Equal-count shades retaining ascending shade order.
- Percentage values compared against the Python reference.
- Parallel shade and percentage arrays always having equal lengths.

### 13.3 Levels fixtures

- Identity: black 0, white 255, gamma 1.
- Linear contrast stretch.
- Gamma below, equal to, and above 1.
- Pixels exactly at black and white.
- Pixels immediately outside black and white.
- Invalid equal/reversed endpoints.
- All 256 possible input values for every representative parameter set.
- Pillow-rounding and ImageMagick-rounding cases that differ by one.
- Automatic gamma golden values.

### 13.4 Posterization fixtures

- Every `bits` value from 1 through 8.
- All 256 input values.
- Midpoint/tie cases.
- Verify exact number and positions of output gray shades.
- Compare the direct mapping with Pillow's subsequent no-dither quantization.

### 13.5 VapourSynth integration

- Odd widths so stride exceeds row width.
- Very small frames, including 1x1.
- Different page dimensions if supported by the input clip.
- Source frame-property preservation, especially `ImgSeqPath`.
- Out-of-order and concurrent frame requests.
- Repeat requests for the same frame.
- Multi-threaded `vspipe` execution.
- Invalid format and invalid argument error messages.
- Plugin unload/core destruction without leaks or dangling references.

### 13.6 Acceptance criteria

- Peak results match the agreed Python reference on all synthetic fixtures.
- Gray-shade values, ordering, and percentages match the Python reference.
- `GRAY8` levels and posterization match the chosen rounding contract for all
  256 input values.
- If ImageMagick parity is not exact, every difference is documented and no
  greater than one 8-bit code value for supported inputs.
- Frame properties survive every filter.
- Results are deterministic regardless of thread count or frame request order.
- No out-of-bounds reads occur on stride-padded frames.

## 14. Performance expectations

These filters are primarily memory-bandwidth-bound:

- Histogram: one input read per pixel plus a small in-cache 256-bin table.
- Gray-shade analysis: the same histogram pass plus sorting at most 256 entries.
- Levels/posterize: one input read, one LUT lookup, and one output write.

Parallelize across frames through VapourSynth. Avoid internal per-frame
threading initially; manga image sequences naturally supply many independent
frames. SIMD is unlikely to materially improve 8-bit LUT application before
memory bandwidth becomes the limit and should only be added after benchmarks.

## 15. Suggested usage

```python
import vapoursynth as vs

core = vs.core

source = core.imgseqs.Read(files=files, mismatch=True)
gray = core.resize.Bicubic(
    source,
    format=vs.GRAY8,
    matrix_s="470bg",
    range_s="full",
)

# Inspect statistics independently.
stats = core.nimages.PeakStats(
    gray,
    upper_limit=60,
    peak_percentage=0.25,
    peak_prominence=0.1,
)

# Analyze significant grayscale values independently.
shades = core.nimages.PeakGrayShades(gray, threshold=0.01)

# Apply the per-frame PeakStats properties.
leveled = core.nimages.Levels(
    stats,
    use_props=True,
    peak_offset=0,
    auto_gamma=True,
)

output = core.nimages.Posterize(leveled, bits=4)
output.set_output()
```

## 16. Recommended decisions

Unless new requirements appear, use these defaults:

- Implement in Rust using `vapoursynth-rs`.
- Target the baseline VapourSynth API 4 plugin ABI.
- Ship `GRAY8` first.
- Use frame properties for all per-frame statistics.
- Configure `xyz.n4o.nimages` as the unique plugin identifier and `nimages` as
  the callable namespace.
- Export the four filters `PeakStats`, `PeakGrayShades`, `Levels`, and
  `Posterize`.
- Define first-version LUT output using Pillow-compatible ties-to-even rounding.
- Leave alpha untouched in all future multi-plane implementations unless
  explicitly requested.
- Correct CLI `peak_offset` unit conversion independently of the plugin.
- Treat HDRI float support as a later, separate capability rather than calling
  an 8-bit LUT HDRI-equivalent.

## 17. References

- [Current Python implementation](../nmanga/autolevel.py)
- [Current VapourSynth helpers](../nmanga/vapour.py)
- [VapourSynth API 4 reference](https://www.vapoursynth.com/doc/api/vapoursynth4.h.html)
- [VapourSynth Rust bindings](https://github.com/rust-av/vapoursynth-rs)
- [VapourSynth `std.Levels`](https://www.vapoursynth.com/doc/functions/video/levels.html)
- [VapourSynth `std.Lut`](https://www.vapoursynth.com/doc/functions/video/lut.html)
- [VapourSynth resize/colorspace conversion](https://www.vapoursynth.com/doc/functions/video/resize.html)
- [ImageMagick `-level` documentation](https://imagemagick.org/command-line-options/#level)
- [ImageMagick MagickCore levels implementation](https://github.com/ImageMagick/ImageMagick/blob/main/MagickCore/enhance.c)
