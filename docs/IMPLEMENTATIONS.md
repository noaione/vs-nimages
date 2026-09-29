# vapoursynth-nimages Implementation Plan

Status: M6 implementation in progress
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

The first version uses 256-entry lookup tables and a reusable 256-bin histogram
for `GRAY8`. M6 extends integer mapping and analysis through 16 bits and adds
per-sample float levels for `GRAYS` and `RGBS`. Analysis results stay attached
to frames as properties, and each filter scans only its requested frame.

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

For integer input above 8 bits, the plugin scales `upper_limit` from its 8-bit
units to the native sample range, mirrors that range for white detection, and
returns peaks in native code units. `skip_white` reports the sample maximum.

Both percentage parameters represent percentages, not fractions. For example,
`0.25` means 0.25 percent of all pixels:

```text
minimum_count = ceil(total_pixels * percentage / 100)
```

### 4.2 Gray-shade analysis

The Python reference builds 256 bins. The plugin uses fixed `0..=255` bins for
8-bit input and every native code value for wider integer input. Both:

1. Converts non-`L` Pillow images to grayscale.
2. Builds a 256-bin histogram for 8-bit input and a native-range histogram for
   wider integer input in the plugin.
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

A second mapping, `method=1`, does not space the levels evenly. It solves them
from the frame's own histogram with the Lloyd-Max solver in `lloyd.py`: 40
passes, both endpoints pinned, and a bucket holding no sample keeps the level it
started on. `docs/FINDINGS.md` §6.3 records the verified vectors. The default
stays `method=0`, the mapping above, so the committed posterize golden tables and
`nmanga` parity are unchanged.

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

For Gray formats above 8 bits, peak property values use the frame's native code
units. `upper_limit` remains in 8-bit-equivalent units and scales to the sample
maximum.

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
`NImagesWhiteLevel` from the current input frame. For 8-bit frames it uses the
existing 256-entry LUT; for wider integer frames it uses a native-range LUT.

Float `Levels` accepts `GRAYS` and `RGBS` with `black_float` and `white_float`:

```python
leveled = core.nimages.Levels(
    clip,
    black_float=0.02,
    white_float=0.94,
    gamma=1.18,
)
```

It evaluates each sample without an integer LUT, maps the selected interval to
`0..=1`, clamps outside it, and preserves NaN samples. Float `Levels` does not
accept integer endpoints, `use_props=True`, nonzero `peak_offset`, or
`auto_gamma=True`.

### 5.4 `Posterize`

```python
posterized = core.nimages.Posterize(clip, bits=4)
```

`Posterize` accepts integer Gray, RGB, and YUV formats from 8 through 16 bits.
It requires `bits` in `1..=sample_depth`, disables dithering, and maps every
plane independently over its full native sample range. RGB input is not
converted to gray.

`method=0` (the default) spaces the levels evenly, and `method=1` solves them for
each frame from plane 0's histogram with Lloyd-Max. Plane 0 is the gray plane of
a Gray clip and the luma plane of a YUV one, and the solved table then applies to
every plane; an RGB caller that wants the grayscale behaviour converts first. The
levels are recomputed for every frame, so a clip whose frames differ gets a level
set each.

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

### 6.2 Beyond M6

- Float `Posterize` and float peak analysis.
- Internal RGB-to-gray conversion for peak statistics and grayscale
  posterization.

Integer Gray analysis through 16 bits uses one bin per native code value and
returns native levels. This is more precise than quantizing into 256 bins. Keep
the existing `GRAY8` binning, properties, and output unchanged.

## 7. Peak detection algorithm

### 7.1 Histogram

- Use `[u64; 256]` for `GRAY8` counts and one bin per sample value for wider
  integer formats.
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

The ROI contains at most 256 samples for Gray8 or 65,536 for Gray16. The current
implementation scans the native bins directly and allocates no ROI copy.

### 7.4 Selection and fallback

- Filter candidates by minimum height and optional minimum prominence.
- Choose the candidate with the greatest height.
- Preserve the lowest-bin candidate on equal heights to match the first-index
  behavior of `numpy.argmax`.
- If no candidate qualifies, repeat peak selection without the height or
  prominence thresholds.
- If no fallback candidate exists, return black 0 or the sample maximum and set
  the corresponding `PeakFound` property to false.

### 7.5 Gray-shade analysis

`PeakGrayShades` reuses the Gray histogram implementation, with 256 bins for
Gray8 and one bin per native sample value for wider formats:

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

A 65536-entry LUT provides native integer precision for 16-bit input. `GRAYS`
and `RGBS` use the formula per floating-point sample, because a finite integer
table cannot represent their input domain.

### 8.2 Colorspace and channel behavior

Neither the existing Pillow LUT nor ImageMagick `-level` inherently converts
the stored channel values to linear light before applying the curve. HDRI
describes storage and calculation precision, not automatic colorspace
linearization.

The native filter applies the curve independently to each plane:

- Gray: process plane 0.
- RGB: process R, G, and B independently.
- YUV integer: process Y, U, and V independently.

VapourSynth represents alpha as a separate clip/output rather than a fourth
plane in these formats, so the filter has no alpha-specific behavior.

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
    black: u16,
    white: u16,
    black_found: bool,
    white_found: bool,
}

fn histogram(
    data: &[u8],
    stride: usize,
    width: usize,
    height: usize,
) -> Histogram;

fn find_local_peaks(
    histogram: &Histogram,
    options: &PeakOptions,
) -> PeakResult;

struct GrayShade {
    shade: u16,
    count: u64,
    percentage: f64,
}

fn peak_gray_shades(
    histogram: &Histogram,
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

M6 extends the existing filters without changing their `GRAY8` results or the
current 8-bit argument meanings. RGB and YUV already work for 8-bit
`Levels` and `Posterize`; M6 extends their per-plane behavior to higher sample
depths and adds a separate floating-point path for `Levels`.

The active implementation increment extends integer mapping and analysis
through 16 bits and adds float `Levels`. The release build passes; integration
validation is pending for this increment.

#### Proposed scope

- Extend `Levels` and `Posterize` to integer samples through 16 bits. `bits`
  ranges from 1 through the input sample depth, and posterized values span the
  full native sample range.
- Extend `PeakStats` and `PeakGrayShades` to Gray integer samples through 16
  bits. Use native-value histogram bins and keep property values in native code
  units. This chooses precision over compatibility with the old 256-bin
  analysis policy for new formats; `GRAY8` properties remain unchanged.
- Extend `Levels` to `GRAYS` and `RGBS` with per-sample floating-point math.
  Do not quantize float samples through an 8-bit or 16-bit lookup table. Float
  `Posterize` and float peak analysis stay out of M6.
- Process color components independently. Do not add implicit RGB-to-gray or
  YUV-to-RGB conversion.
- Keep SIMD out of the first implementation. Benchmark the scalar paths before
  proposing it.

#### M6 decisions

the current M6 interface retains separate float endpoint names and normalized
float units. a proposed consolidation is tracked in
[02-unified-level-endpoints.md](improvements/02-unified-level-endpoints.md);
the current argument names and units remain in effect until that proposal is
accepted.

- Keep `upper_limit` in its current 8-bit-equivalent units and scale it to the
  native integer range as `round_ties_even(upper_limit * max_sample / 255)`.
  This preserves the existing default's relative search range and leaves the
  `GRAY8` behavior unchanged. At 16 bits, the default 60 becomes 15420.
- Keep integer `black`, `white`, and `peak_offset` in native code units above
  8 bits. For float `Levels`, choose typed float endpoint arguments that do not
  change the existing integer arguments. Use `black_float` and `white_float`.
- Default float endpoints are 0.0 and 1.0. Float outputs are clamped to
  `0..=1`; NaN samples remain NaN.
- Validate finite float endpoints with `black_float < white_float` and finite
  positive `gamma`. Map values below/above the endpoints to 0/1. Preserve NaN
  samples as NaN; positive and negative infinity clamp through the endpoint
  comparisons. Float `Levels` does not accept integer endpoints,
  `use_props=True`, nonzero `peak_offset`, or `auto_gamma=True` because the peak
  properties are integer-only.
- VapourSynth carries alpha as a separate clip/output, not as a fourth plane in
  an RGB video format. M6 operates on the planes in the input video format; it
  does not add alpha-specific arguments.

#### Implementation sequence

1. Update the public argument and property tables to match the decisions above.
2. Add pure sample-level integer and float operations outside the VapourSynth
   filter layer. Keep the 8-bit implementations as compatibility paths until
   exhaustive comparison proves the shared implementation is byte-identical.
3. Add stride-aware 16-bit plane reads and writes. Use the frame's sample width,
   plane dimensions, and stride; never treat 16-bit samples as individual
   bytes.
4. Extend Gray analyzers with native-value histograms and frame properties.
   Preserve per-frame independence and the existing threshold and peak rules.
5. Extend integer mapping filters to 9–16-bit Gray, RGB, and YUV formats, then
   add `GRAYS` and `RGBS` support to `Levels`.
6. Add integration coverage for real plugin frames and run the existing 8-bit
   golden and concurrency checks unchanged.
7. Benchmark scalar 8-bit LUT, 16-bit LUT, and float paths on representative
   clips. Consider SIMD only if the scalar measurements show a meaningful
   bottleneck.

#### M6 test plan

- Keep every existing 8-bit golden output unchanged.
- Test 16-bit endpoints, full-scale mapping, monotonicity, gamma, values around
  black/white, and posterization to exactly `2 ** bits` output levels.
- Test peak selection and shade properties at values above 255, including
  peaks near 0 and the native maximum, tied peaks, empty regions, and threshold
  boundaries.
- Test odd-width `GRAY16`, `RGB48`, and subsampled `YUV420P16` frames with
  stride padding and distinct plane values. Verify pixels outside each plane's
  active row are neither read nor written.
- Test `GRAYS` and `RGBS` with fractional samples, endpoint clipping, gamma,
  NaN and infinity samples, and multiple consecutive `Levels` operations.
- Test property preservation through the built plugin, including concurrent,
  repeated, and out-of-order frame requests.
- Compare scalar performance with the current 8-bit path and record the tested
  frame sizes and formats before considering SIMD.

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

## 14. Performance plan

The performance plan is maintained in
[01-performance-plan.md](improvements/01-performance-plan.md). Future
improvement documents use the numbered filename pattern
`XX-plan-name.md` in `docs/improvements/`.

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
- Preserve `GRAY8` results while supporting the M6 integer and float formats.
- Use frame properties for all per-frame statistics.
- Configure `xyz.n4o.nimages` as the unique plugin identifier and `nimages` as
  the callable namespace.
- Export the four filters `PeakStats`, `PeakGrayShades`, `Levels`, and
  `Posterize`.
- Define first-version LUT output using Pillow-compatible ties-to-even rounding.
- Process every plane present in the VapourSynth video format independently;
  alpha is a separate clip/output rather than a fourth plane.
- Correct CLI `peak_offset` unit conversion independently of the plugin.
- Use per-sample math for `GRAYS` and `RGBS`; never route float samples through
  an integer LUT.

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
