# vapoursynth-nimages

[![uv powered](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/github/license/noaione/vs-nimages)](https://github.com/noaione/vs-nimages/blob/master/LICENSE)
![VapourSynth Version](https://img.shields.io/badge/vapoursynth-%3E%3DR79-blue)

a rust vapoursynth plugin that analyzes and manipulates images.

## features

- peak detection that matches `scipy.signal.find_peaks` without depending on
  scipy, including plateau-aware maxima and prominence
- peak statistics attached to frames as properties, so analysis stays observable
  and `Levels` can consume it
- significant gray shades as parallel property arrays, binned over the full
  integer sample range
- levels matching the ImageMagick `-level` curve for integer samples through
  16 bits, with per-sample float support for `GRAYS` and `RGBS`
- posterization to any depth from 1 through the input sample depth, without
  dithering
- edge-masked sharpening in one `Deblur` node, with a Richardson-Lucy style
  deconvolution or an unsharp mask, on Gray, RGB and YUV integer and float
  clips
- automatic gamma derived from the detected black point

## requirements

- VapourSynth R79 or newer
- Python 3.12 or newer when installing the wheel
- Rust 1.88 or newer when building from source
- `PeakStats` and `PeakGrayShades` need a Gray integer clip from 8 to 16 bits;
  `Levels` and `Posterize` take Gray, RGB or YUV integer samples from 8 to 16
  bits; `Levels` also accepts `GRAYS` and `RGBS`; `Deblur` takes Gray, RGB and
  YUV at any integer depth from 8 to 16 bits and their 32 bit float formats
- a clip whose dimensions are not known until a frame is asked for, such as
  `imgseqs.Read(..., mismatch=True)` over pages of different sizes
- normalize an image sequence as in [use](#use) below

## install

the python wheel is plugin-only. it installs a `vapoursynth/plugins/nimages/`
directory holding a `manifest.vs` and the native library, and adds no python
module.

```console
python -m pip install vapoursynth-nimages
```

the x86-64 wheels carry two builds of the same plugin. `manifest.vs` names
`vs_nimages`, and vapoursynth picks the `.<variant>` file that matches the host
CPU, falling back to the plain one:

| variant | windows | linux | needs |
| --- | --- | --- | --- |
| baseline | `vs_nimages.dll` | `libvs_nimages.so` | SSE4.2 (Nehalem, 2008) |
| avx2 | `vs_nimages.avx2.dll` | `libvs_nimages.avx2.so` | AVX2 (Haswell, 2013) |
a macOS wheel is arm64 only, so it ships one file, `libvs_nimages.dylib`.

you can also take the matching file from a release and copy it into your
vapoursynth plugin directory:
if vapoursynth does not find it, load it explicitly:

```python
core.std.LoadPlugin(r"path/to/vs_nimages.dll")
```

## use

normalize the source to `GRAY8` first. `PeakStats` and `Levels` are separate
filters on purpose: compose them when automatic per-frame levels are wanted, and
inspect or override the detected levels in between.

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

# inspect statistics independently
stats = core.nimages.PeakStats(
    gray,
    upper_limit=60,
    peak_percentage=0.25,
    peak_prominence=0.1,
)

# report significant shades independently
shades = core.nimages.PeakGrayShades(gray, threshold=0.01)

# apply the per-frame levels PeakStats found
leveled = core.nimages.Levels(
    stats,
    use_props=True,
    peak_offset=0,
    auto_gamma=True,
)

posterized = core.nimages.Posterize(leveled, bits=4)
posterized.set_output()
```

### `PeakStats`

finds the black and white levels of each frame. pixels pass through untouched.

```python
stats = core.nimages.PeakStats(clip, upper_limit=60, peak_percentage=0.25)
with stats.get_frame(n) as frame:
    black = frame.props["NImagesBlackLevel"]
    white = frame.props["NImagesWhiteLevel"]
```

| argument | default | meaning |
| --- | --- | --- |
| `upper_limit` | `60` | highest 8-bit-equivalent shade searched for the black peak, scaled to the sample range and mirrored for the white peak |
| `peak_percentage` | `0.25` | minimum share of the frame a peak covers, in percent |
| `peak_prominence` | none | minimum prominence of a peak, in percent; unset disables it |
| `skip_white` | `false` | skip the white analysis and report the sample maximum |
| `debug` | `false` | log the resolved arguments and each frame's stage timings |

| property | type | meaning |
| --- | --- | --- |
| `NImagesBlackLevel` | int | selected black level, in source sample units |
| `NImagesWhiteLevel` | int | selected white level, in source sample units |
| `NImagesBlackPeakFound` | int bool | whether a black peak was found rather than defaulting |
| `NImagesWhitePeakFound` | int bool | whether a white peak was found rather than defaulting |

both percentages are percentages, not fractions: `0.25` means 0.25 percent of all
pixels, and the required count is `ceil(total_pixels * percentage / 100)`.

`peak_percentage` cannot change the result on its own. the search falls back to
the tallest peak with both thresholds removed, and the height test is monotone,
so a height threshold that rejects the tallest peak rejects every peak. set
`peak_prominence` for a threshold that has an effect. `docs/FINDINGS.md` §3.7 has
the detail.

### `PeakGrayShades`

reports every shade whose share of the frame exceeds `threshold` percent, sorted
by descending share. pixels pass through untouched.

```python
shades = core.nimages.PeakGrayShades(clip, threshold=0.01)
with shades.get_frame(n) as frame:
    values = list(frame.props.get("NImagesGrayShades", []))
    percentages = list(frame.props.get("NImagesGrayShadePercentages", []))
```

| argument | default | meaning |
| --- | --- | --- |
| `threshold` | `0.01` | minimum share of the frame a shade needs, in percent |
| `debug` | `false` | log the resolved arguments and each frame's stage timings |

| property | type | meaning |
| --- | --- | --- |
| `NImagesGrayShades` | int array | significant shade values, sorted by descending share |
| `NImagesGrayShadePercentages` | float array | share of the frame for each shade |

both arrays are always present and always the same length, including when that
length is zero. a shade is included only when its count is strictly greater than
`ceil(total_pixels * threshold / 100)`, and equal shares keep ascending shade
order.

### `Levels`

applies the ImageMagick `-level` curve to every plane of every frame. Integer
Gray, RGB and YUV clips from 8 to 16 bits use native-range tables. `GRAYS` and
`RGBS` use per-sample float math without quantizing through an integer lookup
table. The curve applies to each sample as it stands rather than to luma.

```python
# integer clips take code values, table built once
leveled = core.nimages.Levels(clip, black=12, white=245, gamma=1.18)

# float clips take the same argument in 8-bit code values
leveled = core.nimages.Levels(clip, black=5.1, white=239.7, gamma=1.18)

# per-frame parameters read from PeakStats
leveled = core.nimages.Levels(stats, use_props=True, peak_offset=0, auto_gamma=True)
```

| argument | default | meaning |
| --- | --- | --- |
| `black` | `0` | black point: a code value on an integer clip, 8-bit code values on a float one |
| `white` | sample maximum, `255` on a float clip | white point, in the same units as `black` |
| `gamma` | `1.0` | gamma of the curve |
| `use_props` | `false` | read `NImagesBlackLevel` and `NImagesWhiteLevel` from each input frame instead |
| `peak_offset` | `0` | added to the black point, in source sample units |
| `auto_gamma` | `false` | derive gamma from the effective black point, ignoring `gamma` |
| `debug` | `false` | log the resolved curve and each frame's stage timings |

for an input `x`, black `b`, white `w` and sample maximum `Q`:

```text
x < b  -> 0
x > w  -> Q
else   -> round(Q * ((x - b) / (w - b)) ** (1 / gamma))
```

rounding is ties-to-even at every step, matching the pillow path. `peak_offset`
is a code-value offset, so `peak_offset=1` on `black=12` levels from 13 rather
than from one percentage point, which is about 2.55 code values.

`black` and `white` are one argument set for both domains. On an integer clip
they are code values, and an endpoint that is not a finite whole number or that
falls outside the sample range is refused. On a float clip they are 8-bit code
values: `white=245` becomes `245 / 255`, and the default `255` becomes `1.0`.
Float outputs are clamped to `[0, 1]`, values outside the endpoints clamp to 0
or 1, and NaN samples remain NaN. Float `Levels` does not accept
`use_props=True`, nonzero `peak_offset`, or `auto_gamma=True`, because the peak
properties are integer-only.

`auto_gamma` normalizes the black point by `Q` before applying its formula. the
expression is undefined at or above half the sample range, so the filter rejects
such a black point.

### `Posterize`

by default maps every plane of each frame to `2 ** bits` evenly spaced values,
dithering. Like `Levels` it takes 8 to 16 bit integer Gray, RGB or YUV formats.

```python
posterized = core.nimages.Posterize(clip, bits=4)
```

posterizing an RGB clip's planes independently is not the same operation as
posterizing its luma. an RGB caller that wants the grayscale behaviour converts
first, as in [use](#use) above.

| argument | default | meaning |
| --- | --- | --- |
| `bits` | required | number of bits, from 1 through the input sample depth |
| `use_props` | `false` | read `NImagesGrayShades` from each input frame instead, this is the same as auto bits detection. |
| `debug` | `false` | log the resolved depth and each frame's stage timings |
| `method` | `0` | `0` spaces the levels evenly, `1` solves them from the frame's histogram with Lloyd-Max |

`bits` equal to the input sample depth is the identity. the mapping uses the
full native sample maximum `Q`:

```text
colors = 2 ** bits
level  = round(x * (colors - 1) / Q)
out    = round(level * Q / (colors - 1))
```

the pillow path follows this with `quantize(colors, dither=NONE)`, which is
provably redundant here: the mapping already produces exactly `colors` distinct
values, and pillow's quantized output is byte-identical to its input.

`method=1` solves the levels instead of spacing them. it runs the Lloyd-Max
solver over the frame's own histogram: 40 passes move each interior level to
the mean of its bucket, both endpoints stay pinned to `0` and the sample maximum,
and a bucket holding no sample keeps its even level. the histogram comes from
plane 0, which is the gray plane of a Gray clip and the luma plane of a YUV one,
and the solved table then applies to every plane. the levels are solved per
frame, so a clip whose frames differ gets a level set each, which a page-per-frame
caller wants and a moving clip would show as flicker.

**note**: when `bits` equals the sample depth, the frame is returned unchanged.

### `Deblur`

sharpens the luma of every frame, with the edge mask, halo clamp and blend that
`nmanga/deblur.py` uses:

```python
sharp = core.nimages.Deblur(clip, method=1)
```

`method=0` runs a Richardson-Lucy style deconvolution for `iterations` passes
and `method=1` an unsharp mask against a gaussian reference. both build their
candidate from the luma alone, clamp it to the 3x3 local extremes of that luma
plus `overshoot`, and blend it back through a soft edge mask, so flat areas and
fine texture come back untouched.

| argument | default | meaning |
| --- | --- | --- |
| `method` | `0` | `0` deconvolves, `1` is the unsharp mask |
| `radius` | `0.8` | gaussian sigma of the assumed blur, in pixels, above `0` and at most `16` |
| `strength` | `0.65` for `0`, `0.85` for `1` | how much of the candidate is blended back |
| `iterations` | `6` | refinement passes, `0` through `64`, `method=0` only |
| `threshold` | `2` | edge threshold in 8-bit levels, from `0` |
| `overshoot` | `0` | extra excursion allowed past the local extremes, in 8-bit steps |
| `debug` | `false` | log the resolved parameters and each frame's stage timings |

the defaults follow the method in effect, so `Deblur(clip, method=1)` is the
unsharp mask at `0.85`, and an explicit `strength` overrides either default.
`method=1` has no refinement loop, so it ignores `iterations` entirely,
including a value outside the range the deconvolution accepts.

only luma moves. on RGB the delta is added to the three planes as one equal
offset per pixel, limited to the gamut the pixel has left, which keeps the
channel differences instead of clipping each channel on its own. on YUV it is
added to the luma plane with chroma carried through byte for byte, so a
subsampled format never runs a kernel over its chroma. `threshold` and
`overshoot` are 8-bit units whatever the sample depth, and a float clip is read
as the reference's `[0, 1]` range.

the arithmetic is not the reference's. the kernels are a separable direct
convolution in `f32` whose interior reads a bounds-free window, which is several
times faster than `scipy.ndimage.gaussian_filter` and agrees with it to a
fraction of a code value. `tests/fixtures/deblur.json` carries the frozen
tolerance, and `tools/golden.py` refuses to write the fixtures when the port and
`nmanga.deblur` disagree.

### debug

every filter takes `debug`. when it is set, the first frame the filter is asked
for logs the arguments it resolved and the input it accepted, and each frame logs
its stage timings and total, both through the VapourSynth log:

```text
[nimages][debug] PeakStats: upper_limit=60 peak_percentage=Some(0.25) peak_prominence=None skip_white=true input=Gray 8 bit 1404x2000
[nimages][debug] PeakStats frame 3: black=12 white=245 histogram=2.104 ms peaks=0.081 ms copy=2.130 ms total=4.315 ms
[nimages][debug] Levels frame 3: black=12 white=245 gamma=1.07 curve=0.014 ms map=6.902 ms total=6.916 ms
```

collect them with a log handler:

```python
import vapoursynth as vs

vs.core.add_log_handler(lambda kind, message: print(message))
```

the settings line comes from the first frame rather than from creation, because
VapourSynth drops a message logged from a filter's create function.
`docs/FINDINGS.md` §2.5 and §2.6 record that, and `tools/bench.py` reads these
same lines to split a workflow into stages.

## differences from the pillow pipeline

the filters port `nmanga/autolevel.py`. two places deliberately differ, both
recorded in `docs/FINDINGS.md` with fixtures that pin the divergence:

- **gray shades use fixed native-range bins.** the reference calls
  `np.histogram(arr, bins=256)` with no `range=`, so numpy bins over the frame's
  observed min..max and `shade` comes back as a bin index rather than a gray
  value. this plugin bins 8-bit input over `0..=255` and wider input over every
  native code value. a constant image reports shade 128 at 100% in the
  reference, and an image holding only 10 and 20 reports shades 0 and 255. this
  plugin reports the actual sample values.
- **invalid arguments are rejected.** the reference's percentage guard is
  `value <= 0 and value >= 100`, a conjunction of opposites that never fires, so
  out-of-range values reach the analysis unchanged.

## build from source

```console
cargo build --release --locked
```

the artifact lands in `target/release/`. copy it into a
`vapoursynth/plugins/nimages/` directory next to a `manifest.vs` that holds:

```text
[VapourSynth Manifest V1]
vs_nimages
```

to build the wheel, which runs cargo and stages the plugin for you:

```console
python -m build
```

## development

```powershell
cargo test --locked
cargo clippy --all-targets
cargo fmt --check
```

`cargo test` runs the algorithm unit tests and the integration tests in
`tests/`, which replay the committed golden vectors in `tests/fixtures/`. those
vectors come from the reference python implementation and are regenerated with

```powershell
uv run --extra golden tools\golden.py --nmanga-path ..\nao-manga-rls
```

to exercise the built plugin itself, sync the test extras and run the validator,
which replays the same vectors through the real filters:

```powershell
uv sync --extra dev --extra dev-tests
.venv\Scripts\python.exe tests\check-nimages.py
```

`docs/BENCH.md` records what the filters cost against the python pipeline they
replace, measured on the same pages:

```powershell
uv run --extra golden --extra dev-tests tools\bench.py
```

`AGENTS.md` documents the repository layout, the crate's pitfalls and the
commands in more detail.

## license

MPL-2.0. see `LICENSE`.
