# vapoursynth-nimages

[![uv powered](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/github/license/noaione/vs-nimages)](https://github.com/noaione/vs-nimages/blob/master/LICENSE)
![VapourSynth Version](https://img.shields.io/badge/vapoursynth-%3E%3DR79-blue)

a rust vapoursynth plugin that analyzes and manipulates images. it finds the
black and white points of a page, reports which gray shades matter, and applies
levels or posterization from either fixed parameters or per-frame statistics.

## status

all four filters are implemented and verified against the reference python
implementation. the analysis and mapping algorithms are replayed against
committed golden vectors, and `tests/check-nimages.py` replays the same vectors
through the built plugin. see `docs/FINDINGS.md` for what was verified and
`docs/IMPLEMENTATIONS.md` for the plan the interface comes from.

input is 8 bit integer only. `PeakStats` and `PeakGrayShades` take a Gray clip,
because a histogram of one plane only means something for one. `Levels` and
`Posterize` take any 8 bit integer family and rewrite every plane. every filter
handles a clip whose dimensions are not known until a frame is asked for. 9 to 16
bit integer input and float input are later milestones.

## features

- peak detection that matches `scipy.signal.find_peaks` without depending on
  scipy, including plateau-aware maxima and prominence
- peak statistics attached to frames as properties, so analysis stays observable
  and `Levels` can consume it
- significant gray shades as parallel property arrays, binned over the real
  `0..=255` range
- levels matching the ImageMagick `-level` curve, built as a 256-entry table
- posterization to any depth from 1 to 8 bits, without dithering
- automatic gamma derived from the detected black point

## requirements

- VapourSynth R79 or newer
- Python 3.12 or newer when installing the wheel
- Rust 1.88 or newer when building from source
- 8 bit integer input. `PeakStats` and `PeakGrayShades` need a Gray clip;
  `Levels` and `Posterize` take Gray, RGB or YUV
- a clip whose dimensions are not known until a frame is asked for, such as
  `imgseqs.Read(..., mismatch=True)` over pages of different sizes
- normalize an image sequence as in [use](#use) below

## install

the python wheel is plugin-only. it installs the native library at
`vapoursynth/plugins/nimages/` with `manifest.vs` and adds no python module.

```console
python -m pip install vapoursynth-nimages
```

you can also take the matching file from a release and copy it into your
vapoursynth plugin directory:

| platform | file |
| --- | --- |
| windows | `vs_nimages.dll` |
| linux | `libvs_nimages.so` |
| macos | `libvs_nimages.dylib` |

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
| `upper_limit` | `60` | highest shade searched for the black peak, mirrored for the white peak |
| `peak_percentage` | `0.25` | minimum share of the frame a peak covers, in percent |
| `peak_prominence` | none | minimum prominence of a peak, in percent; unset disables it |
| `skip_white` | `false` | skip the white analysis and report 255 |
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

applies the ImageMagick `-level` curve as a 256-entry table, to every plane of
every frame. Gray, RGB and YUV clips are all accepted, subsampled or not, and the
curve applies to each sample as it stands rather than to luma.

```python
# constant parameters, table built once
leveled = core.nimages.Levels(clip, black=12, white=245, gamma=1.18)

# per-frame parameters read from PeakStats
leveled = core.nimages.Levels(stats, use_props=True, peak_offset=0, auto_gamma=True)
```

| argument | default | meaning |
| --- | --- | --- |
| `black` | `0` | black point, in source sample units |
| `white` | `255` | white point, in source sample units |
| `gamma` | `1.0` | gamma of the curve |
| `use_props` | `false` | read `NImagesBlackLevel` and `NImagesWhiteLevel` from each input frame instead |
| `peak_offset` | `0` | added to the black point, in source sample units |
| `auto_gamma` | `false` | derive gamma from the effective black point, ignoring `gamma` |
| `debug` | `false` | log the resolved curve and each frame's stage timings |

for an input `x`, black `b`, white `w` and output maximum `255`:

```text
x < b  -> 0
x > w  -> 255
else   -> round(255 * ((x - b) / (w - b)) ** (1 / gamma))
```

rounding is ties-to-even at every step, matching the pillow path. `peak_offset`
is a code-value offset, so `peak_offset=1` on `black=12` levels from 13 rather
than from one percentage point, which is about 2.55 code values.

`auto_gamma` uses `round(1 / (ln(0.5) / ln((0.5 - b/255) / (1 - b/255))), 2)`.
that expression is undefined for a black point of 128 or more, so the filter
rejects one instead of producing NaN.

### `Posterize`

maps every plane of each frame to `2 ** bits` evenly spaced values, without
dithering. Like `Levels` it takes any 8 bit integer family.

```python
posterized = core.nimages.Posterize(clip, bits=4)
```

posterizing an RGB clip's planes independently is not the same operation as
posterizing its luma. an RGB caller that wants the grayscale behaviour converts
first, as in [use](#use) above.

| argument | default | meaning |
| --- | --- | --- |
| `bits` | required | number of bits, from 1 to 8 |
| `debug` | `false` | log the resolved depth and each frame's stage timings |

`bits=8` is the identity. the mapping is

```text
colors = 2 ** bits
level  = round(x * (colors - 1) / 255)
out    = round(level * 255 / (colors - 1))
```

the pillow path follows this with `quantize(colors, dither=NONE)`, which is
provably redundant here: the mapping already produces exactly `colors` distinct
values, and pillow's quantized output is byte-identical to its input.

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

- **gray shades use the fixed `0..=255` binning.** the reference calls
  `np.histogram(arr, bins=256)` with no `range=`, so numpy bins over the frame's
  observed min..max and `shade` comes back as a bin index rather than a gray
  value. a constant image always reports shade 128 at 100%, and an image holding
  only 10 and 20 reports shades 0 and 255. this plugin reports 10 and 20.
- **invalid arguments are rejected.** the reference's percentage guard is
  `value <= 0 and value >= 100`, a conjunction of opposites that never fires, so
  out-of-range values reach the analysis unchanged.

## build from source

```console
cargo build --release --locked
```

the artifact lands in `target/release/`. copy it, or the whole staged
`vapoursynth/plugins/nimages/` directory, into vapoursynth's plugin directory.

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
