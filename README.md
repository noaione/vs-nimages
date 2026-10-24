# vapoursynth-nimages

a rust-based vapoursynth plugin for analyzing and manipulating images.

[![uv powered](https://img.shields.io/endpoint?url=https://raw.githubusercontent.com/astral-sh/uv/main/assets/badge/v0.json)](https://github.com/astral-sh/uv)
[![License](https://img.shields.io/github/license/noaione/vs-nimages)](https://github.com/noaione/vs-nimages/blob/master/LICENSE)
![VapourSynth Version](https://img.shields.io/badge/vapoursynth-%3E%3DR80-blue) [![build](https://github.com/noaione/vs-nimages/actions/workflows/build.yml/badge.svg)](https://github.com/noaione/vs-nimages/actions/workflows/build.yml)

## install

you need vapoursynth R80 or newer and python 3.12 or newer. install the wheel
with the python environment used by your vapoursynth host:

```bash
python -m pip install vapoursynth-nimages --index-url https://pypi.serik.at/simple
```

the wheel installs the plugin under vapoursynth's plugin directory. restart your
host, then use the filters through `core.nimages`.

## quick start

replace the sample clip with a node from your source filter. `PeakStats` finds
per-frame black and white levels without changing the pixels. `Levels` can read
those values from the frame properties.

```py
import vapoursynth as vs

core = vs.core
clip = core.std.BlankClip(width=1920, height=1080, format=vs.GRAY8)

stats = core.nimages.PeakStats(clip)
with stats.get_frame(0) as frame:
    print(frame.props["NImagesBlackLevel"])
    print(frame.props["NImagesWhiteLevel"])

leveled = core.nimages.Levels(stats, use_props=True)
posterized = core.nimages.Posterize(leveled, bits=4)
posterized.set_output()
```

## filters

| filter | accepted formats | what it does |
| --- | --- | --- |
| `PeakStats` | integer `Gray`, `YUV`, or `RGB`, 8–16 bit | finds black and white levels; leaves pixels unchanged |
| `PeakGrayShades` | integer `Gray`, `YUV`, or `RGB`, 8–16 bit | reports significant shades; leaves pixels unchanged |
| `Levels` | integer `Gray`, `YUV`, or `RGB`, 8–16 bit; `GRAYS` and `RGBS` | adjusts every plane |
| `Posterize` | integer `Gray`, `YUV`, or `RGB`, 8–16 bit | reduces every plane to a chosen number of levels |
| `Deblur` | `Gray`, `YUV`, or `RGB`, integer 8–16 bit or 32-bit float | sharpens luma |

frames are processed independently. clips may have different dimensions from
frame to frame.

`PeakStats` and `PeakGrayShades` analyze plane 0 for `YUV` and the fixed-point luma
of the three planes for `RGB`. `Levels` and `Posterize` map each plane independently.
`Deblur` changes `Gray`'s only plane, `YUV`'s luma plane, or `RGB` with an equal
per-pixel channel offset; it leaves `YUV` chroma unchanged.

### `PeakStats`

```py
stats = core.nimages.PeakStats(
    clip,
    upper_limit=60,
    peak_percentage=0.25,
    peak_prominence=0.1,
    skip_white=False,
)
```

| argument | default | meaning |
| --- | --- | --- |
| `upper_limit` | `60` | search limit in 8-bit-equivalent values, from 1 to 255 |
| `peak_percentage` | `0.25` | minimum peak height as a percent of the frame |
| `peak_prominence` | `unset` | minimum peak prominence as a percent of the frame |
| `skip_white` | `false` | skip white detection and use the sample maximum |

`0.25` means `0.25` percent. `peak_percentage` alone does not change the selected
peak; use `peak_prominence` to exclude peaks. results are attached to each frame
as `NImagesBlackLevel`, `NImagesWhiteLevel`, `NImagesBlackPeakFound`, and
`NImagesWhitePeakFound`. peak levels use the input's native sample range.

### `PeakGrayShades`

```py
shades = core.nimages.PeakGrayShades(clip, threshold=0.01)
with shades.get_frame(0) as frame:
    values = list(frame.props["NImagesGrayShades"])
    percentages = list(frame.props["NImagesGrayShadePercentages"])
```

`threshold` is a percent of the frame. the two properties contain matching arrays:
shade values and their percentages. both arrays are present when no shades pass
the threshold.

### `Levels`

```py
leveled = core.nimages.Levels(clip, black=12, white=245, gamma=1.18)
```

| argument | default | meaning |
| --- | --- | --- |
| `black` | `0` | input black point |
| `white` | sample maximum | input white point |
| `gamma` | `1.0` | curve gamma |
| `use_props` | `false` | read black and white from `PeakStats` frame properties |
| `peak_offset` | `0` | add this value to the detected black point |
| `auto_gamma` | `false` | derive gamma from the detected black point |

integer endpoints use native code values. `GRAYS` and `RGBS` endpoints use 8-bit
code values, converted to the 0–1 float range. float clips use fixed endpoints
and do not support `use_props`, `peak_offset`, or `auto_gamma`.

to use per-frame levels, connect `PeakStats` to `Levels` with `use_props=True`:

```py
stats = core.nimages.PeakStats(clip)
leveled = core.nimages.Levels(stats, use_props=True)
```

### `Posterize`

```py
posterized = core.nimages.Posterize(clip, bits=4)
```

| argument | default | meaning |
| --- | --- | --- |
| `bits` | required | output depth, from 1 to the input sample depth |
| `method` | `0` | `0` spaces levels evenly; `1` fits levels to each frame's histogram |
| `use_props` | `false` | infer the output depth from `NImagesGrayShades` on each frame |

for property-based depth, connect `PeakGrayShades` and omit `bits`:

```py
shades = core.nimages.PeakGrayShades(clip)
posterized = core.nimages.Posterize(shades, use_props=True)
```

`Posterize` does not dither. with `method=1`, levels are fitted separately for each
frame.

### `Deblur`

```py
sharp = core.nimages.Deblur(clip, method=1)
```

| argument | default | meaning |
| --- | --- | --- |
| `method` | `0` | `0` uses deconvolution; `1` uses an edge-masked unsharp mask |
| `radius` | `0.8` | Gaussian blur sigma in pixels, above 0 and at most 16 |
| `strength` | `0.65 or 0.85` | blend strength; the default depends on method |
| `iterations` | `6` | deconvolution passes, from 0 to 64 |
| `threshold` | `2` | edge threshold in 8-bit levels |
| `overshoot` | `0` | allowed excursion beyond nearby pixel values, in 8-bit steps |

`method=1` uses `strength` 0.85 by default and does not use `iterations`. `Deblur`
preserves `YUV` chroma and keeps `RGB` channel differences when sharpening.

## optional settings

all filters accept `gpu=1` to use a Vulkan device. This requires vapoursynth R80
and a Vulkan 1.4 device. gpu processing requires the clip format to stay
constant. `Levels` on the gpu supports integer formats; float `Levels` uses the cpu.

all filters accept `debug=1` to write settings and per-frame timings to the
vapoursynth log.

## license

MPL-2.0. see `LICENSE`.
