# color inputs for the analyzers

status: landed. `Gray`, `YUV` and `RGB` integer 8 to 16 are accepted on both
engines; RGB goes through the reference's fixed-point luma, and
`Posterize(method=1)` reads the same definition. depends on nothing; touches
[19](19-vulkan-histogram-analyzers.md)'s surface and the device kernel it left.

## current state

`PeakStats` and `PeakGrayShades` accept `Gray` integer 8 to 16 only
(`Accept::GrayInteger8To16`), so a `YUV420P8` or `RGB24` page is refused at
creation, on both engines:

```text
PeakStats refused: PeakStats needs a constant Gray 8 to 16 bit integer clip,
got YUV 8 bit with subsampling 1x1
```

the counting machinery underneath is already family-agnostic. `plane_histogram`
and the device `count` read plane 0 with the frame's own width, height and
stride, and plane 0 is luma for every `YUV` format: subsampling applies to the
chroma planes, never to plane 0. `Posterize(method=1)` takes any integer family
and its Lloyd solve already reads plane 0, so `posterize/yuv420p8/lloyd-bits4`
runs the same kernel on a YUV frame today and is byte-identical between the
engines.

`RGB` is the one family with real work in it: plane 0 is red, and a red-channel
shade histogram is not what the reference computes.

## what the reference does

`nmanga/autolevel.py` forces gray before it counts, both in the peak search and
in `analyze_gray_shades`:

```python
force_gray = image.mode != "L"
if force_gray:
    image = image.convert("L")  # force grayscale
hist, bin_edges = np.histogram(img_array, bins=256, range=(0, 256))
```

so an RGB page and a YUV page both arrive as `L`, and the histogram is over the
converted luma. `force_gray` is also returned to the caller, which is how the
orchestrator knows the analysis did not run on the original pixels.

## the luma definition

Pillow's `RGB` to `L` is fixed point with rounding, not the float `0.299/0.587/
0.114` anyone writes first. measured on 4096 seeded random colours through
`Image.convert("L")`:

| formula | max error | mismatched pixels |
| --- | ---: | ---: |
| `(r * 19595 + g * 38470 + b * 7471 + 32768) >> 16` | 0 | 0 of 4096 |
| `(r * 299 + g * 587 + b * 114 + 500) / 1000` | 1 | 2 of 4096 |
| `(r * 299 + g * 587 + b * 114) / 1000` | 1 | 2084 of 4096 |
| `round(0.2126r + 0.7152g + 0.0722b)` | 31 | 3975 of 4096 |

the first row is the one to implement: byte-identical to the reference on every
colour tried, and the coefficients sum to 65536 so the shift is exact.

`0.2126/0.7152/0.0722` is BT.709, which `Deblur` uses to limit an RGB delta
relative to luma. It is not the reference's luma and must not be reused here.

the same expression on native code values is the 9 to 16 bit extension, and at 8
bit it reduces to Pillow's exactly. the 16 bit sum peaks at
`65535 * 65536 = 4 294 901 760`, and adding the rounding term reaches
4 294 934 528, still under `u32::MAX`, so neither engine needs a 64 bit path.

## what moves

| piece | gray | yuv | rgb |
| --- | --- | --- | --- |
| host count | plane 0, unchanged | plane 0, unchanged | new three-plane reader |
| device count | plane 0, unchanged | plane 0, unchanged | three source bindings, luma per sample |
| output frame | `copy_frame`, unchanged | same | same |
| properties | unchanged | unchanged | unchanged |

nothing else moves. the peak search, the shade ordering, the percentages, the
property names and the pass-through frame stay where they are, so the two
engines still cannot disagree about what a histogram means.

## decisions to take

**YUV is analyzed as it arrives.** plane 0 is read as-is rather than converting
through the clip's matrix and range. the plugin reads no `_Matrix` or `_Range`
property anywhere today, and doing it properly means chroma upsampling plus a
matrix and a range expansion on both engines, for a result a caller can already
get with one `resize`/`std.SetFrameProp` upstream. the cost of the choice is
that a limited-range page reports limited-range luma, which is the same
divergence [BENCH](../BENCH.md#what-a-webp-actually-hands-out) already records
for the webp set: the reference decodes to RGB and converts with
`Image.convert("L")`, which is full range. the argument table has to say so.

**RGB is analyzed through the luma in the table above**, not through a channel.
this is what makes an RGB page and the same page decoded as gray agree.

**`Posterize(method=1)` follows the same definition.** today it solves from plane
0, which is red for `RGB` and luma for `YUV`. once the analyzers report luma, an
RGB caller chaining `PeakGrayShades` into `Posterize(use_props=True,
method=1)` would get a level count from luma and a level set from red. aligning
it is the only public behaviour change in this plan, and `posterize/rgb*/lloyd-*`
in `tools/map_hash.py` records the new bytes. the alternative is to leave it and
document the mismatch; that is the wrong default for an image pipeline.

**Float stays refused.** the reference is 8 bit and the property contract is code
values in the source's native integer range. a float clip converts to an integer
one before the analyzers, which is a caller decision about the range, not a
filter's.

## files

| file | role |
| --- | --- |
| `src/histogram.rs` | `from_rgb_planes`, the three-plane reader with the fixed-point luma |
| `src/gpu/histogram.rs` | a fourth binding and a family word, the luma path in GLSL |
| `src/filters/mod.rs` | the analyzer `Accept` variant, `analyzer_histogram` that picks the reader from the frame's family |
| `src/filters/{peak_stats,peak_gray_shades}.rs` | call the selector instead of `plane_histogram` |
| `src/filters/posterize.rs` | the Lloyd solve uses the same selector |
| `src/posterize.rs` | no change, it consumes a histogram |
| `tests/check-nimages.py` | the RGB error case flips to an acceptance case, with YUV and RGB reports |
| `tools/histogram_hash.py` | RGB and YUV groups, and the RGB-equals-gray cross-check |
| `tools/map_hash.py` | `posterize/rgb*/lloyd-*` bytes |
| `docs/{IMPLEMENTATIONS,README}.md`, `AGENTS.md` | the input policy and the range caveat |

## exactness

- the luma expression is integer and exact, so an RGB frame's histogram is a
  pure function of its three planes and the two engines must produce the same
  table, element for element.
- the properties of an RGB page whose `r == g == b` must equal the properties of
  the same page in the matching Gray format. that check catches a formula
  mistake, a plane mix-up and a stride mistake at once, and needs no reference
  implementation.
- for a general RGB page, the expected histogram comes from a numpy
  reimplementation of the formula in the harness, so the test is independent of
  the Rust reader.
- YUV has no new arithmetic: the properties of a YUV page must equal those of
  its extracted plane 0 in the matching Gray format, which is a one-line check
  with `std.ShufflePlanes`.

## expected

the counting cost is one plane read per sample for Gray and YUV, and three for
RGB. at 12 MP, 8 bit, the host histogram is 7.7 ms for one plane, so an RGB page
is expected around 20 ms; the device kernel reads three planes per invocation,
which is a memory cost rather than an arithmetic one. the point of the plan is
the acceptance, not the speed: an RGB page has no cheaper path today at all,
because it is refused.

## risks

- **the range caveat.** accepting YUV makes it easy to analyse limited-range
  luma and compare it against a full-range reference. the docs and the harness
  have to say which numbers are being reported.
- **the RGB cost.** three plane reads per sample triples the read side of the
  histogram. it is measured in the harness, and if the device path loses there,
  the RGB case stays on the host the way float `Levels` does.
- **`Posterize` bytes move on RGB.** recorded, expected, and the only case.
- **the device kernel's bindings.** one pipeline with four bindings, and the
  Gray and YUV paths bind plane 0 for the two luma-only slots rather than
  compiling a second pipeline. the alternative, two pipelines, would mean two
  pools and two recordings for one frame.
- **variable formats.** the host path already re-checks the frame's format, so a
  clip that switches family between frames keeps working. the device path
  refuses a variable format at creation, unchanged.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | `from_rgb_planes`, the fixed-point luma reader, with unit tests | done: the formula table, both depths, padded strides |
| M2 | `Accept` widened, YUV and RGB on the host | done: 50 harness cases and the three views |
| M3 | the device luma kernel | done: identical on every RGB and YUV case, 12 MP included |
| M4 | `Posterize(method=1)` on the same definition | done: 34 map cases, the RGB bytes recorded |
| M5 | measurement and the docs above | done: 23 ms of RGB counting, kept on both engines |

## outcome so far

written 2026-10-24. both analyzers take `Gray`, `YUV` and `RGB` from 8 to 16 bits
on both engines, and `Posterize(method=1)` solves its levels from the same luma
for an RGB clip. the gate is `tools/histogram_hash.py`, now 50 cases:

| group | cases | result |
| --- | ---: | --- |
| `PeakStats`, three gray depths and six argument sets | 21 | identical levels and flags |
| `PeakGrayShades`, same depths, four thresholds | 14 | identical shade and percentage arrays |
| `RGB24` and `RGB48`, flat, four-tone, sparse, `1x1` | 10 | identical, on both engines |
| `YUV420P8` and `YUV444P16` | 5 | identical, on both engines |
| float `GRAY32` and `RGB32` | 2 | refused by name on both engines |

a case table only compares the two engines against each other, so the
definition is checked separately, in the same run:

| check | why |
| --- | --- |
| a YUV clip equals `std.ShufflePlanes(clip, 0, vs.GRAY)` | plane 0 is the luma the count reads |
| an RGB page with equal planes equals the same page as gray | the conversion is luma, not a channel |
| flat `RGB` primaries report shades `76`, `150` and `29` | the reference's `convert("L")` values |

`tools/map_hash.py` is 34 cases: the RGB Lloyd bytes moved to the luma-defined
table, the harness records them, and they are still byte-identical between the
engines. `tests/check-nimages.py` carries the same two views plus the three
primaries, so the validator catches a formula change without a device.

### the cost

12 MP, flat page, median of 20 frames, from `tools/histogram_hash.py --bench`:

| format | counting, host | counting, device chain | `Posterize(use_props, method=1)` host | device |
| --- | ---: | ---: | ---: | ---: |
| `GRAY8` | 7.7 to 8.4 ms | 0.2 to 0.5 ms | 20.0 to 21.1 ms | 6.2 to 6.4 ms |
| `YUV420P8` | 8.7 to 9.1 ms | under 1 ms | 26.4 to 27.1 ms | 12.1 to 12.4 ms |
| `RGB24` | 23.1 to 24.5 ms | under 1 ms | 70.8 to 71.0 ms | 25.7 to 26.3 ms |

[21](21-vector-rgb-luma.md) later vectorised the host luma and gave the RGB
reader the striped counters, so the `RGB24` host columns read 10.4 to 14.1 ms
and 40.2 to 43.1 ms against a device 21.6 to 27.8 ms. the table above is the
reading this plan landed with, and the device margin it quotes has narrowed.

RGB counting was three times the gray cost on the host at this point, which is
plane reads and the arithmetic, and it is still 20x cheaper on the device. the
chain ratio is 2.7x for RGB and 2.2x for YUV: a 4:2:0 clip needs 1.5 samples per
pixel on the bus, so the roundtrip is 10.8 ms against 6.5 ms for `GRAY8`, and
the map is still a single-plane lookup.

some device marginals came out negative in one run, down to -3.2 ms. that is the
run-to-run spread of this machine, which the rest of the docs record as 25% to
50%, rather than a counting kernel that runs backwards; the numbers above are the
stable readings.

**decision taken: both engines, all three families.** the three choices at the
top of this plan stand as written, and the only public byte change is
`Posterize(method=1)` on RGB, which now solves from luma.

## done when
an integer `Gray`, `YUV` or `RGB` clip from 8 to 16 bits is analysed by both
analyzers, on either engine, with properties that match the reference's
`convert("L")` view, and the `Posterize(method=1)` definition is the same one.
the YUV range caveat is in the argument table rather than only in this plan.
