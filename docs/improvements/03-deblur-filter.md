# deblur filter

status: implemented

the optimization candidates are collected in
[04-optimization-review.md](04-optimization-review.md). the horizontal SIMD
regression recorded under [where the time goes](#where-the-time-goes) is now
resolved in [05-horizontal-blur-dispatch.md](05-horizontal-blur-dispatch.md),
and the vertical loop is in
[06-vertical-register-accumulation.md](06-vertical-register-accumulation.md);
the stage tables below remain historical measurements.

## problem

`nmanga/deblur.py` exposes two sharpening entry points:

- `deblur_deconv` — a Richardson-Lucy style multiplicative deconvolution.
- `deblur_edge_sharp` — an edge-masked unsharp mask.

Both take and return a `PIL.Image` and are pure numpy/scipy, so the only caller is
the CLI action `image_ops_deblur` (`nmanga/cli/image_operations.py:317`), which
walks a directory, runs them through a thread pool
(`_runner_imops_deblur`, `image_operations.py:209`) and always writes a PNG
(`image_operations.py:241`).

Nothing in this plugin sharpens anything, and none of the plugins installed here
does either, so a VapourSynth pipeline has to leave the graph for Pillow and run
its own pool. The pool is the part that goes away: VapourSynth already runs a
filter frame-parallel across its worker threads, so a `Deblur` node gets the same
concurrency from the core, without a second copy of the page and without the
decode/encode round trip.

## what the reference computes

Verified by reading `nmanga/deblur.py`:

| entry point | candidate | defaults |
| --- | --- | --- |
| `deblur_deconv` (`deblur.py:270`) | `estimate *= blur(observed / max(blur(estimate), 1e-7))` for `iterations`, then `y + strength * (estimate - pedestal - y)`, `pedestal = 1/255` (`deblur.py:187`) | radius `0.8`, strength `0.65`, iterations `6`, threshold `2`, overshoot `0` |
| `deblur_edge_sharp` (`deblur.py:329`) | `y + strength * (y - blur(y, radius))` | radius `0.8`, strength `0.85`, threshold `2`, overshoot `0` |

Both then share four stages:

- **luma** from the gamma-encoded RGB values, `0.2126 R + 0.7152 G + 0.0722 B` on
  `[0, 1]` floats (`deblur.py:266`). The comment there is explicit that encoded
  luma is deliberate: the input is resampled artwork, not a linear-light capture.
- **edge mask** (`deblur.py:95`): Gaussian prefilter with sigma `0.5` on the
  detector only, Sobel magnitude, scaled by `255 / 8`, then
  `smoothstep((m - threshold) / max(3 * threshold, 1e-6))`, then a `0.45` Gaussian
  blur of the mask. So the mask is a soft `[0, 1]` ramp that opens on strong edges
  and leaves flat areas and fine texture alone.
- **halo clamp** (`deblur.py:230`): the candidate is clipped to the 3x3 local
  minimum and maximum of `y`, widened by `overshoot / 255` in 8-bit steps.
- **blend** (`deblur.py:233`): `clip(y + mask * (candidate - y), 0, 1)`.

Colour is handled in one step (`deblur.py:318`): the luma delta is added to R, G
and B as an **equal offset**, clipped to the gamut the pixel has left,
`delta = clip(delta, -rgb.min(axis=2), 1 - rgb.max(axis=2))`. That preserves the
channel differences instead of clipping each channel separately, and it is a
behaviour worth keeping rather than simplifying to per-plane processing.

The reference's kernel is `scipy.ndimage.gaussian_filter` with `mode="reflect"`
and `truncate=4` (`deblur.py:128`), and its rounding is `np.rint` (ties-to-even)
on `clip(rgb + delta) * 255`.

## prior art

Checked in this environment, not assumed:

- The installed plugin set is `avscompat`, `libvship_NVIDIA`, `vs_imageseqs` and
  `nimages` (plus `vsogsov` in the sibling venv). None of them sharpens.
- Stock VapourSynth has the **primitives** but no deblur filter. Verified
  signatures: `std.Convolution` (`matrix:float[]`, `bias`, `divisor` — any kernel,
  so a Gaussian is expressible), `std.BoxBlur`, `std.Expr` (multi-clip per-pixel
  arithmetic), `std.Sobel` / `std.Prewitt`, `std.Minimum` / `std.Maximum`
  (`threshold`, `coordinates`), `std.Median`, `std.Merge`, `std.MakeDiff`,
  `std.Lut` / `std.Lut2`.
  So the graph *can* express both methods — six `Expr`/`Convolution` levels for the
  deconvolution plus the mask and clamp — but there is no Gaussian beyond
  `BoxBlur`'s box approximation, `std.Convolution` would have to be fed a kernel
  built to match `gaussian_filter(truncate=4)`, and the result is a graph the size
  of the iteration count rather than one filter.
- External pointers, from a web search and **not** checked against this machine:
  learned restoration (DPIR, SCUNet) through
  [vs-mlrt](https://github.com/AmusementClub/vs-mlrt), CAS sharpening through
  [VapourSynth-CAS](https://github.com/HolyWu/VapourSynth-CAS), the
  AVISynth "LimitedSharpen" family whose mask-clamp-blend design is the closest
  existing relative of this reference
  ([sharpeners guide](https://www.aquilinestudios.org/avsfilters/sharpeners.html)),
  and the RL algorithm itself
  ([deconvlucy](https://www.mathworks.com/help/matlab/ref/deconvlucy.html)).

The reference is not a learned model, so a port here is real work but bounded work:
one module of kernels that allocates nothing per frame.

## proposed contract

```python
sharp = core.nimages.Deblur(
    clip,
    method=0,        # 0 deconvolution, 1 edge-masked unsharp
    radius=0.8,
    strength=0.65,   # 0.85 when method=1
    iterations=6,    # method 0 only, ignored by method 1
    threshold=2,
    overshoot=0,
    debug=False,
)
```

| argument | default | meaning |
| --- | --- | --- |
| `method` | `0` | `0` runs the deconvolution, `1` the unsharp mask |
| `radius` | `0.8` | Gaussian sigma of the assumed blur / blurred reference, in pixels |
| `strength` | follows the method: `0.65` for `0`, `0.85` for `1` | how much of the candidate is blended back |
| `iterations` | `6` | refinement passes, `method=0` only. An explicit value overrides the default, and `method=1` ignores it |
| `threshold` | `2` | edge threshold in 8-bit levels |
| `overshoot` | `0` | extra excursion allowed past the local extremes, in 8-bit steps |
| `debug` | `false` | log the resolved settings and per-stage timings |

Every default follows the method that is in effect, so `Deblur(clip, method=1)` is
`deblur_edge_sharp` with its own defaults and an omitted `strength` is `0.85`, not
the deconvolution's `0.65`.

Formats: integer Gray, RGB and YUV from 8 to 16 bits, and the 32-bit float Gray,
RGB and YUV formats.

Plane behaviour:

- **RGB** — the luma delta is added to the three planes as one equal offset per
  pixel, clipped to the gamut that pixel has left, exactly as the reference does.
  This is what preserves the channel differences.
- **Gray** — the delta applies to the single plane.
- **YUV** — the delta applies to the luma plane and chroma is left untouched. That
  is the analogue of the equal-offset rule for a format that already carries luma
  separately, and it means the kernel never runs over subsampled chroma.
- Alpha needs nothing: VapourSynth carries alpha outside the video format
  (`IMPLEMENTATIONS.md` M6), so the reference's RGBA pass-through has no
  counterpart here.

Output keeps the input format and the input's properties (`prop_src`), and the
filter is `Parallel` with a strict spatial dependency, like the other four.

## behaviour and exactness

The reference defines the **stages, the parameters and their defaults**. It does
not define the arithmetic: the port is free to choose its own kernel
implementation, its own summation order, and a faster blur than scipy's
`gaussian_filter(truncate=4)` where that buys real time. Speed is an explicit goal
of this port, so a faster kernel that stays inside the tolerance below is the
preferred outcome, not a compromise.

What must not change, because it is what makes the output recognisably this
operation:

- the four stages and their order: mask from the prefiltered gradient, candidate,
  clamp, blend;
- the soft, edge-only mask (zero on flat areas, so flat artwork is untouched);
- the halo clamp against the 3x3 local extremes plus `overshoot`;
- the equal-offset colour rule on RGB, and luma-only on YUV;
- the RL pedestal and the division floor, or the loop diverges near black;
- ties-to-even rounding, which `src/round.rs` already provides and `np.rint`
  matches;
- the per-method defaults.

Proposed tolerance to freeze in the fixtures, pending agreement:

- at 8 bits, no pixel differs by more than **1 code value**, and at least 99.9% of
  pixels differ by **0**;
- at 16 bits the same bound scales with the sample range (257 code values);
- float paths are compared with an absolute tolerance of `1/255`.

A completely different sharpening scheme would be a separate filter proposal. This
one keeps the reference's shape so that a page can be compared between the old CLI
and the plugin without translating the result.

## blur kernel, measured

The blur is the hot loop. `deblur_deconv` runs two blurs per refinement pass for six
passes, and the mask adds two more, so about 14 blurs a frame.

Measured on a synthetic 2048x2048 page (4 194 304 px), best of five runs, rayon on 12
threads, against `scipy.ndimage.gaussian_filter(..., mode="reflect", truncate=4)`.
Times are milliseconds for the whole page and the accuracy column is the worst-pixel
difference from scipy in 8-bit code values at sigma 0.8.

| candidate | 0.45 | 0.5 | 0.8 | 2 | 8 | max diff at 0.8 | note |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | --- |
| separable direct, f32, rayon | 11.9 | 11.7 | 14.3 | 27.7 | 93.6 | 0.0001 | `reflect` per tap |
| **separable direct, f32, window + rayon** | **8.8** | **7.6** | **7.7** | **10.3** | **31.5** | **0.0001** | **the choice** |
| separable direct, f32, window, one thread | 22.1 | 21.7 | 25.8 | 46.4 | 222.6 | 0.0001 | faster than scipy at every sigma |
| separable direct, f32, one thread | 45.8 | 44.2 | 62.5 | 137.5 | 517.4 | 0.0001 | |
| separable direct, f64, rayon | 34.7 | 34.6 | 35.1 | 51.1 | 119.9 | 0.0000 | scipy's arithmetic |
| `image::imageops::blur` (u8, three box passes) | 28.5 | 26.0 | 26.8 | 32.0 | 69.7 | 14.9 | box, not a Gaussian |
| `image::imageops::fast_blur` (f32) | 43.5 | 43.1 | 43.3 | 44.4 | 45.3 | 74.5 | box approximation |
| `image::imageops::fast_blur` (u8) | 76.3 | 73.4 | 73.9 | 75.1 | 71.7 | 74.6 | box approximation |
| `zune-imageprocs` git (u8) | — | 3.6 | 4.0 | 3.7 | 3.9 | 14.8 | refuses sigma <= 0.456 |
| `zune-imageprocs` git (f32) | — | 13.9 | 14.4 | 14.6 | 14.9 | 14.8 | refuses sigma <= 0.456 |
| three box passes, rayon | 152.4 | 153.9 | 164.4 | 145.5 | 144.9 | 115.6 | radii round to 0 |
| two-pass van Vliet, rayon | 60.7 | 59.4 | 60.6 | 59.0 | 63.6 | 24.2 | inaccurate below sigma 2 |
| `scipy.ndimage.gaussian_filter` f64 | 109.2 | 110.6 | 111.8 | 128.3 | 244.9 | — | the reference |
| `scipy.ndimage.gaussian_filter` f32 | 105.2 | 89.1 | 95.7 | 108.4 | 227.6 | — | |

Verdict: the **separable direct convolution in f32 with rayon, using a bounds-free
window for the interior**. It is 12.7x to 16.2x faster than scipy f64 — 10.4x to 12.6x
against scipy f32 — at the sigmas this filter uses, and matches scipy to 0.0001 code
values (one or two ULPs), because it builds the same kernel and the same `reflect`
padding. The first run of this comparison used the `mirror` convention by mistake and
showed a 3.7 code value border error, which is exactly the kind of thing the accuracy
column exists to catch.

The tap loop matters as much as the algorithm. The first version called `reflect()` per
tap, and `reflect` does an integer modulo: one division per tap per pixel, 272 million
of them per page at sigma 8, which on its own accounted for the whole 500 ms and is why
that variant fell behind scipy there. Splitting the interior from the borders so the tap
loop reads a bounds-free window (free to unroll and vectorise) is 2.1x faster at sigma
0.45, 2.4x at 0.8, 3.1x at 2 and 2.3x at 8, with the accuracy unchanged to the last
digit. The implementation uses the windowed form; the `reflect`-per-tap row stays in
the table as the reason why.

Why the alternatives lose:

- **`zune-imageprocs` is the fastest thing measured** — its u8 path runs at 0.87 ns/px,
  about 3x the direct f32 path — but it cannot be used here. The git revision errors
  out at sigma 0.45 with `Gaussian Blur radius is too small at 1`: its radius is
  `round(sqrt(6 * sigma^2 + 1))` and it refuses a radius of 1 or less, i.e. everything
  at or below sigma 0.456, which is the mask blur the reference uses. Where it does
  run it is a two-box approximation whose worst pixel is 14.8 code values from the
  Gaussian at sigma 0.8, with 10% of pixels off by more than half a code value. The
  mask reads the gradient of a blurred copy and the deconvolution divides by one, so
  that error is not affordable. Worth revisiting only if a future filter needs a large
  sigma and can live with a box approximation.
- **`image::imageops::blur`** is a three-pass box blur: 2x slower than the direct path
  at sigma 0.45 and 14.9 code values off at sigma 0.8. `fast_blur` is both slower and
  far worse (74.5 code values) at these radii.
- **Three box passes** degenerate at small sigma: Kutskir's radius formula rounds to 0
  for sigma <= 1.2, so the filter is three identity passes and the page comes back
  unchanged (115.6 code values from the reference).
- **The recursive van Vliet IIR** only pays from roughly sigma 2 upwards, and the
  two-pass form measured here is still 24 code values off at sigma 0.8.
- The crossover is smaller than it first looked: with the windowed tap loop the direct
  path is 31.5 ms at sigma 8, so only zune's u8 box blur (3.9 ms) is still ahead of it
  there. `Deblur` never goes above sigma 0.8 anyway.

At the deblur's sigmas the windowed path costs about 1.8 ns/px, so the ~14 blurs a
`method=0` frame needs are roughly 110 ms at 2048x2048, against about 1.7 s for the same
sequence in scipy. The harness that produced this table is `.tmpbuild/gaussbench/`
(a throwaway crate plus a scipy comparison), so, as with the performance work, the
numbers are reproducible only while that directory exists.

## what was built

1. `src/deblur.rs` — dependency-free kernels, no VapourSynth types: the reflected
   direct gaussian in f32 with the windowed interior, Sobel magnitude, smoothstep,
   3x3 minimum/maximum, the two candidates, and the mask/overshoot blend. A
   `Workspace` owns the five planes, so a frame allocates nothing.
2. `src/filters/deblur.rs` — `Accept::Deblur` covering the formats above, defaults
   resolved per method, luma taken from plane 0 on Gray/YUV and computed from the
   planes on RGB, the kernels run once over that luma, and the delta written back
   per the plane rules. The debug line carries `luma`, `restore` and `write`; the
   kernels are one call, so the mask and the candidate cannot be timed apart
   without splitting it.
3. `src/filters/mod.rs` — the `Accept::Deblur` variant. The filter copies the
   input frame rather than allocating, so the planes it does not touch and the
   input's properties both survive byte for byte.
4. `src/lib.rs` — the fifth registration, plus the filter list in `README.md`,
   `AGENTS.md` and `IMPLEMENTATIONS.md`.
5. Fixtures — `tools/golden.py` builds a 64x64 page, generates the expectations
   from a float64 port of `nmanga.deblur` and refuses to write them when that port
   and `nmanga.deblur` disagree. `tests/test_golden.rs` replays them through the
   algorithms and `tests/check-nimages.py` through the built plugin.

## validation

- Both methods against the reference on the synthetic page, within the frozen
  tolerance, at 8 and 16 bits. All 14 cases pass, and every 8 bit case is
  bit-exact against the reference; the 16 bit deconvolution case has 10 of 4096
  samples one 16 bit step apart, which is what the `f32` kernels cost.
- `Deblur(clip, method=1)` uses `strength=0.85` with no explicit `strength`, and an
  explicit `strength` overrides either method's default.
- `iterations` is ignored by `method=1` and honoured by `method=0`.
- Float Gray, RGB and YUV inputs work, and the float tolerance holds. The worst
  float sample differs from the reference by 2.5e-7, which is 15000x inside the
  frozen `1 / 255`.
- YUV chroma is byte-identical to the input, including 4:2:0 and 4:2:2
  subsampling.
- `threshold=0` opens the mask fully; `overshoot=0` allows no excursion, checked
  against the 3x3 extremes of the page.
- A frame with all-black and all-white areas produces no NaN and no overflow, and a
  flat page comes back unchanged because the mask is zero there.
- `cargo test --locked`, `cargo clippy --all-targets --locked -- -D warnings`,
  `cargo fmt --check`, and `tests/check-nimages.py`.

## resolved decisions

| question | decision |
| --- | --- |
| bit-exactness against scipy | not required. A different, faster implementation is fine, and faster is preferred; the tolerance above is the correctness bar |
| naming | one `Deblur` with a `method` parameter |
| defaults | follow the method that is in effect, per method |
| formats | support YUV and float, not just integer Gray and RGB |
| the sibling checkout | not touched, including its CLI action |
| mask output | not exposed |
| `iterations` cost | if it measures badly, a different or faster scheme is acceptable rather than changing the reference's shape for its own sake |
| the blur | the separable direct gaussian in f32 with a bounds-free interior, one thread per frame; rayon inside a frame is deferred until the frame-level parallelism is shown to be too little |
| the tolerance | the numbers above are frozen in `tests/fixtures/deblur.json`. 8 bits are bit-exact against the reference, so the `1` code value bound and the `0.999` exact fraction hold with room, and the 16 bit exact fraction is `0.99` because f32 lands a handful of samples on the other side of a rounding boundary |

## measured

`tools/bench.py` over the whole `posterize-check` set, both deblur workflows, as it
is recorded in [`docs/BENCH.md`](../BENCH.md): 49 pages of 2902x4128, 2903x4128
and one 5806x4128 spread.

| method | pipeline | apply | total | per page | peak rss |
| --- | --- | ---: | ---: | ---: | ---: |
| `0` deconvolution | nmanga `deblur_deconv` | 328.74 s | 330.73 s | 6749.7 ms | 1569 MiB |
| `0` deconvolution | `nimages.Deblur` | 74.87 s | 77.76 s | 1586.9 ms | 969 MiB |
| `1` unsharp | nmanga `deblur_edge_sharp` | 170.56 s | 172.52 s | 3520.9 ms | 1569 MiB |
| `1` unsharp | `nimages.Deblur` | 33.12 s | 36.04 s | 735.4 ms | 969 MiB |

The table is the first build measured. The three passes under [where the time
goes](#where-the-time-goes) then took the plugin from 1586.9 to 1302.3 ms a page
on the deconvolution and 735.4 to 548.7 on the unsharp mask, so the current build
is 4.88x and 6.13x faster than the reference; the reference's own run to run
spread is wider than that difference, so read the plugin column, not the ratio.
[`docs/BENCH.md`](../BENCH.md) is the machine generated record of the current
build.

It is also the one workflow where the plugin holds less memory than the reference,
0.62x, because scipy keeps several megapixel-sized float64 temporaries alive per
stage of a 24 Mpx page while the plugin's cost is its own scratch. The unsharp mask
is 2.3x cheaper than the deconvolution, so it is the better choice for a caller who
just wants a page sharpened.

### where the time goes

The filter reports one `restore=` number, so the split comes from timing the same
12.0 Mpx page in four configurations that differ only in how many blurs they run,
one node per configuration so the scratch is allocated once and the first frame
is dropped. Medians of five warmed frames, over the three builds this work
produced, run to run spread is about 5%:

| configuration | blurs | first | row slices | plus `sqrt` |
| --- | ---: | ---: | ---: | ---: |
| `method=0 iterations=0` | 0 candidate | 522.2 ms | 454.9 ms | 379.1 ms |
| `method=1` | 1 | 565.3 ms | 502.1 ms | 427.2 ms |
| `method=0 iterations=1` | 2 | 671.0 ms | 604.0 ms | 519.2 ms |
| `method=0 iterations=6` | 12 | 1414.7 ms | 1338.7 ms | 1197.2 ms |
| mask and blend, derived as `2B - C` | | 459.6 ms | 400.2 ms | 335.2 ms |
| read the luma, `luma` stage | | 10.6 ms | 10.9 ms | 8.9 ms |
| write the plane back, `write` stage | | 50.3 ms | 52.4 ms | 48.7 ms |

The stage table is the second pass; the third pass below then took the `write` row
from 48.7 to 30.8 ms without moving anything else, so read the last column as the
second pass plus that. End to end over the 49 page bench, the plugin went from
1586.9 to 1302.3 ms a page for `method=0` and 735.4 to 548.7 for `method=1`,
against a reference that only moved by its own noise: **18% off the deconvolution
and 25% off the unsharp mask**.

**The fourth pass: AVX2 for the blend.** The stencil is elementwise, so eight
columns at a time computes what one column computed, which is what the probe in
`.tmpbuild/simdprobe/` established. `Workspace::blend` now dispatches to a
`#[target_feature(enable = "avx2")]` path when `is_x86_feature_detected!("avx2")`
says the feature is there, with the scalar loop kept as the fallback on every
other target and the two end columns and the row tail still going through the
shared scalar `blend_sample`. Measured at 12.0 Mpx:

| configuration | scalar | AVX2 | |
| --- | ---: | ---: | ---: |
| `method=0 iterations=0` | 378.3 ms | 171.7 ms | -55% |
| `method=1` | 420.8 ms | 223.5 ms | **-47%** |
| `method=0 iterations=1` | 494.5 ms | 309.2 ms | -37% |
| `method=0 iterations=6` | 1157.1 ms | 954.3 ms | -18% |
| mask and blend, derived as `2B - C` | 347.1 ms | 137.7 ms | **-60%** |

The output is byte-identical. The hash harness now carries two float pages full of
`NaN` and reports no difference at all across all twelve cases, and the validator
still passes its 2150 checks. The `NaN` pages are there for a reason: the
branchless vector clamp would snap a `NaN` to a window bound where the scalar form
lets it through, so `clamp8` blends the `NaN` back over the result with an
unordered compare. Getting that wrong would have made float clips behave
differently on machines with AVX2 than without.

**The fifth pass: the vertical blur accumulate.** `target[i] += tap[i] * weight`
down a row is elementwise on the same terms, so the inner loop dispatches to an
`accumulate_avx2` helper eight lanes wide, with the scalar loop left in place
directly below it as the fallback. The product and the sum stay separate
instructions, so no `fmadd` contracts them.

It is worth much less than the blend, and the reason is instructive: the scalar
loop was a plain contiguous `zip` and LLVM had already vectorised it four lanes
wide with SSE2, so the hand written eight lane version only bought the width
difference.

| | before | after | |
| --- | ---: | ---: | ---: |
| twelve blur passes | 808.9 ms | 744.6 ms | -7.9% |
| `method=0 iterations=6` | 977.9 ms | 916.0 ms | -6.3% |
| `method=1` | 220.2 ms | 215.7 ms | -2% |

All twelve hash cases are still identical, validator still 2150 checks.

**The sixth pass: the Sobel and smoothstep.** Same pattern, and this time the
guess was right. `sobel_mask` dispatches to a `sobel_mask_avx2` eight lanes wide,
with a shared `sobel_sample` for the two end columns and the row tail so the
vector path cannot drift from the scalar one at the edges. The three terms of each
response are added in the same order, the division by the ramp stays a division,
and `1 / SOBEL_SPAN` is a power of two, so the multiply the vector path uses is
the same scaling the scalar division is.

| configuration | before | after | |
| --- | ---: | ---: | ---: |
| mask and blend, derived as `2B - C` | 153.2 ms | 101.7 ms | **-34%** |
| `method=1` | 222.9 ms | 180.7 ms | **-19%** |
| `method=0 iterations=6` | 960.2 ms | 868.6 ms | -9.5% |

That is where the fifth pass's lesson pays off. The vertical accumulate bought
7.9% because LLVM had already vectorised it; the Sobel bought 34% because its
scalar form clamps an index per pixel and clamps the ramp with branches, which is
what stops a loop from vectorising. Hand vectorising is worth it where the scalar
form fights the compiler, not where it is already a clean contiguous loop.

Twelve hash cases still identical, validator still 2150 checks.

**The horizontal blur sum, first attempt: a regression that was the dispatch
site.** Vectorising across pixels
rather than taps should have been the safe way to do this one, because each lane
keeps its own taps in weight order and the accumulator starts at zero exactly as
the scalar `sum()` fold does. It was byte-identical as expected, and it measured
**2.9x slower**: twelve blurs went from 759.7 to 2217.6 ms, and `method=0`'s
restore from 892.2 to 2519.5 ms. It was reverted, and the reverted build reports
768.9 ms and the same twelve hashes.

That regression was the dispatch site, not the loop. It is now implemented: the
feature check moved from `blur_row`, which runs once per row, to `blur`, which
runs once per gaussian, and the horizontal pass measures 2.2x to 3.0x faster with
byte-identical output. See
[05-horizontal-blur-dispatch.md](05-horizontal-blur-dispatch.md).

Two explanations were tried and neither accounts for a factor of three: the
`set1` broadcast inside the tap loop, and the single accumulator chain. A
9x-longer loop body with the same memory traffic should not lose a factor of
three, so this needs a profiler rather than another guess. It is the only loop in
the filter that is still scalar, and it is worth about half of the blur, which is
most of `method=0`.

**Then the same loop measured 3x faster in isolation.** `.tmpbuild/simdprobe/`
has a `hsum` binary that runs the row interior three ways over the same plane and
checks the outputs first:

| shape | time |
| --- | ---: |
| scalar left fold, what the crate does | 2.10 ns/px |
| eight columns at a time, taps inner, register accumulator | **0.70 ns/px** |
| eight columns at a time, taps outer, memory accumulator | 0.93 ns/px |

The shape that cost 2.9x inside the plugin is 3.0x *faster* on its own, so the
regression is not a property of the loop. The one structural difference between
the probe and the plugin run is where the feature check sits: `blend` and
`sobel_mask` test `is_x86_feature_detected!` once per plane, while the attempt
put it inside `blur_row`, which runs once per row, 49.5k times for the twelve blur
passes of one frame. That is the thing to change before trying again, and it is a
guess with a measurement behind it rather than a comfortable one.

The probe's own two vector shapes differ from its scalar one on 1536 of 1.5M
samples, which is a probe artifact from the slack window it slices (the row ends
reflect into the neighbouring row); the crate's version was byte-identical on all
twelve cases, so that discrepancy belongs to the harness and not to the idea.

**The first pass: row slices.** `blend` and `sobel_mask` called a helper per pixel
that recomputed `row * width + column` and bounds-checked into a plane-sized slice
for each of the nine taps. They now slice the three rows they need out of the plane
once per output row, so a tap is indexing into a slice whose length is the plane
width. The arithmetic is unchanged and the output is byte-identical on all ten
cases: -13% on the mask and blend stage, -11% on `method=1`.

**The second pass: `sqrt` instead of `hypot`.** `f32::hypot` is a libcall, and it
was about a sixth of the mask and blend stage on its own. `sqrt(a * a + b * b)` is
safe here because the Sobel responses are bounded by four times the sample range
and the result is clamped to `[0, 1]` before it is used, so nothing can overflow or
underflow into a different mask. It is **not** bit-exact: the magnitude can differ
in the last bit. The integer paths absorb it, and the fixture cases still report
`max 0` and `zero 1.0` at 8 and 16 bits; the float cases deviate from the reference
by 2.50e-7, unchanged from before, against a `1 / 255` bound. The hash harness
shows the difference in exactly one of its ten cases, `gray32 method0`.
-16% on the mask and blend stage, -12% on `method=1`.

**The third pass: rounding without `roundsd`.** The `write` stage was 45.9 ms a
frame, which is 10% of `method=1`'s wall time for what should be a byte store.
`f64::round_ties_even` needs SSE4.1's `roundsd`, which a baseline x86-64 build does
not enable, so it was a library call per sample. Adding and subtracting `2^52`
leans on the FPU's own round-to-nearest-even and is exact for every value here, a
code value in `0..=65535`, so it is bit-exact: 45.9 to 30.8 ms, -31% on that
stage, -3% on `method=1`'s wall.

**A pass that did not measure: four samples per horizontal step.** `blur_row`'s
interior was rewritten to sum four windows at once, four independent chains
instead of one left fold's single chain, with each sample still summed in tap
order so the values were unchanged. Twelve blurs measured 773 to 783 ms against a
775 ms baseline, which is the noise floor, and the output was again byte-identical.
It was reverted: the simpler loop is what stayed, and the note is here so the next
round does not spend the same afternoon on it.

What it says:

- **the edge mask and the blend are still 78% of `method=1`** and 28% of
  `method=0`. Together they are 28 ns/px against 8.5 ns/px for a sigma 0.8 blur.
  Sharpening a page with the unsharp mask costs almost nothing on top of masking
  it.
- the twelve deconvolution blur passes are 784 ms, 65% of `method=0`.
- the plan's tolerance is what makes the second pass possible. Bit-exactness was
  never the bar; it was a convenient check while the arithmetic stayed put.

### what the loops are bound by

`restore` per pixel for `method=1` at three page sizes, one build throughout:

| page | one plane | `restore` | twelve blurs |
| --- | ---: | ---: | ---: |
| 1024x1024 | 4 MB | 38.1 ns/px | 62.2 ns/px |
| 2048x2048 | 16 MB | 36.0 ns/px | 63.8 ns/px |
| 2902x4128 | 48 MB | 37.6 ns/px | 65.4 ns/px |

A working set twelve times larger costs 5%. The first plane fits in a 32 MB L3 and
the last does not, so if DRAM bandwidth were the limit these columns would climb
steeply. They do not, which says the loops are bound by instructions rather than
by memory traffic. Two things follow: fusing passes to save traversals buys a few
percent at most, and rayon would help by using more cores rather than by using more
bandwidth.

### what is left

Scalar micro-optimisation is exhausted. Three passes took 24% off `method=1` and
15% off the deconvolution, and what is left needs a different kind of change:

| loop | work per pixel | measured |
| --- | --- | ---: |
| one sigma 0.8 blur, both passes | 27 mul/add, 14 loads, 7 stores | 5.3 ns/px |
| edge mask and blend | six blurs, two stencils and two passes | 26.5 ns/px |

5.3 ns/px for a blur is about 18 cycles for roughly 48 operations, which is
2.6 operations a cycle, and the stencils run at about 2. A four sample unroll of
the horizontal sum was tried and measured inside the noise, which is what near
peak scalar throughput looks like, and the blend is now vectorised, which is the
second lever and the one that landed:

1. **The Sobel stencil**, which is the last elementwise loop: nine taps, the same
   order, and a `sqrt` that vectors. The same pattern and the same harness apply.
   It is the smaller half of the mask and blend stage, so expect less than the
   blend bought.

One thing worth stating in the other direction: the vector path is
`#[cfg(target_arch = "x86_64")]`, so a build for any other target falls back to the
scalar loop and is exactly as fast as it was before this pass.
Smaller, in case that is not wanted: a rolling three-value window in `blend` (a
few percent, since the loads are L1 hits and the sixteen `min`/`max` dominate).
A chunk-based write, walking `chunks_exact_mut` and handing each sample its own
slice, was tried and **regressed the stage threefold**, 29.4 to 94.9 ms: the
chunk length is a runtime value there, so the constant-length bounds elision the
indexed form gets goes away. That is the second idea in this plan to measure
worse than the code it replaced, after the four sample unroll.

**Rayon is not on the list any more, and the measurement says so.** A single
threaded puller, which is what the bench does, gives the filter no frame level
parallelism at all. A temporal fan-in gives it the shape a real graph has: 13
frames of a 1 Mpx page through `std.AverageFrames`, one pull, with each Deblur
frame reporting its own stage times:

| | wall | reported `restore` |
| --- | ---: | ---: |
| one frame on its own | 108.5 ms | 104.1 ms |
| 13 frames through a fan-in | 527.6 ms | 5002.2 ms |

That is **9.48x**: 12 worker threads already run nine and a half of these frames
at once, and each frame costs 3.7x more under that contention (104 to 385 ms) as
the 12 threads share cache and execution ports. There is no idle core for rayon
to use, and 12 rayon threads inside each of 9 frames would oversubscribe the
machine. The bench's per page column is one frame's latency on a quiet machine,
not the throughput a graph sees, which is about 9x better than it looks.

### concurrency and memory

The stage split above is one frame at a time on one thread. The filter is
`Parallel` with a strict spatial dependency, so VapourSynth already runs whole
frames concurrently and the scratch pool grows to the number of frames in
flight. Each workspace is five f32 planes, so **240 MB for a 12 Mpx frame and
479 MB for a 5806x4128 one**, per frame in flight.

That is visible in the bench: the plugin's peak resident set is 969 MiB on the
deblur workflow against 511 MiB for `Posterize` on the same pages, and against the
reference's 1569 MiB, which is the one comparison it wins. Splitting a single
frame across rayon threads would not add to the scratch, but it is worth measuring
the frame-level parallelism on a real graph before adding it.

## remaining questions

None. Both of the questions this plan left open were answered before the work
started: the tolerance numbers above are what the fixtures assert, and `method=1`
with an explicit `iterations` ignores it silently, including a value outside the
range the deconvolution accepts.
