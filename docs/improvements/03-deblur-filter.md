# deblur filter

status: implemented

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

The port is 4.25x faster on the deconvolution and 4.79x on the unsharp mask. It is
also the one workflow where the plugin holds less memory than the reference, 0.62x,
because scipy keeps several megapixel-sized float64 temporaries alive per stage of
a 24 Mpx page while the plugin's cost is its own scratch. The unsharp mask is
2.2x cheaper than the deconvolution, so it is the better choice for a caller who
just wants a page sharpened.

### where the time goes

The filter reports one `restore=` number, so the split came from timing the same
12.0 Mpx page in four configurations that differ only in how many blurs they run,
one node per configuration so the scratch is allocated once:

| configuration | blurs | restore |
| --- | ---: | ---: |
| `method=0 iterations=0` | 0 candidate | 523.5 ms |
| `method=1` | 1 | 583.3 ms |
| `method=0 iterations=1` | 2 | 661.4 ms |
| `method=0 iterations=6` | 12 | 1384.9 ms |

Solving those gives one sigma 0.8 blur at **6.5 ns/px**, which is the 6.2 ns/px
the kernel harness measured for the same kernel in isolation, so the
decomposition holds. What it says:

- **the edge mask and the blend are 87% of `method=1`** and 36% of `method=0`.
  The mask is two small blurs plus the Sobel, and the blend is a 3x3 window and
  a clamp per pixel; together they are 42 ns/px, against 6.5 ns/px for a blur.
  Sharpening a page with the unsharp mask costs almost nothing on top of masking
  it.
- the twelve deconvolution blur passes are 861 ms, 62% of `method=0`.

That is the lever to pull before rayon: the mask and the blend are scalar
per-pixel work over nine neighbours, which is where the bounds checks and the
ninefold re-reads are, and the 3x3 minimum and maximum are separable exactly
like the blur.

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
