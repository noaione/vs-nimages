# bench

what the plugin costs against the python pipeline it replaces, measured on the
same pages.

`sandbox/` holds the pages and is a private working tree, so it is not committed.
`tools/bench.py` runs both pipelines and refreshes the results block below.

## what is compared

two pipelines over one image list:

- **nmanga** — `nmanga.autolevel` on Pillow images with numpy and scipy, one page
  at a time in a single thread. this is the behaviour `IMPLEMENTATIONS.md`
  describes as the source of the algorithms.
- **vapoursynth** — `imgseqs.Read(files, mismatch=True)` through
  `resize.Bicubic` to `GRAY8`, then `PeakStats` + `Levels(use_props=True)`, or
  `PeakGrayShades`, or `Posterize`, or `Deblur`.

each pipeline runs in its own process, so the peak resident set of one is
comparable with the other. the measurement is a high water mark taken from
`GetProcessMemoryInfo` on windows and `ru_maxrss` elsewhere.

## the page sets

| set | pages | sizes | notes |
| --- | ---: | --- | --- |
| `level-check` | 129 | 1404×2000 | jpeg, one size |
| `level-webp-check` | 44 | 2903×4128 | lossy webp, one size, **odd width** |
| `posterize-check` | 49 | 2902×4128, 2903×4128, 5806×4128 | png plus one jpeg, four sizes |

`posterize-check` is why the filters read their dimensions from the frame. a clip
of those pages has no single width, and two of the pages differ by one pixel, so
a filter that trusted the node's `width` would walk the wrong allocation. see
`docs/FINDINGS.md` §8.1.

`level-webp-check` is the harshest set and the only one that needs a conversion
before the filters see it, for two reasons recorded in
[what a webp actually hands out](#what-a-webp-actually-hands-out): `imgseqs`
returns limited range YUV for a lossy webp, and its 2903 pixel width is not
divisible by the format's subsampling factor.

## the workflows

| workflow | nmanga | plugin |
| --- | --- | --- |
| `levels` | `find_local_peak` then `apply_levels` | `PeakStats` then `Levels(use_props=True)` |
| `shades` | `analyze_gray_shades` | `PeakGrayShades` |
| `posterize` | `posterize_image_by_bits` | `Posterize` |
| `deblur` | `deblur_deconv` | `Deblur(method=0)` |
| `deblur-unsharp` | `deblur_edge_sharp` | `Deblur(method=1)` |

both sides use `upper_limit=60`, `peak_percentage=0.25`, white peaks skipped,
`threshold=0.01` and `bits=4`, which are the `nmanga` orchestrator defaults.

the two deblur workflows share `radius=0.8`, `iterations=6`, `threshold=2` and
`overshoot=0`, and each runs the `strength` its own method defaults to, `0.65`
for the deconvolution and `0.85` for the unsharp mask. the reference has no
refinement loop in `deblur_edge_sharp`, so `deblur-unsharp` runs `Deblur` with
`method=1`, which ignores `iterations`.

the plugin side keeps decoded frames in the VapourSynth frame cache, so its peak
memory covers more than one page. every run here uses a **512 MiB** cache, which
is what a caller would set for a manga volume: enough to hold a whole volume of
SD to HD-ish pages as `GRAY8`, and far below the core's own default of several
gigabytes. `--cache MB` changes it.

`imgseqs.Read` hands out whatever the container holds, so the plugin side needs a
conversion before `PeakStats` can see one plane: `to_gray8` in `tools/bench.py`
picks the cheapest graph that is still correct, which is one `resize` for an RGB
sequence, a trim plus an RGB round trip for a YUV one, and a per-frame form only
when the sequence mixes formats.

## how to run

```powershell
uv sync --extra dev --extra dev-tests
uv run --extra golden --extra dev-tests tools\bench.py
uv run --extra golden --extra dev-tests tools\bench.py --write docs\BENCH.md
```

`--limit N` caps the page count, `--suite` and `--workflow` narrow the run, and
`--keep-logs` leaves the per-run json under `target/bench/`. `--cache MB` moves
the plugin side's frame cache off its 512 MiB default.

## results

wall seconds per stage and peak resident set for the whole run. `levels agreed`
compares black and white levels for `levels`, and the number of reported shades
for `shades`. `posterize` records zero placeholders on both sides, so its 49/49
does not compare pixels. the deblur rows record no comparison and show 0/0.
golden fixtures and `tests/check-nimages.py` cover the filter outputs.

`resize` is the plugin's `resize.Bicubic` to `GRAY8` plus the frame plumbing
around it, so it stays `0.00 s` for the reference, which converts inside its
decode instead. read the plugin's `decode + resize` against the reference's
`decode`.

<!-- bench:start -->
| pages | workflow | pipeline | decode | resize | analyze | apply | total | per page | peak rss |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | levels (levels) | nmanga | 2.20 s | 0.00 s | 5.84 s | 0.26 s | **8.30 s** | 64.3 ms | 119 MiB |
| 129 | levels (levels) | vapoursynth | 1.93 s | 0.30 s | 0.23 s | 0.12 s | **2.59 s** | 20.0 ms | 128 MiB |
| 49 | shades (posterize) | nmanga | 1.94 s | 0.00 s | 10.54 s | 0.00 s | **12.49 s** | 254.8 ms | 146 MiB |
| 49 | shades (posterize) | vapoursynth | 2.02 s | 0.30 s | 0.38 s | 0.00 s | **2.70 s** | 55.2 ms | 489 MiB |
| 49 | posterize (posterize) | nmanga | 1.78 s | 0.00 s | 0.00 s | 15.61 s | **17.39 s** | 354.8 ms | 273 MiB |
| 49 | posterize (posterize) | vapoursynth | 2.08 s | 0.33 s | 0.00 s | 0.21 s | **2.63 s** | 53.6 ms | 511 MiB |
| 49 | deblur (posterize) | nmanga | 1.80 s | 0.00 s | 0.00 s | 297.02 s | **298.83 s** | 6098.5 ms | 1569 MiB |
| 49 | deblur (posterize) | vapoursynth | 1.91 s | 0.42 s | 0.00 s | 17.66 s | **19.99 s** | 408.0 ms | 880 MiB |
| 49 | deblur-unsharp (posterize) | nmanga | 1.75 s | 0.00 s | 0.00 s | 151.17 s | **152.92 s** | 3120.8 ms | 1569 MiB |
| 49 | deblur-unsharp (posterize) | vapoursynth | 1.91 s | 0.41 s | 0.00 s | 4.92 s | **7.24 s** | 147.8 ms | 880 MiB |
| 44 | levels (webp) | nmanga | 11.20 s | 0.00 s | 5.68 s | 0.30 s | **17.18 s** | 390.5 ms | 283 MiB |
| 44 | levels (webp) | vapoursynth | 7.08 s | 1.32 s | 0.34 s | 0.20 s | **8.95 s** | 203.4 ms | 400 MiB |

| pages | workflow | pipeline | speedup | memory ratio | levels agreed |
| ---: | --- | --- | ---: | ---: | --- |
| 129 | levels (levels) | vapoursynth vs nmanga | 3.21x | 1.08x | 129/129 |
| 49 | shades (posterize) | vapoursynth vs nmanga | 4.62x | 3.34x | 49/49 |
| 49 | posterize (posterize) | vapoursynth vs nmanga | 6.62x | 1.87x | 49/49 |
| 49 | deblur (posterize) | vapoursynth vs nmanga | 14.95x | 0.56x | 0/0 |
| 49 | deblur-unsharp (posterize) | vapoursynth vs nmanga | 21.12x | 0.56x | 0/0 |
| 44 | levels (webp) | vapoursynth vs nmanga | 1.92x | 1.41x | 44/44 |
<!-- bench:end -->

## what a webp actually hands out

two things about the `level-webp-check` set are worth recording, because both
come from `vapoursynth-imageseqs` rather than from `nimages`.

**the luma is limited range, and the reference's is not.** for a lossy webp,
`imgseqs` hands out `YUV420P8` with `_Range=limited` and `_Matrix=5`. taking that
luma plane straight into `PeakStats` analyses different numbers than the
reference does: the reference decodes the webp to RGB and converts with
`Image.convert("L")`, which is full range. on the same page, plane 0 spans
34..244 and Pillow's luma spans 21..255, a difference of about 13 code values on
average. a black level found on one of those is not the black level of the other.

the fix is to route the plugin through RGB as well, so both sides analyse the
same samples. `resize.Bicubic(yuv, format=vs.RGB24)` uses the frame's own
`_Matrix` and `_Range`, and converting that to `GRAY8` matches Pillow's luma to
within **0.01** of a code value. the bench does exactly that, which is why the
webp row still agrees 44/44.

**the frames are an illegal size for their own format.** these pages are 2903
pixels wide, and 2903 is odd, so it is not divisible by the 4:2:0 subsampling
factor. VapourSynth refuses it everywhere else:

```text
core.std.BlankClip(width=2903, format=vs.YUV420P8)  ->  BlankClip: invalid width
core.resize.Bicubic(clip, format=vs.GRAY8)          ->  Resize error 1027:
                                                        image dimensions must be
                                                        divisible by subsampling factor
```

`imgseqs` produces the frame anyway, so any downstream filter that checks
dimensions rejects it. the documented workaround from the `vapoursynth-imageseqs`
README is to trim the odd edge, convert, and put the edge back; that is what the
bench does. its `FrameEval` form is for a clip whose format varies, which these
suites are not, so the trim happens once on the clip instead of per frame.

both are worth fixing on the `imgseqs` side: a lossy webp is full range in
practice, and a frame whose width breaks its own format's subsampling rule cannot
be used by anything.

## what the numbers say

### correctness

the `levels` workflow agrees on black and white levels for all 129 pages in
`level-check` and all 44 pages in `level-webp-check`. the `shades` workflow
agrees on the number of shades for all 49 pages in `posterize-check`, including
the spread and the two pages whose widths differ by one pixel. the benchmark
does not compare shade values or percentages.

the posterize placeholders and deblur's 0/0 carry no pixel-parity evidence.
`tests/check-nimages.py` compares posterize pixels and shade properties against
the golden vectors, and checks deblur against its frozen fixture tolerance.

### analysis

the analysis stage is more than an order of magnitude faster in both workflows.
`find_local_peak` costs about 45.3 ms per page against 1.8 ms for `PeakStats`,
and `analyze_gray_shades` costs about 215.1 ms against 7.8 ms for
`PeakGrayShades`. these are stage times, excluding decode and resize.

the reference materialises a full `ndarray` of the page and converts it to
grayscale again inside the call, so the page is copied before the histogram
starts, and then `np.histogram` does generic binning over a `range` rather than
a 256 bin count over `uint8`. the plugin samples at most 256 adjacent pairs on
planes of at least 16384 pixels, then counts into either one `[u64; 256]` table
or four striped tables, without building a second image copy. smaller planes
use the direct reader without probing. the current `shades` analysis takes
0.38 s over 49 pages. [09-striped-histogram-counters.md](improvements/09-striped-histogram-counters.md)
records the isolated reader comparisons and the noise/size tradeoffs.

the peak search is not the difference. it runs over at most 61 bins either way,
and `scipy.signal.find_peaks` and `src/peaks.rs` both finish in microseconds.

### the level curve

`Levels` and `Posterize` both apply one 256 entry table to the plane, so `apply`
is a memory pass, and the plugin's is the cheaper one. over 129 pages the plugin
spends 0.12 s and the reference 0.26 s, about 0.9 ms against 2.0 ms a page.

`apply_levels` is not slow. Pillow's `image.point` is a tight native loop over an
image it has already decoded, and the plugin also allocates the output frame,
copies the source properties onto it and hands the result back through the frame
cache. at 2.8 megapixels a page that per-frame overhead is a visible share.

### posterize

posterize finishes 6.62x faster overall. almost all of the difference is in the
apply stage: 0.21 s against 15.61 s over 49 pages, about 4.3 ms against 318.6 ms
a page. deblur has the larger total speedups.

### deblur

in both deblur workflows the plugin's own work still dominates the wall time,
while the reference spends seconds per page here: 19.99 s against
298.83 s over 49 pages, **14.95x** for the deconvolution and **21.12x** for the
unsharp mask (7.24 s against 152.92 s). levels and shades gain mainly in
analysis; posterize gains in apply.

the plugin side of those two rows started this work at 77.76 s and 36.04 s. the
current pipeline totals are about 74% and 80% lower than those historical runs.
the separate optimization measurements are recorded in
[03-deblur-filter.md](improvements/03-deblur-filter.md) and
[04-optimization-review.md](improvements/04-optimization-review.md): reading each
window from row slices instead of a per-tap helper, `sqrt` instead of the
`f32::hypot` libcall, a ties-to-even round that does not need `roundsd`, an AVX2
path for the blend stencil, then the horizontal pass vectorized eight columns at a
time with its feature check hoisted out of the row loop and the vertical sum kept
in a register across every tap of a column block. the last two are bit-identical
to the scalar code. [12-deblur-conversion-dispatch.md](improvements/12-deblur-conversion-dispatch.md)
records the later sample-width writer specialization. the current totals combine
these changes with decode and frame plumbing; they do not isolate kernel gains.

the unsharp mask is the cheaper operation by a wide margin, 147.8 ms a page
against 408.0 ms. method 0 has twelve candidate blurs at six iterations and
method 1 has one; both also run the mask's two blurs. `method=1` is the cheaper
choice for a caller who wants the unsharp operation.

where that time goes is not visible in this table: `Deblur` reports `luma`,
`restore` and `write`, all three of which land in `apply`, and the reference's
deblur branch times one call, so `apply` is the whole operation on both sides.
`docs/improvements/03-deblur-filter.md` measured the inside of `restore`
separately and found the edge mask and the halo clamp were 78% of the
unsharp mask and 28% of the deconvolution, with the twelve blur passes 65% of the
deconvolution in that build. those historical shares are not a stage split of
the current results.

the per page column is one frame's latency on a quiet machine. a graph that asks
for several frames can run them concurrently on 12 worker threads.
[03-deblur-filter.md](improvements/03-deblur-filter.md) measured about 9.5x
concurrent work on 1 megapixel frames, with higher per-frame cost under
contention and about 2.7x aggregate throughput over sequential pulls. it does
not establish a throughput gain for the current full-page workload.

peak memory runs the other way for once. 880 MiB against the reference's
1569 MiB makes both deblur rows smaller processes than the reference: scipy holds
several megapixel-sized float64 temporaries per stage of a 24 megapixel page,
while the plugin's scratch is four f32 planes plus a ring of at most 129 filtered
rows: 194 MB for a 12 megapixel frame and 387 MB for the 5806x4128 spread.

### the webp set

this is the harshest set and the narrowest win, 1.92x, because the decode is no
longer a rounding error. a 12 megapixel lossy webp costs the reference 254.5 ms a
page of decode and the plugin 160.9 ms of decode plus 30.0 ms of resize.
decoding varies by library and set.

its `resize` is 30.0 ms a page rather than the 2.3 ms the jpeg set pays, because
the conversion is a crop, a YUV to RGB resize and an RGB to Gray resize over 12
megapixels instead of one resize over 2.8.

even so the analysis is about seventeen times faster, 7.7 ms a page against
129 ms, and the level decisions are identical on all 44 pages. the lesson is that
on a set this large the plugin's advantage is bounded by the decoder it has to
sit behind, not by its own work.

### memory

the plugin's peak is higher on four of the six workflow rows, and the reason is
concurrency rather than leakage.
VapourSynth keeps frames in flight across its worker threads and holds decoded
frames in its cache, so a 12 megapixel page is about 12 MiB as `GRAY8` and 36 MiB
as `RGB24`, and a dozen of those are live at once. the reference holds one page
plus numpy's temporaries at a time, in one thread. the 512 MiB cache budget
limits cached frames; in-flight frames and pooled scratch also contribute to
process RSS.

both deblur methods are the exceptions, at 0.56x. their pages are the largest
here, so the reference's float64 temporaries dominate its side of the comparison.
the plugin's scratch, while large, is allocated once per frame in flight rather
than per stage.

the cache is a dial for peak memory. an earlier `shades` run compared three
budgets; this separate probe was not rerun with the results block above:

| cache | plugin total | peak rss | memory ratio |
| --- | ---: | ---: | ---: |
| 128 MiB | 3.66 s | 256 MiB | 1.75x |
| 512 MiB | 3.43 s | 488 MiB | 3.35x |
| 2048 MiB | 3.85 s | 488 MiB | 3.34x |

most of the 49 pages are about 12 megapixels, with a 24 megapixel spread. in that
probe, going from 512 MiB to 2 GiB left peak RSS unchanged, and dropping to
128 MiB reduced it from 488 MiB to 256 MiB. totals ranged from 3.43 s to 3.85 s;
repeated runs are needed to separate timing effects from noise.

`levels` barely notices any of it (1.08x) because its pages are 2.8 megapixels
and the core evicts most of them either way.

### how the stages are attributed

both sides are timed from inside, which is the only way to split a VapourSynth
graph correctly.

- **nmanga** is sequential python, so each stage is timed around its own call.
  its `decode` covers `Image.open`, `load` and `convert("L")`.
- **vapoursynth** reads three clocks. `vapoursynth-imageseqs` reports its own
  frame build under `debug=1`, which is the `decode` column. `nimages` reports
  each filter's stages under `debug=1`, which is `analyze` and `apply`. the
  `resize` column is what is left of the wall time after those three, so it
  covers `resize.Bicubic` and the frame plumbing around it.

`Deblur` reports `luma`, `restore` and `write`, and all three land in the `apply`
column; the reference's deblur branches time one call, so `apply` is the whole
operation on both sides. the split inside `restore` is not reported, so the mask
and the kernels cannot be told apart from this table.
`docs/improvements/03-deblur-filter.md` has that split, measured separately.

timing the plugin from outside does not work. VapourSynth does not keep an
intermediate frame between two external requests, so pulling a `PeakStats` node
and then a `Levels` node re-runs the analysis and counts it twice. an earlier
version of this document did exactly that, and reported the plugin's `apply`
stage as three times slower than it is.

### what is not measured

- **decoding is not comparable.** the plugin side goes through
  `vapoursynth-imageseqs`, which decodes with the `image` crate and then
  `resize.Bicubic` converts to `GRAY8`, while the reference decodes with Pillow
  and converts with `Image.convert("L")`. against `decode + resize` the plugin
  path is between 26% faster and 35% slower than the reference's `decode`, in no
  consistent direction, and none of it is `nimages` code,
  so read the `analyze` and `apply` columns as the comparison.
- **neither side writes files.** no VapourSynth writer is installed here, so both
  stop once the adjusted page is in memory. an encoder would add to both.
- **the grayscale conversion differs.** the plugin path uses
  `resize.Bicubic(matrix_s="470bg", range_s="full")` and the reference uses
  `Image.convert("L")`. the coefficients are close but not identical, so a page
  whose peak sits within a code value of a bin boundary could disagree. none did
  in these sets.
- **these numbers are one machine.** 12 VapourSynth worker threads, windows, and
  a warm page cache. the reference side moves by a third between runs on the same
  pages, so read the ratios as the shape of the difference, not a guarantee.
