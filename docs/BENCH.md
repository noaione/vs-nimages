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
  `PeakGrayShades` and `Posterize`.

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

both sides use `upper_limit=60`, `peak_percentage=0.25`, white peaks skipped,
`threshold=0.01` and `bits=4`, which are the `nmanga` orchestrator defaults.

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

wall seconds per stage, peak resident set for the whole run, and how many pages
both pipelines gave the same black and white level. the level comparison is the
correctness claim: the plugin has to agree with the reference on every page, not
just run faster.

`resize` is the plugin's `resize.Bicubic` to `GRAY8` plus the frame plumbing
around it, so it stays `0.00 s` for the reference, which converts inside its
decode instead. read the plugin's `decode + resize` against the reference's
`decode`.

<!-- bench:start -->
| pages | workflow | pipeline | decode | resize | analyze | apply | total | per page | peak rss |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | levels (levels) | nmanga | 3.92 s | 0.00 s | 6.65 s | 0.31 s | **10.89 s** | 84.4 ms | 119 MiB |
| 129 | levels (levels) | vapoursynth | 2.59 s | 0.44 s | 0.52 s | 0.18 s | **3.74 s** | 29.0 ms | 128 MiB |
| 49 | shades (posterize) | nmanga | 3.08 s | 0.00 s | 15.51 s | 0.00 s | **18.59 s** | 379.4 ms | 146 MiB |
| 49 | shades (posterize) | vapoursynth | 4.12 s | 0.64 s | 1.06 s | 0.00 s | **5.82 s** | 118.8 ms | 488 MiB |
| 49 | posterize (posterize) | nmanga | 2.49 s | 0.00 s | 0.00 s | 19.77 s | **22.25 s** | 454.2 ms | 273 MiB |
| 49 | posterize (posterize) | vapoursynth | 3.06 s | 0.57 s | 0.00 s | 0.32 s | **3.95 s** | 80.5 ms | 511 MiB |
| 44 | levels (webp) | nmanga | 12.71 s | 0.00 s | 6.94 s | 0.36 s | **20.00 s** | 454.6 ms | 283 MiB |
| 44 | levels (webp) | vapoursynth | 6.99 s | 1.35 s | 0.74 s | 0.19 s | **9.27 s** | 210.7 ms | 399 MiB |

| pages | workflow | pipeline | speedup | memory ratio | levels agreed |
| ---: | --- | --- | ---: | ---: | --- |
| 129 | levels (levels) | vapoursynth vs nmanga | 2.91x | 1.07x | 129/129 |
| 49 | shades (posterize) | vapoursynth vs nmanga | 3.19x | 3.35x | 49/49 |
| 49 | posterize (posterize) | vapoursynth vs nmanga | 5.64x | 1.87x | 49/49 |
| 44 | levels (webp) | vapoursynth vs nmanga | 2.16x | 1.41x | 44/44 |
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

every page agrees. 129 of 129 on `level-check` and 49 of 49 on `posterize-check`
produced the same black and white level from the plugin as from the reference,
including the spread and the two pages whose widths differ by one pixel. that is
the claim that matters; the timings are the reason to bother.

### analysis

this is where most of the win is. `find_local_peak` costs about 50 ms per page
against 3.7 ms for `PeakStats`, and `analyze_gray_shades` costs about 280 ms
against 16 ms for `PeakGrayShades`. an order of magnitude either way.

the reference materialises a full `ndarray` of the page and converts it to
grayscale again inside the call, so the page is copied before the histogram
starts, and then `np.histogram` does generic binning over a `range` rather than a
256 bin count over `uint8`. the plugin walks the plane once into a `[u64; 256]`
array and never builds a second copy.

the peak search is not the difference. it runs over at most 61 bins either way,
and `scipy.signal.find_peaks` and `src/peaks.rs` both finish in microseconds.

### the level curve

`Levels` and `Posterize` both apply one 256 entry table to the plane, so `apply`
is a memory pass, and the plugin's is the cheaper one. over 129 pages the plugin
spends 0.20 s and the reference 0.33 s, about 1.6 ms against 2.6 ms a page.

`apply_levels` is not slow. Pillow's `image.point` is a tight native loop over an
image it has already decoded, and the plugin also allocates the output frame,
copies the source properties onto it and hands the result back through the frame
cache. at 2.8 megapixels a page that per-frame overhead is a visible share.

### posterize

the biggest gap, 6.25x, and almost all of it is the apply stage: 0.24 s against
16.14 s over 49 pages, about 5 ms against 329 ms a page.

`posterize_image_by_bits` builds its table by calling a python lambda 256 times,
maps the page, then runs `quantize(colors, dither=NONE)` and `convert("L")`, which
is two more passes over a 12 megapixel page plus a palette conversion.
`docs/FINDINGS.md` §6.1 shows the quantization is a no-op, so the plugin omits it
and writes straight into the output frame.

### the webp set

this is the harshest set and the narrowest win, 1.89x, because the decode is no
longer a rounding error. a 12 megapixel lossy webp costs Pillow 258 ms a page and
`imgseqs` 163 ms, so the plugin's decode is the faster one here — the opposite of
the jpeg and png sets.

its `resize` is 32 ms a page rather than the 2.6 ms the jpeg set pays, because
the conversion is a crop, a YUV to RGB resize and an RGB to Gray resize over 12
megapixels instead of one resize over 2.8.

even so the analysis is still about nine times faster, 16 ms a page against
141 ms, and the level decisions are identical on all 44 pages. the lesson is that
on a set this large the plugin's advantage is bounded by the decoder it has to
sit behind, not by its own work.

### memory

the plugin's peak is higher, and the reason is concurrency rather than leakage.
VapourSynth keeps frames in flight across its worker threads and holds decoded
frames in its cache, so a 12 megapixel page is about 12 MiB as `GRAY8` and 36 MiB
as `RGB24`, and a dozen of those are live at once. the reference holds one page
plus numpy's temporaries at a time, in one thread. the 512 MiB cache is what caps
the plugin's side of it.

the cache is the dial for peak memory, and 512 MiB sits at the top of the curve
for this set. the `shades` run repeated at three sizes:

| cache | plugin total | peak rss | memory ratio |
| --- | ---: | ---: | ---: |
| 128 MiB | 3.66 s | 256 MiB | 1.75x |
| 512 MiB | 3.43 s | 488 MiB | 3.35x |
| 2048 MiB | 3.85 s | 488 MiB | 3.34x |

the 49 pages are 12 megapixels each, so 588 MiB as `GRAY8`, and 512 MiB already
holds nearly all of them: going to 2 GiB changes nothing, and dropping to 128 MiB
halves the memory without costing wall time. the differences in the total column
are run-to-run noise, so read this as a memory dial rather than a speed one.

`levels` barely notices any of it (1.07x) because its pages are 2.8 megapixels
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
  path is 25% to 39% slower across these sets, and none of it is `nimages` code,
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
