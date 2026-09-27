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
| `posterize-check` | 49 | 2902×4128, 2903×4128, 5806×4128 | png plus one jpeg, four sizes |

`posterize-check` is why the filters read their dimensions from the frame. a clip
of those pages has no single width, and two of the pages differ by one pixel, so
a filter that trusted the node's `width` would walk the wrong allocation. see
`docs/FINDINGS.md` §8.1.

## the workflows

| workflow | nmanga | plugin |
| --- | --- | --- |
| `levels` | `find_local_peak` then `apply_levels` | `PeakStats` then `Levels(use_props=True)` |
| `shades` | `analyze_gray_shades` | `PeakGrayShades` |
| `posterize` | `posterize_image_by_bits` | `Posterize` |

both sides use `upper_limit=60`, `peak_percentage=0.25`, white peaks skipped,
`threshold=0.01` and `bits=4`, which are the `nmanga` orchestrator defaults.

the plugin side keeps the decoded frame for each page in the VapourSynth frame
cache, so its peak memory covers the whole clip rather than one page. that is the
real cost of asking the core for frames, and it is what the memory column shows.
`--cache MB` bounds it if a smaller number is wanted.

## how to run

```powershell
uv sync --extra dev --extra dev-tests
uv run --extra golden --extra dev-tests tools\bench.py
uv run --extra golden --extra dev-tests tools\bench.py --write docs\BENCH.md
```

`--limit N` caps the page count, `--suite` and `--workflow` narrow the run, and
`--keep-logs` leaves the per-run json under `target/bench/`.

## results

wall seconds per stage, peak resident set for the whole run, and how many pages
both pipelines gave the same black and white level. the level comparison is the
correctness claim: the plugin has to agree with the reference on every page, not
just run faster.

<!-- bench:start -->
| pages | workflow | pipeline | decode | analyze | apply | total | per page | peak rss |
| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 129 | levels (levels) | nmanga | 2.34 s | 5.09 s | 0.28 s | **7.71 s** | 59.8 ms | 118 MiB |
| 129 | levels (levels) | vapoursynth | 2.92 s | 0.87 s | 1.01 s | **4.80 s** | 37.2 ms | 127 MiB |
| 49 | shades (posterize) | nmanga | 1.97 s | 11.33 s | 0.00 s | **13.30 s** | 271.5 ms | 145 MiB |
| 49 | shades (posterize) | vapoursynth | 2.85 s | 1.18 s | 0.00 s | **4.03 s** | 82.2 ms | 488 MiB |
| 49 | posterize (posterize) | nmanga | 1.91 s | 0.00 s | 16.19 s | **18.11 s** | 369.5 ms | 273 MiB |
| 49 | posterize (posterize) | vapoursynth | 2.67 s | 0.00 s | 0.58 s | **3.24 s** | 66.2 ms | 510 MiB |

| pages | workflow | pipeline | speedup | memory ratio | levels agreed |
| ---: | --- | --- | ---: | ---: | --- |
| 129 | levels (levels) | vapoursynth vs nmanga | 1.61x | 1.07x | 129/129 |
| 49 | shades (posterize) | vapoursynth vs nmanga | 3.30x | 3.35x | 49/49 |
| 49 | posterize (posterize) | vapoursynth vs nmanga | 5.58x | 1.87x | 49/49 |
<!-- bench:end -->

## what the numbers say

### correctness

every page agrees. 129 of 129 on `level-check` and 49 of 49 on `posterize-check`
produced the same black and white level from the plugin as from the reference,
including the spread and the two pages whose widths differ by one pixel. that is
the claim that matters; the timings are the reason to bother.

### analysis

this is where the plugin wins. `find_local_peak` costs about 39 ms per page
against 6.7 ms for `PeakStats`, and `analyze_gray_shades` costs about 231 ms
against 24 ms for `PeakGrayShades`, roughly a six to ten times difference.

the reference materialises a full `ndarray` of the page and converts it to
grayscale again inside the call, so the page is copied before the histogram
starts, and then `np.histogram` does generic binning over a `range` rather than a
256 bin count over `uint8`. the plugin walks the plane once into a `[u64; 256]`
array and never builds a second copy.

the peak search is not the difference. it runs over at most 61 bins either way,
and `scipy.signal.find_peaks` and `src/peaks.rs` both finish in microseconds.

### the level curve

`Levels` and `Posterize` both apply one 256 entry table to the plane, so `apply`
is a memory pass. the reference's `apply_levels` is fast for the same reason and
beats the plugin here: 0.28 s against 1.01 s over 129 pages. Pillow's
`image.point` is a tight native loop over an image it has already decoded, while
the plugin also allocates the output frame, copies the source properties onto it
and hands the result back through the frame cache. at 2.8 megapixels a page that
per-frame overhead is a large share of the work, which is why the same stage
looks three times faster per megapixel on the 12 megapixel `posterize-check`
pages.

since the total is what a caller pays, the plugin still finishes the whole
`levels` workflow faster: 1.61x, because the analysis more than makes up for the
apply.

### posterize

the biggest gap, 5.58x. `posterize_image_by_bits` builds its table by calling a
python lambda 256 times, maps the page, then runs `quantize(colors, dither=NONE)`
and `convert("L")`, which is two more passes over a 12 megapixel page plus a
palette conversion. `docs/FINDINGS.md` §6.1 shows that quantization is a no-op,
so the plugin omits it and writes straight into the output frame: 0.58 s against
16.19 s for the same 49 pages.

### memory

the plugin's peak is higher, and the reason is concurrency rather than leakage.
VapourSynth keeps frames in flight across its worker threads and in its frame
cache, so a 12 megapixel page is about 12 MiB as `GRAY8` and 36 MiB as `RGB24`,
and a dozen of those are live at once. the reference holds one page plus numpy's
temporaries at a time, in one thread.

the cache is a real dial. the `shades` run repeated with `--cache 64`:

| cache | total | peak rss | memory ratio |
| --- | ---: | ---: | ---: |
| core default | 4.03 s | 488 MiB | 3.35x |
| `--cache 64` | 4.00 s | 256 MiB | 1.75x |

same wall time, half the memory. bound the cache when 12 megapixel pages are
being processed and the extra is worth avoiding.

`levels` shows almost no difference (1.07x) because its pages are 2.8 megapixels
and the core ends up evicting most of them anyway.

### what is not measured

- **decoding is not comparable.** the plugin side goes through
  `vapoursynth-imageseqs`, which decodes with the `image` crate, and the
  reference decodes with Pillow, which is libjpeg-turbo and libpng. the plugin's
  `decode` column is consistently the larger one, by 25% to 45% across these
  sets, and none of that is `nimages` code. the `analyze` and `apply` columns are
  the comparison.
- **stage times inside the VapourSynth graph are approximate.** the core may
  serve a request from the cache or schedule work ahead of the request that
  needed it, so splitting `decode`, `analyze` and `apply` by request boundary is
  a good approximation rather than an exact attribution. the totals are exact.
- **neither side writes files.** no VapourSynth writer is installed here, so both
  stop once the adjusted page is in memory. an encoder would add to both.
- **the grayscale conversion differs.** the plugin path uses
  `resize.Bicubic(matrix_s="470bg", range_s="full")` and the reference uses
  `Image.convert("L")`. the coefficients are close but not identical, so a page
  whose peak sits within a code value of a bin boundary could disagree. none did
  in these sets.
- **these numbers are one machine.** 12 VapourSynth worker threads, windows, and
  a warm page cache. they describe the shape of the difference, not a guarantee.
