# book peak benchmark

comparison with book fallback enabled in the original nmanga Python path and
in native `PeakStats` followed by `Levels`. measured 2026-11-14.

## method

`tools/book_peak_bench.py` runs each pipeline and each repetition in a fresh
process. the median of three repetitions is reported. `scan` includes the book
prepass and consensus resolution. `output frames` includes opening/requesting
every page, applying levels only to eligible pages, and materializing excluded
pages without levels. native graph construction is timed, including its
synchronous `PeakStats` scan and `Levels` setup. `total` is the measured scan
plus output-frame wall time, including worker scheduling.

lazy scipy imports and the initial native/device binding call are warmed
outside the timer. the image files are already in the OS cache after generation;
this is not a cold-disk measurement. each native graph has a fresh core/process,
a 512 MiB frame cache, and decoder prefetch 0. the Pillow scan and output stages
use the stated number of Python workers. the native output stage uses that many
Python request workers and core threads; its book prepass remains serial.
the 24-page 12 MP gray book occupies about 274 MiB of source pixels and fits
inside that cache. a volume larger than the cache can require additional
decoding in the native output pass.

both implementations consume the same PNG files. the reference calls nmanga's
public `analyze_book_peak_page` and `resolve_book_peaks` with `LevellingPolicy`,
then its
`gamma_correction` and `apply_levels`. the native graph is `imgseqs.Read -> PeakStats(book_peak_fallback=1) -> Levels` on eligible
pages. the GPU row uses the same host prepass, uploads the requested pages for
mapping, and downloads output frames for the consumer. it does not accelerate
the book prepass's median or consensus. regular `PeakStats(gpu=1)` without
book mode computes its median histogram on the device.

all three paths match reference commit `6bd2775`: `upper_limit=64`,
`median_filter=True`, `peak_percentage=0.25`, disabled prominence, white detection
enabled, bounds `(8, 55)`, `book_peak_tolerance=6`, and offset 0. hashing runs separately,
after each timed pass. every repeated run is checked against the first Python
run for black/white levels, both local threshold flags, eligibility, inheritance,
and all output-plane bytes. any difference returns a nonzero exit status.

peak RSS is the process high-water mark, including imports, frame caches and
the subsequent validation pass. it is not a stage's incremental allocation.
macOS `ru_maxrss` is read in bytes; Linux is converted from KiB.

## input

the private `sandbox/level-check`, `sandbox/level-webp-check`, and
`sandbox/posterize-check` sets are absent from this checkout. this run uses
24 generated `GRAY8` PNG pages at each listed size. eight page types repeat
three times: three pages with local black 12, one with local black 20, one with
a weak connected 3×3 dark patch at 20, one with one dark noise pixel, one bright
page, and one with qualified black 0. local-peak pages also contain midtones.

there are 12 usable votes; 9 vote for 12. the consensus is 12. 15 pages are
eligible, including 6 using the book peak; 9 are excluded. the three pages at
local peak 20 exceed book 12 plus tolerance 6, so they now use 12 too. the weak patch is
below the default height threshold but survives the median. the noise pixel
does not survive it. input generation and PNG encoding are outside the timer.
no private page is copied or included in the report.

encoding output PNGs, copying skipped source files, destination checks, and
terminal progress are excluded from both timings. these are decoded-output
pipeline timings, not complete CLI timings. the real Python CLI can copy a
skipped file without decoding its pixels again; this benchmark instead
materializes skipped output pixels on both sides for a common consumer.

## results

macOS 27.2.1 arm64, Apple M4, Python 3.12.10, numpy 2.5.3, scipy 1.18.1,
Pillow 13.0.0, VapourSynth R80, and the local release build of this working
tree. Vulkan uses the Apple M4 device.

<!-- book-bench:start -->
| book | threads | pipeline | scan | output frames | total | peak rss | decisions | pixels |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 768x512 | 1 | nmanga | 0.110 s | 0.008 s | 0.117 s | 100 MiB | 24/24 | 24/24 |
| 768x512 | 1 | vapoursynth | 0.049 s | 0.003 s | 0.052 s | 51 MiB | 24/24 | 24/24 |
| 768x512 | 1 | vapoursynth-gpu | 0.049 s | 0.014 s | 0.063 s | 115 MiB | 24/24 | 24/24 |
| 768x512 | 4 | nmanga | 0.088 s | 0.003 s | 0.091 s | 102 MiB | 24/24 | 24/24 |
| 768x512 | 4 | vapoursynth | 0.049 s | 0.002 s | 0.051 s | 51 MiB | 24/24 | 24/24 |
| 768x512 | 4 | vapoursynth-gpu | 0.049 s | 0.006 s | 0.056 s | 119 MiB | 24/24 | 24/24 |
| 2902x4128 | 1 | nmanga | 3.258 s | 0.188 s | 3.454 s | 157 MiB | 24/24 | 24/24 |
| 2902x4128 | 1 | vapoursynth | 1.449 s | 0.061 s | 1.508 s | 293 MiB | 24/24 | 24/24 |
| 2902x4128 | 1 | vapoursynth-gpu | 1.452 s | 0.095 s | 1.547 s | 716 MiB | 24/24 | 24/24 |
| 2902x4128 | 4 | nmanga | 2.815 s | 0.050 s | 2.865 s | 238 MiB | 24/24 | 24/24 |
| 2902x4128 | 4 | vapoursynth | 1.458 s | 0.018 s | 1.477 s | 306 MiB | 24/24 | 24/24 |
| 2902x4128 | 4 | vapoursynth-gpu | 1.460 s | 0.043 s | 1.503 s | 729 MiB | 24/24 | 24/24 |

| book | threads | pipeline | total speedup vs Pillow |
| --- | ---: | --- | ---: |
| 768x512 | 1 | vapoursynth | 2.24x |
| 768x512 | 1 | vapoursynth-gpu | 1.87x |
| 768x512 | 4 | vapoursynth | 1.79x |
| 768x512 | 4 | vapoursynth-gpu | 1.64x |
| 2902x4128 | 1 | vapoursynth | 2.29x |
| 2902x4128 | 1 | vapoursynth-gpu | 2.23x |
| 2902x4128 | 4 | vapoursynth | 1.94x |
| 2902x4128 | 4 | vapoursynth-gpu | 1.91x |
<!-- book-bench:end -->

## reproduce

```sh
uv sync --extra dev --extra dev-tests --extra golden
.venv/bin/python tools/book_peak_bench.py --gpu --threads 1 --threads 4 \
    --write docs/BOOK_PEAK_BENCH.md
```

raw per-repetition measurements and validation hashes are written to
`target/bench/book-peak.json`. `--size WIDTHxHEIGHT`, `--pages`, `--repeats`,
`--threads`, and `--cache` control the synthetic book and execution. sizes and
worker counts can be repeated. omit `--gpu` on a host without Vulkan.

when the real page set is present, read it in place:

```sh
.venv/bin/python tools/book_peak_bench.py --directory sandbox/level-check \
    --threads 1 --threads 4 --output target/bench/book-peak-real.json
```

the real-input path accepts 8-bit `L` and `RGB` images without palette or alpha
conversion. lossy decoders or YUV-vs-RGB luma can differ, so the tool fails
parity rather than hiding that difference behind a timing row. use lossless
L/RGB pages for an exact-pixel comparison.

`--no-white` repeats the check with white analysis disabled. nonzero offsets
also have an exact-pixel gate: both paths now derive gamma from the raw peak
before shifting the curve endpoint. the offset-1 smoke run passed every decision
and pixel hash on an eight-page 768×512 book, for native CPU and Vulkan. separate
checks cover negative offsets and white detection disabled.

these results replace the earlier raw-histogram/guard-only comparison. the
median now scans every page before peak detection, and both paths implement
the relative tolerance. the native book scan remains serial; Vulkan mapping
adds transfers around that host scan and is slightly slower than CPU here.
