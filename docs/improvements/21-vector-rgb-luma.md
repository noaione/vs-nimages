# a vector luma for the RGB readers

status: landed. the RGB readers compute luma through `std::arch` kernels with a
scalar fallback, and the 8-bit one picks a four-table reader from a luma probe.
touches [20](20-analyzer-color-inputs.md)'s reader and nothing public.

## current state

`Histogram::from_rgb_planes` walks the three rows and calls `luma_u8` per sample
into one `[u64; 256]` table. `Histogram::from_rgb_u16_planes` does the same at
native word depth. 12 MP `RGB24` (4096x3000, three 12 MB planes), median of
seven runs, each part timed on its own:

| page | luma alone | scatter, one table | together |
| --- | ---: | ---: | ---: |
| flat | 10.7 ms | 7.5 ms, striped | 21.1 ms |
| noise | 10.4 ms | 4.4 ms | 12.0 ms |

on a noise page the scatter nearly disappears behind the arithmetic, so the
luma is the whole cost. on a flat page the arithmetic and a single counter
serialise into 21 ms, and the same walk through the gray reader, which picks a
four-table reader from the plane's own samples
([09](09-striped-histogram-counters.md)), is 7.5 ms of scatter rather than 16.

two facts set the ceiling. the expression is three multiply-adds and a shift,
and scalar code needs 2.5 cycles per sample for it. the three planes are 36 MB
per frame, which this machine reads in about 6 ms, so the arithmetic is
above the memory floor but the two are the same order.

## what moves

| file | role |
| --- | --- |
| `src/luma.rs` | new: the scalar expression, the vector kernels, the dispatch, the tests |
| `src/histogram.rs` | the RGB readers compute a chunk of luma through the kernel, then count it, and choose a striped table from a luma probe |
| `src/lib.rs` | the module |
| `AGENTS.md` | the source layout entry |
| `docs/BENCH.md`, [01](01-performance-plan.md), [20](20-analyzer-color-inputs.md) | the new host numbers |

the histogram layout, the properties, the device kernel and every filter
argument stay where they are. the counts are unchanged by construction: the
kernels evaluate the same integer expression the scalar code does.

## decisions

**the formula stays the integer one.** `(r * 19595 + g * 38470 + b * 7471 +
32768) >> 16`, as [20](20-analyzer-color-inputs.md) measured against
`Image.convert("L")`. a vector kernel is only worth having if it computes the
same bytes, so there is one definition and the kernels are tested against it.

**`std::arch` with runtime detection, no crate.** the workspace has one
dependency, the binding. `is_x86_feature_detected!` costs one cached load when
the feature is not implied by the build's target cpu, and compiles away when it
is. every non-`x86_64` target, and an `x86_64` one without the feature, keeps
the scalar loop, so the crate still builds and still counts correctly anywhere.

**the kernel writes a chunk, the counters stay scalar.** a histogram scatter has
no vector form on the baseline target (`avx2` scatter is `avx512`), so the
kernel computes 1024 samples into a stack buffer and the existing increment loop
reads them. the buffer is in the first level cache, so the two halves of the row
interleave rather than splitting the plane into two passes over memory.

**striping applies to RGB through a luma probe.** the gray reader samples 256
neighbour pairs; the RGB one samples the same grid, computes their luma and
compares that. flat artwork then reaches the four-table reader, and noise keeps
the direct one, which is the same decision the gray reader makes.

**the 16-bit reader stays direct.** four 512 KiB tables per depth exceed the
cache the striping depends on, which is why the gray 16-bit reader never
stripes. it still gets the vector luma.

## exactness

- the kernels evaluate the reference expression in exact integer arithmetic: the
  8-bit sum is at most `255 * 65536 = 16 711 680`, and the 16-bit one is at most
  `65535 * 65536 + 32768 = 4 294 934 528`, both inside `u32`, so the vector and
  scalar forms cannot diverge.
- `src/luma.rs` tests every length from 0 to 129 against the scalar expression,
  which covers a whole number of 16-sample and 8-sample bodies plus every tail.
  a second case builds rows where each lane carries a different value, so a
  lane-ordering mistake in the pack cannot pass.
- the existing gates are the behaviour gate:
  `tools/histogram_hash.py --device` compares the host path against the device
  kernel on 50 cases, `--check` compares the host path against the recorded
  table, `tools/map_hash.py` records the `Posterize(method=1)` bytes, and
  `tests/check-nimages.py` carries the RGB-equals-gray view.

## expected

the arithmetic drops by the vector width, so an 8-bit noise page should approach
what the three plane reads cost, and the flat page should approach its 7.5 ms of
striped scatter. the 16-bit reader is arithmetic bound in the same way, and its
scatter is a 1024-bin table read back cold in the harness, so the same
subtraction moves a smaller share. the outcome corrects the first of those
estimates: the reads cost more than this section assumed.

the numbers that have to stay honest are the ones in
[BENCH.md](../BENCH.md#the-histogram-analyzers) and
[20](20-analyzer-color-inputs.md): a host RGB count that no longer costs three
times the gray one changes the chain ratios those tables quote.

## risks

- **`unsafe` in an algorithm module.** `src/luma.rs` is the first module outside
  the filter and device layers to use it. the kernels index through raw pointers
  by a length the caller clamps to all four slices, the target features are
  checked at runtime, and the scalar path stays reachable, so a mistake is a
  test failure rather than a crash. `#![deny(unsafe_op_in_unsafe_fn)]`-style
  discipline applies: every block carries its own SAFETY comment.
- **the AVX2 build.** the crate ships a `-C target-cpu=x86-64-v3` variant, but
  the kernel is written in 128-bit intrinsics. a wider kernel is a follow-up if
  the 8-bit path stays above the memory floor; it is not part of this plan.
- **the flat page is still scatter bound.** a vector luma cannot help a single
  counter that every sample increments. the recorded flat number will improve by
  the arithmetic alone, and the honest way to read it is against the 7.5 ms that
  the scatter costs on its own.
- **measurement spread.** this laptop's throughput moves 25% to 50% between
  sessions. the plan's numbers are one run's medians and the comparisons that
  matter are within the same run.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | `src/luma.rs`, the scalar expression and the two kernels | done: lengths 0 to 129 and both lane orders agree with the scalar rows |
| M2 | the RGB readers through a chunk buffer | done: 113 unit tests and the validator at 2157 checks |
| M3 | the striped RGB reader from a luma probe | done: a flat page takes four tables, noise takes one |
| M4 | measurement and the docs above | done: the harness numbers and the recorded hashes |

## outcome so far

written 2026-10-25. `src/luma.rs` holds the scalar expression, a 16-sample
`sse2` kernel for 8 bit and an 8-sample `sse4.1` kernel for native words, with
the scalar loop as the fallback on every other target. the 8-bit RGB reader
computes a chunk of luma and counts it, and chooses one table or four from a
probe over sampled luma pairs.

the reader before and after, in one process, 12 MP, best of five:

| page | before | after | change |
| --- | ---: | ---: | ---: |
| flat `RGB24` | 20.6 to 21.2 ms | 11.3 to 11.9 ms | 1.8x |
| noise `RGB24` | 13.8 to 15.3 ms | 7.5 to 8.3 ms | 1.9x |
| `RGB48` over 10-bit words | 14.4 to 15.9 ms | 14.8 to 16.9 ms | unchanged |

the kernels against the memory floor, same page:

| depth | three-plane read | vector rows |
| --- | ---: | ---: |
| 8 bit | 6.2 to 6.6 ms for 37 MB | 4.5 to 5.2 ms |
| 16 bit | 13.8 ms for 74 MB | 8.2 ms |

the noise page landed at 7.5 to 8.3 ms rather than the 3 ms this section
assumed: the three reads are 6.2 ms of the total on their own, so the arithmetic
is hidden behind them rather than removed. nothing below that floor is reachable
without reading fewer bytes.

the 16-bit reader is the negative worth recording. three planes of native words
are 74 MB, the machine streams them at 4.8 GB/s, and the scatter and the luma
both fit behind that read: the kernel beats a per-sample scalar row loop 3.4x and
reader does not move. it keeps the kernel anyway, so the definition and the code
path are one, and the cost is under the spread.

`tools/histogram_hash.py --bench`, 12 MP flat, median of 20, two runs:

| row | 20 | 21 |
| --- | ---: | ---: |
| host `RGB24` counting | 23.1 to 24.5 ms | 10.4 to 14.1 ms |
| `PeakGrayShades` into `Posterize(use_props, method=1)`, host | 70.8 to 71.0 ms | 40.2 to 43.1 ms |
| the same, device | 25.7 to 26.3 ms | 21.6 to 27.8 ms |
| ratio | 2.7x | 1.6x to 1.9x |

the plan expected this. the device path was winning against a host
count that cost three times the gray one, and that count now costs 1.1 to 1.5
times the gray one, so most of the margin is gone. `GRAY8`, `GRAY16` and
`YUV420P8` are unchanged in kind: 9.5, 23.4 and 9.6 ms of host counting, and
3.3x, 3.0x and 2.7x on the `Posterize` chain, in the same session.

### what the first cut got wrong

the first version passed `striped: bool` into the reader and masked the counter
table with it. `tables.get_mut(offset & mask)` cannot be proved in range, so
every sample paid a bounds check and a branch: flat `RGB24` reached 14.8 ms
instead of 11.3, and noise `RGB24` 11.8 instead of 8.1, which looked like no
win at all. making the stripe count a const generic, `rgb_counts::<4>` or
`rgb_counts::<1>`, folds the mask to a constant and removes the check. that
was worth more than the kernel.

the chunk length was measured at 512, 1024 and 4096 samples with no difference
outside the run-to-run spread, so the reader keeps 1024.

the gates: `tools/histogram_hash.py --device` identical on 50 cases and the six
views, `--check` unchanged on all 50, `tools/map_hash.py --device` byte-identical
on 34 cases and `--check` unchanged, and `tests/check-nimages.py` at 2157
checks. no recorded hash moved, because the kernels evaluate the same integer
expression the scalar rows do.

## done when

an RGB page counts through a vector luma on both integer depths, the scalar
expression is still the definition, every other target still builds and counts,
and the recorded host numbers in [BENCH.md](../BENCH.md) and
[20](20-analyzer-color-inputs.md) are the new ones.
