# performance plan

the current implementation prioritizes reference parity, bounded frame-local
work, and safe stride handling. do not optimize by changing histogram binning,
rounding, endpoint behavior, properties, or per-frame independence.

## candidate status

checked against the code as it stands:

| # | candidate | state |
| ---: | --- | --- |
| 1 | cache repeated `Levels(use_props=True)` tables | **rejected.** `curve` is 0.14 ms per 16 bit frame and 0.00 at 8 bit, so a total hit rate saves under 6% of the frame, and a volume repeats no tuple |
| 2 | typed `u16` plane walk | **rejected, measured below.** the unreachable error path was the only difference, and removing it is -4% |
| 3 | integer-count shade ordering | **done.** `analyze_gray_shades` sorts by `Reverse(count)` with a stable sort, so ties keep ascending shade order and no percentage is compared |
| 4 | Gray16 histogram accumulation | **rejected, measured below.** half-width counters change nothing, so the scatter's latency is the cost rather than the table size |
| 5 | float `pow` cost | **done** for the per-sample curve, measured below |
| 6 | SIMD for the 8-bit LUT | **rejected.** the 8 bit `map` is 0.34 ns per sample, about one cycle, so it already sits at the load and store issue limit |
| 7 | `Posterize(method=1)` refinement passes | **done.** a bucket is one range of code values, so prefix sums make a pass cost `colors` steps instead of one per code value |

every candidate now has an outcome: done, or measured and rejected. `docs/BENCH.md`
still measures `GRAY8` only, and its reference side is 8-bit Pillow, so the
high-depth and float formats this document asks for cannot be measured by
`tools/bench.py` at all. the numbers below come from a stage harness under
`.tmpbuild/` that drives one filter directly and reads its own `debug=1` stage
timings.

### rejected candidates, measured

both rejections come from the same harness on a 2048x2048 clip, 9 frames and four
repeats, against the 15% bar in the adoption criteria:

| candidate | case | stage | before | after | change |
| --- | --- | --- | ---: | ---: | ---: |
| 2 | `levels8` | `map` | 1.42 ms | 1.36 ms | -4% |
| 2 | `levels16` | `map` | 2.41 ms | 2.31 ms | -4% |
| 2 | `levels48` | `map` | 8.52 ms | 8.29 ms | -3% |
| 4 | `peakstats16` | `histogram` | 3.63 ms | 3.65 ms | 0% |
| 4 | `shades16` | `histogram` | 3.70 ms | 3.70 ms | 0% |

the only difference a typed walk can make on this path is dropping the
unreachable error path, and the branch predictor had already removed it. the
half-width counters cost nothing because the histogram is scatter bound: the bin
table is a chain of dependent loads, so halving its width halves traffic that was
never the limit. `GRAY8` already runs that pass at 0.34 ns per sample, which is
about one cycle, so nothing there is left either.

### candidate 5, measured

`apply_float_level` evaluated `powf(1.0 / gamma)` for every sample, and `gamma`
defaults to `1.0`, so the default float `Levels` spent a libm call per sample on
the identity. `x ** 1.0` is `x`, so the curve skips the call when `gamma == 1.0`.

the `map` stage from `debug=1`, one 2048x2048 clip per format of 15 identical
frames, no `gamma` argument. five `GRAYS` runs and four `RGBS` runs before the
change, three of each after, and the table holds the median of the per-run
medians:

| format | samples per frame | before | after | change |
| --- | ---: | ---: | ---: | ---: |
| `GRAYS` | 4.19 M | 10.94 ms | 5.84 ms | -47% |
| `RGBS` | 12.58 M | 33.01 ms | 20.39 ms | -38% |

the median is the stable statistic here, because one run in the set moved by
30%. the fastest run moves the same way: 9.51 ms to 5.19 ms for `GRAYS` and
28.58 ms to 15.80 ms for `RGBS`.

output is bit-identical rather than merely close. the guard's unit test passes
against the `powf` form as well, which is the direct evidence that `powf(x, 1.0)`
returns `x` for every finite `x`, and `tests/check-nimages.py` passes unchanged
at 1950 checks.

the same shortcut applies to `levels_lut_u16`, which rebuilt up to 65,536 entries
per frame under `use_props=True`. the `curve` stage from `debug=1` on
`levels16-props`, 9 frames and five repeats:

| stage | before | after | change |
| --- | ---: | ---: | ---: |
| `curve` | 0.30 ms | 0.14 ms | -53% |

`levels16-props-gamma` is the control and stays at 1.11 ms in both, because the
branch only fires for the identity.

the harness is a throwaway script under `.tmpbuild/`, which is not committed, so
these numbers are not reproducible from the tree yet. promoting it to `tools/`
is the way to make them so.

### the Lloyd solver, measured

`lloyd_max_levels` walked every code value on each of its 40 passes, which is
2.6 M steps per frame at 16 bit. a bucket is one range of code values, so the
weight and total of every bucket now come from two prefix sums and a pass costs
`colors` steps. the table builders use the same ranges instead of a binary search
per code value.

the frame total from `debug=1`, 2048x2048, 7 frames and three repeats:

| case | before | after | change |
| --- | ---: | ---: | ---: |
| `posterize-lloyd16` | 12.77 ms | 6.72 ms | -47% |
| `posterize-lloyd8` | 3.09 ms | 3.19 ms | unchanged |

the 8 bit path is unchanged because its solver was already 40 steps over 256 code
values. the output is identical: the pinned vectors in the unit tests and the
1950 checks in `tests/check-nimages.py` pass against the reference unchanged.

## establish a baseline

measure the current release build before changing hot loops. record compiler
and CPU, format and sample depth, frame dimensions, row strides, frame-cache
size, warm-up policy, and the number of repeated runs. include odd widths and
subsampled planes so the measurements cover the production row paths.

- measure `PeakStats` and `PeakGrayShades` separately on Gray8 and Gray16.
- measure `Levels` and `Posterize` separately on Gray8, RGB48, and YUV420P16.
- measure `Levels` on GRAYS and RGBS with both identity-like and nonlinear
  curves.
- use `debug` stage timings to locate time in histogram, peaks, curve
  construction, and plane mapping. also record end-to-end frames per second,
  since VapourSynth scheduling and frame copies affect the caller-visible cost.
- keep `tools/bench.py` for the existing 8-bit reference comparison. add a
  deterministic native microbenchmark only if the current tools cannot isolate
  a stage or format.
- run private page benchmarks locally and report only aggregate measurements.

## candidate improvements

implement one candidate at a time, only when its stage is a measured bottleneck.

1. reuse a bounded set of `Levels(use_props=True)` tables when frames repeat
   the same parameter tuple. compare cache hit rate and lock cost against
   rebuilding the table. keep cache memory bounded and preserve deterministic
   behavior under concurrent frame requests.
2. measure a typed `u16` plane walk against the current byte-chunk conversion.
   use it only when pointer alignment and byte stride satisfy the typed path.
   retain the safe byte path for every other case.
3. measure `PeakGrayShades` enumeration and sorting on Gray16. if sorting
   dominates, compare a stable integer-count ordering strategy that keeps
   ascending code values for ties and avoids sorting by computed percentages.
4. measure Gray16 histogram accumulation strategies. a 65,536-bin table costs
   more cache traffic than the Gray8 table, so compare direct accumulation with
   bounded block-local accumulation before considering parallel reduction.
5. measure float `pow` cost separately from plane reads and writes. any vector
   or approximation path must stay within a documented float error bound and
   preserve endpoint, NaN, infinity, and out-of-range behavior.
6. keep the Gray8 LUT mapping path as the control. add SIMD only if the measured
   mapping stage gains enough to outweigh dispatch, alignment, and tail handling.

## adoption criteria

- require a repeatable improvement of at least 15% in the targeted stage and no
  more than 5% regression in total clip throughput on the same machine.
- require exact output equality for integer paths, including every 8-bit LUT
  entry, all native endpoints, and padded-row sentinels.
- for float paths, require at most one ULP of error from the scalar output for
  finite samples, plus unchanged handling of endpoints, NaNs, infinities, and
  samples outside the endpoints.
- record before/after throughput, peak memory, hit rates where relevant, and the
  correctness checks used. keep an optimization only when the results justify
  its maintenance cost.

## performance expectations

these filters are primarily memory-bandwidth-bound:

- histogram: one input read per pixel plus 256 bins for Gray8 or up to 65,536
  bins for Gray16.
- gray-shade analysis: the same histogram pass plus sorting at most the number
  of sample values in the format.
- integer `Levels`/`Posterize`: one input read, one LUT lookup, and one output write.
- float `Levels`: one input read, one curve evaluation, and one output write.

parallelize across frames through VapourSynth. avoid internal per-frame
threading initially; manga image sequences naturally supply many independent
frames. SIMD is unlikely to materially improve 8-bit LUT application before
memory bandwidth becomes the limit and should only be added after benchmarks.
