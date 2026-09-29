# performance plan

the current implementation prioritizes reference parity, bounded frame-local
work, and safe stride handling. do not optimize by changing histogram binning,
rounding, endpoint behavior, properties, or per-frame independence.

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
