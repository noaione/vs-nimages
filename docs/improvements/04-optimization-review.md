# optimization review

status: initial review on 2026-10-03 at
`eabc817b5fc3693d1ebf5a4ca69167ce8432c806`, with later outcomes recorded below.

the initial review follows that source revision, including native integer depths,
float Levels, property-driven Posterize, and the existing AVX2 Deblur paths. the old
[performance plan](01-performance-plan.md) and [deblur record](03-deblur-filter.md)
retain their measured outcomes. the candidate order below records the original
proposals; the outcome section records implementation and measurement.

## candidates and order

| order | candidate | scope | expected benefit | main risk |
| ---: | --- | --- | --- | --- |
| 1 | [05 horizontal blur dispatch](05-horizontal-blur-dispatch.md) | both Deblur methods, especially method 0 | largest unresolved compute opportunity | an earlier vector attempt regressed inside the plugin |
| 2 | [06 vertical register accumulation](06-vertical-register-accumulation.md) | every Deblur gaussian | fewer output loads/stores and feature dispatches | register pressure and large-kernel cache behavior |
| 3 | [07 Lloyd fixed-point exit](07-lloyd-fixed-point-exit.md) | Posterize method 1 | skip repeated refinement after exact convergence | solver is already a small part of large-frame cost |
| 4 | [08 property-driven posterize tables](08-property-driven-posterize-tables.md) | Posterize use_props, method 0, above 8 bits | eliminate repeated native LUT construction | cache synchronization and retained memory |
| 5 | [09 striped histogram counters](09-striped-histogram-counters.md) | Gray analyzers and Lloyd input | break dependencies on repeated shades | wider histograms increase the working set |
| 6 | [10 fresh Deblur output planes](10-fresh-deblur-output-planes.md) | Deblur write/allocation | avoid copying planes that are then overwritten | chroma sharing, properties, and separate strides |
| 7 | [11 streaming blur scratch](11-streaming-blur-scratch.md) | Deblur memory and concurrent throughput | replace one full scratch plane with a row ring | reflection and producer/consumer scheduling |
| 8 | [12 Deblur conversion dispatch](12-deblur-conversion-dispatch.md) | Deblur luma/write, especially RGB | specialize and vectorize sample conversion | f64 quantization and NaN behavior |
| 9 | [13 peak prominence pruning](13-peak-prominence-pruning.md) | PeakStats with prominence enabled | avoid scans for candidates that cannot win | no gain when prominence is disabled |

05 and 06 attack the same gaussian from different directions. compare each
against the unchanged baseline first, then benchmark the combination separately.
10 and 12 both affect write time; use the same separation there. 11 follows a
working direct blur and is primarily a memory experiment.

## outcome

the implemented candidates were measured against the unchanged baseline, on
one machine with rustc 1.99.0. their records report `cargo test --locked`,
`cargo clippy --all-targets --locked -- -D warnings`, `cargo fmt --check` and
`tests/check-nimages.py` validation for each build. rejected and deferred
candidates retain their measurement or implementation rationale. the twelve case
Deblur hash harness in `.tmpbuild/deblur_output_hash.py` backs the exactness claims
for 05, 06 and 12.

| order | candidate | outcome | evidence |
| ---: | --- | --- | --- |
| 1 | 05 horizontal blur dispatch | kept | bit-identical, 2.2x to 3.0x on the horizontal pass alone, `apply` down 20% to 52% over three rounds |
| 2 | 06 vertical register accumulation | kept, 32 tap threshold | bit-identical, 1.29x to 1.41x on the vertical pass at five to seventeen taps, separable blur 5.6% to 10.2% |
| 3 | 07 Lloyd fixed-point exit | kept | same level bits in 56 cases, 87.7% off the refinement section, no frame-level effect |
| 4 | 08 property-driven posterize tables | rejected | the 16-bit build is 0.197 ms against a 15.44 ms walk, 1.3% |
| 5 | 09 striped histogram counters | kept for 8 bits with bounded probe | 2.11x median, 1.30x worst on 16 real planes for the bounded v2 reader; 256-pair cap and direct fallback below 16384 pixels; 16-bit excluded |
| 6 | 10 fresh Deblur output planes | deferred | `write` is 10.8 to 12.2 ms of a 46 to 156 ms frame, and the copy is part of that |
| 7 | 11 streaming blur scratch | not implemented | 48 MB per concurrent frame at 12 Mpx, pinned and planned in its record |
| 8 | 12 Deblur conversion dispatch | kept | the write stage falls 39% and 58%, 18.59 to 11.39 ms and 21.86 to 9.15 ms over two rounds, at the same bytes |
| 9 | 13 peak prominence pruning | kept | same winner in six workloads, 228,000 to 76,000 prominence scans |

the two combinations the review asked for separately are both in place and were
measured apart first: 05 and 06 each have their own single-change reading, and
the integrated rounds in 05 use the 06 build as the baseline.

the implemented candidates' records carry source revisions, DLL hashes, methods
and decisions. the numbers are one machine with a warm page cache and a run to run
spread wider than several of the effects; the records say so where that is the
case rather than rounding a difference up to a win.

the refreshed full-page run in [BENCH.md](../BENCH.md) records 4.12x for shades,
7.09x for posterize, 12.81x for deconvolution and 19.59x for the unsharp mask
against the reference. these are whole-workflow ratios for the combined build;
the individual records below establish each candidate's isolated effect.

## source at the initial review

these notes describe the initial source revision, before the kept changes above.

- `Workspace::deconvolution` calls `blur` twice per iteration. the default six
  iterations mean twelve candidate blurs, plus two blurs in `edge_mask`.
  `method=1` has one candidate blur plus those same two mask blurs.
- `blur_row` remains scalar. vertical `blur` calls `accumulate_avx2` once per
  tap of every output row, detecting AVX2 inside that loop. `blend` and
  `sobel_mask` already dispatch once per plane. proposing either stencil as a
  new SIMD opportunity would repeat completed work.
- `lloyd_max_levels` already uses prefix sums and bucket ranges. it still runs
  all 40 refinements. rebuilding prefix sums is not the candidate in 07.
- fixed-format, fixed-bit even Posterize already borrows its table. with
  `use_props=True`, `resolved` stays empty and wider tables are built per frame.
  this is a different key space from the rejected Levels property cache.
- histogram counters still have one read/modify/write chain per bin. the older
  half-width counter experiment did not break that chain.
- the Deblur workspace has five full f32 planes and grows without shrinking.
  frame concurrency multiplies that storage. frame-level throughput and one
  frame's latency are different measurements.

## correctness gates

keep argument names, defaults, properties, errors, per-frame format validation,
reflection, and native sample units. keep scalar fallbacks and bounded,
allocation-fallible storage. do not accept a faster candidate by editing golden
expectations or widening tolerances.

for exact candidates, compare the unchanged baseline and candidate directly as
well as replaying fixtures. compare integer pixels byte for byte, properties
including array order and empty arrays, and finite float samples by their bits.
check NaNs, infinities, signed zero, odd widths, padded rows, narrow frames,
variable dimensions/depths, and repeated/out-of-order/concurrent requests.

Deblur's frozen fixture limits differ from ordinary integer LUT parity:

| fixture sample type | maximum absolute difference | mean absolute difference | exact fraction |
| --- | ---: | ---: | ---: |
| u8 | 1 | 0.05 | 0.999 |
| u16 | 257 | 1 | 0.99 |
| f32 | 1 / 255 | 0.00001 | 0 |

the source of truth is each case in `tests/fixtures/deblur.json`, replayed by
`deblur_fixtures_stay_inside_the_frozen_tolerance` and `check_deblur`. the
order-preserving candidates target baseline parity in addition to those limits.
an approximation that passes only a 64x64 fixture is not established for large
pages or 64 iterations. none of these proposals requires an approximate blur,
FMA contraction, fewer requested iterations, or a reduced-precision Levels LUT.

## measurement protocol

1. save a release baseline and candidate as separately identified DLLs. record
   source revision, rustc version, CPU, enabled features, and file hashes.
   load each in a fresh process and verify the plugin path; a pre-existing
   installed wheel is not proof that current source is loaded.
2. warm scratch allocation with a discarded frame, then request distinct frame
   numbers through one node. a cached `get_frame(0)` loop does not time the
   filter. verify one debug frame line per evaluated frame.
3. use at least five fresh-process repeats, alternate baseline/candidate order,
   and report the median and spread. keep geometry, source pixels, cache size,
   core thread count, and request pattern identical.
4. measure targeted stages with `debug=1`, then caller-visible wall time with
   `debug=0`. logging overhead and source generation belong to the latter, so
   report source-only wall time too. preserve existing debug line syntax.
5. run sequential pulls for latency and bounded concurrent distinct-frame pulls
   for throughput. include peak RSS, cold-start allocation, and retained scratch
   after a large frame followed by smaller frames.

`tools/bench.py` covers the existing 8-bit page workflows, not high-depth,
float, Lloyd, or property-driven Posterize comparisons. the scripts
`.tmpbuild/bench_stage.py`, `.tmpbuild/deblur_own_stages.py`,
`.tmpbuild/deblur_parallelism.py`, and `.tmpbuild/deblur_output_hash.py` exist
locally, but are untracked helpers. their measurements are not a reproducible
public harness. promote/adapt the needed synthetic cases to `tools/` as a
separate preparation step. the Deblur stage helper infers blur cost by
subtracting configurations; use a directly timed pure blur to confirm it.

the existing page command uses suite name `posterize`, not its directory name:

```powershell
uv run --extra golden --extra dev-tests tools\bench.py --suite posterize --workflow deblur --workflow deblur-unsharp --limit 8 --keep-logs
```

use the full page set for a finalist. keep private pages in their existing
`sandbox/` directories and report only aggregate results.

## validation and result record

after each implementation:

```powershell
cargo test --locked
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check
cargo build --release --locked
uv sync --locked --extra dev --extra dev-tests
.venv\Scripts\python.exe tests\check-nimages.py
```

verify that the installed plugin matches the release build before timing it.
if generator inputs change, run `tools/golden.py --check` with the golden extra.
these proposals change no generator input and require no fixture regeneration.

record one result per candidate: source revision and DLL hash, cases and repeat
count, stage/wall medians and spread, peak RSS, exact/tolerance comparisons,
integration outcome, and keep/reject decision. retain the old plan's bar of a
repeatable 15% targeted-stage gain and at most 5% throughput regression. for 11,
judge the explicit memory reduction and concurrent throughput too. percentages
saved in a stage do not automatically equal percentages saved end to end:
`total fraction saved = stage share * stage fraction saved`.

review-time checks: `cargo test --locked` passed 86 unit tests and 11 golden
tests. a dependency-free Python mathematical probe compared exact fixed-point
exit with 40 Lloyd passes on 27 seeded synthetic histograms: all level bits
matched, with exit after 1 through 40 passes. this supports 07's stopping rule,
not a Rust speedup claim. the VapourSynth import failed before integration could
run: `_ctypes` reported `DLL initialization routine failed`. no fresh plugin
integration pass or candidate timing is claimed for this review.
