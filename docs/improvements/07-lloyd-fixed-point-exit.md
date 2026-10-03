# Lloyd fixed-point exit

status: proposed. low implementation risk, bounded speed benefit.

## current cost

[`lloyd_max_levels`](../../src/posterize.rs) already constructs prefix sums
once and computes bucket totals in O(colors) per pass. it executes all
`LLOYD_ITERATIONS = 40` passes even if every level is unchanged. both endpoints
are pinned and empty buckets retain their previous level.

## one experiment

while updating interior levels, compare each replacement's `to_bits()` with
the old level's bits. if no level changed after the complete pass, break. keep
the 40-pass maximum and all current prefix accumulation, bucket boundaries,
means, and final rounding. this needs one changed flag and no extra level array.

this stopping rule is exact: identical levels produce identical bucket starts,
and identical starts query the same immutable prefix sums. the next pass then
has the same means and leaves the levels identical again. an epsilon or a
rounded-output comparison does not establish that invariant and is excluded.

## existing evidence

a read-only Python model on 2026-10-03 compared full 40-pass and bit-equality
exit using 256, 1024 and 65536 bins, 4/16/64 colors, and uniform, three-cluster,
and random counts seeded with `20261003`. all 27 final level arrays matched by float bits.
stopping passes ranged from 1 to 40; one case still consumed the full budget.
this is a mathematical probe of the current formulas, not a Rust benchmark or
a replacement for the Python/plugin reference comparison.

## parity checks

compare the original 40-pass solver and candidate on large seeded histogram
sets, empty buckets, pinned-only mass, halfway levels, counts near saturation,
and histograms that converge slowly. compare unrounded level bits and every
resulting LUT entry. retain the full-depth identity bypass in the filter.

the generated golden fixtures cover even posterization; Lloyd also needs its
unit vectors and the numpy reference in `tests/check-nimages.py`. passing only
`posterize_tables_cover_every_depth_and_input` does not validate Lloyd.

## measure and decision

record pass counts and solver-only time for 8/10/12/16 bits with 2, 4, 16, 64,
and high color counts. time integrated method 1 on sparse palettes, uniform
ramps, random noise, and private pages. add pass counts to a benchmark helper
without changing the production log grammar.

if p passes are needed, the refinement portion loses approximately
`(40 - p) / 40` of its work. histogram construction, prefixes, LUT creation,
and pixel mapping remain. reject an integrated regression; do not advertise
the refinement saving as a whole-frame speedup.

## result

status: implemented. kept.

source revision `eabc817b5fc3693d1ebf5a4ca69167ce8432c806`, rustc 1.99.0. the
candidate DLL hashes `5966b3347d781f5e4f158780ec5799b6c94b8c8518b8f3ea2dc791c05760d544`
against the unchanged baseline `27fabb356fcf285339a6868368eb9e0f311ea1f67760445b3d3bc7097ea8c4da`.
`src/posterize.rs` breaks out of the refinement loop when a complete pass
leaves every interior level at the same bits.

`cargo test --locked` passed 87 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks.

a standalone probe at `.tmpbuild/lloydb` transcribes the solver with a pass
counter and a switch that keeps the 40 pass budget, so the two runs differ
only in the stopping rule. on seven synthetic histograms covering flat,
two-shade, ramp, three-cluster, seeded-noise, low-key and two-spike mass, plus
12 `sandbox/posterize-check` pages, at 2 through 256 colors, every final level
matched by float bits and no pass moved a level:

| cases | colors tried | mean passes | stopped early | solver-only saving |
| ---: | --- | ---: | ---: | ---: |
| 56 | 2, 4, 8, 16, 32, 64, 128, 256 | 2.54 | 56 of 56 | 87.7% |

the refinement section is microseconds per frame, so that 87.7% is not a
frame-level win. the integrated reading is `map` at 1.29 ms against 1.32 ms on
2048x2048 `GRAY8` and 2.31 ms against 2.31 ms on `GRAY16`, inside the run to run
spread, and the rest of `method=1` is histogram construction and lookup table
building, which the exit does not touch. keep it as a bounded, exact work
reduction; do not claim a measurable frame speedup.
