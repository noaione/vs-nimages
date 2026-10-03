# striped histogram counters

status: proposed. content-sensitive experiment, especially flat artwork.

## current cost

[`src/histogram.rs`](../../src/histogram.rs) increments one u64 counter table.
long runs of one shade serialize updates to the same location. shrinking the
counters, as tried in 01, leaves that dependency chain intact.

this affects PeakStats, PeakGrayShades and the plane-0 histogram used by Lloyd
Posterize. an even Posterize or ordinary Levels frame builds no histogram.

## one experiment

for Gray8, compare two and four independent `[u64; 256]` tables. assign
successive active samples to successive tables, then merge bins with saturating
addition. four tables occupy 8 KiB instead of 2 KiB. handle the short tail
without touching row padding.

keep u64 saturating counters, format clamping, total pixel calculation, and
plane-length validation. the pure constructor that accepts arbitrary counts
retains its existing behavior. the merge is deterministic because these are
nonnegative integer counts.

test wider formats separately. four 65536-bin tables require 2 MiB instead of
512 KiB, before other scratch. begin with two fallibly allocated tables and
make selection depend on measured depth/shape behavior. do not assume the Gray8
winner transfers to Gray16 or allocate megabytes on the stack.

## parity checks

compare every histogram bin against the baseline for flat black/white/mid-gray,
alternating pairs, sparse palettes, ramps and random noise. include odd byte
strides, empty geometry, truncated buffers and out-of-range native words.
replay peak and shade properties after changing the reader, not just bin totals.

the sum of saturating partial counts has the same saturated result as direct
nonnegative increments; do not replace it with a wrapping merge or narrow
counter that depends on an undocumented frame-size assumption.

## measure and decision

time histogram-only constructors and integrated analyzers at 8/10/12/16 bits.
include all those distributions: random data rarely exercises the same-bin
chain that flat artwork does. measure the merge overhead on tiny frames and
peak RSS/cache contention under concurrent full-page evaluation.

the older 0.34 ns/sample Gray8 result is a historical workload measurement,
not a lower bound for every distribution. retain stripes only for cases with
repeatable total gains; retain the direct reader for regressing sizes/depths.
this candidate adds no inner-frame threads.

## result

status: implemented for the 8-bit reader, rejected for the wide one. kept.

source revision `41a2bfe`, rustc 1.99.0. the candidate DLL hashes
`3cc955d97e33199878a00dfe710c288f73625ba808428520cdc97a8897fc1534` against the
unchanged baseline `27fabb356fcf285339a6868368eb9e0f311ea1f67760445b3d3bc7097ea8c4da`.
`Histogram::from_plane` now counts into four `[u64; 256]` tables and merges them
with saturating addition; `from_u16_plane` is unchanged.

`cargo test --locked` passed 89 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks. a differential test in
`src/histogram.rs` compares the striped reader against the reader it replaced
over flat, alternating, sparse, ramp and seeded-noise planes at four shapes
with a padding tail, byte for byte.

a standalone probe at `.tmpbuild/histbench` times both readers on one
2903x4128 plane. the direct reader is the baseline in the ratio column:

| shades | direct | 2 stripes | 4 stripes | 8 stripes |
| --- | ---: | ---: | ---: | ---: |
| flat | 1.00x | 1.87x | 2.37x | 2.10x |
| two-tone | 1.00x | 1.34x | 1.17x | 1.31x |
| halftone and mid gray | 1.00x | 1.37x | 1.62x | 1.73x |
| seeded noise | 1.00x | 0.75x | 0.65x | 0.72x |
| 16-bit flat | 1.00x | 1.82x | 0.67x | 0.92x |
| 16-bit noise | 1.00x | 0.63x | 0.36x | 0.44x |

16 raw luma planes dumped from `sandbox/posterize-check` are the workload that
decides it, and four stripes win there on every page:

| planes | regressed | median ratio | best | worst |
| ---: | ---: | ---: | ---: | ---: |
| 16 | 0 of 16 | 1.67x | 1.86x | 1.05x |

two stripes reach a 1.54x median on the same planes but leave one page at
0.99x, so four is the retained count.

the wide reader is excluded on measurement, not on principle: four 512 KiB
tables per 16-bit depth leave flat planes at 0.67x and seeded noise at 0.36x,
and two stripes lose to the direct reader on noise as well. `PeakStats` and
`PeakGrayShades` over `Gray8` through `resize` are the 8-bit callers; the
16-bit analyzer paths keep the direct reader and their measured cost.

integrated, `--suite posterize --workflow shades --limit 16` moves the plugin's
`analyze` stage from 0.25 s to 0.13 s over 16 pages and its total from 1.10 s to
1.04 s. the analyze stage is about half of the plugin's remaining wall time at
this size, so the per-page total gain is smaller than the stage gain. no
workflow regressed.
