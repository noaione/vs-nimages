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

status: implemented for the 8-bit reader behind a per-plane probe, rejected for
the wide one. kept.

source revision `41a2bfe`, rustc 1.99.0. the candidate DLL hashes
`3cc955d97e33199878a00dfe710c288f73625ba808428520cdc97a8897fc1534` against the
unchanged baseline `27fabb356fcf285339a6868368eb9e0f311ea1f67760445b3d3bc7097ea8c4da`.
`Histogram::from_plane` counts into four `[u64; 256]` tables and merges them with
saturating addition; `from_u16_plane` is unchanged.

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

## the noise regression, and the plane probe that removes it

the table above records seeded noise at 0.65x, and that is a real defect: an
8-bit caller with grain, a dithered gradient or any noise-like plane paid 54%
more histogram time for a win it could not use. repeated shades serialize
updates to one counter; adjacent repeats are a cheap indicator of that cost.

commit `eba8e67` samples four evenly spaced rows, counts how many neighbour
pairs hold the same shade, and picks the reader from that. it scans every
active byte in those rows: four rows of a 2903 pixel page are eleven thousand
byte comparisons against twelve million counted. the cost grows with width,
and a plane with four rows or fewer is scanned twice. the threshold is 5%,
and the recorded full-row measurements have a wide gap:

| plane | neighbour pairs that repeat | reader chosen |
| --- | ---: | --- |
| flat | 100% | striped |
| two shades | 87.5% | striped |
| `sandbox/posterize-check`, 8 pages | 48.8% to 78.3% | striped |
| scattered three-tone artwork | 41.7% | striped |
| uniform noise | 0.35% | direct |

uniform noise sits 14x below the threshold and the weakest measured real page
sits 10x above it. this is a heuristic over sampled content. the measured gap
does not guarantee separation on other distributions, and sampled rows can
misrepresent a mixed-content plane.

at `eba8e67`, `.tmpbuild/histbench` times the adaptive reader against the reader
from before striping existed, on one 2903x4128 plane and then on 16 real page
planes:

| workload | before striping | adaptive | ratio |
| --- | ---: | ---: | ---: |
| flat | 23.93 ms | 8.04 ms | 2.97x |
| two shades | 12.03 ms | 7.53 ms | 1.60x |
| scattered three-tone | 15.43 ms | 7.80 ms | 1.98x |
| uniform noise | 4.99 ms | 4.89 ms | 1.02x |
| 16 page planes | | | 2.15x median, 1.32x worst, 0 of 16 regressed |

uniform noise goes from 0.65x to 1.02x, which is parity, and every page improves
on both the always-striped build and the pre-striping reader. every reading is
byte-identical: the probe and the plain reader report the same counts on all 16
planes and on every differential case in `src/histogram.rs`.

the wide reader is excluded on measurement, not on principle: four 512 KiB
tables per 16-bit depth leave flat planes at 0.67x and seeded noise at 0.36x,
and two stripes lose to the direct reader on noise as well. `PeakStats` and
`PeakGrayShades` over `Gray8` through `resize` are the 8-bit callers; the
16-bit analyzer paths keep the direct reader and their measured cost.

integrated, `--suite posterize --workflow shades --limit 16` moves the plugin's
`analyze` stage from 0.25 s to 0.14 s over 16 pages and its total from 1.10 s to
1.16 s, with the reference side moving further in the same run. this records
an analyze-stage gain and a slower total; a repeated controlled comparison is
needed to attribute the total change.

## bound the probe on short planes

the full-row probe adds another pass over short planes. on seeded 65536x1
noise, review measured 45.4 us against 26.7 us for the reader before striping,
about 70% more time with the v3 build.

the probe now compares at most 256 adjacent pairs: four windows of 16 pairs
in each of at most four rows. windows span each sampled row, skip padding,
and have a cost independent of width and height. planes below 16384 active
pixels use the direct reader without probing. at that boundary the probe
compares at most one pair per 64 pixels. the 5% repeat threshold is unchanged.

the size gate trades away striped gains on small flat planes to avoid probe
and merge overhead on small noisy planes. wide flat rows above the gate keep
striping. the bounded sample remains a heuristic, with identical counts from
either reader.

the standalone comparison in `target/review-histogram-bounded.rs` compiles the
production reader alongside `eba8e67` and the source before `41a2bfe`. release
settings use opt-level 3, thin lto, one codegen unit and the two wheel cpu
targets. synthetic planes have five bytes of row padding. times below are
medians of 11 batches with reader order rotated and cpu affinity fixed:

| workload | cpu | before striping | full-row probe | bounded probe |
| --- | --- | ---: | ---: | ---: |
| 65536x1 noise | v2 | 26.57 us | 46.43 us | 26.84 us |
| 65536x1 noise | v3 | 23.29 us | 32.28 us | 24.91 us |
| 256x4 noise | v2 | 0.532 us | 1.124 us | 0.532 us |
| 256x4 noise | v3 | 0.444 us | 0.602 us | 0.435 us |
| 65536x1 flat | v2 | 135.34 us | 65.25 us | 45.06 us |
| 65536x1 flat | v3 | 135.20 us | 49.23 us | 40.04 us |

the bounded reader removes the second full scan. remaining differences on
noise range from 1% to 7% on the wide row in these runs; timings vary, so this
does not claim parity for every input or cpu. all synthetic bins and totals
match both prior readers. all 16 existing private page planes also match, and
their v2 median speedup over the pre-striping reader is 2.11x, with 1.30x worst.

`cargo test --locked` passes 95 unit tests and 11 golden tests, including the
size boundary, wide single rows and padded rows through both readers. strict
clippy, formatting and the release build pass. during that review, the integrated
benchmark failed to initialize Pillow's `_imaging` DLL before timing began.
`tests/check-nimages.py` also stopped before running checks because the `_ctypes`
DLL failed to initialize.

a subsequent full-suite run is recorded in [BENCH.md](../BENCH.md). over all 49
posterize pages, `shades` analysis takes 0.38 s against the reference's 8.96 s,
with totals of 2.60 s and 10.70 s, a 4.12x workflow speedup. the reported shade
counts agree on 49/49 pages. this measures the combined pipeline; the isolated
reader timings above measure the bounded probe. the benchmark compares shade
counts, while the golden tests and integration validator compare properties.
