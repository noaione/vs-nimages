# peak prominence pruning

status: proposed. narrow workload, exact winner selection.

## current cost

[`select`](../../src/peaks.rs) rejects candidates below the minimum height,
then computes their prominence, then checks whether their height beats the
current best. `prominence` walks both sides until a strictly taller sample or
the padded ROI boundary. equal-height peaks do not stop that walk.

many tied peaks in a wide native ROI can therefore trigger repeated long scans,
even after a qualifying peak has already won their height tie. the default
filter has no prominence option, so this is not a default manga-pipeline win.

## one experiment

after the minimum-height check, skip a candidate whose height is less than or
equal to the best qualifying height already found. then compute prominence
only for candidates that can replace the winner.

`LocalMaxima` visits bins in ascending order and the current winner update uses
strictly greater height. a later candidate with an equal height loses the tie
regardless of prominence, and a shorter one cannot win. this pruning changes
neither the selected bin nor whether any peak qualified.

retain the fallback call with height/prominence disabled. the best state stays
local to each `select` call; do not carry a winner from the filtered pass into
fallback. do not change plateau centers, ROI padding or inclusive thresholds.

## parity checks

compare baseline/candidate winners and found flags over all peak fixtures and
seeded native-depth histograms. include equal-height plateaus, descending-height
peaks, increasing record-height peaks, no prominence qualifier, fallback-only
winners and large counts. test both white checking modes and every upper-limit
edge used by the existing tests.

adversarial examples need both outcomes: when the first tied peak qualifies,
later equal peaks skip their scans; when none qualifies, no best winner exists
and those scans still occur. this does not establish a general O(n) prominence
algorithm. a precomputed range-minimum strategy is a separate larger change.

## measure and decision

time peak selection directly, apart from histogram construction, at 8/10/16-bit
ranges. use prominence enabled/disabled, repeated equal peaks, dense noise,
and monotonically increasing peak heights as controls. count prominence calls
in the benchmark helper and measure integrated PeakStats with the same inputs.

expect the largest gain for native-depth histograms with many non-winning peaks
and an early qualifier. keep only if that workload matters and exact reference
checks pass without an integrated regression. no new peak cache or public
argument is needed.

## result

status: implemented. kept on the scan count, not on an integrated gain.

source revision `c167e48`, rustc 1.99.0. the candidate DLL hashes
`f64ce4ecb6e16b361ad27ba3aeb6b0040d51e3df7c26047bc250993d6790cac6`, built from
the revision whose integration run is recorded below.

`cargo test --locked` passed 89 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks.

a probe at `.tmpbuild/peakbench` repeats `select` over the padded region with a
prominence-scan counter and a switch that keeps the scan for every candidate, so
the two readings differ only in the rule under test. the winner matched in all
six workloads:

| workload | scans pruned | scans unpruned | ratio |
| --- | ---: | ---: | ---: |
| equal plateau under one qualifier | 1 | 30 | 30.0x |
| descending record heights | 1 | 30 | 30.0x |
| dense noise ROI | 3 | 21 | 7.0x |
| ascending record heights | 31 | 31 | 1.0x |
| single qualifying peak | 1 | 1 | 1.0x |
| flat then tied | 1 | 1 | 1.0x |
| **total** | **76,000** | **228,000** | **3.0x** |

the winner search itself takes 2.99 ms against 5.99 ms for 12,000 calls, a 50%
cut. `find_local_peak` is about 0.3 us a call over these regions, which is
nothing against the 10 to 15 ms a 12 megapixel histogram costs, so the
integrated `levels` workflow was unchanged at 32 of 32 pages agreeing and
`analyze` at 0.06 s over 32 pages. keep it as a free exact reduction on the
prominence path; no frame-level speedup is claimed.
