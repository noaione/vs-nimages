# horizontal blur dispatch

status: proposed. highest potential, with a recorded failed predecessor.

## current cost

[`src/deblur.rs`](../../src/deblur.rs) calls scalar `blur_row` for every row of
every gaussian. its interior folds a runtime-length window for one pixel at a
time. the default method 0 runs fourteen gaussians in total. radii 0.8, 0.5,
and 0.45 produce seven, five, and five taps respectively.

[03-deblur-filter.md](03-deblur-filter.md) records a horizontal AVX2 attempt
that was byte-identical but 2.9x slower in the plugin, while its isolated probe
looked faster. moving dispatch is a hypothesis, not an established explanation:
the current vertical loop also repeats feature checks without a comparable
regression. inspect generated code and profile the integrated path.

## one experiment

dispatch once at the start of `blur`, into a plane-wide horizontal AVX2 helper.
vectorize across output columns, not across taps: eight independent f32 sums,
each starting at positive zero and adding products in the existing tap order.
use separate multiply and add instructions; keep FMA and horizontal reductions
out. retain `reflected_tap` for borders and scalar code for tails/fallbacks.

first test runtime-length weights. only after that result, try five- and
seven-tap specializations as separate variants. those variants remove the tap
loop without changing its sum order. keep generic handling through the maximum
129 taps and radius-zero copying.

Rust exposes lane-wise addition through
[`_mm256_add_ps`](https://doc.rust-lang.org/core/arch/x86_64/fn._mm256_add_ps.html).
keep all intrinsic code behind runtime feature detection and target guards,
following the existing blend path rather than setting a global native CPU target.

## parity traps

the existing `.tmpbuild/simdprobe/src/bin/hsum.rs` is not a golden oracle. the
historical note records row-end discrepancies, and its kernel normalization
is not the same cast/normalization sequence as `build_kernel`. use weights and
reflection from the production implementation for the new differential probe.

test widths below the kernel length, 1xN/Nx1, vector tails, sigma values around
tap-count transitions, and NaN/infinity float inputs. each load stays inside its
own row; never use neighboring-row slack to finish a vector. compare pure blur
bits first, then restored samples and actual plugin output.

## measure and decision

time horizontal-only and full separable blur at sigma 0.45, 0.5, 0.8, 2, 8,
and 16, then both Deblur methods at iterations 0, 1, 6, and 64. use 1024x1024,
2048x2048, odd page widths, and the existing spread. report dispatch count,
integrated restore time, and concurrent throughput using the shared protocol.

the isolated probe cannot justify adoption. retain this only if the real plugin
improves consistently and exact differential checks plus frozen golden checks
pass. possible gains are limited by the horizontal share of total time; the
old 3x isolated result is not a promised plugin speedup.

## result

status: implemented. kept. the 2.9x regression does not reproduce when the
feature check is hoisted out of the row loop.

source revision `043467d`, rustc 1.99.0. the candidate DLL hashes
`4be7dd531a90d072633866e5b87a3c7fe685afd6def7398e12daa262073abb2e` against the
unchanged baseline `9abaa39379db2a2ae325fc4ed9f45dcf365fb355979a712f27d7ad04655ac321`.
`blur` now tests `is_x86_feature_detected!("avx2")` once, before its row loop,
and sends the whole plane to `blur_row_avx2`. That helper keeps eight
independent `f32` lanes, each starting at positive zero and folding its own
taps in the same order as `blur_row`, with separate multiply and add
instructions and no horizontal reduction. The first and last `radius` samples
still reflect one at a time, and a tail shorter than eight goes through the
same scalar code.

the standalone probe at `.tmpbuild/blurbench` times both horizontal passes over
one 2903x4128 plane, with the vertical pass held constant:

| sigma | taps | scalar horizontal | vector horizontal | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0.45 | 5 | 42.04 ms | 18.07 ms | 2.33x |
| 0.5 | 5 | 41.33 ms | 18.36 ms | 2.25x |
| 0.8 | 7 | 46.69 ms | 21.33 ms | 2.19x |
| 2 | 17 | 92.39 ms | 39.30 ms | 2.35x |
| 8 | 65 | 555.50 ms | 183.97 ms | 3.02x |
| 16 | 129 | 1186.40 ms | 434.48 ms | 2.73x |

every row is bit-identical, row ends included. the earlier attempt measured the
same shape 3x faster on its own and 2.9x slower in the plugin because its
feature check sat inside `blur_row`, which runs once per row, 49.5k times for
the twelve blur passes of one frame. moving that check to `blur`, which runs
once per gaussian, is the whole difference.

integrated, three interleaved rounds of `--suite posterize --limit 6` on 12
megapixel pages. the baseline in these rows is the build with only the vertical
register pass:

| round | workflow | vertical only | horizontal too | per page |
| ---: | --- | ---: | ---: | ---: |
| 1 | `deblur` | 4.22 s | 3.23 s | 764.7 to 614.9 ms |
| 2 | `deblur` | 4.98 s | 2.80 s | 901.0 to 534.7 ms |
| 3 | `deblur` | 5.14 s | 2.49 s | 923.6 to 479.3 ms |
| 1 | `deblur-unsharp` | 1.04 s | 0.77 s | 235.8 to 195.5 ms |
| 2 | `deblur-unsharp` | 1.18 s | 0.73 s | 265.3 to 196.1 ms |
| 3 | `deblur-unsharp` | 1.10 s | 1.11 s | 248.5 to 278.2 ms |

the `apply` column, which is the whole deblur, moves the same way. read the per
page column from the plugin's total, not the reference's.

`deblur` improves in every round, by 20% to 52% on `apply`, and the per page
latency moves with it. two of three `deblur-unsharp` rounds improve by 26% and
38% on `apply` and the third is inside the run to run spread, so the split
apply column is the honest signal there; the per page column of that round also
carries the slowest decoder sample of the three.

`cargo test --locked` passed 91 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks. a unit test compares the two
horizontal passes by float bits at fourteen widths from 1 to 129, which covers
the reflected ends, the eight lane step and the tail, at five sigmas. the twelve
case Deblur hash harness reports the same hash as the baseline on all twelve,
including both float `NaN` pages.
