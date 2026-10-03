# vertical register accumulation

status: proposed. exact arithmetic-order experiment.

## current cost

in [`src/deblur.rs`](../../src/deblur.rs), vertical `blur` zeroes an output row,
then calls `accumulate_avx2` for each tap. each call loads and stores the partial
output sum. AVX2 detection also occurs for every row/tap pair.

for a seven-tap kernel, each eight-pixel output block has seven accumulator
loads and seven stores, plus its initial zero fill. those are largely cache
operations; counting them does not establish a DRAM bottleneck.

## one experiment

dispatch once per blur and invert the vertical loop nesting. for each output
row, resolve reflected source rows once, then for each eight-column block keep
an accumulator in a register while visiting all taps. store the result once.
use the same zero initial value, weight sequence, and separate mul/add as now.

bind a fixed-size array of row references or offsets with at most 129 entries;
only the active taps are read. validate spans before the vector loop. avoid
constructing a heap vector per row. use unaligned loads and safe scalar tails.
keep the existing generic/scalar route as a differential control.

start with one vector accumulator. two independent output blocks can hide
dependency latency, but benchmark that unroll separately because it increases
register pressure. do not combine this experiment with horizontal SIMD first.

## parity traps

resolve top and bottom reflection exactly as `reflect` does, including heights
smaller than the kernel. preserve horizontal-then-vertical pass order and tap
order; pairing symmetric taps or FMA changes rounding. test signed zero,
subnormals, NaNs and infinities as well as ordinary finite pixels.

do not assume an arbitrary source/target length from plane geometry. retain the
shape checks and the source/target/temp non-aliasing requirement.

## measure and decision

time vertical-only, full gaussian, and both integrated methods. cover sigma
0.45 through 16, short and tall planes, odd widths, and both single-frame and
concurrent workloads. for large kernels, visiting many tap rows for each column
block can change cache behavior; a win at seven taps does not establish a win
at 129.

the operation count reduces accumulator traffic from O(taps) to one final
store per output block. the arithmetic count stays the same. adopt on measured
stage/wall gains and exact output checks, not that traffic count alone. if
large kernels regress, use a measured, documented dispatch threshold.

## result

status: implemented with a measured dispatch threshold. kept.

source revision `fd1c098`, rustc 1.99.0. the candidate DLL hashes
`9abaa39379db2a2ae325fc4ed9f45dcf365fb355979a712f27d7ad04655ac321` against the
unchanged baseline `f64ce4ecb6e16b361ad27ba3aeb6b0040d51e3df7c26047bc250993d6790cac6`.
`blur` now dispatches once, before the vertical loop: a kernel at or below
`REGISTER_TAPS` (32) taps goes to `accumulate_register_rows`, which resolves the
reflected source offsets once per output row into a fixed array, keeps one
accumulator in a register across every tap of an eight column block, and stores
once. The per-tap path stays for longer kernels and for a target without AVX2.

the dispatch threshold is measured, not assumed. the vertical pass alone, on a
2903x4128 plane, against the shipped per-tap `accumulate_avx2`:

| sigma | taps | register | per tap | ratio |
| ---: | ---: | ---: | ---: | ---: |
| 0.45 | 5 | 10.8 ms | 13.8 ms | 1.29x |
| 0.5 | 5 | 10.9 ms | 14.8 ms | 1.35x |
| 0.8 | 7 | 11.9 ms | 16.7 ms | 1.41x |
| 2 | 17 | 19.6 ms | 27.2 ms | 1.39x |
| 8 | 65 | 86.9 ms | 90.2 ms | 1.04x |
| 16 | 129 | 284.8 ms | 188.7 ms | 0.66x |

above 32 taps the register form loses, because visiting many tap rows for each
column block leaves the working set, so those kernels keep the per-tap path.
every row above is bit-identical between the two.

the separable blur, which is what a frame actually pays for, over the same
plane:

| sigma | scalar vertical | register vertical | ratio |
| ---: | ---: | ---: | ---: |
| 0.45 | 47.40 ms | 43.97 ms | 1.078x |
| 0.5 | 50.92 ms | 47.31 ms | 1.076x |
| 0.8 | 53.39 ms | 48.43 ms | 1.102x |
| 2 | 100.56 ms | 95.23 ms | 1.056x |
| 8 | 560.85 ms | 559.67 ms | 1.002x |
| 16 | 1254.58 ms | 1369.83 ms | 0.916x |

the mask and the candidate use sigma 0.45, 0.5 and 0.8, so the 5.6% to 10.2%
band is the relevant one.

`cargo test --locked` passed 90 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks. a unit test compares the register
path against the scalar one at five shapes, including a 1x1 and a 2903x4 plane,
and four sigmas, by float bits. the twelve case Deblur hash harness reports the
same hash as the baseline on all twelve, including both float `NaN` pages.

integrated, the noise floor is wider than the effect: two A/B rounds of
`--suite posterize --limit 6` on 12 megapixel pages moved the plugin between
4.74 s and 5.17 s for `deblur` and 1.20 s and 1.24 s for `deblur-unsharp`
whichever pass was in use. the medians are 4.98 s against 5.05 s and 1.63 s
against 1.65 s, so no regression is measured and no frame-level gain is
claimed here either.
