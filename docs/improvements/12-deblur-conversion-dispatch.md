# Deblur conversion dispatch

status: deferred on measurement. specialize conversion, preserve kernel
arithmetic.

## current cost

[`Convert::read` and `Convert::write`](../../src/filters/deblur.rs) dispatch on
sample width for each pixel. reads also perform slice/array conversions and
writes perform f64 scaling and ties-to-even quantization. `read_luma` scans RGB
three times; `write_rgb` reads three source channels, finds their gamut bounds,
and writes their common limited delta.

the old generic chunk-based write regressed threefold. moving the same dynamic
chunk length into a new helper would repeat that failed experiment. generated
code can already hoist some invariant checks, so source branch counts alone do
not establish a speedup.

## one experiment, in two separately measured variants

1. dispatch once per plane/frame to fixed-width u8, u16 or f32 row helpers. use
   validated row spans and fixed-size sample arrays. preserve the existing
   multiplication, accumulation and rounding sequence. inspect assembly before
   adding manual vectors; retain safe scalar tails.
2. if conversion still dominates, vectorize those specialized helpers with
   runtime dispatch. begin with Gray/YUV luma conversion and integer writes,
   then test RGB as a separate extension. keep each source/output plane's own
   stride and the frame format's native maximum.

RGB reads can traverse R/G/B rows together, but preserve the current sequence:
`(R * scale) * wr`, then add `(G * scale) * wg`, then add `(B * scale) * wb`.
computing weighted native samples first and scaling afterward changes rounding.
keep the equal-delta gamut clamp in RGB writes; three independent clamps change
the filter's behavior.

## parity traps

integer output currently scales f32 code values in f64. a float32 multiply
followed by packed integer conversion can cross a quantization midpoint. keep
f64 scaling and ties-to-even rounding, then saturate and pack to the native
range. check values immediately around half steps and each depth's maximum.

float output currently converts `f64(code) * inverse` back to f32. preserve
that sequence, NaN handling, infinities and signed zero. do not globally enable
fast-math or FMA, and do not rely on aligned rows for unaligned vector loads.

the magic-constant rounding helper is already an implemented optimization.
replacing it is outside the scalar-dispatch variant; compare vector rounding
directly against it, including exceptional float-derived code values.

## measure and decision

compare `luma`, `write`, and total time at Gray8/16/S, RGB24/48/S and subsampled
YUV. measure both Deblur methods: the same conversion saving occupies a larger
share of method 1 than method 0. cover tiny/odd widths, vector tails and full
pages under sequential and concurrent pulls.

test this independently of 10's allocation change. differential comparisons
include full output pixels, input preservation and YUV chroma. the total saving
cannot exceed the measured conversion share; reject a fast microbenchmark if
the full filter regresses. keep scalar code for targets without the vector ISA.

## result

status: deferred on measurement. the conversion share is small and a previous
same-code rewrite regressed threefold.

the two stages this candidate covers are `luma` (the read) and `write` (the
round, saturate and store), both already reported by `debug=1`. medians of
three graph builds at 2048x2048, three frames each, on the build with the
horizontal and vertical vector passes:

| case | luma | write | total | read + write share |
| --- | ---: | ---: | ---: | ---: |
| `method=0`, `GRAY8` | 3.50 ms | 10.82 ms | 155.70 ms | 9.2% |
| `method=0`, `GRAY16` | 4.59 ms | 12.20 ms | 156.13 ms | 10.8% |
| `method=1`, `GRAY16` | 4.50 ms | 12.17 ms | 45.95 ms | 36.3% |

the read is 2 to 3% of a `method=0` frame and the write is 7 to 8%; on the
unsharp mask the pair is 36%, but that is because `restore` is four times
cheaper there, not because the conversion got bigger. the absolute saving the
candidate can reach is therefore a fraction of 3.5 to 4.6 ms and a fraction of
10.8 to 12.2 ms.

the review records why that is not free: a chunk-based write that moved the same
arithmetic regressed the stage threefold, 29.4 to 94.9 ms, because the chunk
length was a runtime value and the constant-length bounds elision went away. a
fixed-width dispatch has to keep f64 scaling and ties-to-even rounding exactly,
so the arithmetic cannot move, and that is the same shape of change that already
measured worse once.

both `luma` and `write` are inside the run to run spread of the integrated
benchmark at these sizes, so no adoption is recorded. the scalar dispatch stays,
`Convert::read` and `Convert::write` are unchanged, and `tests/check-nimages.py`
passes its 2150 checks on the shipped build unchanged.
