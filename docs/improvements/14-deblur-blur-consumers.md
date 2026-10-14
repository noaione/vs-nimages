# consume deconvolution blur rows

status: implemented. kept. preserve the arithmetic and change when each
completed blur row is consumed.

## current cost

[`Workspace::deconvolution`](../../src/deblur.rs) performs two gaussians per
iteration. the first writes the blurred estimate to `third`, then a full-plane
loop reads it to form `second`, the ratio. the second writes the blurred ratio
to `third`, then another full-plane loop reads it to multiply `first`, the
estimate. six iterations mean twelve separate consumer walks after twelve
gaussians.

the streaming `blur` already completes one output row at a time. consuming that
row immediately can keep its values in cache instead of reading the whole plane
after the gaussian finishes. the ratio and multiplication work still exist.
the first variant retains the output stores and the current scratch buffers;
its objective is locality, not fewer arithmetic operations or lower rss.

the current [page benchmark](../BENCH.md) records 17.66 s of plugin `apply` in
19.99 s of deconvolution total. that identifies Deblur as a useful target, but
does not measure the share these consumer walks occupy. the historical blur
shares in 03 and 04 are not a profile of the current build.

## one experiment

add a statically dispatched row consumer to the blur path used by
deconvolution. keep ordinary `blur` available for the mask and other callers.

1. after the first gaussian completes a row in `third`, compute that row of
   `second` as `(value + PEDESTAL) / blurred.max(DIVISION_FLOOR)`.
2. after the second gaussian completes a row in `third`, multiply that row of
   `first` by it.
3. remove the corresponding full-plane loops. keep the initialization,
   strength adjustment, mask and blend unchanged.

use a generic consumer with static dispatch, rather than a dynamic callback
per sample. inspect the generated loops for extra checks or copies. keep the
first variant's existing f32 blur stores so the experiment isolates scheduling.
removing those stores is a separate experiment after this one is measured.

## parity and lifetime traps

the first gaussian reads `first` throughout its run. its consumer writes
`second`, leaving every estimate sample intact until that gaussian finishes.
the second gaussian reads `second`; its consumer can update `first` because
that is a different plane. writing into a gaussian's input while its other rows
are still pending changes the pixels.

retain the horizontal and vertical tap order, the f32 intermediate values, the
division floor, and the multiplication order. preserve NaNs, infinities and
signed zero. do not replace division with an approximate reciprocal or contract
operations into FMA.

fill the required ring rows before resolving their pointers, and finish the
vertical read before invoking the consumer. consumers leave the ring and its
source allocation untouched. keep disjoint field borrows and bounded storage;
the callback introduces no extra frame-sized allocation.

## measure and decision

first time the two existing consumer loops directly on synthetic full-size
planes. this establishes their available share without treating the entire
`restore` stage as potential savings.

compare baseline and candidate on default six-iteration deconvolution, then
one and 64 iterations, narrow and odd widths, maximum radius, and variable
geometry. alternate at least five fresh-process runs with the same cache and
thread count. report `restore`, total latency, concurrent throughput and peak
rss. row consumers increase the number of active streams during the gaussian,
so extra cache pressure can offset the saved rereads.

require baseline output bits, unchanged golden tolerances, scalar and avx2
Miri, and the plugin's repeated, out-of-order and concurrent checks. retain the
shared 15% targeted-stage gain and at most 5% throughput regression gates in
[04](04-optimization-review.md). reject the experiment if only the isolated
consumer timing improves.

this is not a measured speedup. keep it only if the current full filter benefits.

## result

status: implemented. kept on an 8% to 12% throughput gain; the 15% stage gate is
borderline. scalar and avx2 Miri did not run here.

source revision `9ea6c35` plus the working-tree change, rustc 1.99.0. the
candidate DLLs hash `4990877259d68d3ec11ead8a2cbe4e46cb922ea2426f6e5ae336efad6fdb44d1`
(`x86-64-v2`) and `5f40c4ac2cd0952531eeff06451140bb4140509b83eff6589d8042cfb01d5e80`
(`x86-64-v3`), built with `--locked`. the baseline is the same tree with
`src/deblur.rs` from `HEAD`, so the consumer change is the only difference.

`cargo test --locked` passed 96 unit tests and 11 golden tests, `cargo clippy
--all-targets --locked -- -D warnings` and `cargo fmt --check` were clean, and
`tests/check-nimages.py` passed 2150 checks against the installed candidate.
the Miri workflow needs nightly, which this machine does not have, so run it
before relying on the change.

a probe at `.tmpbuild/consbench` holds both versions of `deblur.rs`. it matched
output hashes in 282 shape, radius and iteration cases, including 1x1, 1x40,
33x1 and odd sizes, a radius-zero kernel, and NaN, infinity, signed zero and
1e30 inputs. the two existing consumer loops cost 8.3 ms and 11.3 ms on a
2903x4128 plane, 19.6 ms an iteration, about 118 ms of a 317 ms `restore`. the
loops are memory bound, so the row consumers recover a fraction of that.

probe `restore`, best of five fresh processes with the order alternated:

| case | baseline | candidate | ratio |
| --- | ---: | ---: | ---: |
| default, 6 iterations | 317.5 ms | 296.4 ms | 1.07x |
| 1 iteration | 112.9 ms | 110.6 ms | 1.02x |
| 64 iterations | 2988.5 ms | 2559.2 ms | 1.17x |
| sigma 0.45 | 289.8 ms | 260.0 ms | 1.12x |
| sigma 2 | 509.2 ms | 474.4 ms | 1.07x |
| sigma 16, 2 iterations | 1776.7 ms | 1747.1 ms | 1.02x |
| 33x4128 | 8.8 ms | 8.8 ms | 1.00x |

the probe's concurrent throughput, one workspace a thread, was 1.07x at one
thread, 1.22x at four and 1.15x at twelve, so the extra active streams did not
cost cache. no buffer is added or removed, so peak rss is unchanged.

integrated through the built plugin, a Gray8 2903x4128 clip, `Deblur` defaults,
five fresh processes alternating baseline and candidate, identical output
hashes in every run:

| threads | frames | fps baseline | fps candidate | `restore` median baseline | `restore` median candidate |
| ---: | ---: | ---: | ---: | ---: | ---: |
| 1 | 4 | 2.41 | 2.66 | 394 ms | 339 ms |
| 6 | 24 | 3.35 | 3.75 | 1791 ms | 1522 ms |
| 12 | 24 | 3.40 | 3.69 | 3116 ms | 2803 ms |

the stage gain is 14%, 15% and 10% on the medians and 10%, 13% and 12% on the
minima, so the 15% gate in [04](04-optimization-review.md) is met at most once.
throughput rose 10%, 12% and 8.5% on the best runs and nothing regressed. the
run to run spread on this machine is wider than the smaller gaps. keep it as an
exact change with a consistent small gain; do not quote it as a 15% stage win.