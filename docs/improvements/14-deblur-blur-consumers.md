# consume deconvolution blur rows

status: proposed, unmeasured. preserve the arithmetic and change when each
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
