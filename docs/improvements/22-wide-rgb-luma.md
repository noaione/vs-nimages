# a wider luma kernel

status: measured, not kept. the 256-bin scatter sets the floor, not the luma
arithmetic.

## current state

`src/luma.rs` computes an 8-bit row with an `sse2` kernel sixteen samples at a
time and a native 16-bit row with an `sse4.1` kernel eight at a time. both
evaluate the reference's integer expression exactly and both fall back to a
scalar row. [21](21-vector-rgb-luma.md#risks) left a 256-bit kernel as a
follow-up if the 8-bit reader stayed above the memory floor. it does: at 12 MP
follow-up if the 8-bit reader stayed above the memory floor. [21's floor
table](21-vector-rgb-luma.md#outcome-so-far) already said it does not: the
vector rows cost 4.5 to 5.2 ms for a page whose three-plane read is 6.2 to
6.6 ms. the build is the confirmation, and it costs one plan to settle.

## what was built

four functions in `src/luma.rs`: `luma8`, `luma8_wide`, `rows_u8_avx2` and
`rows_u16_avx2`. the 8-bit kernel takes thirty-two samples per iteration and the
16-bit one sixteen, against sixteen and eight for the 128-bit pair, so each
sample costs about half the instructions. `is_x86_feature_detected!("avx2")`
runs the wide kernel first and the 128-bit one still covers a block the wide
kernel cannot fill.

the integer expression, the halved-green trick and the rounding term are the
same, so the counts cannot move. `every_width_agrees_with_the_scalar_rows`
sweeps lengths 0 through 129, which exercises both new block sizes and every
tail.

## the measurement

12 MP, median of seven frames, one page per process. the two builds are the same
source with only the two step constants differ, and they are alternated: each
round starts one of them first, files swapped in
`vapoursynth/plugins/nimages/`, five pairs per row. the absolute milliseconds
move by a third between sessions on this machine, so only the pair ratio means
anything.

`GRAY8` and `RGB24`, noise page:

| row | 128-bit | 256-bit | ratio |
| --- | ---: | ---: | ---: |
| gray `PeakStats` | 7.57 | 7.35 | 1.03 |
| gray `PeakGrayShades` | 7.25 | 7.75 | 0.94 |
| rgb `PeakStats` | 16.27 | 16.08 | 1.01 |
| rgb `PeakGrayShades` | 15.80 | 15.94 | 0.99 |
| rgb `Posterize(method=1)` chain | 16.25 | 14.23 | 1.14 |

`RGB24`, ramp page:

| row | 128-bit | 256-bit | ratio |
| --- | ---: | ---: | ---: |
| gray `PeakStats` | 7.25 | 7.32 | 0.99 |
| gray `PeakGrayShades` | 7.18 | 7.17 | 1.00 |
| rgb `PeakStats` | 15.27 | 15.06 | 1.01 |
| rgb `PeakGrayShades` | 15.15 | 14.46 | 1.05 |
| rgb `Posterize(method=1)` chain | 16.67 | 14.59 | 1.14 |

`RGB48`, noise page:

| row | 128-bit | 256-bit | ratio |
| --- | ---: | ---: | ---: |
| rgb `PeakStats` | 39.78 | 34.38 | 1.16 |
| rgb `PeakGrayShades` | 34.01 | 31.68 | 1.07 |

## what it says

the chain row and the analyzer row read the same histogram, and they disagree by
15% between the two builds, one of them in the opposite direction from the
other. that is the size of the effect: below the spread. the 16-bit reader is
the most arithmetic bound of the three and still lands at 1.07 to 1.16 with a
pair spread of 1.00 to 1.28.

this is what [21's floor table](21-vector-rgb-luma.md#outcome-so-far) already
implied. the vector rows take 4.5 to 5.2 ms against 6.2 to 6.6 ms of three
plane reads at 8 bit, and 8.2 ms against 13.8 ms at 16, so the arithmetic is
under the memory floor and halving it cannot go below the reads. the filter
rows agree from the other end: an RGB `PeakStats` is 15.3 ms against 7.3 ms of
the same page as gray, and the reader is 5 ms of that gap. the frame copy and
the scatter into 256 bins are the rest.

## exactness

the kernels are byte-exact by construction. the first version was not: `packus`
works on each 128-bit lane separately, so the two middle groups of the 16-sample
block came back swapped, and the width sweep caught it at width 16. the shipped
128-bit kernels never needed a permute because a 128-bit lane is the whole
vector. `_mm256_permute4x64_epi64(packed, 0xD8)` fixes the order.

## decision

not kept. `src/luma.rs` is unchanged, and the wide kernels are not in the tree.
the gate is a ratio that repeats above 1.2 on a quiet machine, or a reader whose
cost is not the scatter.

## what would change the answer

a count that costs nothing per sample: a striped second pass, or a histogram
that accumulates wider than one `u64` per bin without a read-modify-write chain.
[09](09-striped-histogram-counters.md) measured the first for 8 bit and rejected
it for 16, and the device path in
[19](19-vulkan-histogram-analyzers.md) is the second, which is where the
counting actually goes faster.

## done when

a reader above the scatter floor exists to widen, or the same protocol reports a
ratio that repeats outside its own spread.
