# the histogram analyzers on the device

status: proposed. last plan of the set in [15](15-vulkan-device-path.md), and the
one that unlocks `Posterize(method=1)` as a chain member. gated by
[16](16-vulkan-validation.md).

## current cost

`PeakStats` and `PeakGrayShades` both build the same 256-bin (or 65,536-bin)
histogram and then analyse it:

| filter | 2.8 MP | 12 MP | normalised |
| --- | ---: | ---: | ---: |
| `PeakStats` | 1.8 ms | 7.7 ms | 0.64 ns per pixel |
| `PeakGrayShades` | 1.8 ms | 7.8 ms | 0.64 ns per pixel |

the histogram is 0.34 ns per sample and scatter-bound: the bin table is a chain of
dependent loads, so halving the counters changed nothing
([01](01-performance-plan.md#rejected-candidates-measured)). the rest is a
`copy_frame` that shares pixel storage, at most 61 bins of peak search, and for
the shades filter a stable sort by count and `f64` percentages. the split is not
instrumented: at 0.34 ns per sample the histogram is about half of the 0.64 ns
per pixel the filters measure.

the peak search and the shade analysis are microseconds. the histogram is the
whole cost, and the histogram is a reduction with a 1 KB result, which is the
cheapest kind of work to move to a device:

- one read per pixel
- no writes to the frame
- a 1 KB readback for `u8`, 256 KB for `u16`
- the analysis after it is the same host function either way

## what moves

only the histogram. the peak search, the shade ordering, the property arrays and
`copy_frame` stay where they are, so the two engines cannot disagree about what a
histogram means.

| piece | where | shape |
| --- | --- | --- |
| zero the bins | device | `vkCmdFillBuffer`, then a barrier before the dispatch |
| count | device | one invocation per four pixels, a workgroup's own bins, one `atomicAdd` per bin at the end |
| read back | device to host | `TRANSFER_DST` buffer, `HOST_VISIBLE`, **preferring `HOST_CACHED`** |
| peaks and shades | host | `find_local_peak`, `analyze_gray_shades`, unchanged |
| properties, frame | host | `copy_frame` plus `properties_mut`, the same as the CPU path |

the `u32` counter is exact for any frame below 2^32 samples, which is every page
this plugin is for: a 24 MP page is 24 million. a frame can be declared large
enough to reach 2^32, so the device path refuses that size instead of wrapping a
count, and widening to `u64` on the readback is lossless.

`u8` gets one 256-entry `u32` table in workgroup shared memory, 1 KB, drained once
per workgroup. `u16` cannot: 65,536 bins is 256 KB, past what any device offers at
that limit, so it counts into global memory with `atomicAdd` and takes the hot-bin
cache only if the measurement says the same-address contention matters. this is
the shape [09](09-striped-histogram-counters.md) rejected on the CPU for 16 bits
and the shape a device has to solve differently.

## files

| file | role |
| --- | --- |
| `src/gpu/mod.rs` | the shared layer, including the `HOST_CACHED` readback buffer |
| `src/gpu/shared.rs` | launch geometry and the workgroup drain |
| `src/gpu/histogram.rs` | zero, count, drain, read back |
| `src/filters/{peak_stats,peak_gray_shades}.rs` | the `gpu` argument, `gpu_output()`, the transfer, the property write |
| `src/histogram.rs` | a constructor from a counted table, so both engines hand `find_local_peak` the same type |

one kernel serves both analyzers: they count the same table and differ in what
the host does with it.

## the readback is the trap

the sibling paid 17 to 20 ms for a 1.3 MB readback because the buffer carried a
`DEVICE_LOCAL` preference and landed in the uncached aperture; `HOST_CACHED` took
it to about 0.1 ms (the sibling's plan 12). this plugin's
readback is three orders of magnitude smaller, so the wrong preference would still cost more than the
histogram it carries.

the rule is therefore explicit in the allocation, not inferred: `TRANSFER_DST`,
`HOST_VISIBLE | HOST_COHERENT`, `HOST_CACHED` first, and the mapped slice read
after the submission's timeline value is waited out. `gpu=1` waits because the
host needs the numbers; the whole point is that the wait is short.

## what it buys

standalone, the transfer is the argument:

| path | 12 MP |
| --- | ---: |
| CPU `PeakStats` | 7.7 ms |
| device histogram, device frame in and out | 2.3 ms |
| the same, with the frame then downloaded for a CPU consumer | 6.3 ms |

so a lone analyzer is 3.3x in latency and a wash once the consumer needs the
frame on the host. the case that pays is a device chain, where the frame is
already resident and the histogram adds 0.3 ms and 1 KB instead of 7.7 ms.

that case is `Posterize(method=1)`: it needs plane 0's histogram, a host solve of
40 passes over 256 bins, a table, and a map. on the device the whole filter becomes
one histogram dispatch, a 1 KB readback, the same host solve and one map dispatch.
on a 2048x2048 frame it is about 1 ms against 3.2 ms at 8 bits and 6.7 ms at
16 bits. the Lloyd solve itself stays on the host, where it belongs: 256 bins and
microseconds.

## exactness

- counts are exact `u32` values widened to the `u64` the host functions take.
- the host histogram is built by `from_plane` with saturating `u64` adds. a count
  that cannot overflow cannot differ, so the two tables compare equal element for
  element.
- peak selection, the threshold comparison, the stable sort and the `f64`
  percentages all run on the same array, so they cannot drift.
- `NImagesGrayShades` and `NImagesGrayShadePercentages` are always both written,
  including at length zero.
- a device frame's properties live in its header, so `copy_frame` plus
  `properties_mut` works the same as on a host frame.

the one difference is which engine counted. the validator compares the two
histogram outputs through the properties, and the properties are the contract.

## expected

| page | histogram dispatch | readback | total on device |
| --- | ---: | ---: | ---: |
| 12 MP `GRAY8`, 256 bins | 0.3 ms | 1 KB, about 0.01 ms | 0.4 ms |
| 24 MP `GRAY8`, 256 bins | 0.6 ms | 1 KB | 0.7 ms |
| 12 MP `GRAY16`, 65,536 bins | 0.5 ms | 256 KB, about 0.05 ms | 0.6 ms |

flat pages are the risk, not the average: the sibling measured 21 ms for a
per-pixel `atomicAdd` histogram on a page of paper and ink against 6.7 ms on the
same pixels drawn as noise, and 1.9 ms once a workgroup counted its own bins
(the sibling's plan 09). this kernel uses the
workgroup-bins shape from the start, and the flat-page probe is a fixture rather
than a footnote.

## risks

- **same-address contention.** a page of flat tones points most pixels at one
  bin. the workgroup table is the mitigation, and the fixture set includes a flat
  page and a four-tone page so the regression cannot hide.
- **the 16-bit table.** 65,536 bins do not fit in shared memory, so `u16` takes
  global atomics. the `u16` histogram is measured separately, and if the device
  loses there, `u16` stays on the host while `u8` moves.
- **the readback preference.** `DEVICE_LOCAL` is free on unified memory and the
  aperture on a discrete card. the allocation states `HOST_CACHED`, and [16](16-vulkan-validation.md)
  measures the readback with the histogram attached.
- **`copy_frame` on a device frame.** the returned frame has to be device
  memory for `gpu=1`. the core refuses a host frame from a node declared
  `GPUOutput`, fatally, so the frame is allocated and returned as device memory
  and the caller downloads it when it needs it.
- **the format gates.** a constant format is checked at create, the frame's own
  format is re-checked per frame, and a variable-dimension clip reports
  `Undefined` at the node. the device path changes none of them.
- **it is the least valuable plan.** this is why it is last. if the Deblur chain
  never materialises, nothing here pays for itself.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | `u8` histogram on the device, read back, compared to `from_plane` | element-for-element equality on every fixture and a flat page |
| M2 | `PeakStats(gpu=1)` and `PeakGrayShades(gpu=1)` | properties identical to `gpu=0`, including the empty arrays |
| M3 | `u16` histogram | same equality; drop `u16` from the device path if it loses |
| M4 | `Posterize(method=1, gpu=1)` using the device histogram | level bits identical, chain measured |
| M5 | measurement | keep or drop by the bar in [01](01-performance-plan.md#adoption-criteria) |

## done when

the histogram on the device equals the host's element for element, both analyzers
write the same properties under `gpu=1`, and the keep-or-drop decision, including
whether `u16` moved, is recorded here with the numbers from
[16](16-vulkan-validation.md).
