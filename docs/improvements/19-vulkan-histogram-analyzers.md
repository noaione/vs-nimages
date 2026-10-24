# the histogram analyzers on the device

status: landed. last plan of the set in [15](15-vulkan-device-path.md), and the
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
measured at 12 MP, flat page, median of 20 frames, in
[the outcome](#outcome-so-far): the counting step is 0.5 ms at 8 bit and 1.3 to
2.4 ms at 16 bit inside a device chain, against 7.7 to 8.0 ms and 19.7 to
23.4 ms on the host. with the frame uploaded and downloaded it is 5.3 ms and
12.7 to 13.9 ms, which is why a lone analyzer gains 1.6x to 1.9x rather than the
3.3x this plan guessed: 12 MB each way is more than the histogram ever was.

the case that pays is a device chain, where the frame is already resident and the
histogram adds 0.5 ms and 1 KB instead of 7.7 ms. that case is
`Posterize(method=1)`: it needs plane 0's histogram, a host solve of 40 passes
over 256 bins, a table, and a map. on the device the whole filter becomes one
histogram dispatch, a 1 KB readback, the same host solve and one map dispatch,
which is 6.2 to 6.4 ms against 20.0 to 21.1 ms at 8 bit and 14.0 to 14.2 ms
against 49.1 to 51.8 ms at 16 bit. the Lloyd solve itself stays on the host,
where it belongs: 256 bins and microseconds.

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

the one difference is which engine counted. `tools/histogram_hash.py` compares
the two histogram outputs through the properties, and the properties are the
contract.

## expected

estimated before the port; [the outcome](#outcome-so-far) has what it measured.
it held for the counting step and was optimistic about a lone analyzer.

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
  bin. the workgroup table is the mitigation: the case table carries flat,
  four-tone, sparse and noise pages, and the flat page is also what `--bench`
  times, so the number above is the worst case rather than the average.
- **the 16-bit table.** 65,536 bins do not fit in shared memory, so `u16` takes
  global atomics and a 256 KB readback. it was measured separately and it wins:
  1.3 to 2.4 ms against 19.7 to 23.4 ms, so `u16` moved too.
- **the readback preference.** `DEVICE_LOCAL` is free on unified memory and the
  aperture on a discrete card. the allocation states `HOST_CACHED`, and [16](16-vulkan-validation.md)
  measures the readback with the histogram attached.
- **`copy_frame` on a device frame.** it shares the planes and keeps the property
  map, so `gpu=1` returns a device frame with the input's pixels and the
  analyzer's properties. the harness compares both, and the caller downloads
  the frame when it needs the pixels.
- **the format gates.** a constant format is checked at create, the frame's own
  format is re-checked per frame, and a variable-dimension clip reports
  `Undefined` at the node. the device path changes none of them.
- **it was the last plan because it looked like the least valuable.** the chain
  measurement says otherwise for `Posterize(method=1)`, at 3.2x to 3.6x; without
  a device producer it is the 1.6x to 1.9x the transfer allows.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | `u8` histogram on the device, read back, compared to `from_plane` | done: identical on 35 cases, the flat page included |
| M2 | `PeakStats(gpu=1)` and `PeakGrayShades(gpu=1)` | done: properties and pixels identical, empty arrays included |
| M3 | `u16` histogram | done: 1.3 to 2.4 ms against 19.7 to 23.4 ms, so it moved |
| M4 | `Posterize(method=1, gpu=1)` using the device histogram | done: bytes identical, 3.2x and 3.6x on the chain |
| M5 | measurement | done: kept at both depths, see the outcome |

## outcome so far

written 2026-10-24. `PeakStats(gpu=1)` and `PeakGrayShades(gpu=1)` count plane 0
on the device and analyse the table with the same host functions, and
`Posterize(method=1, gpu=1)` builds its levels from that table. the gate is
`tools/histogram_hash.py`:

| group | cases | result |
| --- | ---: | --- |
| `PeakStats`, `GRAY8`/`GRAY10`/`GRAY16`, six argument sets | 14 | identical levels and flags |
| `PeakGrayShades`, same depths, four thresholds | 8 | identical shade and percentage arrays |
| flat, four-tone, sparse and noise pages | 8 | identical, the contention cases included |
| `1x1`, `2x2` | 3 | identical |
| a non-Gray clip and a float clip | 2 | refused by name on both engines |

35 cases, all identical or refused as intended, and the pixels come back
byte-identical too because an analyzer only copies the frame. `--check` records
the host properties, so a change to the peak search or the shade ordering shows
up without a device. `--device` also runs the chain this plan exists for:
`PeakGrayShades(gpu=1)` into `Posterize(use_props=1, method=1, gpu=1)` against the
same graph on the host, byte-identical.

### the counting kernel

12 MP, flat page, median of 20 frames, per frame, from
`tools/histogram_hash.py --bench`:

| piece | 8 bit | 16 bit | host |
| --- | ---: | ---: | ---: |
| counting, marginal in a device chain | 0.5 ms | 1.3 to 2.4 ms | 7.7 to 8.0 ms, 19.7 to 23.4 ms |
| counting, with the frame uploaded and downloaded | 5.3 ms | 12.7 to 13.9 ms | 8.5 ms, 24.7 ms |

the counting step is 10x to 18x cheaper than the host's. the 16 bit row is what
65,536 bins cost: it counts into global memory with no workgroup table, and the
readback is 256 KB rather than 1 KB. the standalone row is transfer bound either
way, so a lone analyzer gains 1.6x to 1.9x rather than the 3.3x this plan
guessed: 12 MB of upload plus 12 or 24 MB of download is more than the histogram
ever was.

the flat page is the case the workgroup table exists for, and it is the number
above: a 12 MP page of one tone counts in 0.5 ms at 8 bit, with one `atomicAdd`
per bin per workgroup rather than one per sample.

### the chain that pays

`PeakGrayShades` into `Posterize(use_props=True, method=1)`, 12 MP, median of 20
frames:

| depth | host | device | saved |
| --- | ---: | ---: | ---: |
| `GRAY8` | 20.0 to 21.1 ms | 6.2 to 6.4 ms | 3.2x |
| `GRAY16` | 49.1 to 51.8 ms | 14.0 to 14.2 ms | 3.6x |

most of the win is the histogram the Lloyd solve used to read from a host plane:
7.7 ms to 0.5 ms at 8 bit. the map and the per-frame table upload are the rest.
`Posterize(method=1, gpu=1)` no longer refuses, and `tools/map_hash.py` covers its
output bytes on `GRAY8`, `GRAY16`, `RGB24`, `YUV420P8` and through `use_props`.

**decision: kept, at both depths.** the 16 bit table moved rather than staying on
the host: 1.3 to 2.4 ms against 19.7 to 23.4 ms is worth the 256 KB readback, and
the kernel has no correctness difference to lose. what the numbers do not say: a
standalone 16 bit `Posterize(gpu=1)` is a loss once the frame crosses the
boundary, [18](18-vulkan-pointwise-maps.md#outcome-so-far) records the same for
the maps. this is a chain member at every depth, which is what the plan claimed.

## done when

the histogram on the device equals the host's element for element, both analyzers
write the same properties under `gpu=1`, and the keep-or-drop decision, including
whether `u16` moved, is recorded here with the numbers from
[16](16-vulkan-validation.md).

all three hold, and the gate is `tools/histogram_hash.py`.

both analyzers took `Gray` only when this plan landed. the color input surface is
[20](20-analyzer-color-inputs.md), which widened them to `YUV` and `RGB` the same
day.
