# Levels and Posterize as device chain members

status: landed. third plan of the set in [15](15-vulkan-device-path.md), after
[17](17-vulkan-deblur.md). gated by [16](16-vulkan-validation.md). `gpu=1` runs
both filters on the device for every integer format they accept, byte-identical
to the host path, and [the outcome](#outcome-so-far) records where that pays.

## current cost

both filters are one 256-entry (or native-range) lookup applied per sample, and
both are already at the load/store issue limit:

| filter | measured | normalised |
| --- | ---: | ---: |
| `Levels` integer, 2.8 MP | 0.9 ms | 0.34 ns per sample |
| `Levels` integer, 12 MP | 4.6 ms | 0.34 ns per sample |
| `Posterize` even, 12 MP | 4.3 ms | 0.34 ns per sample |
| `Levels` float `GRAYS`, 4.2 MP | 5.8 ms | 1.4 ns per sample |

0.34 ns is about one cycle, which is what a load, a table lookup and a store cost
when the loads and stores are the limit ([01](01-performance-plan.md#candidate-status)).
there is no arithmetic to move to a device.

the float path is a different matter and is not in scope: it evaluates a
`f64` `powf` per sample (`src/levels.rs:194-217`). `f64` is optional in Vulkan and
a driver's `powf` is not Rust's. float `Levels` stays on the host, and `gpu=1` on
it is refused with a message rather than silently falling back.

## why a device map is a loss alone

a map reads one plane and writes one plane. on a 12 MP `GRAY8` frame that is
24 MB of device traffic, about 0.2 ms, against 0.25 to 0.35 ns per pixel of bus in
each direction ([15](15-vulkan-device-path.md#the-bus-budget)). the bus alone is
2 ms up and 4 ms down for that frame, so a single `Levels(gpu=1)` between two
host filters is about 6.6 ms against 4.6 ms on the host, a loss of 1.4x. a
three-plane frame triples the traffic and triples the transfer, and the loss
grows with it.

| chain | CPU | device |
| --- | ---: | ---: |
| `Levels` alone, host in and out | 4.6 ms | 0.2 ms kernel, 5.9 ms transfers, 0.5 ms copies |
| `Levels` after a device `Deblur` | 4.6 ms plus 4 ms download | 0.2 ms |

## why it earns its place after a device producer

once a device filter exists, the frame is already device-resident and the bus is
already paid. the map then costs 0.2 ms instead of 4.6 ms, and it runs before the
single download at the end of the chain:

| chain | device work | CPU work | transfers | per frame |
| --- | ---: | ---: | ---: | ---: |
| `Deblur(m0)`, download, host `Levels`, host `Posterize` | 28 ms | 8.9 ms | 5.9 ms | about 43 ms |
| `Deblur(m0)`, `Levels(gpu=1)`, `Posterize(gpu=1)`, download | 28 ms | 0 | 5.9 ms | about 34 ms |

that is 20% of the chain, which is the reason to write the kernels at all. the
same measurement is what decides whether they stay.

## what moves

| piece | where | shape |
| --- | --- | --- |
| the table | host | `curve` / `posterize_table`, uploaded once for fixed arguments, rewritten per frame for `use_props` |
| sample read, clamp, lookup, write | device | one dispatch per plane, four samples an invocation |
| plane iteration | host | one dispatch per plane, from the frame's own plane geometry |
| frame and properties | host | `new_gpu_video_frame(format, w, h, Some(&src))`, the same property carry the CPU path gets from `prop_src` |
| `Posterize(method=1)` | host | refused under `gpu=1`: the Lloyd solve reads a histogram |

the table is uploaded into a `GRAY32` device frame and bound with `exec.read_plane`
like any other plane. a `u8` table is 256 words, 1 KB. a native `u16` table is
65,536 words, 256 KB. a fixed table is copied in once at `create`. a `use_props`
table changes per frame, so each change is a fresh table frame uploaded through
the same `vkCmdCopyBuffer2` path, small enough that the staging buffer is the
whole cost. the table is deliberately not rewritten in place: a frame's recording
can still be reading the previous one.

the lookup keeps the CPU's index rule: the table is indexed by
`min(sample, len - 1)` (`src/filters/mod.rs:382`), and `Posterize` with
`bits == depth` returns the input unchanged instead of dispatching an identity
map.

## files

| file | role |
| --- | --- |
| `src/gpu/mod.rs`, `src/gpu/shared.rs` | the shared device layer, launch geometry, and `upload_words` |
| `src/gpu/map.rs` | the lookup kernel both filters use, and the per-plane recording |
| `src/filters/{levels,posterize}.rs` | the `gpu` argument, `gpu_output()`, the transfer, the call |
| `src/filters/mod.rs` | `MappingTable::device_words` and `device_table`, the upload of a table |
| `src/levels.rs`, `src/posterize.rs` | the table builders stay here, and the kernel uploads their output |

one module covers both filters because the kernel is the same lookup; only the
table builder differs, and that stays with the algorithm. `Posterize(method=1)`
has no device form until a histogram exists; its table solve stays in
`src/posterize.rs` and runs on the host.

## exactness

the integer paths are byte-exact by construction, which is why they are the safe
ones to port:

- the table is built by the same host function the CPU path calls, so it already
  carries the ties-to-even rounding, the native ranges and the endpoint rules.
- the kernel does one clamped index and one store. no float arithmetic runs at
  all on the integer paths.
- `Levels(use_props=True)` reads `NImagesBlackLevel` and `NImagesWhiteLevel` from
  the frame header, which is present on a device frame, and rebuilds the table
  the same way.
- the property set is copied by `new_gpu_video_frame` from the input, which is
  the device equivalent of passing `prop_src`.
- a sample is written through a masked word update, exactly as the CPU writes one
  sample at a time, so the bytes it shares a word with are left alone.

`u16` tables are built natively rather than by scaling an 8-bit table, the same
rule the CPU path follows. the table build stays on the host because a
255-to-65535 remap is not something to reimplement in a shader.

## expected

on a 12 MP `GRAY8` frame one plane is 0.2 ms and three planes 0.6 ms at the
assumed 120 GB/s, against 4.6 ms for `Levels` and 4.3 ms for `Posterize` on the
host. neither number is a reason to write the kernel on its own, and both are a
reason to write it once a device `Deblur` exists.

the device map also removes the CPU work from the worker threads, which matters
under the fan-in: the host's 12 workers are already the bottleneck at 2.7x
aggregate throughput ([03](03-deblur-filter.md) lines 564-581), so work that leaves
them is worth more than its stage time suggests. the fan-in measurement in
[16](16-vulkan-validation.md) is where that shows up.

## risks

- **the table changes per frame.** `use_props=True` rebuilds 256 entries and
  `u16` uses 65,536. each change is an upload of 1 KB or 256 KB, so the table
  stays a device frame rather than a host-visible binding. a per-frame rewrite is
  cheap but not free, and the `use_props` measurement includes it. measured, it
  does not move the chain number noticeably.
- **the grid limit.** 12 MP at four samples an invocation is 3.0 M invocations,
  46,875 groups of 64. the 24 MP spread is 93,750 and needs the strided loop.
- **`u16` and 16-bit storage.** `storageBuffer16BitAccess` is optional and the
  pipeline builder does not expose it, so a `u16` plane is read as `u32` words and
  each sample is picked out of a word in host endianness. big-endian is refused.
  the kernel writes one sample through a masked word, so an odd width leaves no
  tail to handle.
- **a silent CPU fallback.** `gpu=1` on a host without a device must fail with the
  name of the missing capability, not run the host kernel and return a frame the
  node declared device-resident. the core treats that mismatch as fatal.
  `gpu::require_device` refuses at creation.
- **`method=1`.** refusing it under `gpu=1` is a public surface question. the
  argument stays one `method`, and the error names the combination. the
  alternative, a silent host solve, makes the residency contract lie.
- **the chain's footprint.** each map allocates its own output frame, so a chain
  of three device filters holds three frames per frame in flight. at a wide fan-in
  that reaches the device's memory ceiling before the arithmetic does, which is
  the same wall [17](17-vulkan-deblur.md#aggregate) records at 24 MP. the
  aggregate table under [the outcome](#outcome-so-far) shows where it starts.

## milestones

| # | delivers | gate |
| --- | --- | --- |
| M1 | `Levels(gpu=1)` integer, `u8` and `u16`, Gray, RGB and YUV, every plane | done: byte-exact on 20 cases, one plane and three planes |
| M2 | `Posterize(gpu=1)` even, `u8` and `u16` | done: byte-exact, including the `bits == depth` identity |
| M3 | `use_props` per-frame table | done: byte-exact for both filters |
| M4 | chain measurement | done: kept, see the outcome, 1.2x to 1.5x while the chain fits |

## outcome so far

written 2026-10-24. `Levels(gpu=1)` and `Posterize(gpu=1)` are in the tree for
every integer format the filters accept. the gate is `tools/map_hash.py`:

| group | cases | result |
| --- | ---: | --- |
| `Levels`, `GRAY8`/`GRAY16`/`RGB24`/`YUV420P8`/`YUV444P16` | 20 | byte-identical |
| `Posterize`, same families, `bits` 3 to 12 | 7 | byte-identical |
| `Posterize(bits == depth)` identity | 1 | byte-identical, no dispatch |
| `Levels(use_props)` and `Posterize(use_props)` | 3 | byte-identical, table rebuilt per frame |
| `Levels(gpu=1)` on a float clip | 1 | refused, names the float curve |
| `Posterize(gpu=1, method=1)` | 1 | refused, names the histogram |
| odd width, 2x2, 1 px, subsampled 4:2:0 | 6 | byte-identical |

29 cases, all of them identical or refused as intended. the host table
(`tools/map_hash.py --check`) is recorded as well, so a later change to a table
builder shows up without needing a device.

### the chain

one frame per pull through `Deblur(gpu=1)` and the two maps, median of eight
distinct frames, on a 12 MP `GRAY8` page:

| method | host maps | device maps | saved |
| --- | ---: | ---: | ---: |
| `1` | 38.3 ms | 22.3 ms | 42% of the frame |
| `0` | 66.9 ms | 50.1 ms | 25% of the frame |

aggregate, 13 frames through `std.AverageFrames`, median of three:

| frames in flight | method | host maps | device maps | ratio |
| --- | ---: | ---: | ---: | ---: |
| 1 | 1 | 28.4 fps | 39.6 fps | 1.40x |
| 1 | 0 | 15.5 fps | 23.2 fps | 1.49x |
| 3 | 1 | 60.1 fps | 72.3 fps | 1.20x |
| 3 | 0 | 28.0 fps | 32.3 fps | 1.15x |
| 5 | 1 | 74.2 fps | 90.9 fps | 1.22x |
| 5 | 0 | 36.2 fps | 35.6 fps | 0.98x |
| 13 | 1 | 114.1 fps | 103.6 fps | 0.91x |
| 13 | 0 | 42.2 fps | 10.3 fps | 0.24x |

the chain wins while it fits: 1.2x to 1.5x up to five frames in flight. past
that the third device frame per stage reaches the card's ceiling and the device
chain becomes slower than downloading and mapping on the host, which is the same
allocator wall [17](17-vulkan-deblur.md#aggregate) measured for a single filter
at 24 MP, reached here at 12 MP by a chain of three device filters.

**decision: kept.** the maps win in the shape a caller actually has — one frame at
a time, or a bounded fan-in — and cost nothing when the chain is narrow. the wide
fan-in case is a footprint problem with the same fix as [17](17-vulkan-deblur.md):
fewer frames per stage, which tiling would give.

## done when

`Levels(gpu=1)` and `Posterize(gpu=1)` are byte-exact against `gpu=0`, the chain
in `docs/BENCH.md` reports the with-and-without numbers, and the keep or drop
decision is recorded here. all three hold.
