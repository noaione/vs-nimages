# the Vulkan device path

status: proposed. research complete on 2026-10-24 against
`vs-mangacolor` at `29da9fb` and the pinned binding rev `685c29d`. no device
kernel is in the tree.

this plan is the entry point for [16](16-vulkan-validation.md) through
[19](19-vulkan-histogram-analyzers.md). it records what the device path is on
this plugin's binding, what it costs on this host, which of the five filters can
use it, and the order the work goes in.

## why this exists

`vs-mangacolor` runs its two per-pixel walks on VapourSynth's own Vulkan device
behind `gpu=1` (`../vs-mangacolor/src/gpu.rs`). the same binding revision is
already pinned in this repository's `Cargo.toml` with the `vs-vulkan` feature, so
the capability is compiled in and unused.

that sibling is a classifier: it reads the frame on the device, reduces it to a
tens of thousands of words and hands those to a model on the host. three of the five
filters here rewrite pixels instead, which changes the transfer question rather
than the kernel question. that difference is what this plan set is about.

## what the device path is

the plugin links no Vulkan loader and owns no device. VapourSynth compiles the
GLSL, allocates the frames, records and submits the work, and publishes a
timeline value per submission. what the plugin does is:

1. compile one shader per kernel with `core.compile_gpu_shader(GLSL, source)`.
   the core caches by source text, so specialisation is a `#define` preamble.
2. build a `ComputePipeline` with one `storage_buffer()` per binding, in binding
   order, plus `push_constants(bytes)`. push blocks are 32 words.
3. take an `ExecPool` for the compute queue, one per filter instance.
4. in `get_frame`, allocate the output with `core.new_gpu_video_frame(format,
   width, height, Some(&src))`, then `pool.record(|exec| ...)`:
   `exec.read_plane(&src, plane)` and `exec.write_plane(&mut dst, plane)` return
   the `DispatchPlane` objects that declaring and binding share, and
   `exec.dispatch(&pipeline, &planes, &push, [groups, 1, 1])` records one
   dispatch.

a dispatch binds **frames**, not buffers. there is no path from a `GpuBuffer` to
a binding, so every intermediate is a `new_gpu_video_frame`. the framework tracks
frames, not hazards inside one recording, so two dispatches that touch the same
plane need an explicit `vkCmdPipelineBarrier2` between them in the same command
buffer.

per-format sample layout on the device is the same linear pitched plane the host
sees, which means the kernels index `y * stride + x` and never assume a width.

## module layout

the device layer is one directory, so the host algorithms and the device kernels
read apart:

```
src/gpu/
├── mod.rs        the device layer: gate, pipeline build, exec pool, plane
│                 binding, dispatch grid, barrier, upload/download, readback,
│                 and the Vulkan-error mapping
├── shared.rs     what every kernel needs and no core names: launch geometry,
│                 the strided loop, plane-word indexing, the GLSL preamble
├── deblur.rs     the luma read, the separable blur pair, the mask, the blend
├── levels.rs     the integer map, and the refusal of the float path
├── posterize.rs  the even map, and the method=1 histogram hand-off
└── histogram.rs  the 256-bin (65,536-bin) count both analyzers share
```

- `src/filters/<name>.rs` owns the argument, the residency contract and
  `gpu_output()`. it owns no kernel.
- `src/<algorithm>.rs` owns the host-side planning a kernel takes: taps,
  geometry, scale constants, mask parameters. no VapourSynth type, so `cargo
  test` covers it.
- `src/gpu/<name>.rs` owns that filter's GLSL and its record closure.
- `src/gpu/mod.rs` holds the device API: gate, pipeline, pool, plane binding,
  dispatch, barriers, transfers. the kernel modules carry GLSL and their
  filter's sequence, and `src/gpu/shared.rs` names no binding type at all, so
  the launch geometry and the kernels' index arithmetic stay unit-testable.

`histogram.rs` is named for the kernel rather than a filter: `PeakStats` and
`PeakGrayShades` count the same table and differ only in what the host does with
it afterwards.

GLSL lives in `const` `CStr` literals beside its Rust. `compile_gpu_shader`
takes a C string, the core caches by source text, and a `.glsl` asset would need a
`build.rs` step and a wheel entry for no gain. a `#define` preamble is how one
kernel serves several sample widths.

the repository rule that keeps `src/` free of VapourSynth types widens to
`src/lib.rs`, `src/filters/` and `src/gpu/`, and `src/gpu/shared.rs` stays outside
it. `AGENTS.md` records the rule.

## residency, and who pays for the transfer

the binding offers two declarations:

| form | who decides the output | does the core move the input |
| --- | --- | --- |
| `clip:vnode:gpu` | the type | yes, for a `gpu` argument |
| `clip:vnode:all` | `Filter::gpu_output()` per instance | no |

measured with the binding's own sample filter, which declares `clip:vnode:gpu`: a
host clip through that argument returns pixels with no explicit `std.GPUUpload`,
so the core inserted one. the same filter asked to return a device frame from a
`clip:vnode:all` argument is stopped by the core with `Filter ResidencyEcho is
declared to return GPU resident frames but returned a CPU frame`, and that
refusal is fatal. the two forms differ in who moves the frame, and `all` is the
one that moves nothing.

`all` is the form a filter with a `gpu:int:opt` switch needs, because one function
has to serve both engines. it also means the argument is not transferred for the
caller: `vs-mangacolor` invokes `std.GPUUpload` or `std.GPUDownload` by plugin id
itself (`../vs-mangacolor/src/filter.rs:983-1022`). that is the pattern to copy.
a host clip under `gpu=1` becomes an uploaded node; a device clip under `gpu=0`
becomes a downloaded one; a frame under `gpu=1` is returned as device memory and
the caller ends the chain with `std.GPUDownload`.

a LUT or a tap table is uploaded into a `GRAY32` device frame at `create` and
bound with `exec.read_plane`, the same way any other plane is. `exec.uses_buffer`
is not a binding: it keeps a staging buffer alive for a copy the recording already
contains, which is what the sibling's table upload uses it for. a table that
changes per frame is copied into that frame with a recorded `vkCmdCopyBuffer2`
and a barrier, which is 1 KB or 256 KB of transfer, not a free write.

the core itself does not bridge the two worlds. asking a CPU filter to read a
device frame fails at frame fetch with `The frame is GPU resident and its pixel
data cannot be accessed directly, pass the clip through std.GPUDownload first`,
measured on this host. a graph is all CPU or all GPU around a device node.

## the gates

| gate | value | failure surface |
| --- | --- | --- |
| plugin ABI | `vs-43` + `vs-vulkan` features, `VAPOURSYNTH_API_VERSION` 4.3 | build error |
| host version | VapourSynth R80 or newer | plugin does not load |
| device API | Vulkan 1.4 | `vulkan_core_info()` returns an error naming the device |
| entry points | `vkCmdFillBuffer`, `vkCmdPipelineBarrier2`, `vkCmdCopyBuffer2` are all `Option` | `InputError` naming the missing function |
| grid | at most 65,535 groups on x | a strided loop in the kernel |
| push block | 32 words | a second parameter block, or a table |
| 16-bit storage | optional `storageBuffer16BitAccess`, not exposed by the pipeline builder | read `u16` planes as `u32` words in host endianness, and refuse big-endian |
| queues | compute only | `create_gpu_exec_pool` error |
| shader language | GLSL only | compile error |

every one of those is resolved before the first frame, so a host without a device
gets a filter that refuses at creation with a message, not a crash and not a
silently different answer.

## this host

recorded because every number in these plans is from it:

| | |
| --- | --- |
| CPU | i5-11400H, 6 cores / 12 threads, 12 VapourSynth workers |
| device | RTX 3050 Laptop, 3,962 MiB, discrete, `unified_memory = False` |
| Vulkan | 1.4.341, PCIe 4.0 x8, resizable BAR |
| VapourSynth VRAM | budget 3.53 GB, limit 2.35 GB |
| second device | Intel UHD, refused: reports 1.2.201 and 1.4 is required |
| VapourSynth | R80 |

the integrated device is unusable here, so a machine whose only GPU is that one
gets the CPU path for every filter. that is the intended degradation, and it is
worth stating in the README when a `gpu` argument ships.

## the bus budget

the sibling's transfer measurements are the useful number, restated per pixel.
both sides are the same 42.5 MB rgb frame of 14.2 MP unless stated.

| operation | host | measured |
| --- | --- | --- |
| `std.GPUUpload` at the API boundary | RTX 3050 | 5.5 to 7.0 ms, 6.1 to 7.7 GB/s |
| upload inside an existing device chain | RTX 3050 | 0.3 ms marginal or less |
| export to host | RTX 3050 | 5.0 to 7.9 ms, 5 to 8 GB/s |
| `std.GPUDownload` | RTX 3050 | 10 to 24 ms, 1.8 to 4.3 GB/s |
| upload, unified memory | Apple M4 unified | 1.5 to 2.0 ms |
| 1.3 MB readback, `HOST_CACHED` | RTX 3050 | about 0.1 ms |
| 1.3 MB readback, `DEVICE_LOCAL` preference | RTX 3050 | 17 to 20 ms, the aperture |

a round trip on this host is 10.5 to 14.9 ms per 42.5 MB, which is **0.74 to
1.05 ns per pixel for three planes**, 0.25 to 0.35 ns per plane. that is the
number every filter in this repository is compared against, because the CPU cost
of the cheap filters is 0.34 ns per sample.

## the per-filter verdict

CPU figures are this repository's measured stages: 0.34 ns per sample for an
8-bit map, 0.64 ns per pixel for a peak histogram, and the Deblur stages from
[03](03-deblur-filter.md).

| filter | class | CPU per 12 MP | device estimate | verdict |
| --- | --- | ---: | ---: | --- |
| `Deblur method=0` | 28-pass f32 stencil | 360 ms | 25 to 31 ms | **win, first** |
| `Deblur method=1` | 6-pass f32 stencil plus mask | 100 ms | 13 to 15 ms | **win** |
| `PeakGrayShades` | one-read histogram | 7.8 ms | 2 to 3 ms alone | marginal |
| `PeakStats` | same histogram, less host work | 7.7 ms | 2 to 3 ms alone | marginal |
| `Levels` integer | pointwise LUT | 4.6 ms | 0.2 ms plus 6 ms bus | loss alone |
| `Posterize` even | pointwise LUT | 4.3 ms | 0.2 ms plus 6 ms bus | loss alone |
| `Levels` float | pointwise, `f64 powf` | 17 ms Gray | not portable | refused |

the estimate for each is in its own plan. the pattern is the one the sibling
measured: a device walk only pays when the work per pixel is above the bus, and
`Deblur` is the only filter here whose work is.

the LUT verdict is not a statement about the kernel. 0.34 ns per sample is about
one cycle, which is the load/store issue limit, so an 8-bit map has nothing for a
device to beat and the 0.25 to 0.35 ns per plane that the bus costs is the whole
comparison. the same map is worth having **after** a device `Deblur`, which is
[18](18-vulkan-pointwise-maps.md).

## the chain is the unit

a `gpu=1` frame is device memory, so what a caller pays depends on where the chain
ends, not on how many filters are in it:

```
host clip --GPUUpload--> Deblur(gpu=1) --> Levels(gpu=1) --> Posterize(gpu=1) --GPUDownload--> host
             1.8 ms        28 ms           0.2 ms            0.2 ms              4 ms
```

one upload and one download per frame, and every pass between them runs on the
device. that is the shape the plans are written for. a single `Levels(gpu=1)`
between two host filters pays both transfers and loses, which is why it is gated
on a device producer existing.

## decisions

1. `Deblur` is the first kernel. it is the only filter whose CPU cost is above
   the bus by an order of magnitude, and it is the one the caller already waits
   on.
2. every filter keeps its name, its arguments and its defaults. `gpu:int:opt`
   is added, defaulting to 0, declared with `clip:vnode:all`, with
   `gpu_output()` answering per instance.
3. `gpu=1` returns a device frame and `gpu=0` a host frame. transfers are the
   filter's own business and use `std.GPUUpload` / `std.GPUDownload` by id.
4. the CPU path is untouched and stays the reference. no kernel replaces it, and
   `gpu=0` remains the default for every filter.
5. float `Levels` never goes to the device. it evaluates `f64` `powf` per sample
   (`src/levels.rs:194-217`), and neither `f64` nor `powf` is portable across
   drivers. it stays on the host.
6. counters on the device are `u32`. 64-bit atomics are optional in Vulkan, a
   page this plugin is for is far below 2^32 samples, and a frame declared large
   enough to reach that is refused at create.
7. the LUT and tap tables are uploaded into a `GRAY32` frame at `create`. a
   table that changes per frame is copied into that frame with a recorded
   `vkCmdCopyBuffer2` and a barrier. nothing
   on the device divides, roots or derives geometry.
8. the plugin identity does not change. the Vulkan path is the same
   `xyz.n4o.nimages` plugin with the same five filters.

## prerequisites, already in the working tree

`Cargo.toml` and `Cargo.lock` are modified and uncommitted: `vapoursynth4-rs` is
pinned to `685c29d` with `vs-43`, `vs-vulkan` and `vsscript-44`. that raises the
floor from R79 to **R80** and turns on the GPU module. verified on this host:

- `cargo build --release --locked` succeeds.
- `cargo test --locked` passes.
- `uv sync --locked --extra dev --extra dev-tests` builds and installs the wheel.
- `tests/check-nimages.py` reports **2150 checks passed** against R80.

two loose ends belong to [M5](../FINDINGS.md): `vsscript-44` is
compiled in and never used, and the README still says R79. both are edits in the
same change that lands the dependency.

## sequencing

| plan | what it delivers | gate before the next |
| --- | --- | --- |
| [16](16-vulkan-validation.md) | the device parity and bench harness | a device run compares equal to `gpu=0` for a `std`-only chain |
| [17](17-vulkan-deblur.md) | `Deblur(gpu=1)` for both methods | fixture tolerance met, latency measured, aggregate measured |
| [18](18-vulkan-pointwise-maps.md) | `Levels` and `Posterize` as chain members | bit-exact against `gpu=0`, chain measured with and without |
| [19](19-vulkan-histogram-analyzers.md) | `PeakStats` and `PeakGrayShades` on the device | only if a device chain or a device source exists to amortise the upload |

## risks

- **the CPU's frame parallelism.** VapourSynth runs frames on 12 workers. a
  1 MP probe measured 9.48x reported work but each frame 3.7x slower and only
  2.7x aggregate throughput (`03-deblur-filter.md:564-581`). the honest baseline
  for a device `Deblur` is therefore an aggregate rate, not one frame's latency,
  and [16](16-vulkan-validation.md) measures both.
- **a driver's fast math.** MoltenVK rewrites a division into a reciprocal
  multiplication, and `precise` does not stop it: 124,652 of 470,083 8-bit inputs
  differed in the sibling's probe. the rule is to divide on the host and upload
  the result.
- **`f32` order.** the Deblur kernels are order-sensitive: taps accumulate in
  ascending order, multiply and add round separately, and the integer write is
  ties-to-even. a shader that contracts to FMA changes the bits. every such
  statement carries `precise`.
- **VRAM concurrency.** a 5806x4128 page needs two 96 MB `f32` ping-pong planes
  at 24 MP. the core's limit on this host is 2.35 GB, so roughly ten frames fit
  in flight, and the reservation makes the core throttle rather than fail.
- **device coverage.** Vulkan 1.4 excludes the Intel UHD on this host and any
  machine older than it. the README has to say so before `gpu=1` ships.

## what this plan does not do

it does not add a kernel, a dependency, a filter or an argument. it records the
verdict so the four plans below can be judged one at a time, the same way
[04](04-optimization-review.md) does for the CPU candidates.
