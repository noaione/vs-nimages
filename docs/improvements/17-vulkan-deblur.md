# Deblur on the device

status: proposed. first kernel of the set in [15](15-vulkan-device-path.md),
gated by [16](16-vulkan-validation.md).

## current cost

`Deblur` is the only filter in this repository whose work per pixel is above the
host/device bus. measured on the `posterize-check` pages, one frame at a time:

| method | per page | stage split, 2048x2048 | shape |
| --- | ---: | --- | --- |
| `0` deconvolution | 408.0 ms | `restore` 133.3 of 155.7 ms | 14 gaussians, 28 passes |
| `1` unsharp | 147.8 ms | `restore` 28.1 of 46.0 ms | 3 gaussians, 6 passes |

the per-page column is the mixed page set in `docs/BENCH.md`. on a 12 MP page
`method=0` is about 360 ms and `method=1` about 100 ms, which is 30 and 8.4 ns per
pixel.

the split of that cost, from [03](03-deblur-filter.md#what-is-left):

| loop | work per pixel | measured |
| --- | --- | ---: |
| one sigma 0.8 blur, both passes | 27 mul/add, 14 loads, 7 stores | 5.3 ns |
| edge mask and blend | six blurs, two stencils, two passes | 26.5 ns |

the blur figure predates [05](05-horizontal-blur-dispatch.md) and
[06](06-vertical-register-accumulation.md), which took the separable blur down
2.2 to 3.0x and 1.29 to 1.41x. the shape is what matters here: 65% of
`method=0` is twelve blur passes and 78% of `method=1` is the mask and blend. both
are stencils over `f32` planes, which is what a compute shader is for.

`method=0` runs 12 candidate blurs at six iterations, plus the mask's two. the
mask is a Sobel magnitude over the luma, a threshold, then two gaussians of its
own, and the blend clamps the delta to the local gamut with a NaN-preserving
min/max chain. `u8` and `u16` samples are converted to `f32` on the way in and
quantized ties-to-even on the way out.

two facts from the CPU path that decide the device shape:

- the kernels are order-sensitive. taps accumulate in ascending order, multiply
  and add round separately, and the vector kernels in
  `src/deblur.rs` were accepted because they reproduce that order bit for bit.
- the workspace is five `f32` planes, and the horizontal scratch is a row ring
  sized for the largest kernel, `min(height, 129)` rows, with the active window
  `2r+1` ([11](11-streaming-blur-scratch.md)). the ring is a memory argument,
  not a numerical one.

## what moves

| stage | where | shape |
| --- | --- | --- |
| gaussian taps | host | `build_kernel`, uploaded once, at most 129 `f32` per axis |
| luma read and `f32` conversion | device | one pass over plane 0, integer or float |
| horizontal blur | device | one invocation per output sample, taps in ascending order |
| vertical blur | device | same, reads the horizontal plane |
| deconvolution consumers | device | folded into the vertical pass, the shape [14](14-deblur-blur-consumers.md) took on the CPU |
| Sobel, threshold, mask blurs | device | 3x3 stencil, then the same gaussian pipeline |
| blend and quantize | device | min/max chain, then ties-to-even |
| properties, frame allocation, debug | host | unchanged |

nothing divides, roots or derives geometry on the device. the taps, the
`255/max`-style scale constants, the `1e-7` floor and the pedestal are computed on
the host and either uploaded or carried in the push block.

## files

| file | role |
| --- | --- |
| `src/gpu/mod.rs` | new: gate, pipeline, pool, plane binding, dispatch, barrier, transfer |
| `src/gpu/shared.rs` | new: launch geometry, the tap loop, the GLSL preamble |
| `src/gpu/deblur.rs` | new: luma read, the blur pair, the mask, the blend |
| `src/filters/deblur.rs` | the `gpu` argument, `gpu_output()`, the transfer, and the call into the device module |
| `src/deblur.rs` | the planning the kernel takes, kept core-free |
| `src/lib.rs` | `mod gpu;` |

the device module owns the GLSL, the filter module owns the contract, and the
algorithm module owns the numbers. no kernel lands in `src/filters/`.

## the recording

one `pool.record` per frame, one command buffer, every dispatch in it:

| method | dispatches | barriers in a serial recording |
| --- | ---: | ---: |
| `1` | 9 | 8 |
| `0` | 31 | 30 |

every gaussian is two dispatches: horizontal into a ping-pong plane, a
`vkCmdPipelineBarrier2` from compute-write to compute-read, then vertical into the
next plane. two gaussians in a row need a barrier between them as well, because
they reuse those two planes. the consumer walks [14](14-deblur-blur-consumers.md)
removed on the CPU become the vertical pass's output stage, which saves a
dispatch and a full plane traversal per iteration.

the submission publishes the output plane's producer pair, so a later device
filter in the same graph waits on the timeline rather than on a host fence.

## exactness

the target is bit equality with the CPU build, not the fixture tolerance. the
rules that make it reachable:

1. taps are `f32`, computed on the host and uploaded, so no `exp`, `sqrt` or
   division happens on the device.
2. every multiply and add is a separate statement, and every one carries
   `precise`. a contracted `a * b + c` rounds once and changes the last bit.
3. the tap order is ascending, the same order `blur_row` and `accumulate_avx2`
   use.
4. the integer write uses `roundEven`, with the scale constant computed on the
   host at the same precision.
5. the clamp keeps the CPU's NaN semantics. GLSL `min`/`max` do not promise
   them, so the kernel selects on `isnan` the way `blend` does.
6. the Sobel divides by 8. the scalar loop divides and the AVX2 loop multiplies
   by `1/8`. 8 is a power of two, so both are exact, and either form reproduces
   them.
7. the RGB write keeps the equal-delta gamut clamp, and YUV keeps chroma byte for
   byte.

item 2 is where a driver's shader compiler can defeat the plan. SPIR-V's
`NoContraction` decoration, which `precise` emits, is the control, and the hash
harness in [16](16-vulkan-validation.md) is what proves it held.

the one statement with no exact device equivalent is the Sobel's `sqrt`. Rust's
`f32::sqrt` is correctly rounded and a driver's is not required to be. the kernel
compares the magnitude against a threshold, so a one-ULP difference changes the
mask only for a pixel whose magnitude lands exactly on the threshold. the port
measures the count of such pixels and reports it. the host's two kernels compute
`f32::sqrt` on both paths, so the count is expected to be zero, and the number
exists to prove it. a nonzero count is recorded against the frozen tolerance
rather than reconciled.

## memory

two `f32` ping-pong planes, the source and the output:

| page | one `f32` plane | ping-pong | total |
| --- | ---: | ---: | ---: |
| 2902x4128, 12.0 MP | 48 MB | 96 MB | about 120 MB with source and output |
| 5806x4128, 24.0 MP | 96 MB | 192 MB | about 240 MB |

the row ring has no device equivalent: a dispatch binds a whole plane, so the
scratch granularity is a plane. two `f32` planes are enough because the consumer
fusion removed the extra walks, which is why the device workspace is smaller than
the five-plane CPU one. the 2.35 GB limit on this host therefore holds roughly nine
24 MP frames in flight, and the core's reservation throttles concurrency rather
than failing an allocation.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | host-side planning split out of the kernels: taps, geometry, scale, mask parameters, in `src/deblur.rs` with no VapourSynth types | the existing unit tests and `cargo test` pass unchanged |
| M2 | one separable gaussian in `src/gpu/deblur.rs` on the device for `GRAY8` | `tools/deblur_hash.py --check` matches on the gaussian's output |
| M3 | `method=1` end to end, mask and blend included | hash matches for `u8`, `u16` and `f32` |
| M4 | `method=0`, twelve blurs with the folded consumers | hash matches, 31 dispatches, one recording |
| M5 | format surface: RGB and YUV, 8 to 16 bit, `f32` | the validator's colour-family checks pass on both engines |
| M6 | measurement | latency and aggregate per [16](16-vulkan-validation.md) |

M1 is worth its own milestone: `AGENTS.md` requires the algorithms to be testable
without a core, and a device kernel's host side is exactly the part that can be.

## expected

assumptions: 120 GB/s of realised device bandwidth on this part (of 168 GB/s
nominal), transfers at the measured 6.5 GB/s up and 3 GB/s down, and a 12 MP
`GRAY8` page.

| method | device passes | pass bytes | device time | transfers | total | CPU |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| `0` | 28 traversals plus mask and blend | 2.98 GB | 25 ms | 5.9 ms | 30 to 32 ms | 360 ms |
| `1` | 6 traversals plus mask and blend | 0.86 GB | 7.2 ms | 5.9 ms | 13 to 15 ms | 100 ms |

that is about 11 to 12x on `method=0` and about 7x on `method=1` in latency. the
estimate is a prediction, not a measurement, and the 120 GB/s assumption is the
part most likely to be wrong on a laptop part under load.

aggregate is the number that decides adoption. the CPU reaches 2.7x its sequential
rate under 12 worker threads, so its aggregate cost for `method=0` is about
133 ms per frame, not 360. the device submits one frame per recording and can
overlap independent frames only as far as VRAM allows, so the aggregate win is
smaller than the latency win and is the first thing M6 measures.

## risks

- **the Sobel `sqrt`.** the only non-portable operation, and the reason the
  report carries a count rather than a claim.
- **contraction.** `precise` is the control, but a driver can still reassociate
  unless the decoration survives. the hash harness is the gate, and it runs on
  every build that ships.
- **upload and download on a discrete host.** 5.9 ms of a 13 to 15 ms `method=1`
  frame is the bus. the filter is a win because the CPU path is 100 ms, not
  because the bus is cheap.
- **24 MP grids.** 24 MP at four pixels an invocation is 6.0 M invocations and
  93,750 groups of 64, over the 65,535 the API guarantees on x. every kernel
  takes a strided loop, and the 24 MP spread is a test case rather than an edge
  case.
- **`stride % 4`.** the sibling's kernels load `u32` words from byte-addressed
  planes and nothing checks the alignment. an 8-bit plane's stride is not
  guaranteed to be a multiple of four, so the input read is byte- or
   `u32`-shaped. a `u16` sample is two bytes picked out of a `u32` word in host
   endianness, and big-endian is refused, because `storageBuffer16BitAccess` is
   optional and the pipeline builder does not expose it. the `f32` planes start
   from an output the plugin allocates.
- **VRAM under concurrency.** nine frames at 24 MP on a 2.35 GB budget leaves no
  room for a second device consumer. the fan-in measurement reports what happens
  when the budget is reached.

## adoption criteria

the same bar as [01](01-performance-plan.md#adoption-criteria):

- `gpu=0` is byte-identical to the current build on the hash harness, for every
  method and format in M3 and M5.
- `gpu=1` equals `gpu=0` by the rules in [16](16-vulkan-validation.md), with any
  deviation from the Sobel `sqrt` recorded as a count and inside the frozen
  fixture tolerance.
- latency improves by at least 15% in the targeted stages, and aggregate
  throughput through the fan-in does not regress by more than 5%.
- the CPU path stays in the tree and stays the default.

## done when

`Deblur(gpu=1)` runs both methods on the format surface, the hash harness and the
validator compare it against `gpu=0`, and `docs/BENCH.md` carries the latency,
aggregate and transfer numbers from [16](16-vulkan-validation.md).
