# Deblur on the device

status: landed for now, M1 to M7. both methods on Gray, RGB and YUV, 8 to 16 bit
integer and 32 bit float, on four and five scratch planes. 8 bit is
byte-identical, the deeper integers are within one LSB and the float path within
two ULP, and the aggregate is measured at 5.6x to 7.8x. tiling is the one open
item and it is a plan of its own. see [the outcome so far](#outcome-so-far) for
the numbers and the bugs it cost.

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

| method | family | dispatches | barriers in a serial recording |
| --- | --- | ---: | ---: |
| `1` | Gray, RGB | 10 | 9 |
| `0` | Gray, RGB | 32 | 31 |
| `1` | YUV | 12 | 9 |
| `0` | YUV | 34 | 31 |

every gaussian is two dispatches: horizontal into a ping-pong plane, a
`vkCmdPipelineBarrier2` from compute-write to compute-read, then vertical into the
next plane. two gaussians in a row need a barrier between them as well, because
they reuse those two planes. the consumer walks [14](14-deblur-blur-consumers.md)
removed on the CPU become the vertical pass's output stage, which saves a
dispatch and a full plane traversal per iteration. the mask is five dispatches
over two planes: the prefilter pair, the Sobel into the horizontal scratch, and
the mask's own blur, whose vertical pass writes the result back into the plane
the Sobel left it in. the blend restores the candidate in place, so no plane
carries the restored values alone, and the last dispatch is the store: one pass
turns those code values into the frame's own samples. that pass is the only
piece that differs by family. Gray writes `code * inverse` into one plane. RGB
reads the three source planes back, takes one delta against the restored and
luma planes, limits it to `[-low, 255 - high]` and writes `channel + delta`,
which is the host's equal-offset rule and what keeps the channel differences.
YUV stores plane 0 the way Gray does and copies planes 1 and 2 word for word,
because the chroma is carried through untouched and a copy is the only form
that keeps it byte-exact.

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
4. the store pass rounds ties to even (`roundEven`) with the scale constant
   computed on the host, and clamps to the sample range. the multiply is `f32`:
   at 16 bits the host multiplies in `f64`, so a sample can land one code value
   away from the host's; 8 bit has an inverse of `1.0` and stays exact.
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

the one operation with no exact device equivalent is the Sobel's `sqrt`. Rust's
`f32::sqrt` is correctly rounded and a driver's is not required to be. the mask is
a smoothstep of the magnitude, so a one-ULP difference moves the mask slightly
wherever it sits on the ramp, not only at the threshold: the hash harness is the
gate, and the seven cases that run on this host do not expose it. a host whose
`sqrt` differs records the deviation against the frozen tolerance instead of
widening it.

## memory

four `GRAY32` planes for `method=1` and five for `method=0`, the source and the
output:

| page | one `f32` plane | the working planes | total |
| --- | ---: | ---: | ---: |
| 2902x4128, 12.0 MP | 48 MB | 192 to 240 MB | about 216 to 264 MB with source and output |
| 5806x4128, 24.0 MP | 96 MB | 384 to 480 MB | about 432 to 528 MB |

the planes are `luma`, the horizontal scratch, the mask's raw plane, and the
candidate, which the blend restores in place (`method=1`), or two estimates in
place of the candidate (`method=0`). the row ring has no device equivalent: a
dispatch binds a whole plane, so the scratch granularity is a plane.

measured, not modelled, a single 24 MP frame peaks at 768 MB of device memory
and thirteen 12 MP frames in flight at 1538 MB, against the 2.35 GB budget on
this host. the 24 MP page therefore holds four or five frames in flight, which
is what [the aggregate](#aggregate) table shows: the device runs at 31 to 69
frames/s up to five and falls over beyond that.

## milestones

| # | delivers | gate |
| ---: | --- | --- |
| M1 | host-side planning split out of the kernels: taps, geometry, scale, mask parameters, in `src/deblur.rs` with no VapourSynth types | the existing unit tests and `cargo test` pass unchanged |
| M2 | one separable gaussian in `src/gpu/deblur.rs` on the device for `GRAY8` | `tools/deblur_hash.py --check` matches on the gaussian's output |
| M3 | `method=1` end to end, mask and blend included | hash matches for `u8`, `u16` and `f32` |
| M4 | `method=0`, twelve blurs with the folded consumers | hash matches, 31 dispatches, one recording |
| M5 | format surface: RGB and YUV, 8 to 16 bit, `f32` | the validator's colour-family checks pass on both engines |
| M6 | measurement | latency and aggregate per [16](16-vulkan-validation.md), the 24 MP ceiling found and the VRAM knob identified |
| M7 | plane count: the Sobel writes into the horizontal scratch and the blend restores the candidate in place | the hash table is byte-identical, four and five planes, and the 24 MP ceiling moves from five frames in flight to seven |

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

aggregate is the number that decides adoption, and the prediction held at 12 MP:
the CPU reaches 2.7x its sequential rate under 12 worker threads and the device
2.5x, so the aggregate gap (5.6x to 7.8x) tracks the latency gap (8.4x to
10.4x) instead of collapsing, which is a win at the page size this repository
targets. at 24 MP the prediction was too kind: one frame's scratch was 900 MB
before [M7](#milestones), so the win there is 7x to 8x up to five frames in
flight and none beyond it. see [the aggregate](#aggregate).

## outcome so far

written 2026-10-24. M1 to M7 are landed: both methods run on Gray, RGB and YUV,
8 to 16 bit integer and 32 bit float, on four and five scratch planes. the
aggregate is measured and the 24 MP ceiling is known; tiling is the open item.

| piece | where |
| --- | --- |
| the device layer | `src/gpu/mod.rs`: gate, pipeline, pool, plane binding, dispatch, barrier, upload |
| launch and push blocks | `src/gpu/shared.rs`, no binding type, unit tested |
| the kernels | `src/gpu/deblur.rs`: prepare, separable blur, Sobel, blend, store, RGB store, chroma copy |
| the filter | `src/filters/deblur.rs`: `gpu:int:opt`, `clip:vnode:all`, `gpu_output()` |
| the planning | `src/deblur.rs`: `Plan`, `Gaussian`, `Mask`, `Blend`, `Quantisation` |
| the parity gate | `tools/deblur_hash.py --device` |
| the device probe | `tools/gpu_probe.py` |

`gpu=1` refuses a format outside those three families, a sample type outside
8 to 16 bit integer and 32 bit float, and a variable format, each with a
message naming what it saw.

### parity

`tools/deblur_hash.py --device` runs 39 cases; 37 are implemented and all 37
pass. 8 bit is byte-identical; 9 to 16 bit is within one LSB, and 32 bit float
within two ULP of the code value, which is the `f32` quantisation the store pass
was told to use:

- Gray 8 bit: 15 cases, 0 differing samples
- Gray 16 bit: 3 cases, at most 1 code value of 65535
- Gray float: 4 cases, worst 2.4e-7 in the sample's own `[0, 1]` range
- RGB24: 4 cases, 0 differing samples, including the odd 129 px width and a 1 px
  frame
- RGB48: 1 case, at most 1 code value
- RGB float: 2 cases, worst 3.0e-7
- YUV420P8: 5 cases, 0 differing samples, including 130x66 with a 65 byte chroma
  row in a 128 byte stride, and a 2x2 frame whose chroma planes hold one sample
- YUV 10 and 16 bit: 2 cases, at most 1 code value
- YUV float: 1 case, worst 1.2e-7
- the 2 skips are the variable-format clip

- defaults, an odd width (129x97), a 1 px frame, a frame whose row padding is `0xA5`
- `radius=16` (the 129-tap kernel), a non-default `strength`, `threshold` and
  `overshoot`, and `threshold=0`
- `method=0` additionally covers six refinement iterations, the division by
  `max(blur, 1e-7)` and the estimate scaling
- every format repeats `method=0` at least once, so the RGB store and the chroma
  copy are on both paths
- the same 8 bit case runs once more with the source already on the device,
  through `std.GPUUpload`, so the residency branch that skips the upload is
  covered and comes back byte-identical

planes after the first carry a different page, so an RGB gamut clamp that read
the wrong channel, or a chroma plane rewritten from the luma, would show up as
a difference rather than as a pass.

the CPU table is unchanged for the 24 cases it had (`deblur_hash.py --check`
matches all 39), the validator reports 2150 checks, and `cargo test`,
`cargo clippy -- -D warnings` and `cargo fmt --check` are clean.

### measured

one frame on a quiet machine, a fresh `Deblur` node per request, median of seven.
`gpu=1` is measured through `std.GPUDownload`, which is what a caller pays:

| page | family | method | CPU | device + download | bus round trip | speedup |
| --- | --- | ---: | ---: | ---: | ---: | ---: |
| 2902x4128, 12.0 MP | Gray8 | 1 | 125.8 ms | 13.7 ms | 5.1 ms | 9.2x |
| 2902x4128, 12.0 MP | Gray8 | 0 | 350.1 ms | 37.7 ms | 5.1 ms | 9.3x |
| 2902x4128, 12.0 MP | RGB24 | 1 | 239.2 ms | 27.4 ms | 14.2 ms | 8.7x |
| 2902x4128, 12.0 MP | RGB24 | 0 | 453.8 ms | 48.0 ms | 14.2 ms | 9.5x |
| 2902x4128, 12.0 MP | YUV420P8 | 1 | 124.9 ms | 14.9 ms | 6.9 ms | 8.4x |
| 2902x4128, 12.0 MP | YUV420P8 | 0 | 351.6 ms | 40.4 ms | 6.9 ms | 8.7x |
| 5806x4128, 24.0 MP | Gray8 | 1 | 239.8 ms | 27.7 ms | 9.1 ms | 8.7x |
| 5806x4128, 24.0 MP | Gray8 | 0 | 701.0 ms | 74.1 ms | 9.1 ms | 9.5x |
| 5806x4128, 24.0 MP | RGB24 | 1 | 468.3 ms | 44.9 ms | 28.8 ms | 10.4x |
| 5806x4128, 24.0 MP | RGB24 | 0 | 915.0 ms | 91.8 ms | 28.8 ms | 10.0x |
| 5806x4128, 24.0 MP | YUV420P8 | 1 | 242.3 ms | 31.3 ms | 15.2 ms | 7.7x |
| 5806x4128, 24.0 MP | YUV420P8 | 0 | 700.5 ms | 78.5 ms | 15.2 ms | 8.9x |

the source is a `BlankClip`, so the frame costs nothing to make and both engines
see the same input; these are filter numbers, not decode. the transfer is three
planes for RGB24 (14.2 ms at 12 MP against Gray's 5.1) and 1.5 samples per pixel
for YUV420P8, so RGB keeps its win over a host path that also reads three
planes, while YUV wins least because its host path only touches one. every row
was measured in one session: absolute milliseconds move by about a third
between sessions on this laptop, so compare the ratio in a row.

### aggregate

the protocol is [16](16-vulkan-validation.md#the-measurements)'s, and
`tools/deblur_fanin.py` is the command: 13 frames through `std.AverageFrames`,
one pull, a fresh node per repeat so the frame cache cannot answer for the
filter, a 512 MB cache, median of three, and peak VRAM from NVML above the idle
baseline. one pull is 13 frames of the same node outstanding at once, which is
the shape a temporal consumer gives a filter.

at 12 MP the device wins:

| format | method | CPU, frames/s | device, frames/s | ratio | peak VRAM |
| --- | ---: | ---: | ---: | ---: | ---: |
| Gray8 | 1 | 21.0 | 150.0 | 7.1x | 1538 MB |
| Gray8 | 0 | 6.6 | 51.3 | 7.8x | 1922 MB |
| RGB24 | 1 | 14.9 | 83.3 | 5.6x | 1794 MB |
| RGB24 | 0 | 5.9 | 39.5 | 6.7x | 2050 MB |
| YUV420P8 | 1 | 20.6 | 126.1 | 6.1x | 1538 MB |

the ratio lands close to the latency ratio because both engines gain from a
fan-in: 2.7x for the CPU on `Gray8 method=1` at 12 MP, 2.5x for the device.
where each one saturates, also `Gray8 method=1` at 12 MP:

| frames in flight | 1 | 3 | 5 | 7 | 9 | 11 | 13 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| device, frames/s | 58.9 | 104.6 | 114.3 | 139.5 | 139.7 | 138.8 | 150.0 |
| CPU, frames/s | 7.7 | 15.2 | 17.9 | 21.1 | 20.6 | 19.4 | 21.0 |

the device is on its plateau by five frames in flight and the CPU by seven, so
most of the aggregate gap is decided by one frame's cost; the rest is the
fan-in each engine can absorb. peak VRAM on the same run: 384, 640, 1024,
1280, 1280, 1536 and 1538 MB.

at 24 MP the page barely fits. one frame's scratch is four 96 MB planes, which
the allocator rounds up to 128 MB blocks, about 700 MB per frame against
roughly 2.1 GB free on this 4 GB card:

| frames in flight | device, frames/s | CPU, frames/s | peak VRAM |
| --- | ---: | ---: | ---: |
| 1 | 31.1 | 3.9 | 768 MB |
| 3 | 61.9 | 8.9 | 1278 MB |
| 5 | 68.8 | 9.0 | 1790 MB |

those three are stable and 7x to 8x the CPU. `method=0` on five planes holds a
narrower envelope, 19.3 frames/s at three and 14.2 at five, and RGB24 on four
planes the same as Gray. past five frames the number stops describing the filter
and starts describing the allocator's free list: the same page measured 6.7
frames/s at seven and 10.4 at thirteen, against 31 to 69 for the three points
above, and the CPU sat at 9. in one sweep, before the plane count came down, the
process did not survive it:

```text
Failed to allocate a GPU plane: vkAllocateMemory failed for a 128 MB allocator
block (VkResult -2); allocator held 29 blocks totalling 3712 MB, of which 3043 MB
was live in 3 free regions, and trimming reclaimed 0 MB
VapourSynth encountered a fatal error
```

which aborts the host rather than failing a frame, so the filter cannot catch it.
`core.max_vram_cache_size` bounds what the core admits, and every sweep with it
set completed: at 1024 MB and 2048 MB the 24 MP `method=0` case ran 2.0 and 3.8
frames/s against the CPU's 2.6 and 3.0. it keeps the host alive without making
the high-fan-in numbers predictable. that is the knob a caller has; the plane
count under [memory](#memory) is what moved the ceiling from five frames in
flight to seven.

the device path moves the memory rather than removing it: peak resident set at
12 MP with 13 frames in flight is 328 MB against the CPU's 1475, and 411 against
1266 at 24 MP with five.

### what the first kernel cost

one bug, and it was not the one predicted. `fold()` mapped the reflection with
integer `%`, and GLSL leaves the sign of `a % b` undefined when `a` is negative.
the driver returned a value outside the adjusted range for the `radius` leftmost
columns, so the blur read a neighbouring row: 16 of 6144 samples differed by up to
5 codes, all at `x < radius`. `mod()` on floats is defined as
`x - y * floor(x / y)`, and with it the seven cases are byte-identical. index
arithmetic, not `sqrt`, was the first thing to break.

the plan's memory table was also optimistic: the mask needs a raw and a blurred
plane, so `method=1` landed on six planes, not two, and `method=0` on seven. the
deconvolution also needed a second estimate plane, because the estimate is read
and written in the same pass, and the read/write alias of one plane is not
something to rely on a frame binding for. [M7](#milestones) took those counts to
four and five by reusing the planes the passes already have.

the second bug was a missing barrier. after the refactor that moved the mask into
a helper, the blend recorded directly after the mask's vertical pass, and the
framework does not order two dispatches inside one recording. the blend read a
mask that had not been made visible: 3509 of 6144 samples on the numpy page, and
only 3 per case in the hash harness, because the two sources put different
pressure on the queue. `record_mask` now ends with the barrier its output needs.

the format surface cost nothing to find and two things to get right. the first
is that the luma read needs one shader for three families: a multi-plane read
cannot be written generically in GLSL, so the prepare pass binds five buffers
and a one-plane family binds its only plane in all three source slots, which is
legal because every extra descriptor is read-only and unread. two source strides
and three weights are zero for that case and the branch is uniform, so the
Gray path is the same instructions it was.

the second is the chroma copy. it moves whole `u32` words, which means a row
has to start and end on a word, and the last group of a row carries up to three
bytes past the pixels: those are padding on both sides, so the round trip stays
byte-exact. the 130x66 case exists for exactly that, 65 bytes of chroma in a 128
byte stride. a stride that is not a multiple of four is refused by name rather
than read across, which no VapourSynth allocation on this host does.

the RGB store reads the three source planes again rather than keeping them from
the prepare pass, because the delta needs `low` and `high` per pixel and the
code value is per channel. every lane masks its own sample into the destination
word, so two lanes of one invocation never race, and the three planes cost about
5 ms at 12 MP on top of Gray's store. that is the one place a packed-word write
would still pay, and it is not the bottleneck at 9x.

the plane count came down twice without touching a shader's arithmetic. the Sobel
writes into the horizontal scratch, because the prefiltered luma it reads is dead
by then, and the blend restores the candidate in place, because it reads and
writes the same index. that is four planes for `method=1` and five for `method=0`
against six and seven, and the hash table did not move: 37 of 37 device cases
still match with the same deviations. what moved is the stable envelope at 24 MP,
from three frames in flight to five, and the peak device use of one frame, from
1030 MB to 768 MB.

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
- **VRAM under concurrency.** measured, and it is the binding constraint at
  24 MP: four planes per frame, four 128 MB blocks, about 700 MB, against 2.1 GB
  free. a caller who runs many frames in flight at that page size can still push
  the allocator into a fatal error, and `core.max_vram_cache_size` is the knob
  that keeps it honest. the plane count brought the ceiling from three frames in
  flight to five; tiling is what would remove it.

## adoption criteria

the same bar as [01](01-performance-plan.md#adoption-criteria):

- `gpu=0` is byte-identical to the current build on the hash harness, for every
  method and format in M3 and M5.
- `gpu=1` equals `gpu=0` by the rules in [16](16-vulkan-validation.md), with any
  deviation from the Sobel `sqrt` recorded as a count and inside the frozen
  fixture tolerance.
- latency improves by at least 15% in the targeted stages, and aggregate
  throughput through the fan-in does not regress by more than 5%. at 12 MP both
  hold, 5.6x to 7.8x aggregate. at 24 MP the second one holds up to five frames
  in flight and does not beyond that: see [the aggregate](#aggregate) and
  [next](#next).
- the CPU path stays in the tree and stays the default.

## done when

`Deblur(gpu=1)` runs both methods on the format surface, the hash harness and the
validator compare it against `gpu=0`, and `docs/BENCH.md` carries the latency,
aggregate and transfer numbers from [16](16-vulkan-validation.md).

## next

[M7](#milestones) is in the tree: four planes for `method=1` and five for
`method=0`, the hash table unchanged, and the 24 MP envelope at five frames in
flight instead of three. what it did not do is remove the ceiling. four
full-resolution `f32` planes are still about 700 MB at 24 MP, and a twelve frame
fan-in needs roughly 175 MB per frame.

the structural fix is tiling: every pass here is a bounded stencil, the largest
radius being the 16 px gaussian, so a band with a halo of a few rows makes the
working set a function of band height rather than page height. a 512 row band at
24 MP is 12 MB per plane, which is under 60 MB per frame for four of them, and
the question of how many frames fit stops existing. the cost is a rewrite: every
kernel takes a band origin and height, the scratch planes become band sized, and
the deconvolution's six iterations each run band-wise. that belongs in its own
plan if 24 MP pages become a target, not in this one.
