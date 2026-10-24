//! `Deblur` on the device: both methods, one recording per frame.
//!
//! The CPU path in [`crate::deblur`] is the definition. Every kernel here repeats
//! its arithmetic in the same order: taps ascending, multiply and add as separate
//! operations, the Sobel chain left to right, the blend's `min`/`max` over the
//! same nine samples, and the write rounded ties to even. `method=0` adds the
//! ratio and the estimate passes, whose division and `sqrt` are the only
//! operations without an exact guarantee from a driver; `precise` and the hash
//! gate are what hold the rest of the order.

use std::mem::size_of;

use vapoursynth4_rs::ColorFamily;
use vapoursynth4_rs::SampleType;
use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::frame::{VideoFormat, VideoFrame};
use vapoursynth4_rs::vulkan::{ExecContext, VulkanError};

use crate::deblur::{Gaussian, Method, Plan};
use crate::error::{NImagesError, Result};

use super::shared::{self, PREAMBLE, bytes_of, shader};
use super::{
    BarrierFn, Device, barrier_fn, compute_barrier, gpu_error, gray32_frame, upload_floats,
};

/// Storage buffers one prepare dispatch binds: the input plane, the `GRAY32`
/// luma it writes, and the estimate either writes or stands in for.
const PREPARE_BINDINGS: u32 = 3;
/// Storage buffers one blur dispatch binds: source, taps, destination, aux.
const BLUR_BINDINGS: u32 = 4;
/// Storage buffers one Sobel dispatch binds: the blurred luma and the mask.
const SOBEL_BINDINGS: u32 = 2;
/// Storage buffers one blend dispatch binds: luma, candidate, mask, the restored
/// `GRAY32` plane it writes. the output samples are a separate pass.
const BLEND_BINDINGS: u32 = 4;
/// Storage buffers one store dispatch binds: the restored plane and the frame's
/// output plane.
const STORE_BINDINGS: u32 = 2;

/// Columns one blend invocation packs into one output word.
const BLEND_LANES: u32 = 4;

/// What the prepare kernel is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct PreparePush {
    units: u32,
    stride: u32,
    width: u32,
    height: u32,
    src_stride_bytes: u32,
    luma_stride: u32,
    estimate_stride: u32,
    src_bytes: u32,
    scale: f32,
    pedestal: f32,
    mode: u32,
    is_float: u32,
}

const _: () = assert!(size_of::<PreparePush>() == 12 * size_of::<u32>());

/// What the separable blur is told.
///
/// `mode` picks what the vertical pass leaves behind: `0` the blurred plane,
/// `1` the unsharp candidate, `2` the deconvolution ratio, `3` the estimate
/// scaled by the blurred ratio.
#[repr(C)]
#[derive(Clone, Copy)]
struct BlurPush {
    units: u32,
    stride: u32,
    width: u32,
    height: u32,
    radius: u32,
    taps: u32,
    mode: u32,
    direction: u32,
    strength: f32,
    pedestal: f32,
    floor_value: f32,
    src_stride: u32,
    dst_stride: u32,
    aux_stride: u32,
}

const _: () = assert!(size_of::<BlurPush>() == 14 * size_of::<u32>());

/// What the Sobel kernel is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct SobelPush {
    units: u32,
    stride: u32,
    width: u32,
    height: u32,
    threshold: f32,
    ramp: f32,
    span_inverse: f32,
    src_stride: u32,
    dst_stride: u32,
}

const _: () = assert!(size_of::<SobelPush>() == 9 * size_of::<u32>());

/// What the blend is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct BlendPush {
    units: u32,
    stride: u32,
    column_groups: u32,
    width: u32,
    height: u32,
    overshoot: f32,
    luma_stride: u32,
    candidate_stride: u32,
    mask_stride: u32,
    dst_stride_words: u32,
    strength: f32,
    pedestal: f32,
    mode: u32,
}

/// What the store kernel is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct StorePush {
    units: u32,
    stride: u32,
    column_groups: u32,
    width: u32,
    height: u32,
    src_stride_words: u32,
    dst_stride_bytes: u32,
    inverse: f32,
    max_value: f32,
    bytes: u32,
    is_float: u32,
}

const _: () = assert!(size_of::<StorePush>() == 11 * size_of::<u32>());
const _: () = assert!(size_of::<BlendPush>() == 13 * size_of::<u32>());

const PREPARE_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint width;
    uint height;
    uint src_stride_bytes;
    uint luma_stride;
    uint estimate_stride;
    uint src_bytes;
    float scale;
    float pedestal;
    uint mode;
    uint is_float;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) writeonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 2) writeonly buffer Estimate { float values[]; } estimate;

// One byte of the source plane. A byte offset is derived from the plane, not
// from a sample, so any row stride is addressable.
uint load_byte(uint offset) {
    uint word = src.words[offset >> 2u];
    return (word >> (8u * (offset & 3u))) & 0xFFu;
}

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint x = unit % p.width;
        uint y = unit / p.width;
        uint byte = y * p.src_stride_bytes + x * p.src_bytes;
        float value;
        if (p.is_float == 1u) {
            value = uintBitsToFloat(src.words[byte >> 2u]) * p.scale;
        } else if (p.src_bytes == 2u) {
            uint pair = load_byte(byte) | (load_byte(byte + 1u) << 8u);
            value = float(pair) * p.scale;
        } else {
            value = float(load_byte(byte)) * p.scale;
        }
        precise float code = value;
        luma.values[y * p.luma_stride + x] = code;
        if (p.mode == 1u) {
            estimate.values[y * p.estimate_stride + x] = code + p.pedestal;
        }
    }
}
"#;

const BLUR_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint width;
    uint height;
    uint radius;
    uint taps;
    uint mode;
    uint direction;
    float strength;
    float pedestal;
    float floor_value;
    uint src_stride;
    uint dst_stride;
    uint aux_stride;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { float values[]; } src;
layout(std430, set = 0, binding = 1) readonly buffer Taps { float values[]; } tap;
layout(std430, set = 0, binding = 2) writeonly buffer Dst { float values[]; } dst;
layout(std430, set = 0, binding = 3) readonly buffer Aux { float values[]; } aux;

// scipy's mode="reflect": the edge sample repeats.
uint fold(int value, int length) {
    int period = 2 * length;
    // GLSL leaves the sign of an integer % with a negative operand undefined.
    // mod() on floats is defined as x - y * floor(x / y), so an index below zero
    // lands in 0..period instead of on whatever the driver chose.
    int folded = int(mod(float(value), float(period)));
    return uint(folded < length ? folded : period - 1 - folded);
}

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint x = unit % p.width;
        uint y = unit / p.width;
        precise float sum = 0.0;
        for (uint index = 0u; index < p.taps; index++) {
            uint offset;
            if (p.direction == 0u) {
                uint at = fold(int(x) + int(index) - int(p.radius), int(p.width));
                offset = y * p.src_stride + at;
            } else {
                uint at = fold(int(y) + int(index) - int(p.radius), int(p.height));
                offset = at * p.src_stride + x;
            }
            sum = sum + src.values[offset] * tap.values[index];
        }
        uint at = y * p.dst_stride + x;
        if (p.mode == 0u) {
            dst.values[at] = sum;
        } else if (p.mode == 1u) {
            precise float value = aux.values[y * p.aux_stride + x];
            dst.values[at] = value + p.strength * (value - sum);
        } else if (p.mode == 2u) {
            precise float value = aux.values[y * p.aux_stride + x];
            dst.values[at] = (value + p.pedestal) / max(sum, p.floor_value);
        } else {
            dst.values[at] = aux.values[y * p.aux_stride + x] * sum;
        }
    }
}
"#;

const SOBEL_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint width;
    uint height;
    float threshold;
    float ramp;
    float span_inverse;
    uint src_stride;
    uint dst_stride;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { float values[]; } src;
layout(std430, set = 0, binding = 1) writeonly buffer Dst { float values[]; } dst;

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint x = unit % p.width;
        uint y = unit / p.width;
        uint left = x == 0u ? 0u : x - 1u;
        uint right = min(x + 1u, p.width - 1u);
        uint above = (y == 0u ? 0u : y - 1u) * p.src_stride;
        uint middle = y * p.src_stride;
        uint below = min(y + 1u, p.height - 1u) * p.src_stride;
        float tl = src.values[above + left];
        float tm = src.values[above + x];
        float tr = src.values[above + right];
        float ml = src.values[middle + left];
        float mr = src.values[middle + right];
        float bl = src.values[below + left];
        float bm = src.values[below + x];
        float br = src.values[below + right];
        precise float vertical = (bl + 2.0 * bm + br) - (tl + 2.0 * tm + tr);
        precise float horizontal = (tr + 2.0 * mr + br) - (tl + 2.0 * ml + bl);
        precise float magnitude = sqrt(vertical * vertical + horizontal * horizontal) * p.span_inverse;
        precise float progress = clamp((magnitude - p.threshold) / p.ramp, 0.0, 1.0);
        dst.values[middle + x] = progress * progress * (3.0 - 2.0 * progress);
    }
}
"#;

const BLEND_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint column_groups;
    uint width;
    uint height;
    float overshoot;
    uint luma_stride;
    uint candidate_stride;
    uint mask_stride;
    uint dst_stride_words;
    float strength;
    float pedestal;
    uint mode;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 1) readonly buffer Candidate { float values[]; } candidate;
layout(std430, set = 0, binding = 2) readonly buffer Mask { float values[]; } mask;
layout(std430, set = 0, binding = 3) writeonly buffer Dst { float values[]; } dst;

// The host's clamp: an unordered bound pair returns the low one, and a NaN
// passes through because neither comparison fires.
float clamp_host(float value, float low, float high) {
    if (high < low) {
        return low;
    }
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint group = unit % p.column_groups;
        uint y = unit / p.column_groups;
        uint x0 = group * 4u;
        for (uint lane = 0u; lane < 4u; lane++) {
            uint x = x0 + lane;
            if (x >= p.width) {
                break;
            }
            uint left = x == 0u ? 0u : x - 1u;
            uint right = min(x + 1u, p.width - 1u);
            uint above = (y == 0u ? 0u : y - 1u) * p.luma_stride;
            uint middle = y * p.luma_stride;
            uint below = min(y + 1u, p.height - 1u) * p.luma_stride;
            float tl = luma.values[above + left];
            float tm = luma.values[above + x];
            float tr = luma.values[above + right];
            float ml = luma.values[middle + left];
            float mm = luma.values[middle + x];
            float mr = luma.values[middle + right];
            float bl = luma.values[below + left];
            float bm = luma.values[below + x];
            float br = luma.values[below + right];
            precise float low = min(min(min(min(min(min(min(min(tl, tm), tr), ml), mm), mr), bl), bm), br);
            precise float high = max(max(max(max(max(max(max(max(tl, tm), tr), ml), mm), mr), bl), bm), br);
            // mode 0 reads the unsharp candidate; mode 1 is method=0, where the
            // candidate is the estimate adjusted towards the luma on the host.
            precise float candidate_value;
            if (p.mode == 0u) {
                candidate_value = candidate.values[y * p.candidate_stride + x];
            } else {
                precise float estimate = candidate.values[y * p.candidate_stride + x];
                candidate_value = mm + p.strength * ((estimate - p.pedestal) - mm);
            }
            precise float limited = clamp_host(
                candidate_value,
                low - p.overshoot,
                high + p.overshoot
            );
            precise float code = clamp_host(
                mm + mask.values[y * p.mask_stride + x] * (limited - mm),
                0.0,
                255.0
            );
            // The output samples are the store pass's business; this plane is
            // the restored luma in code values, 0 to 255.
            dst.values[y * p.dst_stride_words + x] = code;
        }
    }
}
"#;

const STORE_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint column_groups;
    uint width;
    uint height;
    uint src_stride_words;
    uint dst_stride_bytes;
    float inverse;
    float max_value;
    uint bytes;
    uint is_float;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { float values[]; } src;
layout(std430, set = 0, binding = 1) buffer Dst { uint words[]; } dst;

float clamp_host(float value, float low, float high) {
    if (high < low) {
        return low;
    }
    if (value < low) {
        return low;
    }
    if (value > high) {
        return high;
    }
    return value;
}

// One integer sample: the host rounds ties to even, clamps and truncates.
uint sample_of(uint x, uint y) {
    precise float code = src.values[y * p.src_stride_words + x];
    precise float scaled = code * p.inverse;
    precise float rounded = clamp_host(roundEven(scaled), 0.0, p.max_value);
    return uint(rounded);
}

void main() {
    uint words_per_row = p.dst_stride_bytes / 4u;
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint group = unit % p.column_groups;
        uint y = unit / p.column_groups;
        uint x0 = group * 4u;
        if (p.bytes == 1u) {
            uint word = y * words_per_row + x0 / 4u;
            uint packed = dst.words[word];
            for (uint lane = 0u; lane < 4u; lane++) {
                uint x = x0 + lane;
                if (x >= p.width) {
                    break;
                }
                uint shift = 8u * lane;
                packed = (packed & ~(0xFFu << shift)) | (sample_of(x, y) << shift);
            }
            dst.words[word] = packed;
        } else if (p.bytes == 2u) {
            // Four samples are two words, two samples each.
            uint base = y * words_per_row + x0 / 2u;
            for (uint pair = 0u; pair < 2u; pair++) {
                uint word = base + pair;
                uint packed = dst.words[word];
                for (uint lane = 0u; lane < 2u; lane++) {
                    uint x = x0 + pair * 2u + lane;
                    if (x >= p.width) {
                        break;
                    }
                    uint shift = 16u * lane;
                    packed = (packed & ~(0xFFFFu << shift)) | (sample_of(x, y) << shift);
                }
                dst.words[word] = packed;
            }
        } else {
            // 32 bit float samples: no rounding and no clamp, the host's write.
            for (uint lane = 0u; lane < 4u; lane++) {
                uint x = x0 + lane;
                if (x >= p.width) {
                    break;
                }
                precise float code = src.values[y * p.src_stride_words + x];
                dst.words[y * words_per_row + x] = floatBitsToUint(code * p.inverse);
            }
        }
    }
}
"#;

/// The device half of `Deblur`.
pub(crate) struct Engine {
    prepare: Device,
    blur: Device,
    sobel: Device,
    blend: Device,
    store: Device,
    barrier: BarrierFn,
    gaussian: VideoFrame,
    prefilter: VideoFrame,
    mask: VideoFrame,
    plan: Plan,
}

impl Engine {
    /// Builds the four pipelines and uploads the three tap tables.
    ///
    /// Only 8 bit Gray is implemented; the refusal names what was asked for, so
    /// a caller learns the limit at creation.
    pub(crate) fn new(core: &CoreRef, plan: Plan, format: &VideoFormat) -> Result<Self> {
        if format.color_family != ColorFamily::Gray {
            return Err(NImagesError::new(format!(
                "Deblur gpu=1: the device path takes Gray input, got {}",
                describe(format)
            )));
        }
        let supported = match format.sample_type {
            SampleType::Integer => (8..=16).contains(&format.bits_per_sample),
            SampleType::Float => format.bits_per_sample == 32,
        };
        if !supported {
            return Err(NImagesError::new(format!(
                "Deblur gpu=1: the device path takes 8 to 16 bit integer or 32 bit float, got {}",
                describe(format)
            )));
        }
        // The 16 bit read and the 32 bit float read address words on the host's
        // byte order; the sample pair is assembled little-endian.
        if cfg!(target_endian = "big") {
            return Err(NImagesError::new(
                "Deblur gpu=1: the device path is little-endian only",
            ));
        }

        let prepare = Device::new(
            core,
            &shader(&[PREAMBLE, PREPARE_BODY])?,
            PREPARE_BINDINGS,
            size_of::<PreparePush>() as u32,
        )?;
        let blur = Device::new(
            core,
            &shader(&[PREAMBLE, BLUR_BODY])?,
            BLUR_BINDINGS,
            size_of::<BlurPush>() as u32,
        )?;
        let sobel = Device::new(
            core,
            &shader(&[PREAMBLE, SOBEL_BODY])?,
            SOBEL_BINDINGS,
            size_of::<SobelPush>() as u32,
        )?;
        let blend = Device::new(
            core,
            &shader(&[PREAMBLE, BLEND_BODY])?,
            BLEND_BINDINGS,
            size_of::<BlendPush>() as u32,
        )?;
        let store = Device::new(
            core,
            &shader(&[PREAMBLE, STORE_BODY])?,
            STORE_BINDINGS,
            size_of::<StorePush>() as u32,
        )?;

        let barrier = barrier_fn(core)?;
        let gaussian = upload_floats(core, plan.gaussian.taps())?;
        let prefilter = upload_floats(core, plan.prefilter.taps())?;
        let mask = upload_floats(core, plan.mask_blur.taps())?;

        Ok(Self {
            prepare,
            blur,
            sobel,
            blend,
            store,
            barrier,
            gaussian,
            prefilter,
            mask,
            plan,
        })
    }

    /// Runs the chain and returns the device frame the filter hands back.
    pub(crate) fn render(&self, core: &CoreRef, input: &VideoFrame) -> Result<VideoFrame> {
        let width = input.frame_width(0);
        let height = input.frame_height(0);
        if width <= 0 || height <= 0 {
            return Ok(core.copy_frame(input));
        }
        let (w, h) = (
            u32::try_from(width).unwrap_or(u32::MAX),
            u32::try_from(height).unwrap_or(u32::MAX),
        );

        let format = input.get_video_format();
        let mut output = core
            .new_gpu_video_frame(format, width, height, Some(input))
            .ok_or_else(|| {
                NImagesError::new("Deblur gpu=1: the core could not allocate a device frame")
            })?;
        let output_stride = output.stride(0);
        if output_stride < 0 || output_stride % 4 != 0 {
            return Err(NImagesError::new(
                "Deblur gpu=1: the output row stride is not a multiple of four bytes",
            ));
        }

        let mut luma = gray32_frame(core, width, height, None)?;
        let mut temp = gray32_frame(core, width, height, None)?;
        // The Sobel writes `mask`, the mask's own blur writes `mask_raw`, and
        // the blend reads `mask_raw`. On the deconvolution path `mask_raw` also
        // holds the ratio until the mask pass overwrites it.
        let mut mask_raw = gray32_frame(core, width, height, None)?;
        let mut mask = gray32_frame(core, width, height, None)?;
        // The blend writes code values; the store pass turns them into the frame's
        // own samples, one output format per path.
        let mut restored = gray32_frame(core, width, height, None)?;
        let quantisation = self.plan.quantisation;
        let barrier = self.barrier;

        let launch = shared::Launch::per_pixel(w, h);
        let blend_launch = shared::Launch::new(w.div_ceil(BLEND_LANES).saturating_mul(h));
        let geometry = Geometry {
            width: w,
            height: h,
            units: launch.units,
            stride: launch.stride(),
            groups: [launch.groups, 1, 1],
            columns: w.div_ceil(BLEND_LANES),
            blend_units: blend_launch.units,
            blend_stride: blend_launch.stride(),
            blend_groups: [blend_launch.groups, 1, 1],
            luma_stride: words(&luma)?,
            temp_stride: words(&temp)?,
            mask_raw_stride: words(&mask_raw)?,
            mask_stride: words(&mask)?,
            restored_stride: words(&restored)?,
            output_stride: u32::try_from(output_stride).unwrap_or(u32::MAX),
            src_bytes: quantisation.bytes as u32,
            is_float: u32::from(quantisation.float),
        };

        let submit = match self.plan.method {
            Method::EdgeSharpen => {
                let mut candidate = gray32_frame(core, width, height, None)?;
                let candidate_stride = words(&candidate)?;
                let prepare = PreparePush {
                    units: geometry.units,
                    stride: geometry.stride,
                    width: w,
                    height: h,
                    src_stride_bytes: u32::try_from(input.stride(0)).unwrap_or(u32::MAX),
                    luma_stride: geometry.luma_stride,
                    estimate_stride: geometry.temp_stride,
                    scale: quantisation.scale,
                    pedestal: self.plan.blend.pedestal,
                    mode: 0,
                    src_bytes: geometry.src_bytes,
                    is_float: geometry.is_float,
                };
                self.blur.record(|exec| {
                    self.record_prepare(
                        exec,
                        input,
                        &mut luma,
                        None,
                        &temp,
                        &prepare,
                        geometry.groups,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_blur_pair(
                        exec,
                        &geometry,
                        &luma,
                        geometry.luma_stride,
                        &self.plan.gaussian,
                        &self.gaussian,
                        &mut temp,
                        &mut candidate,
                        candidate_stride,
                        1,
                        &luma,
                        geometry.luma_stride,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_mask(exec, &geometry, &luma, &mut temp, &mut mask_raw, &mut mask)?;
                    self.record_blend(
                        exec,
                        &geometry,
                        &luma,
                        &candidate,
                        candidate_stride,
                        &mask_raw,
                        &mut restored,
                        0,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_store(exec, &geometry, &restored, &mut output)
                })
            }
            Method::Deconvolution => {
                let mut estimate_a = gray32_frame(core, width, height, None)?;
                let mut estimate_b = gray32_frame(core, width, height, None)?;
                let a_stride = words(&estimate_a)?;
                let b_stride = words(&estimate_b)?;
                let iterations = self.plan.iterations;
                let prepare = PreparePush {
                    units: geometry.units,
                    stride: geometry.stride,
                    width: w,
                    height: h,
                    src_stride_bytes: u32::try_from(input.stride(0)).unwrap_or(u32::MAX),
                    luma_stride: geometry.luma_stride,
                    estimate_stride: a_stride,
                    scale: quantisation.scale,
                    pedestal: self.plan.blend.pedestal,
                    mode: 1,
                    src_bytes: geometry.src_bytes,
                    is_float: geometry.is_float,
                };
                self.blur.record(|exec| {
                    self.record_prepare(
                        exec,
                        input,
                        &mut luma,
                        Some(&mut estimate_a),
                        &temp,
                        &prepare,
                        geometry.groups,
                    )?;
                    compute_barrier(exec, barrier);

                    // Each iteration turns the estimate into the ratio, then
                    // scales the estimate by the ratio's blur. The second
                    // estimate plane breaks the read/write alias one plane would
                    // need.
                    let mut current_is_a = true;
                    for _ in 0..iterations {
                        let (old, old_stride, new, new_stride) = if current_is_a {
                            (&estimate_a, a_stride, &mut estimate_b, b_stride)
                        } else {
                            (&estimate_b, b_stride, &mut estimate_a, a_stride)
                        };
                        self.record_blur_pair(
                            exec,
                            &geometry,
                            old,
                            old_stride,
                            &self.plan.gaussian,
                            &self.gaussian,
                            &mut temp,
                            &mut mask_raw,
                            geometry.mask_raw_stride,
                            2,
                            &luma,
                            geometry.luma_stride,
                        )?;
                        compute_barrier(exec, barrier);
                        self.record_blur_pair(
                            exec,
                            &geometry,
                            &mask_raw,
                            geometry.mask_raw_stride,
                            &self.plan.gaussian,
                            &self.gaussian,
                            &mut temp,
                            new,
                            new_stride,
                            3,
                            old,
                            old_stride,
                        )?;
                        compute_barrier(exec, barrier);
                        current_is_a = !current_is_a;
                    }

                    let (candidate, candidate_stride) = if current_is_a {
                        (&estimate_a, a_stride)
                    } else {
                        (&estimate_b, b_stride)
                    };
                    self.record_mask(exec, &geometry, &luma, &mut temp, &mut mask_raw, &mut mask)?;
                    self.record_blend(
                        exec,
                        &geometry,
                        &luma,
                        candidate,
                        candidate_stride,
                        &mask_raw,
                        &mut restored,
                        1,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_store(exec, &geometry, &restored, &mut output)
                })
            }
        };

        submit.map_err(|error| gpu_error("could not record the Deblur device passes", &error))?;
        Ok(output)
    }

    /// Widens the input's luma into code values, and seeds the estimate on the
    /// deconvolution path.
    #[expect(
        clippy::too_many_arguments,
        reason = "the prepare binds the input, the luma, the optional estimate and the block"
    )]
    fn record_prepare(
        &self,
        exec: &mut ExecContext<'_>,
        input: &VideoFrame,
        luma: &mut VideoFrame,
        estimate: Option<&mut VideoFrame>,
        fallback: &VideoFrame,
        push: &PreparePush,
        groups: [u32; 3],
    ) -> std::result::Result<(), VulkanError> {
        let src = exec.read_plane(input, 0)?;
        let dst = exec.write_plane(luma, 0)?;
        // The third binding must exist even when nothing writes it, and it must
        // not be the frame this dispatch writes.
        let seed = match estimate {
            Some(frame) => exec.write_plane(frame, 0)?,
            None => exec.read_plane(fallback, 0)?,
        };
        exec.dispatch(
            self.prepare.pipeline(),
            &[src, dst, seed],
            bytes_of(push),
            groups,
        )
    }

    /// Records one separable gaussian: horizontal into the scratch plane, then
    /// the vertical pass with `mode`'s consumer fused into its output.
    #[expect(
        clippy::too_many_arguments,
        reason = "a separable pass names its source, table, scratch, destination and aux"
    )]
    fn record_blur_pair(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        source: &VideoFrame,
        source_stride: u32,
        kernel: &Gaussian,
        taps: &VideoFrame,
        temp: &mut VideoFrame,
        destination: &mut VideoFrame,
        destination_stride: u32,
        mode: u32,
        aux: &VideoFrame,
        aux_stride: u32,
    ) -> std::result::Result<(), VulkanError> {
        let horizontal = BlurPush {
            units: geometry.units,
            stride: geometry.stride,
            width: geometry.width,
            height: geometry.height,
            radius: kernel.horizontal_radius() as u32,
            taps: kernel.count() as u32,
            mode: 0,
            direction: 0,
            strength: self.plan.strength,
            pedestal: self.plan.blend.pedestal,
            floor_value: self.plan.blend.division_floor,
            src_stride: source_stride,
            dst_stride: geometry.temp_stride,
            aux_stride,
        };
        let (src, tap, dst, aux_plane) = (
            exec.read_plane(source, 0)?,
            exec.read_plane(taps, 0)?,
            exec.write_plane(temp, 0)?,
            exec.read_plane(aux, 0)?,
        );
        exec.dispatch(
            self.blur.pipeline(),
            &[src, tap, dst, aux_plane],
            bytes_of(&horizontal),
            geometry.groups,
        )?;
        compute_barrier(exec, self.barrier);

        let vertical = BlurPush {
            mode,
            direction: 1,
            radius: kernel.vertical_radius() as u32,
            src_stride: geometry.temp_stride,
            dst_stride: destination_stride,
            ..horizontal
        };
        let (src, tap, dst, aux_plane) = (
            exec.read_plane(temp, 0)?,
            exec.read_plane(taps, 0)?,
            exec.write_plane(destination, 0)?,
            exec.read_plane(aux, 0)?,
        );
        exec.dispatch(
            self.blur.pipeline(),
            &[src, tap, dst, aux_plane],
            bytes_of(&vertical),
            geometry.groups,
        )
    }

    /// The edge mask: the prefilter, the Sobel and smoothstep, then the mask's
    /// own blur.
    fn record_mask(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        luma: &VideoFrame,
        temp: &mut VideoFrame,
        mask_raw: &mut VideoFrame,
        mask: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        self.record_blur_pair(
            exec,
            geometry,
            luma,
            geometry.luma_stride,
            &self.plan.prefilter,
            &self.prefilter,
            temp,
            mask_raw,
            geometry.mask_raw_stride,
            0,
            luma,
            geometry.luma_stride,
        )?;
        compute_barrier(exec, self.barrier);

        let push = SobelPush {
            units: geometry.units,
            stride: geometry.stride,
            width: geometry.width,
            height: geometry.height,
            threshold: self.plan.mask.threshold,
            ramp: self.plan.mask.ramp,
            span_inverse: 1.0 / self.plan.mask.sobel_span,
            src_stride: geometry.mask_raw_stride,
            dst_stride: geometry.mask_stride,
        };
        let src = exec.read_plane(mask_raw, 0)?;
        let dst = exec.write_plane(mask, 0)?;
        exec.dispatch(
            self.sobel.pipeline(),
            &[src, dst],
            bytes_of(&push),
            geometry.groups,
        )?;
        compute_barrier(exec, self.barrier);

        self.record_blur_pair(
            exec,
            geometry,
            mask,
            geometry.mask_stride,
            &self.plan.mask_blur,
            &self.mask,
            temp,
            mask_raw,
            geometry.mask_raw_stride,
            0,
            luma,
            geometry.luma_stride,
        )?;
        // The blend reads what this last vertical wrote, so the mask's output
        // has to be visible before it records.
        compute_barrier(exec, self.barrier);
        Ok(())
    }

    /// Limits the candidate to the local extremes and blends it back.
    #[expect(
        clippy::too_many_arguments,
        reason = "the blend binds five frames and the candidate's row stride"
    )]
    fn record_blend(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        luma: &VideoFrame,
        candidate: &VideoFrame,
        candidate_stride: u32,
        mask: &VideoFrame,
        output: &mut VideoFrame,
        mode: u32,
    ) -> std::result::Result<(), VulkanError> {
        let push = BlendPush {
            units: geometry.blend_units,
            stride: geometry.blend_stride,
            column_groups: geometry.columns,
            width: geometry.width,
            height: geometry.height,
            overshoot: self.plan.blend.overshoot,
            luma_stride: geometry.luma_stride,
            candidate_stride,
            mask_stride: geometry.mask_raw_stride,
            dst_stride_words: geometry.restored_stride,
            strength: self.plan.strength,
            pedestal: self.plan.blend.pedestal,
            mode,
        };
        let src = exec.read_plane(luma, 0)?;
        let candidate_plane = exec.read_plane(candidate, 0)?;
        let mask_plane = exec.read_plane(mask, 0)?;
        let dst = exec.write_plane(output, 0)?;
        exec.dispatch(
            self.blend.pipeline(),
            &[src, candidate_plane, mask_plane, dst],
            bytes_of(&push),
            geometry.blend_groups,
        )
    }

    /// Writes the restored code values as the frame's own samples.
    fn record_store(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        restored: &VideoFrame,
        output: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        let quantisation = self.plan.quantisation;
        let push = StorePush {
            units: geometry.blend_units,
            stride: geometry.blend_stride,
            column_groups: geometry.columns,
            width: geometry.width,
            height: geometry.height,
            src_stride_words: geometry.restored_stride,
            dst_stride_bytes: geometry.output_stride,
            inverse: quantisation.inverse as f32,
            max_value: quantisation.max_value as f32,
            bytes: quantisation.bytes as u32,
            is_float: u32::from(quantisation.float),
        };
        let src = exec.read_plane(restored, 0)?;
        let dst = exec.write_plane(output, 0)?;
        exec.dispatch(
            self.store.pipeline(),
            &[src, dst],
            bytes_of(&push),
            geometry.blend_groups,
        )
    }
}

/// The frame's own geometry, resolved once and handed to every pass.
struct Geometry {
    width: u32,
    height: u32,
    /// Work units and the grid a per-pixel pass runs them with.
    units: u32,
    stride: u32,
    groups: [u32; 3],
    /// Columns per blend invocation, and the blend's own grid.
    columns: u32,
    blend_units: u32,
    blend_stride: u32,
    blend_groups: [u32; 3],
    /// Row strides, in `f32` words except the output's bytes.
    luma_stride: u32,
    temp_stride: u32,
    mask_raw_stride: u32,
    mask_stride: u32,
    restored_stride: u32,
    output_stride: u32,
    /// The output format: bytes per sample, and whether it is a float.
    src_bytes: u32,
    is_float: u32,
}

/// A row stride in `f32` words, for a `GRAY32` plane.
fn words(frame: &VideoFrame) -> Result<u32> {
    let stride = frame.stride(0);
    if stride < 0 || stride % 4 != 0 {
        return Err(NImagesError::new(
            "Deblur gpu=1: a working row stride is not a multiple of four bytes",
        ));
    }
    Ok(u32::try_from(stride / 4).unwrap_or(u32::MAX))
}

/// Names a format from its fields, because the core's own name carries padding.
fn describe(format: &VideoFormat) -> String {
    let family = match format.color_family {
        ColorFamily::Gray => "Gray",
        ColorFamily::RGB => "RGB",
        ColorFamily::YUV => "YUV",
        _ => "Undefined",
    };
    let kind = match format.sample_type {
        SampleType::Integer => "integer",
        SampleType::Float => "float",
    };
    format!("{family} {kind} {} bit", format.bits_per_sample)
}
