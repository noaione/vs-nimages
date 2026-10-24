//! `Deblur` on the device: the unsharp chain, one recording per frame.
//!
//! The CPU path in [`crate::deblur`] is the definition. Every kernel here repeats
//! its arithmetic in the same order: taps ascending, multiply and add as separate
//! operations, the Sobel chain left to right, the blend's `min`/`max` over the
//! same nine samples, and the write rounded ties to even. `method=1` is the first
//! chain because it has no division in it, so the only operation without an exact
//! device equivalent is the Sobel's `sqrt`.

use std::mem::size_of;

use vapoursynth4_rs::ColorFamily;
use vapoursynth4_rs::SampleType;
use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::frame::{VideoFormat, VideoFrame};

use crate::deblur::{Method, Plan};
use crate::error::{NImagesError, Result};

use super::shared::{self, PREAMBLE, bytes_of, shader};
use super::{
    BarrierFn, Device, barrier_fn, compute_barrier, gpu_error, gray32_frame, upload_floats,
};

/// Storage buffers one prepare dispatch binds: the input plane and the `GRAY32`
/// luma it writes.
const PREPARE_BINDINGS: u32 = 2;
/// Storage buffers one blur dispatch binds: source, taps, destination, aux.
const BLUR_BINDINGS: u32 = 4;
/// Storage buffers one Sobel dispatch binds: the blurred luma and the mask.
const SOBEL_BINDINGS: u32 = 2;
/// Storage buffers one blend dispatch binds: luma, candidate, mask, output.
const BLEND_BINDINGS: u32 = 4;

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
    dst_stride_words: u32,
    scale: f32,
}

const _: () = assert!(size_of::<PreparePush>() == 7 * size_of::<u32>());

/// What the separable blur is told.
///
/// `mode` picks what the vertical pass leaves behind: `0` the blurred plane,
/// `1` the unsharp candidate, `2` the deconvolution ratio, `3` the estimate
/// scaled by the blurred ratio. The last two are unused until `method=0` lands.
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
    dst_stride_bytes: u32,
    inverse: f32,
    max_value: f32,
}

const _: () = assert!(size_of::<BlendPush>() == 12 * size_of::<u32>());

const PREPARE_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint width;
    uint height;
    uint src_stride_bytes;
    uint dst_stride_words;
    float scale;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) writeonly buffer Dst { float values[]; } dst;

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint x = unit % p.width;
        uint y = unit / p.width;
        uint byte = y * p.src_stride_bytes + x;
        uint word = src.words[byte >> 2u];
        uint code = (word >> (8u * (byte & 3u))) & 0xFFu;
        dst.values[y * p.dst_stride_words + x] = float(code) * p.scale;
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
    uint dst_stride_bytes;
    float inverse;
    float max_value;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 1) readonly buffer Candidate { float values[]; } candidate;
layout(std430, set = 0, binding = 2) readonly buffer Mask { float values[]; } mask;
layout(std430, set = 0, binding = 3) buffer Dst { uint words[]; } dst;

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
        uint word_at = (y * p.dst_stride_bytes + x0) / 4u;
        uint packed = dst.words[word_at];
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
            precise float limited = clamp_host(
                candidate.values[y * p.candidate_stride + x],
                low - p.overshoot,
                high + p.overshoot
            );
            precise float code = clamp_host(
                mm + mask.values[y * p.mask_stride + x] * (limited - mm),
                0.0,
                255.0
            );
            float rounded = clamp_host(roundEven(code * p.inverse), 0.0, p.max_value);
            uint shift = 8u * lane;
            packed = (packed & ~(0xFFu << shift)) | (uint(rounded) << shift);
        }
        dst.words[word_at] = packed;
    }
}
"#;

/// The device half of `Deblur`.
pub(crate) struct Engine {
    prepare: Device,
    blur: Device,
    sobel: Device,
    blend: Device,
    barrier: BarrierFn,
    gaussian: VideoFrame,
    prefilter: VideoFrame,
    mask: VideoFrame,
    plan: Plan,
}

impl Engine {
    /// Builds the four pipelines and uploads the three tap tables.
    ///
    /// Only the unsharp method and 8 bit Gray are implemented; both refusals name
    /// what was asked for, so a caller learns the limit at creation.
    pub(crate) fn new(core: &CoreRef, plan: Plan, format: &VideoFormat) -> Result<Self> {
        if plan.method != Method::EdgeSharpen {
            return Err(NImagesError::new(
                "Deblur gpu=1: method=0 is not implemented on the device yet",
            ));
        }
        if format.color_family != ColorFamily::Gray
            || format.sample_type != SampleType::Integer
            || format.bits_per_sample != 8
        {
            return Err(NImagesError::new(format!(
                "Deblur gpu=1: the device path takes 8 bit Gray, got {}",
                describe(format)
            )));
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

        let barrier = barrier_fn(core)?;
        let gaussian = upload_floats(core, plan.gaussian.taps())?;
        let prefilter = upload_floats(core, plan.prefilter.taps())?;
        let mask = upload_floats(core, plan.mask_blur.taps())?;

        Ok(Self {
            prepare,
            blur,
            sobel,
            blend,
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

        let mut luma = gray32_frame(core, width, height, None)?;
        let mut temp = gray32_frame(core, width, height, None)?;
        let mut candidate = gray32_frame(core, width, height, None)?;
        let mut mask_raw = gray32_frame(core, width, height, None)?;
        let mut mask = gray32_frame(core, width, height, None)?;

        let output_stride = output.stride(0);
        if output_stride < 0 || output_stride % 4 != 0 {
            return Err(NImagesError::new(
                "Deblur gpu=1: the output row stride is not a multiple of four bytes",
            ));
        }

        let columns = w.div_ceil(BLEND_LANES);
        let launch = shared::Launch::per_pixel(w, h);
        let blend_launch = shared::Launch::new(columns.saturating_mul(h));
        let groups = [launch.groups, 1, 1];
        let blend_groups = [blend_launch.groups, 1, 1];

        let luma_stride = words(&luma)?;
        let temp_stride = words(&temp)?;
        let candidate_stride = words(&candidate)?;
        let mask_strides = words(&mask_raw)?;
        let mask_blur_stride = words(&mask)?;

        let quantisation = self.plan.quantisation;
        let scale = quantisation.scale;
        let prepare = PreparePush {
            units: launch.units,
            stride: launch.stride(),
            width: w,
            height: h,
            src_stride_bytes: u32::try_from(input.stride(0)).unwrap_or(u32::MAX),
            dst_stride_words: luma_stride,
            scale,
        };

        let blur_push = |radius: usize,
                         taps: usize,
                         mode: u32,
                         direction: u32,
                         src_stride: u32,
                         dst_stride: u32,
                         aux_stride: u32| BlurPush {
            units: launch.units,
            stride: launch.stride(),
            width: w,
            height: h,
            radius: radius as u32,
            taps: taps as u32,
            mode,
            direction,
            strength: self.plan.strength,
            pedestal: self.plan.blend.pedestal,
            floor_value: self.plan.blend.division_floor,
            src_stride,
            dst_stride,
            aux_stride,
        };

        let gaussian = &self.gaussian;
        let prefilter = &self.prefilter;
        let mask_taps = &self.mask;
        let barrier = self.barrier;
        let plan = self.plan;

        let submit = self.blur.record(|exec| {
            // 1. widen the luma plane into code values.
            let src = exec.read_plane(input, 0)?;
            let dst = exec.write_plane(&mut luma, 0)?;
            exec.dispatch(
                self.prepare.pipeline(),
                &[src, dst],
                bytes_of(&prepare),
                groups,
            )?;
            compute_barrier(exec, barrier);

            // 2. the candidate's blur: horizontal, then vertical with the
            //    unsharp mix folded into the same pass.
            let push = blur_push(
                plan.gaussian.horizontal_radius(),
                plan.gaussian.count(),
                0,
                0,
                luma_stride,
                temp_stride,
                luma_stride,
            );
            let src = exec.read_plane(&luma, 0)?;
            let taps = exec.read_plane(gaussian, 0)?;
            let dst = exec.write_plane(&mut temp, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            let push = blur_push(
                plan.gaussian.vertical_radius(),
                plan.gaussian.count(),
                1,
                1,
                temp_stride,
                candidate_stride,
                luma_stride,
            );
            let src = exec.read_plane(&temp, 0)?;
            let taps = exec.read_plane(gaussian, 0)?;
            let dst = exec.write_plane(&mut candidate, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            // 3. the edge detector's prefilter, into `mask_raw`.
            let push = blur_push(
                plan.prefilter.horizontal_radius(),
                plan.prefilter.count(),
                0,
                0,
                luma_stride,
                temp_stride,
                luma_stride,
            );
            let src = exec.read_plane(&luma, 0)?;
            let taps = exec.read_plane(prefilter, 0)?;
            let dst = exec.write_plane(&mut temp, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            let push = blur_push(
                plan.prefilter.vertical_radius(),
                plan.prefilter.count(),
                0,
                1,
                temp_stride,
                mask_strides,
                luma_stride,
            );
            let src = exec.read_plane(&temp, 0)?;
            let taps = exec.read_plane(prefilter, 0)?;
            let dst = exec.write_plane(&mut mask_raw, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            // 4. the Sobel and its smoothstep, then the mask's own blur.
            let sobel_push = SobelPush {
                units: launch.units,
                stride: launch.stride(),
                width: w,
                height: h,
                threshold: plan.mask.threshold,
                ramp: plan.mask.ramp,
                span_inverse: 1.0 / plan.mask.sobel_span,
                src_stride: mask_strides,
                dst_stride: mask_blur_stride,
            };
            let src = exec.read_plane(&mask_raw, 0)?;
            let dst = exec.write_plane(&mut mask, 0)?;
            exec.dispatch(
                self.sobel.pipeline(),
                &[src, dst],
                bytes_of(&sobel_push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            let push = blur_push(
                plan.mask_blur.horizontal_radius(),
                plan.mask_blur.count(),
                0,
                0,
                mask_blur_stride,
                temp_stride,
                luma_stride,
            );
            let src = exec.read_plane(&mask, 0)?;
            let taps = exec.read_plane(mask_taps, 0)?;
            let dst = exec.write_plane(&mut temp, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            let push = blur_push(
                plan.mask_blur.vertical_radius(),
                plan.mask_blur.count(),
                0,
                1,
                temp_stride,
                mask_blur_stride,
                luma_stride,
            );
            let src = exec.read_plane(&temp, 0)?;
            let taps = exec.read_plane(mask_taps, 0)?;
            let dst = exec.write_plane(&mut mask, 0)?;
            let aux = exec.read_plane(&luma, 0)?;
            exec.dispatch(
                self.blur.pipeline(),
                &[src, taps, dst, aux],
                bytes_of(&push),
                groups,
            )?;
            compute_barrier(exec, barrier);

            // 5. limit the candidate to the local extremes and blend it back.
            let blend_push = BlendPush {
                units: blend_launch.units,
                stride: blend_launch.stride(),
                column_groups: columns,
                width: w,
                height: h,
                overshoot: plan.blend.overshoot,
                luma_stride,
                candidate_stride,
                mask_stride: mask_blur_stride,
                dst_stride_bytes: u32::try_from(output_stride).unwrap_or(u32::MAX),
                inverse: quantisation.inverse as f32,
                max_value: quantisation.max_value as f32,
            };
            let src = exec.read_plane(&luma, 0)?;
            let candidate_plane = exec.read_plane(&candidate, 0)?;
            let mask_plane = exec.read_plane(&mask, 0)?;
            let dst = exec.write_plane(&mut output, 0)?;
            exec.dispatch(
                self.blend.pipeline(),
                &[src, candidate_plane, mask_plane, dst],
                bytes_of(&blend_push),
                blend_groups,
            )?;
            Ok(())
        });

        submit.map_err(|error| gpu_error("could not record the Deblur device passes", &error))?;

        Ok(output)
    }
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
