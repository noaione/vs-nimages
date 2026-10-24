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

use crate::deblur::{Gaussian, LUMA_WEIGHTS, Method, Plan};
use crate::error::{NImagesError, Result};

use super::shared::{self, PREAMBLE, bytes_of, shader};
use super::{
    BarrierFn, Device, barrier_fn, compute_barrier, gpu_error, gray32_frame, upload_floats,
};

/// Storage buffers one prepare dispatch binds: the three source planes, the
/// `GRAY32` luma it writes, and the estimate either writes or stands in for.
///
/// A one-plane family binds its only plane in all three source slots and reads
/// only the first, so the kernel is one shader for Gray, YUV and RGB.
const PREPARE_BINDINGS: u32 = 5;
/// Storage buffers one blur dispatch binds: source, taps, destination, aux.
const BLUR_BINDINGS: u32 = 4;
/// Storage buffers one Sobel dispatch binds: the blurred luma and the mask.
const SOBEL_BINDINGS: u32 = 2;
/// Storage buffers one blend dispatch binds: luma, the candidate it restores in
/// place, and the mask. the output samples are a separate pass.
const BLEND_BINDINGS: u32 = 3;
/// Storage buffers one store dispatch binds: the restored plane and the frame's
/// output plane.
const STORE_BINDINGS: u32 = 2;
/// Storage buffers one RGB store dispatch binds: the three source planes, the
/// luma and restored planes the delta is taken from, and the three output
/// planes.
const STORE_RGB_BINDINGS: u32 = 8;
/// Storage buffers one copy dispatch binds: the plane read and the plane
/// written.
const COPY_BINDINGS: u32 = 2;

/// Columns one blend invocation packs into one output word.
const BLEND_LANES: u32 = 4;

/// What the prepare kernel is told.
///
/// `channels` is three for RGB, where the luma is the weighted sum of the three
/// planes, and one otherwise, where plane 0 is the luma on its own.
#[repr(C)]
#[derive(Clone, Copy)]
struct PreparePush {
    units: u32,
    stride: u32,
    width: u32,
    height: u32,
    src0_stride: u32,
    src1_stride: u32,
    src2_stride: u32,
    luma_stride: u32,
    estimate_stride: u32,
    src_bytes: u32,
    channels: u32,
    mode: u32,
    is_float: u32,
    scale: f32,
    pedestal: f32,
    weight0: f32,
    weight1: f32,
    weight2: f32,
}

const _: () = assert!(size_of::<PreparePush>() == 18 * size_of::<u32>());
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
const _: () = assert!(size_of::<BlendPush>() == 12 * size_of::<u32>());

/// What the RGB store kernel is told.
///
/// The host applies one delta to all three channels, limited to the gamut the
/// pixel has left, so this store reads the three source planes as well as the
/// luma and restored planes the delta comes from.
#[repr(C)]
#[derive(Clone, Copy)]
struct StoreRgbPush {
    units: u32,
    stride: u32,
    column_groups: u32,
    width: u32,
    height: u32,
    src0_stride: u32,
    src1_stride: u32,
    src2_stride: u32,
    luma_stride: u32,
    restored_stride: u32,
    dst0_stride: u32,
    dst1_stride: u32,
    dst2_stride: u32,
    scale: f32,
    inverse: f32,
    max_value: f32,
    src_bytes: u32,
    is_float: u32,
}

const _: () = assert!(size_of::<StoreRgbPush>() == 18 * size_of::<u32>());

/// What the chroma copy is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct CopyPush {
    units: u32,
    stride: u32,
    columns: u32,
    src_stride: u32,
    dst_stride: u32,
}

const _: () = assert!(size_of::<CopyPush>() == 5 * size_of::<u32>());

/// The host's clamp, which the blend and both stores share.
///
/// An unordered bound pair returns the low one, and a NaN passes through
/// because neither comparison fires.
const HOST_HELPERS: &str = r#"
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
"#;

/// The sample readers a kernel that binds source planes declares.
///
/// A GLSL function cannot take a buffer, so the reader is a macro expanded once
/// per binding. The byte offset comes from the plane's own row stride, so row
/// padding is addressable, and a two byte sample is assembled little-endian,
/// which is the order `Convert` reads it in. `p.src_bytes` and `p.is_float` are
/// the same for every lane of a dispatch, so the branch is uniform.
const SAMPLE_READERS: &str = r#"
#define READ_SAMPLE(NAME, BYTE, SRC) \
uint BYTE(uint offset) { \
    uint word = SRC.words[offset >> 2u]; \
    return (word >> (8u * (offset & 3u))) & 0xFFu; \
} \
float NAME(uint offset) { \
    if (p.is_float == 1u) { \
        return uintBitsToFloat(SRC.words[offset >> 2u]); \
    } \
    if (p.src_bytes == 2u) { \
        return float(BYTE(offset) | (BYTE(offset + 1u) << 8u)); \
    } \
    return float(BYTE(offset)); \
}
"#;

const PREPARE_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint width;
    uint height;
    uint src0_stride;
    uint src1_stride;
    uint src2_stride;
    uint luma_stride;
    uint estimate_stride;
    uint src_bytes;
    uint channels;
    uint mode;
    uint is_float;
    float scale;
    float pedestal;
    float weight0;
    float weight1;
    float weight2;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src0 { uint words[]; } src0;
layout(std430, set = 0, binding = 1) readonly buffer Src1 { uint words[]; } src1;
layout(std430, set = 0, binding = 2) readonly buffer Src2 { uint words[]; } src2;
layout(std430, set = 0, binding = 3) writeonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 4) writeonly buffer Estimate { float values[]; } estimate;

READ_SAMPLE(read0, read0Byte, src0)
READ_SAMPLE(read1, read1Byte, src1)
READ_SAMPLE(read2, read2Byte, src2)

void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint x = unit % p.width;
        uint y = unit / p.width;
        uint column = x * p.src_bytes;
        precise float value = read0(y * p.src0_stride + column) * p.scale;
        if (p.channels == 3u) {
            // The host reads the three planes in this order, one multiply and
            // one add at a time, so the weighted sum is reproduced rather than
            // left to whatever association the driver picks.
            precise float green = read1(y * p.src1_stride + column) * p.scale;
            precise float blue = read2(y * p.src2_stride + column) * p.scale;
            precise float weighted0 = value * p.weight0;
            precise float weighted1 = green * p.weight1;
            precise float weighted2 = blue * p.weight2;
            precise float sum = weighted0 + weighted1;
            value = sum + weighted2;
        }
        luma.values[y * p.luma_stride + x] = value;
        if (p.mode == 1u) {
            estimate.values[y * p.estimate_stride + x] = value + p.pedestal;
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
    float strength;
    float pedestal;
    uint mode;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 1) buffer Candidate { float values[]; } candidate;
layout(std430, set = 0, binding = 2) readonly buffer Mask { float values[]; } mask;

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
            // The blend restores the candidate in place; the store pass turns
            // those code values into the frame's own samples.
            candidate.values[y * p.candidate_stride + x] = code;
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

const STORE_RGB_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint column_groups;
    uint width;
    uint height;
    uint src0_stride;
    uint src1_stride;
    uint src2_stride;
    uint luma_stride;
    uint restored_stride;
    uint dst0_stride;
    uint dst1_stride;
    uint dst2_stride;
    float scale;
    float inverse;
    float max_value;
    uint src_bytes;
    uint is_float;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src0 { uint words[]; } src0;
layout(std430, set = 0, binding = 1) readonly buffer Src1 { uint words[]; } src1;
layout(std430, set = 0, binding = 2) readonly buffer Src2 { uint words[]; } src2;
layout(std430, set = 0, binding = 3) readonly buffer Luma { float values[]; } luma;
layout(std430, set = 0, binding = 4) readonly buffer Restored { float values[]; } restored;
layout(std430, set = 0, binding = 5) buffer Dst0 { uint words[]; } dst0;
layout(std430, set = 0, binding = 6) buffer Dst1 { uint words[]; } dst1;
layout(std430, set = 0, binding = 7) buffer Dst2 { uint words[]; } dst2;

READ_SAMPLE(read0, read0Byte, src0)
READ_SAMPLE(read1, read1Byte, src1)
READ_SAMPLE(read2, read2Byte, src2)

// One sample written into a plane. A lane touches only its own sample's bytes,
// so the two lanes that share a word never race.
#define STORE_SAMPLE(NAME, DST, STRIDE) \
void NAME(uint x, uint y, float code) { \
    if (p.is_float == 1u) { \
        uint at = (y * STRIDE + x * 4u) >> 2u; \
        DST.words[at] = floatBitsToUint(code * p.inverse); \
        return; \
    } \
    precise float scaled = code * p.inverse; \
    precise float rounded = clamp_host(roundEven(scaled), 0.0, p.max_value); \
    uint value = uint(rounded); \
    uint offset = y * STRIDE + x * p.src_bytes; \
    uint word = offset >> 2u; \
    uint shift = 8u * (offset & 3u); \
    uint mask = p.src_bytes == 1u ? 0xFFu : 0xFFFFu; \
    uint packed = DST.words[word]; \
    DST.words[word] = (packed & ~(mask << shift)) | (value << shift); \
}

STORE_SAMPLE(write0, dst0, p.dst0_stride)
STORE_SAMPLE(write1, dst1, p.dst1_stride)
STORE_SAMPLE(write2, dst2, p.dst2_stride)

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
            uint column = x * p.src_bytes;
            precise float red = read0(y * p.src0_stride + column) * p.scale;
            precise float green = read1(y * p.src1_stride + column) * p.scale;
            precise float blue = read2(y * p.src2_stride + column) * p.scale;
            precise float low = min(min(red, green), blue);
            precise float high = max(max(red, green), blue);
            // The host's delta: one offset for the three channels, limited to
            // the gamut the pixel has left, which is what keeps the channel
            // differences instead of clamping each channel on its own.
            precise float delta = clamp_host(
                restored.values[y * p.restored_stride + x]
                    - luma.values[y * p.luma_stride + x],
                -low,
                255.0 - high
            );
            precise float restored0 = red + delta;
            precise float restored1 = green + delta;
            precise float restored2 = blue + delta;
            write0(x, y, restored0);
            write1(x, y, restored1);
            write2(x, y, restored2);
        }
    }
}
"#;

const COPY_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint columns;
    uint src_stride;
    uint dst_stride;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) writeonly buffer Dst { uint words[]; } dst;

// Copies one four byte group of one row. Both strides are multiples of four
// and at least the row's width, so a row starts and ends on a word and the
// bytes the last group carries past the pixels are padding on both sides.
void main() {
    for (uint unit = gl_GlobalInvocationID.x; unit < p.units; unit += p.stride) {
        uint column = unit % p.columns;
        uint y = unit / p.columns;
        uint source = (y * p.src_stride + column * 4u) >> 2u;
        uint destination = (y * p.dst_stride + column * 4u) >> 2u;
        dst.words[destination] = src.words[source];
    }
}
"#;

/// Which pass turns the restored code values into the frame's own samples.
///
/// Gray stores one plane, RGB stores three from one delta limited to the gamut
/// each pixel has left, and YUV stores the luma plane and copies the chroma
/// planes through untouched.
enum Output {
    Gray(Device),
    Rgb(Device),
    /// The chroma copy is boxed so one instance carries both pipelines without
    /// the enum taking the larger of the two shapes.
    Yuv(Device, Box<Device>),
}

/// One chroma plane carried through from the input to the output.
#[derive(Clone, Copy)]
struct PlaneCopy {
    /// Four byte groups per row.
    columns: u32,
    src_stride: u32,
    dst_stride: u32,
    launch: shared::Launch,
}

/// The device half of `Deblur`.
pub(crate) struct Engine {
    prepare: Device,
    blur: Device,
    sobel: Device,
    blend: Device,
    output: Output,
    barrier: BarrierFn,
    gaussian: VideoFrame,
    prefilter: VideoFrame,
    mask: VideoFrame,
    plan: Plan,
    family: ColorFamily,
}

impl Engine {
    /// Builds the pipelines this format needs and uploads the three tap tables.
    ///
    /// Gray, RGB and YUV 8 to 16 bit integers and their 32 bit float formats are
    /// implemented; the refusal names what was asked for, so a caller learns the
    /// limit at creation.
    pub(crate) fn new(core: &CoreRef, plan: Plan, format: &VideoFormat) -> Result<Self> {
        let family = match format.color_family {
            ColorFamily::Gray | ColorFamily::RGB | ColorFamily::YUV => format.color_family,
            _ => {
                return Err(NImagesError::new(format!(
                    "Deblur gpu=1: the device path takes Gray, RGB or YUV input, got {}",
                    describe(format)
                )));
            }
        };
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
            &shader(&[PREAMBLE, SAMPLE_READERS, PREPARE_BODY])?,
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
            &shader(&[PREAMBLE, HOST_HELPERS, BLEND_BODY])?,
            BLEND_BINDINGS,
            size_of::<BlendPush>() as u32,
        )?;
        // The store is the one pass that differs by family: one plane, three
        // planes from one delta, or a luma plane and two copied chroma planes.
        let output = match family {
            ColorFamily::RGB => Output::Rgb(Device::new(
                core,
                &shader(&[PREAMBLE, HOST_HELPERS, SAMPLE_READERS, STORE_RGB_BODY])?,
                STORE_RGB_BINDINGS,
                size_of::<StoreRgbPush>() as u32,
            )?),
            ColorFamily::YUV => Output::Yuv(
                Device::new(
                    core,
                    &shader(&[PREAMBLE, HOST_HELPERS, STORE_BODY])?,
                    STORE_BINDINGS,
                    size_of::<StorePush>() as u32,
                )?,
                Box::new(Device::new(
                    core,
                    &shader(&[PREAMBLE, COPY_BODY])?,
                    COPY_BINDINGS,
                    size_of::<CopyPush>() as u32,
                )?),
            ),
            _ => Output::Gray(Device::new(
                core,
                &shader(&[PREAMBLE, HOST_HELPERS, STORE_BODY])?,
                STORE_BINDINGS,
                size_of::<StorePush>() as u32,
            )?),
        };

        let barrier = barrier_fn(core)?;
        let gaussian = upload_floats(core, plan.gaussian.taps())?;
        let prefilter = upload_floats(core, plan.prefilter.taps())?;
        let mask = upload_floats(core, plan.mask_blur.taps())?;

        Ok(Self {
            prepare,
            blur,
            sobel,
            blend,
            output,
            barrier,
            gaussian,
            prefilter,
            mask,
            plan,
            family,
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
        let output_stride = output_plane_stride(&output, 0)?;

        let mut luma = gray32_frame(core, width, height, None)?;
        let mut temp = gray32_frame(core, width, height, None)?;
        // Two scratch planes carry the mask: the Sobel writes into the
        // horizontal scratch and the mask's own blur moves between them. On the
        // deconvolution path `mask_raw` also holds the ratio until then.
        let mut mask_raw = gray32_frame(core, width, height, None)?;
        // The blend restores the candidate in place, so there is no separate
        // restored plane to allocate.
        let quantisation = self.plan.quantisation;
        let barrier = self.barrier;
        let family = self.family;
        // RGB synthesises the luma from three weighted planes; every other
        // family reads plane 0 on its own and leaves the weights at one.
        let (channels, weights) = if family == ColorFamily::RGB {
            (3, LUMA_WEIGHTS)
        } else {
            (1, [1.0; 3])
        };

        let launch = shared::Launch::per_pixel(w, h);
        let blend_launch = shared::Launch::new(w.div_ceil(BLEND_LANES).saturating_mul(h));
        let dst_strides = match family {
            ColorFamily::RGB => [
                output_stride,
                output_plane_stride(&output, 1)?,
                output_plane_stride(&output, 2)?,
            ],
            _ => [output_stride; 3],
        };
        let src_stride = plane_stride(input, 0)?;
        let src_strides = match family {
            ColorFamily::RGB => [src_stride, plane_stride(input, 1)?, plane_stride(input, 2)?],
            _ => [src_stride; 3],
        };
        let geometry = Geometry {
            width: w,
            height: h,
            units: launch.units,
            stride: launch.stride(),
            groups: launch.grid(),
            columns: w.div_ceil(BLEND_LANES),
            blend_units: blend_launch.units,
            blend_stride: blend_launch.stride(),
            blend_groups: blend_launch.grid(),
            luma_stride: words(&luma)?,
            temp_stride: words(&temp)?,
            mask_raw_stride: words(&mask_raw)?,
            output_stride,
            dst_strides,
            src_bytes: quantisation.bytes as u32,
            is_float: u32::from(quantisation.float),
            channels,
            weights,
            src_strides,
            chroma: if family == ColorFamily::YUV {
                [
                    copy_of(input, &output, 1, quantisation.bytes)?,
                    copy_of(input, &output, 2, quantisation.bytes)?,
                ]
            } else {
                [None, None]
            },
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
                    src0_stride: geometry.src_strides[0],
                    src1_stride: geometry.src_strides[1],
                    src2_stride: geometry.src_strides[2],
                    luma_stride: geometry.luma_stride,
                    estimate_stride: geometry.temp_stride,
                    src_bytes: geometry.src_bytes,
                    channels: geometry.channels,
                    mode: 0,
                    is_float: geometry.is_float,
                    scale: quantisation.scale,
                    pedestal: self.plan.blend.pedestal,
                    weight0: geometry.weights[0],
                    weight1: geometry.weights[1],
                    weight2: geometry.weights[2],
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
                    self.record_mask(exec, &geometry, &luma, &mut temp, &mut mask_raw)?;
                    self.record_blend(
                        exec,
                        &geometry,
                        &luma,
                        &mut candidate,
                        candidate_stride,
                        &temp,
                        0,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_output(
                        exec,
                        &geometry,
                        input,
                        &luma,
                        &candidate,
                        candidate_stride,
                        &mut output,
                    )
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
                    src0_stride: geometry.src_strides[0],
                    src1_stride: geometry.src_strides[1],
                    src2_stride: geometry.src_strides[2],
                    luma_stride: geometry.luma_stride,
                    estimate_stride: a_stride,
                    src_bytes: geometry.src_bytes,
                    channels: geometry.channels,
                    mode: 1,
                    is_float: geometry.is_float,
                    scale: quantisation.scale,
                    pedestal: self.plan.blend.pedestal,
                    weight0: geometry.weights[0],
                    weight1: geometry.weights[1],
                    weight2: geometry.weights[2],
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
                        (&mut estimate_a, a_stride)
                    } else {
                        (&mut estimate_b, b_stride)
                    };
                    self.record_mask(exec, &geometry, &luma, &mut temp, &mut mask_raw)?;
                    // The blend restores the final estimate in place, so the plane
                    // the deconvolution ended on carries the output code values.
                    self.record_blend(
                        exec,
                        &geometry,
                        &luma,
                        &mut *candidate,
                        candidate_stride,
                        &temp,
                        1,
                    )?;
                    compute_barrier(exec, barrier);
                    self.record_output(
                        exec,
                        &geometry,
                        input,
                        &luma,
                        &*candidate,
                        candidate_stride,
                        &mut output,
                    )
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
        let src0 = exec.read_plane(input, 0)?;
        // A one-plane family binds its only plane in every source slot; the
        // kernel reads the extra two only when `channels` is three.
        let (second, third) = if push.channels == 3 { (1, 2) } else { (0, 0) };
        let src1 = exec.read_plane(input, second)?;
        let src2 = exec.read_plane(input, third)?;
        let dst = exec.write_plane(luma, 0)?;
        // The third binding must exist even when nothing writes it, and it must
        // not be the frame this dispatch writes.
        let seed = match estimate {
            Some(frame) => exec.write_plane(frame, 0)?,
            None => exec.read_plane(fallback, 0)?,
        };
        exec.dispatch(
            self.prepare.pipeline(),
            &[src0, src1, src2, dst, seed],
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
        self.record_horizontal(exec, geometry, source, source_stride, kernel, taps, temp)?;
        compute_barrier(exec, self.barrier);
        self.record_vertical(
            exec,
            geometry,
            &*temp,
            destination,
            destination_stride,
            kernel,
            taps,
            mode,
            aux,
            aux_stride,
        )
    }

    /// The horizontal half: `source` into the scratch plane.
    ///
    /// The aux binding is declared by the pipeline and never read here, so the
    /// source stands in for it rather than making the caller name a plane this
    /// pass has no use for.
    #[expect(
        clippy::too_many_arguments,
        reason = "the horizontal pass names its source, table and scratch"
    )]
    fn record_horizontal(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        source: &VideoFrame,
        source_stride: u32,
        kernel: &Gaussian,
        taps: &VideoFrame,
        scratch: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        let push = BlurPush {
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
            aux_stride: source_stride,
        };
        let (src, aux, tap, dst) = (
            exec.read_plane(source, 0)?,
            exec.read_plane(source, 0)?,
            exec.read_plane(taps, 0)?,
            exec.write_plane(scratch, 0)?,
        );
        exec.dispatch(
            self.blur.pipeline(),
            &[src, tap, dst, aux],
            bytes_of(&push),
            geometry.groups,
        )
    }

    /// The vertical half: the scratch plane into `destination`, with `mode`'s
    /// consumer fused into its output.
    #[expect(
        clippy::too_many_arguments,
        reason = "the vertical pass names its scratch, destination, table and aux"
    )]
    fn record_vertical(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        scratch: &VideoFrame,
        destination: &mut VideoFrame,
        destination_stride: u32,
        kernel: &Gaussian,
        taps: &VideoFrame,
        mode: u32,
        aux: &VideoFrame,
        aux_stride: u32,
    ) -> std::result::Result<(), VulkanError> {
        let push = BlurPush {
            units: geometry.units,
            stride: geometry.stride,
            width: geometry.width,
            height: geometry.height,
            radius: kernel.vertical_radius() as u32,
            taps: kernel.count() as u32,
            mode,
            direction: 1,
            strength: self.plan.strength,
            pedestal: self.plan.blend.pedestal,
            floor_value: self.plan.blend.division_floor,
            src_stride: geometry.temp_stride,
            dst_stride: destination_stride,
            aux_stride,
        };
        let (src, tap, dst, aux_plane) = (
            exec.read_plane(scratch, 0)?,
            exec.read_plane(taps, 0)?,
            exec.write_plane(destination, 0)?,
            exec.read_plane(aux, 0)?,
        );
        exec.dispatch(
            self.blur.pipeline(),
            &[src, tap, dst, aux_plane],
            bytes_of(&push),
            geometry.groups,
        )
    }

    /// The edge mask: the prefilter, the Sobel and smoothstep, then the mask's
    /// own blur.
    ///
    /// The Sobel writes into the horizontal scratch, so the mask needs the two
    /// planes the blurs already have and no third one. After it, the prefiltered
    /// luma is dead and the last vertical pass writes the final mask back over
    /// the plane it read, which leaves the mask in `temp`.
    fn record_mask(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        luma: &VideoFrame,
        temp: &mut VideoFrame,
        mask_raw: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        self.record_blur_pair(
            exec,
            geometry,
            luma,
            geometry.luma_stride,
            &self.plan.prefilter,
            &self.prefilter,
            &mut *temp,
            &mut *mask_raw,
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
            dst_stride: geometry.temp_stride,
        };
        let src = exec.read_plane(mask_raw, 0)?;
        let dst = exec.write_plane(temp, 0)?;
        exec.dispatch(
            self.sobel.pipeline(),
            &[src, dst],
            bytes_of(&push),
            geometry.groups,
        )?;
        compute_barrier(exec, self.barrier);

        // The mask's own blur, recorded as its two halves: the horizontal pass
        // writes over the prefiltered luma and the vertical pass writes the
        // mask back into the plane the Sobel left it in.
        self.record_horizontal(
            exec,
            geometry,
            &*temp,
            geometry.temp_stride,
            &self.plan.mask_blur,
            &self.mask,
            &mut *mask_raw,
        )?;
        compute_barrier(exec, self.barrier);
        self.record_vertical(
            exec,
            geometry,
            &*mask_raw,
            temp,
            geometry.temp_stride,
            &self.plan.mask_blur,
            &self.mask,
            0,
            luma,
            geometry.luma_stride,
        )?;
        // The blend reads what this last vertical wrote, so the mask's output
        // has to be visible before it records.
        compute_barrier(exec, self.barrier);
        Ok(())
    }

    /// Limits the candidate to the local extremes and blends it back into the
    /// candidate's own plane, so the restored luma costs no plane of its own.
    #[expect(
        clippy::too_many_arguments,
        reason = "the blend names three frames and the candidate's row stride"
    )]
    fn record_blend(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        luma: &VideoFrame,
        candidate: &mut VideoFrame,
        candidate_stride: u32,
        mask: &VideoFrame,
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
            mask_stride: geometry.temp_stride,
            strength: self.plan.strength,
            pedestal: self.plan.blend.pedestal,
            mode,
        };
        let src = exec.read_plane(luma, 0)?;
        let mask_plane = exec.read_plane(mask, 0)?;
        // The candidate is read and written in place: it is declared both ways
        // and bound once.
        exec.read(candidate);
        let candidate_plane = exec.write_plane(candidate, 0)?;
        exec.dispatch(
            self.blend.pipeline(),
            &[src, candidate_plane, mask_plane],
            bytes_of(&push),
            geometry.blend_groups,
        )
    }

    /// Writes the restored code values as the frame's own samples.
    fn record_store(
        &self,
        exec: &mut ExecContext<'_>,
        device: &Device,
        geometry: &Geometry,
        restored: &VideoFrame,
        restored_stride: u32,
        output: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        let quantisation = self.plan.quantisation;
        let push = StorePush {
            units: geometry.blend_units,
            stride: geometry.blend_stride,
            column_groups: geometry.columns,
            width: geometry.width,
            height: geometry.height,
            src_stride_words: restored_stride,
            dst_stride_bytes: geometry.output_stride,
            inverse: quantisation.inverse as f32,
            max_value: quantisation.max_value as f32,
            bytes: quantisation.bytes as u32,
            is_float: u32::from(quantisation.float),
        };
        let src = exec.read_plane(restored, 0)?;
        let dst = exec.write_plane(output, 0)?;
        exec.dispatch(
            device.pipeline(),
            &[src, dst],
            bytes_of(&push),
            geometry.blend_groups,
        )
    }

    /// Writes the restored RGB code values as one offset applied to all three
    /// channels, limited to the gamut the pixel has left.
    #[expect(
        clippy::too_many_arguments,
        reason = "the RGB store binds the input, the luma, the restored plane and the frame"
    )]
    fn record_store_rgb(
        &self,
        exec: &mut ExecContext<'_>,
        device: &Device,
        geometry: &Geometry,
        input: &VideoFrame,
        luma: &VideoFrame,
        restored: &VideoFrame,
        restored_stride: u32,
        output: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        let quantisation = self.plan.quantisation;
        let push = StoreRgbPush {
            units: geometry.blend_units,
            stride: geometry.blend_stride,
            column_groups: geometry.columns,
            width: geometry.width,
            height: geometry.height,
            src0_stride: geometry.src_strides[0],
            src1_stride: geometry.src_strides[1],
            src2_stride: geometry.src_strides[2],
            luma_stride: geometry.luma_stride,
            restored_stride,
            dst0_stride: geometry.dst_strides[0],
            dst1_stride: geometry.dst_strides[1],
            dst2_stride: geometry.dst_strides[2],
            scale: quantisation.scale,
            inverse: quantisation.inverse as f32,
            max_value: quantisation.max_value as f32,
            src_bytes: quantisation.bytes as u32,
            is_float: u32::from(quantisation.float),
        };
        let src0 = exec.read_plane(input, 0)?;
        let src1 = exec.read_plane(input, 1)?;
        let src2 = exec.read_plane(input, 2)?;
        let luma_plane = exec.read_plane(luma, 0)?;
        let restored_plane = exec.read_plane(restored, 0)?;
        let dst0 = exec.write_plane(output, 0)?;
        let dst1 = exec.write_plane(output, 1)?;
        let dst2 = exec.write_plane(output, 2)?;
        exec.dispatch(
            device.pipeline(),
            &[
                src0,
                src1,
                src2,
                luma_plane,
                restored_plane,
                dst0,
                dst1,
                dst2,
            ],
            bytes_of(&push),
            geometry.blend_groups,
        )
    }

    /// Copies one chroma plane through to the output.
    fn record_copy(
        &self,
        exec: &mut ExecContext<'_>,
        device: &Device,
        plan: &PlaneCopy,
        input: &VideoFrame,
        output: &mut VideoFrame,
        plane: i32,
    ) -> std::result::Result<(), VulkanError> {
        let push = CopyPush {
            units: plan.launch.units,
            stride: plan.launch.stride(),
            columns: plan.columns,
            src_stride: plan.src_stride,
            dst_stride: plan.dst_stride,
        };
        let src = exec.read_plane(input, plane)?;
        let dst = exec.write_plane(output, plane)?;
        exec.dispatch(
            device.pipeline(),
            &[src, dst],
            bytes_of(&push),
            plan.launch.grid(),
        )
    }

    /// Writes the frame's planes from the restored code values.
    #[expect(
        clippy::too_many_arguments,
        reason = "the output pass names the input, the luma, the restored plane and the frame"
    )]
    fn record_output(
        &self,
        exec: &mut ExecContext<'_>,
        geometry: &Geometry,
        input: &VideoFrame,
        luma: &VideoFrame,
        restored: &VideoFrame,
        restored_stride: u32,
        output: &mut VideoFrame,
    ) -> std::result::Result<(), VulkanError> {
        match &self.output {
            Output::Gray(store) => {
                self.record_store(exec, store, geometry, restored, restored_stride, output)
            }
            Output::Rgb(store) => self.record_store_rgb(
                exec,
                store,
                geometry,
                input,
                luma,
                restored,
                restored_stride,
                output,
            ),
            Output::Yuv(store, copy) => {
                self.record_store(exec, store, geometry, restored, restored_stride, output)?;
                for (index, plan) in geometry.chroma.iter().enumerate() {
                    let Some(plan) = plan else { continue };
                    let plane = i32::try_from(index).unwrap_or(0) + 1;
                    self.record_copy(exec, copy, plan, input, output, plane)?;
                }
                Ok(())
            }
        }
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
    output_stride: u32,
    /// The output plane strides, which only the RGB store needs three of.
    dst_strides: [u32; 3],
    /// The input plane strides the luma reads.
    src_strides: [u32; 3],
    /// The output format: bytes per sample, and whether it is a float.
    src_bytes: u32,
    is_float: u32,
    /// How many planes the luma reads, and the weight of each.
    channels: u32,
    weights: [f32; 3],
    /// The chroma planes a YUV frame carries through, plane 1 first.
    chroma: [Option<PlaneCopy>; 2],
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

/// The byte stride of one output plane, whose words the store writes into.
fn output_plane_stride(frame: &VideoFrame, plane: i32) -> Result<u32> {
    let stride = frame.stride(plane);
    if stride < 0 || stride % 4 != 0 {
        return Err(NImagesError::new(format!(
            "Deblur gpu=1: output plane {plane} rows are not a multiple of four bytes"
        )));
    }
    Ok(u32::try_from(stride).unwrap_or(u32::MAX))
}

/// The byte stride of one plane, which the shaders address rows by.
fn plane_stride(frame: &VideoFrame, plane: i32) -> Result<u32> {
    let stride = frame.stride(plane);
    if stride < 0 {
        return Err(NImagesError::new(format!(
            "Deblur gpu=1: plane {plane} reports a negative row stride"
        )));
    }
    Ok(u32::try_from(stride).unwrap_or(u32::MAX))
}

/// Plans a chroma plane's copy, or `None` when the plane has no pixels.
///
/// The copy moves whole words, so a row has to start and end on one. A stride
/// that is not a multiple of four is refused by name rather than read across
/// whatever sits beside it.
fn copy_of(
    input: &VideoFrame,
    output: &VideoFrame,
    plane: i32,
    bytes: usize,
) -> Result<Option<PlaneCopy>> {
    let width = u32::try_from(input.frame_width(plane)).unwrap_or(u32::MAX);
    let height = u32::try_from(input.frame_height(plane)).unwrap_or(u32::MAX);
    if width == 0 || height == 0 {
        return Ok(None);
    }
    let row_bytes = u32::try_from(bytes)
        .ok()
        .and_then(|bytes| width.checked_mul(bytes))
        .ok_or_else(|| NImagesError::new("Deblur gpu=1: a chroma row is too wide"))?;
    let src_stride = plane_stride(input, plane)?;
    let dst_stride = plane_stride(output, plane)?;
    if src_stride % 4 != 0 || dst_stride % 4 != 0 {
        return Err(NImagesError::new(format!(
            "Deblur gpu=1: plane {plane} rows are not a whole number of words, so the \
             chroma copy cannot move them"
        )));
    }
    if src_stride < row_bytes || dst_stride < row_bytes {
        return Err(NImagesError::new(format!(
            "Deblur gpu=1: plane {plane} reports a stride shorter than its row"
        )));
    }
    let columns = row_bytes.div_ceil(4);
    Ok(Some(PlaneCopy {
        columns,
        src_stride,
        dst_stride,
        launch: shared::Launch::new(columns.saturating_mul(height)),
    }))
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
