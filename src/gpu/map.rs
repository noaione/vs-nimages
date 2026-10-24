//! `Levels` and `Posterize` on the device: one table lookup per sample.
//!
//! Both filters are a pointwise map, so they share this kernel; only the table
//! differs, and the table is built by the host code the CPU path already calls.
//! The kernel does one clamped index and one store, with no arithmetic of its
//! own, which is why the two paths are byte-identical by construction.
//!
//! The table is a one-row `GRAY32` device frame: 1 KB for 8 bit, 256 KB for the
//! native 16 bit range. A fixed table is uploaded once at creation; a table that
//! changes per frame gets its own frame per recording, because a shared table
//! would be rewritten while the previous recording is still reading it.

use std::mem::size_of;

use vapoursynth4_rs::ColorFamily;
use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::frame::{VideoFormat, VideoFrame};

use crate::error::{NImagesError, Result};

use super::shared::{self, PREAMBLE, bytes_of, shader};
use super::{Device, gpu_error};

/// Storage buffers one map dispatch binds: the source plane, the table, and the
/// plane written.
const MAP_BINDINGS: u32 = 3;

/// Samples one invocation handles, packed into whole output words.
const MAP_LANES: u32 = 4;

/// What the map kernel is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct MapPush {
    units: u32,
    stride: u32,
    column_groups: u32,
    width: u32,
    src_stride_bytes: u32,
    dst_stride_bytes: u32,
    table_length: u32,
    bytes: u32,
}

// Eight words of four bytes: the assert is a division so it does not read like
// a bit width.
const _: () = assert!(size_of::<MapPush>() / size_of::<u32>() == 8);

const MAP_BODY: &str = r#"
layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint column_groups;
    uint width;
    uint src_stride_bytes;
    uint dst_stride_bytes;
    uint table_length;
    uint bytes;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) readonly buffer Table { uint words[]; } table;
layout(std430, set = 0, binding = 2) buffer Dst { uint words[]; } dst;

// One byte of the source plane, addressed from the plane's own row stride.
uint load_byte(uint offset) {
    uint word = src.words[offset >> 2u];
    return (word >> (8u * (offset & 3u))) & 0xFFu;
}

// One source sample in its native code value.
uint read_sample(uint x, uint y) {
    uint offset = y * p.src_stride_bytes + x * p.bytes;
    if (p.bytes == 2u) {
        return load_byte(offset) | (load_byte(offset + 1u) << 8u);
    }
    return load_byte(offset);
}

// One sample into the destination, leaving the bytes it shares a word with
// alone: the host writes one sample at a time too, and the neighbour in the
// word belongs to the neighbouring sample or to row padding.
void write_sample(uint x, uint y, uint value) {
    uint offset = y * p.dst_stride_bytes + x * p.bytes;
    uint word = offset >> 2u;
    uint shift = 8u * (offset & 3u);
    uint mask = p.bytes == 1u ? 0xFFu : 0xFFFFu;
    uint packed = dst.words[word];
    dst.words[word] = (packed & ~(mask << shift)) | (value << shift);
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
            // The host indexes with the same clamp, which only matters when a
            // table is shorter than the sample range.
            uint index = min(read_sample(x, y), p.table_length - 1u);
            write_sample(x, y, table.words[index]);
        }
    }
}
"#;

/// One plane's dispatch geometry, resolved before the recording starts.
#[derive(Clone, Copy)]
struct Plane {
    width: u32,
    src_stride: u32,
    dst_stride: u32,
    /// Four sample groups per row.
    columns: u32,
    launch: shared::Launch,
}

/// The device half of a pointwise map.
pub(crate) struct Map {
    pipeline: Device,
}

impl Map {
    /// Compiles the lookup kernel.
    pub(crate) fn new(core: &CoreRef) -> Result<Self> {
        let pipeline = Device::new(
            core,
            &shader(&[PREAMBLE, MAP_BODY])?,
            MAP_BINDINGS,
            size_of::<MapPush>() as u32,
        )?;
        Ok(Self { pipeline })
    }

    /// Maps every plane of `input` through `table` into a new device frame.
    ///
    /// `table` is a one-row `GRAY32` frame holding `table_length` words, which
    /// is what [`crate::gpu::upload_words`] builds from the host's table.
    pub(crate) fn render(
        &self,
        core: &CoreRef,
        input: &VideoFrame,
        table: &VideoFrame,
        table_length: u32,
    ) -> Result<VideoFrame> {
        let width = input.frame_width(0);
        let height = input.frame_height(0);
        let format = input.get_video_format();
        let bytes = sample_bytes(format)?;
        if table_length == 0 {
            return Err(NImagesError::new("gpu: the map's table is empty"));
        }

        let mut output = core
            .new_gpu_video_frame(format, width, height, Some(input))
            .ok_or_else(|| NImagesError::new("gpu: the core could not allocate a device frame"))?;

        let planes = plane_count(format);
        let mut geometry: [Option<Plane>; 3] = [None; 3];
        for (plane, slot) in geometry.iter_mut().enumerate().take(planes) {
            let plane = i32::try_from(plane).unwrap_or(0);
            let plane_width = input.frame_width(plane);
            let plane_height = input.frame_height(plane);
            if plane_width <= 0 || plane_height <= 0 {
                continue;
            }
            let width = u32::try_from(plane_width).unwrap_or(u32::MAX);
            let height = u32::try_from(plane_height).unwrap_or(u32::MAX);
            let row_bytes = width
                .checked_mul(bytes)
                .ok_or_else(|| NImagesError::new("gpu: a map plane row is too wide"))?;
            let src_stride = plane_stride(input, plane)?;
            let dst_stride = output_plane_stride(&output, plane)?;
            if src_stride < row_bytes || dst_stride < row_bytes {
                return Err(NImagesError::new(format!(
                    "gpu: map plane {plane} reports a stride shorter than its row"
                )));
            }
            let columns = width.div_ceil(MAP_LANES);
            *slot = Some(Plane {
                width,
                src_stride,
                dst_stride,
                columns,
                launch: shared::Launch::new(columns.saturating_mul(height)),
            });
        }

        self.pipeline
            .record(|exec| {
                for (plane, geometry) in geometry.iter().enumerate() {
                    let Some(geometry) = geometry else { continue };
                    let plane = i32::try_from(plane).unwrap_or(0);
                    let push = MapPush {
                        units: geometry.launch.units,
                        stride: geometry.launch.stride(),
                        column_groups: geometry.columns,
                        width: geometry.width,
                        src_stride_bytes: geometry.src_stride,
                        dst_stride_bytes: geometry.dst_stride,
                        table_length,
                        bytes,
                    };
                    let src = exec.read_plane(input, plane)?;
                    let mapped = exec.read_plane(table, 0)?;
                    let dst = exec.write_plane(&mut output, plane)?;
                    exec.dispatch(
                        self.pipeline.pipeline(),
                        &[src, mapped, dst],
                        bytes_of(&push),
                        geometry.launch.grid(),
                    )?;
                }
                Ok(())
            })
            .map_err(|error| gpu_error("could not record the map passes", &error))?;

        Ok(output)
    }
}

/// Bytes one sample of `format` occupies, which is the only width the map takes.
fn sample_bytes(format: &VideoFormat) -> Result<u32> {
    if format.bits_per_sample <= 8 {
        Ok(1)
    } else {
        Ok(2)
    }
}

/// How many planes a frame of `format` carries.
fn plane_count(format: &VideoFormat) -> usize {
    match format.color_family {
        ColorFamily::Gray => 1,
        _ => 3,
    }
}

/// The byte stride of one input plane.
fn plane_stride(frame: &VideoFrame, plane: i32) -> Result<u32> {
    let stride = frame.stride(plane);
    if stride < 0 {
        return Err(NImagesError::new(format!(
            "gpu: map plane {plane} reports a negative row stride"
        )));
    }
    Ok(u32::try_from(stride).unwrap_or(u32::MAX))
}

/// The byte stride of one output plane, whose words the map writes.
fn output_plane_stride(frame: &VideoFrame, plane: i32) -> Result<u32> {
    let stride = frame.stride(plane);
    if stride < 0 || stride % 4 != 0 {
        return Err(NImagesError::new(format!(
            "gpu: map output plane {plane} rows are not a multiple of four bytes"
        )));
    }
    Ok(u32::try_from(stride).unwrap_or(u32::MAX))
}
