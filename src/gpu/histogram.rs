//! `PeakStats` and `PeakGrayShades` on the device: plane 0's histogram, read
//! back.
//!
//! The counts are exact integers, so both engines hand the same [`Histogram`]
//! to the same host analysis. Only the counting moves: the peak search, the
//! shade ordering, the property arrays and `copy_frame` stay where they were,
//! which is why the two engines cannot disagree about what a histogram means.
//!
//! One kernel serves both analyzers. 8 bit counts into a workgroup's own 1 KB
//! table and drains it once per workgroup, because a page of flat tones points
//! most pixels at one bin and a per-sample `atomicAdd` serializes on it. 9 to
//! 16 bit cannot hold 256 KB of bins in shared memory, so those depths count
//! into global memory directly.
//!
//! The result comes back through a buffer that asks for `HOST_CACHED`: the
//! readback is 1 KB at 8 bit and 256 KB at 16, which is small enough that the
//! wrong preference would still cost more than the counting did.

use std::mem::size_of;

use vapoursynth4_rs::SampleType;
use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::ffi;
use vapoursynth4_rs::frame::VideoFrame;

use crate::error::{NImagesError, Result};
use crate::histogram::{BINS, Histogram};

use super::shared::{self, PREAMBLE, bytes_of, shader};
use super::{
    BarrierFn, Device, barrier_fn, fill_barrier, gpu_error, gray32_frame, transfer_barrier,
};

/// Storage buffers one count dispatch binds: the source plane and the counters.
const HISTOGRAM_BINDINGS: u32 = 2;

/// Samples one invocation counts, packed into whole rows of four.
const HISTOGRAM_LANES: u32 = 4;

/// Counters one workgroup accumulates before draining them into global memory.
///
/// A page of flat tones points most pixels at one bin, and a `u8` plane has
/// exactly this many bins, so the local table is the whole 8 bit range.
const LOCAL_BINS: u32 = 256;

const _: () = assert!(LOCAL_BINS as usize == BINS);

/// What the histogram kernel is told.
#[repr(C)]
#[derive(Clone, Copy)]
struct HistogramPush {
    units: u32,
    stride: u32,
    columns: u32,
    width: u32,
    src_stride_bytes: u32,
    bins: u32,
    bytes: u32,
}

const _: () = assert!(size_of::<HistogramPush>() / size_of::<u32>() == 7);

const HISTOGRAM_BODY: &str = r#"
#define LOCAL_BINS 256u
#define LANES 4u

layout(push_constant) uniform Params {
    uint units;
    uint stride;
    uint columns;
    uint width;
    uint src_stride_bytes;
    uint bins;
    uint bytes;
} p;

layout(std430, set = 0, binding = 0) readonly buffer Src { uint words[]; } src;
layout(std430, set = 0, binding = 1) buffer Table { uint words[]; } table;

// One workgroup's own counters, drained once so the global table sees one
// atomic per bin per workgroup instead of one per sample.
shared uint local[LOCAL_BINS];

uint load_byte(uint offset) {
    uint word = src.words[offset >> 2u];
    return (word >> (8u * (offset & 3u))) & 0xFFu;
}

uint read_sample(uint x, uint y) {
    uint offset = y * p.src_stride_bytes + x * p.bytes;
    if (p.bytes == 2u) {
        return load_byte(offset) | (load_byte(offset + 1u) << 8u);
    }
    return load_byte(offset);
}

uint bin_of(uint x, uint y) {
    return min(read_sample(x, y), p.bins - 1u);
}

void main() {
    uint unit = gl_GlobalInvocationID.x;
    uint tid = gl_LocalInvocationID.x;

    if (p.bytes == 1u) {
        for (uint i = tid; i < LOCAL_BINS; i += gl_WorkGroupSize.x) {
            local[i] = 0u;
        }
        barrier();

        for (; unit < p.units; unit += p.stride) {
            uint y = unit / p.columns;
            uint x0 = (unit % p.columns) * LANES;
            for (uint lane = 0u; lane < LANES; lane++) {
                uint x = x0 + lane;
                if (x >= p.width) {
                    break;
                }
                atomicAdd(local[bin_of(x, y)], 1u);
            }
        }
        barrier();

        for (uint i = tid; i < LOCAL_BINS; i += gl_WorkGroupSize.x) {
            uint count = local[i];
            if (count != 0u) {
                atomicAdd(table.words[i], count);
            }
        }
        return;
    }

    for (; unit < p.units; unit += p.stride) {
        uint y = unit / p.columns;
        uint x0 = (unit % p.columns) * LANES;
        for (uint lane = 0u; lane < LANES; lane++) {
            uint x = x0 + lane;
            if (x >= p.width) {
                break;
            }
            atomicAdd(table.words[bin_of(x, y)], 1u);
        }
    }
}
"#;

/// The device half of one histogram.
pub(crate) struct Engine {
    pipeline: Device,
    barrier: BarrierFn,
}

impl Engine {
    /// Compiles the counting kernel.
    pub(crate) fn new(core: &CoreRef) -> Result<Self> {
        let pipeline = Device::new(
            core,
            &shader(&[PREAMBLE, HISTOGRAM_BODY])?,
            HISTOGRAM_BINDINGS,
            size_of::<HistogramPush>() as u32,
        )?;
        Ok(Self {
            pipeline,
            barrier: barrier_fn(core)?,
        })
    }

    /// Counts plane 0 of `frame` and hands back the table the host reads.
    ///
    /// The counter table is a one-row `GRAY32` device frame, because a dispatch
    /// binds frames, and it is zeroed by a fill and copied out into a
    /// host-visible buffer before this returns. The wait is the pool's own
    /// submission value, so another thread's recording does not serialize this
    /// one.
    pub(crate) fn count(&self, core: &CoreRef, frame: &VideoFrame) -> Result<Histogram> {
        let format = frame.get_video_format();
        let bits = format.bits_per_sample;
        if format.sample_type != SampleType::Integer || !(8..=16).contains(&bits) {
            return Err(NImagesError::new(
                "gpu: the histogram takes a Gray 8 to 16 bit integer frame",
            ));
        }

        let bytes = if bits == 8 { 1u32 } else { 2u32 };
        let bins = 1u32 << bits;
        let max_value = u16::try_from(bins - 1)
            .map_err(|_| NImagesError::new("gpu: the histogram has no sample maximum"))?;

        let width = u32::try_from(frame.frame_width(0))
            .map_err(|_| NImagesError::new("gpu: plane 0 has a negative width"))?;
        let height = u32::try_from(frame.frame_height(0))
            .map_err(|_| NImagesError::new("gpu: plane 0 has a negative height"))?;
        if width == 0 || height == 0 {
            return empty_histogram(bits, max_value);
        }

        let stride = frame.stride(0);
        if stride < 0 {
            return Err(NImagesError::new("gpu: plane 0 has a negative row stride"));
        }
        let row_bytes = width.checked_mul(bytes).ok_or_else(|| {
            NImagesError::new("gpu: plane 0's row is wider than the address space")
        })?;
        let src_stride_bytes = u32::try_from(stride)
            .map_err(|_| NImagesError::new("gpu: plane 0's row stride is too large"))?;
        if src_stride_bytes < row_bytes {
            return Err(NImagesError::new(
                "gpu: plane 0 reports a stride shorter than its row",
            ));
        }

        let columns = width.div_ceil(HISTOGRAM_LANES);
        let units = columns.checked_mul(height).ok_or_else(|| {
            NImagesError::new(
                "gpu: the plane is too large for one histogram dispatch; count it on the host",
            )
        })?;
        let launch = shared::Launch::new(units);
        let total = u64::from(width).saturating_mul(u64::from(height));

        let mut counters = gray32_frame(
            core,
            i32::try_from(bins)
                .map_err(|_| NImagesError::new("gpu: the histogram has too many bins"))?,
            1,
            None,
        )?;

        let size = u64::from(bins) * (size_of::<u32>() as u64);
        let mut readback = core
            .create_gpu_buffer(
                size,
                ffi::VK_BUFFER_USAGE_TRANSFER_DST_BIT,
                ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT,
                ffi::VK_MEMORY_PROPERTY_HOST_VISIBLE_BIT
                    | ffi::VK_MEMORY_PROPERTY_HOST_COHERENT_BIT
                    | ffi::VK_MEMORY_PROPERTY_HOST_CACHED_BIT,
            )
            .map_err(|error| {
                gpu_error("could not allocate the histogram readback buffer", &error)
            })?;

        let functions = core.vulkan_functions().map_err(|error| {
            gpu_error("the histogram needs the core's Vulkan functions", &error)
        })?;
        let fill = functions
            .vkCmdFillBuffer
            .ok_or_else(|| NImagesError::new("gpu: the core did not resolve vkCmdFillBuffer"))?;
        let copy = functions
            .vkCmdCopyBuffer2
            .ok_or_else(|| NImagesError::new("gpu: the core did not resolve vkCmdCopyBuffer2"))?;

        let push = HistogramPush {
            units,
            stride: launch.stride(),
            columns,
            width,
            src_stride_bytes,
            bins,
            bytes,
        };

        let value = self
            .pipeline
            .record(|exec| {
                let counters = exec.write_plane(&mut counters, 0)?;

                // SAFETY: the recording is live, the counter plane belongs to a
                // frame this call owns, and the range stays inside its buffer.
                unsafe { fill(exec.command_buffer(), counters.buffer(), 0, size, 0) };
                fill_barrier(exec, self.barrier);

                let source = exec.read_plane(frame, 0)?;
                exec.dispatch(
                    self.pipeline.pipeline(),
                    &[source, counters],
                    bytes_of(&push),
                    launch.grid(),
                )?;
                transfer_barrier(exec, self.barrier);

                let region = ffi::VkBufferCopy2 {
                    sType: ffi::VK_STRUCTURE_TYPE_BUFFER_COPY_2,
                    pNext: std::ptr::null(),
                    srcOffset: 0,
                    dstOffset: 0,
                    size,
                };
                let info = ffi::VkCopyBufferInfo2 {
                    sType: ffi::VK_STRUCTURE_TYPE_COPY_BUFFER_INFO_2,
                    pNext: std::ptr::null(),
                    srcBuffer: counters.buffer(),
                    dstBuffer: readback.buffer(),
                    regionCount: 1,
                    pRegions: &raw const region,
                };

                // SAFETY: the recording is live, both buffers outlive the
                // submission, and the structure outlives the call.
                unsafe { copy(exec.command_buffer(), &raw const info) };
                Ok(())
            })
            .map_err(|error| gpu_error("could not record the histogram passes", &error))?;

        self.pipeline.wait(value)?;

        // SAFETY: the submission this readback was recorded into has completed,
        // the buffer is host visible and coherent, and nothing else maps it.
        let Some(mapped) = (unsafe { readback.mapped_slice() }) else {
            return Err(NImagesError::new(
                "gpu: the histogram readback buffer is not host visible",
            ));
        };
        let word_count = usize::try_from(bins)
            .map_err(|_| NImagesError::new("gpu: the histogram has too many bins"))?;
        if mapped.len() < word_count * size_of::<u32>() {
            return Err(NImagesError::new(
                "gpu: the histogram readback buffer is smaller than its table",
            ));
        }

        let mut counts = Vec::new();
        counts
            .try_reserve_exact(word_count)
            .map_err(|_| NImagesError::new("gpu: could not allocate the host histogram table"))?;
        for chunk in mapped
            .as_chunks::<{ size_of::<u32>() }>()
            .0
            .iter()
            .take(word_count)
        {
            counts.push(u64::from(u32::from_ne_bytes(*chunk)));
        }

        if bits == 8 {
            let mut inline = [0u64; BINS];
            for (slot, &count) in counts.iter().enumerate().take(BINS) {
                if let Some(target) = inline.get_mut(slot) {
                    *target = count;
                }
            }
            return Ok(Histogram::from_counts(inline, total));
        }

        Histogram::from_counts_for_range(counts, total, max_value).ok_or_else(|| {
            NImagesError::new("gpu: the histogram returned the wrong number of bins")
        })
    }
}

/// The table an empty plane produces, without a dispatch.
fn empty_histogram(bits: i32, max_value: u16) -> Result<Histogram> {
    let histogram = if bits == 8 {
        Histogram::from_plane(&[], 0, 0, 0)
    } else {
        Histogram::from_u16_plane(&[], 0, 0, 0, max_value)
    };
    histogram.ok_or_else(|| NImagesError::new("gpu: could not allocate the empty histogram"))
}
