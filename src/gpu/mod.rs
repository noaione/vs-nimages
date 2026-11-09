//! The device layer: everything that names VapourSynth's Vulkan API.
//!
//! A filter's `gpu=1` path lives in [`crate::gpu::deblur`]-style modules that
//! hold GLSL and the per-filter sequence. This module owns what they share and
//! what is unsafe: the pipeline, the pool, the barrier between two dispatches,
//! the frames, and the host/device transfers. The rules that make a kernel
//! correct are in `docs/improvements/15-vulkan-device-path.md`.

pub(crate) mod deblur;
pub(crate) mod histogram;
pub(super) mod shared;

use std::ffi::CStr;
use std::ops::Deref;
pub(crate) mod map;
use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::ffi;
use vapoursynth4_rs::ffi::vulkan::vk;
use vapoursynth4_rs::frame::VideoFrame;
use vapoursynth4_rs::key;
use vapoursynth4_rs::map::{AppendMode, Value};
use vapoursynth4_rs::node::VideoNode;

use crate::error::{NImagesError, Result};
use crate::vulkan::{ComputePipeline, ExecContext, ExecPool, VulkanCoreExt, VulkanError};

/// A GPU frame a filter instance keeps for read-only reuse.
///
/// The table and tap frames are filled once in `create` and never written
/// again, and every read of them goes through the core's own thread-safe
/// accessors.  The bindings' frame is `Send` but not `Sync`, so this says so
/// once for exactly that use.
pub(crate) struct SharedFrame(VideoFrame);

// SAFETY: the frame is immutable from the moment the filter instance is
// built.  A recording only reads it, and the core hands read-only frames to
// every worker thread of a filter by design.
unsafe impl Sync for SharedFrame {}

impl SharedFrame {
    pub(crate) fn new(frame: VideoFrame) -> Self {
        Self(frame)
    }
}

impl Deref for SharedFrame {
    type Target = VideoFrame;

    fn deref(&self) -> &Self::Target {
        &self.0
    }
}

/// The entry point a recording needs between two dispatches that touch the same
/// plane.
///
/// Resolved once at filter creation: a core that does not export it cannot run a
/// kernel with an intra-recording dependency, and the error says so before the
/// first frame rather than after a wrong picture.
pub(crate) type BarrierFn = unsafe extern "system" fn(vk::CommandBuffer, *const vk::DependencyInfo);

/// One compute pipeline and the pool it records on, built once per instance.
pub(crate) struct Device {
    pipeline: ComputePipeline,
    pool: ExecPool,
}

impl Device {
    /// Compiles `source` and builds its pipeline.
    ///
    /// `bindings` is the number of storage buffers the kernel declares, in
    /// declaration order, and `push_bytes` the size of its push block.
    pub(crate) fn new(
        core: &CoreRef,
        source: &CStr,
        bindings: u32,
        push_bytes: u32,
    ) -> Result<Self> {
        let shader = core
            .compile_gpu_shader(ffi::VSGPUShaderLanguage::GLSL, source)
            .map_err(|error| gpu_error("could not compile the compute shader", &error))?;

        let mut builder = ComputePipeline::builder(core, &shader);
        for _ in 0..bindings {
            builder = builder.storage_buffer();
        }
        let pipeline = builder
            .push_constants(push_bytes)
            .build()
            .map_err(|error| gpu_error("could not build the compute pipeline", &error))?;

        let pool = core
            .create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)
            .map_err(|error| gpu_error("could not create an exec pool", &error))?;

        Ok(Self { pipeline, pool })
    }

    /// The pipeline every dispatch in this instance binds.
    pub(crate) fn pipeline(&self) -> &ComputePipeline {
        &self.pipeline
    }

    /// Uploads a table on this instance's own compute pool.
    pub(crate) fn upload_words(&self, core: &CoreRef, words: &[u32]) -> Result<SharedFrame> {
        upload_words_with_pool(core, words, &self.pool)
    }

    /// Records and submits one frame's work.
    ///
    /// The closure may only fail with a [`VulkanError`], which is what the
    /// binding's calls return; a failure of the filter's own is raised before
    /// the recording starts.
    pub(crate) fn record<F>(&self, f: F) -> std::result::Result<u64, VulkanError>
    where
        F: FnOnce(&mut ExecContext<'_>) -> std::result::Result<(), VulkanError>,
    {
        self.pool.record(f)
    }

    /// Waits out one submission whose result the host reads.
    ///
    /// A kernel that writes a buffer the host then maps has nothing else to
    /// wait on: the framework tracks frames, and a counter table is not one.
    /// Waiting the submission's own value keeps concurrent recordings on other
    /// threads from serializing behind it.
    pub(crate) fn wait(&self, value: u64) -> Result<()> {
        // SAFETY: `value` is a point this pool published, because `record`
        // returned it.
        unsafe { self.pool.wait_value(value) }
            .map(|_| ())
            .map_err(|error| gpu_error("could not wait for the device result", &error))
    }
}

/// Refuses the device path early when the core has no usable device.
///
/// Called from a filter's `create`, so a host without Vulkan 1.4 learns it there
/// rather than on the first frame.
pub(crate) fn require_device(core: &CoreRef) -> Result<()> {
    core.vulkan_core_info()
        .map(|_| ())
        .map_err(|error| gpu_error("no usable Vulkan device", &error))
}

/// Resolves the barrier entry point, or names what is missing.
pub(crate) fn barrier_fn(core: &CoreRef) -> Result<BarrierFn> {
    let functions = core
        .vulkan_functions()
        .map_err(|error| gpu_error("the core has no Vulkan functions", &error))?;
    functions.vkCmdPipelineBarrier2.ok_or_else(|| {
        NImagesError::new(
            "gpu: the core did not resolve vkCmdPipelineBarrier2; a kernel with two \
             dependent dispatches cannot be recorded",
        )
    })
}

/// Makes one dispatch's writes visible to the next dispatch's reads.
///
/// The framework tracks frames, not hazards inside one recording, so a kernel
/// that writes a scratch plane and then reads it records this between them. The
/// two stages are both compute: a separable blur is compute in, compute out.
pub(crate) fn compute_barrier(exec: &mut ExecContext<'_>, barrier: BarrierFn) {
    barrier_between(
        exec,
        barrier,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    );
}

/// Makes a transfer's writes visible to a kernel that reads them.
///
/// Zeroing a counter table is a fill, which is a transfer command, so the
/// dispatch that increments those counters records this between the two.
pub(crate) fn fill_barrier(exec: &mut ExecContext<'_>, barrier: BarrierFn) {
    barrier_between(
        exec,
        barrier,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_WRITE,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_READ | vk::AccessFlags2::SHADER_WRITE,
    );
}

/// Makes a kernel's writes visible to a transfer that copies them out.
pub(crate) fn transfer_barrier(exec: &mut ExecContext<'_>, barrier: BarrierFn) {
    barrier_between(
        exec,
        barrier,
        vk::PipelineStageFlags2::COMPUTE_SHADER,
        vk::AccessFlags2::SHADER_WRITE,
        vk::PipelineStageFlags2::TRANSFER,
        vk::AccessFlags2::TRANSFER_READ,
    );
}

/// Records one memory barrier between two stages on one command buffer.
fn barrier_between(
    exec: &mut ExecContext<'_>,
    barrier: BarrierFn,
    src_stage: vk::PipelineStageFlags2,
    src_access: vk::AccessFlags2,
    dst_stage: vk::PipelineStageFlags2,
    dst_access: vk::AccessFlags2,
) {
    let memory = vk::MemoryBarrier2::default()
        .src_stage_mask(src_stage)
        .src_access_mask(src_access)
        .dst_stage_mask(dst_stage)
        .dst_access_mask(dst_access);
    let dependency = vk::DependencyInfo::default().memory_barriers(std::slice::from_ref(&memory));

    // SAFETY: the recording is live and owned by this thread, and the call only
    // records a command before reading the structure above.
    unsafe { barrier(exec.command_buffer(), &dependency) };
}

/// Allocates a `GRAY32` frame for a kernel's intermediate plane.
///
/// A dispatch binds frames, not buffers, so scratch is a frame. `prop_src` is
/// only set for the frame that becomes the filter's output.
pub(crate) fn gray32_frame(
    core: &CoreRef,
    width: i32,
    height: i32,
    prop_src: Option<&VideoFrame>,
) -> Result<VideoFrame> {
    let Some(format) = core.query_video_format(
        ffi::VSColorFamily::Gray,
        ffi::VSSampleType::Integer,
        32,
        0,
        0,
    ) else {
        return Err(NImagesError::new(
            "gpu: the core does not know the GRAY32 format",
        ));
    };
    core.new_gpu_video_frame(&format, width, height, prop_src)
        .ok_or_else(|| NImagesError::new("gpu: the core could not allocate a device frame"))
}

/// A device copy of a host clip, for a kernel that runs there.
pub(crate) fn upload(core: &CoreRef, node: &VideoNode) -> Result<VideoNode> {
    transfer(core, node, c"GPUUpload")
}

/// A host copy of a device clip, for a caller that needs the pixels back.
pub(crate) fn download(core: &CoreRef, node: &VideoNode) -> Result<VideoNode> {
    transfer(core, node, c"GPUDownload")
}

/// Moves a clip across the boundary with the core's own filter.
///
/// `std.GPUUpload` and `std.GPUDownload` are what the core inserts for a `gpu`
/// argument. An argument declared `all` gets no insertion, because one instance
/// has to serve both engines, so the transfer is the instance's own business.
fn transfer(core: &CoreRef, node: &VideoNode, function: &CStr) -> Result<VideoNode> {
    let std = core
        .get_plugin_by_id(c"com.vapoursynth.std")
        .ok_or_else(|| NImagesError::new("gpu: the core has no std plugin to move the clip"))?;

    let mut args = core.create_map();
    args.set(
        key!(c"clip"),
        Value::VideoNode(node.clone()),
        AppendMode::Replace,
    )
    .map_err(|error| NImagesError::new(format!("gpu: {error}")))?;

    let result = std.invoke(function, args);
    if let Some(error) = result.get_error() {
        return Err(NImagesError::new(format!(
            "gpu: std.{}: {}",
            function.to_string_lossy(),
            error.to_string_lossy()
        )));
    }

    result.get_video_node(key!(c"clip"), 0).map_err(|error| {
        NImagesError::new(format!("gpu: std.{}: {error}", function.to_string_lossy()))
    })
}

/// Uploads `values` as a one-row `GRAY32` device frame.
///
/// A dispatch binds frames, so a tap table a kernel reads is a frame too. The
/// values are copied as their `f32` bit patterns, which is what the kernel reads.
pub(crate) fn upload_floats(core: &CoreRef, values: &[f32]) -> Result<SharedFrame> {
    let words: Vec<u32> = values.iter().map(|value| value.to_bits()).collect();
    upload_words(core, &words)
}

/// Uploads `words` as a one-row `GRAY32` device frame.
///
/// A dispatch binds frames, so a tap table or a lookup table a kernel indexes
/// is a frame too. The host is the only writer here, so the copy is recorded
/// once and the frame's producer pair is what later recordings wait on.
pub(crate) fn upload_words(core: &CoreRef, words: &[u32]) -> Result<SharedFrame> {
    let pool = core
        .create_gpu_exec_pool(ffi::VSVulkanQueueType::Compute)
        .map_err(|error| gpu_error("could not create the upload pool", &error))?;
    upload_words_with_pool(core, words, &pool)
}

/// Uploads `words` on an existing compute pool.
///
/// A filter that rebuilds a table per frame keeps one pool for the life of the
/// instance, so each frame pays the copy and not a pool creation.
pub(crate) fn upload_words_with_pool(
    core: &CoreRef,
    words: &[u32],
    pool: &ExecPool,
) -> Result<SharedFrame> {
    let bytes = words.len().saturating_mul(size_of::<u32>());
    if bytes == 0 {
        return Err(NImagesError::new("gpu: a table cannot be empty"));
    }
    let size = u64::try_from(bytes)
        .map_err(|_| NImagesError::new("gpu: the table is larger than a device buffer"))?;

    let mut staging = core
        .create_gpu_buffer(
            size,
            vk::BufferUsageFlags::TRANSFER_SRC,
            vk::MemoryPropertyFlags::HOST_VISIBLE | vk::MemoryPropertyFlags::HOST_COHERENT,
            vk::MemoryPropertyFlags::DEVICE_LOCAL,
        )
        .map_err(|error| gpu_error("could not allocate a staging buffer", &error))?;

    {
        // SAFETY: nothing has been submitted against the buffer yet, so the host
        // is the only writer, and the borrow ends before the recording.
        let Some(mapped) = (unsafe { staging.mapped_slice() }) else {
            return Err(NImagesError::new(
                "gpu: the staging buffer is not host visible",
            ));
        };
        if mapped.len() < bytes {
            return Err(NImagesError::new("gpu: the staging buffer is too small"));
        }
        for (slot, word) in words.iter().enumerate() {
            let start = slot * size_of::<u32>();
            mapped[start..start + size_of::<u32>()].copy_from_slice(&word.to_ne_bytes());
        }
    }

    let Some(format) = core.query_video_format(
        ffi::VSColorFamily::Gray,
        ffi::VSSampleType::Integer,
        32,
        0,
        0,
    ) else {
        return Err(NImagesError::new(
            "gpu: the core does not know the GRAY32 format",
        ));
    };
    let width = i32::try_from(words.len())
        .map_err(|_| NImagesError::new("gpu: the table is too wide for one row"))?;
    let mut frame = core
        .new_gpu_video_frame(&format, width, 1, None)
        .ok_or_else(|| NImagesError::new("gpu: the core could not allocate the table frame"))?;

    let functions = core
        .vulkan_functions()
        .map_err(|error| gpu_error("the upload needs the core's Vulkan functions", &error))?;
    let copy = functions
        .vkCmdCopyBuffer2
        .ok_or_else(|| NImagesError::new("gpu: the core did not resolve vkCmdCopyBuffer2"))?;

    pool.record(|exec| {
        let destination = exec.write_video_frame(&mut frame, 0)?;
        let region = vk::BufferCopy2::default()
            .src_offset(0)
            .dst_offset(0)
            .size(size);
        let info = vk::CopyBufferInfo2::default()
            .src_buffer(staging.buffer())
            .dst_buffer(destination.buffer())
            .regions(std::slice::from_ref(&region));

        // SAFETY: the recording is live, both buffers are declared on it, and
        // the structure outlives the call.
        unsafe { copy(exec.command_buffer(), &info) };
        // SAFETY: the submission reads the staging buffer, so ownership passes
        // to the recording for exactly as long as that submission runs.
        unsafe { exec.uses_buffer(staging) };
        Ok(())
    })
    .map_err(|error| gpu_error("could not upload the table", &error))?;

    Ok(SharedFrame::new(frame))
}

/// Turns a Vulkan failure into a message that names the step.
pub(crate) fn gpu_error(what: &str, error: &VulkanError) -> NImagesError {
    NImagesError::new(format!("gpu: {what}: {}", error.message()))
}
