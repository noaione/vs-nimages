//! Safe wrappers over the Vulkan device VapourSynth owns.
//!
//! The GPU side of the API is a table of pointers, not a linked library: the
//! core creates the instance, picks the physical device and resolves every
//! entry point for it, and hands the result back through
//! [`VulkanCoreExt::vulkan_functions()`].  Nothing here loads Vulkan, and no
//! handle is created on a device this module chose — the core's device is the
//! only one in play, which is what lets a frame, a scratch buffer and a
//! recording share producer pairs without any cross-device synchronization.
//!
//! The module splits along the line between what it can check and what it
//! cannot:
//!
//! * **Owned resources, device information and frames.** A [`Shader`], a
//!   [`ComputePipeline`], a [`GpuBuffer`], an [`ExecPool`] and a [`GpuFrame`]
//!   each own exactly one handle and give it back when dropped.
//!   [`VulkanCoreInfo`] is the core's own view of the device as plain data, so
//!   it can outlive the call that filled it.
//!
//! * **Everything that records or publishes GPU work.** Acquiring a recording,
//!   declaring what it reads and writes, submitting it and handing a buffer to
//!   it are `unsafe`: a recording that outlives its thread or a buffer that is
//!   destroyed while a submission still uses it is a memory error, not a wrong
//!   answer, and the wrapper cannot see the device state that would decide it.
//!
//! The one rule the wrapper does enforce is the one that is fatal rather than
//! merely wrong: a thread may hold at most one recording.  It is a
//! thread-local claim taken when a context is acquired and released exactly
//! once, whether the recording is submitted, abandoned or unwound through, so a
//! nested [`ExecPool::record()`] is refused instead of deadlocking the ring.
//!
//! # Ordering rules the wrapper cannot enforce
//!
//! Handing a [`GpuBuffer`] to [`ExecContext::uses_buffer()`] transfers
//! ownership, and submitting consumes the recording; both are `unsafe`.
//! Dropping a buffer a submission still reads aliases live data rather than
//! faulting, exactly as the C API documents, so the promise that the submission
//! has completed has to be made at the `unsafe` call site that used it.  Frames
//! are the exception: their producer pairs carry the synchronization, which is
//! why creating and reading one is safe.

use std::{
    cell::Cell,
    ffi::{CStr, c_char},
    fmt,
    marker::PhantomData,
    mem::{MaybeUninit, forget, size_of},
    ops::Deref,
    ptr::{NonNull, null},
    slice,
};

use vapoursynth4_rs::{
    api::Api,
    core::{Core, CoreHandle, CoreRef},
    ffi,
    ffi::vulkan::vk,
    frame::VideoFrame,
};

/// The size of the error buffers the API writes its messages into.
///
/// The API takes the size alongside the pointer and truncates rather than
/// overflowing, so this only decides how much of a message survives.
const ERROR_BUFFER_SIZE: i32 = 512;

/// A buffer the API writes an error message into.
type ErrorBuffer = [c_char; 512];

/// Calls `f` with a fresh error buffer, and reads the message back out.
///
/// Every failure in this module is reported the same way, so every call site
/// goes through here.  The message is copied out immediately because the
/// library only guarantees it until the next call.
fn with_error_buffer<T>(f: impl FnOnce(*mut c_char, i32) -> T) -> (T, Option<String>) {
    let mut buffer: ErrorBuffer = [0; 512];
    let result = f(buffer.as_mut_ptr(), ERROR_BUFFER_SIZE);
    (result, read_message(&buffer))
}

/// Copies a NUL terminated message out of an error buffer.
fn read_message(buffer: &ErrorBuffer) -> Option<String> {
    let bytes = buffer
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| byte.cast_unsigned())
        .collect::<Vec<_>>();

    (!bytes.is_empty()).then(|| String::from_utf8_lossy(&bytes).into_owned())
}

/// Reads a NUL terminated C string out of a fixed size character array.
fn read_string<const N: usize>(buffer: &[c_char; N]) -> String {
    let bytes = buffer
        .iter()
        .take_while(|byte| **byte != 0)
        .map(|byte| byte.cast_unsigned())
        .collect::<Vec<_>>();

    String::from_utf8_lossy(&bytes).into_owned()
}

/// Converts the `c_int` a drain call reports into a [`ffi::VSGPUDrainResult`].
///
/// The C API returns a plain `int` for these, so a value outside the documented
/// range is a library bug rather than something a caller can act on.
fn drain_result(value: i32) -> Option<ffi::VSGPUDrainResult> {
    match value {
        0 => Some(ffi::VSGPUDrainResult::Drained),
        1 => Some(ffi::VSGPUDrainResult::Incomplete),
        2 => Some(ffi::VSGPUDrainResult::DeviceLost),
        _ => None,
    }
}

/// An error reported by the GPU side of the API.
///
/// Most GPU calls do not return status codes; they either return a handle or
/// write a human readable message out.  This carries that message, and falls
/// back to a description of the call when the library left the buffer empty.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct VulkanError {
    message: String,
}

impl VulkanError {
    /// Wraps a message the library wrote, or `what` when it wrote nothing.
    pub(crate) fn new(context: Option<String>, what: &str) -> Self {
        Self {
            message: context
                .filter(|message| !message.is_empty())
                .unwrap_or_else(|| what.to_owned()),
        }
    }

    /// The message the library wrote, or the fallback it was constructed with.
    #[must_use]
    pub fn message(&self) -> &str {
        &self.message
    }
}

impl fmt::Display for VulkanError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(&self.message)
    }
}

impl std::error::Error for VulkanError {}

/// A [`VSVULKANAPI`](ffi::VSVULKANAPI) table.
///
/// The table is a compile time member of the linked library, so a value of this
/// type is just a borrowed pointer: it owns nothing and never fails to exist
/// once an [`Api`] does.  It is kept as a value rather than a borrow because
/// the resources in this module outlive any one function call and need the
/// table to destroy themselves.
#[derive(Clone, Copy, PartialEq, Eq, Hash)]
#[repr(transparent)]
pub struct VsVulkanApi(*const ffi::VSVULKANAPI);

impl VsVulkanApi {
    /// Fetches the GPU API table of a linked library.
    #[must_use]
    pub fn new(api: &Api) -> Self {
        // SAFETY: the table is a constant of the library, valid for as long as
        // the library is, and the `vulkan` feature guarantees this member.
        Self(api.vulkan())
    }

    /// The raw API table.
    #[must_use]
    // Required by the wrapper's contract; not every accessor has a caller in
    // this tree yet.
    #[allow(dead_code)]
    pub fn as_ptr(self) -> *const ffi::VSVULKANAPI {
        self.0
    }
}

impl fmt::Debug for VsVulkanApi {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_tuple("VsVulkanApi").field(&self.0).finish()
    }
}

impl Deref for VsVulkanApi {
    type Target = ffi::VSVULKANAPI;

    fn deref(&self) -> &Self::Target {
        // SAFETY: the table outlives the library, and `Api::vulkan()` hands out
        // this same pointer for as long as the library is loaded.
        unsafe { &*self.0 }
    }
}

/// What [`VulkanCoreExt::vulkan_core_info()`] reports about a core's device.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct VulkanCoreInfo {
    /// The device name as the driver reports it.
    pub device_name: String,
    /// The device's total local memory in bytes.
    pub device_memory: i64,
    /// The driver's own budget for this process, in bytes.
    pub budget: i64,
    /// How much of the budget the core has already allocated.
    pub allocated: i64,
    /// The VRAM eviction limit the core enforces.
    pub limit: i64,
    /// Whether device memory is host memory.
    pub unified_memory: bool,
}

impl From<&ffi::VSVulkanCoreInfo> for VulkanCoreInfo {
    fn from(info: &ffi::VSVulkanCoreInfo) -> Self {
        Self {
            device_name: read_string(&info.device_name),
            device_memory: info.device_memory,
            budget: info.budget,
            allocated: info.allocated,
            limit: info.limit,
            unified_memory: info.unified_memory != 0,
        }
    }
}

/// One GPU resident frame, created by
/// [`VulkanCoreExt::new_gpu_video_frame()`] and freed when this value drops.
///
/// It wraps a `VSFrame` rather than the bindings' [`VideoFrame`] because the
/// bindings' type cannot be constructed from an owned pointer at this level:
/// the frame has to be freed with the core's own `freeFrame`, and the wrapper
/// is what knows to do that.  It is exactly the frame the core allocated
/// — same planes, same format, same properties — so every accessor here is the
/// core's own, and a plane handed to a recording is the plane the core would
/// hand to anyone else.
pub struct GpuFrame {
    handle: NonNull<ffi::VSFrame>,
    api: Api,
}

// SAFETY: VapourSynth frames are reference counted, and a GPU frame's contents
// are synchronized through its planes' producer pairs rather than through the
// wrapper.  The core hands frames to every worker thread of a filter by
// design; a frame that is only read after the submission that wrote it has
// completed is safe to share, which is what the pair guarantees.
unsafe impl Send for GpuFrame {}
// SAFETY: as above.  Sharing a frame permits only the read-only accessors
// below, and those are the core's own thread safe reads.
unsafe impl Sync for GpuFrame {}

impl GpuFrame {
    /// The raw `VSFrame`.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSFrame {
        self.handle.as_ptr()
    }

    /// Whether this frame lives in host or device memory.
    #[must_use]
    #[allow(dead_code)]
    pub fn residency(&self) -> ffi::VSNodeResidency {
        // SAFETY: the frame is live for the duration of the call.
        unsafe { (self.api.getFrameResidency)(self.handle.as_ptr().cast_const()) }
    }

    /// The plane's buffer and producer pair, when it is GPU resident.
    ///
    /// Returns [`None`] when the frame is not GPU resident, or the plane does
    /// not exist.
    #[must_use]
    #[allow(dead_code)]
    pub fn gpu_plane(&self, plane: i32) -> Option<ffi::VSVulkanPlaneInfo> {
        let vulkan = VsVulkanApi::new(&self.api);
        let mut info = MaybeUninit::<ffi::VSVulkanPlaneInfo>::uninit();

        // SAFETY: the frame is live and the plane index is validated by the
        // library, which reports failure through the return value.
        let result = unsafe {
            (vulkan.getGPUPlane)(self.handle.as_ptr().cast_const(), plane, info.as_mut_ptr())
        };

        (result == 0).then(|| unsafe { info.assume_init() })
    }

    /// The plane's row stride in bytes.
    #[must_use]
    #[allow(dead_code)]
    pub fn stride(&self, plane: i32) -> isize {
        // SAFETY: the frame is live for the duration of the call.
        unsafe { (self.api.getStride)(self.handle.as_ptr().cast_const(), plane) }
    }

    /// The plane's width in pixels.
    #[must_use]
    #[allow(dead_code)]
    pub fn frame_width(&self, plane: i32) -> i32 {
        // SAFETY: the frame is live for the duration of the call.
        unsafe { (self.api.getFrameWidth)(self.handle.as_ptr().cast_const(), plane) }
    }

    /// The plane's height in pixels.
    #[must_use]
    #[allow(dead_code)]
    pub fn frame_height(&self, plane: i32) -> i32 {
        // SAFETY: the frame is live for the duration of the call.
        unsafe { (self.api.getFrameHeight)(self.handle.as_ptr().cast_const(), plane) }
    }

    /// The frame's format.
    ///
    /// The borrow is tied to the frame, which owns the format it was created
    /// with.
    #[must_use]
    #[allow(dead_code)]
    pub fn get_video_format(&self) -> &ffi::VSVideoFormat {
        // SAFETY: the frame is live, and the format it points at lives as long
        // as the frame does.
        unsafe { &*(self.api.getVideoFrameFormat)(self.handle.as_ptr().cast_const()) }
    }
}

impl Drop for GpuFrame {
    fn drop(&mut self) {
        // SAFETY: the frame was created by `newGPUVideoFrame`, this value owns
        // exactly that reference, and `freeFrame` releases it.
        unsafe { (self.api.freeFrame)(self.handle.as_ptr().cast_const()) };
    }
}

/// The parts of the core's GPU API a filter needs.
///
/// `Core`, `CoreRef` and the frame types are foreign types, so their GPU
/// accessors live here as one extension trait implemented for the two core
/// shapes.  Every method but [`core_handle()`](Self::core_handle) is provided:
/// the only difference between an owned and a borrowed core is how the
/// `CoreHandle` is reached, and that is the one thing an implementor writes.
///
/// The methods are safe because each one either owns its result or reports a
/// failure, and none of them can be called with an argument that makes the
/// wrapper's invariants unsound.  Where the API itself is unsafe, the wrapper
/// keeps the unsafe operation behind a method that enforces its contract.
pub trait VulkanCoreExt {
    /// The common handle both core shapes dereference to.
    fn core_handle(&self) -> &CoreHandle;

    /// The Vulkan entry points the core resolved for this device.
    ///
    /// These are the same functions the core uses, resolved through its
    /// instance and device, so dispatching through them needs no loader of
    /// your own.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    fn vulkan_functions(&self) -> Result<&ffi::VSVulkanFunctions, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());

        let (functions, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getVulkanFunctions)(handle.as_ptr(), buffer, size)
        });

        // SAFETY: the table belongs to the core, which outlives the borrow,
        // and it is one struct of function pointers with no interior state.
        (!functions.is_null())
            .then(|| unsafe { &*functions })
            .ok_or_else(|| VulkanError::new(message, "the core has no Vulkan device"))
    }

    /// What the core knows about the device it is on.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    fn vulkan_core_info(&self) -> Result<VulkanCoreInfo, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());
        let mut info = MaybeUninit::<ffi::VSVulkanCoreInfo>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getVulkanCoreInfo)(handle.as_ptr(), info.as_mut_ptr(), buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(message, "the core has no Vulkan device"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(VulkanCoreInfo::from(unsafe { info.assume_init_ref() }))
    }

    /// The device handle and queue indices the core's device was created with.
    ///
    /// The wrapper needs the device to create and destroy its own pipeline
    /// objects; a filter reading this directly is doing something the wrapper
    /// has no typed call for.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the core has no GPU device.
    fn vulkan_handles(&self) -> Result<ffi::VSVulkanCoreHandles, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());
        let mut handles = MaybeUninit::<ffi::VSVulkanCoreHandles>::uninit();

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.getVulkanHandles)(handle.as_ptr(), handles.as_mut_ptr(), buffer, size)
        });

        if result != 0 {
            return Err(VulkanError::new(message, "the core has no Vulkan device"));
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(unsafe { handles.assume_init() })
    }

    /// Creates a GPU resident frame that only the device can write.
    ///
    /// The frame behaves exactly like a host frame everywhere else: same
    /// properties, same dimensions, same `std.FrameEval` and `std.ModifyFrame`
    /// support.  Only the dimensions and the format are fixed at creation.
    ///
    /// Returns [`None`] when the core has no GPU device, which is not the same
    /// as an allocation failure: the core reports that one by returning a frame
    /// with no planes, as it does for host frames.
    #[must_use]
    fn new_gpu_frame(
        &self,
        format: &ffi::VSVideoFormat,
        width: i32,
        height: i32,
    ) -> Option<GpuFrame> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());
        let ptr = unsafe { (api.newGPUVideoFrame)(format, width, height, null(), handle.as_ptr()) };

        NonNull::new(ptr).map(|handle| GpuFrame {
            handle,
            api: self.core_handle().api(),
        })
    }

    /// Creates a GPU resident video frame the bindings can hand back.
    ///
    /// The upstream bindings cannot build a [`VideoFrame`] from an owned
    /// `VSFrame`, so the frame is assembled through `newVideoFrame2`: a seed
    /// frame allocated here owns the planes, and the returned frame shares
    /// them.  Dropping the seed leaves the returned frame as their only owner,
    /// and the frame carries the GPU residency the filter's output flag
    /// promises.
    ///
    /// Returns [`None`] when the core has no GPU device or refuses the
    /// assembly.
    #[must_use]
    fn new_gpu_video_frame(
        &self,
        format: &ffi::VSVideoFormat,
        width: i32,
        height: i32,
        prop_src: Option<&VideoFrame>,
    ) -> Option<VideoFrame> {
        let seed = self.new_gpu_frame(format, width, height)?;
        let count = usize::try_from(format.num_planes).ok()?;
        if count == 0 || count > 3 {
            return None;
        }

        let sources = [seed.as_ptr().cast_const(); 3];
        let planes = [0, 1, 2];
        // SAFETY: every source is the same GPU resident seed frame, so the
        // residency is uniform and the dimensions and sample storage match the
        // requested format; `newVideoFrame2` shares those planes with the new
        // frame and copies `prop_src`'s properties onto it.
        let frame = unsafe {
            self.core_handle().new_video_frame2(
                format,
                width,
                height,
                &sources[..count],
                &planes[..count],
                prop_src,
            )
        };

        (frame.residency() == ffi::VSNodeResidency::GPU).then(|| frame.freeze())
    }

    /// Allocates a scratch buffer from the core's pool.
    ///
    /// Host visible requests come back persistently mapped, which is what
    /// [`GpuBuffer::mapped_slice()`] hands out.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the allocation failed.
    fn create_gpu_buffer(
        &self,
        size: vk::DeviceSize,
        usage: vk::BufferUsageFlags,
        required: vk::MemoryPropertyFlags,
        preferred: vk::MemoryPropertyFlags,
    ) -> Result<GpuBuffer, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());
        let mut info = MaybeUninit::<ffi::VSVulkanBufferInfo>::uninit();

        let (buffer, message) = with_error_buffer(|message_buffer, message_size| unsafe {
            (api.createGPUBuffer)(
                handle.as_ptr(),
                size,
                usage,
                required,
                preferred,
                info.as_mut_ptr(),
                message_buffer,
                message_size,
            )
        });

        NonNull::new(buffer)
            .map(|handle| GpuBuffer {
                handle,
                api,
                // SAFETY: the call reported success, so every field was written.
                info: unsafe { info.assume_init() },
            })
            .ok_or_else(|| VulkanError::new(message, "could not allocate a GPU buffer"))
    }

    /// Compiles a compute shader at runtime.
    ///
    /// Results are cached per core by source text, so repeating a kernel from
    /// many filter instances parses it once.  Specialize by concatenating a
    /// `#define` preamble in front of the kernel body; there is no include
    /// handler.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] with the compiler log if the source did not
    /// compile, including for a language the core does not know.
    fn compile_gpu_shader(
        &self,
        language: ffi::VSGPUShaderLanguage,
        source: &CStr,
    ) -> Result<Shader, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());

        let (shader, message) = with_error_buffer(|buffer, size| unsafe {
            (api.compileGPUShader)(
                handle.as_ptr(),
                language as i32,
                source.as_ptr(),
                buffer,
                size,
            )
        });

        NonNull::new(shader)
            .map(|handle| Shader { handle, api })
            .ok_or_else(|| VulkanError::new(message, "could not compile the shader"))
    }

    /// Creates an exec pool on one of the core's queues.
    ///
    /// The pool's context ring is sized from the core's worker thread count,
    /// two to eight contexts.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the pool could not be created.
    fn create_gpu_exec_pool(&self, queue: ffi::VSVulkanQueueType) -> Result<ExecPool, VulkanError> {
        let handle = self.core_handle();
        let api = VsVulkanApi::new(&handle.api());

        let (pool, message) = with_error_buffer(|buffer, size| unsafe {
            (api.createGPUExecPool)(handle.as_ptr(), queue as i32, buffer, size)
        });

        NonNull::new(pool)
            .map(|handle| ExecPool { handle, api })
            .ok_or_else(|| VulkanError::new(message, "could not create an exec pool"))
    }
}

impl VulkanCoreExt for CoreRef<'_> {
    fn core_handle(&self) -> &CoreHandle {
        self
    }
}

impl VulkanCoreExt for Core {
    fn core_handle(&self) -> &CoreHandle {
        self
    }
}

/// A shader compiled at runtime, holding the SPIR-V words.
///
/// Compilation goes through the core's statically embedded glslang, so no
/// device is touched and no optimizer runs.  The handle is independent of
/// everything else once returned and stays valid after the core that compiled
/// it is freed, which is why it can be held by a filter instance for its whole
/// life.
pub struct Shader {
    handle: NonNull<ffi::VSGPUShader>,
    api: VsVulkanApi,
}

// SAFETY: the shader holds compiled words and nothing device specific, and is
// reference counted inside the core's cache.
unsafe impl Send for Shader {}
// SAFETY: a compiled shader is immutable once returned, and the only thing a
// `&Shader` reaches is [`Shader::code()`], which reads the cached words.
unsafe impl Sync for Shader {}

impl Shader {
    /// The SPIR-V words.
    ///
    /// The slice is valid for as long as this shader is, and is empty only if
    /// the library reported no code, which it does not do for a live handle.
    #[must_use]
    pub fn code(&self) -> &[u32] {
        let mut size = 0usize;

        // SAFETY: the handle is live for as long as `self`, and the returned
        // words are owned by the shader.
        let ptr = unsafe { (self.api.getGPUShaderCode)(self.handle.as_ptr(), &raw mut size) };

        if ptr.is_null() || size < size_of::<u32>() {
            return &[];
        }

        // SAFETY: the library reports a byte count for a `uint32_t` array, so
        // the pointer is aligned and covers `size` readable bytes.
        unsafe { slice::from_raw_parts(ptr, size / size_of::<u32>()) }
    }
}

impl Drop for Shader {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value.
        unsafe { (self.api.freeGPUShader)(self.handle.as_ptr()) };
    }
}

/// A compute pipeline on the core's device.
///
/// Building one turns a compiled [`Shader`] into everything a dispatch needs: a
/// shader module, a descriptor set layout with one storage buffer per declared
/// binding, a pipeline layout carrying the push constants, and the pipeline
/// itself.  Each is created once and destroyed with this value.
///
/// Descriptors are pushed onto the command buffer rather than written into a
/// set allocated from a pool, so a pipeline owns no descriptor pool to reset
/// between frames.
pub struct ComputePipeline {
    vk: ffi::VSVulkanFunctions,
    device: vk::Device,
    shader_module: vk::ShaderModule,
    set_layout: vk::DescriptorSetLayout,
    layout: vk::PipelineLayout,
    pipeline: vk::Pipeline,
    /// How many storage buffer bindings the layout declares, numbered from
    /// zero.  A dispatch has to bind exactly this many buffers.
    bindings: u32,
    /// The push constant block's size in bytes, at offset zero.
    push_constant_size: u32,
}

// SAFETY: every field is a handle or a number, the handles are created on the
// core's device and immutable once built, and everything done with them is a
// Vulkan call, which the specification makes thread safe.  The device outlives
// the pipeline because the core owns it.
unsafe impl Send for ComputePipeline {}
// SAFETY: as above.  A pipeline holds no per thread state, which is what lets
// one instance serve every worker thread a filter is called on.
unsafe impl Sync for ComputePipeline {}

impl ComputePipeline {
    /// Starts building a pipeline that runs `shader`.
    #[must_use]
    pub fn builder<'a, C: VulkanCoreExt>(
        core: &'a C,
        shader: &'a Shader,
    ) -> ComputePipelineBuilder<'a> {
        ComputePipelineBuilder {
            core,
            shader,
            bindings: 0,
            push_constant_size: 0,
        }
    }

    /// How many storage buffer bindings the layout declares.
    #[must_use]
    #[allow(dead_code)]
    pub fn bindings(&self) -> u32 {
        self.bindings
    }

    /// Creates the shader module from the compiled words.
    ///
    /// The first step of [`ComputePipelineBuilder::build()`] is split out from
    /// the rest so that each `unsafe` block is one call with one contract, and
    /// so a failure leaves behind exactly what `Drop` destroys.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateShaderModule` or the driver refused to create the module.
    fn create_shader_module(&mut self, code: &[u32]) -> Result<(), VulkanError> {
        let create = self.vk.vkCreateShaderModule.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCreateShaderModule")
        })?;
        let info = vk::ShaderModuleCreateInfo::default().code(code);
        let mut module = vk::ShaderModule::null();

        // SAFETY: the module is created on the core's device from the create
        // info the specification asks for, and assigned into a field that
        // `Drop` destroys.  The words outlive the call.
        let result = unsafe { create(self.device, &info, null(), &mut module) };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::new(None, "could not create the shader module"));
        }

        self.shader_module = module;
        Ok(())
    }

    /// Creates the descriptor set layout: one pushed storage buffer binding per
    /// declared binding, numbered from zero.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateDescriptorSetLayout` or the driver refused to create the layout.
    fn create_set_layout(&mut self) -> Result<(), VulkanError> {
        let create = self.vk.vkCreateDescriptorSetLayout.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCreateDescriptorSetLayout")
        })?;

        let bindings: Vec<vk::DescriptorSetLayoutBinding<'_>> = (0..self.bindings)
            .map(|binding| {
                vk::DescriptorSetLayoutBinding::default()
                    .binding(binding)
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .descriptor_count(1)
                    .stage_flags(vk::ShaderStageFlags::COMPUTE)
            })
            .collect();
        let info = vk::DescriptorSetLayoutCreateInfo::default()
            // Pushed onto the command buffer rather than written into a set
            // taken from a pool, which the layout has to say.
            .flags(vk::DescriptorSetLayoutCreateFlags::PUSH_DESCRIPTOR_KHR)
            .bindings(&bindings);
        let mut set_layout = vk::DescriptorSetLayout::null();

        // SAFETY: as in [`Self::create_shader_module()`], and the bindings
        // array outlives the call.
        let result = unsafe { create(self.device, &info, null(), &mut set_layout) };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::new(
                None,
                "could not create the descriptor set layout",
            ));
        }

        self.set_layout = set_layout;
        Ok(())
    }

    /// Creates the pipeline layout: set 0, and the push constant block when one
    /// was declared.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreatePipelineLayout` or the driver refused to create the layout.
    fn create_layout(&mut self) -> Result<(), VulkanError> {
        let create = self.vk.vkCreatePipelineLayout.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCreatePipelineLayout")
        })?;

        let set_layouts = [self.set_layout];
        let mut info = vk::PipelineLayoutCreateInfo::default().set_layouts(&set_layouts);
        let ranges = [vk::PushConstantRange::default()
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .size(self.push_constant_size)];

        if self.push_constant_size != 0 {
            info = info.push_constant_ranges(&ranges);
        }

        let mut layout = vk::PipelineLayout::null();

        // SAFETY: as in [`Self::create_shader_module()`], and both arrays
        // outlive the call.
        let result = unsafe { create(self.device, &info, null(), &mut layout) };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::new(
                None,
                "could not create the pipeline layout",
            ));
        }

        self.layout = layout;
        Ok(())
    }

    /// Creates the compute pipeline itself, with `main` as its entry point.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core did not resolve
    /// `vkCreateComputePipelines` or the driver refused to create the pipeline.
    fn create_pipeline(&mut self) -> Result<(), VulkanError> {
        let create = self.vk.vkCreateComputePipelines.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCreateComputePipelines")
        })?;

        let stage = vk::PipelineShaderStageCreateInfo::default()
            .stage(vk::ShaderStageFlags::COMPUTE)
            .module(self.shader_module)
            .name(c"main");
        let info = vk::ComputePipelineCreateInfo::default()
            .stage(stage)
            .layout(self.layout);
        let mut pipeline = vk::Pipeline::null();

        // SAFETY: as in [`Self::create_shader_module()`].  No pipeline cache:
        // this is one pipeline, created once.
        let result = unsafe {
            create(
                self.device,
                vk::PipelineCache::null(),
                1,
                &info,
                null(),
                &mut pipeline,
            )
        };

        if result != vk::Result::SUCCESS {
            return Err(VulkanError::new(
                None,
                "could not create the compute pipeline",
            ));
        }

        self.pipeline = pipeline;
        Ok(())
    }
}

impl Drop for ComputePipeline {
    fn drop(&mut self) {
        // SAFETY: each handle was created on this device and is given back
        // exactly once.  Nothing is still using them: a pipeline only reaches
        // the GPU through a recording, and every recording is submitted or
        // abandoned before the filter instance holding this is dropped.
        unsafe {
            if let Some(destroy) = self.vk.vkDestroyPipeline
                && self.pipeline != vk::Pipeline::null()
            {
                destroy(self.device, self.pipeline, null());
            }
            if let Some(destroy) = self.vk.vkDestroyPipelineLayout
                && self.layout != vk::PipelineLayout::null()
            {
                destroy(self.device, self.layout, null());
            }
            if let Some(destroy) = self.vk.vkDestroyDescriptorSetLayout
                && self.set_layout != vk::DescriptorSetLayout::null()
            {
                destroy(self.device, self.set_layout, null());
            }
            if let Some(destroy) = self.vk.vkDestroyShaderModule
                && self.shader_module != vk::ShaderModule::null()
            {
                destroy(self.device, self.shader_module, null());
            }
        }
    }
}

/// Builds a [`ComputePipeline`].
///
/// Every [`storage_buffer()`](Self::storage_buffer) declares one binding, and
/// the order is the binding order: the first is `set = 0, binding = 0`, the
/// second is `binding = 1`, and so on.  [`ExecContext::dispatch()`] binds its
/// buffers in the same order, which makes a pipeline the one place that
/// decides what a shader sees.
pub struct ComputePipelineBuilder<'a> {
    core: &'a dyn VulkanCoreExt,
    shader: &'a Shader,
    bindings: u32,
    push_constant_size: u32,
}

impl ComputePipelineBuilder<'_> {
    /// Declares one `readonly buffer` or `writeonly buffer` binding.
    #[must_use]
    pub fn storage_buffer(mut self) -> Self {
        self.bindings = self.bindings.saturating_add(1);
        self
    }

    /// Declares a push constant block of `size` bytes at offset zero.
    ///
    /// Zero, which is the default, declares none.  A size that is not a
    /// multiple of four is rounded up, since the specification requires the
    /// range to be one and every member a shader can put in the block is at
    /// least that wide.  A block at another offset needs the raw path through
    /// the API table.
    #[must_use]
    pub fn push_constants(mut self, size: u32) -> Self {
        self.push_constant_size = size.div_ceil(4).saturating_mul(4);
        self
    }

    /// Creates the shader module, both layouts and the pipeline.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the core is on no device, when it did not
    /// resolve an entry point this needs, or when the driver refused to create
    /// one of the objects.  Whatever was created before the failure is
    /// destroyed.
    pub fn build(self) -> Result<ComputePipeline, VulkanError> {
        let vk = *self.core.vulkan_functions()?;
        let device = self.core.vulkan_handles()?.device;
        let mut pipeline = ComputePipeline {
            vk,
            device,
            shader_module: vk::ShaderModule::null(),
            set_layout: vk::DescriptorSetLayout::null(),
            layout: vk::PipelineLayout::null(),
            pipeline: vk::Pipeline::null(),
            bindings: self.bindings,
            push_constant_size: self.push_constant_size,
        };

        pipeline.create_shader_module(self.shader.code())?;
        pipeline.create_set_layout()?;
        pipeline.create_layout()?;
        pipeline.create_pipeline()?;

        Ok(pipeline)
    }
}

/// One plane of a GPU frame, declared and ready to bind into a dispatch.
///
/// [`ExecContext::read_plane()`] and [`ExecContext::write_plane()`] hand one
/// out, which is what ties a binding to a declaration: the call that returns the
/// buffer is the same one that makes the recording keep the frame alive until
/// the submission completes.
#[derive(Clone, Copy)]
pub struct DispatchPlane {
    info: ffi::VSVulkanPlaneInfo,
}

impl DispatchPlane {
    /// The bytes the plane's buffer covers, including row padding.
    #[must_use]
    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.info.buffer_size
    }

    /// The raw `VkBuffer`, for a binding recorded by hand.
    #[must_use]
    pub fn buffer(&self) -> vk::Buffer {
        self.info.buffer
    }
}

impl fmt::Debug for DispatchPlane {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("DispatchPlane")
            .field("buffer", &self.info.buffer)
            .field("buffer_size", &self.info.buffer_size)
            .field("ready_value", &self.info.ready_value)
            .finish_non_exhaustive()
    }
}

/// A scratch buffer from the core's memory pool.
///
/// This comes from the same sub allocator frame planes do, so it is counted
/// against the VRAM limit, predicted by the admission control of the thread
/// pool and recycled through the size buckets; allocating and destroying one per
/// frame is cheap by design.  Host visible requests come back persistently
/// mapped.
pub struct GpuBuffer {
    handle: NonNull<ffi::VSGPUBuffer>,
    api: VsVulkanApi,
    info: ffi::VSVulkanBufferInfo,
}

// SAFETY: the buffer is owned memory from the core's pool, not tied to a
// thread.  A submission that uses it is what has to keep it alive, and that is
// the `unsafe` contract of [`ExecContext::uses_buffer()`].
unsafe impl Send for GpuBuffer {}

impl GpuBuffer {
    /// The raw `VkBuffer`.
    #[must_use]
    pub fn buffer(&self) -> vk::Buffer {
        self.info.buffer
    }

    /// The buffer's size in bytes.
    #[must_use]
    #[allow(dead_code)]
    pub fn size(&self) -> vk::DeviceSize {
        self.info.size
    }

    /// The raw handle.
    #[must_use]
    pub fn as_ptr(&self) -> *mut ffi::VSGPUBuffer {
        self.handle.as_ptr()
    }

    /// The host mapping, when the allocation is host visible.
    ///
    /// # Safety
    ///
    /// The returned slice aliases memory the GPU may be reading or writing
    /// concurrently; the caller must ensure no submission that uses the buffer
    /// is in flight for as long as the borrow lasts.
    #[must_use]
    pub unsafe fn mapped_slice(&mut self) -> Option<&mut [u8]> {
        if self.info.mapped.is_null() {
            return None;
        }

        let size = usize::try_from(self.info.size).ok()?;

        // SAFETY: a host visible allocation is mapped for `size` bytes, and the
        // caller promises nothing else touches them.
        Some(unsafe { slice::from_raw_parts_mut(self.info.mapped.cast::<u8>(), size) })
    }
}

impl Drop for GpuBuffer {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value.  A buffer a
        // submission still uses would only have been handed over by an
        // `unsafe` call that promised the submission had completed.
        unsafe { (self.api.destroyGPUBuffer)(self.handle.as_ptr()) };
    }
}

// Set while a thread holds an [`ExecContext`].
//
// The one context per thread rule is the whole reason acquiring a recording is
// unsafe, and a recording that cannot outlive a call is exactly what makes it
// checkable, so it is checked here instead of documented.
std::thread_local! {
    static RECORDING: Cell<bool> = const { Cell::new(false) };
}

/// Clears this thread's recording claim again, however the context ends.
struct RecordingClaim;

impl RecordingClaim {
    /// Claims the calling thread, or reports that it already holds a recording.
    fn claim() -> Result<Self, VulkanError> {
        RECORDING.with(|recording| {
            if recording.replace(true) {
                return Err(VulkanError::new(
                    None,
                    "this thread already holds an exec context",
                ));
            }

            Ok(RecordingClaim)
        })
    }
}

impl Drop for RecordingClaim {
    fn drop(&mut self) {
        RECORDING.with(|recording| recording.set(false));
    }
}

/// A pooled recording ring on one of the core's queues.
///
/// Every worker thread that records a frame acquires its own context, records
/// and submits, so the pool's timeline values are the only ordering between
/// them.  A pool is created once per filter instance and freed in the filter's
/// free callback.
pub struct ExecPool {
    handle: NonNull<ffi::VSGPUExecPool>,
    api: VsVulkanApi,
}

// SAFETY: the pool is designed to be handed to every worker thread of a filter;
// each acquires its own context, and the ring is internally synchronized.
unsafe impl Send for ExecPool {}
// SAFETY: as above.  The waiting calls stay `unsafe` because of what they do to
// the calling thread's own state, not because of data races.
unsafe impl Sync for ExecPool {}

impl ExecPool {
    /// Claims a recording slot for this thread.
    ///
    /// The returned context is one recording, owned by the calling thread until
    /// it is submitted or abandoned, and it cannot be moved off that thread.
    /// The thread-local claim is taken before the ring is touched, so a nested
    /// acquire is refused rather than deadlocking.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if this thread already holds a recording or no
    /// context could be claimed.
    fn acquire(&self) -> Result<ExecContext<'_>, VulkanError> {
        let claim = RecordingClaim::claim()?;
        let (handle, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuExecAcquire)(self.handle.as_ptr(), buffer, size)
        });

        let Some(handle) = NonNull::new(handle) else {
            return Err(VulkanError::new(
                message,
                "could not acquire an exec context",
            ));
        };

        Ok(ExecContext {
            handle,
            api: self.api,
            _pool: PhantomData,
            claim: Some(claim),
        })
    }

    /// Runs `f` against one recording and submits it.
    ///
    /// This is the safe way to record: the context is claimed for this thread,
    /// handed to `f`, and submitted when `f` returns without an error.  It
    /// cannot outlive the call, so the one context per thread rule that makes
    /// the raw acquire unsafe is checked here, and a nested call is refused
    /// rather than being fatal.
    ///
    /// The returned value is the submission's point on the pool's timeline,
    /// which is what [`ExecPool::wait_value()`] takes.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if this thread already holds a recording, if no
    /// context could be claimed, if `f` returned one — in which case the
    /// recording is abandoned rather than submitted — or if the submission
    /// failed.
    pub fn record<F>(&self, f: F) -> Result<u64, VulkanError>
    where
        F: FnOnce(&mut ExecContext<'_>) -> Result<(), VulkanError>,
    {
        let mut context = self.acquire()?;

        if let Err(error) = f(&mut context) {
            context.abandon();
            return Err(error);
        }

        // SAFETY: the recording is submitted from the thread that claimed it,
        // and every declaration made against it came from this API: frames were
        // declared by the calls that handed out their planes, and nothing was
        // handed over by hand.
        unsafe { context.submit() }
    }

    /// Waits for one value on the pool's timeline.
    ///
    /// This is the same wait as on the pool's own timeline, with the value
    /// check the pool can make and a timeline you signal yourself cannot: a
    /// value past the newest submission is fatal, since no consumer would ever
    /// signal it.
    ///
    /// A GPU reset makes this return [`ffi::VSGPUDrainResult::DeviceLost`],
    /// which is the one result that makes everything the submission named safe
    /// to destroy; [`ffi::VSGPUDrainResult::Incomplete`] leaves it queued and
    /// untouchable.
    ///
    /// # Safety
    ///
    /// The value must be one this pool submitted: a [`ExecContext::submit()`]
    /// result, or anything earlier.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the wait could not be established, or if the
    /// library reported a drain result outside the documented set.
    pub unsafe fn wait_value(&self, value: u64) -> Result<ffi::VSGPUDrainResult, VulkanError> {
        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (self.api.gpuExecWaitValue)(self.handle.as_ptr(), value, buffer, size)
        });

        drain_result(result).ok_or_else(|| {
            VulkanError::new(
                message,
                "waiting on an exec pool reported an unknown result",
            )
        })
    }
}

impl Drop for ExecPool {
    fn drop(&mut self) {
        // SAFETY: the handle is live and owned by this value, and no context can
        // outlive the pool because `ExecContext` borrows it.
        unsafe { (self.api.freeGPUExecPool)(self.handle.as_ptr()) };
    }
}

/// One recording of an [`ExecPool`], claimed by one thread.
///
/// The context is consumed by [`ExecContext::submit()`],
/// [`ExecContext::abandon()`] or by dropping it, which abandons the recording.
/// It is not `Send`: recording, submitting and abandoning all have to happen on
/// the thread that acquired it.
pub struct ExecContext<'pool> {
    handle: NonNull<ffi::VSGPUExecContext>,
    api: VsVulkanApi,
    /// Keeps the pool alive for as long as a recording is outstanding, and keeps
    /// the context off other threads.
    _pool: PhantomData<&'pool ExecPool>,
    /// The thread-local claim, taken by [`ExecPool::acquire()`].
    ///
    /// It is an `Option` so that [`ExecContext::submit()`] can release it by
    /// hand before forgetting the context the C API has already consumed.
    claim: Option<RecordingClaim>,
}

impl ExecContext<'_> {
    /// The command buffer being recorded.
    ///
    /// Put anything Vulkan allows into it, then submit.
    #[must_use]
    pub fn command_buffer(&self) -> vk::CommandBuffer {
        // SAFETY: the context is live until it is submitted or abandoned.
        unsafe { (self.api.gpuExecCommandBuffer)(self.handle.as_ptr()) }
    }

    /// Declares that this submission reads `frame` and hands back the plane to
    /// bind.
    ///
    /// Every plane's producer pair becomes a device side wait, and the frame is
    /// kept alive until the submission completes, which is what keeps a
    /// recording from binding a plane it never declared.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    #[allow(dead_code)]
    pub fn read_plane(
        &mut self,
        frame: &GpuFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let Some(info) = frame.gpu_plane(plane) else {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        };

        // SAFETY: the recording is live, the frame is borrowed for the call,
        // and the context takes its own reference, so the frame's lifetime from
        // here is the core's business.
        unsafe {
            (self.api.gpuExecReadsFrame)(self.handle.as_ptr(), frame.as_ptr().cast_const());
        }

        Ok(DispatchPlane { info })
    }

    /// Declares that this submission reads an input frame and hands back the
    /// plane to bind.
    ///
    /// Input frames arrive as the bindings' [`VideoFrame`] — an upstream
    /// `std.GPUUpload` produced them, not this wrapper — so this is
    /// [`read_plane()`](Self::read_plane) for a frame this module did not
    /// create.  The recording keeps the frame alive exactly as it does a
    /// [`GpuFrame`].
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    pub fn read_video_frame(
        &mut self,
        frame: &VideoFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let ptr = frame.as_ptr();
        let mut info = MaybeUninit::<ffi::VSVulkanPlaneInfo>::uninit();

        // SAFETY: the frame is live and the plane index is validated by the
        // library, which reports failure through the return value.
        let result = unsafe { (self.api.getGPUPlane)(ptr.cast_const(), plane, info.as_mut_ptr()) };
        if result != 0 {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        }

        // SAFETY: the recording is live, the frame is borrowed for the call,
        // and the context takes its own reference, so the frame's lifetime from
        // here is the core's business.
        unsafe {
            (self.api.gpuExecReadsFrame)(self.handle.as_ptr(), ptr.cast_const());
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(DispatchPlane {
            info: unsafe { info.assume_init() },
        })
    }

    /// Declares that this submission writes a plane of a frame the bindings
    /// own and hands back the plane to bind.
    ///
    /// This is [`write_plane()`](Self::write_plane) for an output frame built
    /// through [`VulkanCoreExt::new_gpu_video_frame()`], which the bindings
    /// cannot represent as a [`GpuFrame`].  Submitting publishes the pool's
    /// producer pair on the plane exactly as it does for a scratch frame.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    pub fn write_video_frame(
        &mut self,
        frame: &mut VideoFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let ptr = frame.as_ptr();
        let mut info = MaybeUninit::<ffi::VSVulkanPlaneInfo>::uninit();

        // SAFETY: the frame is live and the plane index is validated by the
        // library, which reports failure through the return value.
        let result = unsafe { (self.api.getGPUPlane)(ptr.cast_const(), plane, info.as_mut_ptr()) };
        if result != 0 {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        }

        // SAFETY: the recording is live, the frame is borrowed mutably for the
        // call, and the plane is the caller's to declare.
        unsafe {
            (self.api.gpuExecWritesPlane)(self.handle.as_ptr(), ptr, plane);
        }

        // SAFETY: the call reported success, so every field was written.
        Ok(DispatchPlane {
            info: unsafe { info.assume_init() },
        })
    }

    /// Declares that this submission writes `plane` and hands it back to bind.
    ///
    /// Submitting publishes the pool's `(timeline, value)` on the plane as its
    /// producer pair, which is how every later consumer knows when it is ready
    /// without waiting for anything.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the frame is not GPU resident or the plane
    /// does not exist.
    #[allow(dead_code)]
    pub fn write_plane(
        &mut self,
        frame: &mut GpuFrame,
        plane: i32,
    ) -> Result<DispatchPlane, VulkanError> {
        let Some(info) = frame.gpu_plane(plane) else {
            return Err(VulkanError::new(
                None,
                "the frame is not GPU resident, or has no plane there",
            ));
        };

        // SAFETY: the recording is live, the frame is borrowed mutably for the
        // call, and the plane is the caller's to declare.
        unsafe {
            (self.api.gpuExecWritesPlane)(self.handle.as_ptr(), frame.as_ptr(), plane);
        }

        Ok(DispatchPlane { info })
    }

    /// Records one compute dispatch into the recording.
    ///
    /// `buffers` are bound to the pipeline's bindings in order, and there have
    /// to be exactly as many as the pipeline declared: a shader reading a
    /// binding nothing was written to reads whatever the driver left there, so
    /// a short list is refused instead of recorded.
    ///
    /// `push_constants` is the block at offset zero and must fit the layout.
    /// `groups` is the workgroup count, so the invocations are that times the
    /// shader's `local_size`.  Nothing waits for the dispatch here: the
    /// submission's timeline value is what says when it has run.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] when the buffer count or the push constant block
    /// does not match the pipeline, or when the core did not resolve an entry
    /// point this needs.
    pub fn dispatch(
        &mut self,
        pipeline: &ComputePipeline,
        buffers: &[DispatchPlane],
        push_constants: &[u8],
        groups: [u32; 3],
    ) -> Result<(), VulkanError> {
        if u32::try_from(buffers.len()).unwrap_or(u32::MAX) != pipeline.bindings {
            return Err(VulkanError::new(
                None,
                "the dispatch binds a different number of buffers than the pipeline declares",
            ));
        }

        let size = u32::try_from(push_constants.len()).unwrap_or(u32::MAX);
        if size % 4 != 0 {
            return Err(VulkanError::new(
                None,
                "the push constants are not a whole number of 32 bit values",
            ));
        }
        if size > pipeline.push_constant_size {
            return Err(VulkanError::new(
                None,
                "the push constants do not fit the pipeline layout",
            ));
        }

        let bind = pipeline
            .vk
            .vkCmdBindPipeline
            .ok_or_else(|| VulkanError::new(None, "the core did not resolve vkCmdBindPipeline"))?;
        let push_descriptors = pipeline.vk.vkCmdPushDescriptorSet.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCmdPushDescriptorSet")
        })?;
        let push_constants_call = pipeline.vk.vkCmdPushConstants2.ok_or_else(|| {
            VulkanError::new(None, "the core did not resolve vkCmdPushConstants2")
        })?;
        let dispatch_call = pipeline
            .vk
            .vkCmdDispatch
            .ok_or_else(|| VulkanError::new(None, "the core did not resolve vkCmdDispatch"))?;

        let infos: Vec<vk::DescriptorBufferInfo> = buffers
            .iter()
            .map(|plane| {
                vk::DescriptorBufferInfo::default()
                    .buffer(plane.info.buffer)
                    .offset(0)
                    .range(plane.info.buffer_size)
            })
            .collect();
        let writes: Vec<vk::WriteDescriptorSet<'_>> = (0..buffers.len())
            .map(|binding| {
                vk::WriteDescriptorSet::default()
                    // Push descriptors have no set to write into; the field is
                    // ignored, and the binding lands on the layout's set 0.
                    .dst_set(vk::DescriptorSet::null())
                    .dst_binding(u32::try_from(binding).unwrap_or(0))
                    .descriptor_type(vk::DescriptorType::STORAGE_BUFFER)
                    .buffer_info(&infos[binding..=binding])
            })
            .collect();
        let constants = vk::PushConstantsInfoKHR::default()
            .layout(pipeline.layout)
            .stage_flags(vk::ShaderStageFlags::COMPUTE)
            .offset(0)
            .values(push_constants);
        let cb = self.command_buffer();

        // SAFETY: the recording is live and is the only one this thread holds.
        // Every buffer bound here was handed out by `read_plane` or
        // `write_plane` and so was declared on this recording, which is what
        // keeps the frames behind them alive until the submission completes.
        // The push constant bytes and the descriptor infos are only read during
        // the call.
        unsafe {
            bind(cb, vk::PipelineBindPoint::COMPUTE, pipeline.pipeline);
            if !push_constants.is_empty() {
                push_constants_call(cb, &constants);
            }
            if !writes.is_empty() {
                push_descriptors(
                    cb,
                    vk::PipelineBindPoint::COMPUTE,
                    pipeline.layout,
                    0,
                    u32::try_from(writes.len()).unwrap_or(0),
                    writes.as_ptr(),
                );
            }
            dispatch_call(cb, groups[0], groups[1], groups[2]);
        }

        Ok(())
    }

    /// Hands a scratch buffer to the context.
    ///
    /// Ownership transfers, and the context destroys the buffer once the
    /// submission completes, so it must not be destroyed by hand afterwards.
    ///
    /// # Safety
    ///
    /// Surrendering a [`GpuBuffer`] means its `Drop` never runs; the caller
    /// gives up ownership for real by moving it here.  Nothing else may hold a
    /// copy of the handle.
    pub unsafe fn uses_buffer(&self, buffer: GpuBuffer) {
        let ptr = buffer.as_ptr();

        // The context takes ownership of the handle, so the local value must
        // not destroy it when this call returns.
        forget(buffer);

        // SAFETY: the context is live and takes ownership of the handle.
        unsafe { (self.api.gpuExecUsesBuffer)(self.handle.as_ptr(), ptr) };
    }

    /// Gives up the recording without submitting.
    ///
    /// This is what dropping the context does; the method exists to make the
    /// intent explicit at the call site.
    pub fn abandon(self) {
        drop(self);
    }

    /// Ends recording and submits, publishing the producer pairs.
    ///
    /// The timeline value is allocated inside the queue lock so that signals
    /// reach the queue in increasing order.  The returned value is what this
    /// submission signals on the pool's timeline; pass it to
    /// [`ExecPool::wait_value()`] to wait for exactly this submission.
    ///
    /// Do not wait on the timeline with `vkWaitSemaphores` yourself: a bare
    /// wait cannot distinguish a completed dispatch from a reset that
    /// force-signalled the timeline, and reports the second as the first.
    ///
    /// # Safety
    ///
    /// The context must be submitted from the thread that acquired it.  Every
    /// declaration made against it must hold: frames it wrote must own their
    /// planes, and a buffer it took ownership of must not be dropped by hand.
    ///
    /// # Errors
    ///
    /// Returns [`VulkanError`] if the submission failed.  The context is
    /// consumed either way, as it is in the C API.
    pub unsafe fn submit(mut self) -> Result<u64, VulkanError> {
        let api = self.api;
        let handle = self.handle.as_ptr();
        let mut value = 0u64;

        let (result, message) = with_error_buffer(|buffer, size| unsafe {
            (api.gpuExecSubmit)(handle, &raw mut value, buffer, size)
        });

        // The context is consumed whether the submission succeeded or not, so
        // the `Drop` that would abandon it must not run -- and the recording
        // claim has to be released by hand before `self` is forgotten.
        drop(self.claim.take());
        forget(self);

        if result != 0 {
            return Err(VulkanError::new(message, "could not submit a recording"));
        }

        Ok(value)
    }
}

impl Drop for ExecContext<'_> {
    fn drop(&mut self) {
        // A recording that was neither submitted nor abandoned would pin its
        // ring slot forever, so give it back.
        //
        // SAFETY: the context is live and owned by this value.
        unsafe { (self.api.gpuExecAbandon)(self.handle.as_ptr()) };
    }
}
