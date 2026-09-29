//! `Deblur`: an edge-masked sharpen, with the two candidates the reference
//! implements and the halo clamp and blend that limit them.
//!
//! Only luma moves. Gray takes the delta on its single plane, YUV on its luma
//! plane with chroma left alone, and RGB as one equal offset per pixel limited
//! to the gamut the pixel has left, which is what preserves the channel
//! differences. The kernels themselves live in [`crate::deblur`] and work in
//! 8 bit code values, so this layer is only the plane bookkeeping.
//!
//! Float samples are read as the reference's `[0, 1]` range, so the blend clips
//! them to that range exactly as `nmanga.deblur` clips its own.

use std::ffi::{CStr, c_void};
use std::sync::Mutex;
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use vapoursynth4_rs::frame::{FrameContext, VideoFormat, VideoFrame};
use vapoursynth4_rs::map::MapRef;
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{ColorFamily, SampleType, core::CoreRef, ffi, key};

use crate::deblur::{DeblurError, Method, Params, Workspace};
use crate::error::{NImagesError, Result};

use super::{
    Accept, FrameTrace, add_filter, check_frame_format, checked_info, describe_frame, input_failed,
    max_sample_value, plane_shape, read_clip, read_float, read_int, report_settings_once,
};

/// The 8 bit code value range the kernels work in.
const CODE_VALUES: f32 = 255.0;
/// Default Gaussian sigma in pixels.
const DEFAULT_RADIUS: f64 = 0.8;
/// Default number of refinement passes for the deconvolution.
const DEFAULT_ITERATIONS: i64 = 6;
/// Default edge threshold in 8 bit levels.
const DEFAULT_THRESHOLD: f64 = 2.0;
/// Default excursion past the local extremes, in 8 bit steps.
const DEFAULT_OVERSHOOT: f64 = 0.0;
/// The luma weights the reference uses on the gamma-encoded values.
const LUMA_WEIGHTS: [f32; 3] = [0.2126, 0.7152, 0.0722];

/// Sharpens the luma plane of every frame.
pub struct Deblur {
    source: VideoNode,
    params: Params,
    /// One workspace per concurrently evaluated frame.
    workspace: Mutex<Vec<Workspace>>,
    debug: bool,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for Deblur {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Deblur";
    const ARGS: &'static CStr = c"clip:vnode;method:int:opt;radius:float:opt;strength:float:opt;iterations:int:opt;threshold:float:opt;overshoot:float:opt;debug:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "Deblur")?;
        let info = checked_info(&source, "Deblur", Accept::Deblur)?;

        let method = match read_int(&input, key!(c"method"))?.unwrap_or(0) {
            0 => Method::Deconvolution,
            1 => Method::EdgeSharpen,
            other => {
                return Err(NImagesError::new(format!(
                    "Deblur: method must be 0 (deconvolution) or 1 (edge-masked unsharp), \
                     got {other}"
                )));
            }
        };

        // `method=1` has no refinement loop, so an `iterations` outside the
        // range the deconvolution accepts is ignored with it rather than
        // refused.
        let iterations = if method == Method::EdgeSharpen {
            0
        } else {
            let requested = read_int(&input, key!(c"iterations"))?.unwrap_or(DEFAULT_ITERATIONS);
            u32::try_from(requested).map_err(|_| iterations_error(requested))?
        };

        let params = Params {
            method,
            radius: read_float(&input, key!(c"radius"))?.unwrap_or(DEFAULT_RADIUS) as f32,
            strength: read_float(&input, key!(c"strength"))?
                .unwrap_or(f64::from(method.default_strength())) as f32,
            iterations,
            threshold: read_float(&input, key!(c"threshold"))?.unwrap_or(DEFAULT_THRESHOLD) as f32,
            overshoot: read_float(&input, key!(c"overshoot"))?.unwrap_or(DEFAULT_OVERSHOOT) as f32,
        };
        params.validate().map_err(deblur_error)?;

        let debug = read_int(&input, key!(c"debug"))?.unwrap_or(0) != 0;

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self {
                source,
                params,
                workspace: Mutex::new(Vec::new()),
                debug,
                reported: AtomicBool::new(false),
            },
            dependency,
        );
        Ok(())
    }

    fn get_frame(
        &self,
        n: i32,
        activation_reason: ffi::VSActivationReason,
        _frame_data: *mut *mut c_void,
        mut frame_ctx: FrameContext,
        mut core: CoreRef,
    ) -> Result<Option<Self::FrameType>> {
        match activation_reason {
            ffi::VSActivationReason::Initial => {
                frame_ctx.request_frame_filter(n, &self.source);
                Ok(None)
            }
            ffi::VSActivationReason::AllFramesReady => {
                let mut trace = FrameTrace::new(self.debug, "Deblur");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "Deblur", Accept::Deblur)?;
                let format = input.get_video_format().clone();
                let family = format.color_family;
                let convert = Convert::new(&format)?;

                report_settings_once(
                    self.debug,
                    &self.reported,
                    &mut core,
                    format_args!(
                        "Deblur: {} input={}",
                        self.describe(),
                        describe_frame(&input)
                    ),
                );

                let width = usize::try_from(input.frame_width(0))
                    .map_err(|_| NImagesError::new("Deblur: plane 0 has a negative width"))?;
                let height = usize::try_from(input.frame_height(0))
                    .map_err(|_| NImagesError::new("Deblur: plane 0 has a negative height"))?;

                // The copy carries the pixels and the properties, so the planes
                // this filter does not touch come back byte for byte.
                let mut output = core.copy_frame(&input);
                if width == 0 || height == 0 {
                    return Ok(Some(output));
                }

                let mut workspace = self.take(width, height)?;

                let mark = Instant::now();
                read_luma(&input, family, workspace.luma_mut())?;
                trace.mark("luma", mark);

                let mark = Instant::now();
                workspace
                    .restore(width, height, &self.params)
                    .map_err(deblur_error)?;
                trace.mark("restore", mark);

                let mark = Instant::now();
                match family {
                    ColorFamily::RGB => write_rgb(
                        &input,
                        &mut output,
                        &convert,
                        workspace.luma(),
                        workspace.restored(),
                    )?,
                    _ => write_plane(&mut output, 0, &convert, |index| {
                        workspace.restored().get(index).copied().unwrap_or(0.0)
                    })?,
                }
                trace.mark("write", mark);

                trace.emit(
                    &mut core,
                    n,
                    format_args!("method={} ", self.params.method.name()),
                );
                self.put(workspace);

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Deblur")),
        }
    }
}

impl Deblur {
    /// The settings the debug line reports, without the parameters the method in
    /// effect ignores.
    fn describe(&self) -> String {
        let params = &self.params;
        if params.method == Method::EdgeSharpen {
            format!(
                "method={} radius={} strength={} threshold={} overshoot={}",
                params.method.name(),
                params.radius,
                params.strength,
                params.threshold,
                params.overshoot
            )
        } else {
            format!(
                "method={} radius={} strength={} iterations={} threshold={} overshoot={}",
                params.method.name(),
                params.radius,
                params.strength,
                params.iterations,
                params.threshold,
                params.overshoot
            )
        }
    }

    /// Takes one workspace out of the pool, sized for this frame.
    fn take(&self, width: usize, height: usize) -> Result<Workspace> {
        let mut free = lock(&self.workspace);
        let mut workspace = free.pop().unwrap_or_default();
        drop(free);
        workspace.prepare(width, height).map_err(deblur_error)?;
        Ok(workspace)
    }

    /// Returns a workspace for the next frame to reuse.
    fn put(&self, workspace: Workspace) {
        lock(&self.workspace).push(workspace);
    }
}

/// Locks the pool, ignoring poisoning: the buffers hold no invariant a panic
/// could have broken.
fn lock<T>(mutex: &Mutex<T>) -> std::sync::MutexGuard<'_, T> {
    mutex.lock().unwrap_or_else(|error| error.into_inner())
}

/// Wraps a kernel failure with the filter's name.
fn deblur_error(error: DeblurError) -> NImagesError {
    NImagesError::new(format!("Deblur: {}", error.message()))
}

/// The error for a refinement count the deconvolution cannot use.
fn iterations_error(requested: i64) -> NImagesError {
    NImagesError::new(format!(
        "Deblur: iterations must be between 0 and {}, got {requested}",
        crate::deblur::MAX_ITERATIONS
    ))
}

/// How one frame's samples turn into 8 bit code values and back.
#[derive(Clone, Copy)]
struct Convert {
    /// Samples to code values.
    scale: f32,
    /// Code values back to samples.
    inverse: f64,
    /// Bytes one sample occupies.
    bytes: usize,
    /// Whether the samples are floats, which have no rounding step.
    float: bool,
    /// The largest integer code value. Unused by the float path.
    max_value: f64,
}

impl Convert {
    /// Resolves the conversion for one format, which the caller has already
    /// checked against `Accept::Deblur`.
    fn new(format: &VideoFormat) -> Result<Self> {
        let bits = format.bits_per_sample;
        if format.sample_type == SampleType::Float {
            if bits != 32 {
                return Err(NImagesError::new(
                    "Deblur: only 32 bit float formats are supported",
                ));
            }
            return Ok(Self {
                scale: CODE_VALUES,
                inverse: 1.0 / f64::from(CODE_VALUES),
                bytes: 4,
                float: true,
                max_value: 0.0,
            });
        }

        let max_value = f64::from(max_sample_value(bits)?);
        let code_values = f64::from(CODE_VALUES);
        Ok(Self {
            scale: (code_values / max_value) as f32,
            inverse: max_value / code_values,
            bytes: if bits <= 8 { 1 } else { 2 },
            float: false,
            max_value,
        })
    }

    /// One sample of one row, in code values.
    #[inline]
    fn read(&self, row: &[u8], index: usize) -> f32 {
        let offset = index * self.bytes;
        match self.bytes {
            1 => row
                .get(offset)
                .copied()
                .map_or(0.0, |value| f32::from(value) * self.scale),
            2 => row
                .get(offset..offset + 2)
                .and_then(|pair| <[u8; 2]>::try_from(pair).ok())
                .map_or(0.0, |pair| f32::from(u16::from_ne_bytes(pair)) * self.scale),
            _ => row
                .get(offset..offset + 4)
                .and_then(|quad| <[u8; 4]>::try_from(quad).ok())
                .map_or(0.0, |quad| f32::from_ne_bytes(quad) * self.scale),
        }
    }

    /// Writes one code value as a sample of one row.
    ///
    /// Integer samples are rounded the way the reference rounds, ties to even,
    /// and clamped to the sample range.
    #[inline]
    fn write(&self, row: &mut [u8], index: usize, code: f32) {
        let offset = index * self.bytes;
        let scaled = f64::from(code) * self.inverse;
        match self.bytes {
            1 => {
                let value = if self.float {
                    scaled
                } else {
                    scaled.round_ties_even().clamp(0.0, self.max_value)
                };
                if let Some(target) = row.get_mut(offset) {
                    *target = value as u8;
                }
            }
            2 => {
                let value = scaled.round_ties_even().clamp(0.0, self.max_value) as u16;
                if let Some(target) = row.get_mut(offset..offset + 2) {
                    target.copy_from_slice(&value.to_ne_bytes());
                }
            }
            _ => {
                let value = scaled as f32;
                if let Some(target) = row.get_mut(offset..offset + 4) {
                    target.copy_from_slice(&value.to_ne_bytes());
                }
            }
        }
    }
}

/// Writes the luma the kernels produced back into one plane.
fn write_plane<F: Fn(usize) -> f32>(
    output: &mut VideoFrame,
    plane: i32,
    convert: &Convert,
    code: F,
) -> Result<()> {
    let width = usize::try_from(output.frame_width(plane))
        .map_err(|_| NImagesError::new(format!("Deblur: plane {plane} has a negative width")))?;
    let (stride, height) = plane_shape(output, plane)?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let row_bytes = width
        .checked_mul(convert.bytes)
        .ok_or_else(|| NImagesError::new(format!("Deblur: plane {plane} row is too large")))?;
    if row_bytes > stride {
        return Err(NImagesError::new(format!(
            "Deblur: plane {plane} needs {row_bytes} row bytes but its stride is {stride}"
        )));
    }
    let length = stride.checked_mul(height).ok_or_else(|| {
        NImagesError::new("Deblur: the output plane is larger than the address space")
    })?;
    let pointer = output.plane_mut(plane);
    if pointer.is_null() {
        return Err(NImagesError::new(format!(
            "Deblur: plane {plane} is not writable"
        )));
    }

    // SAFETY: the output frame owns a writable plane buffer of `stride * height`
    // bytes, the caller holds the frame, and every offset below stays inside
    // `row_bytes` of its own row.
    let bytes = unsafe { std::slice::from_raw_parts_mut(pointer, length) };
    for row in 0..height {
        let start = row * stride;
        let Some(target) = bytes.get_mut(start..start + row_bytes) else {
            return Err(NImagesError::new(format!(
                "Deblur: plane {plane} is smaller than its reported stride"
            )));
        };
        for column in 0..width {
            convert.write(target, column, code(row * width + column));
        }
    }
    Ok(())
}

/// Reads the kernels' luma plane from a frame, in 8 bit code values.
fn read_luma(frame: &VideoFrame, family: ColorFamily, luma: &mut [f32]) -> Result<()> {
    let format = frame.get_video_format();
    let convert = Convert::new(format)?;
    if family == ColorFamily::RGB {
        for (plane, weight) in LUMA_WEIGHTS.iter().enumerate() {
            read_plane(frame, plane as i32, &convert, *weight, plane > 0, luma)?;
        }
    } else {
        read_plane(frame, 0, &convert, 1.0, false, luma)?;
    }
    Ok(())
}

/// Reads one plane into `target` as code values, weighted and optionally added.
fn read_plane(
    frame: &VideoFrame,
    plane: i32,
    convert: &Convert,
    weight: f32,
    add: bool,
    target: &mut [f32],
) -> Result<()> {
    let width = usize::try_from(frame.frame_width(plane))
        .map_err(|_| NImagesError::new(format!("Deblur: plane {plane} has a negative width")))?;
    let (stride, height) = plane_shape(frame, plane)?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    let row_bytes = width
        .checked_mul(convert.bytes)
        .ok_or_else(|| NImagesError::new(format!("Deblur: plane {plane} row is too large")))?;
    if row_bytes > stride {
        return Err(NImagesError::new(format!(
            "Deblur: plane {plane} needs {row_bytes} row bytes but its stride is {stride}"
        )));
    }
    if target.len() < width * height {
        return Err(NImagesError::new(
            "Deblur: the luma plane is smaller than the frame",
        ));
    }
    let length = stride.checked_mul(height).ok_or_else(|| {
        NImagesError::new("Deblur: the input plane is larger than the address space")
    })?;
    let pointer = frame.plane(plane);
    if pointer.is_null() {
        return Err(NImagesError::new(format!(
            "Deblur: plane {plane} is not readable"
        )));
    }

    // SAFETY: VapourSynth guarantees a readable plane buffer of `stride * height`
    // bytes for as long as the frame is alive, and the caller holds the frame.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    for row in 0..height {
        let start = row * stride;
        let Some(source) = bytes.get(start..start + row_bytes) else {
            return Err(NImagesError::new(format!(
                "Deblur: plane {plane} is smaller than its reported stride"
            )));
        };
        let out_start = row * width;
        let Some(out) = target.get_mut(out_start..out_start + width) else {
            return Err(NImagesError::new(
                "Deblur: the luma plane is smaller than the frame",
            ));
        };
        for (column, value) in out.iter_mut().enumerate() {
            let sample = convert.read(source, column) * weight;
            if add {
                *value += sample;
            } else {
                *value = sample;
            }
        }
    }
    Ok(())
}

/// Writes the RGB planes with one equal offset per pixel.
///
/// The offset is limited to the gamut the pixel has left, which is what keeps
/// the channel differences instead of clipping each channel on its own.
fn write_rgb(
    input: &VideoFrame,
    output: &mut VideoFrame,
    convert: &Convert,
    luma: &[f32],
    restored: &[f32],
) -> Result<()> {
    let mut sources: [&[u8]; 3] = [&[]; 3];
    let mut stride = 0usize;
    let mut width = 0usize;
    let mut height = 0usize;
    for plane in 0..3i32 {
        let (bytes, plane_stride) = plane_bytes(input, plane)?;
        let row_width = usize::try_from(input.frame_width(plane))
            .map_err(|_| NImagesError::new("Deblur: an RGB plane has a negative width"))?;
        let row_height = usize::try_from(input.frame_height(plane))
            .map_err(|_| NImagesError::new("Deblur: an RGB plane has a negative height"))?;
        if plane > 0 && (row_width != width || row_height != height || plane_stride != stride) {
            return Err(NImagesError::new(
                "Deblur: the RGB planes do not share one geometry",
            ));
        }
        width = row_width;
        height = row_height;
        stride = plane_stride;
        if let Some(slot) = sources.get_mut(plane as usize) {
            *slot = bytes;
        }
    }
    if width == 0 || height == 0 {
        return Ok(());
    }
    let row_bytes = width
        .checked_mul(convert.bytes)
        .ok_or_else(|| NImagesError::new("Deblur: an RGB plane row is too large"))?;
    if row_bytes > stride {
        return Err(NImagesError::new(format!(
            "Deblur: an RGB plane needs {row_bytes} row bytes but its stride is {stride}"
        )));
    }
    let mut targets = [std::ptr::null_mut(); 3];
    for (plane, target) in targets.iter_mut().enumerate() {
        let pointer = output.plane_mut(plane as i32);
        if pointer.is_null() {
            return Err(NImagesError::new("Deblur: an RGB plane is not writable"));
        }
        *target = pointer;
    }

    for row in 0..height {
        let start = row * stride;
        let mut rows: [&[u8]; 3] = [&[]; 3];
        for (plane, source) in sources.iter().enumerate() {
            let Some(source) = source.get(start..start + row_bytes) else {
                return Err(NImagesError::new(
                    "Deblur: an RGB plane is smaller than its reported stride",
                ));
            };
            if let Some(slot) = rows.get_mut(plane) {
                *slot = source;
            }
        }
        // SAFETY: every pointer comes from a different plane of the same live
        // output frame, so the three rows never overlap and the input frame is a
        // distinct allocation. Each frame reports its own stride and height, and
        // `row_bytes` was checked against that stride, so a row never reaches
        // past the plane it belongs to.
        let mut writes: [&mut [u8]; 3] = [
            unsafe { std::slice::from_raw_parts_mut(targets[0].add(start), row_bytes) },
            unsafe { std::slice::from_raw_parts_mut(targets[1].add(start), row_bytes) },
            unsafe { std::slice::from_raw_parts_mut(targets[2].add(start), row_bytes) },
        ];

        for column in 0..width {
            let index = row * width + column;
            let codes = [
                convert.read(rows[0], column),
                convert.read(rows[1], column),
                convert.read(rows[2], column),
            ];
            let mut low = f32::INFINITY;
            let mut high = f32::NEG_INFINITY;
            for code in codes {
                low = low.min(code);
                high = high.max(code);
            }
            let delta = (restored.get(index).copied().unwrap_or(0.0)
                - luma.get(index).copied().unwrap_or(0.0))
            .clamp(-low, CODE_VALUES - high);
            for (plane, code) in codes.iter().enumerate() {
                if let Some(target) = writes.get_mut(plane) {
                    convert.write(target, column, code + delta);
                }
            }
        }
    }

    Ok(())
}

/// One plane of a frame as bytes, with its stride.
fn plane_bytes(frame: &VideoFrame, plane: i32) -> Result<(&[u8], usize)> {
    let (stride, height) = plane_shape(frame, plane)?;
    let length = stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("Deblur: the plane is larger than the address space"))?;
    let pointer = frame.plane(plane);
    if pointer.is_null() {
        return Err(NImagesError::new(format!(
            "Deblur: plane {plane} is not readable"
        )));
    }
    // SAFETY: VapourSynth guarantees a readable plane buffer of `stride * height`
    // bytes for as long as the frame is alive, and the caller holds the frame.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    Ok((bytes, stride))
}
