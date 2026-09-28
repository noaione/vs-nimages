//! The VapourSynth filter layer: the only place in the crate that touches the
//! plugin API.
//!
//! Every filter follows the same shape:
//!
//! * `create` reads and validates its arguments, checks the format the node
//!   declares, and registers the node through `core.create_video_filter`
//! * `get_frame` requests the input frame on `Initial`, and builds the output on
//!   `AllFramesReady`
//! * the analysis filters copy the input frame so pixels and properties both
//!   survive, and the mapping filters allocate an output frame and copy the
//!   source properties onto it
//!
//! Frame dimensions come from the frame, never from the node, so a clip whose
//! pages differ in size works and so does a clip whose dimensions are not known
//! until a frame is asked for. Every row walk uses the frame's own stride and
//! stops after the samples the plane holds, so stride padding is never read or
//! written, and an RGB frame's interleaved channels are stepped over rather than
//! mixed.

mod levels;
mod peak_gray_shades;
mod peak_stats;
mod posterize;

pub use levels::Levels;
pub use peak_gray_shades::PeakGrayShades;
pub use peak_stats::PeakStats;
pub use posterize::Posterize;

use std::ffi::{CStr, CString};
use std::fmt::{Display, Write};
use std::sync::atomic::{AtomicBool, Ordering};
use std::time::{Duration, Instant};

use vapoursynth4_rs::frame::{VideoFormat, VideoFrame};
use vapoursynth4_rs::map::{KeyStr, MapPropertyError, MapRef};
use vapoursynth4_rs::node::{Dependencies, Filter, RequestPattern, VideoNode};
use vapoursynth4_rs::{ColorFamily, SampleType, VideoInfo, core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::histogram::Histogram;

/// The sample-depth range the integer mapping filters accept.
const MIN_INTEGER_BITS: i32 = 8;
const MAX_INTEGER_BITS: i32 = 16;

/// Stages one frame can report under `debug`.
const TRACE_STAGES: usize = 4;

/// What a filter accepts as input.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) enum Accept {
    /// Gray, 8 bit integer, one plane. what the analyzers take, because a
    /// histogram of one plane only means something for a gray clip.
    Gray8,
    /// Any color family, 8 to 16 bit integer, every plane. what the lookup-table
    /// filters take, because a curve applies per sample whatever the family.
    Integer8To16,
}

impl Accept {
    /// Wording for an error message.
    fn describe(self) -> &'static str {
        match self {
            Self::Gray8 => "a constant Gray 8 bit clip",
            Self::Integer8To16 => "an 8 to 16 bit integer clip",
        }
    }

    /// Whether one format is acceptable.
    ///
    /// An `Undefined` color family means the clip's format varies between
    /// frames, which only the frame itself can answer for.
    fn accepts(self, format: &VideoFormat) -> bool {
        if format.color_family == ColorFamily::Undefined {
            return false;
        }
        match self {
            Self::Integer8To16 => {
                format.sample_type == SampleType::Integer
                    && (MIN_INTEGER_BITS..=MAX_INTEGER_BITS).contains(&format.bits_per_sample)
            }
            Self::Gray8 => {
                format.sample_type == SampleType::Integer
                    && format.bits_per_sample == 8
                    && format.color_family == ColorFamily::Gray
                    && format.sub_sampling_w == 0
                    && format.sub_sampling_h == 0
            }
        }
    }

    /// Rejects a format that is already known to be wrong.
    ///
    /// A clip whose dimensions or format vary reports `Undefined`, and is left
    /// for [`check_frame_format`] to judge once a frame exists.
    fn describe_rejection(self, format: &VideoFormat) -> String {
        format!(
            "needs {}, got {:?} {} bit with subsampling {}x{}",
            self.describe(),
            format.color_family,
            format.bits_per_sample,
            format.sub_sampling_w,
            format.sub_sampling_h,
        )
    }
}

/// Reads the `clip` argument every filter takes.
pub(super) fn read_clip(input: &MapRef, function: &str) -> Result<VideoNode> {
    input
        .get_video_node(key!(c"clip"), 0)
        .map_err(|error| NImagesError::new(format!("{function}: invalid clip argument: {error}")))
}

/// Checks the format a node declares and returns the output info to register.
///
/// The info is the input's own, which is what carries variable dimensions
/// through: `width` and `height` stay 0 and every frame is measured instead.
pub(super) fn checked_info(node: &VideoNode, function: &str, accept: Accept) -> Result<VideoInfo> {
    let info = node.info().clone();
    if info.format.color_family != ColorFamily::Undefined && !accept.accepts(&info.format) {
        return Err(NImagesError::new(format!(
            "{function} {}",
            accept.describe_rejection(&info.format)
        )));
    }
    Ok(info)
}

/// Checks the format of one frame, which is the only place a clip that varies
/// between frames can be judged.
pub(super) fn check_frame_format(frame: &VideoFrame, function: &str, accept: Accept) -> Result<()> {
    let format = frame.get_video_format();
    if accept.accepts(format) {
        return Ok(());
    }
    Err(NImagesError::new(format!(
        "{function} {}",
        accept.describe_rejection(format)
    )))
}

/// Reads an optional integer argument.
pub(super) fn read_int(input: &MapRef, key: &KeyStr) -> Result<Option<i64>> {
    match input.get_int(key, 0) {
        Ok(value) => Ok(Some(value)),
        Err(MapPropertyError::KeyNotFound) => Ok(None),
        Err(error) => Err(NImagesError::property(key, error)),
    }
}

/// Reads an optional float argument.
pub(super) fn read_float(input: &MapRef, key: &KeyStr) -> Result<Option<f64>> {
    match input.get_float(key, 0) {
        Ok(value) => Ok(Some(value)),
        Err(MapPropertyError::KeyNotFound) => Ok(None),
        Err(error) => Err(NImagesError::property(key, error)),
    }
}

/// Returns the largest code value for a supported integer sample depth.
pub(super) fn max_sample_value(bits_per_sample: i32) -> Result<u16> {
    if !(MIN_INTEGER_BITS..=MAX_INTEGER_BITS).contains(&bits_per_sample) {
        return Err(NImagesError::new(format!(
            "integer sample depth must be between {MIN_INTEGER_BITS} and \
             {MAX_INTEGER_BITS}, got {bits_per_sample}"
        )));
    }
    Ok(((1u32 << bits_per_sample) - 1) as u16)
}

/// Checks a percentage argument.
///
/// `None` disables the threshold, and `0` is accepted because the reference
/// treats it as one.
pub(super) fn validate_percentage(value: Option<f64>, name: &str) -> Result<Option<f64>> {
    match value {
        None => Ok(None),
        Some(value) if value.is_finite() && (0.0..=100.0).contains(&value) => Ok(Some(value)),
        Some(value) => Err(NImagesError::new(format!(
            "{name} must be between 0 and 100, got {value}"
        ))),
    }
}

/// Registers the output node against one strictly spatial input.
///
/// The dependency is taken as a raw node pointer so the caller can move the node
/// into the filter without cloning it.
pub(super) fn add_filter<F: Filter>(
    core: &mut CoreRef<'_>,
    output: MapRef,
    name: &CStr,
    info: &VideoInfo,
    filter: F,
    source: *mut ffi::VSNode,
) {
    let dependencies = [ffi::VSFilterDependency {
        source,
        request_pattern: RequestPattern::StrictSpatial,
    }];
    let dependencies = Dependencies::new(&dependencies)
        .expect("a single dependency is always a valid dependency list");
    core.create_video_filter(output, name, info, Box::new(filter), dependencies);
}

/// The stride and height of one plane.
fn plane_shape(frame: &VideoFrame, plane: i32) -> Result<(usize, usize)> {
    let stride = usize::try_from(frame.stride(plane))
        .map_err(|_| NImagesError::new(format!("plane {plane} has a negative stride")))?;
    let height = usize::try_from(frame.frame_height(plane))
        .map_err(|_| NImagesError::new(format!("plane {plane} has a negative height")))?;
    Ok((stride, height))
}

/// Builds the 256-bin histogram of plane 0 without reading stride padding.
pub(super) fn plane_histogram(frame: &VideoFrame) -> Result<Histogram> {
    let width = usize::try_from(frame.frame_width(0))
        .map_err(|_| NImagesError::new("plane 0 has a negative width"))?;
    let (stride, height) = plane_shape(frame, 0)?;
    if width == 0 || height == 0 {
        return Ok(Histogram::new());
    }

    let length = stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("plane 0 is larger than the address space"))?;
    let pointer = frame.plane(0);
    if pointer.is_null() {
        return Err(NImagesError::new("plane 0 is not readable"));
    }

    // SAFETY: VapourSynth guarantees a readable plane buffer of `stride * height`
    // bytes for as long as the frame is alive, and the caller holds the frame.
    let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
    Histogram::from_plane(bytes, stride, width, height)
        .ok_or_else(|| NImagesError::new("plane 0 is smaller than its reported stride"))
}

/// Lookup table selected for the input's integer sample width.
#[derive(Clone)]
pub(super) enum MappingTable {
    /// Eight bit code values.
    U8([u8; 256]),
    /// Nine through sixteen bit code values stored in 16-bit words.
    U16(Vec<u16>),
}

/// Rewrites every sample of one plane through `table`.
fn map_plane(
    source: &VideoFrame,
    output: &mut VideoFrame,
    plane: i32,
    table: &MappingTable,
) -> Result<()> {
    let samples = usize::try_from(source.frame_width(plane))
        .map_err(|_| NImagesError::new(format!("plane {plane} has a negative width")))?;
    let (source_stride, height) = plane_shape(source, plane)?;
    let (output_stride, output_height) = plane_shape(output, plane)?;
    if samples == 0 || height == 0 {
        return Ok(());
    }
    let bits_per_sample = source.get_video_format().bits_per_sample;
    let bytes_per_sample = if bits_per_sample <= 8 { 1usize } else { 2usize };
    let row_bytes = samples
        .checked_mul(bytes_per_sample)
        .ok_or_else(|| NImagesError::new(format!("plane {plane} row is too large")))?;
    if height != output_height || source_stride == 0 || output_stride == 0 {
        return Err(NImagesError::new(
            "the output frame does not match the input frame",
        ));
    }

    if row_bytes > source_stride || row_bytes > output_stride {
        return Err(NImagesError::new(format!(
            "plane {plane} needs {row_bytes} row bytes but its stride is \
             {source_stride} against {output_stride}"
        )));
    }

    let source_length = source_stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("the input plane is larger than the address space"))?;
    let output_length = output_stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("the output plane is larger than the address space"))?;

    let source_pointer = source.plane(plane);
    let output_pointer = output.plane_mut(plane);
    if source_pointer.is_null() || output_pointer.is_null() {
        return Err(NImagesError::new(format!(
            "plane {plane} is not readable or writable"
        )));
    }

    // SAFETY: both pointers are valid for the length computed above, the two
    // frames are distinct objects so the slices cannot alias, and every offset
    // below stays inside `row_bytes` of its row.
    let source_bytes = unsafe { std::slice::from_raw_parts(source_pointer, source_length) };
    let output_bytes = unsafe { std::slice::from_raw_parts_mut(output_pointer, output_length) };

    for row in 0..height {
        let source_start = row * source_stride;
        let output_start = row * output_stride;
        let source_row = &source_bytes[source_start..source_start + row_bytes];
        let output_row = &mut output_bytes[output_start..output_start + row_bytes];

        match (bytes_per_sample, table) {
            (1, MappingTable::U8(table)) => {
                for (input, target) in source_row.iter().zip(output_row.iter_mut()) {
                    *target = table[*input as usize];
                }
            }
            (2, MappingTable::U16(table)) if !table.is_empty() => {
                for (input, target) in source_row
                    .chunks_exact(2)
                    .zip(output_row.chunks_exact_mut(2))
                {
                    let value = u16::from_ne_bytes([input[0], input[1]]);
                    let index = usize::from(value).min(table.len() - 1);
                    target.copy_from_slice(&table[index].to_ne_bytes());
                }
            }
            (1, MappingTable::U16(_)) => {
                return Err(NImagesError::new(
                    "an 8 bit frame needs an 8 bit lookup table",
                ));
            }
            (2, MappingTable::U8(_)) => {
                return Err(NImagesError::new(
                    "a wider integer frame needs a 16 bit lookup table",
                ));
            }
            (2, MappingTable::U16(_)) => {
                return Err(NImagesError::new("the 16 bit lookup table is empty"));
            }
            _ => return Err(NImagesError::new("the input sample width is unsupported")),
        }
    }

    Ok(())
}

/// Allocates an output frame with the input frame's format, copies the source
/// properties onto it, and rewrites every plane through `table`.
///
/// The frame's own dimensions are used, so a clip whose pages differ in size
/// works, and every plane is processed, so an RGB or YUV frame has each of its
/// channels leveled rather than only the first.
pub(super) fn map_frame(
    core: &CoreRef<'_>,
    source: &VideoFrame,
    table: &MappingTable,
) -> Result<VideoFrame> {
    let format = source.get_video_format().clone();
    // A format that varies between frames reports no planes at the clip level,
    // and the per-frame check has already refused it by the time this runs.
    let planes = format.num_planes.max(1);

    let mut output = core.new_video_frame(
        &format,
        source.frame_width(0),
        source.frame_height(0),
        Some(source),
    );

    for plane in 0..planes {
        map_plane(source, &mut output, plane, table)?;
    }

    Ok(output)
}

/// Reports a frame whose input failed to generate.
pub(super) fn input_failed(function: &str) -> NImagesError {
    NImagesError::new(format!("{function}: failed to generate the input frame"))
}

// ---------------------------------------------------------------------------
// debug reporting
//
// Every filter takes `debug:int:opt`. When it is set, the create call logs the
// arguments it resolved and each frame logs how long its stages took, both
// through `core.log` so a host can collect them with `add_log_handler` instead
// of reading stderr. `tools/bench.py` measures the same work from outside; this
// is for looking at one clip, or one frame, from inside a graph.
// ---------------------------------------------------------------------------

/// Writes one debug line to the VapourSynth log.
pub(super) fn log_debug(core: &mut CoreRef<'_>, message: impl Display) {
    let Ok(message) = CString::new(format!("[nimages][debug] {message}")) else {
        return;
    };
    core.log(ffi::VSMessageType::Information, &message);
}

/// Formats a duration the way `vapoursynth-imageseqs` does.
pub(super) fn format_duration(duration: Duration) -> String {
    format!("{:.3} ms", duration.as_secs_f64() * 1000.0)
}

/// Names the format and size of one frame, for the settings line.
///
/// The frame is used rather than the node because a clip whose dimensions or
/// format vary has nothing useful to say at the node level.
///
/// The name is built from the format's own fields rather than from
/// `Core::get_video_format_name`. That helper hands back the API's fixed 32 byte
/// buffer minus its last byte, so the string keeps the NUL padding behind the
/// name, and any log line built from it is dropped because `CString::new`
/// refuses it. See docs/FINDINGS.md §2.5.
pub(super) fn describe_frame(frame: &VideoFrame) -> String {
    let format = frame.get_video_format();
    format!(
        "{:?} {} bit {}x{}",
        format.color_family,
        format.bits_per_sample,
        frame.frame_width(0),
        frame.frame_height(0)
    )
}

/// Whether this frame is the first one this filter instance was asked for.
///
/// VapourSynth drops a message logged from a filter's create function before it
/// reaches a host's log handler, so the settings line is written from the first
/// frame instead. `swap` returns the previous value, so exactly one frame of a
/// filter that VapourSynth calls concurrently reports it.
pub(super) fn report_settings_once(
    enabled: bool,
    reported: &AtomicBool,
    core: &mut CoreRef<'_>,
    detail: impl Display,
) {
    if enabled && !reported.swap(true, Ordering::Relaxed) {
        log_debug(core, detail);
    }
}

/// Times the stages of one frame and reports them when the filter was created
/// with `debug=1`.
pub(super) struct FrameTrace {
    enabled: bool,
    function: &'static str,
    started: Instant,
    stages: [(&'static str, Duration); TRACE_STAGES],
    used: usize,
}

impl FrameTrace {
    /// Starts a frame. The clock is read even when `debug` is off, because one
    /// `Instant::now` costs nothing next to the megapixel scan that follows.
    pub(super) fn new(enabled: bool, function: &'static str) -> Self {
        Self {
            enabled,
            function,
            started: Instant::now(),
            stages: [("", Duration::ZERO); TRACE_STAGES],
            used: 0,
        }
    }

    /// Records the time since `since` under `name`.
    pub(super) fn mark(&mut self, name: &'static str, since: Instant) {
        if !self.enabled || self.used >= self.stages.len() {
            return;
        }
        self.stages[self.used] = (name, since.elapsed());
        self.used += 1;
    }

    /// Writes the frame's line, with `detail` between the frame number and the
    /// timings.
    pub(super) fn emit(&self, core: &mut CoreRef<'_>, n: i32, detail: impl Display) {
        if !self.enabled {
            return;
        }

        let mut stages = String::new();
        for (index, (name, elapsed)) in self.stages[..self.used].iter().enumerate() {
            if index > 0 {
                stages.push(' ');
            }
            let _ = write!(stages, "{name}={}", format_duration(*elapsed));
        }
        if !stages.is_empty() {
            stages.push(' ');
        }

        log_debug(
            core,
            format_args!(
                "{} frame {n}: {detail}{stages}total={}",
                self.function,
                format_duration(self.started.elapsed())
            ),
        );
    }
}
