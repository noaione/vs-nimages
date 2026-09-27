//! The VapourSynth filter layer: the only place in the crate that touches the
//! plugin API.
//!
//! Every filter follows the same shape:
//!
//! * `create` reads and validates its arguments, refuses anything but `GRAY8`,
//!   and registers the node through `core.create_video_filter`
//! * `get_frame` requests the input frame on `Initial`, and builds the output on
//!   `AllFramesReady`
//! * the analysis filters copy the input frame so pixels and properties both
//!   survive, and the mapping filters allocate an output frame and copy the
//!   source properties onto it
//!
//! Frame dimensions come from the frame, never from the node, so a clip whose
//! pages differ in size still works. every row walk uses the frame's own stride
//! and stops after `width` samples, so stride padding is never read or written.

mod levels;
mod peak_gray_shades;
mod peak_stats;
mod posterize;

pub use levels::Levels;
pub use peak_gray_shades::PeakGrayShades;
pub use peak_stats::PeakStats;
pub use posterize::Posterize;

use std::ffi::CStr;

use vapoursynth4_rs::frame::VideoFrame;
use vapoursynth4_rs::map::{KeyStr, MapPropertyError, MapRef};
use vapoursynth4_rs::node::{Dependencies, Filter, RequestPattern, VideoNode};
use vapoursynth4_rs::{ColorFamily, SampleType, VideoInfo, core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::histogram::Histogram;

/// The bit depth every filter in this release accepts.
const GRAY8_BITS: i32 = 8;

/// Reads the `clip` argument every filter takes.
pub(super) fn read_clip(input: &MapRef, function: &str) -> Result<VideoNode> {
    input
        .get_video_node(key!(c"clip"), 0)
        .map_err(|error| NImagesError::new(format!("{function}: invalid clip argument: {error}")))
}

/// Refuses anything but a constant Gray 8 bit clip and returns the output info.
pub(super) fn require_gray8(node: &VideoNode, function: &str) -> Result<VideoInfo> {
    let info = node.info().clone();
    let format = &info.format;
    let is_gray8 = format.color_family == ColorFamily::Gray
        && format.sample_type == SampleType::Integer
        && format.bits_per_sample == GRAY8_BITS
        && format.sub_sampling_w == 0
        && format.sub_sampling_h == 0;

    if !is_gray8 {
        return Err(NImagesError::new(format!(
            "{function} needs a constant Gray 8 bit clip, got {:?} {} bit with \
             subsampling {}x{}",
            format.color_family,
            format.bits_per_sample,
            format.sub_sampling_w,
            format.sub_sampling_h,
        )));
    }

    Ok(info)
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

/// Applies a lookup table to plane 0 of `source`, writing into `output`.
///
/// Both frames must share their dimensions, which holds because the output is
/// allocated from the input frame's own format and size.
pub(super) fn map_plane(
    source: &VideoFrame,
    output: &mut VideoFrame,
    table: &[u8; 256],
) -> Result<()> {
    let width = usize::try_from(source.frame_width(0))
        .map_err(|_| NImagesError::new("plane 0 has a negative width"))?;
    let (source_stride, height) = plane_shape(source, 0)?;
    let (output_stride, output_height) = plane_shape(output, 0)?;
    if width == 0 || height == 0 {
        return Ok(());
    }
    if height != output_height || source_stride == 0 || output_stride == 0 {
        return Err(NImagesError::new(
            "the output frame does not match the input frame",
        ));
    }

    let source_length = source_stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("plane 0 is larger than the address space"))?;
    let output_length = output_stride
        .checked_mul(height)
        .ok_or_else(|| NImagesError::new("the output plane is larger than the address space"))?;

    let source_pointer = source.plane(0);
    let output_pointer = output.plane_mut(0);
    if source_pointer.is_null() || output_pointer.is_null() {
        return Err(NImagesError::new("plane 0 is not readable or writable"));
    }

    // SAFETY: both pointers are valid for the length computed above, the two
    // frames are distinct objects so the slices cannot alias, and the row walk
    // below stays inside `width` samples of each row.
    let source_bytes = unsafe { std::slice::from_raw_parts(source_pointer, source_length) };
    let output_bytes = unsafe { std::slice::from_raw_parts_mut(output_pointer, output_length) };

    for row in 0..height {
        let source_start = row * source_stride;
        let output_start = row * output_stride;
        let source_row = &source_bytes[source_start..source_start + width];
        let output_row = &mut output_bytes[output_start..output_start + width];
        for (input, target) in source_row.iter().zip(output_row.iter_mut()) {
            *target = table[*input as usize];
        }
    }

    Ok(())
}

/// Allocates an output frame with the input frame's format, copies the source
/// properties onto it, and applies `table` to plane 0.
pub(super) fn map_frame(
    core: &CoreRef<'_>,
    source: &VideoFrame,
    table: &[u8; 256],
) -> Result<VideoFrame> {
    let format = source.get_video_format().clone();
    let mut output = core.new_video_frame(
        &format,
        source.frame_width(0),
        source.frame_height(0),
        Some(source),
    );
    map_plane(source, &mut output, table)?;
    Ok(output)
}

/// Reports a frame whose input failed to generate.
pub(super) fn input_failed(function: &str) -> NImagesError {
    NImagesError::new(format!("{function}: failed to generate the input frame"))
}
