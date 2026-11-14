//! The opt-in book prepass, retaining analysis rather than decoded frames.

use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::ffi;
use vapoursynth4_rs::frame::VideoFrame;
use vapoursynth4_rs::node::{Node, VideoNode};

use crate::book_peak::{BookOptions, BookPage, BookResult, consensus, has_dark_median};
use crate::error::{NImagesError, Result};
use crate::gpu;
use crate::luma::luma_u16;
use crate::peaks::{PeakOptions, find_local_peak};

use super::{Accept, analyzer_histogram, check_frame_format, plane_shape};

/// Scans once at creation, before any output frame can be requested.
/// Native peak units require one bit depth, even when dimensions or family vary.
pub(super) fn scan(
    source: &VideoNode,
    core: &CoreRef,
    peaks: &PeakOptions,
    lower: u8,
    upper: u8,
    offset: i64,
) -> Result<Vec<BookResult>> {
    let length = usize::try_from(source.info().num_frames)
        .ok()
        .filter(|length| *length > 0)
        .ok_or_else(|| {
            NImagesError::new("PeakStats: book scan requires a finite, nonempty clip")
        })?;
    let host = if source.residency() == ffi::VSNodeResidency::GPU {
        gpu::download(core, source)?
    } else {
        source.clone()
    };
    let mut pages = Vec::new();
    pages.try_reserve_exact(length).map_err(|_| {
        NImagesError::new("PeakStats: could not allocate the book's per-frame analysis")
    })?;
    let mut maximum = None;
    let mut options = BookOptions {
        lower: 0,
        upper: 0,
        offset,
    };
    for n in 0..source.info().num_frames {
        let frame = host.get_frame(n).map_err(|error| {
            NImagesError::new(format!(
                "PeakStats: book scan frame {n}: {}",
                error.to_string_lossy()
            ))
        })?;
        check_frame_format(&frame, "PeakStats: book scan", Accept::Analyzer)?;
        let histogram = analyzer_histogram(&frame)?;
        let max_value = histogram.max_value();
        if maximum.is_some_and(|previous| previous != max_value) {
            return Err(NImagesError::new(
                "PeakStats: book scan requires the same bit depth on every frame",
            ));
        }
        maximum = Some(max_value);
        // Integer levels above a fractional lower bound exceed its floor;
        // those below a fractional upper bound precede its ceiling.
        options.lower = i64::from(lower) * i64::from(max_value) / 255;
        options.upper = (i64::from(upper) * i64::from(max_value) + 254) / 255;
        let local = find_local_peak(&histogram, peaks);
        let has_dark = if options.usable(local)
            || (local.black_found && i64::from(local.black).saturating_add(offset) <= options.lower)
        {
            false
        } else {
            dark_content(
                &frame,
                (u32::from(upper) * u32::from(max_value) / 255) as u16,
            )?
        };
        pages.push(BookPage {
            peaks: local,
            has_dark,
        });
    }
    let shared = consensus(&pages, options);
    let mut results = Vec::new();
    results.try_reserve_exact(length).map_err(|_| {
        NImagesError::new("PeakStats: could not allocate the book's resolved levels")
    })?;
    results.extend(pages.into_iter().map(|page| options.resolve(page, shared)));
    Ok(results)
}

/// Reads the frame's luma through checked plane buffers and excludes isolated noise.
fn dark_content(frame: &VideoFrame, limit: u16) -> Result<bool> {
    let width = usize::try_from(frame.frame_width(0))
        .map_err(|_| NImagesError::new("PeakStats: negative frame width"))?;
    let height = usize::try_from(frame.frame_height(0))
        .map_err(|_| NImagesError::new("PeakStats: negative frame height"))?;
    let format = frame.get_video_format();
    let rgb = format.color_family == vapoursynth4_rs::ColorFamily::RGB;
    let sample_bytes = if format.bits_per_sample == 8 { 1 } else { 2 };
    let mut planes: [&[u8]; 3] = [&[], &[], &[]];
    let mut strides = [0usize; 3];
    for plane in 0..if rgb { 3 } else { 1 } {
        let (stride, rows) = plane_shape(frame, plane as i32)?;
        let length = stride
            .checked_mul(rows)
            .ok_or_else(|| NImagesError::new("PeakStats: plane exceeds the address space"))?;
        if rows < height
            || width
                .checked_mul(sample_bytes)
                .is_none_or(|row| row > stride)
        {
            return Err(NImagesError::new(
                "PeakStats: plane is smaller than its dimensions",
            ));
        }
        let pointer = frame.plane(plane as i32);
        if pointer.is_null() {
            return Err(NImagesError::new("PeakStats: plane is not readable"));
        }
        // SAFETY: the live frame owns stride * rows readable bytes on the host.
        let bytes = unsafe { std::slice::from_raw_parts(pointer, length) };
        if let Some(slot) = planes.get_mut(plane) {
            *slot = bytes;
        }
        if let Some(slot) = strides.get_mut(plane) {
            *slot = stride;
        }
    }
    let read = |plane: usize, x: usize, y: usize| {
        let bytes = planes.get(plane).copied().unwrap_or(&[]);
        let stride = strides.get(plane).copied().unwrap_or(0);
        let index = y
            .saturating_mul(stride)
            .saturating_add(x.saturating_mul(sample_bytes));
        let low = bytes.get(index).copied().unwrap_or(0);
        if sample_bytes == 1 {
            u16::from(low)
        } else {
            u16::from_ne_bytes([
                low,
                bytes.get(index.saturating_add(1)).copied().unwrap_or(0),
            ])
        }
    };
    Ok(has_dark_median(width, height, limit, |x, y| {
        if rgb {
            luma_u16(read(0, x, y), read(1, x, y), read(2, x, y))
        } else {
            read(0, x, y)
        }
    }))
}
