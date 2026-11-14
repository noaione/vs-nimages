//! The opt-in book prepass, retaining analysis rather than decoded frames.

use vapoursynth4_rs::core::CoreRef;
use vapoursynth4_rs::ffi;
use vapoursynth4_rs::node::{Node, VideoNode};

use crate::book_peak::{BookOptions, BookPage, BookResult, consensus};
use crate::error::{NImagesError, Result};
use crate::gpu;
use crate::peaks::{PeakOptions, find_local_peak};

use super::{Accept, analyzer_histogram, check_frame_format};

/// Public policy values, resolved for each frame's native bit depth.
#[derive(Clone, Copy)]
pub(super) struct Settings {
    pub lower: u8,
    pub upper: u8,
    pub offset: i64,
    pub tolerance: u8,
    pub median: bool,
}

/// Scans once at creation, before any output frame can be requested.
/// Native peak units require one bit depth, even when dimensions or family vary.
pub(super) fn scan(
    source: &VideoNode,
    core: &CoreRef,
    peaks: &PeakOptions,
    settings: Settings,
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
        offset: settings.offset,
        tolerance: 0,
    };
    for n in 0..source.info().num_frames {
        let frame = host.get_frame(n).map_err(|error| {
            NImagesError::new(format!(
                "PeakStats: book scan frame {n}: {}",
                error.to_string_lossy()
            ))
        })?;
        check_frame_format(&frame, "PeakStats: book scan", Accept::Analyzer)?;
        let histogram = if settings.median {
            super::median::histogram(&frame)?
        } else {
            analyzer_histogram(&frame)?
        };
        let max_value = histogram.max_value();
        if maximum.is_some_and(|previous| previous != max_value) {
            return Err(NImagesError::new(
                "PeakStats: book scan requires the same bit depth on every frame",
            ));
        }
        maximum = Some(max_value);
        // Integer levels above a fractional lower bound exceed its floor;
        // those below a fractional upper bound precede its ceiling.
        options.lower = i64::from(settings.lower) * i64::from(max_value) / 255;
        options.upper = (i64::from(settings.upper) * i64::from(max_value) + 254) / 255;
        options.tolerance = i64::from(settings.tolerance) * i64::from(max_value) / 255;
        let local = find_local_peak(&histogram, peaks);
        let darkest = histogram
            .counts()
            .iter()
            .position(|&count| count != 0)
            .unwrap_or(usize::from(max_value));
        let has_dark = (darkest as u64) * 255 <= u64::from(settings.upper) * u64::from(max_value);
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
