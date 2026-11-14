//! Median-filtered luma histogram for PeakStats and its book scan.

use super::{max_sample_value, plane_shape};
use crate::error::{NImagesError, Result};
use crate::histogram::Histogram;
use crate::luma::luma_u16;
use vapoursynth4_rs::frame::VideoFrame;

/// Counts median-filtered luma through checked host plane buffers.
pub(super) fn histogram(frame: &VideoFrame) -> Result<Histogram> {
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
    crate::median::histogram(
        width,
        height,
        max_sample_value(format.bits_per_sample)?,
        |x, y| {
            if rgb {
                luma_u16(read(0, x, y), read(1, x, y), read(2, x, y))
            } else {
                read(0, x, y)
            }
        },
    )
    .ok_or_else(|| {
        NImagesError::new("PeakStats: could not allocate the median histogram rows or bins")
    })
}
