//! `PeakGrayShades`: the significant gray shades of a frame, as frame properties.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::MapRef;
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::gray_shades::analyze_gray_shades;

use super::{add_filter, input_failed, plane_histogram, read_clip, read_float, require_gray8};

/// Default `threshold`, matching `nmanga`.
const DEFAULT_THRESHOLD: f64 = 0.01;

/// Reports every shade whose share of each frame exceeds `threshold` percent.
///
/// Pixels pass through untouched. `NImagesGrayShades` and
/// `NImagesGrayShadePercentages` are always both present and always the same
/// length, including when that length is zero.
pub struct PeakGrayShades {
    source: VideoNode,
    threshold: f64,
}

impl Filter for PeakGrayShades {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"PeakGrayShades";
    const ARGS: &'static CStr = c"clip:vnode;threshold:float:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "PeakGrayShades")?;
        let info = require_gray8(&source, "PeakGrayShades")?;

        let threshold = read_float(&input, key!(c"threshold"))?.unwrap_or(DEFAULT_THRESHOLD);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(NImagesError::new(format!(
                "PeakGrayShades: threshold must be a finite number of percent, got {threshold}"
            )));
        }

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self { source, threshold },
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
        core: CoreRef,
    ) -> Result<Option<Self::FrameType>> {
        match activation_reason {
            ffi::VSActivationReason::Initial => {
                frame_ctx.request_frame_filter(n, &self.source);
                Ok(None)
            }
            ffi::VSActivationReason::AllFramesReady => {
                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                let histogram = plane_histogram(&input)?;
                let shades = analyze_gray_shades(&histogram, self.threshold);

                let values: Vec<i64> = shades.iter().map(|shade| i64::from(shade.shade)).collect();
                let percentages: Vec<f64> = shades.iter().map(|shade| shade.percentage).collect();

                let mut output = core.copy_frame(&input);
                {
                    let mut properties = output.properties_mut().ok_or_else(|| {
                        NImagesError::new("PeakGrayShades: the output frame holds no properties")
                    })?;
                    // A zero-length array is a supported property, so both keys
                    // are written even when nothing qualified.
                    properties
                        .set_int_array(key!(c"NImagesGrayShades"), &values)
                        .map_err(|error| NImagesError::property("NImagesGrayShades", error))?;
                    properties
                        .set_float_array(key!(c"NImagesGrayShadePercentages"), &percentages)
                        .map_err(|error| {
                            NImagesError::property("NImagesGrayShadePercentages", error)
                        })?;
                }

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("PeakGrayShades")),
        }
    }
}
