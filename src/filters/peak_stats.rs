//! `PeakStats`: the black and white levels of a frame, as frame properties.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::{AppendMode, KeyStr, MapRef, Value};
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::peaks::{PeakOptions, find_local_peak};

use super::{
    Accept, add_filter, check_frame_format, checked_info, input_failed, plane_histogram, read_clip,
    read_float, read_int, validate_percentage,
};

/// Lower bound of `upper_limit`, matching the `nmanga` cli and orchestrator.
const MIN_UPPER_LIMIT: i64 = 1;
/// Upper bound of `upper_limit`, the last shade an 8 bit sample holds.
const MAX_UPPER_LIMIT: i64 = 255;
/// Default `upper_limit`, matching `nmanga`.
const DEFAULT_UPPER_LIMIT: i64 = 60;
/// Default `peak_percentage`, matching `nmanga`.
const DEFAULT_PEAK_PERCENTAGE: f64 = 0.25;

/// Finds the black and white levels of each frame and attaches them as properties.
///
/// Pixels pass through untouched, so this filter composes with anything.
pub struct PeakStats {
    source: VideoNode,
    options: PeakOptions,
}

impl Filter for PeakStats {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"PeakStats";
    const ARGS: &'static CStr = c"clip:vnode;upper_limit:int:opt;peak_percentage:float:opt;peak_prominence:float:opt;skip_white:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "PeakStats")?;
        let info = checked_info(&source, "PeakStats", Accept::Gray8)?;

        let upper_limit = read_int(&input, key!(c"upper_limit"))?.unwrap_or(DEFAULT_UPPER_LIMIT);
        if !(MIN_UPPER_LIMIT..=MAX_UPPER_LIMIT).contains(&upper_limit) {
            return Err(NImagesError::new(format!(
                "PeakStats: upper_limit must be between {MIN_UPPER_LIMIT} and \
                 {MAX_UPPER_LIMIT}, got {upper_limit}"
            )));
        }

        // A height threshold cannot change the result on its own, so there is no
        // reason to make it optional here. See docs/FINDINGS.md §3.7.
        let peak_percentage =
            read_float(&input, key!(c"peak_percentage"))?.unwrap_or(DEFAULT_PEAK_PERCENTAGE);
        let peak_percentage =
            validate_percentage(Some(peak_percentage), "PeakStats: peak_percentage")?;

        // Absent means disabled, which the option type already expresses.
        let peak_prominence = validate_percentage(
            read_float(&input, key!(c"peak_prominence"))?,
            "PeakStats: peak_prominence",
        )?;

        let options = PeakOptions {
            upper_limit: upper_limit as u8,
            peak_percentage,
            peak_prominence,
            skip_white: read_int(&input, key!(c"skip_white"))?.unwrap_or(0) != 0,
        };

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self { source, options },
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
                check_frame_format(&input, "PeakStats", Accept::Gray8)?;
                let histogram = plane_histogram(&input)?;
                let peaks = find_local_peak(&histogram, &self.options);

                let mut output = core.copy_frame(&input);
                {
                    let mut properties = output.properties_mut().ok_or_else(|| {
                        NImagesError::new("PeakStats: the output frame holds no properties")
                    })?;
                    set_property(&mut properties, key!(c"NImagesBlackLevel"), peaks.black)?;
                    set_property(&mut properties, key!(c"NImagesWhiteLevel"), peaks.white)?;
                    set_property(
                        &mut properties,
                        key!(c"NImagesBlackPeakFound"),
                        u8::from(peaks.black_found),
                    )?;
                    set_property(
                        &mut properties,
                        key!(c"NImagesWhitePeakFound"),
                        u8::from(peaks.white_found),
                    )?;
                }

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("PeakStats")),
        }
    }
}

/// Writes one integer property, replacing anything already there.
fn set_property(properties: &mut MapRef<'_>, name: &KeyStr, value: u8) -> Result<()> {
    properties
        .set(name, Value::Int(i64::from(value)), AppendMode::Replace)
        .map_err(|error| NImagesError::property(name, error))
}
