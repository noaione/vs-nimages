//! `Levels`: the ImageMagick `-level` curve as a 256-entry lookup table.
//!
//! The curve applies per sample, so any 8 bit integer format is accepted and
//! every plane is rewritten. That covers Gray, RGB and YUV, subsampled or not,
//! and leaves the choice of family and matrix to the caller.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::{KeyStr, MapRef};
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::levels::{automatic_gamma, levels_lut};

use super::{
    Accept, add_filter, check_frame_format, checked_info, input_failed, map_frame, read_clip,
    read_float, read_int,
};

/// Default black point.
const DEFAULT_BLACK: i64 = 0;
/// Default white point.
const DEFAULT_WHITE: i64 = 255;
/// Default gamma.
const DEFAULT_GAMMA: f64 = 1.0;

/// A level operation, either fixed at creation or read from each input frame's
/// properties.
///
/// The size difference between the variants is deliberate: the table stays
/// inline so a constant `Levels` never dereferences a pointer per frame, and
/// `create_video_filter` moves the whole filter behind a `Box`, so the larger
/// variant never lands on the stack.
#[allow(clippy::large_enum_variant)]
enum Curve {
    /// One table, built once and shared by every frame.
    Constant([u8; 256]),
    /// Built per frame from `NImagesBlackLevel` and `NImagesWhiteLevel`.
    FromProperties {
        gamma: f64,
        peak_offset: i64,
        auto_gamma: bool,
    },
}

/// Applies level adjustments to each frame.
pub struct Levels {
    source: VideoNode,
    curve: Curve,
}

impl Filter for Levels {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Levels";
    const ARGS: &'static CStr = c"clip:vnode;black:int:opt;white:int:opt;gamma:float:opt;use_props:int:opt;peak_offset:int:opt;auto_gamma:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "Levels")?;
        let info = checked_info(&source, "Levels", Accept::Integer8)?;

        let peak_offset = read_int(&input, key!(c"peak_offset"))?.unwrap_or(0);
        let auto_gamma = read_int(&input, key!(c"auto_gamma"))?.unwrap_or(0) != 0;
        let use_props = read_int(&input, key!(c"use_props"))?.unwrap_or(0) != 0;

        let curve = if use_props {
            Curve::FromProperties {
                gamma: read_float(&input, key!(c"gamma"))?.unwrap_or(DEFAULT_GAMMA),
                peak_offset,
                auto_gamma,
            }
        } else {
            let black = read_int(&input, key!(c"black"))?.unwrap_or(DEFAULT_BLACK);
            let white = read_int(&input, key!(c"white"))?.unwrap_or(DEFAULT_WHITE);
            let gamma = read_float(&input, key!(c"gamma"))?.unwrap_or(DEFAULT_GAMMA);
            Curve::Constant(build_constant(
                black,
                white,
                gamma,
                peak_offset,
                auto_gamma,
            )?)
        };

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self { source, curve },
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
                check_frame_format(&input, "Levels", Accept::Integer8)?;
                let table = match &self.curve {
                    Curve::Constant(table) => *table,
                    Curve::FromProperties {
                        gamma,
                        peak_offset,
                        auto_gamma,
                    } => table_from_properties(&input, *gamma, *peak_offset, *auto_gamma)?,
                };
                let output = map_frame(&core, &input, &table)?;
                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Levels")),
        }
    }
}

/// Resolves the constant parameter set, folding in `peak_offset` and
/// `auto_gamma` exactly once.
fn build_constant(
    black: i64,
    white: i64,
    gamma: f64,
    peak_offset: i64,
    auto_gamma: bool,
) -> Result<[u8; 256]> {
    let black = black + peak_offset;
    check_endpoints(black, white)?;
    let gamma = if auto_gamma {
        automatic_gamma(black as u8).ok_or_else(gamma_domain_error)?
    } else {
        gamma
    };
    levels_lut(black as f64, white as f64, gamma)
        .map_err(|error| NImagesError::new(format!("Levels: {}", error.message())))
}

/// Builds the table for one frame from the properties `PeakStats` wrote.
fn table_from_properties(
    frame: &VideoFrame,
    gamma: f64,
    peak_offset: i64,
    auto_gamma: bool,
) -> Result<[u8; 256]> {
    let properties = frame
        .properties()
        .ok_or_else(|| NImagesError::new("Levels: the input frame holds no properties"))?;

    let black = read_peak(&properties, c"NImagesBlackLevel")?;
    let white = read_peak(&properties, c"NImagesWhiteLevel")?;
    build_constant(black, white, gamma, peak_offset, auto_gamma)
}

/// Reads one level property written by `PeakStats`.
fn read_peak(properties: &MapRef<'_>, name: &CStr) -> Result<i64> {
    let name = KeyStr::from_cstr(name);
    properties.get_int(name, 0).map_err(|error| {
        NImagesError::new(format!(
            "Levels(use_props=True) needs {name} on the input frame, which \
             PeakStats writes; {error}"
        ))
    })
}

/// Checks that the adjusted black point is still a usable code value.
fn check_endpoints(black: i64, white: i64) -> Result<()> {
    if !(0..=255).contains(&black) {
        return Err(NImagesError::new(format!(
            "Levels: the black point must land between 0 and 255, got {black}"
        )));
    }
    if !(0..=255).contains(&white) {
        return Err(NImagesError::new(format!(
            "Levels: the white point must be between 0 and 255, got {white}"
        )));
    }
    if black >= white {
        return Err(NImagesError::new(format!(
            "Levels: black level {black} must be lower than white level {white}"
        )));
    }
    Ok(())
}

/// The error for a black point the automatic gamma cannot use.
fn gamma_domain_error() -> NImagesError {
    NImagesError::new(
        "Levels: auto_gamma needs a black point below 128, because the gamma \
         expression is undefined above it",
    )
}
