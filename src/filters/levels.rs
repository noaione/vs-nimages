//! `Levels`: the ImageMagick `-level` curve as a native-range lookup table.
//!
//! The curve applies per sample, so any 8 to 16 bit integer format is accepted
//! and every plane is rewritten. That covers Gray, RGB and YUV, subsampled or
//! not, and leaves the choice of family and matrix to the caller.

use std::borrow::Cow;
use std::ffi::{CStr, c_void};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::{KeyStr, MapRef};
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::levels::{automatic_gamma_for_range, levels_lut, levels_lut_u16};

use super::{
    Accept, FrameTrace, MappingTable, add_filter, check_frame_format, checked_info, describe_frame,
    input_failed, map_frame, max_sample_value, read_clip, read_float, read_int,
    report_settings_once,
};

/// Default black point.
const DEFAULT_BLACK: i64 = 0;
/// Default gamma.
const DEFAULT_GAMMA: f64 = 1.0;

/// One frame's curve, with the parameters it was resolved from so `debug` can
/// report what a page was actually leveled with.
#[derive(Clone)]
struct Resolved {
    table: MappingTable,
    max_sample: u16,
    black: i64,
    white: i64,
    gamma: f64,
}

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
    Constant(Resolved),
    /// Parameters deferred until a variable-format frame supplies its depth.
    ConstantForFormat {
        black: i64,
        white: Option<i64>,
        gamma: f64,
        peak_offset: i64,
        auto_gamma: bool,
    },
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
    debug: bool,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for Levels {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Levels";
    const ARGS: &'static CStr = c"clip:vnode;black:int:opt;white:int:opt;gamma:float:opt;use_props:int:opt;peak_offset:int:opt;auto_gamma:int:opt;debug:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "Levels")?;
        let info = checked_info(&source, "Levels", Accept::Integer8To16)?;

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
            let white = read_int(&input, key!(c"white"))?;
            let gamma = read_float(&input, key!(c"gamma"))?.unwrap_or(DEFAULT_GAMMA);
            if info.format.color_family == vapoursynth4_rs::ColorFamily::Undefined {
                Curve::ConstantForFormat {
                    black,
                    white,
                    gamma,
                    peak_offset,
                    auto_gamma,
                }
            } else {
                let max_sample = max_sample_value(info.format.bits_per_sample)?;
                Curve::Constant(resolve(
                    black,
                    white.unwrap_or(i64::from(max_sample)),
                    gamma,
                    peak_offset,
                    auto_gamma,
                    max_sample,
                )?)
            }
        };
        let debug = read_int(&input, key!(c"debug"))?.unwrap_or(0) != 0;

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self {
                source,
                curve,
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
                let mut trace = FrameTrace::new(self.debug, "Levels");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "Levels", Accept::Integer8To16)?;
                let max_sample = max_sample_value(input.get_video_format().bits_per_sample)?;

                if self.debug {
                    let settings = describe_frame(&input);
                    let source = match &self.curve {
                        Curve::Constant(resolved) => format!(
                            "use_props=false resolved black={} white={} gamma={}",
                            resolved.black, resolved.white, resolved.gamma
                        ),
                        Curve::ConstantForFormat {
                            black,
                            white,
                            gamma,
                            peak_offset,
                            auto_gamma,
                        } => format!(
                            "use_props=false black={black} white={white:?} \
                             peak_offset={peak_offset} auto_gamma={auto_gamma} gamma={gamma}"
                        ),
                        Curve::FromProperties {
                            gamma,
                            peak_offset,
                            auto_gamma,
                        } => format!(
                            "use_props=true peak_offset={peak_offset} \
                             auto_gamma={auto_gamma} gamma={gamma}"
                        ),
                    };
                    report_settings_once(
                        self.debug,
                        &self.reported,
                        &mut core,
                        format_args!("Levels: {source} input={settings}"),
                    );
                }

                let mark = Instant::now();
                let resolved: Cow<'_, Resolved> = match &self.curve {
                    Curve::Constant(resolved) if resolved.max_sample == max_sample => {
                        Cow::Borrowed(resolved)
                    }
                    Curve::Constant(_) => {
                        return Err(NImagesError::new(
                            "Levels: the input sample depth changed after filter creation",
                        ));
                    }
                    Curve::ConstantForFormat {
                        black,
                        white,
                        gamma,
                        peak_offset,
                        auto_gamma,
                    } => Cow::Owned(resolve(
                        *black,
                        white.unwrap_or(i64::from(max_sample)),
                        *gamma,
                        *peak_offset,
                        *auto_gamma,
                        max_sample,
                    )?),
                    Curve::FromProperties {
                        gamma,
                        peak_offset,
                        auto_gamma,
                    } => Cow::Owned(resolve_from_properties(
                        &input,
                        *gamma,
                        *peak_offset,
                        *auto_gamma,
                        max_sample,
                    )?),
                };
                trace.mark("curve", mark);

                let mark = Instant::now();
                let output = map_frame(&core, &input, &resolved.table)?;
                trace.mark("map", mark);

                trace.emit(
                    &mut core,
                    n,
                    format_args!(
                        "black={} white={} gamma={:.2} ",
                        resolved.black, resolved.white, resolved.gamma
                    ),
                );

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Levels")),
        }
    }
}

/// Resolves one parameter set, folding in `peak_offset` and `auto_gamma`.
fn resolve(
    black: i64,
    white: i64,
    gamma: f64,
    peak_offset: i64,
    auto_gamma: bool,
    max_sample: u16,
) -> Result<Resolved> {
    let black = black.checked_add(peak_offset).ok_or_else(|| {
        NImagesError::new("Levels: black point plus peak_offset exceeds the integer range")
    })?;
    check_endpoints(black, white, max_sample)?;
    let gamma = if auto_gamma {
        automatic_gamma_for_range(black as u16, max_sample).ok_or_else(gamma_domain_error)?
    } else {
        gamma
    };
    let table = if max_sample == u8::MAX.into() {
        MappingTable::U8(
            levels_lut(black as f64, white as f64, gamma)
                .map_err(|error| NImagesError::new(format!("Levels: {}", error.message())))?,
        )
    } else {
        MappingTable::U16(
            levels_lut_u16(black as u16, white as u16, gamma, max_sample)
                .map_err(|error| NImagesError::new(format!("Levels: {}", error.message())))?,
        )
    };
    Ok(Resolved {
        table,
        max_sample,
        black,
        white,
        gamma,
    })
}

/// Resolves the curve for one frame from the properties `PeakStats` wrote.
fn resolve_from_properties(
    frame: &VideoFrame,
    gamma: f64,
    peak_offset: i64,
    auto_gamma: bool,
    max_sample: u16,
) -> Result<Resolved> {
    let properties = frame
        .properties()
        .ok_or_else(|| NImagesError::new("Levels: the input frame holds no properties"))?;

    let black = read_peak(&properties, c"NImagesBlackLevel")?;
    let white = read_peak(&properties, c"NImagesWhiteLevel")?;
    resolve(black, white, gamma, peak_offset, auto_gamma, max_sample)
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
fn check_endpoints(black: i64, white: i64, max_sample: u16) -> Result<()> {
    if !(0..=i64::from(max_sample)).contains(&black) {
        return Err(NImagesError::new(format!(
            "Levels: the black point must land between 0 and {max_sample}, got {black}"
        )));
    }
    if !(0..=i64::from(max_sample)).contains(&white) {
        return Err(NImagesError::new(format!(
            "Levels: the white point must be between 0 and {max_sample}, got {white}"
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
        "Levels: auto_gamma needs a black point below half of the sample range, \
         because the gamma expression is undefined at or above it",
    )
}
