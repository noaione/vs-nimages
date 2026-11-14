//! `PeakStats`: the black and white levels of a frame, as frame properties.

use std::ffi::{CStr, c_void};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use vapoursynth4_rs::frame::{FrameContext, VideoFrame};
use vapoursynth4_rs::map::{AppendMode, KeyStr, MapMut, MapRef, Value};
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::book_peak::BookResult;
use crate::error::{NImagesError, Result};
use crate::gpu;
use crate::gpu::histogram::Engine;
use crate::peaks::{PeakOptions, find_local_peak};

use super::{
    Accept, FrameTrace, add_filter, analyzer_histogram, check_frame_format, checked_info,
    describe_frame, input_failed, read_clip, read_float, read_int, report_settings_once,
    validate_percentage,
};

/// Lower bound of `upper_limit`, matching the `nmanga` cli and orchestrator.
const MIN_UPPER_LIMIT: i64 = 1;
/// Upper bound of `upper_limit`, the last shade in its 8-bit-equivalent units.
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
    /// Immutable results from the opt-in whole-clip prepass.
    book: Option<Vec<BookResult>>,
    book_settings: (i64, i64, i64),
    debug: bool,
    /// The device kernel, built once when `gpu` is set.
    engine: Option<Engine>,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for PeakStats {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"PeakStats";
    const ARGS: &'static CStr = c"clip:vnode:all;upper_limit:int:opt;peak_percentage:float:opt;peak_prominence:float:opt;skip_white:int:opt;book_peak_fallback:int:opt;levelling_lower_limit:int:opt;levelling_upper_limit:int:opt;peak_offset:int:opt;gpu:int:opt;debug:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode:all;";

    fn create(
        input: MapRef,
        mut output: MapMut,
        _data: Option<&Self::FilterData>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "PeakStats")?;
        let info = checked_info(&source, "PeakStats", Accept::Analyzer)?;

        let upper_limit = read_int(&input, key!(c"upper_limit"))?.unwrap_or(DEFAULT_UPPER_LIMIT);
        if !(MIN_UPPER_LIMIT..=MAX_UPPER_LIMIT).contains(&upper_limit) {
            return Err(NImagesError::new(format!(
                "PeakStats: upper_limit must be between {MIN_UPPER_LIMIT} and \
                 {MAX_UPPER_LIMIT}, got {upper_limit}"
            )));
        }

        // Zero disables the height threshold; threshold status stays observable.
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
        let debug = read_int(&input, key!(c"debug"))?.unwrap_or(0) != 0;

        let book_enabled = read_int(&input, key!(c"book_peak_fallback"))?.unwrap_or(0) != 0;
        let lower = read_int(&input, key!(c"levelling_lower_limit"))?.unwrap_or(8);
        let upper = read_int(&input, key!(c"levelling_upper_limit"))?.unwrap_or(55);
        if !(0..=255).contains(&lower) || !(0..=255).contains(&upper) || lower >= upper {
            return Err(NImagesError::new(
                "PeakStats: levelling limits must satisfy 0 <= lower < upper <= 255",
            ));
        }
        let offset = read_int(&input, key!(c"peak_offset"))?.unwrap_or(0);

        let gpu = read_int(&input, key!(c"gpu"))?.unwrap_or(0) != 0;
        if gpu {
            gpu::require_device(&core)?;
        }
        let book = if book_enabled {
            Some(super::book_peak::scan(
                &source,
                &core,
                &options,
                lower as u8,
                upper as u8,
                offset,
            )?)
        } else {
            None
        };
        let (source, engine) = if gpu {
            let source = if source.residency() == ffi::VSNodeResidency::GPU {
                source
            } else {
                gpu::upload(&core, &source)?
            };
            let engine = if book_enabled {
                None
            } else {
                Some(Engine::new(&core)?)
            };
            (source, engine)
        } else {
            let source = if source.residency() == ffi::VSNodeResidency::GPU {
                gpu::download(&core, &source)?
            } else {
                source
            };
            (source, None)
        };

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            &mut output,
            Self::NAME,
            &info,
            gpu,
            Self {
                source,
                options,
                book,
                book_settings: (lower, upper, offset),
                debug,
                engine,
                reported: AtomicBool::new(false),
            },
            dependency,
        )?;
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
                let mut trace = FrameTrace::new(self.debug, "PeakStats");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "PeakStats", Accept::Analyzer)?;
                let settings = describe_frame(&input);
                let options = &self.options;
                let line = format!(
                    "PeakStats: upper_limit={} peak_percentage={:?} peak_prominence={:?} \
                     skip_white={} book_peak_fallback={} levelling_lower_limit={} \
                     levelling_upper_limit={} peak_offset={} input={settings}",
                    options.upper_limit,
                    options.peak_percentage,
                    options.peak_prominence,
                    options.skip_white,
                    self.book.is_some(),
                    self.book_settings.0,
                    self.book_settings.1,
                    self.book_settings.2,
                );
                report_settings_once(self.debug, &self.reported, &mut core, line);

                let mark = Instant::now();
                let book_result = match &self.book {
                    Some(book) => Some(
                        usize::try_from(n)
                            .ok()
                            .and_then(|n| book.get(n))
                            .ok_or_else(|| {
                                NImagesError::new("PeakStats: frame is outside the scanned book")
                            })?,
                    ),
                    None => None,
                };
                let peaks = if let Some(result) = book_result {
                    trace.mark("peaks", mark);
                    result.peaks
                } else {
                    let histogram = match &self.engine {
                        Some(engine) => engine.count(&core, &input)?,
                        None => analyzer_histogram(&input)?,
                    };
                    trace.mark("histogram", mark);
                    let mark = Instant::now();
                    let peaks = find_local_peak(&histogram, &self.options);
                    trace.mark("peaks", mark);
                    peaks
                };

                let mark = Instant::now();
                let mut output = core.copy_frame(&input);
                {
                    let mut properties = output.properties_mut().ok_or_else(|| {
                        NImagesError::new("PeakStats: the output frame holds no properties")
                    })?;
                    set_property(&mut properties, key!(c"NImagesBlackLevel"), peaks.black)?;
                    set_property(&mut properties, key!(c"NImagesWhiteLevel"), peaks.white)?;
                    if let Some(result) = book_result {
                        set_property(
                            &mut properties,
                            key!(c"NImagesBookPeakUsed"),
                            u16::from(result.used_book_peak),
                        )?;
                        set_property(
                            &mut properties,
                            key!(c"NImagesLevelEligible"),
                            u16::from(result.eligible),
                        )?;
                    }
                    set_property(
                        &mut properties,
                        key!(c"NImagesBlackPeakFound"),
                        if peaks.black_found { 1 } else { 0 },
                    )?;
                    set_property(
                        &mut properties,
                        key!(c"NImagesWhitePeakFound"),
                        if peaks.white_found { 1 } else { 0 },
                    )?;
                }
                trace.mark("copy", mark);

                trace.emit(
                    &mut core,
                    n,
                    format_args!("black={} white={} ", peaks.black, peaks.white),
                );

                Ok(Some(output.freeze()))
            }
            ffi::VSActivationReason::Error => Err(input_failed("PeakStats")),
        }
    }
}

/// Writes one integer property, replacing anything already there.
fn set_property(properties: &mut MapMut<'_>, name: &KeyStr, value: u16) -> Result<()> {
    properties
        .set(name, Value::Int(i64::from(value)), AppendMode::Replace)
        .map_err(|error| NImagesError::property(name, error))
}
