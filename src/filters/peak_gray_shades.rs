//! `PeakGrayShades`: the significant gray shades of a frame, as frame properties.

use std::ffi::{CStr, c_void};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::MapRef;
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::gpu;
use crate::gpu::histogram::Engine;
use crate::gray_shades::analyze_gray_shades;

use super::{
    Accept, FrameTrace, add_filter, analyzer_histogram, check_frame_format, checked_info,
    describe_frame, input_failed, read_clip, read_float, read_int, report_settings_once,
};

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
    debug: bool,
    /// Whether this instance counts on the device.
    gpu: bool,
    /// The device kernel, built once when `gpu` is set.
    engine: Option<Engine>,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for PeakGrayShades {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"PeakGrayShades";
    const ARGS: &'static CStr = c"clip:vnode:all;threshold:float:opt;gpu:int:opt;debug:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode:all;";

    /// `all` is answered by the instance: a `gpu=1` run's frame is device
    /// memory and a `gpu=0` run's is not.
    fn gpu_output(&self) -> bool {
        self.gpu
    }

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "PeakGrayShades")?;
        let info = checked_info(&source, "PeakGrayShades", Accept::Analyzer)?;

        let threshold = read_float(&input, key!(c"threshold"))?.unwrap_or(DEFAULT_THRESHOLD);
        if !threshold.is_finite() || threshold < 0.0 {
            return Err(NImagesError::new(format!(
                "PeakGrayShades: threshold must be a finite number of percent, got {threshold}"
            )));
        }
        let debug = read_int(&input, key!(c"debug"))?.unwrap_or(0) != 0;

        let gpu = read_int(&input, key!(c"gpu"))?.unwrap_or(0) != 0;
        let (source, engine) = if gpu {
            gpu::require_device(&core)?;
            if !vapoursynth4_rs::utils::is_constant_video_format(&info) {
                return Err(NImagesError::new(
                    "PeakGrayShades gpu=1: the clip's format is not constant",
                ));
            }
            let source = if source.residency() == ffi::VSNodeResidency::GPU {
                source
            } else {
                gpu::upload(&core, &source)?
            };
            (source, Some(Engine::new(&core)?))
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
            output,
            Self::NAME,
            &info,
            Self {
                source,
                threshold,
                debug,
                gpu,
                engine,
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
                let mut trace = FrameTrace::new(self.debug, "PeakGrayShades");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "PeakGrayShades", Accept::Analyzer)?;

                let settings = describe_frame(&input);
                report_settings_once(
                    self.debug,
                    &self.reported,
                    &mut core,
                    format_args!(
                        "PeakGrayShades: threshold={} input={settings}",
                        self.threshold
                    ),
                );

                let mark = Instant::now();
                let histogram = match &self.engine {
                    Some(engine) => engine.count(&core, &input)?,
                    None => analyzer_histogram(&input)?,
                };
                trace.mark("histogram", mark);

                let mark = Instant::now();
                let shades = analyze_gray_shades(&histogram, self.threshold);
                trace.mark("shades", mark);

                let mark = Instant::now();
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
                trace.mark("copy", mark);

                trace.emit(&mut core, n, format_args!("shades={} ", values.len()));

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("PeakGrayShades")),
        }
    }
}
