//! `Posterize`: a native-range lookup table mapping each sample to one of
//! `2^bits` levels.
//!
//! Like `Levels`, the mapping applies per sample, so any 8 to 16 bit integer
//! format is accepted and every plane is rewritten. Posterizing the planes of
//! an RGB clip independently is not the same operation as posterizing its
//! luma, so an RGB caller that wants the grayscale behaviour converts first.
//!
//! `method=0`, the default, spaces the levels evenly. `method=1` solves them
//! from each frame's own histogram with the Lloyd-Max solver in
//! [`crate::posterize::lloyd_max_levels`], so the levels sit where the page has
//! mass rather than at fixed intervals.

use std::borrow::Cow;
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
use crate::gpu::map::Map;
use crate::histogram::Histogram;
use crate::posterize::{
    MAX_INTEGER_SAMPLE_BITS, MIN_BITS, PosterizeError, lloyd_max_levels, posterize_lut,
    posterize_lut_from_levels_u8, posterize_lut_from_levels_u16, posterize_lut_u16,
};

use super::{
    Accept, FrameTrace, MappingTable, add_filter, analyzer_histogram, check_frame_format,
    checked_info, describe_frame, device_table, input_failed, map_frame, read_clip, read_int,
    report_settings_once,
};

// Hardcoded table
const TABLE_BITS_1: [u8; 256] = posterize_lut(1).unwrap();
const TABLE_BITS_2: [u8; 256] = posterize_lut(2).unwrap();
const TABLE_BITS_3: [u8; 256] = posterize_lut(3).unwrap();
const TABLE_BITS_4: [u8; 256] = posterize_lut(4).unwrap();
const TABLE_BITS_5: [u8; 256] = posterize_lut(5).unwrap();
const TABLE_BITS_6: [u8; 256] = posterize_lut(6).unwrap();
const TABLE_BITS_7: [u8; 256] = posterize_lut(7).unwrap();
const TABLE_BITS_8: [u8; 256] = posterize_lut(8).unwrap();

/// How a posterization picks its levels.
#[derive(Clone, Copy, PartialEq, Eq)]
enum Method {
    /// `2^bits` levels spaced evenly over the sample range.
    Even,
    /// Levels solved from each frame's histogram with Lloyd-Max.
    Lloyd,
}

impl Method {
    /// The name the debug settings line reports.
    const fn name(self) -> &'static str {
        match self {
            Self::Even => "even",
            Self::Lloyd => "lloyd",
        }
    }
}

/// A posterization operation, fixed by bits or automatically inferred from gray shades
enum Bits {
    /// A fixed number of bits, as specified by the caller.
    Fixed(u8),
    /// The number of bits to use, inferred from the gray shades of each frame.
    FromShades,
}

#[derive(Clone)]
struct ResolvedBits {
    bits: u8,
    sample_depth: u8,
    table: MappingTable,
}

/// Maps each frame to a fixed number of gray values, without dithering.
pub struct Posterize {
    source: VideoNode,
    bits: Bits,
    method: Method,
    resolved: Option<ResolvedBits>,
    debug: bool,
    /// Whether this instance runs the table on the device.
    gpu: bool,
    /// The device kernel, built once when `gpu` is set.
    engine: Option<Map>,
    /// The device counter table `method=1` solves its levels from.
    counters: Option<Engine>,
    /// The table uploaded at creation, and its length. Unset when the bits come
    /// from the frame's shades, which uploads one per frame instead.
    table: Option<(VideoFrame, u32)>,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for Posterize {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Posterize";
    const ARGS: &'static CStr =
        c"clip:vnode:all;bits:int:opt;use_props:int:opt;method:int:opt;gpu:int:opt;debug:int:opt;";
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
        let source = read_clip(&input, "Posterize")?;
        let info = checked_info(&source, "Posterize", Accept::Integer8To16)?;

        let use_props = read_int(&input, key!(c"use_props"))?.unwrap_or(0) != 0;
        let bits = if use_props {
            Bits::FromShades
        } else {
            let bpc = read_int(&input, key!(c"bits"))?.ok_or_else(|| {
                NImagesError::new(format!(
                    "Posterize: bits is required, and must be between {MIN_BITS} and \
                     {MAX_INTEGER_SAMPLE_BITS}"
                ))
            })?;

            // cast from i64 to u8
            let bpc_u8 = match u8::try_from(bpc) {
                Ok(bits) => bits,
                Err(_) => {
                    return Err(NImagesError::new(format!(
                        "Posterize: bits must be between {MIN_BITS} and \
                         {MAX_INTEGER_SAMPLE_BITS}, got {bpc}"
                    )));
                }
            };
            if bpc_u8 > MAX_INTEGER_SAMPLE_BITS {
                return Err(NImagesError::new(format!(
                    "Posterize: bits must be between {MIN_BITS} and \
                     {MAX_INTEGER_SAMPLE_BITS}, got {bpc}"
                )));
            }
            if info.format.color_family != vapoursynth4_rs::ColorFamily::Undefined {
                let sample_depth = info.format.bits_per_sample as u8;
                if bpc_u8 > sample_depth {
                    return Err(NImagesError::new(format!(
                        "Posterize: bits must be between {MIN_BITS} and {sample_depth}, \
                         got {bpc_u8}"
                    )));
                }
            }

            Bits::Fixed(bpc_u8)
        };

        // `method` only changes where the levels sit. The level count still comes
        // from `bits`, or from the shades `PeakGrayShades` attached.
        let method = match read_int(&input, key!(c"method"))?.unwrap_or(0) {
            0 => Method::Even,
            1 => Method::Lloyd,
            other => {
                return Err(NImagesError::new(format!(
                    "Posterize: method must be 0 (even levels) or 1 (Lloyd-Max), got {other}"
                )));
            }
        };
        let resolved = match &bits {
            Bits::Fixed(bits)
                if method == Method::Even
                    && info.format.color_family != vapoursynth4_rs::ColorFamily::Undefined =>
            {
                Some(resolve_from_bits(*bits, info.format.bits_per_sample as u8)?)
            }
            _ => None,
        };

        let debug = read_int(&input, key!(c"debug"))?.unwrap_or(0) != 0;
        let gpu = read_int(&input, key!(c"gpu"))?.unwrap_or(0) != 0;
        let (source, engine, table, counters) = if gpu {
            gpu::require_device(&core)?;
            let source = if source.residency() == ffi::VSNodeResidency::GPU {
                source
            } else {
                gpu::upload(&core, &source)?
            };
            let table = match &resolved {
                Some(resolved) => device_table(&core, &resolved.table)?,
                None => None,
            };
            // `method=1` solves its levels from plane 0's histogram, so it
            // needs the counter table as well as the map.
            let counters = if method == Method::Lloyd {
                Some(Engine::new(&core)?)
            } else {
                None
            };
            (source, Some(Map::new(&core)?), table, counters)
        } else {
            let source = if source.residency() == ffi::VSNodeResidency::GPU {
                gpu::download(&core, &source)?
            } else {
                source
            };
            (source, None, None, None)
        };

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self {
                source,
                bits,
                resolved,
                method,
                debug,
                gpu,
                engine,
                table,
                counters,
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
                let mut trace = FrameTrace::new(self.debug, "Posterize");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "Posterize", Accept::Integer8To16)?;
                let sample_depth = input.get_video_format().bits_per_sample as u8;

                let bits = match &self.bits {
                    Bits::Fixed(bits) => {
                        validate_bits(*bits, sample_depth)?;
                        *bits
                    }
                    Bits::FromShades => shade_bits(&input, sample_depth)?,
                };

                let settings = describe_frame(&input);
                report_settings_once(
                    self.debug,
                    &self.reported,
                    &mut core,
                    format_args!(
                        "Posterize: bits={bits} colors={} method={} input={settings}",
                        1u32 << bits,
                        self.method.name()
                    ),
                );

                // `2^depth` levels cover every code value, so both methods are the
                // identity there and the frame needs no table.
                if bits == sample_depth {
                    return Ok(Some(input));
                }

                let resolved: Cow<'_, ResolvedBits> = match self.method {
                    Method::Even => match &self.resolved {
                        Some(resolved) if resolved.sample_depth == sample_depth => {
                            Cow::Borrowed(resolved)
                        }
                        Some(_) => {
                            return Err(NImagesError::new(
                                "Posterize: the input sample depth changed after filter creation",
                            ));
                        }
                        None => Cow::Owned(resolve_from_bits(bits, sample_depth)?),
                    },
                    Method::Lloyd => Cow::Owned(match &self.counters {
                        Some(counters) => {
                            resolve_lloyd_from(counters.count(&core, &input)?, bits, sample_depth)?
                        }
                        None => resolve_lloyd(&input, bits, sample_depth)?,
                    }),
                };

                // A table uploaded at creation carries the declared depth, so a frame
                // that disagrees is refused rather than read as bytes of the wrong
                // width. A clip that declares no format uploads its table per frame.
                let creation_depth = self.resolved.as_ref().map(|resolved| resolved.sample_depth);
                if self.table.is_some() && creation_depth != Some(sample_depth) {
                    return Err(NImagesError::new(
                        "Posterize gpu=1: the input sample depth changed after filter creation",
                    ));
                }
                let mark = Instant::now();
                let output = match (&self.engine, &self.table) {
                    // A fixed `bits` uploaded its table at creation; bits that come
                    // from the frame's shades upload one per frame.
                    (Some(engine), Some((table, length))) => {
                        engine.render(&core, &input, table, *length)?
                    }
                    (Some(engine), None) => {
                        let (table, length) =
                            device_table(&core, &resolved.table)?.ok_or_else(|| {
                                NImagesError::new(
                                    "Posterize gpu=1: the bits have no table to upload",
                                )
                            })?;
                        engine.render(&core, &input, &table, length)?
                    }
                    _ => map_frame(&core, &input, &resolved.table)?,
                };
                trace.mark("map", mark);

                trace.emit(&mut core, n, format_args!("bits={} ", resolved.bits));

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Posterize")),
        }
    }
}

/// The level count `PeakGrayShades` implies for one frame.
fn shade_bits(frame: &VideoFrame, sample_depth: u8) -> Result<u8> {
    let properties = frame
        .properties()
        .ok_or_else(|| NImagesError::new("Posterize: the input frame holds no properties"))?;
    let shades = properties
        .get_int_array(key!(c"NImagesGrayShades"))
        .map_err(|error| {
            NImagesError::new(format!(
                "Posterize(use_props=True) needs NImagesGrayShades on the input frame, which \
                 PeakGrayShades writes; {error}"
            ))
        })?;

    let shade_count = shades.len();
    let max_shades = 1usize << sample_depth;
    if !(1..=max_shades).contains(&shade_count) {
        return Err(NImagesError::new(format!(
            "Posterize(use_props=True) needs between 1 and {max_shades} values in \
             NImagesGrayShades, got {shade_count}"
        )));
    }

    // Pick the smallest power-of-two level count that can cover all shades.
    if shade_count <= 2 {
        return Ok(MIN_BITS);
    }
    Ok((usize::BITS - (shade_count - 1).leading_zeros()) as u8)
}

/// Solves one frame's levels with Lloyd-Max and builds its lookup table.
///
/// The histogram comes from plane 0 for a Gray or YUV clip and from the
/// reference's fixed-point luma for an RGB one. The table then applies to every
/// plane, exactly as the evenly spaced table does.
fn resolve_lloyd(frame: &VideoFrame, bits: u8, sample_depth: u8) -> Result<ResolvedBits> {
    resolve_lloyd_from(analyzer_histogram(frame)?, bits, sample_depth)
}

/// Solves one frame's levels from a histogram and builds its lookup table.
///
/// The counters are the same numbers whichever engine built them, so the host
/// and device paths reach the same levels from the same table.
fn resolve_lloyd_from(histogram: Histogram, bits: u8, sample_depth: u8) -> Result<ResolvedBits> {
    let colors = 1usize << bits;
    let levels = lloyd_max_levels(histogram.counts(), colors).map_err(table_error)?;

    let table = if sample_depth == 8 {
        MappingTable::U8(posterize_lut_from_levels_u8(&levels).map_err(table_error)?)
    } else {
        MappingTable::U16(
            posterize_lut_from_levels_u16(&levels, histogram.max_value()).map_err(table_error)?,
        )
    };

    Ok(ResolvedBits {
        bits,
        sample_depth,
        table,
    })
}

/// Checks one level count against the frame's sample depth.
fn validate_bits(bits: u8, sample_depth: u8) -> Result<()> {
    if !(MIN_BITS..=sample_depth).contains(&bits) {
        return Err(NImagesError::new(format!(
            "Posterize needs between {MIN_BITS} and {sample_depth} bits, got {bits}"
        )));
    }
    Ok(())
}

/// Wraps a lookup-table failure with the filter's name.
fn table_error(error: PosterizeError) -> NImagesError {
    NImagesError::new(format!("Posterize: {}", error.message()))
}

fn resolve_from_bits(bits: u8, sample_depth: u8) -> Result<ResolvedBits> {
    validate_bits(bits, sample_depth)?;

    if sample_depth == 8 {
        return match bits {
            1 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_1),
                bits: 1,
                sample_depth,
            }),
            2 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_2),
                bits: 2,
                sample_depth,
            }),
            3 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_3),
                bits: 3,
                sample_depth,
            }),
            4 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_4),
                bits: 4,
                sample_depth,
            }),
            5 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_5),
                bits: 5,
                sample_depth,
            }),
            6 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_6),
                bits: 6,
                sample_depth,
            }),
            7 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_7),
                bits: 7,
                sample_depth,
            }),
            8 => Ok(ResolvedBits {
                table: MappingTable::U8(TABLE_BITS_8),
                bits: 8,
                sample_depth,
            }),
            _ => Err(NImagesError::new(format!(
                "Posterize needs between {MIN_BITS} and {sample_depth} bits, got {bits}"
            ))),
        };
    }

    let table = posterize_lut_u16(bits, sample_depth).map_err(table_error)?;
    Ok(ResolvedBits {
        table: MappingTable::U16(table),
        bits,
        sample_depth,
    })
}
