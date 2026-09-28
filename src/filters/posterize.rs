//! `Posterize`: a 256-entry lookup table mapping each sample to one of `2^bits`
//! levels.
//!
//! Like `Levels`, the mapping applies per sample, so any 8 bit integer format is
//! accepted and every plane is rewritten. Posterizing the planes of an RGB clip
//! independently is not the same operation as posterizing its luma, so an RGB
//! caller that wants the grayscale behaviour converts first.

use std::ffi::{CStr, c_void};
use std::sync::atomic::AtomicBool;
use std::time::Instant;

use vapoursynth4_rs::frame::{Frame, FrameContext, VideoFrame};
use vapoursynth4_rs::map::MapRef;
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::posterize::{MAX_BITS, MIN_BITS, posterize_lut};

use super::{
    Accept, FrameTrace, add_filter, check_frame_format, checked_info, describe_frame, input_failed,
    map_frame, read_clip, read_int, report_settings_once,
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

/// A posterization operation, fixed by bits or automatically inferred from gray shades
enum Bits {
    /// A fixed number of bits, as specified by the caller.
    Fixed(u8),
    /// The number of bits to use, inferred from the gray shades of each frame.
    FromShades,
}

struct ResolvedBits {
    bits: u8,
    table: [u8; 256],
}

/// Maps each frame to `2^bits` evenly spaced gray values, without dithering.
pub struct Posterize {
    source: VideoNode,
    bits: Bits,
    debug: bool,
    /// Set once the settings line has been written for this instance.
    reported: AtomicBool,
}

impl Filter for Posterize {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Posterize";
    const ARGS: &'static CStr = c"clip:vnode;bits:int:opt;use_props:int:opt;debug:int:opt;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "Posterize")?;
        let info = checked_info(&source, "Posterize", Accept::Integer8)?;

        let use_props = read_int(&input, key!(c"use_props"))?.unwrap_or(0) != 0;
        let bits = if use_props {
            Bits::FromShades
        } else {
            let bpc = read_int(&input, key!(c"bits"))?.ok_or_else(|| {
                NImagesError::new(format!(
                    "Posterize: bits is required, and must be between {MIN_BITS} and {MAX_BITS}"
                ))
            })?;

            // cast from i64 to u8
            let bpc_u8 = match u8::try_from(bpc) {
                Ok(bits) => bits,
                Err(_) => {
                    return Err(NImagesError::new(format!(
                        "Posterize: bits must be between {MIN_BITS} and {MAX_BITS}, got {bpc}"
                    )));
                }
            };

            Bits::Fixed(bpc_u8)
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
                bits,
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
                let mut trace = FrameTrace::new(self.debug, "Posterize");

                let input = self.source.get_frame_filter(n, &mut frame_ctx);
                check_frame_format(&input, "Posterize", Accept::Integer8)?;

                let resolved = match self.bits {
                    Bits::Fixed(bits) => resolve_from_bits(bits, "bits=...")?,
                    Bits::FromShades => resolve_from_properties(&input)?,
                };

                let settings = describe_frame(&input);
                report_settings_once(
                    self.debug,
                    &self.reported,
                    &mut core,
                    format_args!(
                        "Posterize: bits={} colors={} input={settings}",
                        resolved.bits,
                        1u32 << resolved.bits
                    ),
                );

                // skip if 8bpc?
                if resolved.bits == 8 {
                    return Ok(Some(input));
                }

                let mark = Instant::now();
                let output = map_frame(&core, &input, &resolved.table)?;
                trace.mark("map", mark);

                trace.emit(&mut core, n, format_args!("bits={} ", resolved.bits));

                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Posterize")),
        }
    }
}

/// Resolves the lookup table for one frame from `PeakGrayShades` properties.
fn resolve_from_properties(frame: &VideoFrame) -> Result<ResolvedBits> {
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
    if !(1..=256).contains(&shade_count) {
        return Err(NImagesError::new(format!(
            "Posterize(use_props=True) needs between 1 and 256 values in \
             NImagesGrayShades, got {shade_count}"
        )));
    }

    // Pick the smallest power-of-two level count that can cover all shades.
    let bits = if shade_count <= 2 {
        MIN_BITS
    } else {
        (usize::BITS - (shade_count - 1).leading_zeros()) as u8
    };

    resolve_from_bits(bits, "use_props=true")
}

fn resolve_from_bits(bits: u8, whence: &str) -> Result<ResolvedBits> {
    match bits {
        0 => Ok(ResolvedBits {
            table: TABLE_BITS_1,
            bits: 1,
        }),
        1 => Ok(ResolvedBits {
            table: TABLE_BITS_1,
            bits: 1,
        }),
        2 => Ok(ResolvedBits {
            table: TABLE_BITS_2,
            bits: 2,
        }),
        3 => Ok(ResolvedBits {
            table: TABLE_BITS_3,
            bits: 3,
        }),
        4 => Ok(ResolvedBits {
            table: TABLE_BITS_4,
            bits: 4,
        }),
        5 => Ok(ResolvedBits {
            table: TABLE_BITS_5,
            bits: 5,
        }),
        6 => Ok(ResolvedBits {
            table: TABLE_BITS_6,
            bits: 6,
        }),
        7 => Ok(ResolvedBits {
            table: TABLE_BITS_7,
            bits: 7,
        }),
        8 => Ok(ResolvedBits {
            table: TABLE_BITS_8,
            bits: 8,
        }),
        other => Err(NImagesError::new(format!(
            "Posterize({whence}): needs between 1 and 8 bpc, got {other} instead"
        ))),
    }
}
