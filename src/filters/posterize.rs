//! `Posterize`: a 256-entry lookup table mapping each sample to one of `2^bits`
//! levels.
//!
//! Like `Levels`, the mapping applies per sample, so any 8 bit integer format is
//! accepted and every plane is rewritten. Posterizing the planes of an RGB clip
//! independently is not the same operation as posterizing its luma, so an RGB
//! caller that wants the grayscale behaviour converts first.

use std::ffi::{CStr, c_void};

use vapoursynth4_rs::frame::{FrameContext, VideoFrame};
use vapoursynth4_rs::map::MapRef;
use vapoursynth4_rs::node::{Filter, Node, VideoNode};
use vapoursynth4_rs::{core::CoreRef, ffi, key};

use crate::error::{NImagesError, Result};
use crate::posterize::{MAX_BITS, MIN_BITS, posterize_lut};

use super::{
    Accept, add_filter, check_frame_format, checked_info, input_failed, map_frame, read_clip,
    read_int,
};

/// Maps each frame to `2^bits` evenly spaced gray values, without dithering.
pub struct Posterize {
    source: VideoNode,
    table: [u8; 256],
}

impl Filter for Posterize {
    type Error = NImagesError;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"Posterize";
    const ARGS: &'static CStr = c"clip:vnode;bits:int;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<()> {
        let source = read_clip(&input, "Posterize")?;
        let info = checked_info(&source, "Posterize", Accept::Integer8)?;

        let bits = read_int(&input, key!(c"bits"))?.ok_or_else(|| {
            NImagesError::new(format!(
                "Posterize: bits is required, and must be between {MIN_BITS} and {MAX_BITS}"
            ))
        })?;
        let table = u8::try_from(bits)
            .ok()
            .and_then(posterize_lut)
            .ok_or_else(|| {
                NImagesError::new(format!(
                    "Posterize: bits must be between {MIN_BITS} and {MAX_BITS}, got {bits}"
                ))
            })?;

        let dependency = source.as_ptr();
        add_filter(
            &mut core,
            output,
            Self::NAME,
            &info,
            Self { source, table },
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
                check_frame_format(&input, "Posterize", Accept::Integer8)?;
                let output = map_frame(&core, &input, &self.table)?;
                Ok(Some(output))
            }
            ffi::VSActivationReason::Error => Err(input_failed("Posterize")),
        }
    }
}
