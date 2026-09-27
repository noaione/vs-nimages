//! `vapoursynth-nimages` — a VapourSynth plugin for analyzing and manipulating images.
//!
//! The crate is split in two halves:
//!
//! * safe, VapourSynth-free algorithm modules ([`histogram`], [`peaks`],
//!   [`gray_shades`], [`levels`], [`posterize`]) that hold all of the behaviour,
//!   so they can be tested without a core
//! * the filter layer, which is the only place that touches the API
//!
//! The filter layer is still a scaffolding spike: a pass-through filter that
//! proves the registration and frame lifecycle work end to end.

pub mod gray_shades;
pub mod histogram;
pub mod levels;
pub mod peaks;
pub mod posterize;
pub mod round;

use std::ffi::{CStr, CString, c_void};

use vapoursynth4_rs::{
    core::CoreRef,
    ffi,
    frame::{FrameContext, VideoFrame},
    key,
    map::{MapPropertyError, MapRef},
    node::{Dependencies, Filter, Node, RequestPattern, VideoNode},
};

fn error(message: impl Into<String>) -> CString {
    let mut bytes = message.into().into_bytes();
    bytes.retain(|byte| *byte != 0);
    CString::new(bytes).expect("filter error message is NUL-free")
}

/// Copies its input frame, pixels and properties alike.
struct PassThrough {
    node: VideoNode,
}

impl Filter for PassThrough {
    type Error = CString;
    type FrameType = VideoFrame;
    type FilterData = ();

    const NAME: &'static CStr = c"PassThrough";
    const ARGS: &'static CStr = c"clip:vnode;";
    const RETURN_TYPE: &'static CStr = c"clip:vnode;";

    fn create(
        input: MapRef,
        output: MapRef,
        _data: Option<Box<Self::FilterData>>,
        mut core: CoreRef,
    ) -> Result<(), Self::Error> {
        let node = input
            .get_video_node(key!(c"clip"), 0)
            .map_err(|err: MapPropertyError| error(format!("clip: {err}")))?;

        let info = node.info().clone();
        let dependencies = [ffi::VSFilterDependency {
            source: node.as_ptr(),
            request_pattern: RequestPattern::StrictSpatial,
        }];
        let dependencies = Dependencies::new(&dependencies)
            .ok_or_else(|| error("too many filter dependencies"))?;

        core.create_video_filter(
            output,
            Self::NAME,
            &info,
            Box::new(Self { node }),
            dependencies,
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
    ) -> Result<Option<Self::FrameType>, Self::Error> {
        match activation_reason {
            ffi::VSActivationReason::Initial => {
                frame_ctx.request_frame_filter(n, &self.node);
                Ok(None)
            }
            ffi::VSActivationReason::AllFramesReady => {
                let source = self.node.get_frame_filter(n, &mut frame_ctx);
                Ok(Some(core.copy_frame(&source)))
            }
            ffi::VSActivationReason::Error => Err(error("failed to generate the input frame")),
        }
    }
}

vapoursynth4_rs::declare_plugin!(
    c"xyz.n4o.nimages",
    c"nimages",
    c"A collection of analyzer and tooling to manipulate images",
    (0, 1),
    vapoursynth4_rs::VAPOURSYNTH_API_VERSION,
    0,
    (PassThrough, None)
);
