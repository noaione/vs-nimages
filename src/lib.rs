//! `vapoursynth-nimages` — a VapourSynth plugin for analyzing and manipulating images.
//!
//! The crate is split in two halves:
//!
//! * safe, VapourSynth-free algorithm modules ([`histogram`], [`peaks`],
//!   [`gray_shades`], [`levels`], [`posterize`]) that hold all of the behaviour,
//!   so they can be tested without a core
//! * the filter layer in [`filters`], the only place that touches the plugin API
//!
//! `PeakStats` and `PeakGrayShades` leave pixels alone and attach their results
//! as frame properties, so a caller composes `PeakStats` -> `Levels(use_props=True)`
//! instead of asking for automatic levels in one call.

mod error;
mod filters;

pub mod gray_shades;
pub mod histogram;
pub mod levels;
pub mod peaks;
pub mod posterize;
pub mod round;

use filters::{Levels, PeakGrayShades, PeakStats, Posterize};

vapoursynth4_rs::declare_plugin!(
    c"xyz.n4o.nimages",
    c"nimages",
    c"A collection of analyzer and tooling to manipulate images",
    (0, 1),
    vapoursynth4_rs::VAPOURSYNTH_API_VERSION,
    0,
    (PeakStats, None),
    (PeakGrayShades, None),
    (Levels, None),
    (Posterize, None)
);
