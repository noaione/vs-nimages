//! Launch geometry and the data every device kernel shares.
//!
//! Nothing here names a VapourSynth type. That is deliberate: the workgroup
//! arithmetic, the grid limit and the push-block encoding are the parts of a
//! device kernel that can be wrong without a GPU, so they are ordinary unit
//! tests.

use std::ffi::CString;
use std::mem::size_of;

use crate::error::{NImagesError, Result};

/// Invocations per workgroup. Every kernel declares `local_size_x` as this.
pub(super) const WORKGROUP: u32 = 64;

/// The largest number of workgroups the API guarantees along x. A dispatch for a
/// larger plane uses [`Launch::stride`] to loop.
pub(super) const GRID_LIMIT: u32 = 65_535;

/// The first line of every kernel, and the declarations each one repeats.
///
/// A kernel is `shader(&[PREAMBLE, body])`: the core compiles GLSL and caches by
/// source text, so specialising on a format is a `#define` at the top of `body`
/// rather than a second shader file.
pub(super) const PREAMBLE: &str = "\
#version 460
layout(local_size_x = 64) in;
";

/// Work units and the grid one dispatch runs them with.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub(super) struct Launch {
    /// Work units the kernel loops over. One invocation handles one unit, so a
    /// pointwise kernel sets this to the sample count and a per-pixel kernel to
    /// `width * height`.
    pub(super) units: u32,
    /// Workgroups along x. Never zero and never past [`GRID_LIMIT`].
    pub(super) groups: u32,
}

impl Launch {
    /// Plans `units` work units.
    pub(super) fn new(units: u32) -> Self {
        let groups = units.div_ceil(WORKGROUP).clamp(1, GRID_LIMIT);
        Self { units, groups }
    }

    /// The strided loop's step, one group's invocations.
    pub(super) fn stride(self) -> u32 {
        self.groups * WORKGROUP
    }

    /// This launch's grid, for [`ExecContext::dispatch`].
    pub(super) fn grid(self) -> [u32; 3] {
        [self.groups, 1, 1]
    }

    /// Work units for `width * height` pixels.
    pub(super) fn per_pixel(width: u32, height: u32) -> Self {
        Self::new(width.saturating_mul(height))
    }
}

/// Builds one shader source from a body and the shared preamble.
///
/// The NUL check is the only way this fails, and it can only fire on a source
/// literal, which is a build mistake rather than a runtime condition.
pub(super) fn shader(parts: &[&str]) -> Result<CString> {
    let mut source = String::new();
    for part in parts {
        source.push_str(part);
    }
    CString::new(source).map_err(|_| NImagesError::new("gpu: a shader source contains a NUL byte"))
}

/// The bytes of one push-constant struct, in field order.
///
/// Only `#[repr(C)]` structs of 32 bit fields are passed here: those have no
/// padding, and each kernel pins its own size with a `const _: () =
/// assert!(size_of::<T>() == WORDS * 4)` beside the struct. The kernel reads the
/// block as a `layout(push_constant)` struct of the same fields.
pub(super) fn bytes_of<T>(value: &T) -> &[u8] {
    // SAFETY: the caller's const assert proves the struct has no padding, so
    // every byte of it is initialized and the slice covers exactly `T`.
    unsafe { std::slice::from_raw_parts((value as *const T).cast::<u8>(), size_of::<T>()) }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn a_small_launch_uses_one_group_per_workgroup_of_units() {
        let launch = Launch::new(1);
        assert_eq!(launch.units, 1);
        assert_eq!(launch.groups, 1);
        // A partial group still runs a whole workgroup; the kernel masks.
        let launch = Launch::new(WORKGROUP);
        assert_eq!(launch.groups, 1);
        let launch = Launch::new(WORKGROUP + 1);
        assert_eq!(launch.groups, 2);
    }

    #[test]
    fn a_large_launch_stops_at_the_grid_limit_and_strides_the_rest() {
        let units = 24_000_000;
        let launch = Launch::new(units);
        assert_eq!(launch.groups, GRID_LIMIT);
        assert!(u64::from(launch.stride()) < u64::from(units));
        // Every unit is covered: the last pass may overrun, never underrun.
        let stride = u64::from(launch.stride());
        let passes = u64::from(units).div_ceil(stride);
        assert!(stride * (passes - 1) < u64::from(units));
        assert!(u64::from(units) <= stride * passes);
    }

    #[test]
    fn an_empty_plane_still_dispatches_one_group() {
        let launch = Launch::per_pixel(0, 0);
        assert_eq!(launch.units, 0);
        assert_eq!(launch.groups, 1);
    }

    #[test]
    fn a_shader_carries_the_preamble_and_rejects_a_nul() {
        let source = shader(&[PREAMBLE, "void main() {}"]).expect("no NUL");
        let text = source.to_str().expect("ascii");
        assert!(text.starts_with("#version 460"));
        assert!(text.contains("local_size_x = 64"));
        assert!(text.ends_with("void main() {}"));
        assert!(shader(&["#version 460\0"]).is_err());
    }

    #[repr(C)]
    #[derive(Clone, Copy)]
    struct Block {
        count: u32,
        scale: f32,
    }

    const _: () = assert!(size_of::<Block>() == 8);

    #[test]
    fn a_push_block_is_its_fields_in_order_with_no_padding() {
        let block = Block {
            count: 3,
            scale: 0.5,
        };
        let bytes = bytes_of(&block);
        assert_eq!(bytes.len(), 8);
        assert_eq!(&bytes[..4], &3u32.to_ne_bytes());
        assert_eq!(&bytes[4..], &0.5f32.to_ne_bytes());
    }
}
