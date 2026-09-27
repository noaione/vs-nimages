//! Rounding shared by the pixel-mapping filters.
//!
//! [`IMPLEMENTATIONS.md`] fixes the first release on Pillow-compatible
//! ties-to-even rounding, which is what `round()` does in Python. Rust's
//! `f64::round()` rounds half away from zero instead, so the distinction is
//! stated here rather than inherited by accident.
//!
//! [`IMPLEMENTATIONS.md`]: ../../docs/IMPLEMENTATIONS.md

/// Converts a rounded sample to `u8`, clamping out-of-range and non-finite
/// values instead of relying on the cast.
///
/// Total for every `f64`: `f64::clamp` propagates `NaN` and the `as` cast maps
/// it to `0`, so no input can panic.
#[inline]
#[must_use]
pub fn to_u8(value: f64) -> u8 {
    value.round_ties_even().clamp(0.0, 255.0) as u8
}

/// Rounds to `places` decimal places with ties to even, matching Python's
/// `round(value, places)`.
#[inline]
#[must_use]
pub fn round_to_places(value: f64, places: i32) -> f64 {
    let scale = 10f64.powi(places);
    if !scale.is_finite() || scale == 0.0 {
        return value;
    }
    (value * scale).round_ties_even() / scale
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn ties_go_to_the_even_neighbour() {
        assert_eq!(to_u8(0.5), 0, "0.5 -> 0, not 1");
        assert_eq!(to_u8(1.5), 2);
        assert_eq!(to_u8(2.5), 2);
        assert_eq!(to_u8(3.5), 4);
        assert_eq!(to_u8(126.5), 126);
        assert_eq!(to_u8(127.5), 128);
    }

    #[test]
    fn to_u8_is_total() {
        assert_eq!(to_u8(-10.0), 0);
        assert_eq!(to_u8(255.0), 255);
        assert_eq!(to_u8(300.0), 255);
        assert_eq!(to_u8(f64::NAN), 0);
        assert_eq!(to_u8(f64::INFINITY), 255);
        assert_eq!(to_u8(f64::NEG_INFINITY), 0);
    }

    #[test]
    fn places_matches_python_round() {
        assert_eq!(round_to_places(1.0 / 0.125, 2), 8.0);
        assert_eq!(round_to_places(1.005, 2), 1.0);
        assert_eq!(round_to_places(1.015, 2), 1.01);
        assert_eq!(round_to_places(1.53, 2), 1.53);
    }
}
