//! Level adjustment: a 256-entry lookup table for `GRAY8`.
//!
//! For an input `x`, black point `b`, white point `w`, output maximum `Q = 255`
//! and gamma `g` the transfer curve is
//!
//! ```text
//! x < b  -> 0
//! x > w  -> Q
//! else   -> round(Q * ((x - b) / (w - b)) ** (1 / g))
//! ```
//!
//! which is the same curve `ImageMagick -level` applies. Rounding is
//! ties-to-even at every step, per `IMPLEMENTATIONS.md` §16.

use crate::round::{round_to_places, to_u8};

/// Automatic gamma derived from a black point, as `nmanga.gamma_correction`
/// computes it.
///
/// Returns [`None`] when the logarithm is not defined: the expression divides by
/// `1 - black` and takes the logarithm of `(0.5 - black) / (1 - black)`, so any
/// `black_level` of 128 or more is a domain error and 255 divides by zero. The
/// default `upper_limit` of 60 keeps callers well inside the domain, but
/// `upper_limit` may be as large as 255.
#[must_use]
pub fn automatic_gamma(black_level: u8) -> Option<f64> {
    let black = black_level as f64 / 255.0;
    if black >= 0.5 {
        return None;
    }

    let ratio = (0.5 - black) / (1.0 - black);
    if ratio <= 0.0 {
        return None;
    }

    let internal = 0.5f64.ln() / ratio.ln();
    if !internal.is_finite() || internal == 0.0 {
        return None;
    }

    let gamma = 1.0 / internal;
    if !gamma.is_finite() || gamma <= 0.0 {
        return None;
    }

    Some(round_to_places(gamma, 2))
}

/// Reason a parameter set cannot be turned into a curve.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum LevelError {
    /// A parameter was `NaN` or infinite.
    NotFinite,
    /// The black point was not strictly below the white point.
    Reversed,
    /// Gamma was not strictly positive.
    Gamma,
}

impl LevelError {
    /// A message for the frame context or the create function.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::NotFinite => "level parameters must be finite numbers",
            Self::Reversed => "black level must be lower than white level",
            Self::Gamma => "gamma must be greater than zero",
        }
    }
}

/// Validates a level parameter set.
///
/// # Errors
///
/// Returns the first violated rule of `IMPLEMENTATIONS.md` §9.2.
pub fn validate(black: f64, white: f64, gamma: f64) -> Result<(), LevelError> {
    if !black.is_finite() || !white.is_finite() || !gamma.is_finite() {
        return Err(LevelError::NotFinite);
    }
    if black >= white {
        return Err(LevelError::Reversed);
    }
    if gamma <= 0.0 {
        return Err(LevelError::Gamma);
    }
    Ok(())
}

/// Builds the 256-entry `GRAY8` lookup table for a level operation.
///
/// # Errors
///
/// Returns the first violated rule of [`validate`].
pub fn levels_lut(black: f64, white: f64, gamma: f64) -> Result<[u8; 256], LevelError> {
    validate(black, white, gamma)?;

    let delta = white - black;
    let inverse_gamma = 1.0 / gamma;
    let mut table = [0u8; 256];

    for (value, slot) in table.iter_mut().enumerate() {
        let value = value as f64;
        *slot = if value < black {
            0
        } else if value > white {
            255
        } else {
            to_u8(((value - black) / delta).powf(inverse_gamma) * 255.0)
        };
    }

    Ok(table)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identity_is_the_trivial_table() {
        let table = levels_lut(0.0, 255.0, 1.0).expect("valid");
        assert_eq!(table, core::array::from_fn(|index| index as u8));
    }

    #[test]
    fn automatic_gamma_matches_the_reference() {
        let cases = [
            (0u8, 1.0),
            (1, 1.01),
            (5, 1.03),
            (12, 1.07),
            (37, 1.27),
            (60, 1.53),
            (100, 2.49),
            (127, 8.0),
        ];
        for (black_level, expected) in cases {
            assert_eq!(
                automatic_gamma(black_level),
                Some(expected),
                "black {black_level}"
            );
        }
    }

    #[test]
    fn automatic_gamma_rejects_the_domain_errors() {
        for black_level in 128..=255u8 {
            assert_eq!(automatic_gamma(black_level), None, "black {black_level}");
        }
    }

    #[test]
    fn validation_covers_every_rule() {
        assert_eq!(validate(0.0, 255.0, 1.0), Ok(()));
        assert_eq!(validate(f64::NAN, 255.0, 1.0), Err(LevelError::NotFinite));
        assert_eq!(
            validate(0.0, f64::INFINITY, 1.0),
            Err(LevelError::NotFinite)
        );
        assert_eq!(validate(0.0, 255.0, f64::NAN), Err(LevelError::NotFinite));
        assert_eq!(validate(10.0, 10.0, 1.0), Err(LevelError::Reversed));
        assert_eq!(validate(20.0, 10.0, 1.0), Err(LevelError::Reversed));
        assert_eq!(validate(0.0, 255.0, 0.0), Err(LevelError::Gamma));
        assert_eq!(validate(0.0, 255.0, -1.0), Err(LevelError::Gamma));
    }

    #[test]
    fn the_curve_clamps_outside_the_endpoints() {
        let table = levels_lut(10.0, 245.0, 1.0).expect("valid");
        assert_eq!(table[0], 0);
        assert_eq!(table[9], 0);
        assert_eq!(table[245], 255);
        assert_eq!(table[255], 255);
        assert_eq!(table[10], 0, "the black point itself maps to 0");
    }

    #[test]
    fn a_wider_span_increases_contrast() {
        let table = levels_lut(10.0, 245.0, 1.0).expect("valid");
        assert_eq!(table[10], 0, "the black point maps to 0");
        assert_eq!(table[245], 255, "the white point maps to 255");
        // Everything below the midpoint is pushed down, everything above is
        // pushed up: the identity would leave 20 and 200 unchanged.
        assert_eq!(table[20], 11);
        assert_eq!(table[200], 206);
    }

    #[test]
    fn gamma_orders_the_midtones() {
        let low = levels_lut(10.0, 245.0, 0.73).expect("valid");
        let unity = levels_lut(10.0, 245.0, 1.0).expect("valid");
        let high = levels_lut(10.0, 245.0, 1.37).expect("valid");
        assert_eq!((low[128], unity[128], high[128]), (99, 128, 154));
        assert!(low[128] < unity[128] && unity[128] < high[128]);
        // The endpoints are independent of gamma.
        assert_eq!((low[10], low[245]), (high[10], high[245]));
    }

    #[test]
    fn bad_parameters_do_not_produce_a_table() {
        assert!(levels_lut(10.0, 10.0, 1.0).is_err());
        assert!(levels_lut(20.0, 10.0, 1.0).is_err());
        assert!(levels_lut(0.0, 255.0, f64::NAN).is_err());
    }

    #[test]
    fn extreme_but_valid_parameters_do_not_panic() {
        assert!(levels_lut(0.0, 255.0, 1e-9).is_ok());
        assert!(levels_lut(0.0, 255.0, 1e9).is_ok());
        assert!(levels_lut(-1e6, 1e6, 1.0).is_ok());
    }
}
