//! Posterization: a 256-entry lookup table mapping `GRAY8` to `2^bits` evenly
//! spaced gray values.
//!
//! ```text
//! colors = 2 ** bits
//! level  = round(x * (colors - 1) / 255)
//! out    = round(level * 255 / (colors - 1))
//! ```
//!
//! Both steps round ties to even, matching Python's `round()` and Pillow's
//! `Image.point()` with a float-returning lambda.
//!
//! The reference then calls `quantize(colors, dither=NONE)` and converts back to
//! `L`. That step is redundant and is deliberately omitted: for every `bits` in
//! `1..=8` the mapping above already produces exactly `colors` distinct gray
//! values, and Pillow's quantized output is byte-identical to its input. See
//! `IMPLEMENTATIONS.md` §4.4 and `docs/FINDINGS.md` §6.1.

/// The smallest number of bits the filter accepts.
pub const MIN_BITS: u8 = 1;
/// The largest number of bits the filter accepts.
pub const MAX_BITS: u8 = 8;

/// Maximum bit depth held by a VapourSynth integer sample.
pub const MAX_INTEGER_SAMPLE_BITS: u8 = 16;

/// Builds the 256-entry `GRAY8` lookup table for one posterization depth.
///
/// Returns [`None`] when `bits` is outside `1..=8`.
#[must_use]
pub const fn posterize_lut(bits: u8) -> Option<[u8; 256]> {
    if bits < MIN_BITS || bits > MAX_BITS {
        return None;
    }

    let levels = (1u32 << bits) - 1;
    let mut table = [0u8; 256];
    let mut value = 0u32;

    while value < 256 {
        let level = round_ratio_ties_even(value * levels, 255);
        table[value as usize] = round_ratio_ties_even(level * 255, levels) as u8;
        value += 1;
    }

    Some(table)
}

/// Reason a wider posterization table could not be built.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum PosterizeError {
    /// `bits` is outside `1..=sample_depth` or `sample_depth` is outside `1..=16`.
    InvalidBits,
    /// The table could not reserve its bounded storage.
    Allocation,
}

impl PosterizeError {
    /// A message suitable for the filter boundary.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::InvalidBits => "posterization bits must be between 1 and the sample depth",
            Self::Allocation => "could not allocate the posterization lookup table",
        }
    }
}

/// Builds a lookup table for an integer sample depth above 8 bits.
///
/// Both the input and output range span the complete legal range for
/// `sample_depth`, even when that range is narrower than `u16`.
pub fn posterize_lut_u16(bits: u8, sample_depth: u8) -> Result<Vec<u16>, PosterizeError> {
    if !(1..=MAX_INTEGER_SAMPLE_BITS).contains(&sample_depth) || bits == 0 || bits > sample_depth {
        return Err(PosterizeError::InvalidBits);
    }

    let max_value = (1u32 << sample_depth) - 1;
    let levels = (1u32 << bits) - 1;
    let length = max_value as usize + 1;
    let mut table = Vec::new();
    table
        .try_reserve_exact(length)
        .map_err(|_| PosterizeError::Allocation)?;
    table.resize(length, 0);

    for (value, target) in table.iter_mut().enumerate() {
        let level = round_ratio_ties_even(value as u32 * levels, max_value);
        *target = round_ratio_ties_even(level * max_value, levels) as u16;
    }

    Ok(table)
}

/// Rounds a nonnegative rational number to its nearest integer, with ties to even.
const fn round_ratio_ties_even(numerator: u32, denominator: u32) -> u32 {
    let quotient = numerator / denominator;
    let remainder = numerator % denominator;
    let doubled_remainder = remainder * 2;

    if doubled_remainder > denominator
        || (doubled_remainder == denominator && !quotient.is_multiple_of(2))
    {
        quotient + 1
    } else {
        quotient
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn every_depth_produces_exactly_two_to_the_bits_levels() {
        for bits in MIN_BITS..=MAX_BITS {
            let table = posterize_lut(bits).expect("in range");
            let mut seen = [false; 256];
            for value in table {
                seen[value as usize] = true;
            }
            let levels = seen.iter().filter(|present| **present).count();
            assert_eq!(levels, 1usize << bits, "bits {bits}");
        }
    }

    #[test]
    fn the_endpoints_are_preserved() {
        for bits in MIN_BITS..=MAX_BITS {
            let table = posterize_lut(bits).expect("in range");
            assert_eq!(table[0], 0, "bits {bits}");
            assert_eq!(table[255], 255, "bits {bits}");
        }
    }

    #[test]
    fn the_levels_are_evenly_spaced() {
        for bits in MIN_BITS..=MAX_BITS {
            let table = posterize_lut(bits).expect("in range");
            let mut levels: Vec<u8> = table.to_vec();
            levels.sort_unstable();
            levels.dedup();
            let colors = 1usize << bits;
            assert_eq!(levels.len(), colors);
            for (index, level) in levels.iter().enumerate() {
                let expected =
                    ((index as f64) * 255.0 / (colors - 1) as f64).round_ties_even() as u8;
                assert_eq!(*level, expected, "bits {bits} level {index}");
            }
        }
    }

    #[test]
    fn eight_bits_is_the_identity() {
        let table = posterize_lut(8).expect("in range");
        assert_eq!(table, core::array::from_fn(|index| index as u8));
    }

    #[test]
    fn the_mapping_is_monotonic() {
        for bits in MIN_BITS..=MAX_BITS {
            let table = posterize_lut(bits).expect("in range");
            assert!(
                table.windows(2).all(|pair| pair[0] <= pair[1]),
                "bits {bits} is not monotonic"
            );
        }
    }

    #[test]
    fn a_depth_outside_the_range_is_rejected() {
        assert!(posterize_lut(0).is_none());
        assert!(posterize_lut(9).is_none());
        assert!(posterize_lut(u8::MAX).is_none());
        assert!(posterize_lut(MIN_BITS).is_some());
        assert!(posterize_lut(MAX_BITS).is_some());
    }

    #[test]
    fn the_midpoint_lands_on_the_nearest_level() {
        // 8 levels: 255/7 = 36.43, so the midpoints are 36, 73, 109, 146, 182, 219.
        let table = posterize_lut(3).expect("in range");
        let mut levels: Vec<u8> = table.to_vec();
        levels.sort_unstable();
        levels.dedup();
        assert_eq!(levels, [0, 36, 73, 109, 146, 182, 219, 255]);
    }

    #[test]
    fn sixteen_bit_posterization_uses_the_full_native_range() {
        for bits in [1, 2, 8, 15, 16] {
            let table = posterize_lut_u16(bits, 16).expect("valid depth");
            assert_eq!(table.len(), 65_536);
            assert_eq!(table[0], 0, "bits {bits}");
            assert_eq!(table[65_535], u16::MAX, "bits {bits}");
            assert!(table.windows(2).all(|pair| pair[0] <= pair[1]));
            let mut seen = vec![false; 65_536];
            for value in table {
                seen[usize::from(value)] = true;
            }
            assert_eq!(
                seen.into_iter().filter(|present| *present).count(),
                1usize << bits,
                "bits {bits}"
            );
        }
    }

    #[test]
    fn lower_integer_depth_uses_its_own_peak_value() {
        let table = posterize_lut_u16(8, 10).expect("valid depth");
        assert_eq!(table.len(), 1_024);
        assert_eq!(table[1_023], 1_023);
        assert_eq!(
            table
                .iter()
                .copied()
                .collect::<std::collections::BTreeSet<_>>()
                .len(),
            256
        );
    }

    #[test]
    fn wide_posterization_rejects_bits_above_the_sample_depth() {
        assert_eq!(posterize_lut_u16(0, 16), Err(PosterizeError::InvalidBits));
        assert_eq!(posterize_lut_u16(11, 10), Err(PosterizeError::InvalidBits));
        assert_eq!(posterize_lut_u16(1, 17), Err(PosterizeError::InvalidBits));
    }
}
