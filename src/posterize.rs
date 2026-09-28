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
}
