//! Posterization: a 256-entry lookup table mapping `GRAY8` to a fixed number
//! of gray levels.
//!
//! The default places the levels evenly:
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
//!
//! [`lloyd_max_levels`] places the levels where a frame's histogram has mass
//! instead, by running the Lloyd-Max solver. The levels come back unrounded,
//! and [`posterize_lut_from_levels_u8`] turns them into the table the plane
//! walk applies. See `docs/FINDINGS.md` §6.3.

/// The smallest number of bits the filter accepts.
pub const MIN_BITS: u8 = 1;
/// The largest number of bits the filter accepts.
pub const MAX_BITS: u8 = 8;

/// Maximum bit depth held by a VapourSynth integer sample.
pub const MAX_INTEGER_SAMPLE_BITS: u8 = 16;

/// Lloyd-Max refinement passes
pub const LLOYD_ITERATIONS: u32 = 40;

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

/// Chooses `colors` gray levels for `counts` with Lloyd-Max, the one-dimensional
/// k-means solver.
///
/// `counts[i]` is how many samples hold code value `i`, so the histogram's own
/// length fixes the code-value range. The first and last level stay pinned to
/// zero and the highest code value, which is what the reference does, and a
/// bucket that holds no sample keeps the even level it started on. Levels come
/// back ascending and distinct, and unrounded: the reference keeps them as
/// floats and rounds only when it writes the table.
///
/// Returns [`PosterizeError::InvalidBits`] for fewer than two colors, for more
/// colors than code values, or for an empty histogram, and
/// [`PosterizeError::Allocation`] when the bounded working storage cannot be
/// reserved.
pub fn lloyd_max_levels(counts: &[u64], colors: usize) -> Result<Vec<f64>, PosterizeError> {
    let code_values = counts.len();
    let max_value = code_values
        .checked_sub(1)
        .ok_or(PosterizeError::InvalidBits)?;
    if colors < 2 || colors > code_values {
        return Err(PosterizeError::InvalidBits);
    }

    let last = colors - 1;
    let mut levels: Vec<f64> = Vec::new();
    let mut sums: Vec<u64> = Vec::new();
    let mut weights: Vec<u64> = Vec::new();
    levels
        .try_reserve_exact(colors)
        .map_err(|_| PosterizeError::Allocation)?;
    sums.try_reserve_exact(colors)
        .map_err(|_| PosterizeError::Allocation)?;
    weights
        .try_reserve_exact(colors)
        .map_err(|_| PosterizeError::Allocation)?;

    for index in 0..colors {
        levels.push(max_value as f64 * index as f64 / last as f64);
    }
    sums.resize(colors, 0);
    weights.resize(colors, 0);

    for _ in 0..LLOYD_ITERATIONS {
        sums.fill(0);
        weights.fill(0);

        // One pass collects every bucket. The edges ascend with the levels, so
        // the bucket index only moves forward as the code value does. A value
        // below its edge goes in the lower bucket, which is the half-open bin
        // `numpy.digitize` produces from the reference's edge list.
        let mut bucket = 0usize;
        for (value, &weight) in counts.iter().enumerate() {
            while bucket < last {
                let edge = (levels[bucket] + levels[bucket + 1]) / 2.0;
                if (value as f64) < edge {
                    break;
                }
                bucket += 1;
            }
            if weight == 0 {
                continue;
            }
            // A frame cannot make either accumulator overflow. The saturating
            // forms keep the pass total whatever a histogram claims.
            let total = weight.saturating_mul(value as u64);
            sums[bucket] = sums[bucket].saturating_add(total);
            weights[bucket] = weights[bucket].saturating_add(weight);
        }

        // Every interior level moves to the mean of its bucket. Both ends are
        // pinned, which is the reference's `range(1, colors - 1)`.
        for index in 1..last {
            let weight = weights[index];
            if weight > 0 {
                levels[index] = sums[index] as f64 / weight as f64;
            }
        }
    }

    Ok(levels)
}

/// Builds a 256-entry 8-bit table that maps every code value to its nearest
/// level.
///
/// `levels` must be ascending and hold 8-bit code values. A value exactly on
/// the midpoint between two levels takes the higher level, matching
/// `numpy.digitize`'s half-open bins, and a level itself rounds ties to even the
/// way `numpy.round` does.
pub fn posterize_lut_from_levels_u8(levels: &[f64]) -> Result<[u8; 256], PosterizeError> {
    if levels.is_empty() {
        return Err(PosterizeError::InvalidBits);
    }

    let mut table = [0u8; 256];
    for (value, target) in table.iter_mut().enumerate() {
        let level = nearest_level(levels, value as u16).ok_or(PosterizeError::InvalidBits)?;
        *target = level.round_ties_even().clamp(0.0, 255.0) as u8;
    }
    Ok(table)
}

/// Builds a table over every code value through `max_value` from `levels`.
///
/// The range spans the format's complete legal range, as [`posterize_lut_u16`]
/// does, and `levels` must be ascending and hold values no greater than
/// `max_value`.
pub fn posterize_lut_from_levels_u16(
    levels: &[f64],
    max_value: u16,
) -> Result<Vec<u16>, PosterizeError> {
    if levels.is_empty() {
        return Err(PosterizeError::InvalidBits);
    }

    let maximum = f64::from(max_value);
    let length = usize::from(max_value) + 1;
    let mut table = Vec::new();
    table
        .try_reserve_exact(length)
        .map_err(|_| PosterizeError::Allocation)?;
    for value in 0..length {
        let level = nearest_level(levels, value as u16).ok_or(PosterizeError::InvalidBits)?;
        table.push(level.round_ties_even().clamp(0.0, maximum) as u16);
    }
    Ok(table)
}

/// The level nearest to `value`, with a midpoint going to the higher level.
///
/// `levels` must be ascending, which makes the midpoints ascending, so the
/// binary search below is the assignment performs with
/// `numpy.digitize` over its list of midpoints.
fn nearest_level(levels: &[f64], value: u16) -> Option<f64> {
    let last = levels.len().checked_sub(1)?;
    let x = f64::from(value);
    let (mut low, mut high) = (0usize, last);
    while low < high {
        let middle = low + (high - low) / 2;
        let edge = (levels[middle] + levels[middle + 1]) / 2.0;
        if x < edge {
            high = middle;
        } else {
            low = middle + 1;
        }
    }
    levels.get(low).copied()
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

    #[test]
    fn lloyd_levels_land_on_the_cluster_means() {
        // One cluster per bucket, so every interior level has mass. The pinned
        // ends stay on the code-value extremes rather than on the clusters.
        // These values come from `lloyd.py` itself; see `docs/FINDINGS.md` §6.3.
        let mut counts = [0u64; 256];
        for shade in [40usize, 90, 200, 250] {
            counts[shade] = 100;
        }

        let levels = lloyd_max_levels(&counts, 4).expect("four colors");
        assert_eq!(levels, [0.0, 90.0, 200.0, 255.0]);
    }

    #[test]
    fn lloyd_keeps_the_even_level_of_an_empty_bucket() {
        // Mass at 40 and 200 only. Bucket 1 spans 43..=127 and stays empty, so
        // it keeps the 85 it started on, and 40 is nearest to the pinned 0.
        let mut counts = [0u64; 256];
        counts[40] = 1_000;
        counts[200] = 1_000;

        let levels = lloyd_max_levels(&counts, 4).expect("four colors");
        assert_eq!(levels, [0.0, 85.0, 200.0, 255.0]);

        let table = posterize_lut_from_levels_u8(&levels).expect("four levels");
        assert_eq!(table[40], 0);
        assert_eq!(table[200], 200);
    }

    #[test]
    fn lloyd_levels_stay_ordered_and_distinct() {
        let mut counts = [0u64; 256];
        for (index, count) in counts.iter_mut().enumerate() {
            *count = ((index * index) % 41) as u64;
        }

        for colors in [2usize, 3, 4, 8, 16, 64, 256] {
            let levels = lloyd_max_levels(&counts, colors).expect("valid colors");
            assert_eq!(levels.len(), colors);
            assert_eq!(levels[0], 0.0, "colors {colors}");
            assert_eq!(levels[colors - 1], 255.0, "colors {colors}");
            assert!(
                levels.windows(2).all(|pair| pair[0] < pair[1]),
                "colors {colors} is not ascending: {levels:?}"
            );
        }
    }

    #[test]
    fn lloyd_is_the_identity_at_the_sample_depth() {
        let counts = [1u64; 256];
        let levels = lloyd_max_levels(&counts, 256).expect("one level per code value");
        let expected: Vec<f64> = (0..=255).map(f64::from).collect();
        assert_eq!(levels, expected);

        let table = posterize_lut_from_levels_u8(&levels).expect("256 levels");
        assert_eq!(table, core::array::from_fn(|index| index as u8));
    }

    #[test]
    fn lloyd_refuses_impossible_color_counts() {
        assert_eq!(lloyd_max_levels(&[], 2), Err(PosterizeError::InvalidBits));

        let counts = [1u64; 4];
        assert_eq!(
            lloyd_max_levels(&counts, 0),
            Err(PosterizeError::InvalidBits)
        );
        assert_eq!(
            lloyd_max_levels(&counts, 1),
            Err(PosterizeError::InvalidBits)
        );
        assert_eq!(
            lloyd_max_levels(&counts, 5),
            Err(PosterizeError::InvalidBits)
        );
        assert!(lloyd_max_levels(&counts, 4).is_ok());
    }

    #[test]
    fn a_table_from_levels_takes_the_nearest_level_with_ties_to_even() {
        // The hand-solved 8-value case: mass at 2 and 6 with four colors settles
        // on levels [0, 2, 14/3, 7], so the emitted 14/3 rounds to 5 and the
        // boundaries sit on the midpoints 3.333 and 5.833.
        let mut counts = [0u64; 8];
        counts[2] = 4;
        counts[6] = 4;

        let levels = lloyd_max_levels(&counts, 4).expect("four colors");
        assert_eq!(levels[0], 0.0);
        assert_eq!(levels[1], 2.0);
        assert!((levels[2] - 14.0 / 3.0).abs() < 1e-12, "got {}", levels[2]);
        assert_eq!(levels[3], 7.0);

        let table = posterize_lut_from_levels_u8(&levels).expect("four levels");
        assert_eq!(table[..8], [0, 2, 2, 2, 5, 5, 7, 7]);

        // A level of its own rounds ties to even, so 2.5 and 4.5 become 2 and 4.
        let even = posterize_lut_from_levels_u8(&[0.0, 2.5, 4.5, 7.0]).expect("four levels");
        assert_eq!(even[..8], [0, 0, 2, 2, 4, 4, 7, 7]);
    }

    #[test]
    fn a_wide_table_from_levels_spans_the_native_range() {
        let levels = [0.0, 512.0, 1_023.0];
        let table = posterize_lut_from_levels_u16(&levels, 1_023).expect("three levels");
        assert_eq!(table.len(), 1_024);
        assert_eq!(table[0], 0);
        assert_eq!(table[255], 0);
        assert_eq!(table[256], 512, "the midpoint goes to the higher level");
        assert_eq!(table[767], 512);
        assert_eq!(
            table[768], 1_023,
            "the second midpoint goes to the higher level"
        );
        assert_eq!(table[1_023], 1_023);
        assert!(table.windows(2).all(|pair| pair[0] <= pair[1]));

        assert_eq!(
            posterize_lut_from_levels_u16(&[], 1_023),
            Err(PosterizeError::InvalidBits)
        );
        assert_eq!(
            posterize_lut_from_levels_u8(&[]),
            Err(PosterizeError::InvalidBits)
        );
    }
}
