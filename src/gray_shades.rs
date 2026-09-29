//! Significant gray shades, a dependency-free replacement for
//! `nmanga.autolevel.analyze_gray_shades`.
//!
//! Two deliberate differences from the reference are documented in
//! [`FINDINGS.md`] §5 and `IMPLEMENTATIONS.md` §4.2:
//!
//! * 8-bit input uses the fixed range `0..=255`, while wider integer input
//!   uses every native code value, so `shade` is always a real gray value
//! * the shade/percentage pair is returned as a value instead of being written
//!   into a frame property by the caller
//!
//! A shade is included only when its count is *strictly* greater than
//! `ceil(total_pixels * threshold / 100)`, and the result is sorted by
//! descending count, keeping ascending shade order on a tie.
//!
//! [`FINDINGS.md`]: ../../docs/FINDINGS.md

use crate::histogram::Histogram;

/// One significant gray shade.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct GrayShade {
    /// The gray value, in source sample units.
    pub shade: u16,
    /// How many pixels have it.
    pub count: u64,
    /// Its share of the complete frame, in percent.
    pub percentage: f64,
}

/// Finds every shade whose share of the frame exceeds `threshold` percent.
///
/// A `threshold` that is negative, `NaN` or infinite is treated as zero, so the
/// function is total for every input.
#[must_use]
pub fn analyze_gray_shades(histogram: &Histogram, threshold: f64) -> Vec<GrayShade> {
    let total_pixels = histogram.total_pixels();
    if total_pixels == 0 {
        return Vec::new();
    }

    let threshold = if threshold.is_finite() && threshold > 0.0 {
        threshold
    } else {
        0.0
    };
    let required = (total_pixels as f64 * (threshold / 100.0)).ceil();
    let required = if required.is_finite() && required > 0.0 {
        required.min(u64::MAX as f64) as u64
    } else {
        0
    };

    let mut shades: Vec<GrayShade> = histogram
        .counts()
        .iter()
        .copied()
        .enumerate()
        .filter_map(|(shade, count)| {
            (count > required).then_some(GrayShade {
                shade: shade as u16,
                count,
                percentage: (count as f64 / total_pixels as f64) * 100.0,
            })
        })
        .collect();

    // `sort_by_key` is stable, so equal counts keep the ascending shade order
    // that the filter_map above produced.
    shades.sort_by_key(|shade| core::cmp::Reverse(shade.count));
    shades
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::histogram::BINS;

    fn histogram(counts: &[(u8, u64)], total_pixels: u64) -> Histogram {
        let mut bins = [0u64; BINS];
        for &(shade, count) in counts {
            bins[shade as usize] = count;
        }
        Histogram::from_counts(bins, total_pixels)
    }

    #[test]
    fn shades_are_gray_values_not_adaptive_bin_indices() {
        // The reference reports shades 0 and 255 here because it bins over the
        // observed 10..20 range. See docs/FINDINGS.md §5.
        let result = analyze_gray_shades(&histogram(&[(10, 50), (20, 50)], 100), 0.01);
        assert_eq!(
            result.iter().map(|shade| shade.shade).collect::<Vec<_>>(),
            [10, 20]
        );
        assert!(result.iter().all(|shade| shade.percentage == 50.0));
    }

    #[test]
    fn a_constant_frame_reports_its_own_shade() {
        for value in [0u8, 1, 128, 200, 255] {
            let result = analyze_gray_shades(&histogram(&[(value, 100)], 100), 0.01);
            assert_eq!(result.len(), 1, "shade {value}");
            assert_eq!(result[0].shade, u16::from(value));
            assert_eq!(result[0].percentage, 100.0);
        }
    }

    #[test]
    fn the_threshold_is_exclusive() {
        // 1000 pixels at 1% is ceil(10) = 10, so 10 is out and 11 is in.
        let at = histogram(&[(10, 10), (20, 990)], 1000);
        let above = histogram(&[(10, 11), (20, 989)], 1000);
        assert!(
            !analyze_gray_shades(&at, 1.0)
                .iter()
                .any(|shade| shade.shade == 10)
        );
        assert!(
            analyze_gray_shades(&above, 1.0)
                .iter()
                .any(|shade| shade.shade == 10)
        );
    }

    #[test]
    fn equal_counts_keep_ascending_shade_order() {
        let result =
            analyze_gray_shades(&histogram(&[(200, 500), (10, 500), (100, 500)], 1500), 0.01);
        assert_eq!(
            result.iter().map(|shade| shade.shade).collect::<Vec<_>>(),
            [10, 100, 200]
        );
    }

    #[test]
    fn larger_counts_come_first() {
        let result =
            analyze_gray_shades(&histogram(&[(10, 300), (20, 800), (30, 100)], 1200), 0.01);
        assert_eq!(
            result.iter().map(|shade| shade.shade).collect::<Vec<_>>(),
            [20, 10, 30]
        );
        let percentages: Vec<f64> = result.iter().map(|shade| shade.percentage).collect();
        assert_eq!(percentages[0], 800.0 / 1200.0 * 100.0);
    }

    #[test]
    fn nothing_significant_is_an_empty_slice() {
        assert!(analyze_gray_shades(&histogram(&[(10, 1), (20, 1)], 100), 0.01).is_empty());
    }

    #[test]
    fn an_empty_frame_produces_nothing() {
        assert!(analyze_gray_shades(&Histogram::new(), 0.01).is_empty());
    }

    #[test]
    fn a_zero_threshold_accepts_every_present_shade() {
        let result = analyze_gray_shades(&histogram(&[(10, 1), (20, 1), (30, 1)], 3), 0.0);
        assert_eq!(result.len(), 3);
    }

    #[test]
    fn absurd_thresholds_do_not_panic() {
        let histogram = histogram(&[(10, 1)], 1);
        for threshold in [f64::NAN, f64::INFINITY, f64::NEG_INFINITY, -1.0, 1e300] {
            let _ = analyze_gray_shades(&histogram, threshold);
        }
        assert!(
            analyze_gray_shades(&histogram, f64::NAN).len() == 1,
            "NaN behaves as zero"
        );
    }

    #[test]
    fn huge_counts_do_not_overflow() {
        let big = 1u64 << 32;
        let result = analyze_gray_shades(
            &histogram(&[(10, big), (200, big), (0, 1)], 2 * big + 1),
            0.01,
        );
        assert_eq!(result.len(), 2);
        assert_eq!(result[0].shade, 10);
        assert_eq!(result[1].shade, 200);
    }

    #[test]
    fn wide_shades_keep_native_code_values_and_sort_ties_by_value() {
        let mut bins = vec![0u64; 65_536];
        bins[60_000] = 50;
        bins[300] = 50;
        bins[1_000] = 100;
        let histogram = Histogram::from_counts_for_range(bins, 200, 65_535)
            .expect("the bins cover the 16-bit range");

        let result = analyze_gray_shades(&histogram, 0.01);
        assert_eq!(
            result.iter().map(|shade| shade.shade).collect::<Vec<_>>(),
            [1_000, 300, 60_000]
        );
        assert_eq!(result[0].percentage, 50.0);
        assert_eq!(result[1].percentage, 25.0);
        assert_eq!(result[2].percentage, 25.0);
    }
}
