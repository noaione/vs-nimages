//! Peak detection over a [`Histogram`], a dependency-free replacement for
//! `scipy.signal.find_peaks` as used by `nmanga.autolevel.find_local_peak`.
//!
//! The rules in [`IMPLEMENTATIONS.md`] §7 are reproduced exactly:
//!
//! * a plateau is one peak, placed at its centre with an even-length plateau
//!   rounded down;
//! * the region of interest is padded with one virtual zero bin at each end, so
//!   a peak on a region boundary is still detected — without that padding the
//!   equivalent of `find_peaks` never reports an array edge;
//! * the minimum height and minimum prominence are both inclusive;
//! * the tallest qualifying candidate wins, and a tie keeps the lowest bin;
//! * if nothing qualifies the search repeats with no thresholds at all;
//! * an all-zero region yields nothing, because a peak must be strictly higher
//!   than both neighbours.
//!
//! [`IMPLEMENTATIONS.md`]: ../../docs/IMPLEMENTATIONS.md

use crate::histogram::{BINS, Histogram};

/// The largest region of interest plus its two padding bins.
const PADDED_CAPACITY: usize = BINS + 2;

/// Parameters of one automatic level analysis.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct PeakOptions {
    /// Highest shade considered for the black peak, and the mirror of it for
    /// the white peak.
    pub upper_limit: u8,
    /// Minimum share of the frame a peak must cover, as a percentage. `None`
    /// disables the height threshold.
    pub peak_percentage: Option<f64>,
    /// Minimum prominence a peak must have, as a percentage of the frame.
    /// `None` disables the prominence threshold.
    pub peak_prominence: Option<f64>,
    /// Skip the white analysis and report 255.
    pub skip_white: bool,
}

impl Default for PeakOptions {
    fn default() -> Self {
        Self {
            upper_limit: 60,
            peak_percentage: Some(0.25),
            peak_prominence: None,
            skip_white: false,
        }
    }
}

/// The selected black and white levels.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct PeakResult {
    /// Selected black level in source sample units.
    pub black: u8,
    /// Selected white level in source sample units.
    pub white: u8,
    /// Whether a black peak was found, rather than falling back to 0.
    pub black_found: bool,
    /// Whether a white peak was found, rather than falling back to 255.
    pub white_found: bool,
}

/// Smallest count that satisfies a percentage of the frame.
///
/// `ceil` matches the reference so that a peak sitting exactly on the threshold
/// still qualifies.
fn minimum_count(total_pixels: u64, percentage: Option<f64>) -> u64 {
    match percentage {
        None => 0,
        Some(percentage) => {
            let required = total_pixels as f64 * (percentage / 100.0);
            if !required.is_finite() || required <= 0.0 {
                0
            } else {
                // Saturating: the cast is already saturating, this only makes
                // the intent explicit.
                required.ceil().min(u64::MAX as f64) as u64
            }
        }
    }
}

/// Iterates the plateau-aware local maxima of `counts`, ascending.
struct LocalMaxima<'a> {
    counts: &'a [u64],
    index: usize,
}

impl<'a> LocalMaxima<'a> {
    fn new(counts: &'a [u64]) -> Self {
        Self { counts, index: 1 }
    }
}

impl Iterator for LocalMaxima<'_> {
    type Item = usize;

    fn next(&mut self) -> Option<usize> {
        let counts = self.counts;
        let length = counts.len();
        if length < 3 {
            return None;
        }

        let mut index = self.index;
        while index < length - 1 {
            if counts[index - 1] < counts[index] {
                let mut end = index + 1;
                while end < length && counts[end] == counts[index] {
                    end += 1;
                }
                if end == length {
                    break;
                }
                if counts[end] < counts[index] {
                    // Plateau spanning index..end, centre rounded down.
                    let peak = (index + end - 1) / 2;
                    self.index = end;
                    return Some(peak);
                }
                index = end;
            } else {
                index += 1;
            }
        }

        self.index = length;
        None
    }
}

/// SciPy-compatible prominence of one peak.
///
/// Walking each side stops at the first sample *strictly* higher than the peak;
/// the base is the higher of the two side minima.
fn prominence(counts: &[u64], peak: usize) -> u64 {
    let height = counts[peak];

    let mut left_min = height;
    let mut index = peak;
    while index > 0 {
        index -= 1;
        if counts[index] > height {
            break;
        }
        left_min = left_min.min(counts[index]);
    }

    let mut right_min = height;
    let mut index = peak + 1;
    while index < counts.len() {
        if counts[index] > height {
            break;
        }
        right_min = right_min.min(counts[index]);
        index += 1;
    }

    height - left_min.max(right_min)
}

/// Picks the tallest candidate in a padded region, returning its bin index.
fn select(
    padded: &[u64],
    minimum_height: u64,
    minimum_prominence: u64,
    use_prominence: bool,
) -> Option<usize> {
    let mut best: Option<(usize, u64)> = None;

    for peak in LocalMaxima::new(padded) {
        let height = padded[peak];
        if height < minimum_height {
            continue;
        }
        if use_prominence && prominence(padded, peak) < minimum_prominence {
            continue;
        }
        // Strictly greater keeps the first (lowest bin) on a tie, which is
        // what numpy.argmax does.
        match best {
            Some((_, best_height)) if height <= best_height => {}
            _ => best = Some((peak, height)),
        }
    }

    // Drop the leading padding bin to get back to the bin index.
    best.map(|(peak, _)| peak - 1)
}

/// Analyses one padded region of interest, returning a bin index in
/// `first..=last`.
fn analyse_region(
    counts: &[u64; BINS],
    first: u8,
    last: u8,
    options: &PeakOptions,
    total_pixels: u64,
) -> Option<usize> {
    let region = counts.get(first as usize..=last as usize)?;

    // One virtual zero bin at each end, so region boundaries can be peaks.
    let mut padded = [0u64; PADDED_CAPACITY];
    padded[1..=region.len()].copy_from_slice(region);
    let padded = &padded[..region.len() + 2];

    let minimum_height = minimum_count(total_pixels, options.peak_percentage);
    let minimum_prominence = minimum_count(total_pixels, options.peak_prominence);
    let use_prominence = options.peak_prominence.is_some();

    let found = select(padded, minimum_height, minimum_prominence, use_prominence)
        .or_else(|| select(padded, 0, 0, false));

    found.map(|index| first as usize + index)
}

/// Finds the black and white levels of a frame.
#[must_use]
pub fn find_local_peak(histogram: &Histogram, options: &PeakOptions) -> PeakResult {
    let counts = histogram.counts();
    let total_pixels = histogram.total_pixels();

    let black_index = analyse_region(counts, 0, options.upper_limit, options, total_pixels);
    let black = black_index.map_or(0, |index| index as u8);

    if options.skip_white {
        return PeakResult {
            black,
            white: 255,
            black_found: black_index.is_some(),
            white_found: false,
        };
    }

    let white_first = 255 - options.upper_limit;
    let white_index = analyse_region(counts, white_first, 255, options, total_pixels);
    let white = white_index.map_or(255, |index| index as u8);

    PeakResult {
        black,
        white,
        black_found: black_index.is_some(),
        white_found: white_index.is_some(),
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    fn histogram(counts: &[(u8, u64)], total_pixels: u64) -> Histogram {
        let mut bins = [0u64; BINS];
        for &(shade, count) in counts {
            bins[shade as usize] = count;
        }
        Histogram::from_counts(bins, total_pixels)
    }

    fn defaults(upper_limit: u8) -> PeakOptions {
        PeakOptions {
            upper_limit,
            ..PeakOptions::default()
        }
    }

    #[test]
    fn plateau_centres_round_down() {
        let cases: &[(&[u64], &[usize])] = &[
            (&[0, 1, 5, 5, 5, 1, 0], &[3]),
            (&[0, 5, 5, 0], &[1]),
            (&[0, 5, 5, 5, 5, 0], &[2]),
            (&[0, 5, 5, 5, 5, 5, 0], &[3]),
            (&[0, 1, 5, 5, 1, 0], &[2]),
        ];
        for (values, expected) in cases {
            let found: Vec<usize> = LocalMaxima::new(values).collect();
            assert_eq!(&found, expected, "{values:?}");
        }
    }

    #[test]
    fn boundaries_are_never_peaks_without_padding() {
        assert!(LocalMaxima::new(&[5, 1, 0]).next().is_none());
        assert!(LocalMaxima::new(&[0, 1, 5]).next().is_none());
        assert!(LocalMaxima::new(&[5, 5, 0]).next().is_none());
        assert!(LocalMaxima::new(&[0, 5, 5]).next().is_none());
    }

    #[test]
    fn prominence_stops_at_the_first_higher_sample() {
        // Naively taking the minimum of the whole side would give 1 and a
        // prominence of 3; the correct answer stops at the 5 on the left.
        assert_eq!(prominence(&[3, 1, 4, 5, 2, 4, 1], 5), 2);
        assert_eq!(prominence(&[0, 3, 5, 0], 2), 5);
        assert_eq!(prominence(&[4, 3, 9, 2, 6, 0], 2), 6);
        assert_eq!(prominence(&[4, 3, 9, 2, 6, 0], 4), 4);
        // Equal neighbours do not stop the scan.
        assert_eq!(prominence(&[0, 2, 5, 5, 2, 0], 2), 5);
    }

    #[test]
    fn thresholds_are_inclusive() {
        let options = PeakOptions {
            upper_limit: 60,
            peak_percentage: None,
            peak_prominence: None,
            skip_white: true,
        };
        // 100 pixels, 1% -> 1.
        let at = histogram(&[(10, 1), (200, 99)], 100);
        let below = histogram(&[(10, 0), (200, 100)], 100);
        assert!(
            find_local_peak(
                &at,
                &PeakOptions {
                    peak_percentage: Some(1.0),
                    ..options
                }
            )
            .black_found
        );
        assert!(
            !find_local_peak(
                &below,
                &PeakOptions {
                    peak_percentage: Some(1.0),
                    ..options
                }
            )
            .black_found
        );
    }

    #[test]
    fn the_tallest_peak_wins_and_ties_keep_the_lowest_bin() {
        let options = PeakOptions {
            peak_percentage: None,
            peak_prominence: None,
            skip_white: true,
            ..defaults(60)
        };
        let taller_later = histogram(&[(10, 900), (20, 4000), (200, 1)], 4901);
        assert_eq!(find_local_peak(&taller_later, &options).black, 20);

        let tied = histogram(&[(5, 700), (40, 700), (100, 700)], 2100);
        assert_eq!(find_local_peak(&tied, &options).black, 5);
    }

    #[test]
    fn region_boundaries_can_be_peaks() {
        let options = PeakOptions {
            peak_percentage: None,
            peak_prominence: None,
            skip_white: false,
            ..defaults(60)
        };
        let low = histogram(&[(0, 100), (200, 50)], 150);
        assert_eq!(find_local_peak(&low, &options).black, 0);
        assert!(find_local_peak(&low, &options).black_found);

        let edge = histogram(&[(60, 100), (200, 50)], 150);
        assert_eq!(find_local_peak(&edge, &options).black, 60);

        let mirror = histogram(&[(195, 100), (10, 50)], 150);
        assert_eq!(find_local_peak(&mirror, &options).white, 195);
        assert!(find_local_peak(&mirror, &options).white_found);
    }

    #[test]
    fn nothing_to_find_reports_the_defaults() {
        let options = defaults(60);
        let empty = Histogram::from_counts([0; BINS], 4096);
        let result = find_local_peak(&empty, &options);
        assert_eq!(
            result,
            PeakResult {
                black: 0,
                white: 255,
                black_found: false,
                white_found: false
            }
        );

        let flat = histogram(&[(128, 4096)], 4096);
        let result = find_local_peak(&flat, &options);
        assert_eq!(
            result.black, 0,
            "one bin in the black region with no neighbours"
        );
        assert!(!result.black_found);
        assert_eq!(result.white, 255);
        assert!(!result.white_found);
    }

    #[test]
    fn skipping_white_returns_255() {
        let options = PeakOptions {
            skip_white: true,
            ..defaults(60)
        };
        let black_and_white = histogram(&[(10, 2000), (245, 2000)], 4000);
        let result = find_local_peak(&black_and_white, &options);
        assert_eq!(result.white, 255);
        assert!(!result.white_found);
        assert_eq!(result.black, 10);
    }

    #[test]
    fn the_fallback_ignores_the_thresholds() {
        let options = PeakOptions {
            peak_percentage: Some(50.0),
            peak_prominence: None,
            skip_white: true,
            ..defaults(60)
        };
        let small = histogram(&[(10, 9), (20, 10), (200, 981)], 1000);
        let result = find_local_peak(&small, &options);
        assert_eq!(result.black, 20, "the fallback still picks the tallest");
        assert!(result.black_found);
    }

    #[test]
    fn percentages_are_percentages_not_fractions() {
        // 0.25 means 0.25%, so 1000 pixels need ceil(2.5) = 3.
        assert_eq!(minimum_count(1000, Some(0.25)), 3);
        assert_eq!(minimum_count(1000, Some(0.1)), 1);
        assert_eq!(minimum_count(1000, Some(1.0)), 10);
        assert_eq!(minimum_count(1000, Some(100.0)), 1000);
        assert_eq!(minimum_count(1000, None), 0);
        assert_eq!(minimum_count(1000, Some(0.0)), 0);
        assert_eq!(minimum_count(1000, Some(-1.0)), 0);
        assert_eq!(minimum_count(1000, Some(f64::NAN)), 0);
        assert_eq!(
            minimum_count(1, Some(0.25)),
            1,
            "a 1x1 frame still has ceil(0.0025)"
        );
        assert_eq!(minimum_count(u64::MAX, Some(100.0)), u64::MAX);
    }

    #[test]
    fn the_height_threshold_only_matters_together_with_prominence() {
        // The tallest peak always clears a height threshold if any peak does,
        // and the fallback ignores the thresholds, so `peak_percentage` alone
        // cannot change the answer. Combined with a prominence threshold it can,
        // by emptying the first pass and letting the fallback run.
        let counts = &[(10u8, 4000u64), (20, 50), (200, 46)];
        let total = 4096;

        let height_only = PeakOptions {
            upper_limit: 60,
            peak_percentage: Some(50.0), // needs 2048 pixels
            peak_prominence: None,
            skip_white: true,
        };
        // Nothing clears 2048, so the fallback picks the tallest: 10.
        assert_eq!(
            find_local_peak(&histogram(counts, total), &height_only).black,
            10
        );

        // With prominence the first pass is not empty, so the height threshold
        // removes the otherwise-winning bin 20.
        let combined_low = PeakOptions {
            peak_percentage: Some(0.25), // needs 11 pixels -> 4000 and 50 qualify
            peak_prominence: Some(0.1),  // needs 5
            ..height_only
        };
        assert_eq!(
            find_local_peak(&histogram(counts, total), &combined_low).black,
            10
        );

        let combined_high = PeakOptions {
            peak_percentage: Some(50.0), // needs 2048 -> only 4000 qualifies
            ..combined_low
        };
        assert_eq!(
            find_local_peak(&histogram(counts, total), &combined_high).black,
            10,
            "only the tallest clears the height, so it wins either way"
        );

        // Emptying the first pass is what makes the difference observable.
        let starved = PeakOptions {
            peak_percentage: Some(50.0),
            peak_prominence: Some(50.0), // needs 2048 prominence, nothing has it
            ..combined_low
        };
        assert_eq!(
            find_local_peak(&histogram(counts, total), &starved).black,
            10
        );
    }

    #[test]
    fn huge_counts_do_not_overflow() {
        let huge = 1u64 << 40;
        let options = PeakOptions {
            peak_percentage: Some(0.25),
            peak_prominence: Some(0.1),
            skip_white: false,
            ..defaults(60)
        };
        let counts = histogram(
            &[(12, huge), (200, huge), (0, huge / 2), (255, huge / 2)],
            0,
        );
        let result = find_local_peak(&counts, &options);
        assert_eq!(result.black, 12);
        assert_eq!(result.white, 200);
        assert!(result.black_found && result.white_found);
    }

    #[test]
    fn absurd_limits_do_not_panic() {
        let options = defaults(255);
        let full = histogram(&[(0, 10), (255, 10)], 20);
        let result = find_local_peak(&full, &options);
        assert!(result.black_found && result.white_found);

        let narrow = defaults(0);
        assert!(find_local_peak(&full, &narrow).black_found);
        assert!(find_local_peak(&full, &narrow).white_found);
    }
}
