//! The 256-bin `GRAY8` histogram shared by every analyzer.
//!
//! Counts are `u64` so that a very large frame cannot overflow, and pixels are
//! read row by row over `width` samples only: stride padding is never counted.

/// Number of shades in an 8-bit sample.
pub const BINS: usize = 256;

/// Pixel counts for every shade from 0 to 255.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    counts: [u64; BINS],
    total_pixels: u64,
}

impl Histogram {
    /// An empty histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: [0; BINS],
            total_pixels: 0,
        }
    }

    /// Counts `height` rows of `width` samples each, skipping `stride - width`
    /// padding bytes at the end of every row.
    ///
    /// Returns [`None`] when the described rows do not fit in `data`, so a
    /// malformed frame cannot cause an out-of-bounds read.
    #[must_use]
    pub fn from_plane(data: &[u8], stride: usize, width: usize, height: usize) -> Option<Self> {
        if width > stride {
            return None;
        }

        if height > 0 {
            let last_row = height - 1;
            let end = last_row.checked_mul(stride)?.checked_add(width)?;
            if end > data.len() {
                return None;
            }
        }

        let mut counts = [0u64; BINS];
        for row in 0..height {
            let start = row * stride;
            let row_data = &data[start..start + width];
            for &value in row_data {
                // A frame cannot hold enough pixels to overflow u64.
                counts[value as usize] += 1;
            }
        }

        Some(Self {
            counts,
            total_pixels: (width as u64).saturating_mul(height as u64),
        })
    }

    /// Builds a histogram from counts and an independent pixel total.
    ///
    /// The total is normally `counts.iter().sum()`, but the analyzer's
    /// thresholds are defined against the frame's pixel count, so tests and
    /// callers may pass a different value.
    #[must_use]
    pub const fn from_counts(counts: [u64; BINS], total_pixels: u64) -> Self {
        Self {
            counts,
            total_pixels,
        }
    }

    /// The count of every shade.
    #[must_use]
    pub const fn counts(&self) -> &[u64; BINS] {
        &self.counts
    }

    /// The number of pixels the histogram was built from.
    #[must_use]
    pub const fn total_pixels(&self) -> u64 {
        self.total_pixels
    }

    /// The count of one shade.
    #[must_use]
    pub const fn count(&self, shade: u8) -> u64 {
        self.counts[shade as usize]
    }

    /// The region of interest `first..=last`, if it is a valid bin range.
    #[must_use]
    pub fn region(&self, first: u8, last: u8) -> Option<&[u64]> {
        if first > last {
            return None;
        }
        self.counts.get(first as usize..=last as usize)
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_ignore_stride_padding() {
        // Two rows of three samples, padded to a stride of eight.
        let data = [
            1, 2, 3, 200, 200, 200, 200, 200, //
            4, 5, 6, 201, 201, 201, 201, 201,
        ];
        let histogram = Histogram::from_plane(&data, 8, 3, 2).expect("the rows fit");
        assert_eq!(histogram.total_pixels(), 6, "padding is not a pixel");
        for shade in [1u8, 2, 3, 4, 5, 6] {
            assert_eq!(histogram.count(shade), 1);
        }
        for shade in [200u8, 201] {
            assert_eq!(histogram.count(shade), 0, "padding must not be counted");
        }
        assert_eq!(histogram.counts().iter().sum::<u64>(), 6);
    }

    #[test]
    fn a_tightly_packed_plane_works() {
        let data: Vec<u8> = (0..=255).collect();
        let histogram = Histogram::from_plane(&data, 256, 256, 1).expect("the row fits");
        assert_eq!(histogram.total_pixels(), 256);
        assert!(histogram.counts().iter().all(|&count| count == 1));
    }

    #[test]
    fn zero_sized_planes_are_accepted() {
        let histogram = Histogram::from_plane(&[], 0, 0, 0).expect("nothing is read");
        assert_eq!(histogram.total_pixels(), 0);
        assert_eq!(histogram.counts().iter().sum::<u64>(), 0);
    }

    #[test]
    fn a_row_that_does_not_fit_is_rejected() {
        assert!(Histogram::from_plane(&[0; 7], 8, 8, 1).is_none());
        assert!(Histogram::from_plane(&[0; 15], 8, 8, 2).is_none());
        assert!(Histogram::from_plane(&[0; 16], 8, 8, 2).is_some());
        // A width larger than the stride would read the next row's pixels.
        assert!(Histogram::from_plane(&[0; 64], 4, 8, 1).is_none());
    }

    #[test]
    fn absurd_sizes_do_not_panic() {
        assert!(Histogram::from_plane(&[0; 8], usize::MAX, usize::MAX, 2).is_none());
        assert!(Histogram::from_plane(&[0; 8], usize::MAX, 0, 2).is_none());
    }

    #[test]
    fn region_selects_an_inclusive_range() {
        let mut counts = [0u64; BINS];
        counts[0] = 5;
        counts[60] = 7;
        counts[255] = 9;
        let histogram = Histogram::from_counts(counts, 21);
        assert_eq!(histogram.region(0, 60).map(<[u64]>::len), Some(61));
        assert_eq!(histogram.region(0, 255).map(<[u64]>::len), Some(256));
        assert_eq!(histogram.region(195, 255).map(<[u64]>::len), Some(61));
        assert!(histogram.region(60, 0).is_none());
    }
}
