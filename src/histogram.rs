//! A stride-aware histogram for integer gray samples through 16 bits.
//!
//! `GRAY8` keeps its fixed 256-bin array. Wider formats allocate one bin for
//! every native code value, and all plane readers visit active samples only.

/// Number of shades in an 8-bit sample.
pub const BINS: usize = 256;

// Keep the 8-bit histogram inline to avoid a per-frame heap allocation.
#[expect(
    clippy::large_enum_variant,
    reason = "the inline 8-bit histogram avoids a per-frame heap allocation"
)]
#[derive(Clone, Debug, PartialEq, Eq)]
enum Counts {
    U8([u64; BINS]),
    Wide(Vec<u64>),
}

impl Counts {
    fn as_slice(&self) -> &[u64] {
        match self {
            Self::U8(counts) => counts,
            Self::Wide(counts) => counts,
        }
    }
}

/// Pixel counts for every shade from zero through `max_value`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct Histogram {
    counts: Counts,
    total_pixels: u64,
    max_value: u16,
}

impl Histogram {
    /// An empty 8-bit histogram.
    #[must_use]
    pub const fn new() -> Self {
        Self {
            counts: Counts::U8([0; BINS]),
            total_pixels: 0,
            max_value: u8::MAX as u16,
        }
    }

    /// Counts `height` rows of `width` 8-bit samples, skipping stride padding.
    ///
    /// Returns [`None`] when the described rows do not fit in `data`.
    #[must_use]
    pub fn from_plane(data: &[u8], stride: usize, width: usize, height: usize) -> Option<Self> {
        if width > stride {
            return None;
        }
        validate_plane_length(data.len(), stride, width, height)?;

        let mut counts = [0u64; BINS];
        for row in 0..height {
            let start = row * stride;
            for &value in &data[start..start + width] {
                if let Some(count) = counts.get_mut(usize::from(value)) {
                    *count = count.saturating_add(1);
                }
            }
        }

        Some(Self {
            counts: Counts::U8(counts),
            total_pixels: pixel_count(width, height),
            max_value: u8::MAX as u16,
        })
    }

    /// Counts 16-bit words in rows whose stride is measured in bytes.
    ///
    /// Values above `max_value` are clamped to the format maximum. Returns
    /// [`None`] when a row does not fit or the bounded histogram cannot be
    /// allocated.
    #[must_use]
    pub fn from_u16_plane(
        data: &[u8],
        stride: usize,
        width: usize,
        height: usize,
        max_value: u16,
    ) -> Option<Self> {
        let row_bytes = width.checked_mul(2)?;
        if row_bytes > stride {
            return None;
        }
        validate_plane_length(data.len(), stride, row_bytes, height)?;

        let mut counts = Vec::new();
        let length = usize::from(max_value).checked_add(1)?;
        counts.try_reserve_exact(length).ok()?;
        counts.resize(length, 0u64);

        for row in 0..height {
            let start = row * stride;
            for sample in data[start..start + row_bytes].as_chunks::<2>().0 {
                let value = u16::from_ne_bytes(*sample).min(max_value);
                if let Some(count) = counts.get_mut(usize::from(value)) {
                    *count = count.saturating_add(1);
                }
            }
        }

        Some(Self {
            counts: Counts::Wide(counts),
            total_pixels: pixel_count(width, height),
            max_value,
        })
    }

    /// Builds an 8-bit histogram from counts and an independent pixel total.
    #[must_use]
    pub const fn from_counts(counts: [u64; BINS], total_pixels: u64) -> Self {
        Self {
            counts: Counts::U8(counts),
            total_pixels,
            max_value: u8::MAX as u16,
        }
    }

    /// Builds a wider histogram from counts and an independent pixel total.
    #[must_use]
    pub fn from_counts_for_range(
        counts: Vec<u64>,
        total_pixels: u64,
        max_value: u16,
    ) -> Option<Self> {
        let length = usize::from(max_value).checked_add(1)?;
        if counts.len() != length {
            return None;
        }
        Some(Self {
            counts: Counts::Wide(counts),
            total_pixels,
            max_value,
        })
    }

    /// The count of every shade in ascending code-value order.
    #[must_use]
    pub fn counts(&self) -> &[u64] {
        self.counts.as_slice()
    }

    /// The maximum code value represented by this histogram.
    #[must_use]
    pub const fn max_value(&self) -> u16 {
        self.max_value
    }

    /// The number of pixels the histogram was built from.
    #[must_use]
    pub const fn total_pixels(&self) -> u64 {
        self.total_pixels
    }

    /// The count of one shade, or zero when it is outside the range.
    #[must_use]
    pub fn count(&self, shade: u16) -> u64 {
        self.counts().get(usize::from(shade)).copied().unwrap_or(0)
    }

    /// The region of interest `first..=last`, if it is a valid bin range.
    #[must_use]
    pub fn region(&self, first: u16, last: u16) -> Option<&[u64]> {
        if first > last {
            return None;
        }
        self.counts().get(usize::from(first)..=usize::from(last))
    }
}

impl Default for Histogram {
    fn default() -> Self {
        Self::new()
    }
}

fn pixel_count(width: usize, height: usize) -> u64 {
    (width as u64).saturating_mul(height as u64)
}

fn validate_plane_length(
    data_length: usize,
    stride: usize,
    row_bytes: usize,
    height: usize,
) -> Option<()> {
    if row_bytes > stride {
        return None;
    }
    if height > 0 {
        let last_row = height - 1;
        let end = last_row.checked_mul(stride)?.checked_add(row_bytes)?;
        if end > data_length {
            return None;
        }
    }
    Some(())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counts_ignore_stride_padding() {
        let data = [
            1, 2, 3, 200, 200, 200, 200, 200, //
            4, 5, 6, 201, 201, 201, 201, 201,
        ];
        let histogram = Histogram::from_plane(&data, 8, 3, 2).expect("the rows fit");
        assert_eq!(histogram.total_pixels(), 6, "padding is not a pixel");
        for shade in [1u16, 2, 3, 4, 5, 6] {
            assert_eq!(histogram.count(shade), 1);
        }
        for shade in [200u16, 201] {
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
    fn wide_words_ignore_byte_stride_padding() {
        let samples = [1u16, 2, 3, 4, 5, 6];
        let mut data = vec![0xff; 16];
        for (index, value) in samples.iter().enumerate() {
            let row = index / 3;
            let column = index % 3;
            let offset = row * 8 + column * 2;
            data[offset..offset + 2].copy_from_slice(&value.to_ne_bytes());
        }
        let histogram =
            Histogram::from_u16_plane(&data, 8, 3, 2, u16::MAX).expect("the padded rows fit");
        assert_eq!(histogram.total_pixels(), 6);
        for shade in 1u16..=6 {
            assert_eq!(histogram.count(shade), 1);
        }
        assert_eq!(histogram.count(u16::MAX), 0, "padding is not a sample");
    }

    #[test]
    fn values_above_the_declared_sample_maximum_are_clamped() {
        let data = [u16::MAX.to_ne_bytes()].concat();
        let histogram = Histogram::from_u16_plane(&data, 2, 1, 1, 1023).expect("one sample");
        assert_eq!(histogram.count(1023), 1);
        assert_eq!(histogram.max_value(), 1023);
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
        assert!(Histogram::from_plane(&[0; 64], 4, 8, 1).is_none());
        assert!(Histogram::from_u16_plane(&[0; 7], 8, 4, 1, 1023).is_none());
        assert!(Histogram::from_u16_plane(&[0; 16], 6, 4, 1, 1023).is_none());
    }

    #[test]
    fn absurd_sizes_do_not_panic() {
        assert!(Histogram::from_plane(&[0; 8], usize::MAX, usize::MAX, 2).is_none());
        assert!(Histogram::from_plane(&[0; 8], usize::MAX, 0, 2).is_none());
        assert!(Histogram::from_u16_plane(&[0; 8], usize::MAX, usize::MAX, 2, 65_535).is_none());
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

    #[test]
    fn a_wide_histogram_requires_one_bin_per_code_value() {
        assert!(Histogram::from_counts_for_range(vec![0; 1023], 0, 1023).is_none());
        let histogram = Histogram::from_counts_for_range(vec![0; 1024], 0, 1023)
            .expect("1024 bins cover 10 bits");
        assert_eq!(histogram.counts().len(), 1024);
        assert_eq!(histogram.max_value(), 1023);
    }
}
