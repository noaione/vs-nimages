//! Book consensus and page eligibility from `nmanga.cli.autolevel`.

use crate::peaks::PeakResult;

/// Eligibility bounds in native sample units, after the offset is applied.
#[derive(Clone, Copy, Debug)]
pub struct BookOptions {
    pub lower: i64,
    pub upper: i64,
    pub offset: i64,
}

/// One page's local analysis and median-filtered dark-content guard.
#[derive(Clone, Copy, Debug)]
pub struct BookPage {
    pub peaks: PeakResult,
    pub has_dark: bool,
}

/// The resolved level and the decision a caller uses to bypass excluded pages.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub struct BookResult {
    pub peaks: PeakResult,
    pub eligible: bool,
    pub used_book_peak: bool,
}

impl BookOptions {
    /// A qualified local peak strictly inside the levelling bounds.
    pub fn usable(&self, peaks: PeakResult) -> bool {
        let level = i64::from(peaks.black).saturating_add(self.offset);
        peaks.black_found && self.lower < level && level < self.upper
    }

    /// Keeps usable local peaks and uses the consensus on eligible fallback pages.
    pub fn resolve(&self, page: BookPage, consensus: Option<u16>) -> BookResult {
        let mut result = BookResult {
            peaks: page.peaks,
            eligible: self.usable(page.peaks),
            used_book_peak: false,
        };
        let level = i64::from(page.peaks.black).saturating_add(self.offset);
        if result.eligible || (page.peaks.black_found && level <= self.lower) {
            return result;
        }
        if let Some(black) = consensus.filter(|_| page.has_dark) {
            result.peaks.black = black;
            result.eligible = true;
            result.used_book_peak = true;
        }
        result
    }
}

/// A strict majority among usable local peaks, with no per-shade allocation.
pub fn consensus(pages: &[BookPage], options: BookOptions) -> Option<u16> {
    let mut candidate = 0;
    let mut balance = 0usize;
    let mut total = 0usize;
    for page in pages.iter().filter(|page| options.usable(page.peaks)) {
        total += 1;
        if balance == 0 {
            candidate = page.peaks.black;
        }
        if page.peaks.black == candidate {
            balance += 1;
        } else {
            balance -= 1;
        }
    }
    let count = pages
        .iter()
        .filter(|page| options.usable(page.peaks) && page.peaks.black == candidate)
        .count();
    (count > total / 2).then_some(candidate)
}

/// Whether a replicated-edge 3x3 median contains a sample at or below `limit`.
/// Only the threshold matters: five low samples make the median low, so no
/// sorted window or full median image is needed.
pub fn has_dark_median(
    width: usize,
    height: usize,
    limit: u16,
    sample: impl Fn(usize, usize) -> u16,
) -> bool {
    if width == 0 || height == 0 {
        return false;
    }
    for y in 0..height {
        let rows = [y.saturating_sub(1), y, y.saturating_add(1).min(height - 1)];
        for x in 0..width {
            let columns = [x.saturating_sub(1), x, x.saturating_add(1).min(width - 1)];
            let mut low = 0;
            for row in rows {
                for column in columns {
                    low += u8::from(sample(column, row) <= limit);
                }
            }
            if low >= 5 {
                return true;
            }
        }
    }
    false
}
