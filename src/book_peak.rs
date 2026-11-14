//! Book consensus and page eligibility from `nmanga.autolevel.LevellingPolicy`.

use crate::peaks::PeakResult;

/// Eligibility bounds in native sample units, after the offset is applied.
#[derive(Clone, Copy, Debug)]
pub struct BookOptions {
    pub lower: i64,
    pub upper: i64,
    pub offset: i64,
    /// Maximum excess above the book peak, in native sample units.
    pub tolerance: i64,
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

    /// The absolute limit or a dark tone above the book's usual ink.
    fn too_high(&self, peak: u16, book: Option<u16>) -> bool {
        i64::from(peak).saturating_add(self.offset) >= self.upper
            || book.is_some_and(|book| {
                i64::from(peak) > i64::from(book).saturating_add(self.tolerance)
            })
    }

    /// Applies the reference's local/relative limits before the darkest-bin guard.
    pub fn resolve(&self, page: BookPage, consensus: Option<u16>) -> BookResult {
        let too_high = self.too_high(page.peaks.black, consensus);
        let mut result = BookResult {
            peaks: page.peaks,
            eligible: self.usable(page.peaks) && !too_high,
            used_book_peak: false,
        };
        if result.eligible || (page.peaks.black_found && !too_high) {
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
