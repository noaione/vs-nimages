//! A replicated-edge 3x3 median histogram, matching Pillow's rank filter.

use crate::histogram::Histogram;

/// The middle of nine samples, using a fixed selection network.
fn middle(values: [u16; 9]) -> u16 {
    let [
        mut a,
        mut b,
        mut c,
        mut d,
        mut e,
        mut f,
        mut g,
        mut h,
        mut i,
    ] = values;
    order(&mut b, &mut c);
    order(&mut e, &mut f);
    order(&mut h, &mut i);
    order(&mut a, &mut b);
    order(&mut d, &mut e);
    order(&mut g, &mut h);
    order(&mut b, &mut c);
    order(&mut e, &mut f);
    order(&mut h, &mut i);
    order(&mut a, &mut d);
    order(&mut f, &mut i);
    order(&mut e, &mut h);
    order(&mut d, &mut g);
    order(&mut b, &mut e);
    order(&mut c, &mut f);
    order(&mut e, &mut h);
    order(&mut e, &mut c);
    order(&mut g, &mut e);
    order(&mut e, &mut c);
    e
}

#[inline]
fn order(a: &mut u16, b: &mut u16) {
    let low = (*a).min(*b);
    *b = (*a).max(*b);
    *a = low;
}

/// Counts median-filtered samples while retaining only three padded luma rows.
/// The callback reads native luma, before any median is applied.
#[must_use]
pub fn histogram(
    width: usize,
    height: usize,
    max_value: u16,
    mut sample: impl FnMut(usize, usize) -> u16,
) -> Option<Histogram> {
    let bins = usize::from(max_value).checked_add(1)?;
    let mut counts = Vec::new();
    counts.try_reserve_exact(bins).ok()?;
    counts.resize(bins, 0u64);
    let total = u64::try_from(width)
        .ok()?
        .checked_mul(u64::try_from(height).ok()?)?;
    if width == 0 || height == 0 {
        return Histogram::from_counts_for_range(counts, total, max_value);
    }
    let length = width.checked_add(2)?;
    let mut rows = [Vec::new(), Vec::new(), Vec::new()];
    for row in &mut rows {
        row.try_reserve_exact(length).ok()?;
        row.resize(length, 0u16);
    }
    let mut load = |row: &mut [u16], y| {
        for (x, value) in row.iter_mut().enumerate() {
            *value = sample(x.saturating_sub(1).min(width - 1), y).min(max_value);
        }
    };
    for (index, row) in rows.iter_mut().enumerate() {
        load(row, index.saturating_sub(1).min(height - 1));
    }
    for y in 0..height {
        let [top, center, bottom] = &rows;
        for ((top, center), bottom) in top.windows(3).zip(center.windows(3)).zip(bottom.windows(3))
        {
            let ([a, b, c], [d, e, f], [g, h, i]) = (top, center, bottom) else {
                return None;
            };
            let value = middle([*a, *b, *c, *d, *e, *f, *g, *h, *i]);
            let count = counts.get_mut(usize::from(value))?;
            *count = count.saturating_add(1);
        }
        if y < height - 1 {
            rows.rotate_left(1);
            load(rows.last_mut()?, y.saturating_add(2).min(height - 1));
        }
    }
    Histogram::from_counts_for_range(counts, total, max_value)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn selection_network_matches_sorted_windows() {
        // Binary windows cover every ordering relevant to a selection network.
        for bits in 0u16..512 {
            let values = std::array::from_fn(|index| (bits >> index) & 1);
            let mut sorted = values;
            sorted.sort_unstable();
            assert_eq!(middle(values), sorted[4]);
        }
        let mut seed = 12345u32;
        for _ in 0..10000 {
            let values = std::array::from_fn(|_| {
                seed = seed.wrapping_mul(1664525).wrapping_add(1013904223);
                seed as u16
            });
            let mut sorted = values;
            sorted.sort_unstable();
            assert_eq!(middle(values), sorted[4]);
        }
    }

    #[test]
    fn empty_and_overflowing_geometry_are_total() {
        assert_eq!(
            histogram(0, 9, 255, |_, _| panic!("empty"))
                .expect("empty")
                .total_pixels(),
            0
        );
        assert!(histogram(usize::MAX, 2, 255, |_, _| 0).is_none());
    }
}
