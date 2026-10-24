//! The luma `Image.convert("L")` computes, scalar and vectorised.
//!
//! The reference pipeline forces every non-gray image to `L` before it counts a
//! histogram, and Pillow's conversion is BT.601 luma at 16 bit fixed point with
//! rounding. The weights sum to 65536, so a gray triple comes back as itself,
//! which is what makes an RGB page and the same page decoded as gray report the
//! same properties.
//!
//! [`rows_u8`] and [`rows_u16`] compute a row of samples at a time into a caller
//! buffer. The scalar expression here is the definition; the kernels compute the
//! same integer expression in the same order, so both forms agree bit for bit
//! and neither can drift from the reference.

/// The red weight, out of 65536.
const RED: u32 = 19_595;
/// The green weight, out of 65536.
const GREEN: u32 = 38_470;
/// The blue weight, out of 65536.
const BLUE: u32 = 7_471;
/// The rounding term, half of the 65536 divisor.
const ROUND: u32 = 32_768;

/// Luma of one 8-bit triple.
///
/// Three 8-bit samples peak at 255 * 65536 before the rounding term, which fits
/// a `u32`, and the shift brings the result back under 256.
pub(crate) fn luma(red: u8, green: u8, blue: u8) -> u8 {
    let sum = u32::from(red) * RED + u32::from(green) * GREEN + u32::from(blue) * BLUE + ROUND;
    u8::try_from(sum >> 16).unwrap_or(u8::MAX)
}

/// The same luma over native 16-bit code values.
///
/// Three 16-bit samples peak at 65535 * 65536 = 4 294 901 760 before the
/// rounding term and at 4 294 934 528 with it, both under `u32::MAX`, so the
/// wide samples need no 64-bit path.
pub(crate) fn luma_u16(red: u16, green: u16, blue: u16) -> u16 {
    let sum = u32::from(red) * RED + u32::from(green) * GREEN + u32::from(blue) * BLUE + ROUND;
    u16::try_from(sum >> 16).unwrap_or(u16::MAX)
}

/// Fills `out` with the luma of three 8-bit rows.
///
/// The sample count is the shortest of the four lengths. Every caller here cuts
/// the four slices to the same width, so the minimum is the width; a mismatched
/// call writes fewer samples rather than panicking, because the release profile
/// aborts the host on a panic.
pub(crate) fn rows_u8(red: &[u8], green: &[u8], blue: &[u8], out: &mut [u8]) {
    let count = red.len().min(green.len()).min(blue.len()).min(out.len());
    if count == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        // `sse2` is part of the x86-64 baseline, so this compiles to a constant
        // and the branch disappears on every build that can take it.
        if count >= 16 && std::arch::is_x86_feature_detected!("sse2") {
            // SAFETY: `count` is the minimum of all four lengths, which is
            // exactly what the kernel's contract asks for.
            unsafe { x86::rows_u8(red, green, blue, out, count) };
            return;
        }
    }

    for index in 0..count {
        out[index] = luma(red[index], green[index], blue[index]);
    }
}

/// Fills `out` with the luma of three native 16-bit rows, whose strides are
/// byte strides.
///
/// The sample count is the shortest of the four lengths, measured in samples;
/// each row holds `count * 2` bytes. The same reasoning as [`rows_u8`] applies
/// to a mismatched call.
pub(crate) fn rows_u16(red: &[u8], green: &[u8], blue: &[u8], out: &mut [u16]) {
    let count = (red.len() / 2)
        .min(green.len() / 2)
        .min(blue.len() / 2)
        .min(out.len());
    if count == 0 {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    {
        if count >= 8 && std::arch::is_x86_feature_detected!("sse4.1") {
            // SAFETY: `count` is the minimum sample count of all four slices,
            // so each byte slice holds `count * 2` bytes.
            unsafe { x86::rows_u16(red, green, blue, out, count) };
            return;
        }
    }

    for (index, sample) in out.iter_mut().enumerate().take(count) {
        let start = index * 2;
        *sample = luma_u16(
            u16::from_ne_bytes([red[start], red[start + 1]]),
            u16::from_ne_bytes([green[start], green[start + 1]]),
            u16::from_ne_bytes([blue[start], blue[start + 1]]),
        );
    }
}

#[cfg(target_arch = "x86_64")]
mod x86 {
    //! The vector kernels. `pmaddwd` and `pmulld` carry the weights, and the
    //! pack instructions put the results back in sample order.
    use super::{BLUE, GREEN, RED, ROUND, luma, luma_u16};
    use std::arch::x86_64::{
        __m128i, _mm_add_epi32, _mm_cvtepu16_epi32, _mm_loadu_si128, _mm_madd_epi16,
        _mm_mullo_epi32, _mm_packs_epi32, _mm_packus_epi16, _mm_packus_epi32, _mm_set1_epi32,
        _mm_setr_epi16, _mm_setzero_si128, _mm_srli_epi32, _mm_srli_si128, _mm_storeu_si128,
        _mm_unpackhi_epi8, _mm_unpackhi_epi16, _mm_unpacklo_epi8, _mm_unpacklo_epi16,
    };

    /// The green weight halved, so both of its products fit a signed 16-bit
    /// source lane. `pmaddwd` reads both operands as signed and the weight does
    /// not fit, so the green term is summed twice at half weight instead.
    const GREEN_HALF: i16 = (GREEN / 2) as i16;
    /// Eight samples per iteration: one 128-bit vector of 16-bit words.
    const U8_STEP: usize = 16;
    /// Eight samples per iteration at native 16-bit depth.
    const U16_STEP: usize = 8;

    /// Four luma values as `i32` lanes.
    ///
    /// `red_green` is four `(red, green)` pairs and `green_blue` the matching
    /// `(green, blue)` pairs, so the sum carries each green sample twice at half
    /// weight.
    #[inline]
    #[target_feature(enable = "sse2")]
    unsafe fn luma4(
        red_green: __m128i,
        green_blue: __m128i,
        rg: __m128i,
        gb: __m128i,
        round: __m128i,
    ) -> __m128i {
        let sum = _mm_add_epi32(
            _mm_madd_epi16(red_green, rg),
            _mm_madd_epi16(green_blue, gb),
        );
        _mm_srli_epi32(_mm_add_epi32(sum, round), 16)
    }

    /// Four luma values as `i32` lanes at native 16-bit depth.
    #[inline]
    #[target_feature(enable = "sse4.1")]
    unsafe fn luma4_wide(
        red: __m128i,
        green: __m128i,
        blue: __m128i,
        weights: (__m128i, __m128i, __m128i),
        round: __m128i,
    ) -> __m128i {
        let (red_weight, green_weight, blue_weight) = weights;
        let sum = _mm_add_epi32(
            _mm_mullo_epi32(red, red_weight),
            _mm_mullo_epi32(green, green_weight),
        );
        let sum = _mm_add_epi32(sum, _mm_mullo_epi32(blue, blue_weight));
        // The total is exact modulo 2^32 because it is under 2^32, so the
        // wrapped sum's high half is the answer.
        _mm_srli_epi32(_mm_add_epi32(sum, round), 16)
    }

    /// Computes `count` samples of 8-bit luma, sixteen at a time.
    ///
    /// # Safety
    /// `red`, `green`, `blue` and `out` hold at least `count` samples, and
    /// `count` is at most `out.len()`.
    #[target_feature(enable = "sse2")]
    pub(super) unsafe fn rows_u8(
        red: &[u8],
        green: &[u8],
        blue: &[u8],
        out: &mut [u8],
        count: usize,
    ) {
        let rg = _mm_setr_epi16(
            RED as i16, GREEN_HALF, RED as i16, GREEN_HALF, RED as i16, GREEN_HALF, RED as i16,
            GREEN_HALF,
        );
        let gb = _mm_setr_epi16(
            GREEN_HALF,
            BLUE as i16,
            GREEN_HALF,
            BLUE as i16,
            GREEN_HALF,
            BLUE as i16,
            GREEN_HALF,
            BLUE as i16,
        );
        let round = _mm_set1_epi32(ROUND as i32);
        let zero = _mm_setzero_si128();
        let whole = count & !(U8_STEP - 1);
        let mut index = 0;

        // SAFETY: every access is a 16-byte load or store at `index`, and
        // `index + 16 <= count` throughout, so each one stays inside the slice
        // it reads or writes. The loads are unaligned because a row starts
        // wherever the frame and the caller's chunk put it.
        unsafe {
            while index < whole {
                let r = _mm_loadu_si128(red.as_ptr().add(index).cast());
                let g = _mm_loadu_si128(green.as_ptr().add(index).cast());
                let b = _mm_loadu_si128(blue.as_ptr().add(index).cast());

                let r_lo = _mm_unpacklo_epi8(r, zero);
                let g_lo = _mm_unpacklo_epi8(g, zero);
                let b_lo = _mm_unpacklo_epi8(b, zero);
                let r_hi = _mm_unpackhi_epi8(r, zero);
                let g_hi = _mm_unpackhi_epi8(g, zero);
                let b_hi = _mm_unpackhi_epi8(b, zero);

                let lo = _mm_packs_epi32(
                    luma4(
                        _mm_unpacklo_epi16(r_lo, g_lo),
                        _mm_unpacklo_epi16(g_lo, b_lo),
                        rg,
                        gb,
                        round,
                    ),
                    luma4(
                        _mm_unpackhi_epi16(r_lo, g_lo),
                        _mm_unpackhi_epi16(g_lo, b_lo),
                        rg,
                        gb,
                        round,
                    ),
                );
                let hi = _mm_packs_epi32(
                    luma4(
                        _mm_unpacklo_epi16(r_hi, g_hi),
                        _mm_unpacklo_epi16(g_hi, b_hi),
                        rg,
                        gb,
                        round,
                    ),
                    luma4(
                        _mm_unpackhi_epi16(r_hi, g_hi),
                        _mm_unpackhi_epi16(g_hi, b_hi),
                        rg,
                        gb,
                        round,
                    ),
                );
                let bytes = _mm_packus_epi16(lo, hi);
                _mm_storeu_si128(out.as_mut_ptr().add(index).cast(), bytes);
                index += U8_STEP;
            }
        }

        while index < count {
            out[index] = luma(red[index], green[index], blue[index]);
            index += 1;
        }
    }

    /// Computes `count` samples of native 16-bit luma, eight at a time.
    ///
    /// # Safety
    /// `red`, `green` and `blue` hold at least `count * 2` bytes and `out`
    /// holds at least `count` samples.
    #[target_feature(enable = "sse4.1")]
    pub(super) unsafe fn rows_u16(
        red: &[u8],
        green: &[u8],
        blue: &[u8],
        out: &mut [u16],
        count: usize,
    ) {
        let weights = (
            _mm_set1_epi32(RED as i32),
            _mm_set1_epi32(GREEN as i32),
            _mm_set1_epi32(BLUE as i32),
        );
        let round = _mm_set1_epi32(ROUND as i32);
        let whole = count & !(U16_STEP - 1);
        let mut index = 0;

        // SAFETY: a 16-byte load at `index * 2` reads samples `index..index + 8`
        // and `index + 8 <= count`, so each one stays inside its row; the store
        // is the same eight samples of `out`.
        unsafe {
            while index < whole {
                let offset = index * 2;
                let r = _mm_loadu_si128(red.as_ptr().add(offset).cast());
                let g = _mm_loadu_si128(green.as_ptr().add(offset).cast());
                let b = _mm_loadu_si128(blue.as_ptr().add(offset).cast());

                let low = luma4_wide(
                    _mm_cvtepu16_epi32(r),
                    _mm_cvtepu16_epi32(g),
                    _mm_cvtepu16_epi32(b),
                    weights,
                    round,
                );
                let high = luma4_wide(
                    _mm_cvtepu16_epi32(_mm_srli_si128(r, 8)),
                    _mm_cvtepu16_epi32(_mm_srli_si128(g, 8)),
                    _mm_cvtepu16_epi32(_mm_srli_si128(b, 8)),
                    weights,
                    round,
                );
                let packed = _mm_packus_epi32(low, high);
                _mm_storeu_si128(out.as_mut_ptr().add(index).cast(), packed);
                index += U16_STEP;
            }
        }

        while index < count {
            let start = index * 2;
            out[index] = luma_u16(
                u16::from_ne_bytes([red[start], red[start + 1]]),
                u16::from_ne_bytes([green[start], green[start + 1]]),
                u16::from_ne_bytes([blue[start], blue[start + 1]]),
            );
            index += 1;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The reference conversion measured against Pillow on 4096 colours.
    fn scalar_u8(red: &[u8], green: &[u8], blue: &[u8], out: &mut [u8]) {
        for (index, sample) in out.iter_mut().enumerate() {
            *sample = luma(red[index], green[index], blue[index]);
        }
    }

    fn scalar_u16(red: &[u8], green: &[u8], blue: &[u8], out: &mut [u16]) {
        for (index, sample) in out.iter_mut().enumerate() {
            let start = index * 2;
            *sample = luma_u16(
                u16::from_ne_bytes([red[start], red[start + 1]]),
                u16::from_ne_bytes([green[start], green[start + 1]]),
                u16::from_ne_bytes([blue[start], blue[start + 1]]),
            );
        }
    }

    /// A seeded xorshift, so a failure is reproducible.
    struct Noise(u64);

    impl Noise {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
    }

    #[test]
    fn the_conversion_matches_the_reference() {
        assert_eq!(luma(255, 0, 0), 76);
        assert_eq!(luma(0, 255, 0), 150);
        assert_eq!(luma(0, 0, 255), 29);
        assert_eq!(luma(10, 200, 30), 124);

        // A gray triple is its own luma, which is what makes an RGB page and the
        // same page decoded as gray report the same properties.
        for value in 0..=255u8 {
            assert_eq!(luma(value, value, value), value);
        }
        assert_eq!(luma_u16(0, 0, 0), 0);
        assert_eq!(luma_u16(65_535, 65_535, 65_535), 65_535);
        assert_eq!(luma_u16(65_535, 0, 0), 19_595);
        for value in [0u16, 1, 1023, 4095, 65_535] {
            assert_eq!(luma_u16(value, value, value), value);
        }
    }

    #[test]
    fn every_width_agrees_with_the_scalar_rows() {
        // Lengths 0 to 129 cover whole 16-sample and 8-sample bodies, and every
        // tail the kernels can be handed.
        let mut noise = Noise(0x2026_1025);
        for length in 0..=129 {
            let red: Vec<u8> = (0..length).map(|_| noise.next() as u8).collect();
            let green: Vec<u8> = (0..length).map(|_| noise.next() as u8).collect();
            let blue: Vec<u8> = (0..length).map(|_| noise.next() as u8).collect();

            let mut want = vec![0u8; length];
            scalar_u8(&red, &green, &blue, &mut want);
            let mut got = vec![0u8; length];
            rows_u8(&red, &green, &blue, &mut got);
            assert_eq!(got, want, "width {length}");

            let wide = length * 2;
            let red: Vec<u8> = (0..wide).map(|_| noise.next() as u8).collect();
            let green: Vec<u8> = (0..wide).map(|_| noise.next() as u8).collect();
            let blue: Vec<u8> = (0..wide).map(|_| noise.next() as u8).collect();
            let mut want = vec![0u16; length];
            scalar_u16(&red, &green, &blue, &mut want);
            let mut got = vec![0u16; length];
            rows_u16(&red, &green, &blue, &mut got);
            assert_eq!(got, want, "width {length} at 16 bits");
        }
    }

    #[test]
    fn every_lane_carries_its_own_sample() {
        // A ramp puts a different value in every lane, so a pack that reorders
        // the halves or the pairs cannot pass. The three planes are offset from
        // each other, so a swapped plane cannot pass either.
        let length = 64;
        let red: Vec<u8> = (0..length).map(|index| index as u8).collect();
        let green: Vec<u8> = (0..length).map(|index| 255 - index as u8).collect();
        let blue: Vec<u8> = (0..length).map(|index| (index * 3) as u8).collect();

        let mut got = vec![0u8; length];
        rows_u8(&red, &green, &blue, &mut got);
        for index in 0..length {
            let want = luma(red[index], green[index], blue[index]);
            assert_eq!(got[index], want, "sample {index}");
        }

        // The same at 16 bits, over samples whose value changes every lane.
        let samples = 24;
        let mut planes = [
            vec![0u8; samples * 2],
            vec![0u8; samples * 2],
            vec![0u8; samples * 2],
        ];
        for index in 0..samples {
            let values = [
                (index * 4091 % 65_536) as u16,
                65_535 - (index * 2500 % 65_536) as u16,
                (index * 37 % 1024 * 64) as u16,
            ];
            for plane in 0..planes.len() {
                let start = index * 2;
                planes[plane][start..start + 2].copy_from_slice(&values[plane].to_ne_bytes());
            }
        }
        let mut got = vec![0u16; samples];
        rows_u16(&planes[0], &planes[1], &planes[2], &mut got);
        for (index, &value) in got.iter().enumerate() {
            let start = index * 2;
            let want = luma_u16(
                u16::from_ne_bytes([planes[0][start], planes[0][start + 1]]),
                u16::from_ne_bytes([planes[1][start], planes[1][start + 1]]),
                u16::from_ne_bytes([planes[2][start], planes[2][start + 1]]),
            );
            assert_eq!(value, want, "sample {index} at 16 bits");
        }
    }

    #[test]
    fn the_shortest_input_limits_the_output() {
        // A mismatched call writes the samples every slice has rather than
        // reading past the shortest one.
        let mut out = [7u8; 4];
        rows_u8(&[1, 2], &[3, 4, 5], &[6, 7, 8, 9], &mut out);
        assert_eq!(out[0], luma(1, 3, 6));
        assert_eq!(out[1], luma(2, 4, 7));
        assert_eq!(&out[2..], &[7, 7], "only two samples were written");
    }
}
