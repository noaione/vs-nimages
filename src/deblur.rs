//! The deblurring kernels behind `Deblur`: a reflected Gaussian blur, the edge
//! mask, the two sharpening candidates, and the masked and clamped blend that
//! limits them.
//!
//! Everything here works in 8-bit code values, which is the reference's `[0, 1]`
//! range times 255. One unit is one 8-bit code value, so the deconvolution's
//! pedestal is `1`, the Sobel normalizer is `255 / 8`, and `threshold` and
//! `overshoot` are the caller's 8-bit step counts unchanged. The filter layer
//! converts a plane's samples to this scale and back.
//!
//! The module is dependency-free and a sized [`Workspace`] allocates nothing
//! during a frame. The blur is the separable direct convolution with the interior
//! split from the borders, so the tap loop reads a bounds-free window and only
//! the border samples pay for the reflection.

/// The sample range the kernels work in: the reference's `[0, 1]` times 255.
const CODE_VALUES: f32 = 255.0;
/// The pedestal the deconvolution adds, in code values (`1 / 255` of `[0, 1]`).
///
/// It keeps the multiplicative update away from zero-locking at pure black.
const PEDESTAL: f32 = 1.0;
/// Floor under the deconvolution's division, in code values.
const DIVISION_FLOOR: f32 = 1e-7;
/// Sigma of the edge detector's prefilter, which never touches the artwork.
const PREFILTER_SIGMA: f32 = 0.5;
/// Sigma the edge mask itself is blurred with.
const MASK_SIGMA: f32 = 0.45;
/// The Sobel response of a full black to white edge, which puts the gradient in
/// 8-bit levels per pixel.
const SOBEL_SPAN: f32 = 8.0;
/// Ramp width of the smoothstep, as a multiple of the threshold.
const RAMP: f32 = 3.0;
/// Floor under the ramp divisor, so `threshold = 0` opens the mask instead of
/// dividing by zero.
const RAMP_FLOOR: f32 = 1e-6;
/// The kernel rule the reference asks scipy for, `truncate=4`.
const TRUNCATE: f32 = 4.0;

/// The largest `radius` a caller may ask for.
pub const MAX_RADIUS: f32 = 16.0;
/// The largest number of refinement passes a caller may ask for.
pub const MAX_ITERATIONS: u32 = 64;
/// The most taps any kernel can have, which is what `MAX_RADIUS` truncates to.
const MAX_KERNEL_TAPS: usize = 2 * ((TRUNCATE * MAX_RADIUS + 0.5) as usize) + 1;

/// How a deblur sharpens.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum Method {
    /// Richardson-Lucy style multiplicative deconvolution, then a scaled delta.
    Deconvolution,
    /// Edge-masked unsharp mask.
    EdgeSharpen,
}

impl Method {
    /// The name the debug settings line reports.
    #[must_use]
    pub const fn name(self) -> &'static str {
        match self {
            Self::Deconvolution => "deconvolution",
            Self::EdgeSharpen => "unsharp",
        }
    }

    /// The `strength` in effect when the caller leaves it out.
    #[must_use]
    pub const fn default_strength(self) -> f32 {
        match self {
            Self::Deconvolution => 0.65,
            Self::EdgeSharpen => 0.85,
        }
    }
}

/// One resolved set of deblurring parameters, in the units the reference uses.
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct Params {
    /// Which candidate is built.
    pub method: Method,
    /// Gaussian sigma in pixels of the assumed blur.
    pub radius: f32,
    /// How much of the candidate is blended back.
    pub strength: f32,
    /// Refinement passes, `Method::Deconvolution` only.
    pub iterations: u32,
    /// Edge threshold in 8-bit levels.
    pub threshold: f32,
    /// Extra excursion past the local extremes, in 8-bit steps.
    pub overshoot: f32,
}

impl Default for Params {
    fn default() -> Self {
        Self {
            method: Method::Deconvolution,
            radius: 0.8,
            strength: Method::Deconvolution.default_strength(),
            iterations: 6,
            threshold: 2.0,
            overshoot: 0.0,
        }
    }
}

impl Params {
    /// Rejects a parameter a kernel cannot act on.
    ///
    /// The bounds keep the work and the kernel length finite. They are wider
    /// than anything the operation is meaningful at.
    pub fn validate(&self) -> Result<(), DeblurError> {
        if !self.radius.is_finite() || self.radius <= 0.0 || self.radius > MAX_RADIUS {
            return Err(DeblurError::Radius);
        }
        if !self.strength.is_finite() {
            return Err(DeblurError::Strength);
        }
        if self.iterations > MAX_ITERATIONS {
            return Err(DeblurError::Iterations);
        }
        if !self.threshold.is_finite() || self.threshold < 0.0 {
            return Err(DeblurError::Threshold);
        }
        if !self.overshoot.is_finite() || self.overshoot < 0.0 {
            return Err(DeblurError::Overshoot);
        }
        Ok(())
    }
}

/// Reason a deblur could not run.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum DeblurError {
    /// `radius` is not a finite sigma in `(0, MAX_RADIUS]`.
    Radius,
    /// `strength` is not finite.
    Strength,
    /// `iterations` is above `MAX_ITERATIONS`.
    Iterations,
    /// `threshold` is not a finite number of 8-bit levels.
    Threshold,
    /// `overshoot` is not a finite number of 8-bit steps.
    Overshoot,
    /// A plane's width and height do not describe a buffer.
    Shape,
    /// The scratch buffers could not be reserved.
    Allocation,
}

impl DeblurError {
    /// A message suitable for the filter boundary.
    #[must_use]
    pub const fn message(self) -> &'static str {
        match self {
            Self::Radius => "radius must be greater than 0 and at most 16, in pixels",
            Self::Strength => "strength must be a finite number",
            Self::Iterations => "iterations must be between 0 and 64",
            Self::Threshold => "threshold must be a finite number of 8-bit levels",
            Self::Overshoot => "overshoot must be a finite number of 8-bit steps",
            Self::Shape => "the frame's width and height do not describe a buffer",
            Self::Allocation => "could not reserve the deblur scratch buffers",
        }
    }
}

/// The shape of one plane the kernels work on.
#[derive(Clone, Copy)]
struct Plane {
    width: usize,
    height: usize,
}

/// The scratch space one frame needs.
///
/// The five planes are the luma the kernels read, the plane the result lands in,
/// and three working planes that the candidate, the mask and the
/// deconvolution's ratio share. A caller keeps one of these per concurrently
/// evaluated frame and reuses it, so a frame allocates nothing.
#[derive(Default)]
pub struct Workspace {
    luma: Vec<f32>,
    first: Vec<f32>,
    second: Vec<f32>,
    third: Vec<f32>,
    temp: Vec<f32>,
    weights: Vec<f32>,
    length: usize,
}

impl Workspace {
    /// An unsized workspace.
    #[must_use]
    pub fn new() -> Self {
        Self::default()
    }

    /// Sizes every buffer for a plane of `width * height` samples.
    ///
    /// Buffers grow and never shrink, so a clip whose pages differ in size keeps
    /// the largest one and every frame slices what it needs.
    pub fn prepare(&mut self, width: usize, height: usize) -> Result<(), DeblurError> {
        let length = width.checked_mul(height).ok_or(DeblurError::Shape)?;
        ensure(&mut self.luma, length)?;
        ensure(&mut self.first, length)?;
        ensure(&mut self.second, length)?;
        ensure(&mut self.third, length)?;
        ensure(&mut self.temp, length)?;
        ensure(&mut self.weights, MAX_KERNEL_TAPS)?;
        self.length = length;
        Ok(())
    }

    /// The luma plane the kernels read, filled by the caller.
    #[must_use]
    pub fn luma(&self) -> &[f32] {
        &self.luma[..self.length]
    }

    /// The luma plane the kernels read, for the caller to fill.
    pub fn luma_mut(&mut self) -> &mut [f32] {
        &mut self.luma[..self.length]
    }

    /// Runs the operation over the luma plane and returns the restored plane.
    ///
    /// The result is in the same 8-bit code values as the luma, clipped to
    /// `[0, 255]` the way the reference clips its blend to `[0, 1]`.
    pub fn restore(
        &mut self,
        width: usize,
        height: usize,
        params: &Params,
    ) -> Result<&[f32], DeblurError> {
        self.prepare(width, height)?;
        if self.length == 0 {
            return Ok(&self.first[..0]);
        }
        params.validate()?;
        let plane = Plane { width, height };

        match params.method {
            Method::Deconvolution => self.deconvolution(plane, params),
            Method::EdgeSharpen => self.unsharp(plane, params),
        }
        self.edge_mask(plane, params.threshold);
        self.blend(plane, params.overshoot);

        Ok(&self.first[..self.length])
    }

    /// The plane the last [`Workspace::restore`] wrote.
    #[must_use]
    pub fn restored(&self) -> &[f32] {
        &self.first[..self.length]
    }

    /// Builds the deconvolution candidate in `first`.
    fn deconvolution(&mut self, plane: Plane, params: &Params) {
        let length = self.length;
        let radius = build_kernel(params.radius, &mut self.weights);
        for (estimate, value) in self.first[..length].iter_mut().zip(&self.luma[..length]) {
            *estimate = value + PEDESTAL;
        }

        for _ in 0..params.iterations {
            // `third` holds the blurred estimate, `second` the ratio it
            // divides into `observed`, and then `third` again the blurred
            // ratio the estimate is scaled by.
            blur(
                &self.first[..length],
                &mut self.third[..length],
                &mut self.temp[..length],
                plane,
                &self.weights,
                radius,
            );
            for (ratio, (blurred, value)) in self.second[..length]
                .iter_mut()
                .zip(self.third[..length].iter().zip(&self.luma[..length]))
            {
                *ratio = (value + PEDESTAL) / blurred.max(DIVISION_FLOOR);
            }
            blur(
                &self.second[..length],
                &mut self.third[..length],
                &mut self.temp[..length],
                plane,
                &self.weights,
                radius,
            );
            for (estimate, factor) in self.first[..length].iter_mut().zip(&self.third[..length]) {
                *estimate *= factor;
            }
        }

        let strength = params.strength;
        for (candidate, value) in self.first[..length].iter_mut().zip(&self.luma[..length]) {
            *candidate = value + strength * (*candidate - PEDESTAL - value);
        }
    }

    /// Builds the unsharp candidate in `first`.
    fn unsharp(&mut self, plane: Plane, params: &Params) {
        let length = self.length;
        let radius = build_kernel(params.radius, &mut self.weights);
        blur(
            &self.luma[..length],
            &mut self.third[..length],
            &mut self.temp[..length],
            plane,
            &self.weights,
            radius,
        );
        for (candidate, (value, blurred)) in self.first[..length]
            .iter_mut()
            .zip(self.luma[..length].iter().zip(&self.third[..length]))
        {
            *candidate = value + params.strength * (value - blurred);
        }
    }

    /// Writes the edge mask of the luma plane into `third`.
    fn edge_mask(&mut self, plane: Plane, threshold: f32) {
        let length = self.length;
        let radius = build_kernel(PREFILTER_SIGMA, &mut self.weights);
        blur(
            &self.luma[..length],
            &mut self.third[..length],
            &mut self.temp[..length],
            plane,
            &self.weights,
            radius,
        );
        sobel_mask(
            &self.third[..length],
            &mut self.second[..length],
            plane,
            threshold,
        );

        let radius = build_kernel(MASK_SIGMA, &mut self.weights);
        blur(
            &self.second[..length],
            &mut self.third[..length],
            &mut self.temp[..length],
            plane,
            &self.weights,
            radius,
        );
    }

    /// Limits the candidate against the local extremes of the luma plane and
    /// blends it back through the mask, in place.
    fn blend(&mut self, plane: Plane, overshoot: f32) {
        let length = self.length;
        for row in 0..plane.height {
            for column in 0..plane.width {
                let index = row * plane.width + column;
                let value = self.luma[index];
                let (low, high) = local_extremes(&self.luma[..length], plane, column, row);
                let limited = clamp(self.first[index], low - overshoot, high + overshoot);
                self.first[index] = clamp(
                    value + self.third[index] * (limited - value),
                    0.0,
                    CODE_VALUES,
                );
            }
        }
    }
}

/// Grows `buffer` to at least `length` samples without ever shrinking it.
fn ensure(buffer: &mut Vec<f32>, length: usize) -> Result<(), DeblurError> {
    if buffer.len() >= length {
        return Ok(());
    }
    buffer
        .try_reserve_exact(length - buffer.len())
        .map_err(|_| DeblurError::Allocation)?;
    buffer.resize(length, 0.0);
    Ok(())
}

/// Builds the normalized Gaussian weights for `sigma` and returns the radius.
///
/// The radius reproduces scipy's `lw = int(truncate * sigma + 0.5)`, so a small
/// sigma gives a radius of zero and the blur is the identity, exactly as the
/// reference's `gaussian_filter` is.
fn build_kernel(sigma: f32, weights: &mut Vec<f32>) -> usize {
    let radius = (TRUNCATE * sigma + 0.5) as usize;
    let length = 2 * radius + 1;
    weights.clear();
    weights.resize(length, 0.0);

    let scale = -0.5 / f64::from(sigma) / f64::from(sigma);
    let mut total = 0.0f64;
    for (index, weight) in weights.iter_mut().enumerate() {
        let offset = index as f64 - radius as f64;
        let value = (scale * offset * offset).exp();
        *weight = value as f32;
        total += value;
    }
    if total > 0.0 {
        for weight in weights.iter_mut() {
            *weight = (f64::from(*weight) / total) as f32;
        }
    }
    radius
}

/// Blurs `source` into `target` with a separable Gaussian, using `temp` for the
/// horizontal pass.
///
/// A radius of zero copies, which is what a radius-zero kernel does. Both passes
/// reflect at the border the way scipy's `mode="reflect"` does: the edge sample
/// is repeated, so index `-1` reads sample `0`.
fn blur(
    source: &[f32],
    target: &mut [f32],
    temp: &mut [f32],
    plane: Plane,
    weights: &[f32],
    radius: usize,
) {
    if radius == 0 {
        target.copy_from_slice(source);
        return;
    }
    let width = plane.width;

    for row in 0..plane.height {
        let start = row * width;
        let (Some(source_row), Some(temp_row)) = (
            source.get(start..start + width),
            temp.get_mut(start..start + width),
        ) else {
            return;
        };
        blur_row(source_row, temp_row, weights, radius);
    }

    for row in 0..plane.height {
        let start = row * width;
        let Some(target_row) = target.get_mut(start..start + width) else {
            return;
        };
        target_row.fill(0.0);
        let interior = row >= radius && row + radius < plane.height;
        for (index, weight) in weights.iter().enumerate() {
            let tap = if interior {
                row + index - radius
            } else {
                reflect(
                    row as isize + index as isize - radius as isize,
                    plane.height,
                )
            };
            let bump = tap * width;
            let Some(tap_row) = temp.get(bump..bump + width) else {
                return;
            };
            for (value, tap) in target_row.iter_mut().zip(tap_row) {
                *value += tap * weight;
            }
        }
    }
}

/// One horizontal pass: the interior reads a bounds-free window and only the
/// first and last `radius` samples reflect.
fn blur_row(source: &[f32], target: &mut [f32], weights: &[f32], radius: usize) {
    let width = source.len();
    let interior_start = radius.min(width);
    let interior_end = width.saturating_sub(radius);

    for (index, value) in target.iter_mut().enumerate().take(interior_start) {
        *value = reflected_tap(source, index as isize, weights, radius);
    }
    for index in interior_start..interior_end {
        let Some(window) = source.get(index - radius..index + radius + 1) else {
            continue;
        };
        let sum = window
            .iter()
            .zip(weights)
            .map(|(value, weight)| value * weight)
            .sum();
        if let Some(target) = target.get_mut(index) {
            *target = sum;
        }
    }
    for (index, value) in target
        .iter_mut()
        .enumerate()
        .skip(interior_end.max(interior_start))
    {
        *value = reflected_tap(source, index as isize, weights, radius);
    }
}

/// Sums one reflected window around `index`.
fn reflected_tap(source: &[f32], index: isize, weights: &[f32], radius: usize) -> f32 {
    let mut sum = 0.0f32;
    for (offset, weight) in weights.iter().enumerate() {
        let tap = reflect(index + offset as isize - radius as isize, source.len());
        sum += source.get(tap).copied().unwrap_or(0.0) * weight;
    }
    sum
}

/// Folds an out-of-range index back into `0..length` by repeating the edge
/// sample, which is scipy's `mode="reflect"`.
fn reflect(index: isize, length: usize) -> usize {
    if length == 0 {
        return 0;
    }
    let period = 2 * length as isize;
    let folded = index.rem_euclid(period);
    if folded < length as isize {
        folded as usize
    } else {
        (period - 1 - folded) as usize
    }
}

/// Writes the soft edge mask of `base` into `mask`.
///
/// The Sobel kernels are the unnormalized ones scipy uses, so a full black to
/// white step responds with `4`; dividing by `SOBEL_SPAN` puts the magnitude in
/// 8-bit levels per pixel, which is the unit `threshold` is in.
fn sobel_mask(base: &[f32], mask: &mut [f32], plane: Plane, threshold: f32) {
    let ramp = (RAMP * threshold).max(RAMP_FLOOR);
    for row in 0..plane.height {
        for column in 0..plane.width {
            let index = row * plane.width + column;
            let (vertical, horizontal) = sobel(base, plane, column, row);
            let magnitude = vertical.hypot(horizontal) / SOBEL_SPAN;
            let progress = clamp((magnitude - threshold) / ramp, 0.0, 1.0);
            if let Some(slot) = mask.get_mut(index) {
                *slot = progress * progress * (3.0 - 2.0 * progress);
            }
        }
    }
}

/// The two Sobel responses at one sample, with the edge sample repeated.
#[inline]
fn sobel(base: &[f32], plane: Plane, column: usize, row: usize) -> (f32, f32) {
    let left = column.saturating_sub(1);
    let right = (column + 1).min(plane.width.saturating_sub(1));
    let top = row.saturating_sub(1);
    let bottom = (row + 1).min(plane.height.saturating_sub(1));

    let at = |column: usize, row: usize| -> f32 {
        base.get(row * plane.width + column).copied().unwrap_or(0.0)
    };
    let (top_left, top_middle, top_right) = (at(left, top), at(column, top), at(right, top));
    let (middle_left, middle_right) = (at(left, row), at(right, row));
    let (bottom_left, bottom_middle, bottom_right) =
        (at(left, bottom), at(column, bottom), at(right, bottom));

    let vertical = (bottom_left + 2.0 * bottom_middle + bottom_right)
        - (top_left + 2.0 * top_middle + top_right);
    let horizontal = (top_right + 2.0 * middle_right + bottom_right)
        - (top_left + 2.0 * middle_left + bottom_left);
    (vertical, horizontal)
}

/// The 3x3 minimum and maximum of `source` around one sample, with the edge
/// sample repeated.
#[inline]
fn local_extremes(source: &[f32], plane: Plane, column: usize, row: usize) -> (f32, f32) {
    let left = column.saturating_sub(1);
    let right = (column + 1).min(plane.width.saturating_sub(1));
    let top = row.saturating_sub(1);
    let bottom = (row + 1).min(plane.height.saturating_sub(1));

    let mut low = f32::INFINITY;
    let mut high = f32::NEG_INFINITY;
    for row in top..=bottom {
        for column in left..=right {
            let Some(value) = source.get(row * plane.width + column).copied() else {
                continue;
            };
            low = low.min(value);
            high = high.max(value);
        }
    }
    (low, high)
}

/// Clamps `value` into `[low, high]`, mapping a `NaN` to `low`.
///
/// `f32::clamp` panics when its bounds are not ordered, and this runs on the
/// frame path where a panic aborts the host, so the comparison is explicit.
#[inline]
fn clamp(value: f32, low: f32, high: f32) -> f32 {
    if high < low {
        return low;
    }
    if value < low {
        low
    } else if value > high {
        high
    } else {
        value
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    /// The scipy reference for a 0.8 Gaussian over the ramp `0..=4` with
    /// `mode="reflect"` and `truncate=4`.
    const RAMP: [f32; 5] = [0.0, 1.0, 2.0, 3.0, 4.0];
    const RAMP_BLURRED: [f32; 5] = [0.296_245_4, 1.023_232_5, 2.0, 2.976_767_5, 3.703_754_6];

    fn blurred(values: &[f32], width: usize, height: usize, sigma: f32) -> Vec<f32> {
        let mut weights = Vec::new();
        let radius = build_kernel(sigma, &mut weights);
        let mut target = vec![0.0; values.len()];
        let mut temp = vec![0.0; values.len()];
        blur(
            values,
            &mut target,
            &mut temp,
            Plane { width, height },
            &weights,
            radius,
        );
        target
    }

    #[test]
    fn the_blur_matches_scipy_along_a_row() {
        let target = blurred(&RAMP, 5, 1, 0.8);
        for (got, want) in target.iter().zip(RAMP_BLURRED) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn the_blur_matches_scipy_along_a_column() {
        let target = blurred(&RAMP, 1, 5, 0.8);
        for (got, want) in target.iter().zip(RAMP_BLURRED) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    #[test]
    fn the_kernel_is_normalized_and_symmetric() {
        let mut weights = Vec::new();
        let radius = build_kernel(0.8, &mut weights);
        assert_eq!(radius, 3);
        assert_eq!(weights.len(), 7);
        let total: f32 = weights.iter().sum();
        assert!((total - 1.0).abs() < 1e-6, "sum is {total}");
        for index in 0..weights.len() / 2 {
            let mirrored = weights.len() - 1 - index;
            assert!((weights[index] - weights[mirrored]).abs() < 1e-9);
        }
    }

    #[test]
    fn a_radius_of_zero_copies() {
        let mut weights = Vec::new();
        let radius = build_kernel(0.1, &mut weights);
        assert_eq!(radius, 0);
        let target = blurred(&RAMP, 5, 1, 0.1);
        assert_eq!(target, RAMP);
    }

    #[test]
    fn the_reflection_repeats_the_edge_sample() {
        assert_eq!(reflect(-1, 5), 0);
        assert_eq!(reflect(-2, 5), 1);
        assert_eq!(reflect(5, 5), 4);
        assert_eq!(reflect(6, 5), 3);
        assert_eq!(reflect(0, 5), 0);
        assert_eq!(reflect(4, 5), 4);
        assert_eq!(reflect(-1, 0), 0);
    }

    #[test]
    fn the_deconvolution_sharpens_a_soft_edge() {
        let hard: Vec<f32> = (0..32)
            .map(|index| if index < 16 { 64.0 } else { 192.0 })
            .collect();
        let softened = {
            let mut weights = Vec::new();
            let radius = build_kernel(1.0, &mut weights);
            let mut out = vec![0.0; hard.len()];
            let mut temp = vec![0.0; hard.len()];
            blur(
                &hard,
                &mut out,
                &mut temp,
                Plane {
                    width: 32,
                    height: 1,
                },
                &weights,
                radius,
            );
            out
        };

        let mut workspace = Workspace::new();
        workspace.prepare(32, 1).expect("sized");
        workspace.luma_mut().copy_from_slice(&softened);
        let restored = workspace
            .restore(32, 1, &Params::default())
            .expect("restored")
            .to_vec();

        // The deconvolution is there to undo the blur that softened the step,
        // so the steepest transition in the plane has to grow.
        let slope = |values: &[f32]| {
            values
                .windows(2)
                .map(|pair| (pair[1] - pair[0]).abs())
                .fold(0.0f32, f32::max)
        };
        assert!(
            slope(&restored) > slope(&softened),
            "{} is not steeper than {}",
            slope(&restored),
            slope(&softened)
        );
        // The flat sides sit outside the mask and keep their value.
        assert!((restored[0] - softened[0]).abs() < 1e-3);
        assert!((restored[31] - softened[31]).abs() < 1e-3);
    }

    #[test]
    fn a_black_to_white_frame_stays_finite() {
        let mut workspace = Workspace::new();
        workspace.prepare(8, 8).expect("sized");
        for (index, value) in workspace.luma_mut().iter_mut().enumerate() {
            *value = if index < 32 { 0.0 } else { 255.0 };
        }
        let restored = workspace
            .restore(8, 8, &Params::default())
            .expect("restored");
        for value in restored {
            assert!(value.is_finite());
            assert!((0.0..=255.0).contains(value));
        }
    }

    #[test]
    fn a_flat_plane_comes_back_unchanged() {
        for method in [Method::Deconvolution, Method::EdgeSharpen] {
            let mut workspace = Workspace::new();
            workspace.prepare(8, 8).expect("sized");
            workspace.luma_mut().fill(128.0);
            let params = Params {
                method,
                ..Params::default()
            };
            let restored = workspace.restore(8, 8, &params).expect("restored");
            for value in restored {
                assert_eq!(*value, 128.0, "{} moved a flat plane", method.name());
            }
        }
    }

    #[test]
    fn a_black_plane_comes_back_unchanged() {
        let mut workspace = Workspace::new();
        workspace.prepare(8, 8).expect("sized");
        workspace.luma_mut().fill(0.0);
        let restored = workspace
            .restore(8, 8, &Params::default())
            .expect("restored")
            .to_vec();
        for value in restored {
            assert_eq!(value, 0.0);
        }
    }

    #[test]
    fn the_mask_opens_on_an_edge_and_stays_shut_on_a_flat_plane() {
        let mut workspace = Workspace::new();
        workspace.prepare(16, 1).expect("sized");
        {
            let luma = workspace.luma_mut();
            for (index, value) in luma.iter_mut().enumerate() {
                *value = if index < 8 { 0.0 } else { 255.0 };
            }
        }
        workspace.edge_mask(
            Plane {
                width: 16,
                height: 1,
            },
            2.0,
        );
        let mask = &workspace.third[..16];
        assert!(mask[0] < 1e-3, "the flat side is open: {}", mask[0]);
        let strongest = mask.iter().copied().fold(0.0f32, f32::max);
        assert!(strongest > 0.9, "the edge did not open: {strongest}");

        let flat = vec![128.0f32; 16];
        let mut open = vec![0.0f32; 16];
        sobel_mask(
            &flat,
            &mut open,
            Plane {
                width: 16,
                height: 1,
            },
            2.0,
        );
        assert!(open.iter().all(|value| *value == 0.0));
    }

    #[test]
    fn a_zero_threshold_opens_the_mask_fully() {
        let mut workspace = Workspace::new();
        workspace.prepare(16, 1).expect("sized");
        {
            let luma = workspace.luma_mut();
            for (index, value) in luma.iter_mut().enumerate() {
                *value = if index < 8 { 0.0 } else { 255.0 };
            }
        }
        workspace.edge_mask(
            Plane {
                width: 16,
                height: 1,
            },
            0.0,
        );
        let mask = &workspace.third[..16];
        assert!(mask.iter().any(|value| *value > 0.9));
        let ramp: Vec<f32> = (0..16).map(|index| index as f32 * 15.0).collect();
        let mut open = vec![0.0f32; 16];
        sobel_mask(
            &ramp,
            &mut open,
            Plane {
                width: 16,
                height: 1,
            },
            0.0,
        );
        assert!(open.iter().all(|value| *value > 0.9));
    }

    #[test]
    fn the_extremes_are_the_3x3_window_with_a_repeated_edge() {
        let plane: Vec<f32> = (0..9).map(|value| value as f32).collect();
        let shape = Plane {
            width: 3,
            height: 3,
        };
        assert_eq!(local_extremes(&plane, shape, 0, 0), (0.0, 4.0));
        assert_eq!(local_extremes(&plane, shape, 2, 2), (4.0, 8.0));
        assert_eq!(local_extremes(&plane, shape, 1, 1), (0.0, 8.0));
    }

    #[test]
    fn the_parameters_are_bounded() {
        assert!(Params::default().validate().is_ok());
        for broken in [
            Params {
                radius: 0.0,
                ..Params::default()
            },
            Params {
                radius: MAX_RADIUS + 1.0,
                ..Params::default()
            },
        ] {
            assert_eq!(broken.validate(), Err(DeblurError::Radius));
        }
        assert_eq!(
            Params {
                strength: f32::NAN,
                ..Params::default()
            }
            .validate(),
            Err(DeblurError::Strength)
        );
        assert_eq!(
            Params {
                iterations: MAX_ITERATIONS + 1,
                ..Params::default()
            }
            .validate(),
            Err(DeblurError::Iterations)
        );
        assert_eq!(
            Params {
                threshold: -1.0,
                ..Params::default()
            }
            .validate(),
            Err(DeblurError::Threshold)
        );
        assert_eq!(
            Params {
                overshoot: f32::INFINITY,
                ..Params::default()
            }
            .validate(),
            Err(DeblurError::Overshoot)
        );
    }
}
