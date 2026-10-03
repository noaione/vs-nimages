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
/// The tap count at which the register accumulator stops paying.
///
/// Measured on a 2903x4128 plane: the vertical pass is 1.29x to 1.41x faster
/// with one accumulator per eight column block at five to seventeen taps, 1.04x
/// at sixty-five, and 0.66x at the 129 taps a radius of sixteen gives. The
/// deblur's own sigmas are 0.45, 0.5 and 0.8, so five and seven taps.
#[cfg(target_arch = "x86_64")]
const REGISTER_TAPS: usize = 32;

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
        // The horizontal scratch is a ring of filtered rows, not a whole plane:
        // the largest kernel is `MAX_KERNEL_TAPS` rows and a shorter plane holds
        // all of its own rows. A larger window is never active for one gaussian.
        let ring = width
            .checked_mul(height.min(MAX_KERNEL_TAPS))
            .ok_or(DeblurError::Shape)?;
        ensure(&mut self.temp, ring)?;
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
                &mut self.temp,
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
                &mut self.temp,
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
            &mut self.temp,
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
            &mut self.temp,
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
            &mut self.temp,
            plane,
            &self.weights,
            radius,
        );
    }

    /// Limits the candidate against the local extremes of the luma plane and
    /// blends it back through the mask, in place.
    ///
    /// The window walk is the hot loop. Three whole rows are sliced out once per
    /// output row so the taps are plain indexing into slices whose length is the
    /// plane's width, and the two end columns are the only ones that repeat a
    /// sample. `min` and `max` are exact, so chaining them gives the same window
    /// as walking it.
    fn blend(&mut self, plane: Plane, overshoot: f32) {
        let width = plane.width;
        let height = plane.height;
        if width == 0 || height == 0 {
            return;
        }
        let length = self.length;

        #[cfg(target_arch = "x86_64")]
        if width > 2 && is_x86_feature_detected!("avx2") {
            // SAFETY: `avx2` was just detected, and the three slices are all
            // `length` long, which is the plane the vector loop walks.
            unsafe {
                blend_avx2(
                    &self.luma[..length],
                    &self.third[..length],
                    &mut self.first[..length],
                    plane,
                    overshoot,
                );
            }
            return;
        }

        let luma = &self.luma[..length];
        let mask = &self.third[..length];

        for row in 0..height {
            let start = row * width;
            let above = row.saturating_sub(1) * width;
            let below = (row + 1).min(height - 1) * width;
            let (Some(above), Some(current), Some(below), Some(mask_row)) = (
                luma.get(above..above + width),
                luma.get(start..start + width),
                luma.get(below..below + width),
                mask.get(start..start + width),
            ) else {
                return;
            };
            let Some(target) = self.first.get_mut(start..start + width) else {
                return;
            };

            for column in 0..width {
                blend_sample(above, current, below, mask_row, target, column, overshoot);
            }
        }
    }
}

/// Grows `buffer` to at least `length` samples without ever shrinking it.
/// One sample of the scalar blend.
///
/// The two end columns and the tail of a row go through this on both paths, so
/// the vector path cannot drift from the scalar one at the edges.
#[inline]
fn blend_sample(
    above: &[f32],
    current: &[f32],
    below: &[f32],
    mask_row: &[f32],
    target: &mut [f32],
    column: usize,
    overshoot: f32,
) {
    let left = column.saturating_sub(1);
    let right = (column + 1).min(current.len().saturating_sub(1));
    let low = above[left]
        .min(above[column])
        .min(above[right])
        .min(current[left])
        .min(current[column])
        .min(current[right])
        .min(below[left])
        .min(below[column])
        .min(below[right]);
    let high = above[left]
        .max(above[column])
        .max(above[right])
        .max(current[left])
        .max(current[column])
        .max(current[right])
        .max(below[left])
        .max(below[column])
        .max(below[right]);
    let value = current[column];
    let limited = clamp(target[column], low - overshoot, high + overshoot);
    target[column] = clamp(
        value + mask_row[column] * (limited - value),
        0.0,
        CODE_VALUES,
    );
}

/// The blend, eight columns at a time.
///
/// Every operation is the one `blend_sample` applies to the same column in the
/// same order: the same nine taps into the same `min`/`max` chain, the same
/// clamp, the same mix. Nothing is reassociated and no `fmadd` contracts a
/// multiply into an add, so the two paths produce the same bytes.
///
/// The two end columns repeat a sample and the tail of a row is shorter than
/// eight, so both go through `blend_sample`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_op_in_unsafe_fn,
    reason = "the body is one unsafe operation, and it is entered only after the feature is detected"
)]
unsafe fn blend_avx2(luma: &[f32], mask: &[f32], first: &mut [f32], plane: Plane, overshoot: f32) {
    use std::arch::x86_64::*;

    let width = plane.width;
    let height = plane.height;
    let overshoot8 = _mm256_set1_ps(overshoot);
    let ceiling = _mm256_set1_ps(CODE_VALUES);
    let zero = _mm256_setzero_ps();

    for row in 0..height {
        let start = row * width;
        let above_start = row.saturating_sub(1) * width;
        let below_start = (row + 1).min(height - 1) * width;
        let (Some(above), Some(current), Some(below), Some(mask_row), Some(target)) = (
            luma.get(above_start..above_start + width),
            luma.get(start..start + width),
            luma.get(below_start..below_start + width),
            mask.get(start..start + width),
            first.get_mut(start..start + width),
        ) else {
            return;
        };

        blend_sample(above, current, below, mask_row, target, 0, overshoot);

        // SAFETY: the row slices are exactly `width` long, the loop keeps
        // widest load and every store inside the row.
        let mut column = 1;
        while column + 8 < width {
            let up_left = _mm256_loadu_ps(above.as_ptr().add(column - 1));
            let up_mid = _mm256_loadu_ps(above.as_ptr().add(column));
            let up_right = _mm256_loadu_ps(above.as_ptr().add(column + 1));
            let mid_left = _mm256_loadu_ps(current.as_ptr().add(column - 1));
            let mid_mid = _mm256_loadu_ps(current.as_ptr().add(column));
            let mid_right = _mm256_loadu_ps(current.as_ptr().add(column + 1));
            let low_left = _mm256_loadu_ps(below.as_ptr().add(column - 1));
            let low_mid = _mm256_loadu_ps(below.as_ptr().add(column));
            let low_right = _mm256_loadu_ps(below.as_ptr().add(column + 1));

            let mut low = _mm256_min_ps(up_left, up_mid);
            low = _mm256_min_ps(low, up_right);
            low = _mm256_min_ps(low, mid_left);
            low = _mm256_min_ps(low, mid_mid);
            low = _mm256_min_ps(low, mid_right);
            low = _mm256_min_ps(low, low_left);
            low = _mm256_min_ps(low, low_mid);
            low = _mm256_min_ps(low, low_right);
            let mut high = _mm256_max_ps(up_left, up_mid);
            high = _mm256_max_ps(high, up_right);
            high = _mm256_max_ps(high, mid_left);
            high = _mm256_max_ps(high, mid_mid);
            high = _mm256_max_ps(high, mid_right);
            high = _mm256_max_ps(high, low_left);
            high = _mm256_max_ps(high, low_mid);
            high = _mm256_max_ps(high, low_right);

            let value = _mm256_loadu_ps(current.as_ptr().add(column));
            let candidate = _mm256_loadu_ps(target.as_ptr().add(column));
            let limited = clamp8(
                candidate,
                _mm256_sub_ps(low, overshoot8),
                _mm256_add_ps(high, overshoot8),
            );
            let blended = _mm256_add_ps(
                value,
                _mm256_mul_ps(
                    _mm256_loadu_ps(mask_row.as_ptr().add(column)),
                    _mm256_sub_ps(limited, value),
                ),
            );
            let bounded = clamp8(blended, zero, ceiling);
            _mm256_storeu_ps(target.as_mut_ptr().add(column), bounded);

            column += 8;
        }
        while column < width - 1 {
            blend_sample(above, current, below, mask_row, target, column, overshoot);
            column += 1;
        }
        blend_sample(
            above,
            current,
            below,
            mask_row,
            target,
            width - 1,
            overshoot,
        );
    }
}

/// `clamp` in eight lanes.
///
/// `min` then `max` is the scalar comparison chain without the branches, but it
/// snaps a `NaN` to a bound where the scalar form lets it through, so a `NaN`
/// value is blended back over the result.
#[cfg(target_arch = "x86_64")]
#[inline]
#[target_feature(enable = "avx2")]
unsafe fn clamp8(
    value: std::arch::x86_64::__m256,
    low: std::arch::x86_64::__m256,
    high: std::arch::x86_64::__m256,
) -> std::arch::x86_64::__m256 {
    use std::arch::x86_64::*;

    let bounded = _mm256_max_ps(_mm256_min_ps(value, high), low);
    let unordered = _mm256_cmp_ps(value, value, _CMP_UNORD_Q);
    _mm256_blendv_ps(bounded, value, unordered)
}
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
/// A radius of zero copies, which is what a radius-zero kernel does. The
/// horizontal pass reflects at the border the way scipy's `mode="reflect"` does:
/// the edge sample is repeated, so index `-1` reads sample `0`.
///
/// `temp` is a ring of `min(height, 2 * radius + 1)` filtered rows rather than a
/// whole plane. One output row consumes exactly that window, and
/// `source_row % ring_rows` sends the window's distinct rows to distinct slots:
/// the window is a run of at most `ring_rows` consecutive source rows, so no two
/// of them can share a slot. Each slot remembers the source row it holds, which
/// is what makes moving the window safe. A row is filtered once as the window
/// advances and refiltered only where the reflection asks for it again.
///
/// The horizontal fold, the vertical tap order and the f32 stores are the ones a
/// whole-plane `temp` produced, so the output is unchanged.
///
/// The x86-64 feature check runs once here, not once per row or tap.
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
    let height = plane.height;
    let taps = weights.len().min(MAX_KERNEL_TAPS);
    if width == 0 || height == 0 || taps == 0 {
        return;
    }
    let ring_rows = ring_rows(radius, height);
    if temp.len() < ring_rows * width {
        return;
    }

    #[cfg(target_arch = "x86_64")]
    let vector = is_x86_feature_detected!("avx2");
    #[cfg(not(target_arch = "x86_64"))]
    let vector = false;

    // The source row each slot holds now, or `usize::MAX` while it holds
    // nothing. Local to one gaussian, so no frame state outlives a call.
    let mut held = [usize::MAX; MAX_KERNEL_TAPS];
    let mut ring_slots = [0usize; MAX_KERNEL_TAPS];
    let mut rows: [*const f32; MAX_KERNEL_TAPS] = [std::ptr::null(); MAX_KERNEL_TAPS];

    for row in 0..height {
        let interior = row >= radius && row + radius < height;
        for (index, slot) in ring_slots.iter_mut().enumerate().take(taps) {
            let tap = if interior {
                row + index - radius
            } else {
                reflect(row as isize + index as isize - radius as isize, height)
            };
            let ring_slot = tap % ring_rows;
            *slot = ring_slot;
            if held[ring_slot] != tap {
                let Some(source_row) = source.get(tap * width..tap * width + width) else {
                    return;
                };
                let Some(ring_row) = temp.get_mut(ring_slot * width..ring_slot * width + width)
                else {
                    return;
                };
                filter_row(source_row, ring_row, weights, radius, vector);
                held[ring_slot] = tap;
            }
        }

        // Resolve pointers after every ring write. A mutable borrow of `temp`
        // invalidates pointers from earlier shared borrows even when the row
        // ranges are disjoint. No ring write occurs before these are consumed.
        for (slot, &ring_slot) in rows.iter_mut().zip(&ring_slots).take(taps) {
            match temp.get(ring_slot * width..ring_slot * width + width) {
                Some(ring_row) => *slot = ring_row.as_ptr(),
                None => return,
            }
        }

        let Some(target_row) = target.get_mut(row * width..row * width + width) else {
            return;
        };
        // SAFETY: every entry of `rows[..taps]` addresses a `width` sample row
        // inside `temp`, which is a distinct allocation from `target`.
        unsafe { accumulate_row(&rows, taps, target_row, weights, vector) };
    }
}

/// How many filtered rows the ring holds for one gaussian.
fn ring_rows(radius: usize, height: usize) -> usize {
    (2 * radius + 1).min(height)
}

/// Filters one source row into one ring row.
///
/// The caller hoists the x86-64 feature check, so this branches on a bool
/// rather than detecting the feature once per row.
#[inline]
fn filter_row(source: &[f32], target: &mut [f32], weights: &[f32], radius: usize, vector: bool) {
    #[cfg(target_arch = "x86_64")]
    if vector {
        // SAFETY: the caller detected `avx2` for this plane, and both rows are
        // `target.len()` samples.
        unsafe { blur_row_avx2(source, target, weights, radius) };
        return;
    }
    let _ = vector;
    blur_row(source, target, weights, radius);
}

/// The vertical pass for one output row, one accumulator per eight columns.
///
/// Every product and every sum is the instruction the per-tap loop uses, in the
/// same tap order and from the same zero start, so the result is the same bytes;
/// what changes is that the partial sum stays in a register across the taps and
/// the block is written once.
///
/// # Safety
///
/// Every entry of `rows[..taps]` must address at least `target_row.len()`
/// readable `f32` samples, and none of them may overlap `target_row`.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn vertical_row_registers(
    rows: &[*const f32; MAX_KERNEL_TAPS],
    taps: usize,
    target_row: &mut [f32],
    weights: &[f32],
) {
    use std::arch::x86_64::*;

    let width = target_row.len();
    let target_base = target_row.as_mut_ptr();

    // SAFETY: the caller resolved every row to `width` readable samples.
    unsafe {
        let mut column = 0;
        while column + 8 <= width {
            let mut sum = _mm256_setzero_ps();
            for index in 0..taps {
                let tap = _mm256_loadu_ps(rows.get_unchecked(index).add(column));
                let weight = _mm256_set1_ps(*weights.get_unchecked(index));
                sum = _mm256_add_ps(sum, _mm256_mul_ps(tap, weight));
            }
            _mm256_storeu_ps(target_base.add(column), sum);
            column += 8;
        }
        while column < width {
            let mut sum = 0.0f32;
            for index in 0..taps {
                sum += *rows.get_unchecked(index).add(column) * weights.get_unchecked(index);
            }
            *target_base.add(column) = sum;
            column += 1;
        }
    }
}

/// The vertical pass for one output row, one tap at a time.
///
/// This covers a kernel longer than `REGISTER_TAPS` and a target without AVX2.
/// AVX2 still handles each tap eight lanes wide where it is available.
///
/// # Safety
///
/// Every entry of `rows[..taps]` must address at least `target_row.len()`
/// readable `f32` samples, and none of them may overlap `target_row`.
unsafe fn vertical_row_per_tap(
    rows: &[*const f32; MAX_KERNEL_TAPS],
    taps: usize,
    target_row: &mut [f32],
    weights: &[f32],
    vector: bool,
) {
    let _ = vector;
    target_row.fill(0.0);
    for index in 0..taps {
        let Some(weight) = weights.get(index).copied() else {
            return;
        };
        // SAFETY: the caller resolved every row to `target_row.len()` samples.
        let tap_row =
            unsafe { std::slice::from_raw_parts(*rows.get_unchecked(index), target_row.len()) };
        #[cfg(target_arch = "x86_64")]
        if vector {
            // SAFETY: both rows are `target_row.len()` long.
            unsafe { accumulate_avx2(target_row, tap_row, weight) };
            continue;
        }
        for (value, tap) in target_row.iter_mut().zip(tap_row) {
            *value += tap * weight;
        }
    }
}

/// Runs the vertical pass for one output row from resolved source rows.
///
/// # Safety
///
/// Every entry of `rows[..taps]` must address at least `target_row.len()`
/// readable `f32` samples, and none of them may overlap `target_row`.
#[inline]
unsafe fn accumulate_row(
    rows: &[*const f32; MAX_KERNEL_TAPS],
    taps: usize,
    target_row: &mut [f32],
    weights: &[f32],
    vector: bool,
) {
    #[cfg(target_arch = "x86_64")]
    if vector && taps <= REGISTER_TAPS {
        // SAFETY: the caller resolved every row.
        unsafe { vertical_row_registers(rows, taps, target_row, weights) };
        return;
    }
    // SAFETY: the caller resolved every row.
    unsafe { vertical_row_per_tap(rows, taps, target_row, weights, vector) };
}

/// `target[i] += tap[i] * weight`, eight lanes at a time.
///
/// The accumulator is the one the scalar loop keeps and the product and the sum
/// are separate instructions, so no `fmadd` contracts them and the bytes do not
/// move. This is the vertical pass, which is elementwise down a row; the
/// horizontal pass is a dot product of runtime length and has to stay scalar.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_op_in_unsafe_fn,
    reason = "the body is one unsafe operation, and it is entered only after the feature is detected"
)]
unsafe fn accumulate_avx2(target: &mut [f32], tap: &[f32], weight: f32) {
    use std::arch::x86_64::*;

    let weight8 = _mm256_set1_ps(weight);
    let mut column = 0;
    while column + 8 <= target.len() {
        let sum = _mm256_add_ps(
            _mm256_loadu_ps(target.as_ptr().add(column)),
            _mm256_mul_ps(_mm256_loadu_ps(tap.as_ptr().add(column)), weight8),
        );
        _mm256_storeu_ps(target.as_mut_ptr().add(column), sum);
        column += 8;
    }
    for (value, tap) in target[column..].iter_mut().zip(&tap[column..]) {
        *value += tap * weight;
    }
}

/// One horizontal pass: the interior reads a bounds-free window and only the
/// first and last `radius` samples reflect.
///
/// Four samples at a time were tried here, to put four independent sum chains in
/// flight instead of one left fold's single chain. It measured inside the run to
/// run noise, so the simpler loop is what stayed. A vector attempt over the same
/// window regressed 2.9x inside the plugin while measuring 3x faster on its own;
/// that attempt put its feature check inside this per-row function, and
/// [`blur_row_avx2`] is dispatched from [`blur`] instead.
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

/// One horizontal pass, eight columns at a time.
///
/// Each lane starts at positive zero and folds its own taps in the same order
/// as [`blur_row`], with the multiply and the add as separate instructions, so
/// the two paths produce the same bytes. The window is `radius` wide on each
/// side and every load stays inside the row. The first and last `radius`
/// samples still reflect one at a time, and a tail shorter than eight goes
/// through the same scalar code.
///
/// [`blur`] detects the feature once per gaussian and calls this, rather than
/// checking inside the per-row loop the way the reverted attempt did.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
unsafe fn blur_row_avx2(source: &[f32], target: &mut [f32], weights: &[f32], radius: usize) {
    use std::arch::x86_64::*;

    let width = source.len();
    let interior_start = radius.min(width);
    let interior_end = width.saturating_sub(radius);

    for (index, value) in target.iter_mut().enumerate().take(interior_start) {
        *value = reflected_tap(source, index as isize, weights, radius);
    }

    // SAFETY: `avx2` was detected by the caller. For an `index` below
    // `width - radius`, the widest read is `source[index + radius]` and the
    // store is eight lanes wide, so both stay inside the row.
    unsafe {
        let mut index = interior_start;
        while index < interior_end {
            if index + 8 <= interior_end {
                let mut sum = _mm256_setzero_ps();
                for (offset, weight) in weights.iter().enumerate() {
                    let window = source.as_ptr().add(index + offset - radius);
                    let row = _mm256_loadu_ps(window);
                    let weight8 = _mm256_set1_ps(*weight);
                    sum = _mm256_add_ps(sum, _mm256_mul_ps(row, weight8));
                }
                _mm256_storeu_ps(target.as_mut_ptr().add(index), sum);
                index += 8;
                continue;
            }
            let end = interior_end.min(index + 8);
            for one in index..end {
                let window = source.as_ptr().add(one - radius);
                let mut sum = 0.0f32;
                for (offset, weight) in weights.iter().enumerate() {
                    sum += *window.add(offset) * weight;
                }
                *target.as_mut_ptr().add(one) = sum;
            }
            index = end;
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
///
/// Three rows are sliced out once per output row, so the nine taps are direct
/// reads instead of an index computation each.
fn sobel_mask(base: &[f32], mask: &mut [f32], plane: Plane, threshold: f32) {
    let width = plane.width;
    let height = plane.height;
    if width == 0 || height == 0 {
        return;
    }
    let ramp = (RAMP * threshold).max(RAMP_FLOOR);

    #[cfg(target_arch = "x86_64")]
    if is_x86_feature_detected!("avx2") && width > 2 {
        // SAFETY: `avx2` was just detected, and the three rows are `width` long,
        // so every eight lane load and store stays inside them.
        unsafe { sobel_mask_avx2(base, mask, plane, threshold, ramp) };
        return;
    }

    for row in 0..height {
        let start = row * width;
        let above = row.saturating_sub(1) * width;
        let below = (row + 1).min(height - 1) * width;
        let (Some(above), Some(current), Some(below), Some(target)) = (
            base.get(above..above + width),
            base.get(start..start + width),
            base.get(below..below + width),
            mask.get_mut(start..start + width),
        ) else {
            return;
        };

        for column in 0..width {
            let left = column.saturating_sub(1);
            let right = (column + 1).min(width - 1);
            let (top_left, top_middle, top_right) = (above[left], above[column], above[right]);
            let (middle_left, middle_right) = (current[left], current[right]);
            let (bottom_left, bottom_middle, bottom_right) =
                (below[left], below[column], below[right]);

            let vertical = (bottom_left + 2.0 * bottom_middle + bottom_right)
                - (top_left + 2.0 * top_middle + top_right);
            let horizontal = (top_right + 2.0 * middle_right + bottom_right)
                - (top_left + 2.0 * middle_left + bottom_left);
            let magnitude = (vertical * vertical + horizontal * horizontal).sqrt() / SOBEL_SPAN;
            let progress = clamp((magnitude - threshold) / ramp, 0.0, 1.0);
            target[column] = progress * progress * (3.0 - 2.0 * progress);
        }
    }
}

/// One sample of the scalar Sobel and smoothstep.
///
/// The two end columns and the tail of a row go through this on the vector path,
/// so it cannot drift from the scalar one at the edges.
#[cfg(target_arch = "x86_64")]
#[inline]
fn sobel_sample(
    above: &[f32],
    current: &[f32],
    below: &[f32],
    target: &mut [f32],
    column: usize,
    threshold: f32,
    ramp: f32,
) {
    let left = column.saturating_sub(1);
    let right = (column + 1).min(current.len().saturating_sub(1));
    let (top_left, top_middle, top_right) = (above[left], above[column], above[right]);
    let (middle_left, middle_right) = (current[left], current[right]);
    let (bottom_left, bottom_middle, bottom_right) = (below[left], below[column], below[right]);

    let vertical = (bottom_left + 2.0 * bottom_middle + bottom_right)
        - (top_left + 2.0 * top_middle + top_right);
    let horizontal = (top_right + 2.0 * middle_right + bottom_right)
        - (top_left + 2.0 * middle_left + bottom_left);
    let magnitude = (vertical * vertical + horizontal * horizontal).sqrt() / SOBEL_SPAN;
    let progress = clamp((magnitude - threshold) / ramp, 0.0, 1.0);
    target[column] = progress * progress * (3.0 - 2.0 * progress);
}

/// The Sobel and smoothstep, eight columns at a time.
///
/// Every operation is the one `sobel_sample` applies to the same column in the
/// same order, including the order the three terms of each response are added
/// in and the division by the ramp. `1 / SOBEL_SPAN` is a power of two, so the
/// multiply the vector path uses is the same scaling the scalar division is.
#[cfg(target_arch = "x86_64")]
#[target_feature(enable = "avx2")]
#[expect(
    unsafe_op_in_unsafe_fn,
    reason = "the body is one unsafe operation, and it is entered only after the feature is detected"
)]
unsafe fn sobel_mask_avx2(base: &[f32], mask: &mut [f32], plane: Plane, threshold: f32, ramp: f32) {
    use std::arch::x86_64::*;

    let width = plane.width;
    let height = plane.height;
    let two = _mm256_set1_ps(2.0);
    let three = _mm256_set1_ps(3.0);
    let span = _mm256_set1_ps(1.0 / SOBEL_SPAN);
    let threshold8 = _mm256_set1_ps(threshold);
    let ramp8 = _mm256_set1_ps(ramp);
    let zero = _mm256_setzero_ps();
    let one = _mm256_set1_ps(1.0);

    for row in 0..height {
        let start = row * width;
        let above_start = row.saturating_sub(1) * width;
        let below_start = (row + 1).min(height - 1) * width;
        let (Some(above), Some(current), Some(below), Some(target)) = (
            base.get(above_start..above_start + width),
            base.get(start..start + width),
            base.get(below_start..below_start + width),
            mask.get_mut(start..start + width),
        ) else {
            return;
        };

        sobel_sample(above, current, below, target, 0, threshold, ramp);

        // SAFETY: the row slices are exactly `width` long, and the loop keeps
        // the widest load and every store inside the row.
        let mut column = 1;
        while column + 8 < width {
            let up_left = _mm256_loadu_ps(above.as_ptr().add(column - 1));
            let up_mid = _mm256_loadu_ps(above.as_ptr().add(column));
            let up_right = _mm256_loadu_ps(above.as_ptr().add(column + 1));
            let mid_left = _mm256_loadu_ps(current.as_ptr().add(column - 1));
            let mid_right = _mm256_loadu_ps(current.as_ptr().add(column + 1));
            let low_left = _mm256_loadu_ps(below.as_ptr().add(column - 1));
            let low_mid = _mm256_loadu_ps(below.as_ptr().add(column));
            let low_right = _mm256_loadu_ps(below.as_ptr().add(column + 1));

            let vertical = _mm256_sub_ps(
                _mm256_add_ps(
                    _mm256_add_ps(low_left, _mm256_mul_ps(low_mid, two)),
                    low_right,
                ),
                _mm256_add_ps(_mm256_add_ps(up_left, _mm256_mul_ps(up_mid, two)), up_right),
            );
            let horizontal = _mm256_sub_ps(
                _mm256_add_ps(
                    _mm256_add_ps(up_right, _mm256_mul_ps(mid_right, two)),
                    low_right,
                ),
                _mm256_add_ps(
                    _mm256_add_ps(up_left, _mm256_mul_ps(mid_left, two)),
                    low_left,
                ),
            );
            let squared = _mm256_add_ps(
                _mm256_mul_ps(vertical, vertical),
                _mm256_mul_ps(horizontal, horizontal),
            );
            let magnitude = _mm256_mul_ps(_mm256_sqrt_ps(squared), span);
            let progress = clamp8(
                _mm256_div_ps(_mm256_sub_ps(magnitude, threshold8), ramp8),
                zero,
                one,
            );
            let value = _mm256_mul_ps(
                _mm256_mul_ps(progress, progress),
                _mm256_sub_ps(three, _mm256_mul_ps(two, progress)),
            );
            _mm256_storeu_ps(target.as_mut_ptr().add(column), value);

            column += 8;
        }
        while column < width - 1 {
            sobel_sample(above, current, below, target, column, threshold, ramp);
            column += 1;
        }
        sobel_sample(above, current, below, target, width - 1, threshold, ramp);
    }
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

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_vector_horizontal_pass_matches_the_scalar_one() {
        // Widths below, around and past the eight lane step, so the tail and
        // the reflected ends are both exercised, and tap counts from the
        // mask's five to the largest kernel.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x2026_1003u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for width in [1usize, 2, 3, 4, 5, 7, 8, 9, 15, 16, 17, 33, 64, 129] {
            let source: Vec<f32> = (0..width).map(|_| (next() % 512) as f32 - 256.0).collect();
            for sigma in [0.45f32, 0.5, 0.8, 2.0, 16.0] {
                let mut weights = Vec::new();
                let radius = build_kernel(sigma, &mut weights);
                let mut want = vec![0.0f32; width];
                let mut got = vec![0.0f32; width];
                blur_row(&source, &mut want, &weights, radius);
                // SAFETY: `avx2` was just detected, and the row is exactly
                // `width` samples long in both directions.
                unsafe { blur_row_avx2(&source, &mut got, &weights, radius) };
                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(bits(&got), bits(&want), "width {width} sigma {sigma} moved");
            }
        }
    }

    #[test]
    fn the_blur_matches_scipy_along_a_row() {
        let target = blurred(&RAMP, 5, 1, 0.8);
        for (got, want) in target.iter().zip(RAMP_BLURRED) {
            assert!((got - want).abs() < 1e-6, "got {got}, want {want}");
        }
    }

    /// The scalar vertical pass, which the register accumulator replaced.
    #[cfg(target_arch = "x86_64")]
    fn vertical_scalar(
        source: &[f32],
        target: &mut [f32],
        plane: Plane,
        weights: &[f32],
        radius: usize,
    ) {
        let width = plane.width;
        for row in 0..plane.height {
            let start = row * width;
            let target_row = &mut target[start..start + width];
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
                let tap_row = &source[bump..bump + width];
                for (value, tap) in target_row.iter_mut().zip(tap_row) {
                    *value += tap * weight;
                }
            }
        }
    }

    #[cfg(target_arch = "x86_64")]
    #[test]
    fn the_register_accumulator_matches_the_scalar_vertical_pass() {
        // Odd widths put a tail after the last full eight lane block, and the
        // short heights exercise the reflected rows. Both paths start from
        // positive zero and add the same products in the same tap order, so
        // every sample has to come back at the same bits.
        if !is_x86_feature_detected!("avx2") {
            return;
        }
        let mut state = 0x2026_1003u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for (width, height) in [(1usize, 1usize), (9, 3), (17, 9), (37, 5), (2903, 4)] {
            let source: Vec<f32> = (0..width * height)
                .map(|_| (next() % 4096) as f32 - 2048.0)
                .collect();
            for sigma in [0.45f32, 0.5, 0.8, 2.0, 16.0] {
                let mut weights = Vec::new();
                let radius = build_kernel(sigma, &mut weights);
                let plane = Plane { width, height };
                let taps = weights.len();
                let mut want = vec![0.0f32; source.len()];
                vertical_scalar(&source, &mut want, plane, &weights, radius);
                let mut got = vec![0.0f32; source.len()];

                for row in 0..height {
                    let mut rows: [*const f32; MAX_KERNEL_TAPS] =
                        [std::ptr::null(); MAX_KERNEL_TAPS];
                    for (index, slot) in rows.iter_mut().enumerate().take(taps) {
                        // `reflect` leaves an in-range index alone, so this is
                        // the same row set the vertical pass resolves.
                        let tap = reflect(row as isize + index as isize - radius as isize, height);
                        *slot = source[tap * width..].as_ptr();
                    }
                    let target_row = &mut got[row * width..row * width + width];
                    // SAFETY: every row points at `width` samples inside
                    // `source`, which does not overlap `got`.
                    unsafe { accumulate_row(&rows, taps, target_row, &weights, true) };
                }

                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&got),
                    bits(&want),
                    "{width}x{height} sigma {sigma} moved"
                );
            }
        }
    }

    #[test]
    fn the_ring_temp_matches_a_whole_plane_temp() {
        // The ring holds `min(height, 2 * radius + 1)` filtered rows and reuses
        // them, so this is the check that moving the window never serves a stale
        // row. A 64x129 plane at sigma 2 keeps seventeen rows for 129 output
        // rows, which is the reuse case; sigma 16 needs all 129, which is the
        // whole-plane case; the short planes store every row they have.
        let mut state = 0x2026_1003u64;
        let mut next = move || {
            state ^= state << 13;
            state ^= state >> 7;
            state ^= state << 17;
            state
        };

        for (width, height) in [
            (1usize, 1usize),
            (9, 3),
            (17, 9),
            (37, 5),
            (64, 129),
            (129, 64),
        ] {
            let source: Vec<f32> = (0..width * height)
                .map(|_| (next() % 4096) as f32 - 2048.0)
                .collect();
            for sigma in [0.45f32, 0.5, 0.8, 2.0, 16.0] {
                let mut weights = Vec::new();
                let radius = build_kernel(sigma, &mut weights);
                let plane = Plane { width, height };

                let mut whole_temp = vec![0.0f32; width * height];
                let mut whole = vec![0.0f32; width * height];
                blur(
                    &source,
                    &mut whole,
                    &mut whole_temp,
                    plane,
                    &weights,
                    radius,
                );

                let ring = ring_rows(radius, height) * width;
                let mut ring_temp = vec![0.0f32; ring];
                let mut ringed = vec![0.0f32; width * height];
                blur(
                    &source,
                    &mut ringed,
                    &mut ring_temp,
                    plane,
                    &weights,
                    radius,
                );

                let bits = |values: &[f32]| {
                    values
                        .iter()
                        .map(|value| value.to_bits())
                        .collect::<Vec<_>>()
                };
                assert_eq!(
                    bits(&ringed),
                    bits(&whole),
                    "{width}x{height} sigma {sigma} with {} ring rows moved",
                    ring_rows(radius, height)
                );
            }
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
    fn the_blend_limits_the_candidate_to_the_3x3_window() {
        // A fully open mask leaves the clamp alone, so a candidate past the
        // window comes back as the local extreme, with the edge sample
        // repeated at the border.
        let plane = Plane {
            width: 3,
            height: 3,
        };
        let mut workspace = Workspace::new();
        workspace.prepare(3, 3).expect("sized");
        for (index, value) in workspace.luma_mut().iter_mut().enumerate() {
            *value = index as f32;
        }
        workspace.third[..9].fill(1.0);

        workspace.first[..9].fill(CODE_VALUES);
        workspace.blend(plane, 0.0);
        assert_eq!(
            workspace.first[..9].to_vec(),
            vec![4.0, 5.0, 5.0, 7.0, 8.0, 8.0, 7.0, 8.0, 8.0]
        );

        workspace.first[..9].fill(0.0);
        workspace.blend(plane, 0.0);
        assert_eq!(
            workspace.first[..9].to_vec(),
            vec![0.0, 0.0, 1.0, 0.0, 0.0, 1.0, 3.0, 3.0, 4.0]
        );

        // Overshoot widens the window in both directions.
        workspace.first[..9].fill(CODE_VALUES);
        workspace.blend(plane, 300.0);
        assert_eq!(workspace.first[..9].to_vec(), vec![CODE_VALUES; 9]);
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
