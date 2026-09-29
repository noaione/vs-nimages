# vapoursynth-nimages — Pre-implementation Findings & Plan

Status: original findings verified; M6 extension in progress
Companion to: [IMPLEMENTATIONS.md](./IMPLEMENTATIONS.md)

Reference implementation under study:
`I:\Work\GitHub\nao-manga-rls\nmanga\autolevel.py`
(the `IMPLEMENTATIONS.md` relative link `../nmanga/autolevel.py` does not resolve;
the sibling checkout is `../nao-manga-rls`).

---

## 1. Verified environment

| Item | Value |
|---|---|
| Rust | 1.98.1 (2026-09-01), edition 2024 |
| `cargo build` | clean, baseline crate compiles |
| `vapoursynth4-rs` | 0.5.1 (features `vs-42`, `vsscript-43`) |
| `vapoursynth4-sys` | 0.4.1+R79 → `VAPOURSYNTH_API_VERSION` = 4.2 |
| VapourSynth runtime (`.venv`) | R80, reports `api_version=R4.3` |
| Golden-reference venv (`nao-manga-rls/.venv`) | Python 3.13.9, numpy 2.5.3, scipy 1.18.1, Pillow 12.3.0, vapoursynth R80 |

Loading a plugin compiled against API 4.2 into the R80 (API 4.3) runtime works.

### 1.1 Plugin skeleton proof (done)

A scaffolding spike (`src/lib.rs`, `PassThrough`) was built and loaded successfully:

```text
core.nimages.PassThrough(src)     -> loads, identifier xyz.n4o.nimages / namespace nimages
13x7 GRAY8 frame n=2              -> correct dims + pixel value
ImgSeqPath / custom props         -> preserved through core.copy_frame
parallel out-of-order frame grabs -> correct, deterministic
missing clip argument             -> "PassThrough: argument clip is required"
```

Required entry point shape (verified working):

```rust
#[unsafe(no_mangle)]
pub unsafe extern "system-unwind" fn VapourSynthPluginInit2(
    plugin: *mut ffi::VSPlugin,
    vspapi: *const ffi::VSPLUGINAPI,
) {
    let configured = ((*vspapi).configPlugin)(
        c"xyz.n4o.nimages".as_ptr(),
        c"nimages".as_ptr(),
        c"vapoursynth-nimages".as_ptr(),
        ffi::vs_make_version(0, 1),
        VAPOURSYNTH_API_VERSION,
        0,
        plugin,
    );
    if configured == 0 { return; }
    FilterRegister::<MyFilter>::new(None).register(plugin, vspapi);
}
```

`core.copy_frame(&input)` copies pixels **and** properties — sufficient for the
analysis filters that must not touch pixels.

---

## 2. Blocking crate issues and the chosen workarounds

These were found by reading the vendored crate source and confirmed against the
working spike.

### 2.1 `VideoNode::new` has an inverted null check — do not use it

`vapoursynth4-rs-0.5.1/src/node.rs:213`

```rust
ptr.is_null()
    .then_some(unsafe { Self::from_ptr(ptr, core.api()) })
```

`then_some` yields `Some` when the predicate is *true*, so this returns
`Some(VideoNode { handle: null() })` exactly when filter creation failed, and
`None` on success. Any caller doing `.ok_or(...)` gets the logic backwards, and
any caller unwrapping it dereferences a null node.

**Workaround:** register the output through
`CoreRef::create_video_filter(output, name, &vi, Box::new(filter), deps)`, which
has no such check and is what the spike uses. Worth reporting upstream.

### 2.2 No free callback for plugin function data

API 4's `VSPLUGINAPI::registerFunction` has no `free` parameter (unlike
`VSAPI::registerFunction`), and `FilterRegister::register` therefore cannot pass
one. Anything handed to `FilterRegister::new(Some(data))` leaks for the lifetime
of the plugin.

**Workaround:** pass `None` and parse every parameter from the input map inside
`Filter::create`. The spike does this. Any static table (e.g. a constant `Levels`
LUT) belongs in the filter struct created in `create`, not in registration data.

### 2.3 Panics are unsound *and* fatal — filter code must be panic-free

`vapoursynth4-rs-0.5.1/src/node/internal.rs:44-47` and `:80-83` recover a caught
panic with

```rust
let e = p.downcast::<&str>().unwrap_unchecked();
```

A `panic!` with a format argument produces a `String` payload, so the downcast
fails and `unwrap_unchecked()` yields an invalid value — undefined behaviour
before the error message is even used. Separately, `Cargo.toml` sets
`panic = "abort"` in the release profile, so a panic terminates the *host*
(`vspipe`, the Python interpreter) rather than the frame.

There is no configuration that makes both paths safe, so the contract is:

- keep `panic = "abort"`, and
- make all filter and algorithm code total: fixed-size `[u64; 256]` arrays,
  checked conversions, no `unwrap`/`expect`/indexing that can fail on
  attacker-controlled arithmetic, and
- validate every parameter at filter construction so `get_frame` cannot hit an
  error branch that panics.

This is a hard review constraint for Milestone 2 and 3, not a stylistic one.

### 2.4 `impl<const N> From<[T; N]> for &Dependencies` is a footgun

It takes the array **by value** and returns a reference to that temporary, which
cannot outlive the statement (it is only accepted because `&Dependencies` is
`Copy`-ish through the raw pointer cast). Use `Dependencies::new(&array)`
instead, as the spike does.

### 2.5 `Core::get_video_format_name` returns NUL padding

`vapoursynth4-rs-0.5.1/src/frame/format.rs` builds the name with

```rust
CStr::from_bytes_with_nul_unchecked(&self.buffer).to_bytes()
```

over the API's fixed `[u8; 32]` buffer. `CStr::to_bytes` returns
`inner.len() - 1` bytes, so a name VapourSynth wrote as `Gray8\0` comes back as
`Gray8` followed by the 25 remaining padding zeros, and `str::from_utf8_unchecked`
turns that into a `&str` with interior NULs. `CString::new` then refuses it.

The failure is silent and confusing: `format!` renders the string fine, so a
message built from it looks correct in a debugger, but the `CString::new` guard
in `log_debug` returns early and **the whole log line disappears**. It cost an
afternoon of probing to find, because every other line in the same function was
delivered normally.

**Workaround:** build the description from the format's own fields. This is what
`describe_frame` in `src/filters/mod.rs` does.

### 2.6 VapourSynth drops a message logged from a filter's `create`

`core.log` works from `get_frame` and reaches a host's `add_log_handler`
callback. The same call from `Filter::create` is never delivered, so a settings
line written there is invisible to the only consumer that would read it.

**Workaround:** write it from the first frame the filter is asked for, guarded by
an `AtomicBool` whose `swap` returns the previous value, so exactly one frame of
a filter that the core calls concurrently reports it. `report_settings_once` in
`src/filters/mod.rs` does this. The input format is named from the frame rather
than the node, which is also more accurate for a clip whose dimensions vary.

---

## 3. Algorithm semantics pinned against SciPy 1.18.1

SciPy has no `find_peaks` equivalent in Rust and the doc forbids depending on it,
so the port needs a clean-room implementation. These are the exact rules, each
cross-checked against the real SciPy before writing any Rust.

### 3.1 Plateau-aware local maxima

- A plateau spanning indices `i..=j` (all equal, `j` chosen as one past the last
  equal element in the implementation below) is one peak at `(i + j - 1) / 2`,
  integer division — i.e. the centre rounded **down** for even-length plateaus.
- SciPy never reports index `0` or `n-1`: boundary samples are never peaks.
- A strictly increasing/decreasing run has no peak.

```rust
// run of equal values starting at i, ending at j-1
i = 1;
while i < n - 1 {
    if x[i - 1] < x[i] {
        j = i + 1;
        while j < n && x[j] == x[i] { j += 1; }
        if j == n { break }
        if x[j] < x[i] { peaks.push((i + j - 1) / 2) }
        i = j;
    } else { i += 1; }
}
```

Verified: **0 mismatches / 20 000 random arrays** against `scipy.signal.find_peaks`.

### 3.2 Prominence (doc §7.3 is correct as written)

From the peak, walk left while `x[k] <= x[peak]`, tracking the minimum; stop at
the boundary or the first sample **strictly greater** than the peak height.
Repeat right. `prominence = x[peak] - max(left_min, right_min)`.

Verified: **0 mismatches / 41 699 (array, peak) pairs**.

> Note: taking the minimum over the *whole* side instead of stopping at the first
> higher sample is wrong and disagrees with SciPy on ~6% of random arrays. The
> doc's wording is right; this is recorded because the naive reading is tempting.

### 3.3 Thresholds are inclusive

`height` and `prominence` both keep a peak when the value is **equal** to the
threshold (`>=`), verified directly. This matters for the doc's §13.1 fixture
"a peak exactly at the minimum height threshold".

### 3.4 ROI handling and bin mapping

`nmanga` pads the ROI with one zero bin at each end; that padding is what makes
ROI-edge peaks detectable at all (SciPy would otherwise ignore them). The
`peaks >= 0 && peaks < len(roi)` filter after the `-1` remap is a no-op, because
SciPy can never return the padding indices.

The `bins_roi` lookup is the identity mapping, so no array is needed:

```text
black_roi = hist[0 ..= upper_limit]                 -> black = roi_index
white_roi = hist[(255 - upper_limit) ..= 255]       -> white = 255 - upper_limit + roi_index
```

### 3.5 Selection and fallback

- Candidates are filtered by height (and prominence when requested), then the
  **tallest** wins; ties resolve to the **lowest bin** (`np.argmax` semantics —
  first maximum index).
- If nothing qualifies, detection is rerun with `height = 0` and prominence
  disabled, using the *same* padded ROI.
- An all-zero ROI yields no peak in either pass (`0 > 0` is false), so
  `black = 0` / `white = 255` with `*PeakFound = 0`.
- `skip_white_check` short-circuits **before** the white ROI is computed and
  returns `white = 255`.
- `upper_limit` is not validated by the function; the CLI clamps it to `1..=255`.
  The plugin must validate it (`255 - upper_limit` must stay in range).

### 3.7 `peak_percentage` alone is a no-op

Found while writing the fixture cases. The first pass keeps peaks with
`height >= h` and, when requested, `prominence >= p`; when it keeps nothing, the
fallback reruns with **both** thresholds removed. Because the height test is
monotone, if the tallest peak fails it then every peak fails it, so the first
pass is empty and the fallback returns the tallest peak anyway. `peak_percentage`
therefore cannot change the selected black or white level on its own.

It does matter when a prominence threshold is also set, because it can empty the
first pass and hand the decision to the fallback, which ignores prominence. Both
behaviours are covered by fixtures (`height-below-threshold-fallback`,
`prominence-changes-the-winner`, `prominence-empties-the-pass`) and by
`peaks::tests::the_height_threshold_only_matters_together_with_prominence`.

`*PeakFound` is also weaker than it looks: it is false only when the region of
interest has no non-zero bin at all, since the fallback finds any peak.

### 3.8 End-to-end equivalence

A clean-room reimplementation of `find_local_peak` reproduced `nmanga`'s output
**exactly** on 640 (image × parameter) combinations:

- all-black, all-white, uniform mid-gray, odd dimensions (37×21)
- 60 pseudo-random images across six generators (uniform, low-key, high-key,
  constant, sparse-palette, uniform/3)
- `upper_limit ∈ {1, 60, 255}`
- `peak_percentage ∈ {default, None, 0.0, 100.0, 50.0}`
- `peak_prominence ∈ {default, None, 0.1, 0.01}`
- `skip_white_check ∈ {False, True}`

This is the algorithm the Rust port must implement, and it is the basis for the
golden-vector generator.

---

## 4. Levels

### 4.1 Rounding mode is practically unobservable for `GRAY8`

Exhaustive comparison of Python `round()` (ties-to-even) against half-up rounding
over all 256 inputs produced **0 differing values** for every parameter set
tested:

```text
(10,245,1.00) (10,245,1.37) (37,231,0.73) (0,255,1.0) (12,245,1.18)
(1,254,1.0)   (60,255,1.0)  (10,245,0.5)  (10,245,2.0) (5,250,1.05)
```

No input produced an exact `.5` tie. The doc's §8.1 ImageMagick comparison
(0/1/1 differing values) is consistent with this.

**Consequence:** honour the doc's §16 default (`pillow` / ties-to-even) for
documented fidelity and cheap exhaustiveness, but do not treat byte-parity with
ImageMagick as a hard requirement — it is already unattainable and irrelevant at
this precision.

`f64::round()` in Rust rounds half away from zero, which is *not* ties-to-even.
Implement an explicit helper anyway so the contract is stated in code rather than
inherited from an accident.

### 4.2 Automatic gamma has a hard domain limit

```text
black_normalized = black / 255
internal_gamma   = ln(0.5) / ln((0.5 - black_normalized) / (1 - black_normalized))
level_gamma      = round(1 / internal_gamma, 2)
```

Measured behaviour of the reference:

| `black_level` | result |
|---|---|
| 0 | `level_gamma = 1.0` |
| 12 | `1.07` |
| 60 | `1.53` |
| 127 | `8.0` |
| **128 … 254** | **`ValueError: math domain error`** |
| **255** | **`ZeroDivisionError`** |

The default `upper_limit = 60` keeps callers safe, but `upper_limit` may be up to
255, and `peak_offset` moves the black point too. The plugin must reject
`adjusted_black >= 128` (or clamp) with a clear message rather than producing NaN.

### 4.3 Channel behaviour

`nmanga` uses `image.point(lut * len(image.getbands()))`. The `* len(bands)`
replicates the table per band, which is the alpha-hazard noted in doc §8.2. The
plugin processes plane 0 only for `GRAY8`, so this is moot in the first release —
but the `alpha` policy must be written down before RGB support is added.

---

## 5. Gray shades: `analyze_gray_shades` bins adaptively (new bug, not in the plan)

Found while generating the golden vectors. `nmanga/autolevel.py:294` reads

```python
hist, _ = np.histogram(img_array, bins=256)
```

with **no `range=`**, unlike `find_local_peak` a few lines above it, which
correctly passes `range=(0, 256)`. With `range=None` NumPy derives the bin edges
from the data's `(min, max)`, so the 256 bins span the *observed* value range
instead of `0..255`, and the value reported as `shade` is a bin index over that
adaptive range rather than a gray value.

Measured on `nmanga` directly:

| input | reported shades | correct shades |
|---|---|---|
| constant 0 | `[(128, 100.0)]` | `[(0, 100.0)]` |
| constant 128 | `[(128, 100.0)]` | `[(128, 100.0)]` |
| constant 200 | `[(128, 100.0)]` | `[(200, 100.0)]` |
| only values 10 and 20 | `[(0, 50.0), (255, 50.0)]` | `[(10, 50.0), (20, 50.0)]` |
| values 10..137 | every other integer | each integer once |

Consequences:

- A constant-valued image always reports shade 128 at 100%, whatever its value.
- Any image that does not contain both 0 and 255 reports bin indices that are
  unrelated to its gray values, and collides distinct values into one bin, so the
  percentages are wrong too.
- Consumers that treat the result as gray values are silently broken:
  `pad_shades_to_bpc` documents "a list of gray shade values (0-255)" and is fed
  straight from `analyze_gray_shades` by `nmanga/cli/posterize.py`.

The plugin's `PeakGrayShades` cannot meaningfully reproduce this, because
`IMPLEMENTATIONS.md` §5.2 defines `NImagesGrayShades` as "Significant shade
values". The doc's §2 goal — "Match the existing Python behavior where it is
intentional, while documenting and correcting existing validation and
unit-conversion bugs" — covers it.

**Locked 8-bit contract:** use the fixed `0..=255` binning that `find_local_peak`
already uses, the same `[u64; 256]` histogram as every other filter, and true
gray values in `NImagesGrayShades`. M6 extends wider integer input to one bin per
native code value and reports shades in native sample units without changing
the `GRAY8` behavior.

The golden generator has already been written against this contract and records
the divergence rather than hiding it: every shade case carries
`"parity": "nmanga" | "diverges" | "reference-only"`, and diverging cases keep
`nmanga_expect` alongside the expected values. Of the current fixtures, 27 of 65
histogram cases and 11 of 26 frame cases diverge.

---

## 6. Posterization

### 6.1 The trailing `quantize()` is provably redundant (doc §4.4 answered)

For every `bits` in `1..=8`, `posterized.quantize(colors=2**bits,
dither=NONE).convert("L")` is **byte-identical** to `posterized`, and the mapped
image already contains exactly `2**bits` distinct gray levels:

| bits | colors | distinct output levels | quantize identical |
|---:|---:|---:|:--:|
| 1 | 2 | 2 | yes |
| 2 | 4 | 4 | yes |
| 3 | 8 | 8 | yes |
| 4 | 16 | 16 | yes |
| 5 | 32 | 32 | yes |
| 6 | 64 | 64 | yes |
| 7 | 128 | 128 | yes |
| 8 | 256 | 256 | yes |

**Consequence:** the native implementation omits the quantization and the golden
tests assert the table above, so the omission is proven rather than assumed.

### 6.2 Exact mapping contract

```text
colors = 2 ** bits
level  = round_ties_even(x * (colors - 1) / 255)     // integer 0..=colors-1
out    = round_ties_even(level * 255 / (colors - 1)) // integer 0..=255
```

Two details that are easy to get wrong:

- Pillow's `Image.point()` with a float-returning lambda **rounds** to the nearest
  integer, it does not truncate. Verified for every `bits`; `bits=3`, `5`, `6`, `7`
  actually differ between truncation and rounding.
- An integer ties-to-even formulation reproduces Python `round()` exactly for all
  256 inputs at all 8 bit depths (0 mismatches).

No exact `.5` ties occur in either rounding step for `bits = 1..=8`, so Rust's
`f64::round` would in practice also agree — but the explicit helper is kept for
the same reason as in §4.1.

### 6.3 Lloyd-Max levels (`method=1`)

`Posterize(method=1)` does not use the §6.2 table. It solves the levels per
frame from the frame's own histogram with the Lloyd-Max solver in `lloyd.py`,
which sits beside `posterize_image_by_bits` as the second reference for this
filter:

```text
counts[i] = how many samples of plane 0 hold code value i
levels    = linspace(0, maximum, colors)
repeat 40 times:
    edges[j]  = (levels[j-1] + levels[j]) / 2
    levels[j] = sum(counts[i] * i) / sum(counts[i])   over bucket j, interior j only
out[x]    = round_ties_even(levels[bucket of x])
```

Vectors taken from running `lloyd.py` itself on 8 bit pages:

| page | colors | levels |
| --- | ---: | --- |
| mass at 40, 90, 200 and 250, one cluster per bucket | 4 | 0, 90, 200, 255 |
| mass at 40 and 200 only | 4 | 0, 85, 200, 255 |
| every code value, one sample each | 256 | 0..=255, the identity |

- Both ends are **pinned**. Level 0 and level `colors-1` never move, whatever
  mass sits there, so the 40/90/200/250 page maps 0 and 255 rather than the
  outermost cluster means.
- A bucket with **no mass** keeps the evenly spaced level it was initialised
  with. On the 40/200 page, bucket 1 spans 43..=127 and stays at 85, and 40
  itself lands on the pinned 0.
- The levels stay strictly ascending, because a bucketed mean can never cross
  the midpoint that bounds its own bucket. They also stay distinct, so an empty
  bucket widens a gap instead of collapsing two levels.
- The rounding in the last step is ties-to-even, so a level exactly on `.5`
  rounds to the even integer. The Rust solver keeps the levels as `f64` and
  rounds only the emitted value, which matters: the bucket boundary stays at
  the *unrounded* midpoint, as it does in the reference.
- `bits` equal to the sample depth is the identity for this method too, because
  `2^depth` levels with pinned ends assign every code value to itself. The
  filter then returns the input frame without building a table.
- The levels come from **the frame being processed**, not from the clip. A clip
  whose frames differ gets a level set each, which is what a page-per-frame
  caller wants and what a moving clip would show as flicker.

The accumulator and the monotone edge scan that the Rust solver uses were
checked against `lloyd.py` over 200 random histograms of 8, 16, 32, 256 and
1024 code values with color counts from 2 through 64: 0 mismatches. The Rust
unit tests freeze the vectors above, and `tests/check-nimages.py` re-derives
the same algorithm in numpy and compares it against the built plugin for an
8 bit and a 16 bit page.

---

## 7. Gaps in `IMPLEMENTATIONS.md` to resolve

1. **`force_gray` is dropped.** `find_local_peak` returns
   `(black, white, force_gray)`. `PeakStats` exposes only black/white/found flags.
   In a VapourSynth graph the clip is already `GRAY8`, so the decision belongs
   upstream — but the Milestone 4 wrapper must replace `nmanga`'s
   `force_gray`-driven branching explicitly rather than silently lose it.
2. **`*PeakFound` has no Python counterpart**, so golden vectors for it must come
   from the reimplementation in §3, not from `nmanga` (§9.1 records the
   invariant this creates).
3. **`peak_offset` units.** The plugin contract must be
   `black_percent = (black_level + peak_offset) / 255 * 100`, matching the
   orchestrator action. The CLI's `create_magick_params` adds the offset *after*
   converting to percent (`nmanga/autolevel.py:252`), so `peak_offset=1` means one
   percentage point ≈ 2.55 code values. Doc §9.4 is confirmed; the CLI fix is a
   separate change in the sibling repo.
4. **`peak_percentage=0` / `peak_prominence=0`.** Doc §9.1 proposes
   `0 < x <= 100`. In the reference, the guard
   `value <= 0 and value >= 100` is a conjunction of opposites that never fires,
   so `0` reaches `ceil(total * 0 / 100) = 0` and simply disables the height
   filter. The fixtures record `0.0` as accepted and equivalent to `None`; the
   filter layer should accept `0 <= x <= 100` and document `None` as the explicit
   way to disable a threshold. Note §3.7: the height threshold is a no-op on its
   own either way.
5. **Empty `NImagesGrayShades`** — resolved by decision 2: both arrays are always
   present, empty when nothing qualifies.
6. **`total_pixels` for 1×1 frames** makes the default 0.25% threshold `ceil(1 *
   0.0025) = 1`, so only a completely uniform 1×1 image has a peak. Covered by the
   `tiny-1x1` and `tiny-1x1-mid` frame fixtures.
7. **The plugin description is the only human-readable name VapourSynth stores.**
   `configPlugin` takes one name argument, surfaced as `core.plugins()[i].name`.
   `IMPLEMENTATIONS.md` §5 lists both a display name and a description, so only one
   can be used; this plugin passes the description, matching `vs-imageseqs`.

---

## 8. Milestones

Each milestone ends with something executable and verifiable.

### M1 — Freeze behaviour and generate golden vectors — **done**

- `tools/golden.py` drives `nmanga.autolevel` from the sibling checkout
  (`--nmanga-path`, default `../nao-manga-rls`) plus the §3 reference
  implementation, and writes deterministic fixtures to `tests/fixtures/`.
- It self-checks the reference against `scipy.signal.find_peaks` before writing
  anything, aborts on any disagreement with `nmanga`, and `--check` reports stale
  fixtures.
- Fixture payload: raw `GRAY8` planes in `tests/fixtures/frames/*.bin`, plus
  `manifest.json`, `peaks.json`, `shades.json`, `levels.json`, `posterize.json` and
  `frames.json`. Histograms are deduplicated into a table per file.
- Current contents: 288 peak cases over 36 histograms, 65 gray-shade cases, 13
  level tables, 8 posterize tables, and 13 frames with 104 peak and 26 shade cases.
- Locked here: rounding mode, empty-array representation, `*PeakFound`
  derivation, `upper_limit` and gamma validation, and the §5 shade-binning fix.

Every shade case records `parity` as `nmanga`, `diverges` or `reference-only`, and
diverging cases keep `nmanga_expect` so the difference stays visible and tested:
27 of 65 histogram cases and 11 of 26 frame cases diverge, all because of §5.

### M2 — Pure Rust algorithms — **done**

- `src/histogram.rs` — stride-aware `[u64; 256]` histogram, `total_pixels`, and a
  `region` accessor; rejects rows that do not fit rather than reading out of bounds.
- `src/peaks.rs` — §3 rules, `PeakOptions`/`PeakResult`. The region of interest is
  assembled in a fixed `[u64; 258]` buffer, so frame evaluation allocates nothing.
- `src/gray_shades.rs` — `ceil` threshold, strict `>`, stable descending count
  sort, `f64` percentages.
- `src/levels.rs` — LUT, automatic gamma with domain validation, `LevelError`.
- `src/posterize.rs` — LUT.
- `src/round.rs` — the ties-to-even helper the two mapping filters share.
- `tests/test_golden.rs` — replays every fixture as an integration test, which is
  why the crate also builds an `rlib`; frame fixtures are rebuilt into a
  stride-padded buffer with a non-shade padding byte so an over-read cannot pass.
- No VapourSynth types outside `lib.rs`, and every entry point is total for its
  input type, which is what the `panic = "abort"` constraint in §2.3 requires.

Verified: `cargo test` 58 passed (48 unit, 10 integration),
`cargo clippy --all-targets` clean, `cargo fmt --check` clean,
`cargo build --release --locked` produces `vs_nimages.dll`, and that release
artifact still loads and serves frames in VapourSynth R80.

### M3 — Plugin filters — **done**

- `src/error.rs` holds `NImagesError`, the type that crosses the boundary.
- `src/filters/mod.rs` holds the shared layer: reading a clip, refusing anything
  but `GRAY8`, reading optional arguments, registering the node, building a
  frame's histogram, and applying a lookup table to a plane.
- `src/filters/{peak_stats,peak_gray_shades,levels,posterize}.rs` are the four
  filters, all `Parallel`, all with a strict spatial dependency on their input.
- Dimensions come from the frame, never from the node, and every row walk stops
  after `width` samples, so stride padding is neither read nor written.
- The analysis filters `core.copy_frame`, which keeps pixels and properties, and
  then write their own properties. The mapping filters allocate from the input
  frame's format and pass it as `prop_src`, which copies the properties onto the
  output.
- `Levels` keeps its table inline in the instance data when the parameters are
  constant, and rebuilds one per frame when `use_props=True`.
- `tests/check-nimages.py` is the integration validator. It replays the golden
  vectors through the built plugin, including 248 histogram peak cases and 60
  histogram shade cases materialised as real frames, and then checks geometry,
  per-frame independence, repeated and out-of-order and concurrent requests,
  property preservation, determinism, the color families each filter takes,
  channel separation on RGB, dynamic dimensions and the error messages.

Verified: `cargo test` 61 passed (51 unit, 10 integration),
`cargo clippy --all-targets` clean, `cargo fmt --check` clean,
`cargo build --release --locked` loads in VapourSynth R80 with all four filters
registered, and `tests/check-nimages.py` reports **1886 checks passed**.

Two things the plan called for that are not here: `dev-tests` gained `numpy`
rather than `pytest`, because the validator is a plain script in the style of
`vs-imageseqs` rather than a pytest suite, and there is no `vspipe` run because the
concurrent-request and determinism checks cover the same ground without needing an
encoder.

### 8.1 Input surface, widened after review

The first M3 draft took `GRAY8` only and refused a clip whose dimensions vary.
Both were wrong for how the filters are used. M3 widened the family surface to
8-bit formats; M6 adds integer formats through 16 bits and float `Levels`:

| filter | families | planes | variable dimensions |
|---|---|---|---|
| `PeakStats` | Gray integer 8–16 bit | plane 0 | yes |
| `PeakGrayShades` | Gray integer 8–16 bit | plane 0 | yes |
| `Levels` | Gray/RGB/YUV integer 8–16 bit; GrayS/RGBS | all | yes |
| `Posterize` | Gray/RGB/YUV integer 8–16 bit | all | yes |

Formats outside those rows are refused, with the message naming what was
received. Float `Levels` takes `black` and `white` in 8-bit code values; peak
properties, `peak_offset`, and `auto_gamma` remain integer-only.

A clip whose dimensions or format vary reports `Undefined` at the node, so the
format is checked twice: once at creation when the node declares one, and again
per frame when it does not. The output `VideoInfo` is the input's own, which is
what carries `width = height = 0` through.

**`getFrameWidth` is the sample count for every plane.** VapourSynth gives RGB
frames three *separate* plane buffers rather than one interleaved buffer.
`getReadPtr` for planes 0, 1 and 2 of a 6x4 RGB24 frame returned addresses
differing by whole plane allocations, not by one byte. Each plane walk advances
by its sample width (one byte for 8-bit integer, two for wider integer, four for
32-bit float) and its own byte stride. The first implementation assumed an
interleaved layout and mixed RGB channels; the RGB channel-separation check in
`tests/check-nimages.py` is what catches that, and it stays in the suite.

### M4 — `nmanga` integration — **deferred, out of scope**

Decision 3 keeps this repository self-contained, so the `nmanga` wrapper and the
§9.4 `create_magick_params` fix are a separate change in that repository:

- Compose `PeakStats` -> `Levels(use_props=True)`.
- Wire to `vapoursynth-imageseqs` per doc §15.
- Fix the `create_magick_params` `peak_offset` unit bug (§9.4).
- Golden-compare against the Pillow pipeline.

The plugin's own Python integration tests still cover the filter behaviour
against the same golden vectors.

### M5 — Distribution

- Matrix build, `.dll`/`.so`/`.dylib`; `hatch_build.py` already stages the artifact
  and writes `manifest.vs` (`vs_nimages`), which matches the crate's `cdylib` output
  name.
- Drop the unused `vsscript-43` feature; the plugin never links VSScript.
- Check whether `--locked` builds are viable (a `Cargo.lock` is committed).

### M6 — Higher precision and colour (later, unchanged)

---

## 9. Locked decisions

Recorded after review; these are now part of the contract.

1. **Rounding contract: Pillow, ties-to-even.** `Levels` and `Posterize` use an
   explicit ties-to-even helper so the rule is stated in code rather than
   inherited from `f64::round`. §4.1 shows the choice is unobservable at `GRAY8`
   against half-up, and it keeps byte-parity with the existing Python pipeline.
   ImageMagick byte-parity is explicitly *not* a goal.
2. **Empty gray-shade result: always set both properties.** `mapSetIntArray` with
   `size = 0` creates an empty property (documented in `VSAPI::mapSetIntArray`),
   so `NImagesGrayShades` and `NImagesGrayShadePercentages` are always present and
   always the same length, including when that length is zero.
3. **Scope: `vs-nimages` stays self-contained.** `nao-manga-rls` is not modified
   in this effort, so `IMPLEMENTATIONS.md` Milestone 4 (the `nmanga` wrapper) and
   the §9.4 `create_magick_params` fix are deferred to a separate change in that
   repository. This repo still gets its own Python integration tests against the
   installed plugin.
4. **Golden vectors: committed to the repository.** `tests/fixtures/` is checked
   in so Rust and Python tests run without the sibling checkout, numpy, scipy or
   Pillow. `tools/golden.py` regenerates them and is the only thing that needs
   those dependencies.

### 9.1 Invariant accepted with decision 4

`nmanga` does not expose the `*PeakFound` flags, so those expected values come
from the reference implementation inside `tools/golden.py` rather than from
`nmanga`. The generator asserts `nmanga` parity for black and white on every
case and refuses to emit fixtures on any mismatch, so only the two flags rest on
the reference implementation.
