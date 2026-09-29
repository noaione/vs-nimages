//! Golden-vector tests over the committed fixtures in `tests/fixtures`.
//!
//! `tools/golden.py` generates those fixtures from `nmanga.autolevel` and from
//! the reference implementation documented in `docs/FINDINGS.md`, and refuses to
//! write them when the reference disagrees with `nmanga`. These tests replay
//! them, so the Rust port is checked against the same numbers the reference was.
//!
//! The peak and gray-shade fixtures are also replayed against real frame bytes
//! written into a stride-padded buffer, which is what a VapourSynth frame looks
//! like and what the histogram must not over-read.

use std::collections::HashMap;
use std::path::PathBuf;

use serde::Deserialize;

use vs_nimages::deblur::{
    Method as DeblurMethod, Params as DeblurParams, Workspace as DeblurWorkspace,
};
use vs_nimages::gray_shades::{self, GrayShade};
use vs_nimages::histogram::{BINS, Histogram};
use vs_nimages::levels;
use vs_nimages::peaks::{self, PeakOptions};
use vs_nimages::posterize;

/// Padding byte used when rebuilding a frame: deliberately not a valid shade
/// value, so any accidental inclusion of stride padding fails the histogram.
const PAD: u8 = 0xAA;
/// Extra bytes appended to every row, so the stride never equals the width.
const PAD_BYTES: usize = 7;

fn fixtures_dir() -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("tests")
        .join("fixtures")
}

fn load<T: for<'de> Deserialize<'de>>(name: &str) -> T {
    let path = fixtures_dir().join(name);
    let text = std::fs::read_to_string(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    serde_json::from_str(&text)
        .unwrap_or_else(|error| panic!("cannot parse {}: {error}", path.display()))
}

fn counts_from(values: &[u64]) -> [u64; BINS] {
    assert_eq!(
        values.len(),
        BINS,
        "a fixture histogram must have {BINS} bins"
    );
    let mut counts = [0u64; BINS];
    counts.copy_from_slice(values);
    counts
}

fn histogram(total_pixels: u64, values: &[u64]) -> Histogram {
    Histogram::from_counts(counts_from(values), total_pixels)
}

/// Copies tightly packed rows into a wider buffer, returning it and the stride.
fn pad_rows(bytes: &[u8], width: usize, height: usize) -> (Vec<u8>, usize) {
    let stride = width + PAD_BYTES;
    let mut buffer = vec![PAD; stride * height];
    for row in 0..height {
        buffer[row * stride..row * stride + width]
            .copy_from_slice(&bytes[row * width..(row + 1) * width]);
    }
    (buffer, stride)
}

fn histograms_of(fixtures: &HashMap<String, Vec<u64>>) -> &HashMap<String, Vec<u64>> {
    assert!(!fixtures.is_empty(), "the fixture file has no histograms");
    fixtures
}

// ---------------------------------------------------------------------------
// Peaks
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PeakFixtures {
    histograms: HashMap<String, Vec<u64>>,
    cases: Vec<PeakCase>,
}

#[derive(Deserialize)]
struct PeakCase {
    name: String,
    histogram: String,
    total_pixels: u64,
    upper_limit: u8,
    peak_percentage: Option<f64>,
    peak_prominence: Option<f64>,
    skip_white: bool,
    expect: PeakExpect,
}

#[derive(Deserialize)]
struct PeakExpect {
    black: u16,
    white: u16,
    black_found: bool,
    white_found: bool,
}

#[test]
fn peak_histogram_fixtures_match_the_reference() {
    let fixtures: PeakFixtures = load("peaks.json");
    let histograms = histograms_of(&fixtures.histograms);
    assert!(fixtures.cases.len() >= 200, "the fixture set was truncated");

    for case in &fixtures.cases {
        let values = histograms
            .get(&case.histogram)
            .unwrap_or_else(|| panic!("{}: no histogram named {}", case.name, case.histogram));
        let histogram = histogram(case.total_pixels, values);
        let result = peaks::find_local_peak(&histogram, &options_of(case));

        assert_eq!(result.black, case.expect.black, "{}: black", case.name);
        assert_eq!(result.white, case.expect.white, "{}: white", case.name);
        assert_eq!(
            result.black_found, case.expect.black_found,
            "{}: black_found",
            case.name
        );
        assert_eq!(
            result.white_found, case.expect.white_found,
            "{}: white_found",
            case.name
        );
    }
}

fn options_of(case: &PeakCase) -> PeakOptions {
    PeakOptions {
        upper_limit: case.upper_limit,
        peak_percentage: case.peak_percentage,
        peak_prominence: case.peak_prominence,
        skip_white: case.skip_white,
    }
}

// ---------------------------------------------------------------------------
// Gray shades
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct ShadeFixtures {
    histograms: HashMap<String, Vec<u64>>,
    diverging_cases: usize,
    cases: Vec<ShadeCase>,
}

#[derive(Deserialize)]
struct ShadeCase {
    name: String,
    histogram: String,
    total_pixels: u64,
    threshold: f64,
    expect: Vec<ShadeEntry>,
    parity: String,
    nmanga_expect: Option<Vec<ShadeEntry>>,
}

#[derive(Deserialize, Debug, PartialEq)]
struct ShadeEntry {
    shade: u16,
    percentage: f64,
}

fn shades_to_entries(shades: &[GrayShade]) -> Vec<ShadeEntry> {
    shades
        .iter()
        .map(|shade| ShadeEntry {
            shade: shade.shade,
            percentage: shade.percentage,
        })
        .collect()
}

#[test]
fn gray_shade_fixtures_match_the_reference() {
    let fixtures: ShadeFixtures = load("shades.json");
    let histograms = histograms_of(&fixtures.histograms);
    assert!(!fixtures.cases.is_empty(), "the fixture set was truncated");

    for case in &fixtures.cases {
        let values = histograms
            .get(&case.histogram)
            .unwrap_or_else(|| panic!("{}: no histogram named {}", case.name, case.histogram));
        let histogram = histogram(case.total_pixels, values);
        let shades = gray_shades::analyze_gray_shades(&histogram, case.threshold);
        assert_eq!(shades_to_entries(&shades), case.expect, "{}", case.name);
    }
}

#[test]
fn the_recorded_nmanga_divergence_is_real() {
    // Every case marked as diverging must actually differ, so the recorded
    // divergence keeps describing the reference instead of going stale.
    let fixtures: ShadeFixtures = load("shades.json");
    let mut diverging = 0;

    for case in &fixtures.cases {
        match case.parity.as_str() {
            "diverges" => {
                let nmanga = case
                    .nmanga_expect
                    .as_ref()
                    .unwrap_or_else(|| panic!("{}: diverging cases need nmanga_expect", case.name));
                assert_ne!(
                    *nmanga, case.expect,
                    "{}: marked diverging but matches",
                    case.name
                );
                diverging += 1;
            }
            _ => assert!(
                case.nmanga_expect.is_none(),
                "{}: only diverging cases carry nmanga_expect",
                case.name
            ),
        }
    }

    assert_eq!(diverging, fixtures.diverging_cases);
    assert!(diverging > 0, "the divergence must stay documented");
}

#[test]
fn shades_without_a_divergence_agree_with_nmanga() {
    let fixtures: ShadeFixtures = load("shades.json");
    let histograms = histograms_of(&fixtures.histograms);

    for case in &fixtures.cases {
        if case.parity != "nmanga" {
            continue;
        }
        let values = histograms
            .get(&case.histogram)
            .unwrap_or_else(|| panic!("{}: no histogram named {}", case.name, case.histogram));
        let histogram = histogram(case.total_pixels, values);
        let shades = gray_shades::analyze_gray_shades(&histogram, case.threshold);
        assert_eq!(
            shades_to_entries(&shades),
            case.expect,
            "{}: this case is supposed to be nmanga-identical",
            case.name
        );
    }
}

// ---------------------------------------------------------------------------
// Levels
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct LevelFixtures {
    cases: Vec<LevelCase>,
    gamma_cases: Vec<GammaCase>,
    invalid_gamma_levels: Vec<u8>,
}

#[derive(Deserialize)]
struct LevelCase {
    black: f64,
    white: f64,
    gamma: f64,
    expect: Vec<u8>,
}

#[derive(Deserialize)]
struct GammaCase {
    black_level: u8,
    expect: f64,
}

#[test]
fn level_tables_cover_every_input_value() {
    let fixtures: LevelFixtures = load("levels.json");
    assert!(fixtures.cases.len() >= 10, "the fixture set was truncated");

    for case in &fixtures.cases {
        let table =
            levels::levels_lut(case.black, case.white, case.gamma).unwrap_or_else(|error| {
                panic!(
                    "b={} w={} g={}: {error:?}",
                    case.black, case.white, case.gamma
                )
            });
        assert_eq!(
            table.as_slice(),
            case.expect.as_slice(),
            "b={} w={} g={}",
            case.black,
            case.white,
            case.gamma
        );
    }
}

#[test]
fn automatic_gamma_fixtures_match() {
    let fixtures: LevelFixtures = load("levels.json");
    for case in &fixtures.gamma_cases {
        assert_eq!(
            levels::automatic_gamma(case.black_level),
            Some(case.expect),
            "black_level={}",
            case.black_level
        );
    }
    for black_level in &fixtures.invalid_gamma_levels {
        assert_eq!(
            levels::automatic_gamma(*black_level),
            None,
            "black_level={black_level} must stay out of the domain"
        );
    }
}

// ---------------------------------------------------------------------------
// Posterize
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct PosterizeFixtures {
    cases: Vec<PosterizeCase>,
}

#[derive(Deserialize)]
struct PosterizeCase {
    bits: u8,
    colors: usize,
    levels: Vec<u8>,
    expect: Vec<u8>,
}

#[test]
fn posterize_tables_cover_every_depth_and_input() {
    let fixtures: PosterizeFixtures = load("posterize.json");
    assert_eq!(fixtures.cases.len(), 8, "one case per supported depth");

    for case in &fixtures.cases {
        let table = posterize::posterize_lut(case.bits)
            .unwrap_or_else(|| panic!("bits={} was rejected", case.bits));
        assert_eq!(
            table.as_slice(),
            case.expect.as_slice(),
            "bits={}",
            case.bits
        );

        // The mapping alone already produces exactly `colors` gray values, which
        // is why the reference's trailing quantize() is omitted.
        let mut produced: Vec<u8> = table.to_vec();
        produced.sort_unstable();
        produced.dedup();
        assert_eq!(produced, case.levels, "bits={}: distinct levels", case.bits);
        assert_eq!(
            produced.len(),
            case.colors,
            "bits={}: level count",
            case.bits
        );
    }
}

// ---------------------------------------------------------------------------
// Frames
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct FrameFixtures {
    frames: HashMap<String, FrameInfo>,
    peak_cases: Vec<FramePeakCase>,
    shade_cases: Vec<FrameShadeCase>,
}

#[derive(Deserialize)]
struct FrameInfo {
    width: usize,
    height: usize,
    file: String,
    bytes: usize,
    hist: Vec<u64>,
}

#[derive(Deserialize)]
struct FramePeakCase {
    name: String,
    frame: String,
    upper_limit: u8,
    peak_percentage: Option<f64>,
    peak_prominence: Option<f64>,
    skip_white: bool,
    expect: PeakExpect,
}

#[derive(Deserialize)]
struct FrameShadeCase {
    name: String,
    frame: String,
    threshold: f64,
    expect: Vec<ShadeEntry>,
    parity: String,
    nmanga_expect: Option<Vec<ShadeEntry>>,
}

/// Reads a frame fixture and rebuilds it with stride padding.
fn load_frame(info: &FrameInfo) -> Histogram {
    let path = fixtures_dir().join(&info.file);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    assert_eq!(bytes.len(), info.bytes, "{}: unexpected size", info.file);
    assert_eq!(
        bytes.len(),
        info.width * info.height,
        "{}: the packed size must match the dimensions",
        info.file
    );

    let (padded, stride) = pad_rows(&bytes, info.width, info.height);
    let histogram = Histogram::from_plane(&padded, stride, info.width, info.height)
        .unwrap_or_else(|| panic!("{}: the padded rows must fit", info.file));

    let expected_counts = counts_from(&info.hist);
    assert_eq!(
        histogram.counts(),
        expected_counts.as_slice(),
        "{}: the histogram must ignore stride padding",
        info.file
    );
    assert_eq!(histogram.total_pixels(), (info.width * info.height) as u64);
    histogram
}

#[test]
fn frame_histograms_ignore_stride_padding() {
    let fixtures: FrameFixtures = load("frames.json");
    assert!(fixtures.frames.len() >= 10, "the fixture set was truncated");
    for info in fixtures.frames.values() {
        let _ = load_frame(info);
    }
}

#[test]
fn frame_peak_fixtures_match_the_reference() {
    let fixtures: FrameFixtures = load("frames.json");
    assert!(!fixtures.peak_cases.is_empty());

    for case in &fixtures.peak_cases {
        let info = fixtures
            .frames
            .get(&case.frame)
            .unwrap_or_else(|| panic!("{}: no frame named {}", case.name, case.frame));
        let histogram = load_frame(info);
        let result = peaks::find_local_peak(
            &histogram,
            &PeakOptions {
                upper_limit: case.upper_limit,
                peak_percentage: case.peak_percentage,
                peak_prominence: case.peak_prominence,
                skip_white: case.skip_white,
            },
        );

        assert_eq!(result.black, case.expect.black, "{}: black", case.name);
        assert_eq!(result.white, case.expect.white, "{}: white", case.name);
        assert_eq!(
            result.black_found, case.expect.black_found,
            "{}: black_found",
            case.name
        );
        assert_eq!(
            result.white_found, case.expect.white_found,
            "{}: white_found",
            case.name
        );
    }
}

#[test]
fn frame_shade_fixtures_match_the_reference() {
    let fixtures: FrameFixtures = load("frames.json");
    assert!(!fixtures.shade_cases.is_empty());

    for case in &fixtures.shade_cases {
        let info = fixtures
            .frames
            .get(&case.frame)
            .unwrap_or_else(|| panic!("{}: no frame named {}", case.name, case.frame));
        let histogram = load_frame(info);
        let shades = gray_shades::analyze_gray_shades(&histogram, case.threshold);
        assert_eq!(shades_to_entries(&shades), case.expect, "{}", case.name);

        match case.parity.as_str() {
            "nmanga" => {}
            "diverges" => assert_ne!(
                case.nmanga_expect
                    .as_ref()
                    .expect("diverging cases need nmanga_expect"),
                &case.expect,
                "{}: marked diverging but matches",
                case.name
            ),
            other => panic!("{}: unexpected parity {other}", case.name),
        }
    }
}

// ---------------------------------------------------------------------------
// Deblur
// ---------------------------------------------------------------------------

#[derive(Deserialize)]
struct DeblurFixtures {
    cases: Vec<DeblurCase>,
}

#[derive(Deserialize)]
struct DeblurCase {
    name: String,
    format: String,
    sample: String,
    width: usize,
    height: usize,
    method: i64,
    radius: f64,
    strength: f64,
    iterations: u32,
    threshold: f64,
    overshoot: f64,
    planes: Vec<DeblurPlane>,
    expect: Vec<DeblurPlane>,
    tolerance: DeblurTolerance,
}

#[derive(Deserialize)]
struct DeblurPlane {
    file: String,
    width: usize,
    height: usize,
    bytes: usize,
}

#[derive(Deserialize)]
struct DeblurTolerance {
    max_abs_diff: f64,
    mean_abs_diff: f64,
    zero_fraction: f64,
}

/// One fixture plane, decoded.
enum DeblurData {
    U8(Vec<u8>),
    U16(Vec<u16>),
    F32(Vec<f32>),
}

impl DeblurData {
    fn decode(bytes: &[u8], sample: &str) -> Self {
        match sample {
            "u8" => Self::U8(bytes.to_vec()),
            "u16" => Self::U16(
                bytes
                    .as_chunks::<2>()
                    .0
                    .iter()
                    .map(|pair| u16::from_ne_bytes(*pair))
                    .collect(),
            ),
            "f32" => Self::F32(
                bytes
                    .as_chunks::<4>()
                    .0
                    .iter()
                    .map(|quad| f32::from_ne_bytes(*quad))
                    .collect(),
            ),
            other => panic!("unknown fixture sample type {other}"),
        }
    }

    fn len(&self) -> usize {
        match self {
            Self::U8(values) => values.len(),
            Self::U16(values) => values.len(),
            Self::F32(values) => values.len(),
        }
    }

    /// One sample in the 8 bit code values the kernels work in.
    fn code(&self, index: usize) -> f32 {
        match self {
            Self::U8(values) => values.get(index).copied().map_or(0.0, f32::from),
            Self::U16(values) => values
                .get(index)
                .copied()
                .map_or(0.0, |value| f32::from(value) * (255.0 / 65535.0)),
            Self::F32(values) => values
                .get(index)
                .copied()
                .map_or(0.0, |value| value * 255.0),
        }
    }

    /// The stored sample at one index, in this plane's own units.
    fn value(&self, index: usize) -> f64 {
        match self {
            Self::U8(values) => values.get(index).copied().map_or(0.0, f64::from),
            Self::U16(values) => values.get(index).copied().map_or(0.0, f64::from),
            Self::F32(values) => values.get(index).copied().map_or(0.0, f64::from),
        }
    }

    /// One code value in this plane's own units, rounded the way the reference
    /// rounds.
    fn sample(&self, code: f32) -> f64 {
        match self {
            Self::U8(_) => f64::from(code).round_ties_even(),
            Self::U16(_) => (f64::from(code) * (65535.0 / 255.0)).round_ties_even(),
            Self::F32(_) => f64::from(code) / 255.0,
        }
    }
}

fn load_deblur_plane(plane: &DeblurPlane, sample: &str) -> Vec<u8> {
    let path = fixtures_dir().join(&plane.file);
    let bytes = std::fs::read(&path)
        .unwrap_or_else(|error| panic!("cannot read {}: {error}", path.display()));
    let width = match sample {
        "u8" => 1,
        "u16" => 2,
        "f32" => 4,
        other => panic!("unknown fixture sample type {other}"),
    };
    assert_eq!(bytes.len(), plane.bytes, "{}: unexpected size", plane.file);
    assert_eq!(
        bytes.len(),
        plane.width * plane.height * width,
        "{}: the plane does not match its dimensions",
        plane.file
    );
    bytes
}

/// The luma the kernels run on, in 8 bit code values.
///
/// Gray and YUV carry it in plane 0; RGB is `0.2126 R + 0.7152 G + 0.0722 B`
/// over the encoded values, which is what the reference does on purpose.
fn deblur_luma(case: &DeblurCase, planes: &[DeblurData]) -> Vec<f32> {
    let length = case.width * case.height;
    let mut luma = vec![0.0f32; length];
    match case.format.as_str() {
        "Gray" | "YUV" => {
            for (index, value) in luma.iter_mut().enumerate() {
                *value = planes.first().map_or(0.0, |plane| plane.code(index));
            }
        }
        "RGB" => {
            for (index, value) in luma.iter_mut().enumerate() {
                let red = planes.first().map_or(0.0, |plane| plane.code(index));
                let green = planes.get(1).map_or(0.0, |plane| plane.code(index));
                let blue = planes.get(2).map_or(0.0, |plane| plane.code(index));
                *value = 0.2126 * red + 0.7152 * green + 0.0722 * blue;
            }
        }
        other => panic!("{}: unknown fixture format {other}", case.name),
    }
    luma
}

/// Writes the restored luma back through the plane rules, in code values.
///
/// Gray takes the restored plane, YUV takes it on luma and copies chroma, and
/// RGB adds one equal offset per pixel limited to the gamut the pixel has left.
fn deblur_planes(
    case: &DeblurCase,
    planes: &[DeblurData],
    luma: &[f32],
    restored: &[f32],
) -> Vec<Vec<f32>> {
    let mut produced: Vec<Vec<f32>> = Vec::with_capacity(planes.len());
    for (index, plane) in planes.iter().enumerate() {
        let length = case
            .planes
            .get(index)
            .map_or(0, |shape| shape.width * shape.height);
        produced.push(
            (0..length.min(plane.len()))
                .map(|offset| plane.code(offset))
                .collect(),
        );
    }

    match case.format.as_str() {
        "Gray" | "YUV" => {
            if let Some(target) = produced.first_mut() {
                target.copy_from_slice(restored);
            }
        }
        "RGB" => {
            for (index, (restored_value, luma_value)) in restored.iter().zip(luma).enumerate() {
                let mut low = f32::INFINITY;
                let mut high = f32::NEG_INFINITY;
                for plane in &produced {
                    let value = plane.get(index).copied().unwrap_or(0.0);
                    low = low.min(value);
                    high = high.max(value);
                }
                let delta = (*restored_value - *luma_value).clamp(-low, 255.0 - high);
                for plane in &mut produced {
                    if let Some(value) = plane.get_mut(index) {
                        *value += delta;
                    }
                }
            }
        }
        other => panic!("{}: unknown fixture format {other}", case.name),
    }
    produced
}

#[test]
fn deblur_fixtures_stay_inside_the_frozen_tolerance() {
    let fixtures: DeblurFixtures = load("deblur.json");
    assert!(fixtures.cases.len() >= 14, "the fixture set was truncated");
    let mut constrained = 0;

    for case in &fixtures.cases {
        let planes: Vec<DeblurData> = case
            .planes
            .iter()
            .map(|plane| DeblurData::decode(&load_deblur_plane(plane, &case.sample), &case.sample))
            .collect();
        let expect: Vec<DeblurData> = case
            .expect
            .iter()
            .map(|plane| DeblurData::decode(&load_deblur_plane(plane, &case.sample), &case.sample))
            .collect();
        assert_eq!(
            planes.len(),
            expect.len(),
            "{}: the input and expectation planes differ in count",
            case.name
        );

        let params = DeblurParams {
            method: if case.method == 0 {
                DeblurMethod::Deconvolution
            } else {
                DeblurMethod::EdgeSharpen
            },
            radius: case.radius as f32,
            strength: case.strength as f32,
            iterations: case.iterations,
            threshold: case.threshold as f32,
            overshoot: case.overshoot as f32,
        };
        let luma = deblur_luma(case, &planes);
        let mut workspace = DeblurWorkspace::new();
        workspace.prepare(case.width, case.height).expect("sized");
        workspace.luma_mut().copy_from_slice(&luma);
        let restored = workspace
            .restore(case.width, case.height, &params)
            .unwrap_or_else(|error| panic!("{}: {error:?}", case.name))
            .to_vec();
        let produced = deblur_planes(case, &planes, &luma, &restored);

        let mut worst = 0.0f64;
        let mut total = 0.0f64;
        let mut exact = 0usize;
        let mut count = 0usize;
        for (index, plane) in produced.iter().enumerate() {
            let want = expect
                .get(index)
                .unwrap_or_else(|| panic!("{}: no expectation plane {index}", case.name));
            assert_eq!(
                plane.len(),
                want.len(),
                "{}: plane {index} length",
                case.name
            );
            for (offset, code) in plane.iter().enumerate() {
                let got = want.sample(*code);
                let difference = (got - want.value(offset)).abs();
                worst = worst.max(difference);
                total += difference;
                if difference == 0.0 {
                    exact += 1;
                }
                count += 1;
            }
        }
        let mean = total / count as f64;
        let zero = exact as f64 / count as f64;

        assert!(
            worst <= case.tolerance.max_abs_diff,
            "{}: worst sample differs by {worst}, over {}",
            case.name,
            case.tolerance.max_abs_diff
        );
        assert!(
            mean <= case.tolerance.mean_abs_diff,
            "{}: mean sample differs by {mean}, over {}",
            case.name,
            case.tolerance.mean_abs_diff
        );
        assert!(
            zero >= case.tolerance.zero_fraction,
            "{}: only {zero} of the samples match exactly, under {}",
            case.name,
            case.tolerance.zero_fraction
        );
        if case.tolerance.zero_fraction > 0.0 {
            constrained += 1;
        }
    }

    assert!(
        constrained > 0,
        "no case constrains the exact-match fraction"
    );
}
