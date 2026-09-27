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
    black: u8,
    white: u8,
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
    shade: u8,
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

    assert_eq!(
        histogram.counts().as_slice(),
        counts_from(&info.hist).as_slice(),
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
