//! Replays the sibling's book policy and Pillow median fixtures without a core.

use serde::Deserialize;
use vs_nimages::book_peak::{BookOptions, BookPage, consensus};
use vs_nimages::histogram::Histogram;
use vs_nimages::peaks::{PeakOptions, find_local_peak};

#[derive(Deserialize)]
struct Fixtures {
    cases: Vec<Case>,
}

#[derive(Deserialize)]
struct Case {
    name: String,
    lower: i64,
    upper: i64,
    offset: i64,
    tolerance: i64,
    parameters: Parameters,
    consensus: Option<u16>,
    pages: Vec<Page>,
}

#[derive(Deserialize)]
struct Parameters {
    upper_limit: u8,
    peak_percentage: Option<f64>,
    peak_prominence: Option<f64>,
    skip_white: bool,
    median_filter: bool,
}

#[derive(Deserialize)]
struct Page {
    width: usize,
    height: usize,
    pixels: Vec<u8>,
    darkest_gray: u16,
    expect: Expected,
}

#[derive(Deserialize)]
struct Expected {
    black: u16,
    white: u16,
    black_found: bool,
    white_found: bool,
    eligible: bool,
    used_book_peak: bool,
}

#[test]
fn book_resolution_matches_nmanga_and_pillow() {
    let data = std::fs::read_to_string(
        std::path::Path::new(env!("CARGO_MANIFEST_DIR")).join("tests/fixtures/book.json"),
    )
    .expect("book fixtures");
    let fixtures: Fixtures = serde_json::from_str(&data).expect("book fixture schema");
    assert!(fixtures.cases.len() >= 69);
    for case in fixtures.cases {
        let options = BookOptions {
            lower: case.lower,
            upper: case.upper,
            offset: case.offset,
            tolerance: case.tolerance,
        };
        let peak_options = PeakOptions {
            upper_limit: case.parameters.upper_limit,
            peak_percentage: case.parameters.peak_percentage,
            peak_prominence: case.parameters.peak_prominence,
            skip_white: case.parameters.skip_white,
        };
        let pages: Vec<_> = case
            .pages
            .iter()
            .map(|page| {
                let histogram = if case.parameters.median_filter {
                    vs_nimages::median::histogram(page.width, page.height, 255, |x, y| {
                        u16::from(page.pixels[y * page.width + x])
                    })
                    .expect("median fixture")
                } else {
                    Histogram::from_plane(&page.pixels, page.width, page.width, page.height)
                        .expect("fixture dimensions")
                };
                let darkest = histogram
                    .counts()
                    .iter()
                    .position(|&count| count != 0)
                    .unwrap_or(255) as u16;
                assert_eq!(darkest, page.darkest_gray, "{}: Pillow median", case.name);
                let has_dark = darkest <= case.upper as u16;
                BookPage {
                    peaks: find_local_peak(&histogram, &peak_options),
                    has_dark,
                }
            })
            .collect();
        let shared = consensus(&pages, options);
        assert_eq!(shared, case.consensus, "{}: majority", case.name);
        for (index, (page, fixture)) in pages.into_iter().zip(case.pages).enumerate() {
            let result = options.resolve(page, shared);
            let want = fixture.expect;
            assert_eq!(
                result.peaks.black, want.black,
                "{} {index}: black",
                case.name
            );
            assert_eq!(
                result.peaks.white, want.white,
                "{} {index}: white",
                case.name
            );
            assert_eq!(
                result.peaks.black_found, want.black_found,
                "{} {index}: black status",
                case.name
            );
            assert_eq!(
                result.peaks.white_found, want.white_found,
                "{} {index}: white status",
                case.name
            );
            assert_eq!(
                result.eligible, want.eligible,
                "{} {index}: eligibility",
                case.name
            );
            assert_eq!(
                result.used_book_peak, want.used_book_peak,
                "{} {index}: fallback",
                case.name
            );
        }
    }
}

#[test]
fn empty_and_extreme_inputs_are_total() {
    let options = BookOptions {
        lower: 8,
        upper: 55,
        offset: i64::MAX,
        tolerance: 6,
    };
    assert_eq!(consensus(&[], options), None);
    let histogram = Histogram::from_plane(&[12], 1, 1, 1).expect("one pixel");
    let peaks = find_local_peak(&histogram, &PeakOptions::default());
    let page = BookPage {
        peaks,
        has_dark: true,
    };
    assert_eq!(consensus(&[page], options), None);
    assert!(!options.resolve(page, None).eligible);
    let negative = BookOptions {
        offset: i64::MIN,
        ..options
    };
    assert!(!negative.resolve(page, Some(12)).eligible);
}
