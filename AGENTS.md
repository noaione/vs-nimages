# agents

## project

`vs-nimages` is a rust VapourSynth plugin that analyzes and manipulates images,
shipped as `vapoursynth-nimages`. keep the public identity unchanged:

- project name: `vapoursynth-nimages`
- plugin identifier: `xyz.n4o.nimages`
- callable namespace: `nimages`
- filters: `PeakStats`, `PeakGrayShades`, `Levels`, `Posterize`
- crate: `vs-nimages`
- native artifact: `vs_nimages.dll`, `libvs_nimages.so`, `libvs_nimages.dylib`
- python distribution: `vapoursynth-nimages`

the first release is `GRAY8` only. `PeakStats` and `PeakGrayShades` leave pixels
alone and attach their results as frame properties, so a caller composes
`PeakStats` -> `Levels(use_props=True)` instead of asking for automatic levels in
one call. `Levels` and `Posterize` are 256-entry lookup tables.

`docs/IMPLEMENTATIONS.md` is the plan of record for the filter surface, the
argument names and the property names. read it before changing anything public.
`docs/FINDINGS.md` is what was verified against the python reference, what the
crate gets wrong, and which decisions are already locked.

## status

the four filters are implemented for `GRAY8` and covered by
`tests/check-nimages.py`. `docs/FINDINGS.md` §8 tracks the milestones: M1, M2 and
M3 are done, M4 is deferred because it touches the sibling checkout, and M5 is
distribution work.

## where the behaviour comes from

the filters port `nmanga/autolevel.py` from the sibling checkout
`../nao-manga-rls`. that checkout is reference material: read it, run it, do not
modify it.

`tools/golden.py` imports it and refuses to write fixtures when the reference
implementation disagrees with it, so the golden vectors cannot drift away from
the python pipeline silently. the places where this plugin deliberately differs
are recorded in `docs/FINDINGS.md` §3.7 and §5, and every affected shade case
carries a `parity` of `nmanga`, `diverges` or `reference-only` plus the
`nmanga_expect` value it diverges from.

## repository rules

- use `pyproject.toml` and hatchling. do not add `setup.py`.
- keep the wheel plugin-only. it installs `vapoursynth/plugins/nimages/` with
  `manifest.vs` and adds no python module.
- never commit anything yourself. ask the maintainer first.
- there is no `CHANGELOG.md` yet, although `pyproject.toml` already points at
  one. ask before adding it.
- treat `tests/fixtures/` as generated. change it by running `tools/golden.py`,
  never by editing the json or the `.bin` files.
- keep `src/` free of VapourSynth types outside `lib.rs` and `src/filters/`. the
  algorithms must stay testable without a core.
- pass `--locked` to cargo so the committed lockfile is what gets built.

## source layout

- `src/lib.rs`: plugin declaration through `declare_plugin!`, and the four filter
  registrations.
- `src/error.rs`: `NImagesError`, the type that crosses the boundary.
- `src/filters/mod.rs`: the shared filter layer. reading a clip, refusing
  anything but `GRAY8`, reading optional arguments, registering the node,
  building a frame's histogram, and applying a lookup table to a plane.
- `src/filters/{peak_stats,peak_gray_shades,levels,posterize}.rs`: the four
  filters. all `Parallel`, all with a strict spatial dependency on their input.
- `src/histogram.rs`: the stride-aware `[u64; 256]` histogram every analyzer
  shares. `from_plane` refuses rows that do not fit instead of reading past them.
- `src/peaks.rs`: `find_local_peak`, a dependency-free replacement for
  `scipy.signal.find_peaks` as `nmanga` uses it. the region of interest is
  assembled in a fixed `[u64; 258]` buffer, so frame evaluation allocates nothing.
- `src/gray_shades.rs`: `analyze_gray_shades` over the fixed `0..=255` binning.
- `src/levels.rs`: the level lookup table, `automatic_gamma`, and `validate`.
- `src/posterize.rs`: the posterization lookup table.
- `src/round.rs`: the ties-to-even helper `levels` and `posterize` share.
- `tests/test_golden.rs`: replays every fixture through the algorithms. frame
  fixtures are rebuilt into a stride-padded buffer whose padding byte is not a
  shade value, so a histogram that over-reads a row cannot pass.
- `tests/check-nimages.py`: the integration validator. replays the same fixtures
  through the built plugin, then checks geometry, per-frame independence,
  repeated and out-of-order and concurrent requests, property preservation,
  determinism and the error messages.
- `tools/golden.py`: the golden-vector generator. see below.
- `hatch_build.py`: cargo build, plugin staging, wheel tagging, license
  inclusion.

## hard constraints from the crate

`vapoursynth4-rs` 0.5.1 has four traps. `docs/FINDINGS.md` §2 has the detail.

- register the output with `core.create_video_filter(...)`. do not use
  `VideoNode::new`, whose null check is inverted and hands back a node with a
  null handle exactly when creation failed.
- pass `None` to `FilterRegister::new`. API 4's plugin `registerFunction` takes
  no free callback, so any `functionData` leaks for the life of the plugin. read
  every argument from the input map in `Filter::create` instead.
- write total code. the release profile sets `panic = "abort"`, so a panic
  aborts the host process rather than failing one frame, and the binding's own
  panic recovery is unsound for a formatted panic payload. no `unwrap`, no
  panicking index, fixed-size arrays over `Vec` where the bound is known, and
  clamp or saturate rather than assert.
- bind the dependency array before taking a reference to it.
  `Dependencies::new(&array)` is correct; `impl From<[T; N]> for &Dependencies`
  returns a reference to a temporary.

two smaller ones worth knowing: property keys must be ascii alphanumeric or
underscore because `KeyStr::from_cstr` asserts, and `MapRef::from_ptr` is
crate-private, so a mutable property map comes from `Frame::properties_mut`.

## local build and test

the crate builds both a `cdylib` for VapourSynth and an `rlib` for `tests/`.
keep both in `crate-type`, or the integration tests stop linking.

```powershell
cargo build --release --locked
cargo test --locked
cargo clippy --all-targets
cargo fmt --check
```

the development python is `.venv\Scripts\python.exe` (3.12, VapourSynth R80). the
test extras add numpy, which the validator needs:

```powershell
uv sync --extra dev --extra dev-tests
.venv\Scripts\python.exe tests\check-nimages.py
```

run the validator after any change to `src/filters/`, `src/lib.rs` or a fixture.
`uv sync` builds the wheel, which runs cargo and installs the plugin into the
venv, so the validator always sees the current source.

to install a hand-built plugin instead:

```powershell
cargo build --release
copy target\release\vs_nimages.dll .venv\Lib\site-packages\vapoursynth\plugins\nimages\
.venv\Scripts\python.exe -c "import vapoursynth as vs; print([p.identifier for p in vs.core.plugins()])"
```

## filter contract

things worth knowing before touching `src/filters/`. `README.md` has the full
argument and property tables.

- every filter refuses anything but a constant `Gray` 8 bit clip, at creation.
- `PeakStats` and `PeakGrayShades` copy the input frame, so pixels and properties
  both survive, and then attach their own properties.
- `Levels` and `Posterize` allocate from the input frame's format and pass the
  input as `prop_src`, which is what carries the properties onto the output.
- `NImagesGrayShades` and `NImagesGrayShadePercentages` are always both present
  and always the same length. a zero-length array is a valid property, so the
  empty case writes both rather than omitting them.
- `Levels(use_props=True)` reads `NImagesBlackLevel` and `NImagesWhiteLevel` per
  frame, so it fails at frame time, not at creation, when they are missing.
- `peak_offset` is a code value, added to the black point before the curve is
  built. it is not a percentage point, which is the `nmanga` cli bug recorded in
  `docs/FINDINGS.md` §7.
- `auto_gamma` refuses a black point of 128 or more, because the expression is
  undefined there.

## golden vectors

`tools/golden.py` needs numpy, scipy and pillow, which the `golden` extra
provides:

```powershell
uv run --extra golden tools\golden.py --nmanga-path ..\nao-manga-rls
uv run --extra golden tools\golden.py --check
```

`--check` fails when a committed fixture is stale, so run it after touching
anything that changes a fixture's inputs. it compares the data files only:
`manifest.json` records the interpreter and library versions a run used, so it
differs between machines even when the data is current, and a mismatch there is
reported as a note rather than a failure. the generator self-checks the peak
reference against `scipy.signal.find_peaks` on every run and aborts on any
disagreement with `nmanga`, so a failure there means the reference moved, not
that the fixtures need accepting.

## writing style

prose in `AGENTS.md`, `README.md` and code comments is lowercase and direct.

- say what the code does and why, not what it visibly is
- no `we`, no `our`, no `should`, no `might`. state the rule or the fact
- prefer the concrete number or path over a description of it
- do not end a list item with `;`. a full stop is fine, as in the sibling
  repositories
- code comments keep the same directness but may use any capitalization
