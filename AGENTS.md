# agents

## project

`vs-nimages` is a rust VapourSynth plugin that analyzes and manipulates images,
shipped as `vapoursynth-nimages`. keep the public identity unchanged:

- project name: `vapoursynth-nimages`
- plugin identifier: `xyz.n4o.nimages`
- callable namespace: `nimages`
- filters: `PeakStats`, `PeakGrayShades`, `Levels`, `Posterize`, `Deblur`
- crate: `vs-nimages`
- native artifact: `vs_nimages.dll`, `libvs_nimages.so`, `libvs_nimages.dylib`
- python distribution: `vapoursynth-nimages`

the first release is 8 bit integer only. `PeakStats` and `PeakGrayShades` take a
Gray clip and leave pixels alone, attaching their results as frame properties, so
a caller composes `PeakStats` -> `Levels(use_props=True)` instead of asking for
automatic levels in one call. `Levels` and `Posterize` take any 8 bit integer
family and rewrite every plane. `Deblur` takes Gray, RGB and YUV formats, at
any integer depth from 8 to 16 bits and the 32 bit float ones. Every filter
handles a clip whose dimensions are not known until a frame is asked for.

`docs/IMPLEMENTATIONS.md` is the plan of record for the filter surface, the
argument names and the property names. read it before changing anything public.
`docs/FINDINGS.md` is what was verified against the python reference, what the
crate gets wrong, and which decisions are already locked.

## status

the five filters are implemented and covered by `tests/check-nimages.py`.
`docs/FINDINGS.md` §8 tracks the milestones: M1, M2 and M3 are done, M4 is
deferred because it touches the sibling checkout, and M5 is distribution work.

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
- keep the wheel plugin-only. it installs the native library as
  `vapoursynth/plugins/vs_nimages.dll`, `.so` or `.dylib` and adds no python
  module.
- never commit anything yourself. ask the maintainer first.
- there is no `CHANGELOG.md` yet, although `pyproject.toml` already points at
  one. ask before adding it.
- treat `tests/fixtures/` as generated. change it by running `tools/golden.py`,
  never by editing the json or the `.bin` files.
- keep `src/` free of VapourSynth types outside `lib.rs` and `src/filters/`. the
  algorithms must stay testable without a core.
- pass `--locked` to cargo so the committed lockfile is what gets built.

## source layout

- `src/lib.rs`: plugin declaration through `declare_plugin!`, and the five filter
  registrations.
- `src/error.rs`: `NImagesError`, the type that crosses the boundary.
- `src/filters/mod.rs`: the shared filter layer. reading a clip, the `Accept`
  rules for what each filter takes, reading optional arguments, registering the
  node, building a frame's histogram, and rewriting a plane through a table.
- `src/filters/{peak_stats,peak_gray_shades,levels,posterize,deblur}.rs`: the
  five filters. all `Parallel`, all with a strict spatial dependency on their
  input.
- `src/histogram.rs`: the stride-aware `[u64; 256]` histogram every analyzer
  shares. `from_plane` refuses rows that do not fit instead of reading past them.
- `src/peaks.rs`: `find_local_peak`, a dependency-free replacement for
  `scipy.signal.find_peaks` as `nmanga` uses it. the region of interest is
  assembled in a fixed `[u64; 258]` buffer, so frame evaluation allocates nothing.
- `src/gray_shades.rs`: `analyze_gray_shades` over the fixed `0..=255` binning.
- `src/levels.rs`: the level lookup table, `automatic_gamma`, and `validate`.
- `src/posterize.rs`: the posterization lookup table and the Lloyd-Max level
  solver.
- `src/round.rs`: the ties-to-even helper `levels` and `posterize` share.
- `src/deblur.rs`: the reflected gaussian blur, the edge mask, the two
  sharpening candidates and the masked blend, over caller-owned scratch space.
- `tests/test_golden.rs`: replays every fixture through the algorithms. frame
  fixtures are rebuilt into a stride-padded buffer whose padding byte is not a
  shade value, so a histogram that over-reads a row cannot pass.
- `tests/check-nimages.py`: the integration validator. replays the same fixtures
  through the built plugin, then checks geometry, per-frame independence,
  repeated and out-of-order and concurrent requests, property preservation,
  determinism and the error messages.
- `tools/golden.py`: the golden-vector generator. see below.
- `tools/bench.py`: times the python reference against the plugin on the same
  pages. see below.
- `docs/BENCH.md`: the method and the measured results.
- `hatch_build.py`: cargo build, plugin staging, wheel tagging, license
  inclusion.

## sandbox

`sandbox/level-check`, `sandbox/level-webp-check` and `sandbox/posterize-check`
hold real manga pages for manual and benchmark runs. they are a private working
tree: read them, run them through the plugin, and name the directories where a
command needs a path, but do not commit them, copy them anywhere else, or quote
their contents. `.gitignore` already excludes `sandbox/`.

`posterize-check` is the useful one for dimension handling, because its pages are
four different sizes including a 5806x4128 spread and two widths that differ by a
single pixel. `level-webp-check` is the useful one for conversion, because a
lossy webp arrives as limited range YUV420P8 at an odd width, which `resize`
refuses; `docs/BENCH.md` has the detail.

## hard constraints from the crate

`vapoursynth4-rs` 0.5.1 has six traps. `docs/FINDINGS.md` §2 has the detail.

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
cargo clippy --all-targets --locked -- -D warnings
cargo fmt --check
```

clippy warnings are failures. keep the `-D warnings` flag so clippy cannot
report success while emitting warnings.

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
copy target\release\vs_nimages.dll .venv\Lib\site-packages\vapoursynth\plugins\
.venv\Scripts\python.exe -c "import vapoursynth as vs; print([p.identifier for p in vs.core.plugins()])"
```

## filter contract

things worth knowing before touching `src/filters/`. `README.md` has the full
argument and property tables.

- a filter checks the format the node declares, and re-checks the format of each
  frame. a clip whose dimensions or format vary reports `Undefined`, and only the
  frame can answer for it.
- `PeakStats` and `PeakGrayShades` take a Gray clip, because a histogram of one
  plane only means something for one. `Levels` and `Posterize` take any 8 bit
  integer family and rewrite every plane. `Deblur` takes Gray, RGB and YUV
  integer and float formats and moves luma alone, with chroma carried through
  and an RGB delta limited to the gamut each pixel has left.
- every plane walk uses the frame's own `frame_width`, `frame_height` and
  `stride`, so variable dimensions and subsampled chroma both work. samples are
  one byte apart: VapourSynth hands an RGB24 frame out as three separate plane
  buffers, not as one interleaved buffer, so the walk is the same as for Gray.
- `PeakStats` and `PeakGrayShades` copy the input frame, so pixels and properties
  both survive, and then attach their own properties.
- `Levels` and `Posterize` allocate from the input frame's format and pass the
  input as `prop_src`, which is what carries the properties onto the output.
- `Deblur` copies the input frame before it writes, so the planes it does not
  touch and the input's properties both survive byte for byte. its kernels take
  their scratch space from a pool, one workspace per concurrently evaluated
  frame.
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
- `Posterize(method=1)` solves the levels for each frame from plane 0's
  histogram with Lloyd-Max, and `method=0`, the default, spaces them evenly.
- every filter takes `debug:int:opt`. it writes a settings line on the first
  frame it is asked for and a stage timing line per frame, both through
  `core.log`, so a host collects them with `add_log_handler`. `tools/bench.py`
  parses those lines, so changing their shape means changing `DebugLog` too.
- do not build a log line from `Core::get_video_format_name`. see
  `docs/FINDINGS.md` §2.5: the name carries NUL padding and silently kills the
  line.

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

## benchmarks

`tools/bench.py` times the python reference against the plugin on the same pages,
one pipeline per process so peak resident memory is comparable, and reports the
per-stage split. it needs numpy, scipy and pillow, so it runs with both extras:

```powershell
uv run --extra golden --extra dev-tests tools\bench.py
uv run --extra golden --extra dev-tests tools\bench.py --write docs\BENCH.md
```

`--write` replaces the block between the `<!-- bench:start -->` and
`<!-- bench:end -->` markers in `docs/BENCH.md`. every run uses a 512 MiB frame
cache, which is what a caller would set for a manga volume and far below the
core's default; `--cache MB` moves it. run it before changing anything that
affects per-page cost, and keep the prose around the block honest about what is
and is not comparable: decoding differs by library, and the plugin side does not
write files because no VapourSynth writer is installed.

## writing style

prose in `AGENTS.md`, `README.md` and code comments is lowercase and direct.

- say what the code does and why, not what it visibly is
- no `we`, no `our`, no `should`, no `might`. state the rule or the fact
- prefer the concrete number or path over a description of it
- do not end a list item with `;`. a full stop is fine, as in the sibling
  repositories
- code comments keep the same directness but may use any capitalization
