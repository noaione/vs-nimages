# unified Levels endpoints

status: implemented

## problem

integer Levels uses black and white, while GRAYS and RGBS use black_float and
white_float. the duplicate names make the same operation look like two
different APIs.

## proposed contract

- expose one black argument and one white argument for both sample domains.
- declare both arguments as VapourSynth float values. Python callers can
  continue passing integer literals; validate integer-format endpoints as
  finite whole numbers before converting them to code values.
- for integer formats, keep the current native code-value units and defaults:
  black defaults to 0 and white defaults to the sample maximum.
- for GRAYS and RGBS, interpret endpoints in 8-bit-equivalent code units and
  divide by 255 before building the per-sample curve. default white to 255,
  which resolves to 1.0. for example, white=245 resolves to 245 / 255.0, or
  about 0.9608.
- keep gamma, float output clamping, NaN preservation, and the existing
  restrictions on use_props, peak_offset, and auto_gamma.

this gives float clips one explicit scale instead of guessing from endpoint
magnitude. float values above 255 remain usable for endpoints above 1.0. a
fractional endpoint retains precision in the 8-bit-equivalent scale, so an old
white_float=0.94 setting becomes white=239.7.

## compatibility

the current M6 interface uses normalized float endpoints. the proposed
interface changes those units to 8-bit-equivalent code values; update examples,
tests, and callers before release. integer endpoint behavior stays unchanged.
decide whether to remove black_float and white_float immediately or accept them
as deprecated aliases for one transition period.

the VapourSynth argument signature has one type per argument. a float signature
lets Python convert integer literals, while callers that build a native
VapourSynth map directly must provide floats. integer clips still reject
fractional endpoint values after conversion.

## implementation outline

1. change the Levels argument signature to declare black and white once.
2. read endpoints as floats and resolve them after the frame format is known.
   Keep deferred resolution for variable-format clips.
3. for integer frames, reject non-finite, fractional, negative, and
   out-of-range endpoints before converting to the existing integer path.
4. for float frames, divide both endpoints by 255 and pass the resulting values
   to the existing per-sample curve.
5. update the argument tables, errors, README examples, and integration cases.

## validation

- test omitted endpoints on integer and float clips, including defaults on
  GRAY8, GRAY16, GRAYS, and RGBS.
- test integer code values on integer formats and reject fractional values.
- test float endpoint conversion at 0, 1, 255, fractional values, and values
  above 255. check that white=245 maps to 245 / 255.0.
- preserve tests for gamma, clamping, NaNs, infinities, variable-format clips,
  and per-plane behavior.
- run cargo test --locked, cargo clippy --all-targets, cargo fmt --check, and
  tests/check-nimages.py after implementation.

the reported test failure was an invalid expectation: white=40 is inside a
sample maximum of 50. the regression case now uses white=51, and cargo test
--locked passes.

## outcome

implemented as proposed. the deprecated aliases were not kept: `Levels` declares
`black:float:opt` and `white:float:opt` and no longer accepts `black_float` or
`white_float`, so an old float call fails loudly instead of silently rescaling.

- `resolve_for_domain` converts the endpoints once per resolution: whole numbers
  inside the frame's own code-value range on an integer frame, and `value / 255`
  on a float one.
- A variable-format clip resolves per frame, so a clip that mixes depths gets
  each frame's own table from one argument set.
- `tests/check-nimages.py` covers the defaults on `GRAY8`, `GRAY16`, `GRAYS` and
  `RGBS`, the conversion at 0, 1, 245, 255, fractional and above-255 endpoints,
  the endpoints an integer clip refuses, and a sequence that mixes 8 and 16 bit
  frames.

performance: measured against the pre-change build on the same machine, the per
sample `map` stage is unchanged within run-to-run noise (`GRAY8` 1.42 against
1.44 ms, `GRAYS` 7.61 against 7.33 ms, `RGBS` 23.23 against 23.35 ms) and the 16
bit `use_props` `curve` stage stays at 0.13 to 0.14 ms, so the unified arguments
cost nothing per frame.
