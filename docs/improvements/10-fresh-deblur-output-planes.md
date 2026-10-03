# fresh Deblur output planes

status: deferred on measurement. exact pixels, allocation/write cost only.

## current cost

[`src/filters/deblur.rs`](../../src/filters/deblur.rs) copies the input frame,
then acquires writable output planes and overwrites every active sample of each
processed plane. the copy is not an immediate full pixel copy.

[VapourSynth's copyFrame contract](https://www.vapoursynth.com/doc/api/vapoursynth4.h.html#copyframe)
specifies shared pixel storage with copy-on-write. acquiring a writable shared
plane can therefore copy its old contents before this filter replaces them.
the analyzers modify properties alone and do not trigger that pixel write, so
this proposal does not replace their cheap property copies.

## one experiment

allocate fresh Gray/RGB output with `core.new_video_frame`, carrying the input
as `prop_src`. for YUV, use `core.new_video_frame2` with fresh plane 0 and the
input's corresponding chroma planes. the binding in `vapoursynth4-rs` 0.5.1
already exposes both methods; no new raw frame wrapper is needed.

[`newVideoFrame2`](https://www.vapoursynth.com/doc/api/vapoursynth4.h.html#newvideoframe2)
accepts a null source entry for a fresh uninitialized plane and source entries
for the planes to retain. bind the plane-source and plane-index arrays before
passing slices, with the exact validated plane count and source-frame lifetime.

keep the current copy path for zero geometry before any allocation API that
requires positive dimensions. fill every active sample of every fresh plane.
YUV chroma and all input properties remain byte-identical.

## necessary stride fix

`write_plane` already reads the output's stride. `write_rgb` currently offsets
its output pointers using the input stride. fresh allocation must read and
validate each output plane's own stride and dimensions, separately from the
source. do not assume the two allocations have matching layout, or that output
RGB planes share a single stride without checking it.

leave conversion arithmetic unchanged in this experiment. 12 tests specialized
conversion later, against whichever allocation result is retained.

## measure and decision

compare output allocation plus writes, total latency, cold-start allocation,
and peak RSS. source `copy_frame` occurs before the timed `write` stage, while
first writable access occurs inside it; report total as well as `write` so work
cannot appear to vanish merely by moving it outside the timer.

cover Gray8/16/S, RGB24/48/S, YUV420/422/444 integer/float, and variable geometry.
retain input frames across evaluation to check that source pixels remain intact.
test subsampled chroma and properties, then both methods under concurrent pulls.

the upper bound is the copy traffic for processed planes, not the five-plane
restoration cost. an 8-bit Gray plane of 12 million pixels represents roughly
12 MB of copied payload, much less than Deblur scratch. adopt only if the write
or caller-visible result measures better.

## result

status: deferred on measurement. the write stage is inside the noise band this
would have to move.

`core.copy_frame` shares pixel storage, so the first writable acquire of a
processed plane pays to copy its old contents. a stage run at 2048x2048 with
`debug=1`, three frames, medians of three graph builds, on the build that also
carries the horizontal and vertical vector passes:

| case | luma | restore | write | total | write share |
| --- | ---: | ---: | ---: | ---: | ---: |
| `method=0`, `GRAY8` | 3.50 ms | 133.27 ms | 10.82 ms | 155.70 ms | 6.9% |
| `method=0`, `GRAY16` | 4.59 ms | 139.21 ms | 12.20 ms | 156.13 ms | 7.8% |
| `method=1`, `GRAY16` | 4.50 ms | 28.05 ms | 12.17 ms | 45.95 ms | 26.5% |

the whole `write` stage, which is one plane's copy plus one plane's per-pixel
rounding and store, is 10.8 to 12.2 ms against a 46 to 156 ms frame. the copy
is the smaller part of that, because the same stage also does the conversion
and the store, and the review's own upper bound is one plane of copied payload,
which is 4 MB here as `GRAY8` and 8 MB as `GRAY16`.

the change is not free either: it replaces one `copy_frame` call with a fresh
allocation for Gray and RGB and a `new_video_frame2` with a plane-source array
for YUV, plus a rewrite of `write_plane` and `write_rgb` to read each output
plane's own stride. that touches every format family to chase a fraction of one
to four milliseconds, under the 5% throughput bar the review sets. reverted to
the copy path, which is what the current build uses.

`cargo test --locked`, `cargo clippy --all-targets --locked -- -D warnings`,
`cargo fmt --check` and `tests/check-nimages.py` all pass unchanged on the
shipped build, and the twelve case Deblur hash harness reports the same hashes
as the baseline. no code change is recorded for this candidate.
