# streaming blur scratch

status: implemented. the ring replaces the full horizontal plane and removes
13.6% of peak rss at a thirteen frame fan-in.

## current cost

[`Workspace`](../../src/deblur.rs) keeps five full f32 planes: `luma`, `first`,
`second`, `third`, and `temp`. `blur` fills the full horizontal `temp` before
starting its vertical pass. workspaces grow to their largest frame and the pool
retains one per concurrent evaluation.

## one experiment

replace only the full horizontal `temp` with a ring of filtered rows. keep the
other four planes and all candidate/mask/blend stages as they are. each vertical
output row needs at most `2 * radius + 1` horizontal rows. produce source rows
on demand, retain rows until their last consumer, and reuse ring slots afterward.

allocate at most `min(height, 2 * radius + 1) * width` f32 samples for an active
blur, with checked arithmetic and fallible reservation. the maximum kernel has
129 taps. one reusable ring sized for the largest active kernel replaces
`ensure(temp, width * height)`; leaving that full allocation alive defeats the
memory objective.

for tall pages, scratch changes from approximately `20 * width * height` bytes
to `16 * width * height + 4 * width * ring_rows`. it saves almost one of five
planes, or 20% of workspace storage. the process RSS reduction is smaller because
frames, caches and other allocations remain. a 12 million sample plane costs
48 MB in decimal units; removing one saves that payload per workspace.

## parity traps

map logical taps through the existing `reflect`, including repeated edge
samples, height below kernel length, and 1xN/Nx1 frames. track the source-row
identity of each ring slot; modulo addressing alone cannot establish that its
contents still belong to a needed reflected row. when the plane is shorter
than the kernel, storing all its rows is a valid bounded fallback.

every output retains the same horizontal fold, vertical tap order and f32
stores. source and target remain distinct during the blur. a later deconvolution
iteration cannot read overwritten intermediate source data. a reused workspace
also needs the current ring width/row capacity, not a previous frame's layout.

## measure and decision

first compare blur output bits for tiny, odd, tall and large planes through
all kernel lengths. then run restored-output and plugin checks. measure pure
blur time, both methods, and 1/2/4/8 concurrent full frames at the same cache
budget. include largest-to-smallest frame sequences and retained pool bytes.

this can improve cache locality or add scheduling overhead; no latency gain is
established. keep it for a measured memory/throughput benefit within the shared
regression limits. do not add rayon, tile the entire multi-iteration operation,
or shrink all pooled buffers at the same time as this experiment.

## result

status: implemented. the ring removes 44.4 MiB per workspace and 13.6% of peak
rss at a thirteen frame fan-in; no throughput change is claimed.

`Workspace` holds five `f32` planes sized `width * height`: `luma`, `first`,
`second`, `third` and `temp`. that is 20 bytes a sample, so a 12 megapixel
frame's scratch is 240 MB and the 5806x4128 spread's is 479 MB in decimal units.
these are workspace payloads; `docs/BENCH.md` reports whole-process peak RSS of
969 MiB for each deblur method, including frames, caches and pooled workspaces.
replacing the full
horizontal `temp` with a ring of at most `2 * radius + 1` filtered rows, or
`height` when that is smaller, takes the workspace to `16 * width * height +
4 * width * ring_rows`, and at the deblur's radius of 3 and 2 that ring is 7
and 5 rows.

| pin | f32 planes | per 12 Mpx frame |
| --- | ---: | ---: |
| current workspace | 5 | 240 MB |
| one plane removed | 4 | 192 MB |
| saving per workspace | 1 | 48 MB |

concurrent throughput on the current build, from
`.tmpbuild/deblur_parallelism.py` at 1 megapixel, 12 core threads: 53.0 ms of
wall for one frame on its own, 578.6 ms for 13 frames through a
`std.AverageFrames` fan-in, and 9.56x concurrency measured as reported work over
wall. the pool grows to the frames in flight, so the 48 MB is per concurrent
frame, not per process.

## what the scratch costs a process

the review asks whether the workspace is visible in a process at all, and that
was unmeasured. `.tmpbuild/deblur_rss.py` measures it: each case runs in its own
process, builds a `std.AverageFrames` fan-in of `frames` through either `Deblur`
or `PeakGrayShades`, pulls the centre frame and reports the peak resident set.
`PeakGrayShades` copies frames and allocates no scratch, so the gap between the
two fan-ins is the workspace storage in flight. the cache is held at 32 MiB so
it cannot dominate the gap.

| size | case | frames | peak rss | wall |
| --- | --- | ---: | ---: | ---: |
| 2048x2048 | control | 1 | 39.6 MiB | 7.2 ms |
| 2048x2048 | control | 13 | 88.4 MiB | 59.0 ms |
| 2048x2048 | `Deblur` | 1 | 119.4 MiB | 70.3 ms |
| 2048x2048 | `Deblur` | 13 | 728.6 MiB | 717.2 ms |
| 2903x4128 | control | 1 | 54.6 MiB | 15.7 ms |
| 2903x4128 | control | 13 | 194.4 MiB | 225.0 ms |
| 2903x4128 | `Deblur` | 1 | 283.2 MiB | 218.8 ms |
| 2903x4128 | `Deblur` | 13 | 651.7 MiB | 2071.4 ms |

one `Deblur` frame costs 79.8 MiB over the control at 2048x2048 and 228.6 MiB at
2903x4128, against workspace payloads of 83.9 MB and 239.7 MB. that is one
workspace to within a few percent, which is what makes the rest of the table
attributable rather than a guess.

the control's own growth is the frame cache: 139.8 MiB over twelve more frames
of 12 MB is 11.4 MiB a frame, the decoded `GRAY8` page. `Deblur` grows 368.5 MiB
over the same twelve, which is 1.6 further workspaces, so about 2.6 are live at
this size and the scratch is roughly 594 MiB of the 652 MiB peak, 91%.

at the bench's 512 MiB cache the frame cache adds several hundred MiB on top, so
the scratch share there is smaller than this 91%. the ring removes one of five
planes, 20% of a workspace, which is about 119 MiB off this 652 MiB peak and
proportionally less of the bench's 969 MiB.

so the memory objective is measured rather than assumed, and it is real: the
workspace dominates `Deblur`'s peak at a small cache.

## implementation

`Workspace::temp` is now a ring of `min(height, MAX_KERNEL_TAPS)` rows instead of
a whole plane, and `blur` drives the vertical pass one output row at a time from
rows it filters on demand. the four other planes, the candidates, the mask and
the blend are untouched.

the part the review flagged as the wrong-pixel risk is the slot addressing, so it
is resolved rather than assumed: `slot = source_row % ring_rows`, and the window
one output row consumes is a run of at most `ring_rows` consecutive source rows.
a run of at most `n` consecutive integers has distinct residues modulo `n`, so no
two rows of a window can share a slot. that holds for the reflected borders too:
a border window's distinct rows are a subset of the run, so they stay distinct.
each slot also records which source row it holds, and a row is filtered only when
that record disagrees, which is what makes a reused slot safe rather than merely
likely. `ring_rows == height` for a plane shorter than the kernel, which makes
the ring the whole plane and the path identical.

the x86-64 feature check moved to the top of `blur`, once per gaussian. the
previous shape already did that; the per-row loop would have been the 49.5k-times
mistake the horizontal attempt made.

## measured, after

same protocol as the table above: one process per case, `median` of three runs,
2903x4128, cache 32 MiB. the `before` column is the whole-plane `temp` build
measured the same way, in the same session pair.

| case | frames | before peak rss | after peak rss | before wall | after wall |
| --- | ---: | ---: | ---: | ---: | ---: |
| control | 1 | 54.6 MiB | 54.5 MiB | 17.6 ms | 14.4 ms |
| control | 13 | 194.4 MiB | 194.4 MiB | 240.5 ms | 217.9 ms |
| `Deblur` | 1 | 283.2 MiB | **238.8 MiB** | 199.3 ms | 157.4 ms |
| `Deblur` | 13 | 651.5 MiB | **563.0 MiB** | 1924.3 ms | 1934.5 ms |

one workspace costs 44.4 MiB less: 283.2 to 238.8 MiB for a single frame against
a control that is unchanged at 54.5 MiB. that is the ring replacing a whole plane
with 129 rows, four planes of 228.6 MiB plus a 1.5 MB ring, and it is the saving
the plan predicted rather than a new one.

at 13 frames in flight the peak falls 651.5 to 563.0 MiB, 13.6%, while the
scratch-free control is flat at 194.4 MiB. peak rss is a high-water mark and moved
by under 0.3 MiB across runs, so that column is solid.

the wall column is not. the control's own wall moved 17.6 to 14.4 ms at one frame
and 240.5 to 217.9 ms at thirteen with no code change between the two sessions,
which is a larger drift than `Deblur`'s 0.5% at thirteen frames, so this probe
cannot resolve a throughput change at this size.

the full suite can, because the ratio against the reference absorbs the drift.
`docs/BENCH.md` moved the deconvolution from 12.81x to **14.95x** and the unsharp
mask from 19.59x to **21.12x** across the run that added the ring, so the plugin
side takes about 14% and 7% less time against the same reference workload, since
a ratio that rises from 12.81x to 14.95x is a time ratio that falls by 14%. that
is the throughput benefit the review asked for, and it arrives with the memory one.

## correctness

the twelve case Deblur hash harness reports the same hashes as before the change,
on 8, 16 and 32 bit planes and both `NaN` pages. `cargo test --locked` passes 96
unit and 11 golden tests, including one that compares a ring-sized `temp` against
a whole-plane one bit for bit at six shapes and five sigmas, which covers the reuse
case, the whole-plane case and the reflected borders. `cargo clippy --all-targets
--locked -- -D warnings` and `cargo fmt --check` are clean and
`tests/check-nimages.py` passes its 2150 checks.

what the earlier plan asked for and this does not do: two, four and eight
concurrent full frames at the same cache budget, and a largest-to-smallest frame
sequence through one workspace. the pool grows and a smaller frame reuses the
larger frame's buffers, and the ring is sized per blur rather than per workspace,
so the sequence case is exercised by the size boundary test rather than measured.
