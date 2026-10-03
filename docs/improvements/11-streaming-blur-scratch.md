# streaming blur scratch

status: not implemented. larger implementation, explicit memory target.

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

status: not implemented. the objective is real and now measured, 91% of peak rss
at a small cache; the change is the largest in the review and the plan is below.

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
workspace dominates `Deblur`'s peak at a small cache. this table says nothing
about throughput, since it is one pull per case. the decision now rests on the
implementation cost and its wrong-pixel risk, not on whether the saving exists.

the reason this is deferred rather than attempted and reverted: it cannot be
landed as a local change to `blur` alone. `blur_row_avx2` writes one horizontal
row, and the vertical pass needs a window of `2 * radius + 1` of those rows per
output row, so either the producer runs ahead of the consumer or the vertical
loop has to drive the horizontal one. a ring that recomputed its window per
output row would pay the horizontal pass `2 * radius + 1` times, and at radius
3 that is seven times the filter's hottest loop; the alternative is a
slot-to-source-row map, which is the part that can silently read the wrong row.
this is the one candidate in the review whose failure mode is a wrong pixel
rather than a slower frame, so it needs its own round rather than the end of
this one.

the plan, in the order the review sets: replace only `temp`, keep the other four
planes and every candidate, mask and blend stage as they are; allocate the ring
with checked arithmetic and a fallible reservation, reusing it across frames
sized for the largest active kernel; track each ring slot's source row identity
rather than trusting modulo addressing; fall back to storing every row when the
plane is shorter than the kernel. then compare blur output bits for tiny, odd,
tall and large planes through all kernel lengths, and measure 1/2/4/8 concurrent
full frames at the same cache budget plus the largest-to-smallest sequence. the
unit and hash harnesses from 05 and 06 apply unchanged, because both preserve
the arithmetic order.
