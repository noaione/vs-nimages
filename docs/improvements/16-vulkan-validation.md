# validating and measuring a device kernel

status: in part landed. `tools/deblur_hash.py` and its `--device` mode are in
the tree, with 39 cases after M5 added RGB, YUV and the depth variants; the
validator gate and the fan-in measurement are still open. the plan is the gate
for [17](17-vulkan-deblur.md), [18](18-vulkan-pointwise-maps.md) and
[19](19-vulkan-histogram-analyzers.md).

## problem

a device path adds three things the current harness cannot see: a second
implementation of the same pixels, a residency contract the caller can break, and
a cost that moves with the bus rather than with the frame. the existing suite
covers the first only if the two implementations are compared on the same frame,
and covers neither of the other two.

the rule from [04](04-optimization-review.md) still holds: no candidate is
accepted by widening a tolerance or by editing a golden expectation. a device
kernel that cannot meet the CPU's answer is rejected, not reconciled.

## what a device result has to prove

| # | property | how |
| ---: | --- | --- |
| 1 | `gpu=1` equals `gpu=0` on the same frame | fetch both, compare plane bytes |
| 2 | integer outputs are byte-exact | `u8` and `u16` every sample |
| 3 | `f32` outputs are equal by bits | `to_bits()`, not `==`, so NaN and `-0.0` are covered |
| 4 | Deblur meets its frozen fixture tolerance at minimum | `tests/fixtures/deblur.json` |
| 5 | properties are identical | names, types, array order, empty arrays |
| 6 | geometry is unchanged | odd widths, padded rows, 1 px, 24 MP, variable dimensions |
| 7 | requests are independent | repeated, out-of-order, concurrent |
| 8 | a host without a device fails cleanly | named error, CPU path unchanged |
| 9 | residency is what the instance declared | a `gpu=1` node yields device frames |

item 3 is stricter than the fixture tolerance on purpose: the CPU kernels are the
reference and a device port that only lands inside the tolerance has not been
shown to reproduce them. [17](17-vulkan-deblur.md) closes with a recorded
deviation when the driver's `sqrt` differs from the host's, and that deviation
is a number in the report, not a change to the fixture.

## the device gate in the suite

`tests/check-nimages.py` grows a device section that runs after the CPU checks
and is gated on a usable device, not on an argument:

```py
devices = core.vulkan_devices               # list of dicts, one per physical device
usable = [d for d in devices if d["usable"]]
if not usable:
    print("gpu: skipped, no usable Vulkan 1.4 device")
    return 0                                # a skip is exit 0, like the sibling
core.set_vulkan_device(usable[0]["index"])  # pin the device, so two runs compare
```

on this host that list has two entries and only the RTX 3050 is usable, because
the Intel UHD reports 1.2.201 and the core requires 1.4. the skip therefore has a
reason string, and the harness prints it.

the rest of the section is a loop over the filters that have a `gpu` argument:

```py
host = build_chain(core, gpu=0)
dev = core.std.GPUDownload(build_chain(core, gpu=1))
for n in frames:
    assert_planes_equal(dev.get_frame(n), host.get_frame(n))
    assert_props_equal(dev.get_frame(n), host.get_frame(n))
```

`build_chain` is the same source and the same arguments in both cases, so the
only difference is where the kernel ran.

## the Deblur hash harness

the 12-case Deblur hash harness lives in `.tmpbuild/deblur_output_hash.py` and is
not committed, which [04](04-optimization-review.md) lines 141-145 already calls out as
the reason the exactness claims for 05, 06 and 12 are not reproducible from
the tree. the device port needs it more than they did, because a shader changes
the arithmetic environment rather than the loop shape.

promote it to `tools/deblur_hash.py` before the first kernel:

- 12 cases: both methods, `u8`/`u16`/`f32`, odd width, a 1 px frame and a 129-tap kernel,
  a variable-dimension clip, and a frame whose stride padding is non-zero.
- one hash per case over the output plane bytes, printed as a table sorted by
  case, so two builds compare with `diff`.
- a `--write` mode that records the table beside a source revision, and a
  `--check` mode that fails on a mismatch, in the style of `tools/golden.py`.
- the frozen fixture replay stays as it is. the hash is the tighter gate for
  order-preserving changes and the fixture is the contract with the reference.

## the measurements

two numbers per filter, because they answer different questions.

**latency** is one frame's cost on a quiet machine, which is what the current
`per page` column reports. it needs a fresh node per request: the frame cache
serves a repeated `get_frame(0)` and the median then measures the cache.

**aggregate** is frames per second through a temporal fan-in, which is the shape
a real graph has. `03-deblur-filter.md:564-581` is the protocol: 13 frames
through `std.AverageFrames`, one pull, each frame reporting its own stage line.
that probe measured 9.48x reported work but 2.7x aggregate throughput on the CPU,
so the CPU aggregate is the honest baseline and a device frame competes against
that rate, not against 12 times one frame.

each run records:

| field | why |
| --- | --- |
| source revision and library hash | proves which build ran, the way [04](04-optimization-review.md) lines 118-127 requires |
| host, CPU, device, driver, Vulkan version, VapourSynth version | the numbers are host-specific |
| page and geometry | 12 MP and 24 MP are different arguments |
| arguments, including `iterations` and `radius` | the default is not the only case |
| latency median and spread, 5 to 9 repeats | the machine moves 20 to 40% between runs |
| aggregate through the fan-in | the real use case |
| upload and download, measured alone on the same frame | the bus is the comparison |
| peak RSS and peak VRAM | a device frame is not free, and neither is the workspace |
| exactness table | items 1 to 5 above |

the bench alternates CPU and device in one window rather than running one today
and the other tomorrow, and a shader A/B is two processes because
`std.LoadPlugin` refuses a second library for the same plugin id.

## what lands where

| change | file |
| --- | --- |
| device section in the validator | `tests/check-nimages.py` |
| Deblur hash harness | `tools/deblur_hash.py` (new), promoted from `.tmpbuild/` |
| device bench | `tools/bench.py --device`, plus a `gpu` column in `docs/BENCH.md` |
| results | a new section in `docs/BENCH.md`, not a rewrite of the existing block |
| launch geometry, index arithmetic, tap order | `#[cfg(test)]` tests in `src/gpu/shared.rs`, which names no binding type |

`--device` compares `gpu=0` against `gpu=1` on the same pages and reports both
latency and aggregate. it does not replace the reference comparison, which has no
device side.

## continuous integration

the device section is not a wheel gate. a runner without a usable device prints
the skip reason and exits 0, which keeps the existing Windows and Linux jobs
green. a job that has a device sets an environment flag and treats a skip as a
failure, so a broken device path cannot hide as "no device here".

no new wheel variant is needed: the device path is the same binary, and a host
without Vulkan 1.4 resolves the failure at filter creation.

## done when

- the validator has a device section that skips with a reason and compares the
  filters that have `gpu` against `gpu=0`.
- `tools/deblur_hash.py --check` reproduces the committed table on the CPU build.
- `tools/bench.py --device` reports latency, aggregate and transfer cost for the
  filters that have a device path.
- every result lands in `docs/BENCH.md` with the fields above.
