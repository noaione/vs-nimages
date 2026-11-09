# the asynchronous frame request path

status: landed. all five filters already answered `Initial` with a request and
returned no frame; this plan records the audit, adds the gates that hold the
contract, and keeps the probe that measures what the path buys. no frame was
requested differently to land it.

## current state

the filter contract is two-phase:

| activation | work |
| --- | --- |
| `Initial` | request input `n` with `request_frame_filter`, return `Ok(None)` |
| `AllFramesReady` | fetch input `n`, build the output, return it |
| `Error` | report the input failure |

every filter follows it: [Deblur](../../src/filters/deblur.rs#L179), [Levels](../../src/filters/levels.rs#L226),
[PeakStats](../../src/filters/peak_stats.rs#L139), [PeakGrayShades](../../src/filters/peak_gray_shades.rs#L113)
and [Posterize](../../src/filters/posterize.rs#L245). `Initial` does no frame
work, so the core can keep another frame moving while this one waits.

the five filters use the default `FilterMode::Parallel`. `GpuNode` now delegates
`FILTER_MODE` from the wrapped filter ([filters/mod.rs](../../src/filters/mod.rs#L300)),
so the wrapper cannot silently force a different mode.

the GPU paths split the same way as the host paths:

- `Deblur(gpu=1)` and the pointwise maps record a submission and return a frame
  with its producer pair. the worker does not wait for the device.
- `PeakStats`, `PeakGrayShades` and `Posterize(method=1)` wait for the histogram
  readback at [histogram.rs](../../src/gpu/histogram.rs#L366). the wait is part of
  the contract: the properties and the Lloyd solve need the numbers before the
  frame can be returned.

## what was measured

`tools/async_probe.py` is the repeatable probe. it builds a fresh node for every
sample, warms frame 0, submits frames `1..k` with `get_frame_async`, then
consumes every result. the reported value is the median of fresh nodes per
cell. it also compares one asynchronous frame with the same frame pulled
synchronously, every plane and every property.

the run below is `GRAY8`, 1280x720, cache 512 MB, median of two fresh nodes,
on the M4 host:

| filter | engine | 1 frame | 4 in flight | 8 in flight |
| --- | --- | ---: | ---: | ---: |
| `Deblur` | host | 43.8 ms | 12.3 ms | 8.2 ms |
| `Deblur` | device | 7.1 ms | 6.1 ms | 5.2 ms |
| `Levels` | host | 0.22 ms | 0.07 ms | 0.06 ms |
| `Levels` | device | 0.37 ms | 0.20 ms | 0.21 ms |
| `PeakGrayShades` | host | 0.22 ms | 0.09 ms | 0.07 ms |
| `PeakGrayShades` | device | 0.47 ms | 0.31 ms | 0.26 ms |
| `PeakStats` | host | 0.22 ms | 0.11 ms | 0.08 ms |
| `PeakStats` | device | 0.46 ms | 0.35 ms | 0.22 ms |
| `Posterize` | host | 0.31 ms | 0.11 ms | 0.07 ms |
| `Posterize` | device | 0.36 ms | 0.26 ms | 0.17 ms |

the host `Deblur` is the clearest case: 5.3x from one frame to eight, so the
request path is not serialising the worker. the device `Deblur` stays near 5.2
to 7.1 ms because the GPU queue, not the request path, is the limit. the
one-frame device analyzer row carries the readback wait; the same row at four
and eight in flight falls to 0.26 to 0.31 ms because the wait overlaps with
other frames.

the exactness check compared every asynchronous frame with the synchronous
frame of a second node. no case reported a mismatch.

## what the gates hold

`tests/check-nimages.py` has a host section, `asynchronous frame requests`, that
builds a second node and requests frames `3, 0, 2, 1` for all five filters. it
compares pixels and properties against the synchronous node. the device section
adds eight asynchronous device frames: four `Posterize(method=1, gpu=1)` and
four `Levels(use_props=True, gpu=1)` frames through `std.GPUDownload`, which
is the per-frame table path under a shared pool.

the run after the change reports **2232 checks passed**, up from 2184. the 48
new checks are 40 host asynchronous frames plus eight device frames.

## decisions

**no prefetch.** a filter has a strict spatial dependency on input `n`; requesting
`n + 1` would only fill the frame cache with frames the graph has not asked for.
the core already has the next frame's `Initial` callback when the caller pulls a
sequence.

**the readback wait stays.** `PeakStats`, `PeakGrayShades` and
`Posterize(method=1)` cannot return a frame without the histogram. moving the
wait to another thread would move the latency, not remove it.

**no `release_frame_early`.** the frame context holds the input only until
`AllFramesReady` returns, and an output that shares the input's planes already
keeps those planes alive through the frame's own reference. the call would add a
second ownership rule for no measured cache win.

## files

| file | role |
| --- | --- |
| `src/filters/mod.rs` | `GpuNode::FILTER_MODE` delegation |
| `src/filters/{deblur,levels,peak_stats,peak_gray_shades,posterize}.rs` | the two-phase `get_frame` bodies |
| `src/gpu/histogram.rs` | the analyzer readback wait |
| `tests/check-nimages.py` | the host and device asynchronous gates |
| `tools/async_probe.py` | the in-flight probe and its exactness check |

## done when

- all five filters pass the asynchronous host gate, including properties.
- the device gate passes on a device host, or prints its skip reason.
- `tools/async_probe.py --device` reports no exactness mismatch.
- the probe's host `Deblur` rate rises with more frames in flight; a flat rate
  sends the plan back to the request path.
