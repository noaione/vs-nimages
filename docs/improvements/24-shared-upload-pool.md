# a shared upload pool for per-frame tables

status: landed. `Levels(use_props=True, gpu=1)` and `Posterize` with a per-frame
table no longer create an `ExecPool` for every frame; the table copy records on
the map's pool. a fixed table and the tap tables still create a pool at
`create`, where the cost is paid once.

## current state

the map kernel binds a table as a `GRAY32` device frame. a table that changes per
frame is copied from a host staging buffer into a fresh frame by
`upload_words`:

| piece | before |
| --- | --- |
| staging buffer | allocated per table |
| table frame | allocated per table |
| `ExecPool` | **created per table** |
| copy | recorded on that temporary pool |

the pool creation was the avoidable part. [gpu/mod.rs](../../src/gpu/mod.rs#L304)
created a compute pool, submitted the copy, and dropped the pool before
`Map::render` recorded the map dispatch. the map already owns a pool at
[gpu/map.rs](../../src/gpu/map.rs#L124), so the upload can use it.

`Levels` rebuilds a table per frame only under `use_props=True`. `Posterize`
rebuilds one per frame when `bits` comes from `NImagesGrayShades` or when
`method=1` solves the levels from the frame histogram. a fixed table pays the
upload at `create` and keeps the path unchanged.

## the change

| function | role |
| --- | --- |
| `upload_words` | creates one pool, then calls the shared path |
| `upload_words_with_pool` | the copy path, on a caller-owned pool |
| `Device::upload_words` | the map instance's pool |
| `Map::upload_words` | the upload a filter calls |
| `device_table(core, engine, table)` | uploads a `MappingTable` through the map |

the map's pool is `ExecPool`, which is `Send + Sync` and designed for the
filter's worker threads. an upload recording finishes before `Map::render`
starts its own recording on the same thread, so the pool never holds two
recordings for one frame. the table frame's producer pair carries the ordering
from the copy to the map dispatch, exactly as it did with the temporary pool.

`Deblur`'s tap tables still call `upload_floats`, which uses the one-shot
`upload_words`; they are created once per instance, so there is no per-frame
pool to reuse.

## the measurement

`tools/histogram_hash.py --bench --format GRAY8 --frames 20`, 4096x3000, on the
M4 host. the before column is one run before the change; the after column is
five runs after it, so the spread is visible rather than hidden:

| row | before | after | note |
| --- | ---: | ---: | --- |
| `device/lloyd` | 7.3 ms | 4.2, 4.7, 6.7, 6.6, 6.3 ms | median 6.3 ms |
| `device/even` | 2.0 ms | 2.1, 2.1, 2.8, 1.9, 1.8 ms | fixed table, unchanged path |

the machine moves by roughly a millisecond between runs, so the table does not
claim a tighter effect than that. the direction is consistent: the per-frame
table row is lower after the change, and the fixed-table row is not.

## exactness

the table bytes and the producer-pair ordering are unchanged; only the pool that
records the copy is different. the gates:

| gate | result |
| --- | --- |
| `tools/map_hash.py --device` | byte-identical on every case |
| `tools/histogram_hash.py --device` | identical properties and pixels |
| `tests/check-nimages.py` with `NIMAGES_REQUIRE_DEVICE=1` | 2232 checks passed |
| device asynchronous `Posterize(method=1)` gate | 4 frames match `gpu=0` |
| device asynchronous `Levels(use_props=True)` gate | 4 frames match `gpu=0` |

the device asynchronous gates are the important ones: four frames on one map
instance share the pool, and each per-frame table still reaches its own map
dispatch.

## files

| file | role |
| --- | --- |
| `src/gpu/mod.rs` | `upload_words_with_pool`, `Device::upload_words` |
| `src/gpu/map.rs` | `Map::upload_words` |
| `src/filters/mod.rs` | `device_table` takes the map |
| `src/filters/levels.rs` | creates the map before the first table |
| `src/filters/posterize.rs` | creates the map before the first table |
| `docs/improvements/15-vulkan-device-path.md` | the device path entry |

## done when

- a fixed table still uploads once at `create`.
- a per-frame table uploads on the map's pool and creates no pool.
- both device hash gates report no difference.
- the asynchronous device gate sees four per-frame tables on one instance.
