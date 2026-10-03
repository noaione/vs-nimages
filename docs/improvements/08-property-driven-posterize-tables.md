# property-driven posterize tables

status: rejected on measurement. target repeated even tables, not Lloyd or
Levels.

## current cost

[`src/filters/posterize.rs`](../../src/filters/posterize.rs) sets `resolved`
only for fixed bits, even method, and a declared format. property-driven even
Posterize therefore calls `resolve_from_bits` per frame. above 8 bits this
allocates and computes a native-range LUT using integer division per entry.

8-bit resolution already copies a compile-time table. a fixed-format,
fixed-bit even filter already borrows its resolved table. neither is the main
target. Lloyd tables depend on frame contents and cannot share this cache.

## one experiment

reuse immutable tables keyed by `(sample_depth, bits)` within the filter.
resolve `NImagesGrayShades` and validate its length on every frame before lookup.
the table depends on the inferred bit count, not the individual shade values.
retain the full-depth identity return before constructing any table.

for a known sample depth, first compare eager immutable tables with lazy slots.
at 16 bits, the fifteen non-identity tables occupy 1.875 MiB of u16 entries
before container overhead. eager creation trades startup work for lock-free
reads; lazy creation avoids allocating unused bit counts.

for variable-format clips, include depth in every key and use bounded slots
for depths 8 through 16. publish an immutable shared table only after successful
fallible construction. keep locks short, avoid cloning the Vec on cache hits,
and preserve allocation errors. a failed build cannot leave a slot marked ready.

## why this differs from the rejected Levels cache

the old Levels candidate keyed per-frame black, white and gamma, which varied
across a volume. this candidate has at most fifteen non-identity bit counts at
one 16-bit depth. repeated shade counts that infer the same bits reuse exactly
the same table. measure actual key frequencies rather than assuming a hit rate.

## parity checks

compare every entry against `posterize_lut_u16` for each depth/bit pair. test
changing shade counts, two different counts that infer the same bits, malformed
and missing properties, and alternating native depths. exact pixels, properties,
errors, and concurrent out-of-order requests stay unchanged.

the current Posterize trace measures `map`, but table resolution occurs outside
that stage. use pure builder timing or benchmark-only instrumentation plus total
frame time; a stable `map` time does not disprove the table saving.

## measure and decision

compare warm repeats, alternating bits, all cold keys, fixed bits as a control,
and a depth-varying clip. cover Gray16, RGB48 and YUV420P16, small frames where
table cost dominates, and full pages. report creation time, hit/miss counts,
retained bytes, total latency and concurrent throughput.

skip this if callers use fixed bits, or if resolution is negligible at their
page size. choose eager or lazy publication only after measuring startup and
contention; do not use an unbounded global cache.

## result

status: rejected on measurement. resolution is negligible at page size, and the
cache's synchronization is not worth 1.3%.

a probe at `.tmpbuild/lutbench` times `posterize_lut_u16` itself, allocation
included, against a walk over a 2903x4128 plane with the table it built. the
walk is the work the table is for, so the share is the most a cache could
remove:

| case | table build | plane walk | share |
| --- | ---: | ---: | ---: |
| bits 4, 16-bit | 0.197 ms | 15.44 ms | 1.26% |
| bits 8, 16-bit | 0.196 ms | 14.94 ms | 1.29% |
| bits 15, 16-bit | 0.223 ms | 15.24 ms | 1.44% |
| bits 4, 12-bit | 0.012 ms | 16.00 ms | 0.08% |
| bits 4, 10-bit | 0.004 ms | 17.00 ms | 0.02% |
| bits 4, 8-bit | 0.001 ms | 13.68 ms | 0.01% |

the 8-bit tables are already compile-time copies, so a cache can only help 10
bits and above, and there the whole saving is 1.3% of the mapping pass at 16
bits. that is inside the run to run spread of the integrated benchmark, so no
integrated change is claimed either way.

the review's own instruction is to skip this when resolution is negligible at
the caller's page size. it is, at every depth measured. the per-filter cache
would also have to hold `Vec<u16>` behind a lock and be keyed on both depth and
bits, which is the retained memory and synchronization the review asked to
measure before adopting. leave `resolve_from_bits` called per frame.
