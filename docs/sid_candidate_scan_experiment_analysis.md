# SID Candidate Scan Experiments

## Environment

- Host: `8.160.171.231:1030`
- Repository: `/mnt/workspace/aop_lab/collision_exp/all_item_candidate_raw/TorchEasyRec`
- Branch: `perf/sid-candidate-scan`
- Baseline commit: `d8e1e0096ff2da8c3e89061ed22b138a584b1aec`
- Input: `exp_single/part-0.parquet`
- Input rows: 42,864,043
- Input size: 92,428,460,572 bytes (about 87 GiB)
- Schema: `id: string`, `codes: list<int64>`, `candidate_codes: list<int64>`
- Machine: 32 logical CPUs, 128 GiB RAM, no swap
- Initial free disk: 119 GiB (`/mnt/workspace` was 98% used)

All timing runs use `--rate_only` because the target is candidate loading and
matching. This retains planning, candidate scanning, and collision resolution
while avoiding about 9 GiB of unrelated output per run. Any non-rate-only trial
must use a unique output directory and remove it after metrics are recorded.

## Experiment 0: Baseline

Command derived from repository `test.sh`:

```bash
PYTHONPATH=. python -m tzrec.tools.sid.resolve_sid_collisions \
  --input_path 'exp_single/*.parquet' \
  --codebook 8192,8192,8192 \
  --max_items_per_codebook 5 \
  --strategy candidate \
  --item_id_field id \
  --rate_only
```

Results:

| Metric                        |                  Baseline |
| ----------------------------- | ------------------------: |
| End-to-end elapsed            |                 713.836 s |
| User CPU time                 |                 479.555 s |
| System CPU time               |                 132.082 s |
| Average CPU                   | 85.68% of one logical CPU |
| Candidate interval throughput |              72.3K rows/s |
| Estimated candidate scan      |               about 593 s |
| Peak observed RSS             |              about 11 GiB |
| Relocated                     |                 1,403,836 |
| Unresolved                    |                         0 |
| Raw/final collision buckets   |               210,170 / 0 |

The candidate pass consumed about 83% of elapsed time and remained close to one
busy CPU. The first experiment will replace object-array sorting and repeated
binary searches with pandas' reusable hash index. It does not assume stable row
positions and preserves the existing duplicate-ID representative/broadcast
behavior.

## Experiment 1: Reusable hash index

Commit: `64cf94460253bcddeef9990505bd7d3787b8d7a8`

Change under test:

- Factorize the requested overflow item IDs once.
- Build one reusable pandas hash index over the unique IDs.
- Match each source batch with `Index.get_indexer()` instead of repeatedly
  binary-searching an object array of sorted string IDs.
- Preserve the existing semantics: the last source occurrence supplies the
  candidate and that candidate is broadcast to duplicate overflow rows.

The expected lookup complexity changes from approximately `O(N log M)` string
comparisons to expected `O(N + M)` hash work, where `N` is the input row count
and `M` is the overflow row count. The Parquet data and command are unchanged
from Experiment 0 so the two elapsed times are directly comparable.

Results:

| Metric                         |     Baseline |     Hash index |             Change |
| ------------------------------ | -----------: | -------------: | -----------------: |
| End-to-end elapsed             |    713.836 s |      464.519 s |            -34.93% |
| User CPU time                  |    479.555 s |      344.301 s |            -28.20% |
| System CPU time                |    132.082 s |      120.231 s |             -8.97% |
| Average CPU                    |       85.68% |        100.00% |          +14.32 pp |
| Candidate effective throughput | 72.2K rows/s |  120.7K rows/s |            +67.23% |
| Estimated candidate scan       |      593.7 s |        355.0 s |            -40.20% |
| Peak observed RSS              | about 11 GiB | about 11.1 GiB | no material change |

The complete collision statistics are identical: 42,864,043 total items,
210,170 raw collision buckets, 1,403,836 relocated items, no unresolved items,
no final collision buckets, and a maximum final bucket size of five.

The measured end-to-end saving was 249.317 seconds. The candidate scan saving
estimated from progress checkpoints was 238.674 seconds, which accounts for
95.7% of the wall-time improvement. This confirms that the changed lookup is
on the measured bottleneck. The aggregate statistics did not change.

This first A/B is not a fully controlled cold-cache comparison. Experiment 0
populated the page cache before Experiment 1: even the unchanged first SID pass
rose from 3.15M to 4.90M rows/s, while physical read bytes barely changed during
Experiment 1. Therefore 34.93% is the observed run improvement, not an estimate
attributable solely to the hash index. A warm-cache baseline control is required
before accepting the isolated speedup.

## Experiment 0b: Warm-cache baseline control

Run the unchanged baseline commit again after Experiment 1, using the same
input, command, and page-cache state. This controls for the largest known
environmental difference without dropping the host-wide cache or affecting
other workloads.

Results:

| Metric                         | Warm baseline |    Hash index |   Change |
| ------------------------------ | ------------: | ------------: | -------: |
| End-to-end elapsed             |     599.451 s |     464.519 s |  -22.51% |
| Total CPU time                 |     598.698 s |     464.532 s |  -22.41% |
| Average CPU                    |        99.87% |       100.00% | +0.13 pp |
| Candidate effective throughput |  89.5K rows/s | 120.7K rows/s |  +34.94% |
| Estimated candidate scan       |       479.1 s |       355.0 s |  -25.89% |

The warm baseline produced the same complete collision statistics as both prior
runs. The controlled comparison therefore shows a 134.932-second end-to-end
saving (1.290x throughput) with nearly identical cache and CPU utilization. The
candidate phase accounts for about 124.0 seconds of that saving. This is the
preferred estimate of the hash-index improvement; the 34.93% cold-to-warm wall
reduction in Experiment 1 overstates the isolated code effect.

Setup note: two worktree launches stopped before reading any data because
gitignored generated protobuf bindings were absent (3.668 and 3.692 seconds).
The main and baseline commits do not differ under `tzrec/protos/`; the same
generated bindings were copied recursively into the scratch worktree before the
measurement was restarted. Neither failed launch created result data.

## Experiment 2a: Parquet decoding scalability microbenchmark

Before changing the production reader, a read-only microbenchmark used row
groups `[0, 96)` from the same file: 3,145,728 rows, `batch_size=100000`, and
only `id` plus `candidate_codes`. Every run verified the row count. The cache
was hot, so this isolates Parquet decode and Arrow materialization rather than
physical disk reads.

Single-reader results, alternating the setting in one process:

| Mode                | Mean elapsed |    Throughput | Average CPU cores |
| ------------------- | -----------: | ------------: | ----------------: |
| `use_threads=False` |     19.216 s | 163.7K rows/s |              1.00 |
| `use_threads=True`  |     15.981 s | 196.8K rows/s |              1.02 |

Arrow's flag reduced elapsed time by 16.83%, but the dominant candidate column
is one Parquet leaf column, so it did not materially increase CPU parallelism.

Independent readers were then assigned continuous, disjoint row-group ranges:

| Reader workers | Elapsed |      Throughput | Average CPU cores | Decode speedup |
| -------------: | ------: | --------------: | ----------------: | -------------: |
|              2 | 9.846 s |   319.5K rows/s |              1.84 |          1.95x |
|              4 | 4.975 s |   632.3K rows/s |              3.19 |          3.86x |
|              8 | 2.484 s | 1,266.4K rows/s |              7.24 |          7.74x |

This second table is an upper-bound decode test, not an end-to-end result. It
does not execute hash matching, `take`, candidate extraction, or ordered result
updates, and it validates coverage rather than candidate contents. It proves
that independent row-group readers can use multiple cores; a production version
must merge updates in global input order to preserve last-source-wins behavior.

## Experiment 2b: Arrow threaded full pipeline

Commit: `d0b8766b4fcb92b261c28288b4729b4dc73aba01`

After the controlled hash comparison, enable PyArrow's ordered threaded column
decoding for this tool's Parquet readers. `candidate_codes` occupies about
39.0 GiB compressed, while `id` occupies only about 0.35 GiB. The optimized run
still consumed one full logical CPU and performed almost no physical disk I/O,
so decode/conversion work is the next bottleneck. `iter_batches` retains batch
order when `use_threads=True`, so the duplicate-ID last-source-wins behavior is
unchanged.

Results:

| Metric                         |    Hash index | Hash + Arrow threads |   Change |
| ------------------------------ | ------------: | -------------------: | -------: |
| End-to-end elapsed             |     464.519 s |            434.214 s |   -6.52% |
| User CPU time                  |     344.301 s |            345.412 s |   +0.32% |
| System CPU time                |     120.231 s |             90.886 s |  -24.41% |
| Average CPU                    |       100.00% |              100.48% | +0.48 pp |
| Candidate effective throughput | 120.7K rows/s |        135.3K rows/s |  +12.10% |
| Estimated candidate scan       |       355.0 s |              316.7 s |  -10.79% |

The full statistics remain identical to all prior runs. Arrow threading saves
30.305 seconds end to end (1.070x) and 38.317 seconds in the estimated candidate
phase. It mainly reduces system/decoding overhead and still averages only about
one logical CPU. The change is useful and low risk, but it is not a true
multi-core implementation.

## Final comparison and recommendation

The preferred controlled comparison is the warm baseline against the final
version:

| Metric               | Warm original | Final version |           Change |
| -------------------- | ------------: | ------------: | ---------------: |
| End-to-end elapsed   |     599.451 s |     434.214 s | -27.56% / 1.381x |
| Candidate scan       |       479.1 s |       316.7 s |          -33.89% |
| Candidate throughput |  89.5K rows/s | 135.3K rows/s |          +51.26% |

Keep both measured changes. The reusable hash index is the primary fix,
accounting for about 82% of the controlled end-to-end saving. Ordered Arrow
threaded decoding contributes a smaller additional improvement without changing
the observed aggregate statistics.

Do not claim that the final implementation uses the 32-core host effectively.
The next optimization should be a separate change that gives continuous,
disjoint row-group ranges to four independent Parquet reader threads. Workers
must not update the shared candidate matrix concurrently. Their matched updates
must be merged in original input order so duplicate item IDs retain the existing
last-source-wins behavior. The 96-row-group microbenchmark shows that raw decode
can scale to 3.86x with four readers, but a complete implementation needs output
equivalence tests, bounded in-flight memory, multi-file coverage, and a full
end-to-end measurement.

All full runs used `--rate_only` and created only log files. Available disk was
112 GiB after the final run; no generated result directory required cleanup.

## Validation scope

- All four complete runs produced the same six collision-stat fields.
- The 37 SID collision unit tests pass, including duplicate item IDs and the
  last-source-candidate broadcast behavior.
- A differential check over 500 randomized batches for both integer and string
  IDs matched the previous lookup and duplicate broadcast results exactly.
- All three `ParquetReaderTest` cases pass, including explicit propagation of
  `use_threads`; pre-commit checks pass for every modified tracked file.
- Pyre reported no critical type errors, although its installed CLI emitted a
  version-option compatibility warning.

`--rate_only` does not materialize the 9 GiB item map, so the real-data runs do
not constitute a bit-for-bit comparison of every output row. Result equivalence
is supported by the focused unit/differential tests, while the full input
provides aggregate-stat and performance validation.
