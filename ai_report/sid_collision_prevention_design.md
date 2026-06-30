# SID Codebook Collision Prevention Design

## Goal

Add an offline collision-prevention step for Semantic ID (SID) generation. The step runs after SID batch prediction and guarantees that no final SID/codebook bucket contains more than a configured number of items, for example 5. It must support CSV, Parquet, and MaxCompute inputs/outputs and should preserve both the original SID and the reassigned SID for auditability.

This design is based on `/mnt/fangtinglin/al_sid/SID_generation/ID_collision.sql`.

## SQL Review

The reviewed SQL implements the right high-level policy:

1. Build `item_codebook_info(item_id, origin_codebook, codebook, index)`.
2. Keep the first `capacity` items per original `codebook_index`.
3. Treat the remaining items as overflow.
4. Explode each overflow item's nearest candidate SIDs (`sorted_index`) into prioritized candidate rows.
5. Iteratively assign overflow items to non-full candidate SIDs, keeping at most `capacity` items per final SID.

The production implementation should not copy the SQL literally. It has several gaps: manual loop control, in-place overwrites of staging tables, non-deterministic `rand()` ordering, ambiguous candidate string encoding, a few alias issues, no explicit convergence check, no diagnostics, and no support for local CSV/Parquet execution.

## Proposed Workflow

1. Train and export/evaluate the SID model as usual.
2. Run offline prediction with `python -m tzrec.predict`, reserving the item id and emitting SID codes.
3. If reassignment is needed, explicitly provide candidate SID codes either from model prediction or from a separate candidate table. Candidate generation/output is not enabled by default.
4. Run a new collision tool, proposed as `python -m tzrec.tools.sid.collision_prevention`.
5. Write the final mapping table:

```text
item_id, origin_codebook, codebook, index
```

`origin_codebook` is the raw SID from the model. `codebook` is the capacity-safe SID after reassignment. `index` is the 1-based slot number inside the final SID bucket.

## Input and Output Schemas

The collision tool should normalize all inputs into two logical tables.

`raw_sid`:

```text
item_id: string or int64
origin_codebook: string, such as "111,222,333"
```

It should also accept SID prediction outputs where `codes` is `ARRAY<BIGINT>` / `list<int64>` or split columns like `code_0, code_1, code_2`, converting them to `origin_codebook` with a configurable delimiter. For CSV, codes must be stored as strings; nested arrays/lists are not supported in CSV output.

`candidate_sid` is optional only when the raw output already satisfies capacity. It has no implicit default source; if overflow rows exist and no candidate rows are supplied, the tool should fail with a clear error.

```text
item_id
origin_codebook
candidate_codebook
priority: int64       # lower is better
score: double         # optional distance or similarity
```

For compatibility with the existing SQL, the tool may also accept a compact string `sorted_index` column, but the long candidate schema above should be the canonical internal representation. It avoids delimiter ambiguity when a SID itself is a comma-joined tuple.

Primary output remains SQL-compatible:

```text
item_id
origin_codebook
codebook
index
```

An optional diagnostics output should include total items, raw collision buckets, final collision buckets, reassigned count, unassigned count, iteration count, and max final bucket size.

## Assignment Algorithm

Use one deterministic allocator for CSV/Parquet and MaxCompute.

1. Normalize every raw SID to a canonical string.
2. Initial assignment: group by `origin_codebook`; sort by a deterministic tie-breaker such as `hash(seed, item_id)`; keep the first `capacity` rows with `index = row_number`.
3. Mark all other rows as unassigned.
4. Expand or read candidate rows for unassigned items only.
5. On each iteration:
   - Remove candidates for already assigned items.
   - Remove candidates whose target `candidate_codebook` is already full.
   - Rank candidates by `(priority, score, hash(seed, item_id))`.
   - For each target codebook, select up to remaining capacity.
   - If an item is selected by multiple target codebooks, keep its best candidate.
   - Append accepted rows to `item_codebook_info` with the next available `index`.
6. Stop when all items are assigned, no progress is made, or `max_iters` is reached.

Default unassigned behavior should be `error`, because silently keeping an over-capacity SID defeats collision prevention. Optional policies can include `keep_original` for debugging and `drop` for analysis-only runs.

The allocator must enforce these invariants:

- every `item_id` appears at most once in the final table;
- every final `codebook` has `count(*) <= capacity`;
- `origin_codebook` is never overwritten;
- the same inputs and seed produce the same output.

## CSV and Parquet Backend

Local mode should use PyArrow-based readers and writers, consistent with existing `CsvReader`, `ParquetReader`, `CsvWriter`, and `ParquetWriter` behavior. The first implementation can be single-process and in-memory after batch loading, because the assignment requires global grouping and repeated joins. It should fail early with a clear message when estimated rows exceed a configured memory limit.

For larger local corpora, a later version can add external sorting or SQLite/DuckDB, but that should not be required for the initial feature.

Example:

```bash
python -m tzrec.tools.sid.collision_prevention \
  --input_path /path/to/sid_predict/*.parquet \
  --output_path /path/to/final_sid \
  --reader_type ParquetReader \
  --writer_type ParquetWriter \
  --item_id_field item_id \
  --code_field codes \
  --candidate_input_path /path/to/sid_candidates/*.parquet \
  --max_items_per_codebook 5 \
  --seed 2026
```

## MaxCompute Backend

MaxCompute mode should execute generated SQL through PyODPS instead of downloading the corpus. The script should create staging tables with a configurable temp prefix and lifecycle, then orchestrate the same loop used by the local allocator.

Recommended staging tables:

- `${prefix}_raw_sid`
- `${prefix}_candidate_sid`
- `${prefix}_assigned`
- `${prefix}_selected`
- `${prefix}_diagnostics`

The generated SQL should use deterministic hash ordering instead of `rand()`. It should expose `--dry_run_sql` to print SQL without executing it, and it should run `SELECT COUNT(*)` checks between iterations to detect convergence.

Example:

```bash
python -m tzrec.tools.sid.collision_prevention \
  --backend odps \
  --input_path odps://project/tables/raw_sid_table/ds=20260630 \
  --candidate_input_path odps://project/tables/sid_candidates/ds=20260630 \
  --output_path odps://project/tables/item_codebook_info/ds=20260630 \
  --max_items_per_codebook 5 \
  --temp_prefix tmp_sid_collision_20260630 \
  --odps_lifecycle 7
```

## Candidate Generation

Collision prevention quality depends on candidate coverage. The script should support two candidate sources in v1:

1. Existing candidate table, such as the reviewed SQL's `sorted_index_lv3`.
2. Explicitly enabled model-emitted KNN candidates, recommended.

For SID, the v1 candidate strategy is last-layer KNN. Keep the greedy prefix `codes[:-1]`, compute the top-k nearest final-layer code ids from the residual before the last quantization layer, and produce full candidate SID tuples by replacing only the last code. This matches the reviewed SQL's "last level codebook ID" assumption and avoids a full combinatorial search.

V1 supports only explicit candidate tables and explicitly enabled last-layer KNN candidate output. Beam search is out of scope; do not add beam-search APIs, config fields, or implementation paths in v1.

## Required `sid_model` Additions

The collision tool should remain a post-processing tool, not part of training loss. `sid_model` and the SID quantizers should add enough inference outputs to make high-quality post-processing possible.

Proposed additions:

- Add a shared candidate-output switch used by `SidRqvae` and `SidRqkmeans`; it must be disabled by default:
  - `enabled`: whether inference emits candidates; default behavior remains `codes` only.
  - `topk`: number of candidate SIDs per item.
  - `strategy`: v1 supports only `last_layer_knn`.
  - `target_layer`: default `-1` for the last layer.
  - `include_origin`: whether priority 0 is the original SID.
- Add quantizer APIs:
  - `get_codes(input)` already exists and remains the raw SID output.
  - `get_code_candidates(input, topk, strategy, target_layer)` returns `candidate_codes` with shape `[B, K, L]` and `candidate_scores` with shape `[B, K]`.
  - expose residuals before each layer internally so last-layer KNN does not recompute the whole pass twice.
- Add prediction outputs only when candidate output is explicitly enabled:
  - `codes`: `[B, L]`, current behavior.
  - `candidate_codes`: `[B, K, L]`.
  - `candidate_scores`: `[B, K]`.
- For CSV consumption, emit or convert SID codes as strings only. Split-code outputs such as `code_0`, `code_1`, ... may be accepted as input, but CSV outputs should store `origin_codebook` and `codebook` as canonical strings.
- Keep `max_items_per_codebook` out of `sid_model` and all SID protos. Capacity is a business/post-processing policy and must be passed only as a parameter to `python -m tzrec.tools.sid.collision_prevention`.

The existing `unique_sid_ratio` metric is useful but batch-local. Exact global collision statistics should be produced by the collision tool, not by `sid_model`.

## Implementation Plan

1. Add `tzrec/tools/sid/collision_prevention.py` with a backend-independent allocator interface.
2. Implement local CSV/Parquet mode using Arrow batch loading and writer reuse.
3. Implement MaxCompute mode as a SQL generator plus PyODPS executor.
4. Add candidate normalization utilities for `codes`, `code_0...`, long candidate rows, and compact string candidate lists.
5. Add SID quantizer/model candidate outputs behind a config flag.
6. Add unit tests for deterministic assignment, full-bucket filtering, duplicate item prevention, unassigned policy, and CSV/Parquet round trips.
7. Add gated MaxCompute tests using ODPS environment variables, following existing ODPS test patterns.

## Open Decisions

- Whether candidate output should be enabled from SID prediction CLI flags or from model export/predict config, while remaining disabled by default.
- Whether output should include extra audit fields by default or only in a diagnostics table.
