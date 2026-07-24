# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

r"""Offline best-effort SID collision resolution with TorchEasyRec-native I/O.

Item IDs are expected to be unique in the upstream prediction output. To avoid
blocking a large job on rare upstream violations, this tool does not perform a
full-input uniqueness check or remove duplicate rows. Duplicate rows remain
independent items in collision accounting and outputs. When candidate-strategy
overflow rows share an item ID, they reuse one candidate list matched for that
ID. Duplicate data should still be fixed upstream.

The runner retains the first capacity items in each SID bucket, attempts to
relocate overflow items using fixed-width last-layer candidates, delegates
placement to the pure NumPy core, and writes item-level and grouped SID results
through TorchEasyRec readers and writers. CSV encodes each SID/candidate column
as comma-separated codes and item-ID groups as JSON arrays because Arrow's CSV
writer cannot serialize list columns; ``candidate_codes`` is a flat
``topk * n_layers`` run split by the ``--codebook`` length.

The random strategy intentionally preserves the legacy deterministic baseline:
it draws with replacement from the full last-layer space. Placement skips an
item's origin, so an origin draw or a duplicate draw is not replaced. Candidate
generation is independent of the selected first-fit or iterative placement
policy.

The input is a Semantic-ID table from ``tzrec.predict`` (an ``item_id`` column,
a ``codes`` ``list<int>`` column, and -- for the default ``--strategy
candidate`` -- a flat ``candidate_codes`` ``list<int>`` column (``topk *
n_layers`` codes per item)). RQ-VAE also emits ``candidate_scores`` when
candidate output is enabled. This tool does not read those scores: candidates
are already sorted from best to worst, and both placement policies use that
column position as the priority.

Example::

    python -m tzrec.tools.sid.resolve_sid_collisions \
        --input_path 'sid_predict_output/*.parquet' \
        --codebook 256,256,256 --max_items_per_codebook 5 \
        --strategy candidate \
        --output_path sid_collision/map \
        --resolved_sid_groups_output_path sid_collision/resolved_groups

To also write the optional original-SID audit grouping, add
``--original_sid_groups_output_path sid_collision/original_groups``.

For CPU parallel arbitration on one machine, launch the same command with
``torchrun --standalone --nproc-per-node=8``. Rank zero owns all input and
output I/O; ranks exchange numeric collision shards through memory-bounded
PyTorch tensor transfers on a Gloo group.
"""

import argparse
import json
import os
from contextlib import closing, contextmanager
from dataclasses import dataclass
from functools import cached_property
from typing import TYPE_CHECKING, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc

from tzrec.datasets.dataset import create_reader, create_writer
from tzrec.utils.logging_util import ProgressLogger, logger
from tzrec.utils.path_util import check_path_conflict
from tzrec.utils.sid.collision import (
    CodebookItemGrouping,
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionResult,
    CollisionResolutionStats,
    CollisionResolver,
    CollisionShardResult,
    KnnCollisionResolver,
    build_original_item_grouping,
    build_resolved_item_grouping,
    generate_random_candidate_last_codes,
    overflow_band_bucket_mask,
    prepare_collision_plan,
)
from tzrec.utils.sid.collision_sharding import (
    CollisionBandShard,
    partition_collision_bands,
)
from tzrec.utils.sid.distributed_collision import (
    DistributedCollisionContext,
    DistributedCollisionWork,
    broadcast_collision_stats,
    build_collision_work,
    gather_collision_shard_results,
    initialize_distributed_collision,
    receive_collision_work,
    send_collision_work,
    synchronize_collision_workers,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver

if TYPE_CHECKING:
    from tzrec.datasets.dataset import BaseReader, BaseWriter

_MAP_WRITE_ROWS = 1_000_000
_GROUP_WRITE_ITEMS = 1_000_000
_CANDIDATE_BUFFER_BYTES = 128 * 1024 * 1024
_ARROW_LIST_OFFSET_MAX = int(np.iinfo(np.int32).max)


@contextmanager
def _rank_zero_io_environment():
    """Expose rank-zero Reader and Writer construction as single-process I/O."""
    names = ("WORLD_SIZE", "RANK", "LOCAL_WORLD_SIZE", "LOCAL_RANK")
    previous = {name: os.environ.get(name) for name in names}
    os.environ.update(
        {
            "WORLD_SIZE": "1",
            "RANK": "0",
            "LOCAL_WORLD_SIZE": "1",
            "LOCAL_RANK": "0",
        }
    )
    try:
        yield
    finally:
        for name, value in previous.items():
            if value is None:
                os.environ.pop(name, None)
            else:
                os.environ[name] = value


class _ItemIdLookup:
    """Map streamed item IDs to positions in a fixed requested-ID array."""

    def __init__(self, item_ids: np.ndarray) -> None:
        import pandas as pd

        group_indices, unique_ids = pd.factorize(item_ids, sort=False)
        representative_rows = np.full(unique_ids.shape[0], item_ids.shape[0])
        np.minimum.at(
            representative_rows,
            group_indices,
            np.arange(item_ids.shape[0], dtype=np.int64),
        )
        requested_rows = np.arange(item_ids.shape[0], dtype=np.int64)
        self._duplicate_requested_rows = requested_rows[
            requested_rows != representative_rows[group_indices]
        ]
        self._representative_requested_rows = representative_rows[
            group_indices[self._duplicate_requested_rows]
        ]
        self._unique_id_index = pd.Index(unique_ids)
        self._representative_rows = representative_rows

    def match(self, item_ids: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Return matching source rows and requested-ID positions."""
        group_indices = self._unique_id_index.get_indexer(item_ids)
        source_rows = np.flatnonzero(group_indices >= 0)
        return source_rows, self._representative_rows[group_indices[source_rows]]

    def broadcast_duplicate_targets(self, values: np.ndarray) -> None:
        """Copy representative values to duplicate requested-ID positions."""
        values[self._duplicate_requested_rows] = values[
            self._representative_requested_rows
        ]


def _candidate_chunk_rows(candidate_count: int) -> int:
    """Return a row count bounded by the candidate temporary-buffer limit."""
    row_bytes = max(candidate_count * np.dtype(np.int64).itemsize, 1)
    return max(_CANDIDATE_BUFFER_BYTES // row_bytes, 1)


@dataclass(frozen=True)
class ResolveSidCollisionsConfig:
    """Validated configuration for SID collision-resolution orchestration."""

    input_path: str
    output_path: Optional[str]
    original_sid_groups_output_path: Optional[str]
    resolved_sid_groups_output_path: Optional[str]
    reader_type: Optional[str]
    writer_type: Optional[str]
    batch_size: int
    progress_interval: int
    distributed_timeout_seconds: int
    item_id_field: str
    code_field: str
    candidate_codes_field: str
    layer_sizes: Tuple[int, ...]
    max_items_per_codebook: int
    strategy: str
    placement_policy: str
    random_num_candidates: int
    rate_only: bool
    odps_data_quota_name: str

    def __post_init__(self) -> None:
        if self.batch_size < 1:
            raise ValueError(f"batch_size must be >= 1, got {self.batch_size}.")
        if self.progress_interval < 1:
            raise ValueError(
                f"progress_interval must be >= 1, got {self.progress_interval}."
            )
        if self.distributed_timeout_seconds < 1:
            raise ValueError(
                "distributed_timeout_seconds must be >= 1, got "
                f"{self.distributed_timeout_seconds}."
            )
        if self.random_num_candidates < 1:
            raise ValueError(
                f"random_num_candidates must be >= 1, got {self.random_num_candidates}."
            )
        if self.strategy not in {"candidate", "random"}:
            raise ValueError(f"unsupported strategy: {self.strategy!r}.")
        if self.placement_policy not in {"first_fit", "iterative"}:
            raise ValueError(
                f"unsupported placement_policy: {self.placement_policy!r}."
            )
        if not self.rate_only and not self.output_path:
            raise ValueError("output_path is required unless rate_only is set.")
        if not self.rate_only and not self.resolved_sid_groups_output_path:
            raise ValueError(
                "resolved_sid_groups_output_path is required unless rate_only is set."
            )

        # Eagerly build the resolution config so its validation (layer_sizes,
        # capacity) fails fast here rather than lazily inside run().
        _ = self.resolution_config

    def validate_path_conflicts(self) -> None:
        """Reject overlapping input and output paths before any I/O.

        Raises:
            ValueError: If any two configured paths overlap.
        """
        paths = [self.input_path]
        paths.extend(
            path
            for path in (
                self.output_path,
                self.original_sid_groups_output_path,
                self.resolved_sid_groups_output_path,
            )
            if path
        )
        has_conflict, conflict_message = check_path_conflict(paths)
        if has_conflict:
            raise ValueError(conflict_message)

    @classmethod
    def from_namespace(cls, args: argparse.Namespace) -> "ResolveSidCollisionsConfig":
        """Build a validated configuration from parsed CLI arguments."""
        try:
            layer_sizes = tuple(int(value) for value in args.codebook.split(","))
        except ValueError as err:
            raise ValueError(
                "--codebook must be 'int,int,...' with no empty fields, got "
                f"{args.codebook!r}."
            ) from err
        return cls(
            input_path=args.input_path,
            output_path=args.output_path,
            original_sid_groups_output_path=args.original_sid_groups_output_path,
            resolved_sid_groups_output_path=args.resolved_sid_groups_output_path,
            reader_type=args.reader_type,
            writer_type=args.writer_type,
            batch_size=args.batch_size,
            progress_interval=args.progress_interval,
            distributed_timeout_seconds=args.distributed_timeout_seconds,
            item_id_field=args.item_id_field,
            code_field=args.code_field,
            candidate_codes_field=args.candidate_codes_field,
            layer_sizes=layer_sizes,
            max_items_per_codebook=args.max_items_per_codebook,
            strategy=args.strategy,
            placement_policy=args.placement_policy,
            random_num_candidates=args.random_num_candidates,
            rate_only=args.rate_only,
            odps_data_quota_name=args.odps_data_quota_name,
        )

    @cached_property
    def resolution_config(self) -> CollisionResolutionConfig:
        """Return the pure-core configuration."""
        return CollisionResolutionConfig(
            layer_sizes=self.layer_sizes,
            capacity=self.max_items_per_codebook,
        )


class CollisionResolutionRunner:
    """Run best-effort SID collision resolution over repository I/O."""

    def __init__(
        self,
        config: ResolveSidCollisionsConfig,
    ) -> None:
        self._config = config
        if self._config.placement_policy == "iterative":
            self._resolver: CollisionResolver = IterativeCollisionResolver(
                progress_interval=self._config.progress_interval,
            )
        else:
            self._resolver = KnnCollisionResolver(
                progress_interval=self._config.progress_interval
            )
        self._default_writer_type: Optional[str] = None
        self._item_id_type: Optional[pa.DataType] = None

    def run(self) -> CollisionResolutionStats:
        """Read, resolve collisions, and write the resulting SID map."""
        with initialize_distributed_collision(
            self._config.distributed_timeout_seconds,
        ) as context:
            if context.is_coordinator:
                self._config.validate_path_conflicts()
            if not context.distributed:
                return self._run_single_process()
            if context.is_coordinator:
                return self._run_distributed_coordinator(context)
            return self._run_distributed_worker(context)

    def _run_single_process(self) -> CollisionResolutionStats:
        """Execute the existing in-process Reader, resolver, and Writer path."""
        item_ids, codes = self._load_codes()
        plan = prepare_collision_plan(
            item_ids,
            codes,
            self._config.resolution_config,
        )
        collect_grouping = not self._config.rate_only and bool(plan.overflow_rows.size)
        candidate_last_codes = None
        if plan.overflow_rows.size:
            candidate_last_codes = self._prepare_candidate_last_codes(plan)
        result = self._resolver.resolve(
            plan,
            candidate_last_codes,
            collect_grouping=collect_grouping,
        )
        del candidate_last_codes
        return self._finish_coordinator_run(item_ids, codes, plan, result)

    def _finish_coordinator_run(
        self,
        item_ids: np.ndarray,
        codes: np.ndarray,
        plan: CollisionPlan,
        result: CollisionResolutionResult,
    ) -> CollisionResolutionStats:
        """Write coordinator-owned outputs and log final statistics."""
        if self._config.rate_only:
            logger.info("rate_only: skipping map and SID group writes")
            del plan
        else:
            self._write_group_outputs(item_ids, codes, plan, result)
            del plan
            self._write_map(item_ids, codes, result)

        logger.info("SID collision resolution finished: %s", result.stats)
        return result.stats

    def _run_distributed_coordinator(
        self, context: DistributedCollisionContext
    ) -> CollisionResolutionStats:
        """Prepare, distribute, resolve, merge, and write on rank zero."""
        item_ids, codes = self._load_codes()
        plan = prepare_collision_plan(
            item_ids,
            codes,
            self._config.resolution_config,
        )
        shards = partition_collision_bands(
            plan.overflow_bucket_key_prefixes,
            plan.bucket_keys,
            plan.config.layer_sizes[-1],
            context.world_size,
        )
        candidates = (
            self._prepare_candidate_last_codes(plan)
            if plan.overflow_rows.size
            else np.empty((0, 0), dtype=np.int64)
        )
        local_work: Optional[DistributedCollisionWork] = None
        for shard in shards:
            overflow_slice = slice(shard.overflow_start, shard.overflow_end)
            order_hashes = (
                plan.overflow_order_hashes[overflow_slice]
                if self._resolver.requires_order_hashes
                else None
            )
            if shard.rank == context.rank:
                # The local slice is copied so the full matrix can be freed.
                local_work = build_collision_work(
                    plan, shard, candidates[overflow_slice].copy(), order_hashes
                )
            else:
                # Remote sends stream contiguous row views without copying.
                send_collision_work(
                    build_collision_work(
                        plan, shard, candidates[overflow_slice], order_hashes
                    ),
                    dst=shard.rank,
                )
        del candidates, order_hashes
        if local_work is None:
            raise RuntimeError("rank 0 collision work is unavailable.")

        synchronize_collision_workers()
        shard_result = self._resolve_distributed_work(local_work)
        del local_work
        shard_results = gather_collision_shard_results(shard_result, context)
        if shard_results is None:
            raise RuntimeError("rank 0 did not gather collision shard results.")
        collect_grouping = not self._config.rate_only and bool(plan.overflow_rows.size)
        result = self._merge_collision_shards(
            plan,
            shards,
            shard_results,
            collect_grouping,
        )
        del shard_results
        stats = self._finish_coordinator_run(item_ids, codes, plan, result)
        return broadcast_collision_stats(stats, context)

    def _run_distributed_worker(
        self, context: DistributedCollisionContext
    ) -> CollisionResolutionStats:
        """Resolve one received work shard and return the shared statistics."""
        work = receive_collision_work(src=0)
        synchronize_collision_workers()
        shard_result = self._resolve_distributed_work(work)
        del work
        gather_collision_shard_results(shard_result, context)
        del shard_result
        return broadcast_collision_stats(None, context)

    def _resolve_distributed_work(
        self, work: DistributedCollisionWork
    ) -> CollisionShardResult:
        """Apply the configured placement policy to one numeric work shard."""
        return self._resolver.resolve_shard(
            work.work_shard,
            work.candidate_codes,
            work.order_hashes,
        )

    @staticmethod
    def _merge_collision_shards(
        plan: CollisionPlan,
        shards: Tuple[CollisionBandShard, ...],
        shard_results: Sequence[CollisionShardResult],
        collect_grouping: bool,
    ) -> CollisionResolutionResult:
        """Merge and validate rank-ordered complete-band shard results."""
        if len(shard_results) != len(shards):
            raise ValueError(
                f"received {len(shard_results)} results for {len(shards)} shards."
            )
        overflow_count = plan.overflow_rows.size
        covered_until = 0
        resolved_last_codes = plan.original_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_storage = np.empty(overflow_count, dtype=np.int64)
        unresolved_count = 0
        bucket_key_parts: List[np.ndarray] = []
        bucket_count_parts: List[np.ndarray] = []
        last_size = plan.config.layer_sizes[-1]
        capacity = plan.config.capacity
        untouched = ~overflow_band_bucket_mask(
            plan.bucket_keys, plan.overflow_bucket_key_prefixes, last_size
        )
        untouched_counts = np.minimum(plan.bucket_counts[untouched], capacity)
        merged_item_count = int(untouched_counts.sum(dtype=np.int64))
        final_collision_buckets = 0
        max_final_bucket_size = (
            int(untouched_counts.max()) if untouched_counts.size else 0
        )
        if collect_grouping:
            bucket_key_parts.append(plan.bucket_keys[untouched])
            bucket_count_parts.append(untouched_counts)

        for shard, result in zip(shards, shard_results):
            if shard.overflow_start != covered_until:
                raise ValueError("collision shards do not cover overflow contiguously.")
            local_count = shard.overflow_end - shard.overflow_start
            if result.resolved_last_codes.shape != (local_count,):
                raise ValueError(
                    f"rank {shard.rank} returned "
                    f"{result.resolved_last_codes.shape} resolved codes for "
                    f"{local_count} overflow rows."
                )
            if np.any(result.resolved_last_codes < 0) or np.any(
                result.resolved_last_codes >= last_size
            ):
                raise ValueError(
                    f"rank {shard.rank} returned a resolved code outside "
                    f"[0, {last_size})."
                )
            if result.slot_indices.shape != (local_count,):
                raise ValueError(
                    f"rank {shard.rank} returned invalid slot index shape "
                    f"{result.slot_indices.shape}."
                )
            if np.any(result.slot_indices < 1):
                raise ValueError(
                    f"rank {shard.rank} returned nonpositive slot indices."
                )
            rows = plan.overflow_rows[shard.overflow_start : shard.overflow_end]
            if result.unresolved_rows.size and np.any(
                ~np.isin(result.unresolved_rows, rows)
            ):
                raise ValueError(
                    f"rank {shard.rank} returned an unresolved row it does not own."
                )
            if np.unique(result.unresolved_rows).size != result.unresolved_rows.size:
                raise ValueError(
                    f"rank {shard.rank} returned duplicate unresolved rows."
                )
            if result.final_bucket_keys.shape != result.final_bucket_counts.shape:
                raise ValueError(
                    f"rank {shard.rank} returned misaligned final bucket metadata."
                )
            if np.any(result.final_bucket_counts < 1):
                raise ValueError("collision shard bucket counts must be positive.")
            if np.any(np.diff(result.final_bucket_keys) <= 0):
                raise ValueError(
                    "collision shard bucket keys must be strictly increasing."
                )
            shard_prefixes = plan.overflow_bucket_key_prefixes[
                shard.overflow_start : shard.overflow_end
            ]
            if not np.all(
                overflow_band_bucket_mask(
                    result.final_bucket_keys, shard_prefixes, last_size
                )
            ):
                raise ValueError(
                    f"rank {shard.rank} returned a bucket it does not own."
                )
            resolved_last_codes[rows] = result.resolved_last_codes
            slot_indices[rows] = result.slot_indices
            next_unresolved_count = unresolved_count + result.unresolved_rows.size
            unresolved_storage[unresolved_count:next_unresolved_count] = (
                result.unresolved_rows
            )
            unresolved_count = next_unresolved_count
            merged_item_count += int(result.final_bucket_counts.sum(dtype=np.int64))
            final_collision_buckets += int(
                (result.final_bucket_counts > capacity).sum()
            )
            if result.final_bucket_counts.size:
                max_final_bucket_size = max(
                    max_final_bucket_size,
                    int(result.final_bucket_counts.max()),
                )
            if collect_grouping:
                bucket_key_parts.append(result.final_bucket_keys)
                bucket_count_parts.append(result.final_bucket_counts)
            covered_until = shard.overflow_end
        if covered_until != overflow_count:
            raise ValueError(
                f"collision shards cover {covered_until} of {overflow_count} "
                "overflow rows."
            )
        if merged_item_count != plan.item_count:
            raise ValueError(
                "merged collision bucket counts do not match the input row count."
            )
        unresolved_storage.resize(unresolved_count, refcheck=False)
        unresolved_rows = unresolved_storage
        if collect_grouping:
            all_keys = np.concatenate(bucket_key_parts)
            all_counts = np.concatenate(bucket_count_parts)
            order = np.argsort(all_keys, kind="stable")
            all_keys = all_keys[order]
            all_counts = all_counts[order]
            if np.any(all_keys[1:] == all_keys[:-1]):
                raise ValueError("collision shard bucket keys overlap.")
        else:
            all_keys = np.empty(0, dtype=np.int64)
            all_counts = np.empty(0, dtype=np.int64)
        stats = CollisionResolutionStats(
            total_items=plan.item_count,
            raw_collision_buckets=int((plan.bucket_counts > capacity).sum()),
            final_collision_buckets=final_collision_buckets,
            relocated_count=int(overflow_count - unresolved_count),
            unresolved_count=int(unresolved_count),
            max_final_bucket_size=max_final_bucket_size,
        )
        return CollisionResolutionResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_bucket_keys=all_keys,
            final_bucket_counts=all_counts,
            grouping_collected=collect_grouping,
            stats=stats,
        )

    def _make_reader(self, selected_cols: List[str]) -> "BaseReader":
        """Open a repository reader projecting ``selected_cols``."""
        with _rank_zero_io_environment():
            return create_reader(
                input_path=self._config.input_path,
                batch_size=self._config.batch_size,
                selected_cols=selected_cols,
                reader_type=self._config.reader_type,
                quota_name=self._config.odps_data_quota_name,
            )

    def _progress_logger(self, description: str) -> ProgressLogger:
        """Create a logger throttled by the configured progress interval."""
        return ProgressLogger(
            description, start_n=0, miniters=self._config.progress_interval
        )

    @staticmethod
    def _codes_matrix(values: pa.Array) -> np.ndarray:
        """Decode an SID column into an ``(N, n_layers)`` int64 matrix."""
        if (
            pa.types.is_list(values.type)
            or pa.types.is_large_list(values.type)
            or pa.types.is_fixed_size_list(values.type)
        ):
            row_count = len(values)
            flat = (
                values.flatten()
                .to_numpy(zero_copy_only=False)
                .astype(np.int64, copy=False)
            )
        elif pa.types.is_integer(values.type):
            array = values.to_numpy(zero_copy_only=False).astype(np.int64, copy=False)
            return array.reshape(-1, 1)
        else:
            parts = pc.split_pattern(values, ",")
            row_count = len(parts)
            flat = pc.cast(parts.flatten(), pa.int64()).to_numpy(zero_copy_only=False)

        if row_count == 0:
            return np.empty((0, 0), dtype=np.int64)
        layer_count = flat.shape[0] // row_count
        if flat.shape[0] != row_count * layer_count:
            raise ValueError("ragged SID codes: all items must share n_layers.")
        return flat.reshape(row_count, layer_count)

    def _validate_item_ids(self, item_ids: pa.Array) -> None:
        """Validate one item ID batch and retain its Arrow type.

        Args:
            item_ids: Item IDs from one input batch.

        Raises:
            ValueError: If IDs contain nulls or their type changes between batches.
        """
        if item_ids.null_count:
            raise ValueError("item IDs must not contain null values.")
        if self._item_id_type is None:
            self._item_id_type = item_ids.type
        elif self._item_id_type != item_ids.type:
            raise ValueError(
                "item ID type changed between batches: "
                f"{self._item_id_type} vs {item_ids.type}."
            )

    def _load_codes(self) -> Tuple[np.ndarray, np.ndarray]:
        """Load item IDs and SIDs into arrays used by the NumPy core.

        Also resolves the default writer type from the reader class name before
        streaming, so downstream writes can default to the input format.
        """
        reader = self._make_reader(
            [self._config.item_id_field, self._config.code_field]
        )
        self._default_writer_type = self._config.writer_type or (
            reader.__class__.__name__.replace("Reader", "Writer")
        )
        progress = self._progress_logger("Reading SID input")
        read_rows = 0
        id_chunks: List[np.ndarray] = []
        code_chunks: List[np.ndarray] = []
        for batch in reader.to_batches():
            item_ids = batch[self._config.item_id_field]
            self._validate_item_ids(item_ids)
            id_chunks.append(item_ids.to_numpy(zero_copy_only=False))
            code_chunks.append(self._codes_matrix(batch[self._config.code_field]))
            read_rows += len(item_ids)
            progress.log(read_rows)

        if not id_chunks:
            raise ValueError("SID input is empty.")
        item_id_array = np.concatenate(id_chunks)
        del id_chunks
        code_matrix = np.concatenate(code_chunks, axis=0)
        del code_chunks
        return item_id_array, code_matrix

    def _candidate_last_matrix(self, values: pa.Array) -> np.ndarray:
        """Decode a flat candidate column into its per-candidate last codes.

        ``values`` is decoded like ``codes`` (list, integer, or comma string) into
        an ``(N, topk * n_layers)`` matrix, then split into ``topk`` groups of
        ``n_layers`` (from ``--codebook``), keeping each group's last-layer code.

        Args:
            values: One batch's flat ``candidate_codes`` column.

        Returns:
            The last-layer code of every candidate, shape ``(N, topk)``.

        Raises:
            ValueError: If the per-row width is not a multiple of ``n_layers``.
        """
        flat = self._codes_matrix(values)
        per_row = flat.shape[1]
        n_layers = len(self._config.layer_sizes)
        if per_row % n_layers != 0:
            raise ValueError(
                f"candidate_codes width {per_row} is not a multiple of n_layers "
                f"{n_layers}."
            )
        return flat[:, n_layers - 1 :: n_layers]

    def _prepare_candidate_last_codes(
        self,
        plan: CollisionPlan,
    ) -> np.ndarray:
        """Build candidate codes in memory."""
        if self._config.strategy == "candidate":
            return self._load_candidate_last_codes(plan.overflow_item_ids)

        overflow_count = plan.overflow_rows.size
        last_size = plan.config.layer_sizes[-1]
        candidate_count = min(self._config.random_num_candidates, last_size - 1)
        candidates = np.empty((overflow_count, candidate_count), dtype=np.int64)
        chunk_rows = _candidate_chunk_rows(candidate_count)
        for start in range(0, overflow_count, chunk_rows):
            end = min(start + chunk_rows, overflow_count)
            candidates[start:end] = generate_random_candidate_last_codes(
                plan.overflow_item_ids[start:end],
                last_size,
                self._config.random_num_candidates,
            )
        return candidates

    def _load_candidate_last_codes(
        self,
        overflow_item_ids: np.ndarray,
    ) -> np.ndarray:
        """Load fixed-width candidates aligned to ``overflow_item_ids``."""
        item_count = overflow_item_ids.shape[0]
        item_id_lookup = _ItemIdLookup(overflow_item_ids)

        candidates: Optional[np.ndarray] = None
        seen = np.zeros(item_count, dtype=bool)
        for target_rows, batch_candidates in self._iter_candidate_batches(
            item_id_lookup
        ):
            if candidates is None:
                candidates = np.empty(
                    (item_count, batch_candidates.shape[1]), dtype=np.int64
                )
            elif candidates.shape[1] != batch_candidates.shape[1]:
                raise ValueError(
                    "candidate topk changed between batches: "
                    f"{candidates.shape[1]} vs {batch_candidates.shape[1]}."
                )
            candidates[target_rows] = batch_candidates
            seen[target_rows] = True

        if candidates is None:
            raise ValueError(
                "map has overflow items but candidate_codes yielded no candidates."
            )
        item_id_lookup.broadcast_duplicate_targets(candidates)
        item_id_lookup.broadcast_duplicate_targets(seen)
        if not np.all(seen):
            missing = np.flatnonzero(~seen)
            preview = ",".join(str(value) for value in missing[:10])
            raise ValueError(
                f"candidate_codes missing for {missing.size} overflow items; "
                f"first overflow positions: {preview}."
            )
        return candidates

    def _iter_candidate_batches(
        self,
        item_id_lookup: _ItemIdLookup,
    ) -> Iterator[Tuple[np.ndarray, np.ndarray]]:
        """Yield matched candidate batches with duplicate targets resolved."""
        field = self._config.candidate_codes_field
        reader = self._make_reader([self._config.item_id_field, field])
        scanned_items = 0
        progress = self._progress_logger("Scanning candidate input")
        for batch in reader.to_batches():
            if field not in batch:
                raise ValueError(
                    f"candidate_codes field {field!r} is missing from an input batch."
                )
            batch_ids = batch[self._config.item_id_field].to_numpy(zero_copy_only=False)
            scanned_items += batch_ids.shape[0]
            progress.log(scanned_items)
            source_rows, target_rows = item_id_lookup.match(batch_ids)
            if source_rows.size == 0:
                continue

            selected = pc.take(batch[field], pa.array(source_rows, type=pa.int64()))
            batch_candidates = self._candidate_last_matrix(selected)
            if np.unique(target_rows).size != target_rows.size:
                _, reverse_indices = np.unique(target_rows[::-1], return_index=True)
                keep = np.sort(target_rows.size - 1 - reverse_indices)
                target_rows = target_rows[keep]
                batch_candidates = batch_candidates[keep]
            yield target_rows, batch_candidates

    def _make_writer(self, output_path: str) -> "BaseWriter":
        """Create a repository writer for one independently resolved output."""
        if self._default_writer_type is None:
            raise RuntimeError("writer type is unavailable before reading input.")
        with _rank_zero_io_environment():
            return create_writer(
                output_path,
                writer_type=self._default_writer_type,
                quota_name=self._config.odps_data_quota_name,
                world_size=1,
            )

    def _item_id_array(self, values: np.ndarray) -> pa.Array:
        """Encode item IDs with the input Arrow type preserved."""
        if self._item_id_type is None:
            raise RuntimeError("item ID type is unavailable before reading input.")
        return pa.array(values, type=self._item_id_type)

    @staticmethod
    def _grouped_item_ids_column(
        values: pa.Array, offsets: np.ndarray, is_csv: bool
    ) -> pa.Array:
        """Encode flat item IDs and offsets as one grouped output column."""
        arrow_offsets = pa.array(offsets, type=pa.int32())
        if not is_csv:
            return pa.ListArray.from_arrays(arrow_offsets, values)

        if pa.types.is_integer(values.type):
            encoded_values = pc.cast(values, pa.string())
        else:
            try:
                encoded_values = pa.array(
                    [
                        json.dumps(
                            value,
                            ensure_ascii=False,
                            separators=(",", ":"),
                        )
                        for value in values.to_pylist()
                    ],
                    type=pa.string(),
                )
            except (TypeError, ValueError) as error:
                raise ValueError(
                    "CSV SID group output cannot JSON-encode item ID type "
                    f"{values.type}."
                ) from error
        encoded_lists = pa.ListArray.from_arrays(arrow_offsets, encoded_values)
        joined_values = pc.binary_join(encoded_lists, ",")
        return pc.binary_join_element_wise("[", joined_values, "]", "")

    @staticmethod
    def _codes_column(codes: np.ndarray, is_csv: bool) -> pa.Array:
        """Encode an SID matrix for the actual output writer."""
        row_count, layer_count = codes.shape
        if is_csv:
            values = pc.cast(pa.array(codes.reshape(-1)), pa.string())
            offsets = pa.array(
                np.arange(
                    0,
                    (row_count + 1) * layer_count,
                    layer_count,
                    dtype=np.int64,
                )
            )
            return pc.binary_join(pa.LargeListArray.from_arrays(offsets, values), ",")
        if row_count > _ARROW_LIST_OFFSET_MAX // layer_count:
            raise ValueError("SID output exceeds Arrow list offset capacity.")
        values = pa.array(codes.reshape(-1))
        offsets = pa.array(
            np.arange(
                0,
                (row_count + 1) * layer_count,
                layer_count,
                dtype=np.int32,
            )
        )
        return pa.ListArray.from_arrays(offsets, values)

    def _write_map(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        result: CollisionResolutionResult,
    ) -> None:
        """Write the resolved map in chunks without a full final-code copy."""
        output_path = self._config.output_path
        if output_path is None:
            raise RuntimeError("map output path was not validated.")
        with closing(self._make_writer(output_path)) as writer:
            is_csv = writer.__class__.__name__ == "CsvWriter"
            output_count = item_ids.shape[0]
            progress = self._progress_logger("Writing resolved item map")
            write_chunk = _MAP_WRITE_ROWS
            if not is_csv:
                write_chunk = min(
                    write_chunk,
                    _ARROW_LIST_OFFSET_MAX // origin_codes.shape[1],
                )
                if write_chunk < 1:
                    raise ValueError("one SID row exceeds Arrow list offset capacity.")
            for start in range(0, output_count, write_chunk):
                end = min(start + write_chunk, output_count)
                selection = slice(start, end)
                origin_chunk = origin_codes[selection]
                final_chunk = origin_chunk.copy()
                final_chunk[:, -1] = result.resolved_last_codes[selection]
                writer.write(
                    {
                        "item_id": self._item_id_array(item_ids[selection]),
                        "origin_codebook": self._codes_column(origin_chunk, is_csv),
                        "codebook": self._codes_column(final_chunk, is_csv),
                        "index": pa.array(
                            result.slot_indices[selection], type=pa.int64()
                        ),
                    }
                )
                progress.log(end)

    def _write_group_outputs(
        self,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        plan: CollisionPlan,
        result: CollisionResolutionResult,
    ) -> None:
        """Build and write original and resolved SID item groups sequentially.

        Args:
            item_ids: Input item IDs in original row order.
            origin_codes: Original full SID matrix.
            plan: Original SID aggregation and overflow plan.
            result: Completed collision-resolution result.
        """
        original_path = self._config.original_sid_groups_output_path
        resolved_path = self._config.resolved_sid_groups_output_path
        if resolved_path is None:
            raise RuntimeError("resolved SID group output path was not validated.")

        original_grouping: Optional[CodebookItemGrouping] = None
        if original_path is not None:
            original_grouping = build_original_item_grouping(plan)
            self._write_sid_groups(
                original_path,
                item_ids,
                origin_codes,
                original_grouping,
                resolved_last_codes=None,
                progress_description="Writing original SID item groups",
            )

        if plan.overflow_rows.size:
            del original_grouping
            grouping = build_resolved_item_grouping(plan, result)
            resolved_last_codes = result.resolved_last_codes
        else:
            grouping = original_grouping or build_original_item_grouping(plan)
            resolved_last_codes = None
        self._write_sid_groups(
            resolved_path,
            item_ids,
            origin_codes,
            grouping,
            resolved_last_codes=resolved_last_codes,
            progress_description="Writing resolved SID item groups",
        )

    def _write_sid_groups(
        self,
        output_path: str,
        item_ids: np.ndarray,
        origin_codes: np.ndarray,
        grouping: CodebookItemGrouping,
        resolved_last_codes: Optional[np.ndarray],
        progress_description: str,
    ) -> None:
        """Write codebook-to-item-ID groups in SID and slot order.

        Args:
            output_path: Destination passed to the selected repository writer.
            item_ids: Input item IDs in original row order.
            origin_codes: Original full SID matrix.
            grouping: Sorted SID groups and their flattened original row order.
            resolved_last_codes: Final last-layer values, or ``None`` when
                writing original SIDs.
            progress_description: Description identifying the group output.

        Raises:
            ValueError: If one group exceeds Arrow list offset capacity.
        """
        group_count = grouping.counts.shape[0]
        if np.any(grouping.counts > _ARROW_LIST_OFFSET_MAX):
            raise ValueError("one SID group exceeds Arrow list offset capacity.")

        offsets = grouping.offsets
        with closing(self._make_writer(output_path)) as writer:
            is_csv = writer.__class__.__name__ == "CsvWriter"
            progress = self._progress_logger(progress_description)
            max_codebook_rows = (
                group_count
                if is_csv
                else _ARROW_LIST_OFFSET_MAX // origin_codes.shape[1]
            )
            if not is_csv and max_codebook_rows < 1:
                raise ValueError("one SID row exceeds Arrow list offset capacity.")
            group_start = 0
            while group_start < group_count:
                child_limit = int(offsets[group_start]) + _GROUP_WRITE_ITEMS
                group_end = int(np.searchsorted(offsets, child_limit, side="right") - 1)
                group_end = max(group_end, group_start + 1)
                group_end = min(group_end, group_start + max_codebook_rows)

                child_start = int(offsets[group_start])
                child_end = int(offsets[group_end])
                rows = grouping.row_order[child_start:child_end]
                local_offsets = (
                    offsets[group_start : group_end + 1] - child_start
                ).astype(np.int32, copy=False)
                representative_rows = grouping.row_order[offsets[group_start:group_end]]
                code_chunk = origin_codes[representative_rows]
                if resolved_last_codes is not None:
                    code_chunk[:, -1] = resolved_last_codes[representative_rows]
                flat_item_ids = self._item_id_array(item_ids[rows])
                writer.write(
                    {
                        "codebook": self._codes_column(code_chunk, is_csv),
                        "itemids": self._grouped_item_ids_column(
                            flat_item_ids, local_offsets, is_csv
                        ),
                    }
                )
                group_start = group_end
                progress.log(child_end)


def build_parser() -> argparse.ArgumentParser:
    """Build the SID collision-resolution command-line parser."""
    parser = argparse.ArgumentParser(
        description=(
            "Resolve SID codebook collisions within each band on a best-effort "
            "basis; finite candidate sets may leave over-capacity buckets."
        )
    )
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--output_path",
        default=None,
        help="Resolved item map output; required unless --rate_only is set.",
    )
    parser.add_argument(
        "--original_sid_groups_output_path",
        default=None,
        help=("Optional audit output grouping item IDs by their original SID."),
    )
    parser.add_argument(
        "--resolved_sid_groups_output_path",
        default=None,
        help=(
            "Output grouped item IDs keyed by their resolved SID; required "
            "unless --rate_only is set."
        ),
    )
    parser.add_argument(
        "--reader_type",
        choices=["CsvReader", "ParquetReader", "OdpsReader"],
        default=None,
    )
    parser.add_argument(
        "--writer_type",
        choices=["CsvWriter", "ParquetWriter", "OdpsWriter"],
        default=None,
        help="Output writer; defaults to matching the input reader.",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=100000,
        help=(
            "Reader I/O batch size; does not limit the in-memory collision working set."
        ),
    )
    parser.add_argument(
        "--progress_interval",
        type=int,
        default=1_000_000,
        help="Number of processed samples between progress checks.",
    )
    parser.add_argument(
        "--distributed_timeout_seconds",
        type=int,
        default=86400,
        help="Maximum wait for distributed preparation and worker results.",
    )
    parser.add_argument("--item_id_field", default="item_id")
    parser.add_argument("--code_field", default="codes")
    parser.add_argument(
        "--codebook",
        required=True,
        help="Comma-separated per-layer sizes, e.g. '8192,8192,8192'.",
    )
    parser.add_argument("--candidate_codes_field", default="candidate_codes")
    parser.add_argument("--max_items_per_codebook", type=int, required=True)
    parser.add_argument(
        "--strategy",
        choices=["candidate", "random"],
        default="candidate",
        help="Use model candidates or deterministic legacy random draws.",
    )
    parser.add_argument(
        "--placement_policy",
        choices=["first_fit", "iterative"],
        default="first_fit",
        help=(
            "Place candidates greedily or with deterministic SQL-inspired "
            "multi-round arbitration."
        ),
    )
    parser.add_argument(
        "--random_num_candidates",
        type=int,
        default=64,
        help="Full-space random draws per overflow item for random strategy.",
    )
    parser.add_argument(
        "--rate_only",
        action="store_true",
        help="Compute and log metrics without writing map or SID group outputs.",
    )
    parser.add_argument("--odps_data_quota_name", default="pay-as-you-go")
    return parser


def main() -> None:
    """Run SID collision resolution from command-line arguments."""
    config = ResolveSidCollisionsConfig.from_namespace(build_parser().parse_args())
    CollisionResolutionRunner(config).run()


if __name__ == "__main__":
    main()
