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

"""Deterministic multi-round collision resolution for semantic IDs."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger
from tzrec.utils.sid.collision import (
    CollisionResolver,
    CollisionShardResult,
    CollisionWorkShard,
    overflow_band_bucket_mask,
)


@dataclass(frozen=True)
class IterativeCollisionRoundStats:
    """Summary of one proposal and arbitration round.

    Args:
        band_key: Mixed-radix prefix key of the independently resolved band.
        round_index: One-based round number within the band.
        active_items: Number of overflow items not assigned before the round.
        active_edges: Candidate edges remaining after assigned-item and
            full-target filtering.
        proposals: Candidate edges surviving per-origin priority throttling.
        accepts: Proposal edges temporarily accepted by target buckets.
        winners: Items committed after per-item arbitration.
    """

    band_key: int
    round_index: int
    active_items: int
    active_edges: int
    proposals: int
    accepts: int
    winners: int


@dataclass(frozen=True)
class IterativeCollisionDiagnostics:
    """Optional diagnostics for deterministic multi-round arbitration.

    Args:
        assignment_rounds: One-based successful assignment round, or zero for
            an unresolved item that kept its origin SID.
        round_stats: Per-band round summaries, including each terminal
            zero-winner round.
    """

    assignment_rounds: np.ndarray
    round_stats: tuple[IterativeCollisionRoundStats, ...]


def _run_starts(ordered_values: np.ndarray) -> np.ndarray:
    """Return start offsets of equal-value runs in a nonempty sorted array."""
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1,
        )
    )


def _ranks_within_runs(ordered_values: np.ndarray) -> np.ndarray:
    """Return zero-based ranks within equal-value runs of a nonempty sorted array."""
    starts = _run_starts(ordered_values)
    ranks = np.arange(ordered_values.shape[0], dtype=np.int64)
    return ranks - np.repeat(
        starts, np.diff(np.append(starts, ordered_values.shape[0]))
    )


class IterativeCollisionResolver(CollisionResolver):
    """Resolve overflow with deterministic SQL-inspired batch arbitration.

    Each band is independent. In every round, candidate edges for assigned
    items and full targets are removed, every ``(origin, priority)`` group
    emits at most ``capacity`` proposals, targets accept proposals by priority
    and stable item order, and each item commits only its best accepted edge.
    Vacancies created by the final item arbitration are visible in the next
    round.
    """

    requires_order_hashes: bool = True

    def _resolve_shard_impl(
        self,
        shard: CollisionWorkShard,
        candidates: np.ndarray,
        order_hashes: Optional[np.ndarray],
    ) -> CollisionShardResult:
        """Arbitrate one validated-candidate shard made of complete bands.

        Args:
            shard: Numeric overflow and initial-occupancy inputs for one or
                more complete bands. Candidate columns are one-based
                priorities by position and duplicate values remain distinct
                edges.
            candidates: Validated candidate matrix aligned with the shard.
            order_hashes: Stable uint64 hashes aligned with
                ``shard.overflow_rows``.

        Returns:
            Compact assignments and bucket metadata for the selected bands.

        Raises:
            TypeError: If shard or hash arrays have invalid dtypes.
            ValueError: If hashes are missing or shard arrays are malformed.
        """
        if order_hashes is None:
            raise ValueError("iterative collision work is missing stable order hashes.")
        validated_hashes = self._validate_work_shard(shard, order_hashes)
        result, _ = self._resolve_validated_shard(
            shard,
            candidates,
            validated_hashes,
            collect_diagnostics=False,
        )
        return result

    def resolve_shard_with_diagnostics(
        self,
        shard: CollisionWorkShard,
        candidate_codes: np.ndarray,
        overflow_order_hashes: np.ndarray,
    ) -> tuple[CollisionShardResult, IterativeCollisionDiagnostics]:
        """Resolve a shard and retain its per-round diagnostic trace.

        Args:
            shard: Numeric overflow and initial-occupancy inputs for complete
                bands.
            candidate_codes: Two-dimensional candidates aligned with the shard.
            overflow_order_hashes: Precomputed stable uint64 hashes aligned
                with the shard.

        Returns:
            The production shard result plus assignment-round diagnostics.

        Raises:
            TypeError: If shard, candidate, or hash arrays have invalid dtypes.
            ValueError: If inputs are malformed, misaligned, or out of range.
        """
        order_hashes = self._validate_work_shard(shard, overflow_order_hashes)
        candidates = self._validate_candidate_last_codes(
            candidate_codes,
            shard.overflow_rows.shape[0],
            shard.config.layer_sizes[-1],
        )
        result, diagnostics = self._resolve_validated_shard(
            shard, candidates, order_hashes
        )
        if diagnostics is None:
            raise RuntimeError("iterative collision diagnostics were not collected.")
        return result, diagnostics

    def _validate_work_shard(
        self,
        shard: CollisionWorkShard,
        overflow_order_hashes: np.ndarray,
    ) -> np.ndarray:
        """Validate compact numeric worker inputs."""
        overflow_arrays = (
            ("overflow_rows", shard.overflow_rows),
            (
                "overflow_bucket_key_prefixes",
                shard.overflow_bucket_key_prefixes,
            ),
            ("overflow_origin_last_codes", shard.overflow_origin_last_codes),
        )
        overflow_count = shard.overflow_rows.shape[0]
        for name, values in overflow_arrays:
            if values.ndim != 1:
                raise ValueError(f"{name} must be 1-D, got shape {values.shape}.")
            if values.shape[0] != overflow_count:
                raise ValueError(
                    f"{name} has {values.shape[0]} rows, expected {overflow_count}."
                )
            if not np.issubdtype(values.dtype, np.integer):
                raise TypeError(f"{name} must use an integer dtype.")
        order_hashes = np.asarray(overflow_order_hashes)
        if order_hashes.ndim != 1:
            raise ValueError(
                f"overflow_order_hashes must be 1-D, got shape {order_hashes.shape}."
            )
        if order_hashes.shape[0] != overflow_count:
            raise ValueError(
                "overflow_order_hashes must align with overflow rows, got "
                f"{order_hashes.shape[0]} hashes for {overflow_count} rows."
            )
        if order_hashes.dtype != np.dtype(np.uint64):
            raise TypeError("overflow_order_hashes must use uint64 dtype.")
        if shard.bucket_keys.ndim != 1 or shard.bucket_counts.ndim != 1:
            raise ValueError("bucket_keys and bucket_counts must be 1-D.")
        if shard.bucket_keys.shape != shard.bucket_counts.shape:
            raise ValueError("bucket_keys and bucket_counts must align.")
        if not np.issubdtype(shard.bucket_keys.dtype, np.integer):
            raise TypeError("bucket_keys must use an integer dtype.")
        if not np.issubdtype(shard.bucket_counts.dtype, np.integer):
            raise TypeError("bucket_counts must use an integer dtype.")
        if np.any(shard.bucket_counts <= 0):
            raise ValueError("bucket_counts must be positive.")
        if np.any(shard.bucket_keys[1:] <= shard.bucket_keys[:-1]):
            raise ValueError("bucket_keys must be strictly increasing.")

        last_size = shard.config.layer_sizes[-1]
        if np.any(shard.overflow_origin_last_codes < 0) or np.any(
            shard.overflow_origin_last_codes >= last_size
        ):
            raise ValueError(f"overflow_origin_last_codes must be in [0, {last_size}).")
        prefixes = shard.overflow_bucket_key_prefixes
        if np.any(prefixes < 0) or np.any(prefixes % last_size != 0):
            raise ValueError(
                "overflow_bucket_key_prefixes must be nonnegative multiples "
                "of the last-layer size."
            )
        if np.any(prefixes[1:] < prefixes[:-1]):
            raise ValueError("overflow rows must be ordered by nondecreasing band.")
        if overflow_count == 0:
            return order_hashes

        if not np.all(
            overflow_band_bucket_mask(shard.bucket_keys, prefixes, last_size)
        ):
            raise ValueError("bucket_keys contain a band outside the overflow shard.")
        origin_keys = prefixes + shard.overflow_origin_last_codes
        origin_positions = np.searchsorted(shard.bucket_keys, origin_keys)
        in_bounds = origin_positions < shard.bucket_keys.size
        if not np.all(in_bounds) or not np.all(
            shard.bucket_keys[origin_positions] == origin_keys
        ):
            raise ValueError("bucket_keys do not cover every overflow origin.")
        if np.any(shard.bucket_counts[origin_positions] <= shard.config.capacity):
            raise ValueError("every overflow origin must exceed capacity.")
        return order_hashes

    def _resolve_validated_shard(
        self,
        shard: CollisionWorkShard,
        candidates: np.ndarray,
        overflow_order_hashes: np.ndarray,
        collect_diagnostics: bool = True,
    ) -> tuple[
        CollisionShardResult,
        Optional[IterativeCollisionDiagnostics],
    ]:
        """Resolve validated candidates for a compact complete-band shard."""
        if shard.overflow_rows.size == 0:
            result = self._empty_shard_result(shard)
            diagnostics = (
                IterativeCollisionDiagnostics(
                    assignment_rounds=np.empty(0, dtype=np.int64),
                    round_stats=(),
                )
                if collect_diagnostics
                else None
            )
            return result, diagnostics

        local_rows = shard.overflow_rows
        local_hashes = overflow_order_hashes
        local_prefixes = shard.overflow_bucket_key_prefixes
        local_origins = shard.overflow_origin_last_codes
        band_starts = _run_starts(local_prefixes)
        band_stops = np.append(band_starts[1:], local_rows.size)

        resolved_last_codes = np.empty(local_rows.size, dtype=np.int64)
        slot_indices = np.empty(local_rows.size, dtype=np.int64)
        unresolved_rows = np.empty(local_rows.size, dtype=np.int64)
        unresolved_count = 0
        final_bucket_key_parts = []
        final_bucket_count_parts = []
        assignment_rounds = (
            np.empty(local_rows.size, dtype=np.int64) if collect_diagnostics else None
        )
        round_stats = []
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )
        processed_count = 0
        for band_start, band_stop in zip(band_starts, band_stops):
            band_start_int = int(band_start)
            band_stop_int = int(band_stop)
            band_result, band_diagnostics = self._resolve_one_band(
                shard,
                local_rows[band_start_int:band_stop_int],
                local_hashes[band_start_int:band_stop_int],
                local_prefixes[band_start_int:band_stop_int],
                local_origins[band_start_int:band_stop_int],
                candidates[band_start_int:band_stop_int],
            )
            resolved_last_codes[band_start_int:band_stop_int] = (
                band_result.resolved_last_codes
            )
            slot_indices[band_start_int:band_stop_int] = band_result.slot_indices
            band_unresolved_count = band_result.unresolved_rows.size
            unresolved_rows[
                unresolved_count : unresolved_count + band_unresolved_count
            ] = band_result.unresolved_rows
            unresolved_count += band_unresolved_count
            final_bucket_key_parts.append(band_result.final_bucket_keys)
            final_bucket_count_parts.append(band_result.final_bucket_counts)
            if assignment_rounds is not None:
                assignment_rounds[band_start_int:band_stop_int] = (
                    band_diagnostics.assignment_rounds
                )
                round_stats.extend(band_diagnostics.round_stats)
            processed_count += band_stop_int - band_start_int
            progress.log(processed_count)

        unresolved_rows.resize(unresolved_count, refcheck=False)
        result = CollisionShardResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_bucket_keys=np.concatenate(final_bucket_key_parts),
            final_bucket_counts=np.concatenate(final_bucket_count_parts),
        )
        diagnostics = (
            IterativeCollisionDiagnostics(
                assignment_rounds=assignment_rounds,
                round_stats=tuple(round_stats),
            )
            if assignment_rounds is not None
            else None
        )
        return result, diagnostics

    def _resolve_one_band(
        self,
        shard: CollisionWorkShard,
        overflow_rows: np.ndarray,
        item_ties: np.ndarray,
        prefixes: np.ndarray,
        origin_last_codes: np.ndarray,
        candidates: np.ndarray,
    ) -> tuple[CollisionShardResult, IterativeCollisionDiagnostics]:
        """Resolve one nonempty band with synchronous multi-round arbitration."""
        capacity = shard.config.capacity
        last_size = shard.config.layer_sizes[-1]
        prefix = int(prefixes[0])
        band_key = prefix // last_size
        bucket_start = int(np.searchsorted(shard.bucket_keys, prefix))
        bucket_stop = int(np.searchsorted(shard.bucket_keys, prefix + last_size))
        initial_occupancy = np.zeros(last_size, dtype=np.int64)
        bucket_last_codes = shard.bucket_keys[bucket_start:bucket_stop] - prefix
        initial_occupancy[bucket_last_codes] = np.minimum(
            shard.bucket_counts[bucket_start:bucket_stop], capacity
        )
        occupancy = initial_occupancy.copy()

        item_count = overflow_rows.shape[0]
        item_positions = np.arange(item_count, dtype=np.int64)
        proposal_order = np.lexsort((overflow_rows, item_ties, origin_last_codes))

        assigned = np.zeros(item_count, dtype=bool)
        resolved = origin_last_codes.copy()
        assignment_rounds = np.zeros(item_count, dtype=np.int64)
        assignment_priorities = np.full(item_count, -1, dtype=np.int64)
        round_stats = []
        round_index = 0

        while True:
            round_index += 1
            active = proposal_order[~assigned[proposal_order]]
            active_items = int(active.size)
            active_edges = 0
            proposal_item_parts = []
            proposal_priority_parts = []
            proposal_target_parts = []
            if active.size:
                # Every (origin, priority) group proposes at most `capacity`
                # valid edges: rank valid edges within each origin run.
                run_starts = _run_starts(origin_last_codes[active])
                run_lengths = np.diff(np.append(run_starts, active.size))
                for priority in range(candidates.shape[1]):
                    targets = candidates[active, priority]
                    valid = occupancy[targets] < capacity
                    active_edges += int(valid.sum())
                    valid_counts = np.cumsum(valid, dtype=np.int64)
                    run_offsets = np.repeat(
                        valid_counts[run_starts] - valid[run_starts], run_lengths
                    )
                    keep = valid & (valid_counts - run_offsets <= capacity)
                    if not np.any(keep):
                        continue
                    proposal_item_parts.append(active[keep])
                    proposal_priority_parts.append(
                        np.full(int(keep.sum()), priority, dtype=np.int64)
                    )
                    proposal_target_parts.append(targets[keep])

            if proposal_item_parts:
                proposal_items = np.concatenate(proposal_item_parts)
                proposal_priorities = np.concatenate(proposal_priority_parts)
                proposal_targets = np.concatenate(proposal_target_parts)
                target_order = np.lexsort(
                    (
                        overflow_rows[proposal_items],
                        item_ties[proposal_items],
                        proposal_priorities,
                        proposal_targets,
                    )
                )
                ordered_targets = proposal_targets[target_order]
                accepted_in_order = _ranks_within_runs(ordered_targets) < (
                    capacity - occupancy[ordered_targets]
                )
                accepted_edges = target_order[accepted_in_order]
                accepted_items = proposal_items[accepted_edges]
                accepted_priorities = proposal_priorities[accepted_edges]
                accepted_targets = proposal_targets[accepted_edges]

                item_order = np.lexsort(
                    (
                        accepted_targets,
                        overflow_rows[accepted_items],
                        item_ties[accepted_items],
                        accepted_priorities,
                        accepted_items,
                    )
                )
                ordered_items = accepted_items[item_order]
                first_accept = np.empty(item_order.size, dtype=bool)
                first_accept[0] = True
                first_accept[1:] = ordered_items[1:] != ordered_items[:-1]
                winner_edges = item_order[first_accept]
                winner_items = accepted_items[winner_edges]
                winner_priorities = accepted_priorities[winner_edges]
                winner_targets = accepted_targets[winner_edges]
            else:
                proposal_items = np.empty(0, dtype=np.int64)
                accepted_items = np.empty(0, dtype=np.int64)
                winner_items = np.empty(0, dtype=np.int64)
                winner_priorities = np.empty(0, dtype=np.int64)
                winner_targets = np.empty(0, dtype=np.int64)

            winner_count = int(winner_items.size)
            round_stats.append(
                IterativeCollisionRoundStats(
                    band_key=band_key,
                    round_index=round_index,
                    active_items=active_items,
                    active_edges=active_edges,
                    proposals=int(proposal_items.size),
                    accepts=int(accepted_items.size),
                    winners=winner_count,
                )
            )
            if winner_count == 0:
                break

            target_additions = np.bincount(winner_targets, minlength=last_size).astype(
                np.int64, copy=False
            )
            occupancy += target_additions
            assigned[winner_items] = True
            resolved[winner_items] = winner_targets
            assignment_rounds[winner_items] = round_index
            assignment_priorities[winner_items] = winner_priorities

        slot_indices = np.empty(item_count, dtype=np.int64)
        relocated_items = item_positions[assigned]
        if relocated_items.size:
            relocated_order = np.lexsort(
                (
                    overflow_rows[relocated_items],
                    item_ties[relocated_items],
                    assignment_priorities[relocated_items],
                    assignment_rounds[relocated_items],
                    resolved[relocated_items],
                )
            )
            ordered_relocated = relocated_items[relocated_order]
            ordered_targets = resolved[ordered_relocated]
            slot_indices[ordered_relocated] = (
                initial_occupancy[ordered_targets]
                + _ranks_within_runs(ordered_targets)
                + 1
            )

        unresolved_items = item_positions[~assigned]
        if unresolved_items.size:
            unresolved_order = np.lexsort(
                (
                    overflow_rows[unresolved_items],
                    item_ties[unresolved_items],
                    origin_last_codes[unresolved_items],
                )
            )
            ordered_unresolved = unresolved_items[unresolved_order]
            ordered_origins = origin_last_codes[ordered_unresolved]
            slot_indices[ordered_unresolved] = (
                initial_occupancy[ordered_origins]
                + _ranks_within_runs(ordered_origins)
                + 1
            )
            occupancy += np.bincount(ordered_origins, minlength=last_size).astype(
                np.int64, copy=False
            )

        occupied_last_codes = np.flatnonzero(occupancy)
        return CollisionShardResult(
            resolved_last_codes=resolved,
            slot_indices=slot_indices,
            unresolved_rows=overflow_rows[~assigned],
            final_bucket_keys=prefix + occupied_last_codes,
            final_bucket_counts=occupancy[occupied_last_codes],
        ), IterativeCollisionDiagnostics(
            assignment_rounds=assignment_rounds,
            round_stats=tuple(round_stats),
        )
