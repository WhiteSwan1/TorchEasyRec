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

from dataclasses import dataclass
from typing import Optional

import numpy as np

from tzrec.utils.logging_util import ProgressLogger
from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionResult,
    CollisionResolutionStats,
    CollisionResolver,
    lookup_sorted,
    stable_order_hash,
)


@dataclass(frozen=True)
class _BandResolution:
    """Assignments and final occupancy for one independently resolved band."""

    resolved_last_codes: np.ndarray
    slot_indices: np.ndarray
    unresolved_rows: np.ndarray
    bucket_keys: np.ndarray
    bucket_counts: np.ndarray


def _run_starts(ordered_values: np.ndarray) -> np.ndarray:
    """Return start offsets of equal-value runs in a nonempty sorted array."""
    return np.concatenate(
        (
            np.asarray([0], dtype=np.int64),
            np.flatnonzero(ordered_values[1:] != ordered_values[:-1]) + 1,
        )
    )


def _ranks_within_runs(ordered_values: np.ndarray) -> np.ndarray:
    """Return zero-based ranks within equal-value runs in a sorted array."""
    starts = _run_starts(ordered_values)
    ranks = np.arange(ordered_values.shape[0], dtype=np.int64)
    return ranks - np.repeat(
        starts, np.diff(np.append(starts, ordered_values.shape[0]))
    )


class IterativeCollisionResolver(CollisionResolver):
    """Resolve overflow with deterministic SQL-inspired batch arbitration.

    Each SID prefix band is independent. In every round, non-full targets
    receive proposals ordered by candidate priority and stable item identity.
    A target accepts up to its remaining capacity, then each item commits only
    its best accepted proposal. Vacancies from that per-item arbitration become
    available in the next round.

    Published append-state items participate only through their occupancy. They
    are never proposal rows, so they cannot be moved, and an already
    over-capacity published bucket retains its complete count.
    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows through synchronous proposal rounds.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_codes: Ordered last-layer candidates aligned with
                ``plan.overflow_rows``. It may be omitted only without overflow.
            collect_grouping: Whether to retain final bucket metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidates are absent while overflow rows exist.
        """
        if candidate_codes is None:
            if plan.overflow_rows.size:
                raise ValueError(
                    "candidate_codes are required when the collision plan has "
                    "overflow rows."
                )
            candidate_codes = np.empty((0, 0), dtype=np.int64)
        candidates = self._validate_candidate_last_codes(plan, candidate_codes)
        if plan.overflow_rows.size == 0:
            return self._build_no_overflow_result(plan, collect_grouping)

        capacity = plan.config.capacity
        prior_counts = plan.prior_bucket_counts
        combined_counts = prior_counts + plan.bucket_counts
        initial_counts = np.maximum(prior_counts, np.minimum(combined_counts, capacity))
        last_size = plan.config.layer_sizes[-1]
        overflow_band_ids = np.unique(plan.overflow_bucket_key_prefixes // last_size)
        _, in_overflow_band = lookup_sorted(
            overflow_band_ids, plan.bucket_keys // last_size
        )

        resolved_last_codes = plan.original_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_parts = []
        final_band_counts: dict[int, int] = {}
        order_hashes = stable_order_hash(plan.overflow_item_ids)
        prefixes = plan.overflow_bucket_key_prefixes
        band_starts = _run_starts(prefixes)
        band_stops = np.append(band_starts[1:], prefixes.shape[0])
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        for band_start, band_stop in zip(band_starts, band_stops):
            start = int(band_start)
            stop = int(band_stop)
            band = self._resolve_band(
                plan,
                initial_counts,
                plan.overflow_rows[start:stop],
                order_hashes[start:stop],
                prefixes[start:stop],
                plan.overflow_origin_last_codes[start:stop],
                candidates[start:stop],
            )
            rows = plan.overflow_rows[start:stop]
            resolved_last_codes[rows] = band.resolved_last_codes
            slot_indices[rows] = band.slot_indices
            if band.unresolved_rows.size:
                unresolved_parts.append(band.unresolved_rows)
            final_band_counts.update(
                zip(band.bucket_keys.tolist(), band.bucket_counts.tolist())
            )
            progress.log(stop)

        unresolved_rows = (
            np.concatenate(unresolved_parts)
            if unresolved_parts
            else np.empty(0, dtype=np.int64)
        )
        (
            final_bucket_keys,
            final_bucket_counts,
            final_collision_buckets,
            max_final_bucket_size,
        ) = self._summarize_final_buckets(
            plan,
            initial_counts,
            in_overflow_band,
            final_band_counts,
            collect_grouping,
        )
        stats = CollisionResolutionStats(
            total_items=plan.item_count,
            raw_collision_buckets=int((combined_counts > capacity).sum()),
            final_collision_buckets=final_collision_buckets,
            relocated_count=int(plan.overflow_rows.size - unresolved_rows.size),
            unresolved_count=int(unresolved_rows.size),
            max_final_bucket_size=max_final_bucket_size,
        )
        return CollisionResolutionResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_bucket_keys=final_bucket_keys,
            final_bucket_counts=final_bucket_counts,
            grouping_collected=collect_grouping,
            stats=stats,
        )

    def _resolve_band(
        self,
        plan: CollisionPlan,
        initial_counts: np.ndarray,
        overflow_rows: np.ndarray,
        item_ties: np.ndarray,
        prefixes: np.ndarray,
        origin_last_codes: np.ndarray,
        candidates: np.ndarray,
    ) -> _BandResolution:
        """Resolve one nonempty prefix band through multi-round arbitration."""
        capacity = plan.config.capacity
        last_size = plan.config.layer_sizes[-1]
        prefix = int(prefixes[0])
        occupancy = np.zeros(last_size, dtype=np.int64)

        prior_start = int(np.searchsorted(plan.prior.bucket_keys, prefix))
        prior_stop = int(np.searchsorted(plan.prior.bucket_keys, prefix + last_size))
        prior_keys = plan.prior.bucket_keys[prior_start:prior_stop]
        occupancy[prior_keys - prefix] = plan.prior.bucket_counts[
            prior_start:prior_stop
        ]

        bucket_start = int(np.searchsorted(plan.bucket_keys, prefix))
        bucket_stop = int(np.searchsorted(plan.bucket_keys, prefix + last_size))
        current_keys = plan.bucket_keys[bucket_start:bucket_stop]
        occupancy[current_keys - prefix] = initial_counts[bucket_start:bucket_stop]
        initial_occupancy = occupancy.copy()

        item_count = overflow_rows.shape[0]
        item_positions = np.arange(item_count, dtype=np.int64)
        proposal_order = np.lexsort((overflow_rows, item_ties, origin_last_codes))
        assigned = np.zeros(item_count, dtype=bool)
        resolved = origin_last_codes.copy()
        assignment_rounds = np.zeros(item_count, dtype=np.int64)
        assignment_priorities = np.full(item_count, -1, dtype=np.int64)
        round_index = 0

        while True:
            round_index += 1
            active = proposal_order[~assigned[proposal_order]]
            proposal_item_parts = []
            proposal_priority_parts = []
            proposal_target_parts = []
            if active.size:
                origin_starts = _run_starts(origin_last_codes[active])
                origin_lengths = np.diff(np.append(origin_starts, active.size))
                for priority in range(candidates.shape[1]):
                    targets = candidates[active, priority]
                    valid = occupancy[targets] < capacity
                    valid_counts = np.cumsum(valid, dtype=np.int64)
                    origin_offsets = np.repeat(
                        valid_counts[origin_starts] - valid[origin_starts],
                        origin_lengths,
                    )
                    keep = valid & (valid_counts - origin_offsets <= capacity)
                    if not np.any(keep):
                        continue
                    proposal_item_parts.append(active[keep])
                    proposal_priority_parts.append(
                        np.full(int(keep.sum()), priority, dtype=np.int64)
                    )
                    proposal_target_parts.append(targets[keep])

            if not proposal_item_parts:
                break

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
            accepted = target_order[
                _ranks_within_runs(ordered_targets)
                < capacity - occupancy[ordered_targets]
            ]
            accepted_items = proposal_items[accepted]
            accepted_priorities = proposal_priorities[accepted]
            accepted_targets = proposal_targets[accepted]
            if accepted_items.size == 0:
                break

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
            winners = item_order[first_accept]
            winner_items = accepted_items[winners]
            winner_priorities = accepted_priorities[winners]
            winner_targets = accepted_targets[winners]
            if winner_items.size == 0:
                break

            occupancy += np.bincount(winner_targets, minlength=last_size).astype(
                np.int64, copy=False
            )
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

        occupied = np.flatnonzero(occupancy)
        return _BandResolution(
            resolved_last_codes=resolved,
            slot_indices=slot_indices,
            unresolved_rows=overflow_rows[~assigned],
            bucket_keys=prefix + occupied,
            bucket_counts=occupancy[occupied],
        )
