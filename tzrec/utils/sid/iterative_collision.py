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
    concat_ranges,
    lookup_sorted,
    ranks_within_runs,
    run_starts,
    stable_order_hash,
)

_OCCUPANCY_CELLS = 1 << 25
_PROPOSAL_CELLS = 1 << 24
_INT64_MAX = np.iinfo(np.int64).max


@dataclass(frozen=True)
class _BatchResolution:
    """Assignments and final occupancy for one batch of whole prefixes."""

    resolved_last_codes: np.ndarray
    slot_indices: np.ndarray
    unresolved_rows: np.ndarray
    bucket_keys: np.ndarray
    bucket_counts: np.ndarray


class IterativeCollisionResolver(CollisionResolver):
    """Resolve overflow with deterministic synchronous batch arbitration.

    Each SID prefix is independent. In every round, each unplaced
    overflow row proposes to its candidates that still have room, except that
    within one origin bucket only the first ``capacity`` rows (in stable item
    order) with room at a candidate position propose at that position, which
    bounds the proposals of large same-origin groups whose candidate lists are
    near-identical. A target accepts up to its remaining capacity, ordered by
    candidate position then stable item order, and each row commits only its
    earliest-position accepted proposal. Slots accepted for a row that
    committed elsewhere reopen in the next round. A row left with no candidate
    that has room keeps its origin SID over capacity.

    Acceptance favors candidate proximity: a bucket goes to the row that ranks
    it earlier even when that row has other options, so this policy can leave
    more rows unresolved than first-fit placement. Within a bucket, relocated
    rows are numbered after its initial occupancy by round, candidate position
    and stable item order.

    Published append-state items participate only through their occupancy. They
    are never proposal rows, so they cannot be moved, and an already
    over-capacity published bucket retains its complete count.

    Prefixes are resolved in batches that share dense occupancy and run their
    rounds together; a prefix that stops proposing never changes again, so the
    result is the same as resolving each prefix alone.
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
        candidates = self._validate_candidate_last_codes(plan, candidate_codes)
        if plan.overflow_rows.size == 0:
            return self._build_no_overflow_result(plan, collect_grouping)

        capacity = plan.config.capacity
        prior_counts = plan.prior_bucket_counts
        combined_counts = prior_counts + plan.bucket_counts
        initial_counts = np.maximum(prior_counts, np.minimum(combined_counts, capacity))
        last_size = plan.config.layer_sizes[-1]

        resolved_last_codes = plan.original_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        unresolved_parts = []
        touched_key_parts = []
        touched_count_parts = []
        prefix_starts = run_starts(plan.overflow_bucket_key_prefixes)
        _, in_overflow_prefix = lookup_sorted(
            plan.overflow_bucket_key_prefixes[prefix_starts] // last_size,
            plan.bucket_keys // last_size,
        )
        prefix_stops = np.append(prefix_starts[1:], plan.overflow_rows.shape[0])
        prefix_count = prefix_starts.shape[0]
        batch_prefixes = max(1, _OCCUPANCY_CELLS // last_size)
        batch_rows = max(1, _PROPOSAL_CELLS // max(1, candidates.shape[1]))
        occupancy = np.zeros(
            min(batch_prefixes, prefix_count) * last_size, dtype=np.int64
        )
        progress = ProgressLogger(
            "Resolving collision overflow",
            start_n=0,
            miniters=self._progress_interval,
        )

        first_prefix = 0
        while first_prefix < prefix_count:
            start = int(prefix_starts[first_prefix])
            end_prefix = int(
                np.searchsorted(prefix_stops, start + batch_rows, side="right")
            )
            end_prefix = min(
                max(end_prefix, first_prefix + 1), first_prefix + batch_prefixes
            )
            stop = int(prefix_stops[end_prefix - 1])
            batch = self._resolve_batch(
                plan,
                initial_counts,
                occupancy,
                slice(start, stop),
                candidates[start:stop],
            )
            rows = plan.overflow_rows[start:stop]
            resolved_last_codes[rows] = batch.resolved_last_codes
            slot_indices[rows] = batch.slot_indices
            unresolved_parts.append(batch.unresolved_rows)
            touched_key_parts.append(batch.bucket_keys)
            touched_count_parts.append(batch.bucket_counts)
            progress.log(stop)
            first_prefix = end_prefix

        unresolved_rows = np.concatenate(unresolved_parts)
        (
            final_bucket_keys,
            final_bucket_counts,
            final_collision_buckets,
            max_final_bucket_size,
        ) = self._summarize_final_buckets(
            plan,
            initial_counts,
            in_overflow_prefix,
            np.concatenate(touched_key_parts),
            np.concatenate(touched_count_parts),
            collect_grouping,
        )
        stats = CollisionResolutionStats(
            total_items=plan.item_count,
            raw_collision_buckets=int((combined_counts > capacity).sum()),
            final_collision_buckets=final_collision_buckets,
            relocated_count=plan.overflow_rows.size - unresolved_rows.size,
            unresolved_count=unresolved_rows.size,
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

    def _resolve_batch(
        self,
        plan: CollisionPlan,
        initial_counts: np.ndarray,
        occupancy: np.ndarray,
        span: slice,
        candidates: np.ndarray,
    ) -> _BatchResolution:
        """Resolve a contiguous run of whole prefixes through shared rounds.

        Buckets are addressed by a batch-local key, ``prefix position * last_size
        + last code``, into ``occupancy``, which must be all zero on entry and
        is zeroed again on return.

        Args:
            plan: Grouping and append-aware occupancy plan.
            initial_counts: Capped per-bucket counts aligned with
                ``plan.bucket_keys``.
            occupancy: Reusable dense occupancy buffer of at least ``last_size``
                cells per prefix in ``span``.
            span: Slice of the overflow-aligned plan arrays covering whole
                prefixes.
            candidates: Int64 candidate last codes of the rows in ``span``.

        Returns:
            Per-row assignments in ``span`` order and the batch's final
            occupancy keyed by global bucket key.

        Raises:
            ValueError: If the batch is too large to pack arbitration keys.
        """
        capacity = plan.config.capacity
        last_size = plan.config.layer_sizes[-1]
        overflow_rows = plan.overflow_rows[span]
        row_count = overflow_rows.shape[0]
        prefix_starts = run_starts(plan.overflow_bucket_key_prefixes[span])
        prefix_count = prefix_starts.shape[0]
        candidate_count = candidates.shape[1]
        prefix_first_keys = plan.overflow_bucket_key_prefixes[span][prefix_starts]
        prefix_positions = np.arange(prefix_count, dtype=np.int64)
        key_bases = np.repeat(
            prefix_positions * last_size, np.diff(np.append(prefix_starts, row_count))
        )
        origin_keys = key_bases + plan.overflow_origin_last_codes[span]
        if prefix_count * last_size * candidate_count * row_count > _INT64_MAX:
            raise ValueError(
                f"{row_count} overflow rows with {candidate_count} candidates in "
                "one prefix batch are too many to pack arbitration keys."
            )

        seeded_parts = []
        for bucket_keys, bucket_counts in (
            (plan.prior.bucket_keys, plan.prior.bucket_counts),
            (plan.bucket_keys, initial_counts),
        ):
            starts = np.searchsorted(bucket_keys, prefix_first_keys)
            lengths = (
                np.searchsorted(bucket_keys, prefix_first_keys + last_size) - starts
            )
            selected = concat_ranges(starts, lengths)
            owners = np.repeat(prefix_positions, lengths)
            keys = (
                owners * last_size + bucket_keys[selected] - prefix_first_keys[owners]
            )
            occupancy[keys] = bucket_counts[selected]
            seeded_parts.append(keys)

        item_ranks = np.empty(row_count, dtype=np.int64)
        item_ranks[
            np.lexsort((overflow_rows, stable_order_hash(plan.overflow_item_ids[span])))
        ] = np.arange(row_count)

        resolved_keys = origin_keys.copy()
        assigned = np.zeros(row_count, dtype=bool)
        slot_indices = np.empty(row_count, dtype=np.int64)
        pending = np.arange(row_count, dtype=np.int64)
        targets = candidates + key_bases[:, None]
        while True:
            has_room = occupancy[targets] < capacity
            # Occupancy only grows, so a row without room anywhere is final.
            reachable = has_room.any(axis=1)
            if not reachable.all():
                pending = pending[reachable]
                targets = targets[reachable]
                has_room = has_room[reachable]
            if pending.size == 0:
                break

            origin_starts = run_starts(origin_keys[pending])
            origin_lengths = np.diff(np.append(origin_starts, pending.shape[0]))
            room_ranks = np.cumsum(has_room, axis=0, dtype=np.int32)
            room_ranks -= np.repeat(
                room_ranks[origin_starts] - has_room[origin_starts],
                origin_lengths,
                axis=0,
            )
            proposals = np.flatnonzero(has_room & (room_ranks <= capacity))
            proposal_rows, proposal_positions = np.divmod(proposals, candidate_count)
            proposal_items = pending[proposal_rows]
            proposal_targets = targets.ravel()[proposals]

            target_order = np.argsort(
                (proposal_targets * candidate_count + proposal_positions) * row_count
                + item_ranks[proposal_items]
            )
            ordered_targets = proposal_targets[target_order]
            accepted = np.zeros(proposal_items.shape[0], dtype=bool)
            accepted[
                target_order[
                    ranks_within_runs(ordered_targets)
                    < capacity - occupancy[ordered_targets]
                ]
            ] = True
            # Proposals are row-major, so a row's first accepted one is its best.
            accepted_proposals = np.flatnonzero(accepted)
            winners = accepted_proposals[run_starts(proposal_items[accepted_proposals])]
            won = np.zeros(proposal_items.shape[0], dtype=bool)
            won[winners] = True
            winners = target_order[won[target_order]]
            winner_items = proposal_items[winners]
            winner_targets = proposal_targets[winners]

            slot_indices[winner_items] = (
                occupancy[winner_targets] + ranks_within_runs(winner_targets) + 1
            )
            np.add.at(occupancy, winner_targets, 1)
            assigned[winner_items] = True
            resolved_keys[winner_items] = winner_targets
            still_pending = ~assigned[pending]
            pending = pending[still_pending]
            targets = targets[still_pending]

        unresolved = ~assigned
        unresolved_keys = origin_keys[unresolved]
        slot_indices[unresolved] = (
            occupancy[unresolved_keys] + ranks_within_runs(unresolved_keys) + 1
        )
        np.add.at(occupancy, unresolved_keys, 1)

        touched_keys = np.unique(np.concatenate((*seeded_parts, resolved_keys)))
        touched_counts = occupancy[touched_keys]
        occupancy[touched_keys] = 0
        occupied = touched_counts > 0
        touched_keys = touched_keys[occupied]
        owners = touched_keys // last_size
        return _BatchResolution(
            resolved_last_codes=resolved_keys - key_bases,
            slot_indices=slot_indices,
            unresolved_rows=overflow_rows[unresolved],
            bucket_keys=prefix_first_keys[owners] + touched_keys - owners * last_size,
            bucket_counts=touched_counts[occupied],
        )
