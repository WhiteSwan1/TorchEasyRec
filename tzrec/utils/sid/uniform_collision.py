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

"""Candidate-free least-loaded collision resolution for semantic IDs."""

from typing import Optional

import numpy as np

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

_WINDOW_CELLS = 1 << 22
_DENSE_CELLS = 1 << 22


class UniformCollisionResolver(CollisionResolver):
    """Spread overflow evenly over the least-loaded codes of each SID prefix.

    Each bucket keeps the rows :func:`prepare_collision_plan` retains. The
    overflow rows of a prefix take, in stable item order, the prefix's
    least-loaded last-layer codes one at a time, so a prefix with room never
    leaves a row over capacity. Ties between equally loaded codes follow a
    per-prefix rotated code order, which keeps relocated rows from piling onto
    the same codes in every prefix. When a prefix has more rows than room, the
    filling continues past capacity, so its largest bucket is as small as its
    published occupancy allows; rows left over capacity count as unresolved
    even when they changed code.

    Published append-state items participate only through their occupancy. They
    are never overflow rows, so they cannot be moved.
    """

    def resolve(
        self,
        plan: CollisionPlan,
        candidate_codes: Optional[np.ndarray] = None,
        collect_grouping: bool = True,
    ) -> CollisionResolutionResult:
        """Resolve overflow rows by least-loaded filling within each prefix.

        Args:
            plan: Grouping and append-aware occupancy plan.
            candidate_codes: Must be ``None``; this strategy uses no candidates.
            collect_grouping: Whether to retain final bucket metadata.

        Returns:
            Resolved last-layer codes, slot indices, and summary statistics.

        Raises:
            ValueError: If candidate codes are supplied.
        """
        if candidate_codes is not None:
            raise ValueError(
                "UniformCollisionResolver does not accept candidate_codes."
            )
        overflow_count = plan.overflow_rows.shape[0]
        if overflow_count == 0:
            return self._build_no_overflow_result(plan, collect_grouping)

        capacity = plan.config.capacity
        last_size = plan.config.layer_sizes[-1]
        prior_counts = plan.prior_bucket_counts
        combined_counts = prior_counts + plan.bucket_counts
        initial_counts = np.maximum(prior_counts, np.minimum(combined_counts, capacity))

        prefix_starts = run_starts(plan.overflow_bucket_key_prefixes)
        prefix_rows = np.diff(np.append(prefix_starts, overflow_count))
        prefix_keys = plan.overflow_bucket_key_prefixes[prefix_starts]
        prefix_ids = prefix_keys // last_size
        _, in_overflow_prefix = lookup_sorted(prefix_ids, plan.bucket_keys // last_size)

        occupied_keys = plan.bucket_keys[in_overflow_prefix]
        occupied_counts = initial_counts[in_overflow_prefix]
        if not plan.prior.is_empty:
            touched_prior = plan.prior.restrict_to_prefixes(prefix_ids, last_size)
            _, known = lookup_sorted(occupied_keys, touched_prior.bucket_keys)
            occupied_keys = np.concatenate(
                (occupied_keys, touched_prior.bucket_keys[~known])
            )
            occupied_counts = np.concatenate(
                (occupied_counts, touched_prior.bucket_counts[~known])
            )
            order = np.argsort(occupied_keys, kind="stable")
            occupied_keys = occupied_keys[order]
            occupied_counts = occupied_counts[order]
        nonzero = occupied_counts > 0
        occupied_keys = occupied_keys[nonzero]
        occupied_counts = occupied_counts[nonzero]
        occupied_starts = np.searchsorted(occupied_keys, prefix_keys)
        occupied_lengths = (
            np.searchsorted(occupied_keys, prefix_keys + last_size) - occupied_starts
        )
        # uint64 modulus: a Python-int modulus would promote a uint64 scalar to float.
        rotations = (stable_order_hash(prefix_ids) % np.uint64(last_size)).astype(
            np.int64
        )

        prefix_of_row = np.repeat(
            np.arange(prefix_starts.shape[0], dtype=np.int64), prefix_rows
        )
        ordered_rows = plan.overflow_rows[
            np.lexsort(
                (
                    plan.overflow_rows,
                    stable_order_hash(plan.overflow_item_ids),
                    prefix_of_row,
                )
            )
        ]
        del prefix_of_row
        assigned_codes = np.empty(overflow_count, dtype=np.int64)
        assigned_slots = np.empty(overflow_count, dtype=np.int64)

        # Rows that all fit on empty codes need no load tracking.
        fits_empty = prefix_rows <= last_size - occupied_lengths
        fast_prefixes = np.flatnonzero(fits_empty)
        fast_key_parts = []
        if fast_prefixes.size:
            windows = np.minimum(
                last_size, prefix_rows[fast_prefixes] + occupied_lengths[fast_prefixes]
            )
            window_ends = np.cumsum(windows)
            first = 0
            while first < fast_prefixes.shape[0]:
                end = int(
                    np.searchsorted(
                        window_ends,
                        int(window_ends[first] - windows[first]) + _WINDOW_CELLS,
                        side="right",
                    )
                )
                end = max(end, first + 1)
                fast_key_parts.append(
                    self._fill_empty_codes(
                        fast_prefixes[first:end],
                        windows[first:end],
                        prefix_rows,
                        prefix_starts,
                        prefix_keys,
                        rotations,
                        occupied_keys,
                        occupied_starts,
                        occupied_lengths,
                        last_size,
                        assigned_codes,
                        assigned_slots,
                    )
                )
                first = end

        general_prefixes = np.flatnonzero(~fits_empty)
        general_key_parts = []
        chunk = max(1, _DENSE_CELLS // last_size)
        for first in range(0, general_prefixes.shape[0], chunk):
            general_key_parts.append(
                self._water_fill(
                    general_prefixes[first : first + chunk],
                    prefix_rows,
                    prefix_starts,
                    prefix_keys,
                    rotations,
                    occupied_keys,
                    occupied_counts,
                    occupied_starts,
                    occupied_lengths,
                    last_size,
                    assigned_codes,
                    assigned_slots,
                )
            )

        resolved_last_codes = plan.original_last_codes.copy()
        slot_indices = plan.initial_slot_indices.copy()
        resolved_last_codes[ordered_rows] = assigned_codes
        slot_indices[ordered_rows] = assigned_slots
        unresolved_rows = ordered_rows[assigned_slots > capacity]

        general_keys = np.concatenate((np.empty(0, dtype=np.int64), *general_key_parts))
        final_counts = occupied_counts.copy()
        positions, found = lookup_sorted(occupied_keys, general_keys)
        np.add.at(final_counts, positions[found], 1)
        new_keys, new_counts = np.unique(general_keys[~found], return_counts=True)
        fast_keys = np.concatenate((np.empty(0, dtype=np.int64), *fast_key_parts))
        (
            final_bucket_keys,
            final_bucket_counts,
            final_collision_buckets,
            max_final_bucket_size,
        ) = self._summarize_final_buckets(
            plan,
            initial_counts,
            in_overflow_prefix,
            np.concatenate((occupied_keys, fast_keys, new_keys)),
            np.concatenate(
                (final_counts, np.ones(fast_keys.shape[0], dtype=np.int64), new_counts)
            ),
            collect_grouping,
        )
        return CollisionResolutionResult(
            resolved_last_codes=resolved_last_codes,
            slot_indices=slot_indices,
            unresolved_rows=unresolved_rows,
            final_bucket_keys=final_bucket_keys,
            final_bucket_counts=final_bucket_counts,
            grouping_collected=collect_grouping,
            stats=CollisionResolutionStats(
                total_items=plan.item_count,
                raw_collision_buckets=int((combined_counts > capacity).sum()),
                final_collision_buckets=final_collision_buckets,
                relocated_count=overflow_count - unresolved_rows.shape[0],
                unresolved_count=unresolved_rows.shape[0],
                max_final_bucket_size=max_final_bucket_size,
            ),
        )

    @staticmethod
    def _fill_empty_codes(
        prefixes: np.ndarray,
        windows: np.ndarray,
        prefix_rows: np.ndarray,
        prefix_starts: np.ndarray,
        prefix_keys: np.ndarray,
        rotations: np.ndarray,
        occupied_keys: np.ndarray,
        occupied_starts: np.ndarray,
        occupied_lengths: np.ndarray,
        last_size: int,
        assigned_codes: np.ndarray,
        assigned_slots: np.ndarray,
    ) -> np.ndarray:
        """Give every row of prefixes with enough empty codes one empty code.

        The first ``m`` empty codes in rotated order lie within the first
        ``m + occupied`` rotated positions, so only that window is scanned.

        Args:
            prefixes: Indices of the prefixes to fill, ascending.
            windows: Rotated positions to scan for each of ``prefixes``.
            prefix_rows: Overflow row count of every prefix.
            prefix_starts: Offset of every prefix's rows in stable item order.
            prefix_keys: Bucket key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            occupied_keys: Sorted keys of occupied buckets in overflow prefixes.
            occupied_starts: First ``occupied_keys`` position of every prefix.
            occupied_lengths: Occupied bucket count of every prefix.
            last_size: Cardinality of the last SID layer.
            assigned_codes: Per-row last codes in stable item order, updated.
            assigned_slots: Per-row slots in stable item order, updated.

        Returns:
            Global keys of the buckets the rows now occupy.
        """
        owners = np.repeat(np.arange(prefixes.shape[0], dtype=np.int64), windows)
        codes = rotations[prefixes][owners] + concat_ranges(
            np.zeros(prefixes.shape[0], dtype=np.int64), windows
        )
        codes %= last_size
        keys = prefix_keys[prefixes][owners] + codes
        _, taken = lookup_sorted(
            occupied_keys[
                int(occupied_starts[prefixes[0]]) : int(
                    occupied_starts[prefixes[-1]] + occupied_lengths[prefixes[-1]]
                )
            ],
            keys,
        )
        free = np.flatnonzero(~taken)
        free_owners = owners[free]
        free = free[ranks_within_runs(free_owners) < prefix_rows[prefixes][free_owners]]
        rows = concat_ranges(prefix_starts[prefixes], prefix_rows[prefixes])
        assigned_codes[rows] = codes[free]
        assigned_slots[rows] = 1
        return keys[free]

    @staticmethod
    def _water_fill(
        prefixes: np.ndarray,
        prefix_rows: np.ndarray,
        prefix_starts: np.ndarray,
        prefix_keys: np.ndarray,
        rotations: np.ndarray,
        occupied_keys: np.ndarray,
        occupied_counts: np.ndarray,
        occupied_starts: np.ndarray,
        occupied_lengths: np.ndarray,
        last_size: int,
        assigned_codes: np.ndarray,
        assigned_slots: np.ndarray,
    ) -> np.ndarray:
        """Fill prefixes with fewer empty codes than overflow rows.

        Taking the least-loaded ``(load, rotated position)`` for each row in
        turn yields the smallest pairs ``(level, position)`` with ``level`` at
        or above the code's load, so the final fill level is found by bisection
        and the pairs are enumerated directly.

        Args:
            prefixes: Indices of the prefixes to fill.
            prefix_rows: Overflow row count of every prefix.
            prefix_starts: Offset of every prefix's rows in stable item order.
            prefix_keys: Bucket key of code 0 in every prefix.
            rotations: Tie-break rotation of every prefix.
            occupied_keys: Sorted keys of occupied buckets in overflow prefixes.
            occupied_counts: Initial counts aligned with ``occupied_keys``.
            occupied_starts: First ``occupied_keys`` position of every prefix.
            occupied_lengths: Occupied bucket count of every prefix.
            last_size: Cardinality of the last SID layer.
            assigned_codes: Per-row last codes in stable item order, updated.
            assigned_slots: Per-row slots in stable item order, updated.

        Returns:
            Global bucket key of every assigned row.
        """
        group = prefixes.shape[0]
        loads = np.zeros((group, last_size), dtype=np.int64)
        lengths = occupied_lengths[prefixes]
        selected = concat_ranges(occupied_starts[prefixes], lengths)
        owners = np.repeat(np.arange(group, dtype=np.int64), lengths)
        loads[
            owners,
            (
                occupied_keys[selected]
                - prefix_keys[prefixes][owners]
                - rotations[prefixes][owners]
            )
            % last_size,
        ] = occupied_counts[selected]

        rows = prefix_rows[prefixes]
        low = loads.min(axis=1)
        high = loads.max(axis=1) + -(-rows // last_size)
        while np.any(low < high):
            middle = (low + high) // 2
            enough = np.maximum(middle[:, None] + 1 - loads, 0).sum(axis=1) >= rows
            high = np.where(enough, middle, high)
            low = np.where(enough, low, middle + 1)
        level = low
        below = np.maximum(level[:, None] - loads, 0)
        at_level = loads <= level[:, None]
        at_level &= np.cumsum(at_level, axis=1) <= (rows - below.sum(axis=1))[:, None]

        below_cells = np.flatnonzero(below)
        below_counts = below.ravel()[below_cells]
        pair_levels = np.concatenate(
            (
                concat_ranges(loads.ravel()[below_cells], below_counts),
                np.repeat(level, at_level.sum(axis=1)),
            )
        )
        pair_owners, pair_positions = np.divmod(
            np.concatenate(
                (np.repeat(below_cells, below_counts), np.flatnonzero(at_level))
            ),
            last_size,
        )
        order = np.lexsort((pair_positions, pair_levels, pair_owners))
        pair_owners = pair_owners[order]
        pair_codes = (
            pair_positions[order] + rotations[prefixes][pair_owners]
        ) % last_size
        targets = concat_ranges(prefix_starts[prefixes], rows)
        assigned_codes[targets] = pair_codes
        assigned_slots[targets] = pair_levels[order] + 1
        return prefix_keys[prefixes][pair_owners] + pair_codes
