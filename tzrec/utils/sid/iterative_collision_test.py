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

import unittest
from unittest import mock

import numpy as np
from parameterized import parameterized

from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    CollisionWorkShard,
    KnnCollisionResolver,
    build_resolved_item_grouping,
    prepare_collision_plan,
    stable_order_hash,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.test_util import parameterized_name_func


def _plan(layer_sizes, capacity, item_ids, codes):
    return prepare_collision_plan(
        np.asarray(item_ids),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
    )


def _plan_shard(plan, start, stop):
    prefixes = plan.overflow_bucket_key_prefixes[start:stop]
    if prefixes.size:
        last_size = plan.config.layer_sizes[-1]
        bucket_mask = np.isin(
            plan.bucket_keys // last_size, np.unique(prefixes // last_size)
        )
    else:
        bucket_mask = np.zeros(plan.bucket_keys.shape[0], dtype=bool)
    return (
        CollisionWorkShard(
            overflow_rows=plan.overflow_rows[start:stop],
            overflow_bucket_key_prefixes=prefixes,
            overflow_origin_last_codes=plan.overflow_origin_last_codes[start:stop],
            bucket_keys=plan.bucket_keys[bucket_mask],
            bucket_counts=plan.bucket_counts[bucket_mask],
            config=plan.config,
        ),
        stable_order_hash(plan.overflow_item_ids)[start:stop],
    )


def _shard(
    layer_sizes,
    capacity,
    overflow_rows,
    order_hashes,
    prefixes,
    origins,
    bucket_keys,
    bucket_counts,
):
    return (
        CollisionWorkShard(
            overflow_rows=np.asarray(overflow_rows, dtype=np.int64),
            overflow_bucket_key_prefixes=np.asarray(prefixes, dtype=np.int64),
            overflow_origin_last_codes=np.asarray(origins, dtype=np.int64),
            bucket_keys=np.asarray(bucket_keys, dtype=np.int64),
            bucket_counts=np.asarray(bucket_counts, dtype=np.int64),
            config=CollisionResolutionConfig(layer_sizes, capacity),
        ),
        np.asarray(order_hashes, dtype=np.uint64),
    )


class IterativeCollisionResolverTest(unittest.TestCase):
    def test_item_arbitration_vacancy_is_filled_next_round(self) -> None:
        plan = _plan((3,), 1, [10, 11, 12], [[0], [0], [0]])
        candidates = np.tile(np.asarray([1, 2], dtype=np.int64), (2, 1))
        shard, order_hashes = _plan_shard(plan, 0, 2)

        result, diagnostics = (
            IterativeCollisionResolver().resolve_shard_with_diagnostics(
                shard, candidates, order_hashes
            )
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [1, 2])
        np.testing.assert_array_equal(result.slot_indices, [1, 1])
        np.testing.assert_array_equal(diagnostics.assignment_rounds, [1, 2])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        self.assertEqual(
            [
                (
                    stats.active_items,
                    stats.active_edges,
                    stats.proposals,
                    stats.accepts,
                    stats.winners,
                )
                for stats in diagnostics.round_stats
            ],
            [
                (2, 4, 2, 2, 1),
                (1, 1, 1, 1, 1),
                (0, 0, 0, 0, 0),
            ],
        )

    def test_origin_priority_proposals_are_throttled_by_capacity(self) -> None:
        shard, order_hashes = _shard(
            (4,),
            2,
            overflow_rows=[10, 11, 12],
            order_hashes=[0, 1, 2],
            prefixes=[0, 0, 0],
            origins=[0, 0, 0],
            bucket_keys=[0],
            bucket_counts=[5],
        )
        candidates = np.asarray([[1], [2], [3]], dtype=np.int64)

        result, diagnostics = (
            IterativeCollisionResolver().resolve_shard_with_diagnostics(
                shard, candidates, order_hashes
            )
        )

        np.testing.assert_array_equal(diagnostics.assignment_rounds, [1, 1, 2])
        self.assertEqual(
            [(stats.proposals, stats.winners) for stats in diagnostics.round_stats],
            [(2, 2), (1, 1), (0, 0)],
        )

    def test_target_prefers_lower_candidate_priority(self) -> None:
        shard, order_hashes = _shard(
            (4,),
            1,
            overflow_rows=[10, 20],
            order_hashes=[0, 1],
            prefixes=[0, 0],
            origins=[0, 1],
            bucket_keys=[0, 1, 2],
            bucket_counts=[2, 2, 1],
        )
        candidates = np.asarray([[2, 3], [3, 2]], dtype=np.int64)

        result, diagnostics = (
            IterativeCollisionResolver().resolve_shard_with_diagnostics(
                shard, candidates, order_hashes
            )
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 3])
        np.testing.assert_array_equal(diagnostics.assignment_rounds, [0, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [10])
        self.assertEqual(diagnostics.round_stats[0].accepts, 1)
        self.assertEqual(diagnostics.round_stats[0].winners, 1)

    def test_existing_target_members_take_capacity_before_proposals(self) -> None:
        shard, order_hashes = _shard(
            (4,),
            2,
            overflow_rows=[10, 20],
            order_hashes=[0, 1],
            prefixes=[0, 0],
            origins=[0, 2],
            bucket_keys=[0, 1, 2],
            bucket_counts=[3, 1, 3],
        )
        candidates = np.asarray([[1], [1]], dtype=np.int64)

        result, diagnostics = (
            IterativeCollisionResolver().resolve_shard_with_diagnostics(
                shard, candidates, order_hashes
            )
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [1, 2])
        np.testing.assert_array_equal(result.unresolved_rows, [20])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 2, 3])
        self.assertEqual(diagnostics.round_stats[0].accepts, 1)

    def test_duplicate_candidates_remain_distinct_edges(self) -> None:
        shard, order_hashes = _shard(
            (4,),
            2,
            overflow_rows=[10, 20],
            order_hashes=[0, 1],
            prefixes=[0, 0],
            origins=[0, 2],
            bucket_keys=[0, 2, 3],
            bucket_counts=[3, 3, 2],
        )
        candidates = np.asarray([[1, 1], [3, 1]], dtype=np.int64)

        result, diagnostics = (
            IterativeCollisionResolver().resolve_shard_with_diagnostics(
                shard, candidates, order_hashes
            )
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [1, 1])
        np.testing.assert_array_equal(diagnostics.assignment_rounds, [1, 2])
        self.assertEqual(diagnostics.round_stats[0].proposals, 3)
        self.assertEqual(diagnostics.round_stats[0].accepts, 2)
        self.assertEqual(diagnostics.round_stats[0].winners, 1)

    def test_iterative_and_first_fit_can_choose_different_assignments(self) -> None:
        plan = _plan(
            (5,),
            1,
            [10, 11, 20, 21, 30],
            [[0], [0], [1], [1], [2]],
        )
        np.testing.assert_array_equal(plan.overflow_origin_last_codes, [0, 1])
        candidates = np.asarray([[2, 3, 4], [3, 2, 2]], dtype=np.int64)

        iterative = IterativeCollisionResolver().resolve(plan, candidates)
        first_fit = KnnCollisionResolver().resolve(plan, candidates)

        self.assertEqual(iterative.stats.relocated_count, 2)
        self.assertEqual(iterative.stats.unresolved_count, 0)
        self.assertEqual(first_fit.stats.relocated_count, 1)
        self.assertEqual(first_fit.stats.unresolved_count, 1)
        np.testing.assert_array_equal(
            iterative.resolved_last_codes[plan.overflow_rows], [4, 3]
        )
        np.testing.assert_array_equal(
            first_fit.resolved_last_codes[plan.overflow_rows], [3, 1]
        )

    def test_candidate_exhaustion_keeps_origin_and_compacts_slots(self) -> None:
        plan = _plan((3,), 1, [10, 11, 12, 13], [[0], [0], [0], [0]])
        candidates = np.asarray([[1], [2], [0]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(
            result.resolved_last_codes[plan.overflow_rows], [1, 2, 0]
        )
        np.testing.assert_array_equal(
            result.slot_indices[plan.overflow_rows], [1, 1, 2]
        )
        np.testing.assert_array_equal(result.unresolved_rows, [plan.overflow_rows[2]])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2])
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1, 1])
        self.assertEqual(result.stats.final_collision_buckets, 1)
        self.assertEqual(result.stats.max_final_bucket_size, 2)
        grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(np.sort(grouping.row_order), np.arange(4))

    @parameterized.expand(
        [
            ("integer", np.arange(12, dtype=np.int64)),
            ("string", np.asarray([f"item-{index}" for index in range(12)])),
        ],
        name_func=parameterized_name_func,
    )
    def test_resolution_is_deterministic_under_input_reordering(
        self, _, item_ids
    ) -> None:
        codes = np.zeros((item_ids.size, 1), dtype=np.int64)

        def assignments(order):
            ordered_ids = item_ids[order]
            plan = _plan((4,), 2, ordered_ids, codes[order])
            candidates = np.tile(
                np.asarray([1, 2, 3, 1], dtype=np.int64),
                (plan.overflow_rows.size, 1),
            )
            result = IterativeCollisionResolver().resolve(plan, candidates)
            unresolved = set(result.unresolved_rows.tolist())
            return {
                item_id: (
                    int(result.resolved_last_codes[row]),
                    int(result.slot_indices[row]),
                    row in unresolved,
                )
                for row, item_id in enumerate(ordered_ids.tolist())
            }

        expected = assignments(np.arange(item_ids.size))
        actual = assignments(np.random.default_rng(7).permutation(item_ids.size))

        self.assertEqual(actual, expected)

    def test_complete_band_shards_match_full_multi_band_result(self) -> None:
        plan = _plan(
            (2, 3),
            1,
            [10, 11, 12, 20, 21, 22],
            [[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0]],
        )
        candidates = np.tile(np.asarray([1, 2], dtype=np.int64), (4, 1))
        resolver = IterativeCollisionResolver()
        full_result = resolver.resolve(plan, candidates)
        prefixes = plan.overflow_bucket_key_prefixes
        split = int(np.flatnonzero(prefixes[1:] != prefixes[:-1])[0] + 1)
        self.assertEqual(split, 2)

        shard_results = []
        for start, stop in ((0, split), (split, prefixes.size)):
            shard, shard_order_hashes = _plan_shard(plan, start, stop)
            shard_results.append(
                (
                    shard,
                    resolver.resolve_shard(
                        shard,
                        candidates[start:stop],
                        shard_order_hashes,
                    ),
                )
            )
        combined_shard, combined_order_hashes = _plan_shard(plan, 0, prefixes.size)
        combined_result = resolver.resolve_shard(
            combined_shard, candidates, combined_order_hashes
        )
        diagnostic_result, diagnostics = resolver.resolve_shard_with_diagnostics(
            combined_shard,
            candidates,
            combined_order_hashes,
        )

        for shard, shard_result in shard_results:
            np.testing.assert_array_equal(
                full_result.resolved_last_codes[shard.overflow_rows],
                shard_result.resolved_last_codes,
            )
            np.testing.assert_array_equal(
                full_result.slot_indices[shard.overflow_rows],
                shard_result.slot_indices,
            )
        np.testing.assert_array_equal(
            combined_result.resolved_last_codes,
            full_result.resolved_last_codes[plan.overflow_rows],
        )
        np.testing.assert_array_equal(
            combined_result.slot_indices,
            full_result.slot_indices[plan.overflow_rows],
        )
        np.testing.assert_array_equal(
            combined_result.final_bucket_keys, full_result.final_bucket_keys
        )
        np.testing.assert_array_equal(
            combined_result.final_bucket_counts, full_result.final_bucket_counts
        )
        for field in (
            "resolved_last_codes",
            "slot_indices",
            "unresolved_rows",
            "final_bucket_keys",
            "final_bucket_counts",
        ):
            np.testing.assert_array_equal(
                getattr(combined_result, field),
                getattr(diagnostic_result, field),
            )
        self.assertEqual(diagnostics.assignment_rounds.shape, (4,))
        self.assertGreater(len(diagnostics.round_stats), 0)

    def test_full_resolver_reports_progress_at_band_thresholds(self) -> None:
        plan = _plan(
            (2, 3),
            1,
            [10, 11, 12, 20, 21, 22],
            [[0, 0], [0, 0], [0, 0], [1, 0], [1, 0], [1, 0]],
        )
        candidates = np.tile(np.asarray([1, 2], dtype=np.int64), (4, 1))

        with mock.patch(
            "tzrec.utils.sid.iterative_collision.ProgressLogger"
        ) as progress_cls:
            IterativeCollisionResolver(progress_interval=3).resolve(plan, candidates)

        progress_cls.assert_called_once_with(
            "Resolving collision overflow", start_n=0, miniters=3
        )
        self.assertEqual(
            progress_cls.return_value.log.call_args_list,
            [mock.call(2), mock.call(4)],
        )

    def test_empty_overflow_supports_full_and_worker_apis(self) -> None:
        plan = _plan((3,), 1, ["a", "b"], [[0], [1]])
        resolver = IterativeCollisionResolver()

        result = resolver.resolve(plan)
        shard, order_hashes = _plan_shard(plan, 0, 0)
        shard_result, diagnostics = resolver.resolve_shard_with_diagnostics(
            shard,
            np.empty((0, 7), dtype=np.int64),
            order_hashes,
        )

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 1])
        np.testing.assert_array_equal(result.slot_indices, [1, 1])
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(diagnostics.round_stats, ())
        self.assertEqual(shard.overflow_rows.size, 0)
        self.assertEqual(shard_result.final_bucket_keys.size, 0)

    def test_collect_grouping_false_omits_only_bucket_metadata(self) -> None:
        plan = _plan((2,), 1, [10, 11], [[0], [0]])
        result = IterativeCollisionResolver().resolve(
            plan, np.asarray([[1]], dtype=np.int64), collect_grouping=False
        )

        np.testing.assert_array_equal(
            result.resolved_last_codes[plan.overflow_rows], [1]
        )
        self.assertFalse(result.grouping_collected)
        self.assertEqual(result.final_bucket_keys.size, 0)
        self.assertEqual(result.final_bucket_counts.size, 0)

    @parameterized.expand(
        [
            ("one_dimensional", np.asarray([1], dtype=np.int64), "2-D"),
            (
                "wrong_row_count",
                np.empty((0, 1), dtype=np.int64),
                "row-aligned",
            ),
            ("float_dtype", np.asarray([[1.0]]), "integer dtype"),
            ("negative", np.asarray([[-1]], dtype=np.int64), r"\[0, 2\)"),
            ("out_of_range", np.asarray([[2]], dtype=np.int64), r"\[0, 2\)"),
        ],
        name_func=parameterized_name_func,
    )
    def test_full_resolver_rejects_invalid_candidates(
        self, _, candidates, message
    ) -> None:
        plan = _plan((2,), 1, [10, 11], [[0], [0]])

        with self.assertRaisesRegex((TypeError, ValueError), message):
            IterativeCollisionResolver().resolve(plan, candidates)

    def test_full_resolver_requires_candidates_for_overflow(self) -> None:
        plan = _plan((2,), 1, [10, 11], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_codes are required"):
            IterativeCollisionResolver().resolve(plan)

    def test_worker_rejects_invalid_candidates_and_shard(self) -> None:
        shard, order_hashes = _shard(
            (2,),
            1,
            overflow_rows=[10],
            order_hashes=[0],
            prefixes=[0],
            origins=[0],
            bucket_keys=[0],
            bucket_counts=[2],
        )
        resolver = IterativeCollisionResolver()

        with self.assertRaisesRegex(ValueError, "row-aligned with overflow_rows"):
            resolver.resolve_shard(
                shard,
                np.empty((0, 1), dtype=np.int64),
                order_hashes,
            )
        with self.assertRaisesRegex(ValueError, r"\[0, 2\)"):
            resolver.resolve_shard(
                shard,
                np.asarray([[2]], dtype=np.int64),
                order_hashes,
            )
        with self.assertRaisesRegex(TypeError, "uint64"):
            resolver.resolve_shard(
                shard,
                np.asarray([[1]], dtype=np.int64),
                order_hashes.astype(np.int64),
            )

        invalid_shard, invalid_hashes = _shard(
            (2,),
            1,
            overflow_rows=[10],
            order_hashes=[0],
            prefixes=[1],
            origins=[0],
            bucket_keys=[0],
            bucket_counts=[2],
        )
        with self.assertRaisesRegex(ValueError, "multiples"):
            resolver.resolve_shard(
                invalid_shard,
                np.asarray([[1]], dtype=np.int64),
                invalid_hashes,
            )


if __name__ == "__main__":
    unittest.main()
