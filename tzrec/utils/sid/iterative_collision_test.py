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

import numpy as np
from parameterized import parameterized

from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    KnnCollisionResolver,
    PriorOccupancy,
    build_resolved_item_grouping,
    prepare_collision_plan,
)
from tzrec.utils.sid.iterative_collision import IterativeCollisionResolver
from tzrec.utils.test_util import parameterized_name_func


def _plan(layer_sizes, capacity, item_ids, codes, prior=None):
    return prepare_collision_plan(
        np.asarray(item_ids),
        np.asarray(codes, dtype=np.int64),
        CollisionResolutionConfig(layer_sizes, capacity),
        prior=prior,
    )


def _prior(keys, counts):
    return PriorOccupancy(
        np.asarray(keys, dtype=np.int64), np.asarray(counts, dtype=np.int64)
    )


class IterativeCollisionResolverTest(unittest.TestCase):
    def test_no_overflow_needs_no_candidates(self) -> None:
        plan = _plan((3,), 1, ["a", "b"], [[0], [1]])

        result = IterativeCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 1])
        np.testing.assert_array_equal(result.slot_indices, [1, 1])
        np.testing.assert_array_equal(result.unresolved_rows, [])
        np.testing.assert_array_equal(result.final_bucket_counts, [1, 1])

    def test_multi_round_arbitration_fills_item_vacancy(self) -> None:
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

        np.testing.assert_array_equal(
            iterative.resolved_last_codes[plan.overflow_rows], [4, 3]
        )
        np.testing.assert_array_equal(
            first_fit.resolved_last_codes[plan.overflow_rows], [3, 1]
        )
        self.assertEqual(iterative.stats.relocated_count, 2)
        self.assertEqual(iterative.stats.unresolved_count, 0)
        self.assertEqual(first_fit.stats.relocated_count, 1)
        self.assertEqual(first_fit.stats.unresolved_count, 1)

    @parameterized.expand(
        [("top100", 100), ("top200", 200)],
        name_func=parameterized_name_func,
    )
    def test_candidate_width_is_consumed_without_truncation(
        self, _case_name, candidate_count
    ) -> None:
        plan = _plan(
            (256,),
            1,
            [10],
            [[0]],
            prior=_prior(
                np.arange(candidate_count, dtype=np.int64),
                np.ones(candidate_count, dtype=np.int64),
            ),
        )
        candidates = np.arange(1, candidate_count + 1, dtype=np.int64)[None, :]

        result = IterativeCollisionResolver().resolve(plan, candidates)

        self.assertEqual(int(result.resolved_last_codes[0]), candidate_count)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 0)

    def test_exhausted_candidates_keep_origin_and_dense_slots(self) -> None:
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
        np.testing.assert_array_equal(result.final_bucket_counts, [2, 1, 1])
        grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(np.sort(grouping.row_order), np.arange(4))

    @parameterized.expand(
        [
            ("integer", np.arange(12, dtype=np.int64)),
            ("string", np.asarray([f"item-{index}" for index in range(12)])),
        ],
        name_func=parameterized_name_func,
    )
    def test_assignments_are_deterministic_under_input_reordering(
        self, _case_name, item_ids
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

    def test_append_preserves_prior_over_capacity_and_moves_only_new_rows(self) -> None:
        plan = _plan(
            (4,),
            2,
            [10, 11],
            [[0], [0]],
            prior=_prior([0, 1], [3, 1]),
        )
        candidates = np.asarray([[1, 2], [1, 2]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        self.assertEqual(plan.overflow_rows.size, 2)
        self.assertEqual(result.stats.relocated_count, 2)
        self.assertEqual(result.stats.unresolved_count, 0)
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2])
        np.testing.assert_array_equal(result.final_bucket_counts, [3, 2, 1])
        self.assertEqual(result.stats.max_final_bucket_size, 3)
        np.testing.assert_array_equal(
            np.sort(result.slot_indices[plan.overflow_rows]), [1, 2]
        )

    def test_append_unresolved_continues_after_full_prior_count(self) -> None:
        plan = _plan(
            (2,),
            1,
            [10],
            [[0]],
            prior=_prior([0, 1], [3, 1]),
        )

        result = IterativeCollisionResolver().resolve(
            plan, np.asarray([[1, 0]], dtype=np.int64)
        )

        self.assertEqual(result.stats.unresolved_count, 1)
        self.assertEqual(int(result.slot_indices[0]), 4)
        np.testing.assert_array_equal(result.final_bucket_counts, [4, 1])

    def test_append_multi_band_preserves_prior_only_destinations(self) -> None:
        plan = _plan(
            (2, 4),
            1,
            [10, 11, 20, 21],
            [[0, 0], [0, 0], [1, 0], [1, 0]],
            prior=_prior([1, 6], [1, 1]),
        )
        candidates = np.asarray([[1, 2], [2, 1]], dtype=np.int64)

        result = IterativeCollisionResolver().resolve(plan, candidates)

        np.testing.assert_array_equal(
            result.resolved_last_codes[plan.overflow_rows], [2, 1]
        )
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2, 4, 5, 6])
        np.testing.assert_array_equal(result.final_bucket_counts, [1, 1, 1, 1, 1, 1])
        np.testing.assert_array_equal(result.slot_indices[plan.overflow_rows], [1, 1])

    def test_collect_grouping_false_omits_only_bucket_metadata(self) -> None:
        plan = _plan((2,), 1, [10, 11], [[0], [0]])

        result = IterativeCollisionResolver().resolve(
            plan,
            np.asarray([[1]], dtype=np.int64),
            collect_grouping=False,
        )

        self.assertFalse(result.grouping_collected)
        np.testing.assert_array_equal(result.final_bucket_keys, [])
        np.testing.assert_array_equal(result.final_bucket_counts, [])
        self.assertEqual(result.stats.relocated_count, 1)

    def test_requires_candidates_for_overflow(self) -> None:
        plan = _plan((2,), 1, [10, 11], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "candidate_codes are required"):
            IterativeCollisionResolver().resolve(plan)


if __name__ == "__main__":
    unittest.main()
