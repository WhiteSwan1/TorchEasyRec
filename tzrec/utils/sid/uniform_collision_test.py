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

from tzrec.utils.sid import uniform_collision
from tzrec.utils.sid.collision import (
    CollisionResolutionConfig,
    PriorOccupancy,
    build_resolved_item_grouping,
    prepare_collision_plan,
    sid_prefix_ids,
    stable_order_hash,
)
from tzrec.utils.sid.uniform_collision import UniformCollisionResolver
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


def _rotation(prefix_id, last_size):
    return int(stable_order_hash(np.asarray([prefix_id], dtype=np.int64))[0]) % int(
        last_size
    )


def _reference(plan, codes, item_ids):
    """Place overflow rows one at a time on the least-loaded rotated code."""
    last_size = plan.config.layer_sizes[-1]
    prefixes = sid_prefix_ids(np.asarray(codes), plan.config.layer_sizes)
    moving = set(plan.overflow_rows.tolist())
    loads = {}
    for row, prefix in enumerate(prefixes.tolist()):
        if row not in moving:
            key = prefix * last_size + int(codes[row][-1])
            loads[key] = loads.get(key, 0) + 1
    for key, count in zip(
        plan.prior.bucket_keys.tolist(), plan.prior.bucket_counts.tolist()
    ):
        loads[key] = loads.get(key, 0) + count
    hashes = stable_order_hash(np.asarray(item_ids)).tolist()
    codes_out, slots_out = {}, {}
    for row in sorted(moving, key=lambda r: (int(prefixes[r]), hashes[r], r)):
        prefix = int(prefixes[row])
        rotation = _rotation(prefix, last_size)
        code = min(
            range(last_size),
            key=lambda c: (
                loads.get(prefix * last_size + c, 0),
                (c - rotation) % last_size,
            ),
        )
        key = prefix * last_size + code
        loads[key] = loads.get(key, 0) + 1
        codes_out[row] = code
        slots_out[row] = loads[key]
    return codes_out, slots_out


class UniformCollisionResolverTest(unittest.TestCase):
    def test_no_overflow_keeps_assignments(self) -> None:
        plan = _plan((3,), 1, ["a", "b"], [[0], [1]])

        result = UniformCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 1])
        np.testing.assert_array_equal(result.slot_indices, [1, 1])
        self.assertEqual(result.stats.relocated_count, 0)

    def test_rejects_candidate_codes(self) -> None:
        plan = _plan((4,), 1, [10, 11], [[0], [0]])

        with self.assertRaisesRegex(ValueError, "does not accept candidate_codes"):
            UniformCollisionResolver().resolve(plan, np.asarray([[1]]))

    def test_overflow_takes_empty_codes_in_rotated_order(self) -> None:
        plan = _plan((8,), 1, [10, 11, 12], [[3], [3], [3]])
        rotation = _rotation(0, 8)
        rotated = [(rotation + step) % 8 for step in range(8)]
        expected = [code for code in rotated if code != 3][:2]

        result = UniformCollisionResolver().resolve(plan)

        moved = sorted(
            plan.overflow_rows.tolist(),
            key=lambda row: stable_order_hash(np.asarray([10, 11, 12]))[row],
        )
        self.assertEqual([int(result.resolved_last_codes[r]) for r in moved], expected)
        np.testing.assert_array_equal(result.slot_indices[plan.overflow_rows], [1, 1])
        self.assertEqual(result.stats.relocated_count, 2)
        self.assertEqual(result.stats.max_final_bucket_size, 1)

    def test_empty_codes_fill_before_partly_loaded_ones(self) -> None:
        plan = _plan((4,), 2, [10, 11, 12, 13, 14, 20], [[0]] * 5 + [[1]])

        result = UniformCollisionResolver().resolve(plan)

        moved = result.resolved_last_codes[plan.overflow_rows].tolist()
        self.assertTrue({2, 3} <= set(moved))
        np.testing.assert_array_equal(
            np.sort(result.slot_indices[plan.overflow_rows]), [1, 1, 2]
        )
        self.assertEqual(result.stats.unresolved_count, 0)
        self.assertEqual(result.stats.max_final_bucket_size, 2)

    def test_full_prefix_spreads_excess_past_capacity(self) -> None:
        plan = _plan((2,), 1, [10, 11, 12, 13, 14], [[0]] * 5)

        result = UniformCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(np.sort(result.final_bucket_counts), [2, 3])
        self.assertEqual(result.stats.max_final_bucket_size, 3)
        self.assertEqual(result.stats.relocated_count, 1)
        self.assertEqual(result.stats.unresolved_count, 3)
        grouping = build_resolved_item_grouping(plan, result)
        np.testing.assert_array_equal(np.sort(grouping.row_order), np.arange(5))

    def test_single_code_space_keeps_rows_in_the_only_bucket(self) -> None:
        plan = _plan((1,), 1, [10, 11, 12], [[0]] * 3)

        result = UniformCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(result.resolved_last_codes, [0, 0, 0])
        np.testing.assert_array_equal(np.sort(result.slot_indices), [1, 2, 3])
        self.assertEqual(result.stats.unresolved_count, 2)

    def test_append_never_shrinks_published_buckets(self) -> None:
        plan = _plan((4,), 2, [10, 11], [[0], [0]], prior=_prior([0, 1], [3, 1]))

        result = UniformCollisionResolver().resolve(plan)

        np.testing.assert_array_equal(
            np.sort(result.resolved_last_codes[plan.overflow_rows]), [2, 3]
        )
        np.testing.assert_array_equal(result.slot_indices[plan.overflow_rows], [1, 1])
        np.testing.assert_array_equal(result.final_bucket_keys, [0, 1, 2, 3])
        np.testing.assert_array_equal(result.final_bucket_counts, [3, 1, 1, 1])

    def test_append_into_full_prefix_continues_published_slots(self) -> None:
        plan = _plan((2,), 1, [10], [[0]], prior=_prior([0, 1], [1, 1]))

        result = UniformCollisionResolver().resolve(plan)

        self.assertEqual(int(result.slot_indices[0]), 2)
        self.assertEqual(result.stats.unresolved_count, 1)
        np.testing.assert_array_equal(np.sort(result.final_bucket_counts), [1, 2])

    def test_collect_grouping_false_omits_only_bucket_metadata(self) -> None:
        plan = _plan((4,), 1, [10, 11], [[0], [0]])

        grouped = UniformCollisionResolver().resolve(plan)
        rate_only = UniformCollisionResolver().resolve(plan, collect_grouping=False)

        self.assertFalse(rate_only.grouping_collected)
        np.testing.assert_array_equal(rate_only.final_bucket_keys, [])
        np.testing.assert_array_equal(
            rate_only.resolved_last_codes, grouped.resolved_last_codes
        )
        self.assertEqual(rate_only.stats, grouped.stats)

    @parameterized.expand(
        [
            ("integer", np.arange(24, dtype=np.int64)),
            ("string", np.asarray([f"item-{index}" for index in range(24)])),
        ],
        name_func=parameterized_name_func,
    )
    def test_assignments_are_deterministic_under_input_reordering(
        self, _case_name, item_ids
    ) -> None:
        codes = np.stack([np.arange(24) % 2, np.arange(24) % 3], axis=1)

        def assignments(order):
            ordered_ids = item_ids[order]
            result = UniformCollisionResolver().resolve(
                _plan((2, 4), 1, ordered_ids, codes[order])
            )
            return {
                item_id: (
                    int(result.resolved_last_codes[row]),
                    int(result.slot_indices[row]),
                )
                for row, item_id in enumerate(ordered_ids.tolist())
            }

        expected = assignments(np.arange(24))
        actual = assignments(np.random.default_rng(3).permutation(24))

        self.assertEqual(actual, expected)

    @parameterized.expand(
        [
            ("one_prefix_per_batch", 1, 1),
            ("small_batches", 7, 5),
        ],
        name_func=parameterized_name_func,
    )
    def test_batching_matches_single_batch(
        self, _name, window_cells, dense_cells
    ) -> None:
        rng = np.random.default_rng(2026)
        for trial in range(40):
            layer_sizes = (rng.integers(1, 6), rng.integers(2, 9))
            item_count = rng.integers(1, 120)
            codes = np.stack(
                [rng.integers(0, size, item_count) for size in layer_sizes], axis=1
            )
            plan = _plan(layer_sizes, rng.integers(1, 3), np.arange(item_count), codes)

            batched = UniformCollisionResolver().resolve(plan)
            with (
                mock.patch.object(uniform_collision, "_WINDOW_CELLS", window_cells),
                mock.patch.object(uniform_collision, "_DENSE_CELLS", dense_cells),
            ):
                split = UniformCollisionResolver().resolve(plan)

            with self.subTest(trial=trial):
                np.testing.assert_array_equal(
                    split.resolved_last_codes, batched.resolved_last_codes
                )
                np.testing.assert_array_equal(split.slot_indices, batched.slot_indices)
                np.testing.assert_array_equal(
                    split.final_bucket_counts, batched.final_bucket_counts
                )
                self.assertEqual(split.stats, batched.stats)

    def test_random_plans_match_least_loaded_reference(self) -> None:
        rng = np.random.default_rng(7)
        for trial in range(150):
            last_size = rng.integers(1, 9)
            layer_sizes = (rng.integers(1, 4), last_size)
            capacity = rng.integers(1, 3)
            item_count = rng.integers(1, 40)
            codes = np.stack(
                [
                    np.minimum(rng.geometric(0.4, item_count) - 1, size - 1)
                    for size in layer_sizes
                ],
                axis=1,
            )
            prior = None
            if trial % 3 == 0:
                keys = np.unique(rng.integers(0, layer_sizes[0] * last_size, 4))
                prior = _prior(keys, rng.integers(1, capacity + 3, keys.shape[0]))
            item_ids = np.arange(item_count) * 7 + 1000
            plan = _plan(layer_sizes, capacity, item_ids, codes, prior=prior)

            result = UniformCollisionResolver().resolve(plan)
            expected_codes, expected_slots = _reference(plan, codes, item_ids)

            with self.subTest(trial=trial):
                for row in plan.overflow_rows.tolist():
                    self.assertEqual(
                        int(result.resolved_last_codes[row]), expected_codes[row]
                    )
                    self.assertEqual(int(result.slot_indices[row]), expected_slots[row])
                self.assertEqual(
                    result.stats.unresolved_count,
                    sum(slot > capacity for slot in expected_slots.values()),
                )
                build_resolved_item_grouping(plan, result)


if __name__ == "__main__":
    unittest.main()
