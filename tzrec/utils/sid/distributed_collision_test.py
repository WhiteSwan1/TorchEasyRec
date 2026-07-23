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

import os
import unittest
from datetime import timedelta
from unittest import mock

import numpy as np
from parameterized import parameterized

from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionStats,
    CollisionShardResult,
)
from tzrec.utils.sid.collision_sharding import CollisionBandShard
from tzrec.utils.sid.distributed_collision import (
    DistributedCollisionContext,
    broadcast_collision_stats,
    build_collision_work,
    gather_collision_shard_results,
    initialize_distributed_collision,
    receive_collision_work,
    send_collision_work,
)
from tzrec.utils.test_util import parameterized_name_func


def _collision_plan() -> CollisionPlan:
    return CollisionPlan(
        item_count=6,
        original_last_codes=np.asarray([0, 0, 0, 0, 0, 0], dtype=np.int64),
        origin_bucket_indices=np.asarray([0, 0, 0, 1, 2, 2], dtype=np.int64),
        initial_slot_indices=np.asarray([1, 2, 3, 1, 1, 2], dtype=np.int64),
        bucket_keys=np.asarray([0, 4, 8], dtype=np.int64),
        bucket_counts=np.asarray([3, 1, 2], dtype=np.int64),
        overflow_rows=np.asarray([1, 2, 5], dtype=np.int64),
        overflow_item_ids=np.asarray(["one", "two", "five"]),
        overflow_bucket_key_prefixes=np.asarray([0, 0, 8], dtype=np.int64),
        overflow_origin_last_codes=np.asarray([0, 0, 0], dtype=np.int64),
        config=CollisionResolutionConfig((3, 4), 1),
    )


def _candidate_codes() -> np.ndarray:
    return np.asarray([[1, 2], [2, 3], [1, 3]], dtype=np.int64)


def _result() -> CollisionShardResult:
    return CollisionShardResult(
        resolved_last_codes=np.asarray([0, 0, 0], dtype=np.int64),
        slot_indices=np.asarray([2, 3, 2], dtype=np.int64),
        unresolved_rows=np.asarray([1, 2, 5], dtype=np.int64),
        final_bucket_keys=np.asarray([0, 8], dtype=np.int64),
        final_bucket_counts=np.asarray([3, 2], dtype=np.int64),
    )


def _stats() -> CollisionResolutionStats:
    return CollisionResolutionStats(
        total_items=100,
        raw_collision_buckets=8,
        final_collision_buckets=1,
        relocated_count=20,
        unresolved_count=2,
        max_final_bucket_size=3,
    )


def _context(world_size=1, rank=0) -> DistributedCollisionContext:
    return DistributedCollisionContext(
        world_size=world_size,
        rank=rank,
        local_world_size=world_size,
    )


class DistributedCollisionTest(unittest.TestCase):
    def test_initializes_single_process_without_process_group(self) -> None:
        with (
            mock.patch.dict(os.environ, {}, clear=True),
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.is_initialized",
                return_value=False,
            ),
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.init_process_group"
            ) as init_process_group,
        ):
            context = initialize_distributed_collision()

        self.assertEqual(context.world_size, 1)
        self.assertEqual(context.rank, 0)
        self.assertFalse(context.distributed)
        init_process_group.assert_not_called()

    def test_initializes_and_destroys_multi_process_gloo_group(self) -> None:
        environment = {
            "WORLD_SIZE": "2",
            "RANK": "1",
            "LOCAL_WORLD_SIZE": "2",
        }
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.is_initialized",
                side_effect=[False, True],
            ),
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.init_process_group"
            ) as init_process_group,
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.destroy_process_group"
            ) as destroy_process_group,
        ):
            context = initialize_distributed_collision(timeout_seconds=17)
            self.assertTrue(context.owns_process_group)
            context.close()

        init_process_group.assert_called_once_with(
            backend="gloo",
            timeout=timedelta(seconds=17),
        )
        destroy_process_group.assert_called_once_with()
        self.assertFalse(context.owns_process_group)

    @parameterized.expand(
        [
            ("zero_timeout", {"timeout_seconds": 0}, {}, ValueError),
            (
                "multi_node",
                {},
                {
                    "WORLD_SIZE": "4",
                    "RANK": "0",
                    "LOCAL_WORLD_SIZE": "2",
                },
                RuntimeError,
            ),
            (
                "bad_rank",
                {},
                {
                    "WORLD_SIZE": "2",
                    "RANK": "2",
                    "LOCAL_WORLD_SIZE": "2",
                },
                ValueError,
            ),
            (
                "bad_environment",
                {},
                {"WORLD_SIZE": "many"},
                ValueError,
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_process_configuration(
        self, _, arguments, environment, error_type
    ) -> None:
        with (
            mock.patch.dict(os.environ, environment, clear=True),
            self.assertRaises(error_type),
        ):
            initialize_distributed_collision(**arguments)

    def test_builds_only_rank_owned_overflow_bands(self) -> None:
        plan = _collision_plan()
        order_hashes = np.asarray([0, 2**63 + 7, 2**64 - 1], dtype=np.uint64)
        first = build_collision_work(
            plan,
            CollisionBandShard(0, 0, 2, 0, 1),
            _candidate_codes(),
            order_hashes,
        )
        second = build_collision_work(
            plan,
            CollisionBandShard(1, 2, 3, 2, 3),
            _candidate_codes(),
            order_hashes,
        )

        np.testing.assert_array_equal(first.work_shard.overflow_rows, [1, 2])
        np.testing.assert_array_equal(first.work_shard.bucket_keys, [0])
        np.testing.assert_array_equal(first.candidate_codes, [[1, 2], [2, 3]])
        np.testing.assert_array_equal(first.order_hashes, order_hashes[:2])
        self.assertEqual(first.order_hashes.dtype, np.dtype(np.uint64))
        np.testing.assert_array_equal(second.work_shard.overflow_rows, [5])
        np.testing.assert_array_equal(second.work_shard.bucket_keys, [8])
        np.testing.assert_array_equal(second.candidate_codes, [[1, 3]])
        np.testing.assert_array_equal(second.order_hashes, order_hashes[2:])

    def test_built_work_copies_plan_aligned_arrays(self) -> None:
        plan = _collision_plan()
        candidates = _candidate_codes()
        order_hashes = np.asarray([11, 12, 13], dtype=np.uint64)

        work = build_collision_work(
            plan,
            CollisionBandShard(0, 0, 3, 0, 3),
            candidates,
            order_hashes,
        )

        self.assertFalse(np.shares_memory(work.candidate_codes, candidates))
        self.assertFalse(np.shares_memory(work.order_hashes, order_hashes))
        self.assertFalse(
            np.shares_memory(work.work_shard.overflow_rows, plan.overflow_rows)
        )

    def test_filters_non_overflow_buckets_inside_one_range(self) -> None:
        work = build_collision_work(
            _collision_plan(),
            CollisionBandShard(0, 0, 3, 0, 3),
            _candidate_codes(),
        )

        np.testing.assert_array_equal(work.work_shard.bucket_keys, [0, 8])
        np.testing.assert_array_equal(work.work_shard.bucket_counts, [3, 2])
        self.assertIsNone(work.order_hashes)

    def test_builds_empty_work_for_idle_rank(self) -> None:
        work = build_collision_work(
            _collision_plan(),
            CollisionBandShard(2, 3, 3, 3, 3),
            _candidate_codes(),
            np.asarray([1, 2, 3], dtype=np.uint64),
        )

        self.assertEqual(work.candidate_codes.shape, (0, 2))
        self.assertEqual(work.work_shard.overflow_rows.size, 0)
        self.assertEqual(work.work_shard.bucket_keys.size, 0)
        self.assertEqual(work.order_hashes.shape, (0,))
        self.assertEqual(work.order_hashes.dtype, np.dtype(np.uint64))

    @parameterized.expand(
        [
            (
                "misaligned_candidates",
                {"candidate_codes": _candidate_codes()[:2]},
                ValueError,
                "candidate_codes must align",
            ),
            (
                "float_candidates",
                {"candidate_codes": _candidate_codes().astype(np.float64)},
                TypeError,
                "integer dtype",
            ),
            (
                "misaligned_hashes",
                {
                    "candidate_codes": _candidate_codes(),
                    "order_hashes": np.asarray([1, 2], dtype=np.uint64),
                },
                ValueError,
                "order_hashes must have shape",
            ),
            (
                "bad_overflow_range",
                {
                    "candidate_codes": _candidate_codes(),
                    "band_shard": CollisionBandShard(0, 0, 4, 0, 3),
                },
                ValueError,
                "invalid overflow range",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_work_inputs(
        self, _, overrides, error_type, message
    ) -> None:
        arguments = {
            "band_shard": CollisionBandShard(0, 0, 3, 0, 3),
            **overrides,
        }
        with self.assertRaisesRegex(error_type, message):
            build_collision_work(_collision_plan(), **arguments)

    def test_round_trips_work_through_object_collectives(self) -> None:
        expected = build_collision_work(
            _collision_plan(),
            CollisionBandShard(1, 0, 3, 0, 3),
            _candidate_codes(),
            np.asarray([0, 2**63 + 7, 2**64 - 1], dtype=np.uint64),
        )
        sent = []

        def fake_recv(payload, src):
            payload[0] = sent.pop(0)

        with (
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.send_object_list",
                side_effect=lambda objects, dst: sent.append(objects[0]),
            ) as send,
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.recv_object_list",
                side_effect=fake_recv,
            ),
        ):
            send_collision_work(expected, dst=1)
            actual = receive_collision_work(src=0)

        send.assert_called_once_with([expected], dst=1)
        self.assertIs(actual, expected)

    def test_receive_rejects_unexpected_object(self) -> None:
        with (
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.recv_object_list",
                side_effect=lambda payload, src: payload.__setitem__(0, "junk"),
            ),
            self.assertRaisesRegex(RuntimeError, "received str"),
        ):
            receive_collision_work(src=0)

    def test_gather_returns_rank_ordered_results_on_coordinator(self) -> None:
        local = _result()
        remote = _result()

        def fake_gather(value, output, dst):
            if output is not None:
                output[0] = value
                output[1] = remote

        with mock.patch(
            "tzrec.utils.sid.distributed_collision.dist.gather_object",
            side_effect=fake_gather,
        ):
            gathered = gather_collision_shard_results(
                local, _context(world_size=2, rank=0)
            )
            worker_view = gather_collision_shard_results(
                remote, _context(world_size=2, rank=1)
            )

        self.assertEqual(gathered, [local, remote])
        self.assertIsNone(worker_view)

    def test_gather_rejects_unexpected_result_object(self) -> None:
        def fake_gather(value, output, dst):
            output[0] = value
            output[1] = None

        with (
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.gather_object",
                side_effect=fake_gather,
            ),
            self.assertRaisesRegex(RuntimeError, "rank 1 returned NoneType"),
        ):
            gather_collision_shard_results(_result(), _context(world_size=2, rank=0))

    def test_broadcast_shares_coordinator_stats_with_workers(self) -> None:
        expected = _stats()

        def fake_broadcast(payload, src):
            if payload[0] is None:
                payload[0] = expected

        with mock.patch(
            "tzrec.utils.sid.distributed_collision.dist.broadcast_object_list",
            side_effect=fake_broadcast,
        ):
            coordinator_stats = broadcast_collision_stats(
                expected, _context(world_size=2, rank=0)
            )
            worker_stats = broadcast_collision_stats(
                None, _context(world_size=2, rank=1)
            )

        self.assertEqual(coordinator_stats, expected)
        self.assertEqual(worker_stats, expected)

    def test_broadcast_rejects_missing_stats(self) -> None:
        with (
            mock.patch(
                "tzrec.utils.sid.distributed_collision.dist.broadcast_object_list"
            ),
            self.assertRaisesRegex(RuntimeError, "NoneType"),
        ):
            broadcast_collision_stats(None, _context(world_size=2, rank=1))


if __name__ == "__main__":
    unittest.main()
