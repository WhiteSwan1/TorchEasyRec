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

from tzrec.utils.sid.collision_sharding import (
    CollisionBandShard,
    partition_collision_bands,
)
from tzrec.utils.test_util import parameterized_name_func


class CollisionShardingTest(unittest.TestCase):
    def test_partitions_complete_bands_and_matching_buckets(self) -> None:
        overflow_prefixes = np.asarray(
            [0, 0, 0, 4, 4, 12, 12, 12, 12, 20], dtype=np.int64
        )
        bucket_keys = np.asarray(
            [0, 1, 4, 5, 8, 12, 13, 15, 16, 20, 21, 22, 24],
            dtype=np.int64,
        )

        shards = partition_collision_bands(
            overflow_prefixes,
            bucket_keys,
            last_size=4,
            world_size=3,
            candidate_count=4,
        )

        self.assertEqual(
            shards,
            (
                CollisionBandShard(0, 0, 3, 0, 2),
                CollisionBandShard(1, 3, 5, 2, 4),
                CollisionBandShard(2, 5, 10, 5, 12),
            ),
        )
        covered = np.concatenate(
            [np.arange(shard.overflow_start, shard.overflow_end) for shard in shards]
        )
        np.testing.assert_array_equal(covered, np.arange(overflow_prefixes.size))
        for shard in shards:
            shard_prefixes = overflow_prefixes[
                shard.overflow_start : shard.overflow_end
            ]
            if shard.overflow_start:
                self.assertNotEqual(
                    overflow_prefixes[shard.overflow_start - 1],
                    shard_prefixes[0],
                )
            if shard.overflow_end < overflow_prefixes.size:
                self.assertNotEqual(
                    shard_prefixes[-1], overflow_prefixes[shard.overflow_end]
                )
            bucket_bands = bucket_keys[shard.bucket_start : shard.bucket_end] // 4
            self.assertEqual(bucket_bands[0], shard_prefixes[0] // 4)
            self.assertEqual(bucket_bands[-1], shard_prefixes[-1] // 4)

    def test_more_ranks_than_bands_adds_empty_ranges(self) -> None:
        shards = partition_collision_bands(
            np.asarray([0, 0, 4], dtype=np.int64),
            np.asarray([0, 1, 4, 5, 8], dtype=np.int64),
            last_size=4,
            world_size=5,
        )

        self.assertEqual(
            shards,
            (
                CollisionBandShard(0, 0, 2, 0, 2),
                CollisionBandShard(1, 2, 3, 2, 4),
                CollisionBandShard(2, 3, 3, 4, 4),
                CollisionBandShard(3, 3, 3, 4, 4),
                CollisionBandShard(4, 3, 3, 4, 4),
            ),
        )

    def test_candidate_width_does_not_change_partition_boundaries(self) -> None:
        overflow_prefixes = np.asarray(
            [0, 0, 0, 4, 4, 12, 12, 12, 12, 20], dtype=np.int64
        )
        bucket_keys = np.asarray(
            [0, 1, 4, 5, 8, 12, 13, 15, 16, 20, 21, 22, 24],
            dtype=np.int64,
        )

        narrow = partition_collision_bands(
            overflow_prefixes,
            bucket_keys,
            last_size=4,
            world_size=3,
            candidate_count=1,
        )
        wide = partition_collision_bands(
            overflow_prefixes,
            bucket_keys,
            last_size=4,
            world_size=3,
            candidate_count=200,
        )

        self.assertEqual(narrow, wide)

    def test_empty_overflow_returns_empty_ranges(self) -> None:
        shards = partition_collision_bands(
            np.empty(0, dtype=np.int64),
            np.asarray([0, 1, 4], dtype=np.int64),
            last_size=4,
            world_size=2,
        )

        self.assertEqual(
            shards,
            (
                CollisionBandShard(0, 0, 0, 0, 0),
                CollisionBandShard(1, 0, 0, 0, 0),
            ),
        )

    @parameterized.expand(
        [
            ("last_size", {"last_size": 0}, "last_size must be a positive"),
            ("world_size", {"world_size": 0}, "world_size must be a positive"),
            (
                "candidate_count",
                {"candidate_count": 0},
                "candidate_count must be a positive",
            ),
            ("bool_world_size", {"world_size": True}, "world_size must be a positive"),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_scalar_parameters(
        self, _, overrides, error_pattern
    ) -> None:
        arguments = {
            "overflow_bucket_key_prefixes": np.asarray([0], dtype=np.int64),
            "bucket_keys": np.asarray([0], dtype=np.int64),
            "last_size": 4,
            "world_size": 1,
        }
        arguments.update(overrides)

        with self.assertRaisesRegex(ValueError, error_pattern):
            partition_collision_bands(**arguments)

    @parameterized.expand(
        [
            (
                "prefix_not_one_dimensional",
                np.asarray([[0]], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                ValueError,
                "one-dimensional",
            ),
            (
                "bucket_not_one_dimensional",
                np.asarray([0], dtype=np.int64),
                np.asarray([[0]], dtype=np.int64),
                ValueError,
                "one-dimensional",
            ),
            (
                "prefix_not_integer",
                np.asarray([0.0]),
                np.asarray([0], dtype=np.int64),
                TypeError,
                "integer dtype",
            ),
            (
                "bucket_not_integer",
                np.asarray([0], dtype=np.int64),
                np.asarray([0.0]),
                TypeError,
                "integer dtype",
            ),
            (
                "negative_prefix",
                np.asarray([-4], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                ValueError,
                "nonnegative",
            ),
            (
                "negative_bucket",
                np.asarray([0], dtype=np.int64),
                np.asarray([-1], dtype=np.int64),
                ValueError,
                "nonnegative",
            ),
            (
                "unsorted_prefix",
                np.asarray([4, 0], dtype=np.int64),
                np.asarray([0, 4], dtype=np.int64),
                ValueError,
                "sorted",
            ),
            (
                "unsorted_bucket",
                np.asarray([0, 4], dtype=np.int64),
                np.asarray([4, 0], dtype=np.int64),
                ValueError,
                "sorted",
            ),
            (
                "unaligned_prefix",
                np.asarray([1], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                ValueError,
                "multiples",
            ),
            (
                "missing_bucket_band",
                np.asarray([4], dtype=np.int64),
                np.asarray([0], dtype=np.int64),
                ValueError,
                "occupied bucket",
            ),
        ],
        name_func=parameterized_name_func,
    )
    def test_rejects_invalid_arrays(
        self, _, overflow_prefixes, bucket_keys, error_type, error_pattern
    ) -> None:
        with self.assertRaisesRegex(error_type, error_pattern):
            partition_collision_bands(
                overflow_prefixes,
                bucket_keys,
                last_size=4,
                world_size=1,
            )


if __name__ == "__main__":
    unittest.main()
