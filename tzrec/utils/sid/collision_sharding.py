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

"""Band-aligned sharding for SID collision resolution."""

from dataclasses import dataclass
from typing import Tuple

import numpy as np


@dataclass(frozen=True)
class CollisionBandShard:
    """Contiguous overflow and bucket ranges assigned to one rank.

    Args:
        rank: Zero-based worker rank.
        overflow_start: Inclusive index into overflow-aligned arrays.
        overflow_end: Exclusive index into overflow-aligned arrays.
        bucket_start: Inclusive index into the sorted bucket arrays.
        bucket_end: Exclusive index into the sorted bucket arrays.
    """

    rank: int
    overflow_start: int
    overflow_end: int
    bucket_start: int
    bucket_end: int


def _positive_integer(name: str, value: int) -> int:
    """Validate and normalize a positive integer.

    Args:
        name: Parameter name used in the error message.
        value: Value to validate.

    Returns:
        The value as a Python integer.

    Raises:
        ValueError: If the value is not a positive integer.
    """
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, np.integer))
        or value < 1
    ):
        raise ValueError(f"{name} must be a positive integer, got {value}.")
    return int(value)


def _sorted_nonnegative_integer_array(name: str, values: np.ndarray) -> np.ndarray:
    """Validate a sorted, nonnegative one-dimensional integer array.

    Args:
        name: Parameter name used in error messages.
        values: Array to validate.

    Returns:
        The normalized NumPy array.

    Raises:
        TypeError: If the array does not use an integer dtype.
        ValueError: If the array is not one-dimensional, is negative, or is
            not sorted in nondecreasing order.
    """
    array = np.asarray(values)
    if array.ndim != 1:
        raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
    if not np.issubdtype(array.dtype, np.integer):
        raise TypeError(f"{name} must use an integer dtype.")
    if array.size and array[0] < 0:
        raise ValueError(f"{name} must contain only nonnegative values.")
    if array.size > 1 and np.any(array[1:] < array[:-1]):
        raise ValueError(f"{name} must be sorted in nondecreasing order.")
    return array


def partition_collision_bands(
    overflow_bucket_key_prefixes: np.ndarray,
    bucket_keys: np.ndarray,
    last_size: int,
    world_size: int,
    candidate_count: int = 1,
) -> Tuple[CollisionBandShard, ...]:
    """Partition sorted collision work without splitting a SID band.

    Overflow bands are assigned contiguously with a greedy balance based on
    ``overflow_rows * candidate_count``. Bucket ranges include every occupied
    band from the first through the last overflow band assigned to a rank.
    Ranks beyond the number of overflow bands receive empty ranges.

    Args:
        overflow_bucket_key_prefixes: Sorted last-layer-excluded bucket key
            prefixes aligned with overflow rows. Every prefix must be a
            multiple of ``last_size``.
        bucket_keys: Sorted flattened keys for occupied SID buckets.
        last_size: Cardinality of the final SID layer.
        world_size: Number of worker ranks.
        candidate_count: Estimated candidates examined per overflow row.

    Returns:
        One band-aligned range descriptor per rank.

    Raises:
        TypeError: If an input array does not use an integer dtype.
        ValueError: If a scalar parameter or array violates the input
            contract, or an overflow band has no occupied bucket.
    """
    last_size = _positive_integer("last_size", last_size)
    world_size = _positive_integer("world_size", world_size)
    candidate_count = _positive_integer("candidate_count", candidate_count)
    overflow_prefixes = _sorted_nonnegative_integer_array(
        "overflow_bucket_key_prefixes", overflow_bucket_key_prefixes
    )
    bucket_keys = _sorted_nonnegative_integer_array("bucket_keys", bucket_keys)

    if overflow_prefixes.size and np.any(overflow_prefixes % last_size):
        raise ValueError("overflow_bucket_key_prefixes must be multiples of last_size.")
    if not overflow_prefixes.size:
        return tuple(CollisionBandShard(rank, 0, 0, 0, 0) for rank in range(world_size))

    overflow_band_ids = overflow_prefixes // last_size
    bucket_band_ids = bucket_keys // last_size
    band_ids, band_starts, band_counts = np.unique(
        overflow_band_ids, return_index=True, return_counts=True
    )
    bucket_positions = np.searchsorted(bucket_band_ids, band_ids)
    missing_band = bucket_positions == bucket_band_ids.size
    valid_positions = bucket_positions[~missing_band]
    if valid_positions.size:
        missing_band[~missing_band] = (
            bucket_band_ids[valid_positions] != band_ids[~missing_band]
        )
    if np.any(missing_band):
        raise ValueError("every overflow band must have at least one occupied bucket.")

    band_costs = [int(count) * candidate_count for count in band_counts]
    active_shards = min(world_size, len(band_costs))
    shards = []
    band_start = 0
    remaining_cost = sum(band_costs)
    overflow_count = overflow_prefixes.size

    for rank in range(active_shards):
        remaining_shards = active_shards - rank
        if remaining_shards == 1:
            band_end = len(band_costs)
            shard_cost = remaining_cost
        else:
            max_band_end = len(band_costs) - remaining_shards + 1
            band_end = band_start + 1
            shard_cost = band_costs[band_start]
            while band_end < max_band_end:
                expanded_cost = shard_cost + band_costs[band_end]
                if abs(expanded_cost * remaining_shards - remaining_cost) > abs(
                    shard_cost * remaining_shards - remaining_cost
                ):
                    break
                shard_cost = expanded_cost
                band_end += 1

        overflow_start = int(band_starts[band_start])
        overflow_end = (
            int(band_starts[band_end])
            if band_end < len(band_starts)
            else overflow_count
        )
        first_band = band_ids[band_start]
        last_band = band_ids[band_end - 1]
        bucket_start = int(np.searchsorted(bucket_band_ids, first_band, side="left"))
        bucket_end = int(np.searchsorted(bucket_band_ids, last_band, side="right"))
        shards.append(
            CollisionBandShard(
                rank,
                overflow_start,
                overflow_end,
                bucket_start,
                bucket_end,
            )
        )
        band_start = band_end
        remaining_cost -= shard_cost

    empty_overflow_index = overflow_prefixes.size
    empty_bucket_index = shards[-1].bucket_end
    shards.extend(
        CollisionBandShard(
            rank,
            empty_overflow_index,
            empty_overflow_index,
            empty_bucket_index,
            empty_bucket_index,
        )
        for rank in range(active_shards, world_size)
    )
    return tuple(shards)
