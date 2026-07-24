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

"""Single-node distributed exchange for SID collision resolution.

Rank zero sends small work metadata with object communication and transfers
NumPy arrays as bounded-size Gloo tensors. Compact results and final statistics
continue to use object collectives. Failure handling is delegated to the
launcher: a rank that raises terminates the torchrun job.
"""

import os
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, List, Optional, Tuple

import numpy as np
import torch
from torch import distributed as dist

from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionStats,
    CollisionShardResult,
    CollisionWorkShard,
    overflow_band_bucket_mask,
)
from tzrec.utils.sid.collision_sharding import CollisionBandShard

_TENSOR_TRANSFER_CHUNK_BYTES = 128 * 1024 * 1024


@dataclass
class DistributedCollisionContext:
    """Single-node process context initialized from torchrun variables.

    Args:
        world_size: Total number of processes.
        rank: Global process rank.
        owns_process_group: Whether this context initialized the process group.
    """

    world_size: int
    rank: int
    owns_process_group: bool = False

    @property
    def distributed(self) -> bool:
        """Return whether more than one worker participates."""
        return self.world_size > 1

    @property
    def is_coordinator(self) -> bool:
        """Return whether this process owns coordinator-only work."""
        return self.rank == 0

    def close(self) -> None:
        """Destroy the process group initialized by this context."""
        if self.owns_process_group and dist.is_initialized():
            dist.destroy_process_group()
        self.owns_process_group = False

    def __enter__(self) -> "DistributedCollisionContext":
        return self

    def __exit__(self, exc_type: Any, exc_value: Any, traceback: Any) -> None:
        self.close()


@dataclass(frozen=True)
class DistributedCollisionWork:
    """Numeric collision work owned by one process.

    Args:
        band_shard: Global overflow and bucket ranges assigned to the process.
        work_shard: Compact collision work containing complete owned bands.
        candidate_codes: Candidate matrix aligned with local overflow rows.
        order_hashes: Optional stable uint64 hashes for iterative arbitration.
    """

    band_shard: CollisionBandShard
    work_shard: CollisionWorkShard
    candidate_codes: np.ndarray
    order_hashes: Optional[np.ndarray]


@dataclass(frozen=True)
class _CollisionWorkHeader:
    """Array-free description of one transmitted work shard.

    Every transmitted array shape derives from this header: overflow-aligned
    arrays span the ``band_shard`` overflow range, bucket arrays have
    ``bucket_count`` rows, and the candidate matrix is ``(overflow rows,
    candidate_count)``.
    """

    band_shard: CollisionBandShard
    config: CollisionResolutionConfig
    bucket_count: int
    candidate_count: int
    has_order_hashes: bool


def initialize_distributed_collision(
    timeout_seconds: float = 1800,
) -> DistributedCollisionContext:
    """Initialize a single-node CPU collision process context.

    Args:
        timeout_seconds: Positive timeout applied to Gloo operations.

    Returns:
        Context populated from ``WORLD_SIZE``, ``RANK``, and
        ``LOCAL_WORLD_SIZE``.

    Raises:
        RuntimeError: If an existing process group is incompatible or a
            multi-node launch is detected.
        ValueError: If an environment value or the timeout is invalid.
    """
    timeout = _positive_timeout(timeout_seconds)
    world_size = _environment_integer("WORLD_SIZE", 1)
    rank = _environment_integer("RANK", 0)
    local_world_size = _environment_integer("LOCAL_WORLD_SIZE", 1)
    if world_size < 1:
        raise ValueError(f"WORLD_SIZE must be >= 1, got {world_size}.")
    if local_world_size < 1:
        raise ValueError(f"LOCAL_WORLD_SIZE must be >= 1, got {local_world_size}.")
    if rank < 0 or rank >= world_size:
        raise ValueError(f"RANK must be in [0, {world_size}), got {rank}.")
    if world_size > 1 and local_world_size != world_size:
        raise RuntimeError(
            "SID collision resolution supports one machine only: "
            f"LOCAL_WORLD_SIZE={local_world_size}, WORLD_SIZE={world_size}."
        )

    owns_process_group = False
    if world_size > 1:
        if dist.is_initialized():
            if (
                dist.get_world_size() != world_size
                or dist.get_rank() != rank
                or str(dist.get_backend()) != "gloo"
            ):
                raise RuntimeError(
                    "the existing process group does not match the requested "
                    "rank, world size, and Gloo backend."
                )
        else:
            dist.init_process_group(
                backend="gloo",
                timeout=timedelta(seconds=timeout),
            )
            owns_process_group = True
    return DistributedCollisionContext(
        world_size=world_size,
        rank=rank,
        owns_process_group=owns_process_group,
    )


def build_collision_work(
    plan: CollisionPlan,
    band_shard: CollisionBandShard,
    candidate_codes: np.ndarray,
    order_hashes: Optional[np.ndarray] = None,
) -> DistributedCollisionWork:
    """Build one complete-band numeric work shard on rank zero.

    Bucket range descriptors span the first through last owned overflow band.
    Occupied buckets belonging to intervening non-overflow bands are removed.
    Candidate codes and hashes must already be aligned with the local shard.
    Plan slices and contiguous candidate or hash arrays are reused as views
    without copying; the caller owns any copy that must outlive the plan or
    the full candidate matrix.

    Args:
        plan: Global collision plan prepared by rank zero.
        band_shard: Complete-band ranges assigned to one rank.
        candidate_codes: Integer candidate matrix aligned with overflow rows
            owned by ``band_shard``.
        order_hashes: Optional integer hashes aligned with overflow rows owned
            by ``band_shard``.

    Returns:
        Compact work ready for local resolution or transmission.

    Raises:
        TypeError: If candidates or hashes do not use an integer dtype.
        ValueError: If ranges or array shapes violate the work contract.
    """
    overflow_count = int(plan.overflow_rows.shape[0])
    if not (
        0 <= band_shard.overflow_start <= band_shard.overflow_end <= overflow_count
    ):
        raise ValueError("band_shard has an invalid overflow range.")
    if not (
        0
        <= band_shard.bucket_start
        <= band_shard.bucket_end
        <= int(plan.bucket_keys.shape[0])
    ):
        raise ValueError("band_shard has an invalid bucket range.")

    local_overflow_count = band_shard.overflow_end - band_shard.overflow_start
    candidates = np.asarray(candidate_codes)
    if candidates.ndim != 2:
        raise ValueError(f"candidate_codes must be 2-D, got {candidates.shape}.")
    if candidates.shape[0] != local_overflow_count:
        raise ValueError(
            "candidate_codes must align with shard overflow rows, got "
            f"{candidates.shape[0]} candidates for {local_overflow_count} rows."
        )
    if not np.issubdtype(candidates.dtype, np.integer):
        raise TypeError("candidate_codes must use an integer dtype.")
    candidates = np.ascontiguousarray(candidates, dtype=np.int64)
    hashes = None
    if order_hashes is not None:
        hashes = np.asarray(order_hashes)
        if hashes.shape != (local_overflow_count,):
            raise ValueError(
                "order_hashes must align with shard overflow rows, got "
                f"{hashes.shape} for {local_overflow_count} rows."
            )
        if not np.issubdtype(hashes.dtype, np.integer):
            raise TypeError("order_hashes must use an integer dtype.")
        hashes = np.ascontiguousarray(hashes, dtype=np.uint64)

    overflow_slice = slice(band_shard.overflow_start, band_shard.overflow_end)
    prefixes = plan.overflow_bucket_key_prefixes[overflow_slice].astype(
        np.int64, copy=False
    )
    bucket_slice = slice(band_shard.bucket_start, band_shard.bucket_end)
    if prefixes.size:
        bucket_keys = plan.bucket_keys[bucket_slice]
        owned_bucket_mask = overflow_band_bucket_mask(
            bucket_keys, prefixes, plan.config.layer_sizes[-1]
        )
        bucket_keys = bucket_keys[owned_bucket_mask].astype(np.int64, copy=False)
        bucket_counts = plan.bucket_counts[bucket_slice][owned_bucket_mask].astype(
            np.int64, copy=False
        )
    else:
        bucket_keys = np.empty(0, dtype=np.int64)
        bucket_counts = np.empty(0, dtype=np.int64)

    return DistributedCollisionWork(
        band_shard=band_shard,
        work_shard=CollisionWorkShard(
            overflow_rows=plan.overflow_rows[overflow_slice].astype(
                np.int64, copy=False
            ),
            overflow_bucket_key_prefixes=prefixes,
            overflow_origin_last_codes=plan.overflow_origin_last_codes[
                overflow_slice
            ].astype(np.int64, copy=False),
            bucket_keys=bucket_keys,
            bucket_counts=bucket_counts,
            config=plan.config,
        ),
        candidate_codes=candidates,
        order_hashes=hashes,
    )


def send_collision_work(work: DistributedCollisionWork, dst: int) -> None:
    """Send one numeric work shard with bounded tensor transfers.

    ``work`` must come from :func:`build_collision_work`, which guarantees the
    contiguous int64/uint64 storage the fixed transfer order relies on.

    Args:
        work: Complete-band work to transmit.
        dst: Destination process rank.
    """
    header = _CollisionWorkHeader(
        band_shard=work.band_shard,
        config=work.work_shard.config,
        bucket_count=int(work.work_shard.bucket_keys.shape[0]),
        candidate_count=int(work.candidate_codes.shape[1]),
        has_order_hashes=work.order_hashes is not None,
    )
    dist.send_object_list([header], dst=dst)
    for array in _work_arrays(work):
        if array is not None:
            _send_array(array, dst)


def receive_collision_work(src: int) -> DistributedCollisionWork:
    """Receive one numeric work shard from bounded tensor transfers.

    Args:
        src: Source process rank.

    Returns:
        Received complete-band collision work.

    Raises:
        RuntimeError: If the received work header is malformed.
        ValueError: If the header describes a negative array size.
    """
    payload: List[Any] = [None]
    dist.recv_object_list(payload, src=src)
    header = payload[0]
    if not isinstance(header, _CollisionWorkHeader):
        raise RuntimeError(
            f"received {type(header).__name__} instead of collision work metadata."
        )
    band_shard = header.band_shard
    if not isinstance(band_shard, CollisionBandShard) or not isinstance(
        header.config, CollisionResolutionConfig
    ):
        raise RuntimeError("received invalid collision work metadata.")
    local_count = band_shard.overflow_end - band_shard.overflow_start
    if local_count < 0 or header.bucket_count < 0 or header.candidate_count < 0:
        raise ValueError("received a negative collision work array size.")

    # NOTE: transfers must match the _work_arrays transmission order.
    overflow_rows = _receive_array((local_count,), np.int64, src)
    prefixes = _receive_array((local_count,), np.int64, src)
    origin_last_codes = _receive_array((local_count,), np.int64, src)
    bucket_keys = _receive_array((header.bucket_count,), np.int64, src)
    bucket_counts = _receive_array((header.bucket_count,), np.int64, src)
    candidate_codes = _receive_array(
        (local_count, header.candidate_count), np.int64, src
    )
    order_hashes = (
        _receive_array((local_count,), np.uint64, src)
        if header.has_order_hashes
        else None
    )
    return DistributedCollisionWork(
        band_shard=band_shard,
        work_shard=CollisionWorkShard(
            overflow_rows=overflow_rows,
            overflow_bucket_key_prefixes=prefixes,
            overflow_origin_last_codes=origin_last_codes,
            bucket_keys=bucket_keys,
            bucket_counts=bucket_counts,
            config=header.config,
        ),
        candidate_codes=candidate_codes,
        order_hashes=order_hashes,
    )


def synchronize_collision_workers() -> None:
    """Wait until every rank has received its collision work shard."""
    dist.barrier()


def gather_collision_shard_results(
    result: CollisionShardResult,
    context: DistributedCollisionContext,
) -> Optional[List[CollisionShardResult]]:
    """Gather every rank's compact result on the coordinator.

    Args:
        result: This rank's compact collision result.
        context: Active collision process context.

    Returns:
        Rank-ordered results on the coordinator, or ``None`` on workers.

    Raises:
        RuntimeError: If a gathered object is not a compact collision result.
    """
    gathered: Optional[List[Any]] = (
        [None] * context.world_size if context.is_coordinator else None
    )
    dist.gather_object(result, gathered, dst=0)
    if gathered is None:
        return None
    for rank, value in enumerate(gathered):
        if not isinstance(value, CollisionShardResult):
            raise RuntimeError(
                f"rank {rank} returned {type(value).__name__} instead of a "
                "collision shard result."
            )
    return gathered


def broadcast_collision_stats(
    stats: Optional[CollisionResolutionStats],
    context: DistributedCollisionContext,
) -> CollisionResolutionStats:
    """Share the coordinator's final statistics with every rank.

    Args:
        stats: Final statistics on the coordinator; ``None`` on workers.
        context: Active collision process context.

    Returns:
        Final statistics on every rank.

    Raises:
        RuntimeError: If the broadcast object is not collision statistics.
    """
    payload: List[Any] = [stats if context.is_coordinator else None]
    dist.broadcast_object_list(payload, src=0)
    received = payload[0]
    if not isinstance(received, CollisionResolutionStats):
        raise RuntimeError(
            f"rank 0 broadcast {type(received).__name__} instead of collision "
            "statistics."
        )
    return received


def _positive_timeout(value: float) -> float:
    """Validate and normalize a positive finite timeout."""
    if (
        isinstance(value, (bool, np.bool_))
        or not isinstance(value, (int, float, np.integer, np.floating))
        or not np.isfinite(value)
        or value <= 0
    ):
        raise ValueError(
            f"timeout_seconds must be a positive finite number, got {value}."
        )
    return float(value)


def _work_arrays(
    work: DistributedCollisionWork,
) -> Tuple[Optional[np.ndarray], ...]:
    """Return work arrays in their fixed transmission order."""
    return (
        work.work_shard.overflow_rows,
        work.work_shard.overflow_bucket_key_prefixes,
        work.work_shard.overflow_origin_last_codes,
        work.work_shard.bucket_keys,
        work.work_shard.bucket_counts,
        work.candidate_codes,
        work.order_hashes,
    )


def _send_array(array: np.ndarray, dst: int) -> None:
    """Send one contiguous array in bounded-size tensor chunks."""
    tensor = torch.from_numpy(array).reshape(-1)
    elements_per_chunk = max(_TENSOR_TRANSFER_CHUNK_BYTES // tensor.element_size(), 1)
    for start in range(0, tensor.numel(), elements_per_chunk):
        dist.send(tensor[start : start + elements_per_chunk], dst=dst)


def _receive_array(shape: Tuple[int, ...], dtype: type, src: int) -> np.ndarray:
    """Allocate and receive one array in bounded-size tensor chunks."""
    array = np.empty(shape, dtype=dtype)
    tensor = torch.from_numpy(array).reshape(-1)
    elements_per_chunk = max(_TENSOR_TRANSFER_CHUNK_BYTES // tensor.element_size(), 1)
    for start in range(0, tensor.numel(), elements_per_chunk):
        dist.recv(tensor[start : start + elements_per_chunk], src=src)
    return array


def _environment_integer(name: str, default: int) -> int:
    """Read an integer environment variable with a clear validation error."""
    value = os.environ.get(name)
    if value is None:
        return default
    try:
        return int(value)
    except ValueError as error:
        raise ValueError(f"{name} must be an integer, got {value!r}.") from error
