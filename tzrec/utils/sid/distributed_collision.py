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
from typing import Any, List, Optional, Tuple, cast

import numpy as np
import torch
from torch import distributed as dist

from tzrec.utils.sid.collision import (
    CollisionPlan,
    CollisionResolutionConfig,
    CollisionResolutionStats,
    CollisionShardResult,
    CollisionWorkShard,
)
from tzrec.utils.sid.collision_sharding import CollisionBandShard

_TENSOR_TRANSFER_CHUNK_BYTES = 128 * 1024 * 1024
_WORK_ARRAY_NAMES = (
    "overflow_rows",
    "overflow_bucket_key_prefixes",
    "overflow_origin_last_codes",
    "bucket_keys",
    "bucket_counts",
    "candidate_codes",
    "order_hashes",
)


@dataclass
class DistributedCollisionContext:
    """Single-node process context initialized from torchrun variables.

    Args:
        world_size: Total number of processes.
        rank: Global process rank.
        local_world_size: Number of processes on the local machine.
        owns_process_group: Whether this context initialized the process group.
    """

    world_size: int
    rank: int
    local_world_size: int
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
class _ArrayMetadata:
    """Shape and dtype needed to allocate one received array."""

    shape: Tuple[int, ...]
    dtype: str


@dataclass(frozen=True)
class _CollisionWorkMetadata:
    """Small non-array portion of one distributed collision work shard."""

    band_shard: CollisionBandShard
    config: CollisionResolutionConfig
    arrays: Tuple[Optional[_ArrayMetadata], ...]


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
        local_world_size=local_world_size,
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
    Contiguous arrays with the target dtype are reused without copying, so rank
    zero can release each shard immediately after it is transmitted.

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
        np.int64, copy=True
    )
    bucket_slice = slice(band_shard.bucket_start, band_shard.bucket_end)
    if prefixes.size:
        last_size = plan.config.layer_sizes[-1]
        owned_bands = np.unique(prefixes // last_size)
        bucket_keys = plan.bucket_keys[bucket_slice]
        owned_bucket_mask = np.isin(bucket_keys // last_size, owned_bands)
        bucket_keys = bucket_keys[owned_bucket_mask].astype(np.int64, copy=True)
        bucket_counts = plan.bucket_counts[bucket_slice][owned_bucket_mask].astype(
            np.int64, copy=True
        )
    else:
        bucket_keys = np.empty(0, dtype=np.int64)
        bucket_counts = np.empty(0, dtype=np.int64)

    return DistributedCollisionWork(
        band_shard=band_shard,
        work_shard=CollisionWorkShard(
            overflow_rows=plan.overflow_rows[overflow_slice].astype(
                np.int64, copy=True
            ),
            overflow_bucket_key_prefixes=prefixes,
            overflow_origin_last_codes=plan.overflow_origin_last_codes[
                overflow_slice
            ].astype(np.int64, copy=True),
            bucket_keys=bucket_keys,
            bucket_counts=bucket_counts,
            config=plan.config,
        ),
        candidate_codes=candidates,
        order_hashes=hashes,
    )


def send_collision_work(work: DistributedCollisionWork, dst: int) -> None:
    """Send one numeric work shard with bounded tensor transfers.

    Args:
        work: Complete-band work to transmit.
        dst: Destination process rank.

    Raises:
        TypeError: If an array does not use its required dtype.
        ValueError: If work arrays violate the shard contract or are not
            contiguous.
    """
    arrays = _collision_work_arrays(work)
    _validate_collision_work_arrays(work, arrays)
    metadata = _CollisionWorkMetadata(
        band_shard=work.band_shard,
        config=work.work_shard.config,
        arrays=tuple(
            None if array is None else _ArrayMetadata(array.shape, array.dtype.str)
            for array in arrays
        ),
    )
    dist.send_object_list([metadata], dst=dst)
    for array in arrays:
        if array is not None:
            _send_array(array, dst)


def receive_collision_work(src: int) -> DistributedCollisionWork:
    """Receive one numeric work shard from bounded tensor transfers.

    Args:
        src: Source process rank.

    Returns:
        Received complete-band collision work.

    Raises:
        RuntimeError: If the received metadata is malformed.
        TypeError: If received metadata specifies an unsupported dtype.
        ValueError: If received array metadata or work shapes are invalid.
    """
    payload: List[Any] = [None]
    dist.recv_object_list(payload, src=src)
    metadata = payload[0]
    if not isinstance(metadata, _CollisionWorkMetadata):
        raise RuntimeError(
            f"received {type(metadata).__name__} instead of collision work metadata."
        )
    if not isinstance(metadata.band_shard, CollisionBandShard) or not isinstance(
        metadata.config, CollisionResolutionConfig
    ):
        raise RuntimeError("received invalid collision work metadata.")
    if len(metadata.arrays) != len(_WORK_ARRAY_NAMES):
        raise RuntimeError(
            f"received metadata for {len(metadata.arrays)} arrays, expected "
            f"{len(_WORK_ARRAY_NAMES)}."
        )
    arrays = tuple(
        None if array_metadata is None else _receive_array(array_metadata, src)
        for array_metadata in metadata.arrays
    )
    if any(array is None for array in arrays[:-1]):
        raise RuntimeError("received missing required collision work array metadata.")
    overflow_rows = cast(np.ndarray, arrays[0])
    overflow_bucket_key_prefixes = cast(np.ndarray, arrays[1])
    overflow_origin_last_codes = cast(np.ndarray, arrays[2])
    bucket_keys = cast(np.ndarray, arrays[3])
    bucket_counts = cast(np.ndarray, arrays[4])
    candidate_codes = cast(np.ndarray, arrays[5])
    order_hashes = arrays[6]
    work = DistributedCollisionWork(
        band_shard=metadata.band_shard,
        work_shard=CollisionWorkShard(
            overflow_rows=overflow_rows,
            overflow_bucket_key_prefixes=overflow_bucket_key_prefixes,
            overflow_origin_last_codes=overflow_origin_last_codes,
            bucket_keys=bucket_keys,
            bucket_counts=bucket_counts,
            config=metadata.config,
        ),
        candidate_codes=candidate_codes,
        order_hashes=order_hashes,
    )
    _validate_collision_work_arrays(work, arrays)
    return work


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


def _collision_work_arrays(
    work: DistributedCollisionWork,
) -> Tuple[Optional[np.ndarray], ...]:
    """Return transmitted arrays in their fixed wire-protocol order."""
    return (
        work.work_shard.overflow_rows,
        work.work_shard.overflow_bucket_key_prefixes,
        work.work_shard.overflow_origin_last_codes,
        work.work_shard.bucket_keys,
        work.work_shard.bucket_counts,
        work.candidate_codes,
        work.order_hashes,
    )


def _validate_collision_work_arrays(
    work: DistributedCollisionWork,
    arrays: Tuple[Optional[np.ndarray], ...],
) -> None:
    """Validate array shapes, dtypes, and storage before transmission."""
    local_count = work.band_shard.overflow_end - work.band_shard.overflow_start
    if local_count < 0:
        raise ValueError("band_shard has an invalid overflow range.")
    expected_shapes = (
        (local_count,),
        (local_count,),
        (local_count,),
        None,
        None,
        None,
        (local_count,),
    )
    expected_dtypes = (
        np.dtype(np.int64),
        np.dtype(np.int64),
        np.dtype(np.int64),
        np.dtype(np.int64),
        np.dtype(np.int64),
        np.dtype(np.int64),
        np.dtype(np.uint64),
    )
    for index, (name, array, shape, dtype) in enumerate(
        zip(_WORK_ARRAY_NAMES, arrays, expected_shapes, expected_dtypes)
    ):
        if array is None:
            if name != "order_hashes":
                raise ValueError(f"{name} must not be None.")
            continue
        if not isinstance(array, np.ndarray):
            raise TypeError(f"{name} must be a NumPy array.")
        if array.dtype != dtype:
            raise TypeError(f"{name} must use {dtype}, got {array.dtype}.")
        if not array.dtype.isnative:
            raise ValueError(f"{name} must use native byte order.")
        if not array.flags.c_contiguous:
            raise ValueError(f"{name} must be C-contiguous.")
        if shape is not None and array.shape != shape:
            raise ValueError(f"{name} must have shape {shape}, got {array.shape}.")
        if index in (3, 4) and array.ndim != 1:
            raise ValueError(f"{name} must be one-dimensional, got {array.shape}.")
        if name == "candidate_codes" and (
            array.ndim != 2 or array.shape[0] != local_count
        ):
            raise ValueError(
                "candidate_codes must be 2-D and align with shard overflow rows, "
                f"got {array.shape}."
            )
    bucket_keys = cast(np.ndarray, arrays[3])
    bucket_counts = cast(np.ndarray, arrays[4])
    if bucket_keys.shape != bucket_counts.shape:
        raise ValueError("bucket_keys and bucket_counts must have the same shape.")


def _send_array(array: np.ndarray, dst: int) -> None:
    """Send one contiguous array in bounded-size tensor chunks."""
    tensor = torch.from_numpy(array).reshape(-1)
    elements_per_chunk = max(_TENSOR_TRANSFER_CHUNK_BYTES // tensor.element_size(), 1)
    for start in range(0, tensor.numel(), elements_per_chunk):
        dist.send(tensor[start : start + elements_per_chunk], dst=dst)


def _receive_array(metadata: _ArrayMetadata, src: int) -> np.ndarray:
    """Allocate and receive one array in bounded-size tensor chunks."""
    if not isinstance(metadata, _ArrayMetadata):
        raise RuntimeError(
            f"received {type(metadata).__name__} instead of array metadata."
        )
    if not isinstance(metadata.shape, tuple) or any(
        isinstance(size, bool) or not isinstance(size, int) or size < 0
        for size in metadata.shape
    ):
        raise ValueError(f"received invalid array shape {metadata.shape!r}.")
    try:
        dtype = np.dtype(metadata.dtype)
    except TypeError as error:
        raise TypeError(f"received invalid array dtype {metadata.dtype!r}.") from error
    if dtype not in (np.dtype(np.int64), np.dtype(np.uint64)):
        raise TypeError(f"received unsupported array dtype {dtype}.")
    if not dtype.isnative:
        raise ValueError("received array dtype must use native byte order.")
    array = np.empty(metadata.shape, dtype=dtype)
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
