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

"""Build fixed-count, length-balanced Parquet batches from per-row costs."""

import argparse
import heapq

import numpy as np
from pyarrow import parquet

from tzrec.datasets.utils import calc_slice_intervals, list_input_files


def create_row_plan(
    input_path: str,
    costs: np.ndarray,
    batch_size: int,
    world_size: int,
    num_workers: int,
    output_path: str,
) -> None:
    """Balance each existing global microbatch without changing its sample set.

    Args:
        input_path: original Parquet file pattern.
        costs: nonnegative costs in file order, one scalar per original row.
        batch_size: fixed number of parents per rank.
        world_size: number of training ranks.
        num_workers: DataLoader workers per rank, at least one.
        output_path: destination NPZ plan; the source Parquet files stay intact.
    """
    paths = list_input_files(input_path)
    counts = []
    for path in paths:
        with parquet.ParquetFile(path) as source:
            counts.append(source.metadata.num_rows)
    if (
        costs.shape != (sum(counts),)
        or not np.isfinite(costs).all()
        or (costs < 0).any()
    ):
        raise ValueError(
            "Costs must contain one finite nonnegative value per input row."
        )
    streams = []
    for worker in range(world_size * num_workers):
        intervals = calc_slice_intervals(
            [(input_path, sum(counts))],
            worker,
            world_size * num_workers,
            batch_size,
            True,
            0,
            world_size=world_size,
        )[0]
        rows = np.concatenate([np.arange(start, end) for start, end in intervals])
        streams.append(
            [rows[i : i + batch_size] for i in range(0, len(rows), batch_size)]
        )
    rank_batches = []
    for rank in range(world_size):
        local = streams[rank * num_workers : (rank + 1) * num_workers]
        rank_batches.append(
            [
                worker[step]
                for step in range(max(map(len, local)))
                for worker in local
                if step < len(worker)
            ]
        )
    steps = len(rank_batches[0])
    if any(len(batches) != steps for batches in rank_batches):
        raise ValueError("The source reader did not produce equal rank step counts.")
    assignment = np.full((steps, world_size, batch_size), -1, dtype=np.int64)
    for step in range(steps):
        source_rows = [batches[step] for batches in rank_batches]
        if any(len(rows) < batch_size for rows in source_rows):
            for rank, rows in enumerate(source_rows):
                assignment[step, rank, : len(rows)] = rows
            continue
        buckets = [[] for _ in range(world_size)]
        queue = [(0.0, rank) for rank in range(world_size)]
        for row in sorted(
            np.concatenate(source_rows).tolist(), key=lambda i: (-costs[i], i)
        ):
            total, rank = heapq.heappop(queue)
            buckets[rank].append(row)
            if len(buckets[rank]) < batch_size:
                heapq.heappush(queue, (total + float(costs[row]), rank))
        assignment[step] = buckets
    np.savez(
        output_path,
        balanced=assignment,
        input_files=np.asarray(paths),
        num_rows=np.asarray(counts),
        num_workers=num_workers,
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input_path", required=True)
    parser.add_argument(
        "--cost_file", required=True, help="NPY costs in input file/row order"
    )
    parser.add_argument("--batch_size", type=int, required=True)
    parser.add_argument("--world_size", type=int, required=True)
    parser.add_argument("--num_workers", type=int, default=8)
    parser.add_argument("--output_path", required=True)
    args = parser.parse_args()
    if min(args.batch_size, args.world_size, args.num_workers) <= 0:
        parser.error("batch_size, world_size and num_workers must be positive")
    create_row_plan(
        args.input_path,
        np.load(args.cost_file, allow_pickle=False),
        args.batch_size,
        args.world_size,
        args.num_workers,
        args.output_path,
    )
