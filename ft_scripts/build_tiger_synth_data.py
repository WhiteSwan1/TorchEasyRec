# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
"""Generate synthetic TIGER training/eval parquet shards.

Each row represents one user with:
    - history_sids : list<int64>, length = items × num_hierarchies, codes
                     in [0, codebook_sizes[h])
    - label_sids   : list<int64>, length = num_hierarchies, codes in
                     [0, codebook_sizes[h])

For a quick sanity check, we plant a deterministic relationship between
history and label: label = (last_item_in_history + 1) mod codebook_size,
so the model has signal to learn beyond pure noise.

Usage:
    python ft_scripts/build_tiger_synth_data.py \\
        --out_dir data/tiger_synth_train \\
        --num_rows 1000 \\
        --codebook 8192,8192,8192 \\
        --num_items_min 4 --num_items_max 20
"""

import argparse
import os
from typing import List

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq


def _gen_rows(
    num_rows: int,
    codebook_sizes: List[int],
    items_min: int,
    items_max: int,
    seed: int,
) -> pa.Table:
    rng = np.random.default_rng(seed)
    H = len(codebook_sizes)
    histories = []
    labels = []
    user_ids = []
    for i in range(num_rows):
        n_items = int(rng.integers(items_min, items_max + 1))
        # Random SID codes for each item in history.
        history_codes = [
            int(rng.integers(0, codebook_sizes[h]))
            for _ in range(n_items)
            for h in range(H)
        ]
        # Label = (last_item + 1) mod codebook_size per hierarchy — a
        # deterministic non-trivial signal the model can learn.
        last_item = history_codes[-H:]
        label_codes = [
            (last_item[h] + 1) % codebook_sizes[h] for h in range(H)
        ]
        histories.append(history_codes)
        labels.append(label_codes)
        user_ids.append(i)
    table = pa.table(
        {
            "history_sids": pa.array(histories, type=pa.list_(pa.int64())),
            "label_sids": pa.array(labels, type=pa.list_(pa.int64())),
            "user_id": pa.array(user_ids, type=pa.int64()),
        }
    )
    return table


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out_dir", required=True)
    parser.add_argument("--num_rows", type=int, default=1000)
    parser.add_argument(
        "--codebook",
        type=str,
        default="8192,8192,8192",
        help="Comma-separated per-hierarchy codebook sizes.",
    )
    parser.add_argument("--num_items_min", type=int, default=4)
    parser.add_argument("--num_items_max", type=int, default=20)
    parser.add_argument(
        "--shards",
        type=int,
        default=4,
        help="Number of parquet shards to emit.",
    )
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    codebook_sizes = [int(x) for x in args.codebook.split(",") if x.strip()]
    os.makedirs(args.out_dir, exist_ok=True)
    rows_per_shard = (args.num_rows + args.shards - 1) // args.shards
    for s in range(args.shards):
        start = s * rows_per_shard
        end = min(args.num_rows, (s + 1) * rows_per_shard)
        if start >= end:
            break
        table = _gen_rows(
            num_rows=end - start,
            codebook_sizes=codebook_sizes,
            items_min=args.num_items_min,
            items_max=args.num_items_max,
            seed=args.seed + s,
        )
        path = os.path.join(args.out_dir, f"part-{s:04d}.parquet")
        pq.write_table(table, path, compression="snappy")
        print(f"wrote {path} ({table.num_rows} rows)")
    print(f"\nDone. Codebook: {codebook_sizes}, total rows: {args.num_rows}")


if __name__ == "__main__":
    main()
