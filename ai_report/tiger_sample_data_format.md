# TIGER — sample data format

The exact on-disk schema TIGER expects, why it's shaped this way, and how
to produce it from an upstream `SidRqkmeans`/`SidRqvae` model.

## Schema

One **parquet** row per user, with three columns:

| Column | dtype | Length | Meaning |
|---|---|---|---|
| `history_sids` | `list<int64>` | `num_items_in_history × num_hierarchies` | The user's interaction history, flattened to a SID-token stream. Codes are **0-indexed** per hierarchy slot. |
| `label_sids` | `list<int64>` | exactly `num_hierarchies` | The held-out next item's SID tuple. 0-indexed. |
| `user_id` | `int64` | scalar | (Optional) Used only if the model's `num_user_bins > 0`. Otherwise ignored. |

That's it. No raw item IDs anywhere — the offline pipeline that produces
this parquet has already done the `item_id → SID` mapping.

## Layout of `history_sids`

The SID-token stream interleaves per-item codes contiguously. For a user
with N items in history and `num_hierarchies = 3`, the column contains
`3 × N` ints:

```
[ c0_item0, c1_item0, c2_item0,    c0_item1, c1_item1, c2_item1,    ...,    c0_itemN-1, c1_itemN-1, c2_itemN-1 ]
  └────────item 0────────┘         └────────item 1────────┘                 └────────item N-1───────┘
```

Per-item boundaries are **implicit**: every `num_hierarchies` consecutive
tokens belong to one item. The model reshapes to `(B, items, H)`
internally when injecting SEP tokens.

## Value ranges

| Slot in the tuple | Range |
|---|---|
| Slot `h` (0-based) | `[0, codebook_sizes[h])` |

For the production config in `ft_scripts/tiger.config`:

```
codebook: "8192,8192,8192"
```

All codes are in `[0, 8192)`. Hierarchy slots are identified by **position
within the tuple**, not by tagged ID.

## Padding

Padding is handled by **KJT lengths**, not by sentinel values:

- Variable-length `history_sids` is fine — different users with
  different `num_items_in_history` produce different-length lists.
- The dataloader `to_padded_dense`-s them to a `(B, L_tokens)` tensor at
  batch time, filling with `0` (which IS a valid real code; the model's
  attention mask blocks contributions from padded positions, so the
  collision is harmless — see §6.2 Job 1 of `tiger_migration_design.md`).
- `label_sids` is always exactly `num_hierarchies` long — no padding
  considerations.

## Example rows

For `codebook: "256,256,256"`, a user with 4 items in history might look
like:

| `user_id` | `history_sids` | `label_sids` |
|---:|---|---|
| 42 | `[12, 47, 198,  88, 220, 5,  201, 33, 91,  5, 8, 4]` | `[88, 220, 5]` |
| 7 | `[3, 12, 77,  15, 8, 91]` | `[201, 33, 91]` |
| 99 | `[88, 220, 5,  3, 12, 77,  5, 8, 4,  15, 8, 91,  255, 255, 255]` | `[12, 47, 198]` |

For the production `codebook: "8192,8192,8192"`, the values can range
`0..8191`. The user histories should typically be at least 2 items long
(one for input, one for the held-out label is the leave-one-out split).

## How to produce this parquet from upstream

The typical pipeline:

### Step 1: train and run predict on a Sid* model

```bash
# Train RQ-KMeans on item embeddings.
torchrun ... -m tzrec.train_eval \
    --pipeline_config_path ft_scripts/sid_rqkmeans.config

# Run predict to emit (id, codes) for every item.
python -m tzrec.predict_checkpoint \
    --pipeline_config_path ft_scripts/sid_rqkmeans.config \
    --predict_input_path  '<item_embeddings>.parquet' \
    --predict_output_path 'data/sid_rqkmeans_predict_out/' \
    --reserved_columns 'id'
```

Resulting parquet schema:

```
id     : int64
codes  : list<int64>   # 0-indexed SID tuple, length num_hierarchies
```

### Step 2: offline join — produce per-user SID histories

A small pyarrow/pandas script (out of scope of this repo, but
straightforward):

```python
import pandas as pd
import numpy as np
import pyarrow.parquet as pq

# Item-id → SID lookup.
sid_table = pq.read_table('data/sid_rqkmeans_predict_out/*.parquet').to_pandas()
id_to_sid = dict(zip(sid_table['id'], sid_table['codes']))

# User behavior log: user_id, sequence of item_ids in chronological order.
behavior = pd.read_parquet('data/user_behavior/*.parquet')

rows = []
for user_id, group in behavior.groupby('user_id'):
    items = group.sort_values('ts')['item_id'].tolist()
    if len(items) < 2:
        continue  # need at least 2 items for leave-one-out.
    history_items = items[:-1]
    held_out_item = items[-1]
    # Flatten SID tokens per item.
    history_sids = [
        c
        for itm in history_items
        for c in id_to_sid[itm]
    ]
    rows.append({
        'user_id': user_id,
        'history_sids': history_sids,
        'label_sids': id_to_sid[held_out_item],
    })

df = pd.DataFrame(rows)
df.to_parquet('data/tiger_train/*.parquet')
```

Notes:
- **Leave-one-out split** is the v1 convention (one row per user). v2
  could revisit by adding back per-prefix augmentation, but currently
  this gives a single `(history, next-item)` pair per user.
- **No padding/truncation at this stage** — that's the dataloader's job
  via `sequence_length` in `tiger.config`.

## Synthetic data for testing

A reference generator is provided at `ft_scripts/build_tiger_synth_data.py`.
It produces parquets that **deterministically plant a learnable signal**
(label = (last_history_item + 1) mod codebook_size per hierarchy), so a
trained TIGER model should converge to non-trivial recall on synthetic
data.

```bash
# 1000 training rows, 4 shards.
python ft_scripts/build_tiger_synth_data.py \
    --out_dir data/tiger_synth_train \
    --num_rows 1000 \
    --codebook 8192,8192,8192 \
    --num_items_min 4 --num_items_max 20 \
    --shards 4

# 200 eval rows.
python ft_scripts/build_tiger_synth_data.py \
    --out_dir data/tiger_synth_eval \
    --num_rows 200 \
    --codebook 8192,8192,8192 \
    --shards 1 \
    --seed 7
```

Inspect:

```bash
python -c "
import pyarrow.parquet as pq
t = pq.read_table('data/tiger_synth_train/part-0000.parquet')
print('schema:', t.schema)
print('first row:', {k: t.column(k)[0].as_py() for k in t.column_names})
"
```

Expected output (truncated):
```
schema: history_sids: list<element: int64>
        label_sids:   list<element: int64>
        user_id:      int64
first row: {'history_sids': [6340, 5362, 3595, 3547, 7033, 704, 5712, 1650, 771, 4312, 7992, 6027, 6235, 5877, 6439],
            'label_sids':   [6236, 5878, 6440],
            'user_id': 0}
```

(The 15 history tokens = 5 items × 3 hierarchies; `label_sids` =
`[history_sids[-3] + 1, history_sids[-2] + 1, history_sids[-1] + 1]` mod
8192, the planted relationship.)

## Predict-mode input

For `tzrec.predict_checkpoint`, the input parquet should have the same
schema as training **minus the labels** (or with labels still present —
they're ignored when running in inference mode).

The predict-mode **output** parquet schema (per `_write_predictions` in
`tzrec/main.py` + Tiger's `predict()` return value + `--reserved_columns`):

| Column | Type | Source |
|---|---|---|
| `generated_sids` | `list<int64>` (length K × H, flattened per row) | from `Tiger.predict()` |
| `step_scores`    | `list<float>` (length K × H, flattened per row) | from `Tiger.predict()` |
| `user_id`        | `int64` (if passed via `--reserved_columns user_id`) | reserved from input |

The `K` beams are sorted descending by marginal score (product of
step_scores along each beam). Downstream offline retrieval consumes
`(user_id, generated_sids)` to recover the actual item
recommendations.

## Quick sanity checks

For a parquet you produced, confirm:

```python
import pyarrow.parquet as pq

t = pq.read_table('data/tiger_train/*.parquet')

# 1. Schema: exactly the three required columns (plus optional user_id).
assert 'history_sids' in t.column_names
assert 'label_sids'   in t.column_names

# 2. label_sids has the right length (num_hierarchies = 3 for "8192,8192,8192").
import numpy as np
label_lengths = np.array([len(x) for x in t['label_sids'].to_pylist()])
assert (label_lengths == 3).all(), 'expected every label_sids to be length 3'

# 3. history_sids is a multiple of num_hierarchies.
hist_lengths = np.array([len(x) for x in t['history_sids'].to_pylist()])
assert (hist_lengths % 3 == 0).all(), 'history_sids must be a multiple of num_hierarchies'

# 4. All code values are in [0, 8192).
import itertools
all_codes = list(itertools.chain.from_iterable(t['history_sids'].to_pylist())) + \
            list(itertools.chain.from_iterable(t['label_sids'].to_pylist()))
all_codes = np.array(all_codes)
assert (all_codes >= 0).all() and (all_codes < 8192).all(), 'codes out of range'

# 5. histories are non-trivial (at least 1 item).
assert (hist_lengths >= 3).all(), 'each history must have at least one item'

print('All checks passed:', len(t), 'rows')
```

## Pointers

- Model behavior: `ai_report/tiger_model_docs.md`
- Full design: `ai_report/tiger_migration_design.md`
- Config example: `ft_scripts/tiger.config`
- Synthetic data generator: `ft_scripts/build_tiger_synth_data.py`
- Standalone smoke test: `ft_scripts/tiger_smoke.py`
