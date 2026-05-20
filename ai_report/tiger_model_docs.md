# Tiger — generative recommendation with semantic IDs

User-facing guide for the `Tiger` model in TorchEasyRec. See
`tiger_migration_design.md` for the deeper design rationale.

## What it does

Given a user's history of items encoded as **Semantic ID (SID) tokens**,
TIGER autoregressively generates the SID of the next item via beam search.
Each SID is a tuple of `num_hierarchies` integer codes — produced upstream
by a residual quantization model like `SidRqkmeans` or `SidRqvae`. TIGER
does not see raw item IDs; the offline pipeline that prepares its training
parquet is responsible for the `item_id → SID` mapping.

This is a port of GRID's TIGER implementation
(`src/models/modules/semantic_id/tiger_generation_model.py`) to the
TorchEasyRec framework. v1 covers architecture + forward + beam search +
retrieval metrics. Constrained beam search (GRID's `_check_valid_prefix`)
is **deferred to v2**.

## Architecture in one paragraph

T5 encoder (`num_encoder_layers` blocks) + T5 decoder stack with
cross-attention (`num_decoder_layers` blocks). The T5-internal embedding
tables are **deleted** at construction; instead, a single compact
`nn.Embedding(sum(codebook_sizes), embed_dim)` is shared between encoder
and decoder. Per-hierarchy offsets map raw 0-indexed codes into disjoint
slices of this table. The decoder ends with `num_hierarchies` per-position
`Linear(embed_dim, codebook_sizes[h], bias=False)` heads — one per
hierarchy step. A learnable BOS prompts the decoder; a learnable SEP is
inserted between every item's SID block in the encoder sequence. With
`num_user_bins > 0`, a remainder-hashed `nn.Embedding(num_user_bins,
embed_dim)` is prepended to the encoder input.

## Files

| Path | What's in it |
|---|---|
| `tzrec/protos/models/tiger.proto` | `Tiger` proto message — all knobs |
| `tzrec/models/tiger.py` | `Tiger(BaseModel)` model class |
| `tzrec/modules/tiger_ff.py` | `T5MultiLayerFF` — drop-in replacement for `T5LayerFF` when `hidden_dims` has length > 1 |
| `tzrec/models/tiger_test.py` | unittest covering forward, backward, beam search, metrics |
| `tzrec/tests/configs/tiger_mock.config` | minimal config used by the unittest |
| `ft_scripts/tiger.config` | production-shaped config with `codebook: "8192,8192,8192"` |
| `ft_scripts/train_tiger.sh` | torchrun launcher wrapping `tzrec.train_eval` |
| `ft_scripts/build_tiger_synth_data.py` | synthetic data generator for sanity tests |
| `ft_scripts/tiger_smoke.py` | standalone smoke test (no parquet, no torchrun) |
| `ft_scripts/tiger_train_launcher.py` | helper launcher with in-process dynamicemb shim (for environments where the local source's `dynamicemb_util.py` mismatches the installed torchrec) |

## Proto fields (`tzrec/protos/models/tiger.proto`)

| Field | Default | Meaning |
|---|---|---|
| `embed_dim` | 128 | T5 hidden/embedding width (`d_model`). |
| `num_heads` | 6 | Attention heads per T5 block. |
| `d_kv` | 64 | Per-head key/value dim. |
| `hidden_dims` | `"1024"` | Comma-separated FF hidden widths. Length 1 = stock T5 FF; length > 1 triggers the `T5MultiLayerFF` swap (e.g. `"1024,1024"` is GRID's `mlp_layers=2`). |
| `num_encoder_layers` | 4 | T5 blocks in encoder. |
| `num_decoder_layers` | 4 | T5 blocks in decoder. |
| `dropout_rate` | 0.15 | Inside attention + FF. |
| `codebook` | `"256,256,256,256"` | Per-hierarchy code count, comma-separated. Length = `num_hierarchies`; values = `codebook_sizes[h]`. Non-uniform supported (`"256,256,256,128"`). |
| `num_user_bins` | 0 | 0 = no user-id branch. `> 0` enables remainder-hashed user embedding. |
| `use_sep_token` | true | Insert a learnable SEP between every item's SID block in the encoder sequence. |
| `beam_width` | 10 | K for top-K beam search. |
| `history_group_name` | `"history"` | `FeatureGroupConfig.group_name` to look up for the SID-history sequence. Tiger consumes the first `feature_name` in that group. |
| `user_group_name` | `"user"` | `FeatureGroupConfig.group_name` to look up for `user_id` (used only when `num_user_bins > 0`). |

Field numbers 11 / 21 / 31 are reserved for v2 (`semantic_id_path`,
`weight_tying`, `constrained_beam_search`).

## Feature wiring

`feature_groups` are declared at the **ModelConfig** level. Tiger reads
`history_group_name` (and optionally `user_group_name`) from its proto,
matches against the `group_name` in each `feature_groups` entry, and
consumes the **first** `feature_name` listed in the matching group. So
the user can rename freely:

```
model_config {
    feature_groups {
        group_name: "my_renamed_sid_seq"
        feature_names: "my_encoded_history"
        group_type: DEEP
    }
    tiger {
        history_group_name: "my_renamed_sid_seq"
        ...
    }
}
```

Inside the model, the resolution happens in `__init__` via
`_resolve_group_and_feature` — Tiger validates at construction time
(not at first forward) that the named group exists in
`model_config.feature_groups` and that it has at least one
`feature_name`. Misconfiguration produces a clear `ValueError`
identifying the missing group.

## End-to-end workflow

```
┌────────────────────────────────────────────────────────────────────────┐
│ Step 1: Train the SID generator (SidRqkmeans or SidRqvae)             │
│   → produces a per-item SID lookup parquet via tzrec.predict_checkpoint │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 2: Offline join — for each user, look up SIDs for every item     │
│   in their history and for their held-out next item. Emit parquet     │
│   with columns: history_sids, label_sids, user_id (optional).         │
│   See ai_report/tiger_sample_data_format.md for the exact schema.     │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 3: Train Tiger                                                    │
│   torchrun -m tzrec.train_eval                                         │
│     --pipeline_config_path ft_scripts/tiger.config                     │
│     --train_input_path 'data/tiger_train/*.parquet'                    │
│     --eval_input_path  'data/tiger_eval/*.parquet'                     │
│     --model_dir experiments/tiger                                      │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 4: Predict — emit generated SIDs                                  │
│   torchrun -m tzrec.predict_checkpoint                                 │
│     --pipeline_config_path ft_scripts/tiger.config                     │
│     --predict_input_path  'data/tiger_predict_in/*.parquet'            │
│     --predict_output_path 'data/tiger_predict_out/'                    │
│     --reserved_columns user_id                                         │
│   Output columns: user_id (reserved), generated_sids, step_scores      │
└────────────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌────────────────────────────────────────────────────────────────────────┐
│ Step 5: Offline retrieval (out of scope of this model)                 │
│   Join generated_sids against the catalog SID → item table to recover  │
│   item recommendations. Beams whose SID matches no real item are       │
│   naturally dropped.                                                   │
└────────────────────────────────────────────────────────────────────────┘
```

## Conventions

- **0-indexed codes throughout.** `history_sids` and `label_sids` on
  disk, the model's per-hierarchy MLP outputs, and the emitted
  `generated_sids` are all 0-indexed: `c_h ∈ [0, codebook_sizes[h])`.
  Padding identification uses KJT lengths, not sentinel values.
- **History flattened to a token stream.** `history_sids` is
  `list<int64>` of length `num_items × num_hierarchies`. Per-item
  boundaries are implicit (every `num_hierarchies` consecutive tokens
  belong to one item).
- **Labels are pre-encoded SID tuples.** `label_sids` is `list<int64>`
  of length `num_hierarchies`. No model-side `item_id → SID` lookup.
- **Compact embedding table.** Size `sum(codebook_sizes)` rows, not
  `max × num_hierarchies`. Per-hierarchy `code_offsets[h] =
  sum(codebook_sizes[:h])` (precomputed).
- **Output is 0-indexed and sorted.** `generated_sids: (B, K, H)` is
  sorted descending by per-beam marginal score. `step_scores: (B, K, H)`
  is the per-hierarchy softmax probability of the selected code at each
  step.

## Beam-search behavior (v1)

Plain unconstrained autoregressive top-K beam search. No prefix-validity
pruning. Beams may emit SID tuples that don't correspond to any real
item; the **offline retrieval step** (out of scope) maps SIDs back to
items and naturally filters hallucinations.

Concretely, for each user `b` and each step `h ∈ [0, num_hierarchies)`:

1. Run the decoder over BOS + already-emitted prefix.
2. Apply `decoder_mlp[h]` → logits over `codebook_sizes[h]` codes.
3. Softmax → probabilities.
4. At step 0: take top-K codes per user (expand beam from 1 to K).
5. At step h > 0: joint = parent_marginal × per-step probs; take top-K
   from the `K × C` joint space; reorder by parent beam.

The returned `marginal_probs` (implicit) is the product of per-step
probabilities along each beam.

## Loss

Per-hierarchy cross-entropy, summed across hierarchies. With random
init and codebook width `C`, the initial loss is ≈ `num_hierarchies ·
ln(C)` (for `num_hierarchies=3, C=8192`: ≈ 27).

## Metrics

`init_metric()` registers:

- `tiger_ce` — MeanMetric over the summed cross-entropy.
- `recall@K` and `ndcg@K` from `torchmetrics.retrieval`, for K ∈
  `{5, beam_width}`. Match logic: a ground-truth SID is "retrieved" iff
  it equals one of the beam-emitted tuples.

Used by the engine at eval-checkpoint cadence: `update_metric` runs
beam search internally and updates the retrieval modules with `(preds,
target, indexes)` triples constructed from the beam outputs.

## Distributed training

Plain `nn.Embedding` instances (not TorchRec sharded). All parameters go
to the **dense optimizer**. TorchRec's `DistributedModelParallel`
automatically wraps non-sharded modules in `DistributedDataParallel` —
gradients all-reduce across ranks, weights stay in sync. The TorchRec
planner produces a trivial plan; `create_train_pipeline` falls back to
`TrainPipelineBase` (the right choice for a dense-compute-dominated
model like TIGER). See §5.1 of `tiger_migration_design.md` for the full
DDP-correctness argument.

## Parameter count

For the production config `codebook: "8192,8192,8192"`, `embed_dim:
128`, `hidden_dims: "1024"`, 4 encoder + 4 decoder layers:

| Component | Params |
|---|---:|
| Encoder (4 T5Blocks) | ~3.0 M |
| Decoder (4 T5Blocks with cross-attn) | ~3.4 M |
| `sid_embedding` (24576 × 128) | 3.1 M |
| `decoder_mlp` (3 × Linear(128, 8192, bias=False)) | 3.1 M |
| BOS + SEP | 256 |
| **Total** | **~12.6 M** |

(Smoke test reports 6.95 M for the smaller `embed_dim=128,
hidden_dims=256, num_layers=2,2` debug config.)

## Verifying the implementation

```bash
# Quickest sanity check (no parquet needed) — runs the full Tiger logic
# on a synthetic in-memory batch and checks forward, backward, beam
# search, and that loss decreases over 20 optimizer steps:
docker exec -w /workspace/fangtinglin/codework/TorchEasyRec torchgpuv3 \
    python ft_scripts/tiger_smoke.py
```

For the formal unit tests:

```bash
docker exec -w /workspace/fangtinglin/codework/TorchEasyRec torchgpuv3 \
    python -m unittest tzrec.models.tiger_test
```

(Both require a tzrec environment compatible with the local source. If
the container has an older torchrec than the local source expects, see
`ft_scripts/tiger_train_launcher.py` for a shim pattern.)

## Known limitations (v1)

- **No constrained beam search.** Deferred to v2; see §0.1 of
  `tiger_migration_design.md` and the design exploration in
  `tiger_beam_search_pruning_comparison.md` and
  `tiger_semantic_id_path_optimized_design.md`.
- **No `semantic_id_path`.** v1 expects pre-SID-encoded data in the
  training parquet; the upstream offline pipeline does the `item_id →
  SID` mapping. v2 will introduce the proto field and the
  associated `codebooks` buffer.
- **No item-ID mapping in outputs.** Predict emits raw SID tuples;
  mapping back to items is the responsibility of the downstream offline
  retrieval step.
- **TorchScript export untested in v1.** The encoder-only export pattern
  (§7.4 of the design doc) is the recommended path if `tzrec.predict`
  speed matters; `tzrec.predict_checkpoint` works out of the box.

## Pointers

- Design rationale: `ai_report/tiger_migration_design.md`
- Sample data format: `ai_report/tiger_sample_data_format.md`
- v2 pruning design exploration: `ai_report/tiger_beam_search_pruning_comparison.md`,
  `ai_report/tiger_semantic_id_path_optimized_design.md`
- GRID-vs-T5base parameter comparison: `/mnt/fangtinglin/GRID/TIGER_vs_T5base_params.md`
