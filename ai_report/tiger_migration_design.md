# TIGER → TorchEasyRec Migration Design

Status: design draft. **No code in this document** — this is a structural plan that resolves the questions you'll have to answer before writing the migration PR.

## 0. Scope

We're porting **sections 2.3 – 4.4** of `GRID/TIGER_implementation.md` into TorchEasyRec. Concretely:

| In scope (this design covers) | Out of scope |
|---|---|
| §2.3 — model input contract (`input_ids`, `attention_mask_encoder`, `user_id`, target SID) | §1 — Hydra configuration (replaced by proto, §2 below) |
| §2.4 — three-job attention mask | §2.1 — TFRecord reader (replaced by tzrec's `ParquetReader`/`OdpsReader`) |
| §3.x — module composition (encoder, decoder, SID embedding table, BOS/SEP, decoder MLPs, T5MultiLayerFF, optional user embedding) | §2.2 — `collate_with_sid_causal_duplicate` (offline pre-expansion proposed in §3.4 below) |
| §4.1 — encoder forward (offset trick + SEP injection) | §5 — Lightning training loop (replaced by tzrec's `TrainPipelineSparseDist`) |
| §4.2 — decoder forward + summed per-hierarchy CE loss | §6 — parameter-count math (informational only) |
| §4.3 — constrained beam search | §7 — config knobs (folded into proto, §2 below) |
| §4.4 — `SIDRetrievalEvaluator` (`NDCG`/`Recall@K`) | |

The target shape: a new `Tiger` subclass of `tzrec.models.model.BaseModel` (`tzrec/models/tiger.py`) driven by a new `tzrec/protos/models/tiger.proto` message, fully participating in the train/eval/predict lifecycle documented in `ai_report/TZREC_PIPELINE.md`.

---

## 0.1 Version scope — what's in v1, what's deferred to v2

This design covers **v1**. The following features are intentionally **deferred to v2** to keep the first version's implementation small:

| Deferred feature | What it would add | Why deferred from v1 |
|---|---|---|
| **Constrained beam search** | Beam search prunes beams whose prefix doesn't match any real item, eliminating hallucinated SIDs at decode time. GRID's `_check_valid_prefix` is the reference. | Needs a per-item or per-prefix lookup structure (see `tiger_semantic_id_path_optimized_design.md` for a scalable design via lex-sorted `(N, H)` tensor). The lifecycle (bootstrap from disk on train, restore from checkpoint on eval/predict) requires mode-detection plumbing that we'd rather not add until the pruning behavior is actually shipped. |
| **`semantic_id_path` proto field** | Bootstraps the prefix-validity table from upstream `SidRq*` predict output. | Only used by constrained beam search. Removed in v1; reintroduced in v2 alongside the chosen pruning data structure. |
| **`codebooks` persistent buffer** | In-memory representation of the prefix-validity table (also doubled in v1 drafts as the item_id → SID lookup, but that role is now obsoleted by pre-SID-encoded data; see §3 below). | Same as above — exists only to support pruning. |
| **Mode-detection mechanism** (passing `Mode` to `BaseModel.__init__`) | Lets the model know whether to bootstrap from disk (train) or restore from checkpoint (eval/predict). | Required only for the codebooks-buffer lifecycle. Without that buffer, v1 has no mode-dependent `__init__` behavior. |
| **Item → SID lookup inside the model** | The runtime catalog lookup GRID does via `map_sparse_id_to_semantic_id`. | v1 sidesteps this by pre-encoding the data offline: the training/eval/predict parquet rows carry SID tokens directly (see §3.1). The model never sees a raw `item_id`. |

**v1 beam search behavior**: plain unconstrained top-K beam decoding. Emits `(B, K, num_hierarchies)` SID tuples + `(B, K)` scores. A separate **offline retrieval step (out of scope for this design)** is responsible for joining each emitted SID against the catalog to recover item IDs — and naturally drops any hallucinated SIDs that don't decode to an item. The eval-time `recall@K` / `ndcg@K` metrics work on exact-SID match against the ground-truth held-out SID, so they remain well-defined without constrained beam search (hallucinated beams simply count as misses).

**Forward compatibility commitment**: every v1 design decision documented below is chosen so that adding the v2 features is purely additive — no breaking changes to the proto, the model class, the on-disk row schema, or the engine integration. Specifically:

- The `Tiger` proto message will gain `semantic_id_path` and `constrained_beam_search` fields in v2 without renumbering or removing existing fields.
- The model class will gain a `codebooks` buffer and a `_check_valid_prefix`-equivalent helper without changing the `predict()` / `loss()` / `update_metric()` signatures.
- The data parquet schema (SID-encoded history + label) stays identical between v1 and v2.

References for the deferred design work:
- `ai_report/tiger_beam_search_pruning_comparison.md` — compares the pruning strategies (GRID full-prefix vs per-level vocab vs no constraint).
- `ai_report/tiger_semantic_id_path_optimized_design.md` — the scalable lex-sorted `(N, H)` tensor design for full-prefix pruning at arbitrary `(H, C)` scales.

---

## 1. Architectural mapping at a glance

| GRID concept | TorchEasyRec equivalent | Notes |
|---|---|---|
| `Hydra @package _global_` YAML | `tzrec.protos.models.tiger.Tiger` proto | See §2. |
| `LightningModule` + `TransformerBaseModule` | `tzrec.models.model.BaseModel` subclass | See §5. |
| Hydra `_target_` instantiation | Class registry via `BaseModel.create_class(name)` | Auto-registers when class declared with the registry metaclass. |
| `SemanticIDDatasetConfig` + `TFRecordIterator` + `map_sparse_id_to_semantic_id` | `ParquetDataset`/`OdpsDataset` + `feature_configs` (sequence_feature carrying **already-SID-encoded** int tokens) — **no runtime item→SID lookup in v1** | See §3. |
| `collate_with_sid_causal_duplicate` (online contiguous-subseq augmentation) | **Dropped.** One user sequence → one training pair (full history → held-out last item) | See §3.4. |
| `NextKTokenMasking(next_k=num_hierarchies)` label transform | One `list<int64>` label column = the held-out item's SID tuple directly (no mapping needed) | See §3.1. |
| GRID `Batch` (`SequentialModelInputData` + `SequentialModuleLabelData`) | `tzrec.datasets.utils.Batch` with sparse-feature KJT + label tensor | See §4. |
| `T5EncoderModel` + `T5Stack` (decoder) | Same HF classes, instantiated in `__init__` | See §5.2. |
| `item_sid_embedding_table_encoder` (1024×128, padded `max × num_hierarchies`) | `self.sid_embedding`, a plain `nn.Embedding(sum(codebook_sizes), embed_dim)` — compact form, no padding rows. NOT a TorchRec `EmbeddingCollection`. | See §5.3. |
| `decoder_mlp: ModuleList[Linear]` per hierarchy | Same `ModuleList` of dense `Linear` heads | See §5.4. |
| `codebooks` tensor used by constrained beam search | **Deferred to v2** — see §0.1 | — |
| `eval_step` running `generate` + retrieval metric | `update_metric(predictions, batch)` calling `generate` inside `torch.no_grad()` | See §7. |
| GRID `ckpt_path` + Lightning checkpoint | tzrec `model_dir/model.ckpt-N/` (handled by engine) | Nothing model-specific to do. |

---

## 2. Proto definitions

### 2.1 New file: `tzrec/protos/models/tiger.proto`

Defines a single message `Tiger` carrying every TIGER-specific knob. **Field naming follows `sid_model.proto` conventions** (proto2 with `optional ... [default = X]`, snake_case, comma-separated strings for per-layer lists, no `should_` / `do_` prefixes on booleans) so that a config that has both a `SidRqvae` block and a downstream `Tiger` block reads consistently end-to-end.

Two design choices borrowed directly from `SidRqvae`/`SidRqkmeans`:

- **`codebook` is a single comma-separated string**, not two numeric fields. `codebook: "256,256,256,256"` means "4 hierarchies, each with 256 codes". The list length **is** the number of hierarchies; the values give the per-hierarchy code count. This: (a) deletes the `num_hierarchies × codebook_width` consistency-check footgun, (b) supports non-uniform codebooks for free (e.g. `"256,256,256,128"` if the dedup digit only needs 128 buckets), and (c) matches exactly how the upstream SID-generator config declared its codebooks — users copy-paste the same string.
- **`hidden_dims` is a single comma-separated string** describing the feed-forward block, not two fields. `hidden_dims: "1024"` → stock T5 single-hidden FF; `hidden_dims: "1024,1024"` → two-hidden `T5MultiLayerFF` swap (the old `d_ff + mlp_layers` pair). Length = number of FF hidden layers; values = per-layer FF widths. Matches `SidRqvae.hidden_dims`.

#### Field listing (proto2)

All fields `optional` so the message is forward/backward compatible. Field numbers shown to make the proto file order explicit.

**Backbone shape**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 1 | `embed_dim` | `uint32` | `128` | The model's hidden width. Replaces GRID's `d_model`. Aligned with `SidRqvae.embed_dim`. |
| 2 | `num_heads` | `uint32` | `6` | Attention heads per T5 block. |
| 3 | `d_kv` | `uint32` | `64` | Per-head key/value dim (T5 standard name; kept). |
| 4 | `hidden_dims` | `string` | `"1024"` | Comma-separated FF hidden widths. Length controls how many hidden layers the FF block has; if > 1, replace every `T5LayerFF` with `T5MultiLayerFF`. |
| 5 | `num_encoder_layers` | `uint32` | `4` | Number of `T5Block`s in the encoder. |
| 6 | `num_decoder_layers` | `uint32` | `4` | Number of `T5Block`s in the decoder. |
| 7 | `dropout_rate` | `float` | `0.15` | Dropout inside attention + FF. T5 standard name; kept. |

**SID structure**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 10 | `codebook` | `string` | `"256,256,256,256"` | Per-hierarchy code count. List length → number of hierarchies; values → per-layer width. Drives the SID embedding-table size, the per-hierarchy offset stride, and the per-hierarchy `decoder_mlp` head widths. The max value is also handed to `T5Config.vocab_size` as a placeholder (see §5.2 for why that field is still needed). Must match the SID encoding produced by the upstream offline pipeline that generated the training parquet's `history_sids` and `label_sids` columns (see §3.1). |

**Codebook source** — *(deferred to v2, see §0.1; v1 has no `semantic_id_path` field)*

**Optional user features**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 12 | `num_user_bins` | `uint32` | `0` | `0` disables the user-id branch entirely. `> 0` enables remainder-hashed `nn.Embedding(num_user_bins, embed_dim)` prepended to the encoder input. |

**Forward-pass toggles**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 20 | `use_sep_token` | `bool` | `true` | Insert a learnable SEP between every item's `n` SID tokens in the encoder sequence. Replaces GRID's `should_add_sep_token`. |

**Beam search / generation**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 30 | `beam_width` | `uint32` | `10` | Beam size during unconstrained autoregressive decoding. Replaces GRID's `top_k_for_generation`. |

*(v1 omits the constrained-beam-search flag — deferred to v2, see §0.1. Field number 31 is reserved for the v2 addition so the proto remains backward-compatible.)*

**Output naming** — *no proto fields*

Removed in v1. The predict-mode output column names are **hardcoded by the model**: `Tiger.predict()` (in inference mode) returns

```
{
  "generated_sids": (B, K, num_hierarchies) int64,   # 0-indexed (same as label_sids), sorted descending by marginal score
  "step_scores"   : (B, K, num_hierarchies) float32, # per-hierarchy softmax prob at the selected code
}
```

(See §7.2 step 6 for the full spec.) Users include row-identifier columns from the input parquet (e.g. `user_id`) by passing `--reserved_columns user_id` to `tzrec.predict` / `tzrec.predict_checkpoint` — the engine's `_write_predictions` (`tzrec/main.py:998`) merges those reserved columns with the model's predictions into the output parquet. No proto field is needed for either side. This matches the convention used by `SidRqkmeans` (returns `{"codes": ...}`, users pass `--reserved_columns "id"` to get the item ID through).

**Feature wiring**

| # | Name | Type | Default | Meaning |
|---|---|---|---|---|
| 50 | `feature_groups` | `repeated FeatureGroupConfig` | — | Two named groups: `"history"` (the sequence of past raw `item_id`s) and `"user"` (the user_id feature; required only if `num_user_bins > 0`). Conventional in tzrec; the model consumes them via `self.get_features_in_feature_groups(...)`. |

#### Field-naming changes vs an earlier "GRID-direct" port

For anyone who saw an earlier draft of this design with GRID-style field names, here's the rename log:

| Old (GRID-direct) | New (RQVAE-aligned) | Why |
|---|---|---|
| `d_model` | `embed_dim` | Match `SidRqvae.embed_dim`; more descriptive — it's the embedding/hidden width, T5 just calls it `d_model`. |
| `d_ff` + `mlp_layers` | `hidden_dims` (string) | Mirrors `SidRqvae.hidden_dims`. One field instead of two; length encodes `mlp_layers`, values encode `d_ff` per layer. Non-uniform FF widths come for free. |
| `num_hierarchies` + `codebook_width` | `codebook` (string) | Mirrors `SidRqvae.codebook` / `SidRqkmeans.codebook`. Eliminates the `num_hierarchies × codebook_width` consistency-check footgun; non-uniform codebooks come for free. |
| `should_add_sep_token` | `use_sep_token` | Drop verb-prefix per `SidRqvae` style (`shared_codebook`, `normalize_residuals`, `rotation_trick`). |
| `top_k_for_generation` | `beam_width` | Standard NMT terminology; shorter; says what it is. |
| `should_check_prefix` | *(deferred to v2 as `constrained_beam_search`)* | — |

### 2.2 Wiring into `tzrec/protos/model.proto`

The top-level `ModelConfig` is a `oneof model { ... }` containing one entry per supported model. We add:

```
  Tiger tiger = <next_free_field_number>;
```

And add `import "tzrec/protos/models/tiger.proto";` at the top of `model.proto`. Both `_pb2.py` and `_pb2.pyi` are regenerated via `bash scripts/gen_proto.sh` (per `CLAUDE.md`).

### 2.3 Pipeline config layout that users will write

A minimal user-facing config will look like:

```
model_dir: "experiments/tiger_amazon_beauty"
train_input_path: ".../taobao_or_amazon_prefix_pairs/*.parquet"
eval_input_path:  ".../taobao_or_amazon_eval/*.parquet"
train_config { sparse_optimizer { ... }  dense_optimizer { adam_optimizer { lr: 1e-3 } } num_epochs: 10 }
eval_config  { num_steps: ... }
data_config  { batch_size: 32  dataset_type: ParquetDataset  fg_mode: FG_NONE
               label_fields: "label_sids"  num_workers: 8 }
feature_configs { sequence_feature { feature_name: "history_sids" sequence_length: 120 ... } }    # SID-token stream, length = items × num_hierarchies
feature_configs { id_feature       { feature_name: "user_id"      num_buckets: ... embedding_dim: ... } }
model_config { tiger {
    embed_dim: 128
    num_heads: 6
    d_kv: 64
    hidden_dims: "1024,1024"          # 2-hidden-layer T5MultiLayerFF; "1024" alone → stock T5
    num_encoder_layers: 4
    num_decoder_layers: 4
    dropout_rate: 0.15
    codebook: "256,256,256,256"       # 4 hierarchies × 256 codes each
    num_user_bins: 0                  # disable user-id branch
    beam_width: 10
    feature_groups { ... }
    # v1 has no semantic_id_path / constrained_beam_search fields — deferred to v2 (§0.1)
} }
```

We default to `fg_mode: FG_NONE` because TIGER does not need on-the-fly FG (no bucketization, no combo crossing, no hashing) — all features are already integer item IDs that the model itself converts to SID tokens at runtime. If `num_user_bins > 0` we accept `FG_DAG` too (cheap and lets users add user-side raw features later).

---

## 3. Data-pipeline mapping (replaces GRID §2.1, §2.2)

### 3.1 Row contract on disk

**One row per user.** Each row carries the user's full interaction history **already encoded as a flat SID-token stream** plus the held-out next item's **SID tuple**. The item_id → SID mapping is done by the upstream offline pipeline (out of scope for this design); the model never sees raw item IDs.

| Column | dtype | Meaning |
|---|---|---|
| `history_sids` | `list<int64>` | The user's history as a **flat sequence of SID tokens**: `num_items_in_history × num_hierarchies` ints, with codes `[c₀^{item₀}, c₁^{item₀}, …, c_{H-1}^{item₀}, c₀^{item₁}, …]` laid out contiguously. Length is always a multiple of `num_hierarchies`. **Codes are 0-indexed** — `c_h ∈ [0, codebook_sizes[h])` (see §3.6). |
| `user_id` | `int64` (optional) | Used only if `num_user_bins > 0`. |
| `label_sids` | `list<int64>`, length = `num_hierarchies` | The held-out last item's SID tuple — `(c₀, c₁, …, c_{H-1})`, **0-indexed** per the same convention. |
| `__source_id` / `__row_idx` | (injected by `ParquetReader`) | Row-level dataloader checkpoint metadata. |

`reserved_columns` (predict mode) can include `user_id` so the writer can emit `(user_id, generated_sids, scores)` rows; downstream offline retrieval joins those SIDs back to item_ids.

### 3.6 Convention: codes are stored 0-indexed; per-hierarchy offset shift happens inside the model

**SID codes in `history_sids` and `label_sids` are stored 0-indexed** — each per-hierarchy code is in `[0, codebook_sizes[h])`. The same convention applies end-to-end: `label_sids` (input), `generated_sids` (predict output in §7.2), `target_sids` (eval comparison in §8.2) are all 0-indexed. No `+1` / `-1` conversions at any interface.

The per-hierarchy offset shift to map into the compact unified embedding-table index range happens **only inside the model**, in `_add_repeating_offset_to_rows`:

```
lookup_idx[h] = c[h] + cumsum(codebook_sizes)[h]
              = c[h] + sum(codebook_sizes[:h])
```

Concrete example with `codebook: "128,256,512"`:

| SID tuple in input | Lookup indices inside the model |
|---|---|
| `[0, 0, 0]` | `[0, 128, 384]` (first code at each hierarchy) |
| `[1, 2, 3]` | `[1, 130, 387]` |
| `[127, 255, 511]` | `[127, 383, 895]` (last valid row of each hierarchy) |
| `[0, 0, 0, 0]` (padding slot) | `[0, 0, 0, 0]` after mask-multiply; see §6.2 Job 1 |

**Padding identification in tzrec is by KJT lengths, not by a sentinel value.** The attention mask (derived from KJT lengths via `lengths_to_mask`) is what marks padding positions; the actual value sitting in a padded slot is irrelevant — it gets zeroed in the mask-multiply (Job 1) and blocked from attention (Job 3) regardless. So the fact that `to_padded_dense` happens to fill padding with `0` (which is also a valid real code-0-at-hierarchy-0) is harmless: the collision exists but the mask blocks any contribution from those positions.

Why 0-indexed:

1. **Convention alignment.** PyTorch (`embedding_dim`, `num_embeddings`), `F.cross_entropy` (0-indexed class targets), `torch.argmax` (0-indexed result), HuggingFace tokenizers (0-indexed `vocab_size`) — every adjacent piece of the stack uses 0-indexed integer IDs. Going 1-indexed would force `+1` / `-1` conversions at every boundary (input shift, CE target, output restoration) — three extra arithmetic ops, three extra places for off-by-one bugs.
2. **Fewer arithmetic operations in the hot path.** The offset trick becomes `mask * (c + offsets)` — one fewer op than `mask * ((c-1) + offsets)`.
3. **Matches `SequenceFeature` semantics.** When `history_sids` is declared as a `sequence_feature` with `num_buckets = max(codebook_sizes)`, the natural bucket range is `[0, num_buckets)` — declaring `num_buckets: 257` to leave a sentinel slot for "padding=0" would be awkward.
4. **No information loss vs 1-indexed.** Padding identification doesn't need a sentinel value because KJT lengths already encode it; the attention mask propagates through the model correctly under either convention. Choosing 0-indexed gives the same correctness with less indexing-convention drift.

### 3.2 `feature_configs`

- **`history_sids`** declared as a `sequence_feature` of int64 SID tokens with:
  - the sequence's max length = the pipeline's `sequence_length` knob expressed in **tokens**, not items (e.g. 30 items × 4 hierarchies = `sequence_length: 120`),
  - sparse / id-feature type,
  - placed in `data_group: "history"`. Per-item boundaries are implicit: every `num_hierarchies` consecutive tokens belong to one item — the model handles the per-item reshape internally during SEP injection (§6.2).
- **`user_id`** as an `id_feature` in `data_group: "user"`, with `num_buckets = num_user_bins`, `embedding_dim` ignored by us — we use our own remainder-hashed user embedding inside the model, not the embedding produced by tzrec's `EmbeddingGroup`. (Alternative: rely on tzrec's `EmbeddingGroup` and remove the model's `user_embedding`; see §10 — Open Decisions.)
- **No `label_fields` feature_config entry** — `data_config.label_fields = "label_sids"` is enough for the parser to grab the int64 list and produce the right tensors.

### 3.3 What lands in `Batch`

After `DataParser.parse` + `to_batch`, the model receives a `Batch` with:

- `batch.sparse_features["history"]`: a `KeyedJaggedTensor` containing the key `"history_sids"`. Per-sample lengths describe how many SID tokens each history has (always a multiple of `num_hierarchies`). Values are **0-indexed** per-hierarchy codes in `[0, codebook_sizes[h])` (see §3.6 convention). Padding slots from `to_padded_dense` hold `0` — harmless because attention masks them (Job 3) regardless of collision with real code-0-at-h0. The model applies the `c + cumsum_h` offset shift (§6.2 Job 1) at lookup time to map into the unified `sum(codebook_sizes)`-row SID embedding table.
- `batch.sparse_features["user"]`: a `KeyedJaggedTensor` with key `"user_id"` and length 1 per sample. Skipped if `num_user_bins == 0`.
- `batch.labels["label_sids"]`: a `(B, num_hierarchies)` int64 tensor (the `DataParser` materializes a `list<int64>` label column as `label_sids.values` + `label_sids.lengths`; the model reshapes/`view`s into `(B, num_hierarchies)` since every row has a fixed length).
- `batch.checkpoint_info`: row-level resume metadata (handled by the engine; the model ignores it).

### 3.4 No contiguous-subsequence augmentation

**Decision: drop `collate_with_sid_causal_duplicate` entirely.** Each user becomes one training row (full history → held-out last item, the row contract in §3.1). No online expansion, no offline pre-expansion utility, no custom `BaseDataset` subclass.

Rationale:

- **Framework fit.** tzrec's pipeline assumes "one input row = one batch sample"; the augmentation would either fight `ParquetReader`'s row-level checkpointing (a custom dataset that emits a variable number of records per input row breaks the `(__source_id, __row_idx)` accounting) or require regenerating the parquet on every data refresh.
- **Operational simplicity.** No precomputation step in the workflow. Re-training on a new SID generation is one `torchrun -m tzrec.train_eval ...` command.
- **Sufficient signal.** TIGER's amplification trick is most valuable when raw sequences are short and rare. For the Amazon Beauty / Sports / Toys datasets and similar production-scale user sequences, the leave-one-out baseline is already strong; the GRID paper itself reports its main numbers without per-step variance attributable to the augmentation.

Trade-off accepted: training sees fewer `(prefix, next-item)` examples than GRID would for the same raw data. If a benchmark gap opens vs the GRID reference, revisit by either (a) reintroducing the augmentation as a custom dataset subclass after the first milestone lands, or (b) augmenting offline at parquet-prep time as a separate one-off utility — both deferred for now.

### 3.5 Why `fg_mode: FG_NONE` (or `FG_DAG` with empty FG)

TIGER does not bucketize, hash, or combo-cross any input. The history is already SID-encoded by the offline pipeline; the label is already a SID tuple. The encoder just looks up SID-token embeddings via its own table (offset trick on the model side; §6.2). FG is a no-op for this model.

---

## 4. Tensor representation: GRID names → tzrec names

This table is the rosetta stone for the rest of the doc.

| GRID (`SemanticIDEncoderDecoder`) | tzrec `Batch` source | Shape | Type |
|---|---|---|---|
| `input_ids` (per-hierarchy SID tokens, after GRID's `map_sparse_id_to_semantic_id`) | `batch.sparse_features["history"]["history_sids"]` directly — already SID-encoded by the offline pipeline. KJT values are flat SID tokens; KJT lengths give the per-sample token count (always a multiple of `num_hierarchies`). Model densifies via `to_padded_dense` into `(B, L_tokens)`. | `(B, L_tokens)` | int64 |
| `attention_mask_encoder` (1=real, 0=pad) | Derived from KJT lengths via the standard `lengths_to_mask` trick → `(B, L_tokens)` **directly**, no per-item-to-token expansion step. | `(B, L_tokens)` | int64 / bool |
| `user_id` | `batch.sparse_features["user"]["user_id"]` (KJT with length 1 per sample) → squeeze to `(B,)` | `(B,)` | int64 |
| `future_ids` (target item's SID) — training only | `batch.labels["label_sids"]` reshaped to `(B, num_hierarchies)` — **no codebooks lookup needed**, label is already a SID tuple. | `(B, num_hierarchies)` | int64 |
| `attention_mask_decoder` | `None` (target SID is always full-length; mirrors GRID) | — | — |
| `codebooks` (per-item SID table for `_check_valid_prefix`) | *(deferred to v2 — see §0.1; v1 has no `codebooks` buffer)* | — | — |

**Key insight**: padding is represented by **KJT lengths** in tzrec, not by a sentinel value like GRID's `-1`. So the `attention_mask_encoder` is constructed from `lengths` rather than `(input_ids != -1)`. Functionally identical; the construction step differs.

**Second key insight (v1-specific)**: because history is pre-SID-encoded and labels carry SID tuples directly, the encoder pipeline becomes one step shorter than GRID's. Steps that disappear vs GRID: `history_item_id → SID` lookup (was step 2 of `encoder_forward_pass`) and `label_item_id → SID` lookup (was step 5 of `model_step`). They re-appear in v2 only if we want constrained beam search to consult a per-item table.

---

## 5. Model architecture (replaces GRID §3.x)

### 5.1 Class skeleton (no code, structural)

A new class `Tiger(BaseModel)` lives in `tzrec/models/tiger.py`. Constructor signature follows the `BaseModel` contract:

`__init__(model_config, features, labels, sample_weights, sampler_type, **kwargs)`

Inside `__init__`, the order of operations is:

1. Read `self._model_config = model_config.tiger` (the proto sub-message).
2. Stash hyperparameters from the proto:
   - Parse `codebook` (comma-separated string) into a list of ints → derive `num_hierarchies = len(codebook_sizes)` and per-hierarchy widths `codebook_sizes[h]`. The SID embedding table is sized to `sum(codebook_sizes)` (compact, no padding rows; see §5.3). Precompute `self.code_offsets = torch.tensor([0, *cumsum(codebook_sizes[:-1])])` for the per-hierarchy index shift. The `T5Config.vocab_size` placeholder is `1` (no-op; see §5.2).
   - Parse `hidden_dims` into a list of ints → derive `mlp_layers = len(hidden_dims_list)`. If `mlp_layers == 1`, leave stock `T5LayerFF`; if > 1, do the `T5MultiLayerFF` swap with widths from the list.
   - Read `embed_dim`, `num_heads`, `d_kv`, `num_encoder_layers`, `num_decoder_layers`, `dropout_rate`, `num_user_bins`, `beam_width`, `use_sep_token`.
3. Build the modules in §5.2–§5.7 below.

*(No codebooks buffer in v1 — deferred to v2 per §0.1. The model has no on-disk parquet to bootstrap from at `__init__`.)*

Note: TIGER's embedding tables stay as plain `nn.Embedding` instances (not TorchRec `EmbeddingCollection`s). At TIGER's scale (sub-1MB embedding tables) the TorchRec sharding/sparse-optimizer machinery offers no real benefit and adds friction — the offset shift (§3.6) would have to move into KJT manipulation before the lookup. Plain `nn.Embedding` keeps the lookup as one tensor op and the offset trick inside the model.

**No `sparse_parameters()` override needed.** `BaseModel.sparse_parameters()` (`tzrec/models/model.py:151`) only collects instances of `EmbeddingBagCollectionInterface` / `EmbeddingCollectionInterface`; plain `nn.Embedding` is automatically ignored, so the function returns two empty lists for TIGER without any model-side code. The TorchRec planner still runs but produces a trivial plan with no sharded modules; `create_train_pipeline` then falls back to `TrainPipelineBase` (see `TZREC_PIPELINE.md` §2.1), which is the right choice for TIGER.

#### DDP correctness with plain `nn.Embedding`

A natural concern: if we don't use TorchRec sparse for the SID embedding, does DDP still sync its gradients across ranks? **Yes — automatically.** The tzrec/TorchRec contract is "sharded modules go model-parallel, **everything else goes data-parallel via standard DDP**":

| Mechanism | Source |
|---|---|
| `DistributedModelParallel(module=model, plan=plan, ...)` (`tzrec/main.py:689`) wraps the entire non-sharded subtree in `torch.nn.parallel.DistributedDataParallel`. | TorchRec design — non-sharded modules participate in normal DDP. |
| `apply_optimizer_in_backward(sparse_optim_cls, sparse_parameters, ...)` (`tzrec/main.py:668`) is called with TIGER's empty `sparse_parameters()` list, so it's a no-op for us. | Our `nn.Embedding` falls through to the dense optimizer (`tzrec/main.py:708`). |
| `TrainPipelineBase` (selected by `create_train_pipeline` when no sharded modules are present, `tzrec/utils/dist_util.py:343`) runs forward/backward/all-reduce/optimizer-step in the standard eager order. | Slightly less throughput-overlap than `TrainPipelineSparseDist`, but TIGER is dense-compute-dominated so there's nothing meaningful to overlap with. |

Concretely, what happens each training step on `world_size = N` GPUs:

1. Each rank receives a different shard of the global batch (data parallelism).
2. Forward pass on each rank — independent copies of `self.sid_embedding`, `self.encoder_t5`, etc. all produce per-rank gradients.
3. DDP's backward hooks all-reduce every parameter's gradient across ranks (including `self.sid_embedding.weight`, `self.bos_token`, `self.sep_token`, every T5 weight).
4. Dense optimizer step on each rank applies the same averaged gradient → all ranks end the step with identical weights.

Memory footprint per rank: `self.sid_embedding` replicated → `sum(codebook_sizes) × embed_dim × 4 bytes` (fp32) ≈ 400 KB for the `"256,256,256,256"` default. Adam state adds 2×, so ~1.2 MB total per rank for this table. Negligible vs the ~50 MB total model footprint.

What `nn.Embedding` does **not** get from this setup (and doesn't need):
- **Sparse-gradient optimization in backward.** `apply_optimizer_in_backward` fuses the optimizer step with the backward pass for huge embedding tables where each batch only touches a tiny fraction of rows. TIGER's table is small enough that every batch touches a large fraction of rows; the gradient is effectively dense, and the fused-in-backward path wouldn't speed anything up.
- **Row-wise / column-wise / table-wise sharding.** Sharding a 100K-param table across multiple GPUs is overhead with no win — the all-to-all collectives would dwarf any compute saving.
- **TorchRec-specific embedding features** (`DynamicEmbedding` eviction, zero-collision hash, dynamic table resizing). None of these apply to a fixed-size SID codebook.

What `nn.Embedding` **does** get (because plain DDP):
- ✓ Gradient all-reduce across ranks → weights stay in sync.
- ✓ Mixed-precision (`bf16-mixed` / `fp16` via `TrainWrapper`'s `torch.amp.autocast`) — DDP wrap is autocast-aware.
- ✓ `GradScaler` integration via tzrec's `TZRecOptimizer` (when `train_config.grad_scaler` is configured).
- ✓ Standard PyTorch checkpoint/restore — no sharded-state-dict awkwardness.
- ✓ Clean TorchScript / `tzrec.export` path (`nn.Embedding` is fully scriptable; `EmbeddingCollection` requires more elaborate handling).

Empirical confirmation: `SidRqkmeans` (`tzrec/models/sid_rqkmeans.py`) uses plain `nn.Embedding`-style buffers for its centroid table and runs fine under DDP via the same code path. The non-sharded → DDP route is well-trodden in tzrec.

### 5.2 T5 encoder & decoder (mirrors GRID §3.1)

Build a `T5Config` from the proto-derived hyperparameters. Map: `d_model = embed_dim`, `num_heads`, `d_kv`, `d_ff = max(hidden_dims_list)`, `num_layers = num_encoder_layers` (resp. `num_decoder_layers`), `dropout_rate`. **`T5Config.vocab_size = 1`** — it's a true no-op (the only place T5 consumes it is `T5Stack.__init__` to size `embed_tokens`, which we immediately delete; T5 forward always takes `inputs_embeds=...` once `embed_tokens` is gone). Setting it to `1` avoids briefly allocating a discarded embedding tensor at construction.

- `self.encoder_t5 = T5EncoderModel(config=enc_cfg)`
- `self.decoder_t5 = T5Stack(dec_cfg, embed_tokens=<temporary>)` — the `T5Stack` constructor needs an `embed_tokens`; pass a temp `nn.Embedding(1, embed_dim)` we will discard immediately.

Then **delete the T5-internal embedding tables**:

- `delete_module(self.encoder_t5, "shared")`
- `delete_module(self.encoder_t5.encoder, "embed_tokens")`
- `delete_module(self.decoder_t5, "embed_tokens")`

And **re-initialize the rest** of the encoder/decoder weights from scratch (we do not load pretrained T5 weights — TIGER never has). `reset_parameters` recursion on the remaining submodules matches GRID's behavior.

### 5.3 SID embedding table

A single compact embedding:

```
self.sid_embedding = nn.Embedding(
    num_embeddings = sum(codebook_sizes),      # NOT max × num_hierarchies — no padding rows
    embedding_dim  = embed_dim,
)
```

The total row count is the **sum** of per-hierarchy code counts, not the padded `max × num_hierarchies` form GRID uses. Hierarchy `h`'s codes occupy a contiguous block:

```
hierarchy h owns rows [cumsum_h, cumsum_h + codebook_sizes[h])
where cumsum_h = sum(codebook_sizes[:h])
```

So with `codebook: "128,256,512"` the table has `128 + 256 + 512 = 896` rows (vs `3 × 512 = 1536` in the padded form — 42% smaller). For uniform codebooks like `"256,256,256,256"` both forms produce 1024 rows; the compact form is never larger.

The offset tensor `self.code_offsets = torch.tensor([0, *cumsum(codebook_sizes[:-1])])` (shape `(num_hierarchies,)` int64) is precomputed once in `__init__` and used by `_add_repeating_offset_to_rows` (§3.6 shows the formula).

**Used by both encoder and decoder.** The same `self.sid_embedding` is consulted in `encoder_forward_pass` (to embed the history SID stream) and in `decoder_forward_pass` (to embed the teacher-forced target SID + the BOS token). Encoder–decoder sharing is the only "tying" relationship in TIGER and it's hardcoded — no proto flag controls it. See `tiger_weight_tying_explanation.md` for why the GRID `weight_tying` flag was dropped.

### 5.4 Per-hierarchy decoder heads (mirrors GRID §3.5)

`self.decoder_mlp = nn.ModuleList([nn.Linear(embed_dim, codebook_sizes[h], bias=False) for h in range(num_hierarchies)])`. No biases (matches GRID). Each head outputs **its own per-hierarchy width** — for the uniform case all heads are `Linear(embed_dim, 256)`; for `codebook: "256,256,256,128"` the last head is `Linear(embed_dim, 128)`. Each head is used at exactly one decode step.

### 5.5 BOS / SEP parameters, optional user embedding (mirrors GRID §3.2, §3.6)

- `self.bos_token = nn.Parameter(torch.randn(1, embed_dim))` — decoder start token.
- `self.sep_token = nn.Parameter(torch.randn(1, embed_dim))` if `use_sep_token`, else `None`.
- `self.user_embedding = nn.Embedding(num_user_bins, embed_dim)` if `num_user_bins > 0`, else `None`.

### 5.6 The codebooks buffer — *deferred to v2*

In v1 the model has **no `codebooks` persistent buffer** and reads no on-disk SID table at `__init__`. The two GRID use cases that needed a per-item lookup are both eliminated:

- History → SID expansion: done offline by the upstream pipeline; v1 model reads `history_sids` directly from the batch.
- Label → target SID: same — `batch.labels["label_sids"]` already carries the SID tuple.

The third GRID use case (constrained beam search prefix check via `_check_valid_prefix`) is deferred to v2 along with the buffer that supports it. See §0.1 and the design exploration in:

- `ai_report/tiger_beam_search_pruning_comparison.md` — GRID full-prefix vs per-level vocab pruning vs no constraint.
- `ai_report/tiger_semantic_id_path_optimized_design.md` — scalable lex-sorted `(N, H)` tensor design that supports full-prefix pruning at arbitrary `(H, C)` scales with `O(K · log N)` per-beam-step compute.

**Forward-compat impact**: v1's absence of this buffer means there's no mode-detection plumbing needed in `BaseModel.__init__` (every mode goes through the same constructor); no `load_state_dict` override; no rank-zero parquet IO at startup. The v2 PR adds all of this together with the `semantic_id_path` proto field and the chosen pruning implementation.

### 5.7 `T5LayerFF → T5MultiLayerFF` swap (mirrors GRID §3.4)

Triggered only when `len(hidden_dims_list) > 1`. A helper method `_swap_t5_ff_with_multilayer(hidden_dims_list)` walks `self.named_modules()` after construction, locates every `T5LayerFF` instance, and replaces it with a `T5MultiLayerFF` whose MLP widths are exactly `hidden_dims_list`. The `T5MultiLayerFF` itself is reused from a small internal module file (`tzrec/modules/tiger_ff.py`, conceptual location) that mirrors GRID's `T5MultiLayerFF` (layernorm + dropout + `MLP[embed_dim → hidden_dims[0] → ... → hidden_dims[-1] → embed_dim]` + residual). When `len(hidden_dims_list) == 1`, no swap happens and the underlying T5 uses the stock `T5LayerFF` with `d_ff = hidden_dims_list[0]`.

---

## 6. Forward passes (replaces GRID §4.1, §4.2)

### 6.1 Entry point: `predict(self, batch)` — training & teacher-forced eval

`BaseModel.predict` is what `TrainWrapper.forward` calls (see `TZREC_PIPELINE.md` §1.5). For TIGER, `predict` does:

1. **Extract inputs from the batch.**
   - From `batch.sparse_features["history"]`: read `values` (flat SID-token stream — **already per-hierarchy codes**, no item-id lookup needed) and `lengths` (`(B,)`, tokens per sample, always a multiple of `num_hierarchies`).
   - Densify via `KeyedJaggedTensor.to_padded_dense` → `(B, L_tokens)` int64 tensor. Padding positions hold value `0` (default fill); per-hierarchy semantic correctness in those positions is guarded by the attention mask (§6.2 Job 1).
   - From `batch.sparse_features["user"]` (if configured): read user_id (single int per sample, squeeze to `(B,)`).
   - From `batch.labels["label_sids"]` (training/eval only — `None` in pure inference; see §7) → reshape to `(B, num_hierarchies)`. This is `future_ids` directly; no further conversion needed.

2. **Build the encoder attention mask.**
   - From KJT `lengths` → `attention_mask_encoder: (B, L_tokens)` via standard `lengths_to_mask`. No per-item-to-token expansion step (the lengths are already in tokens).
   - This is the GRID `attention_mask_encoder` tensor that does **three jobs** (preserved verbatim from §2.4):

#### 6.2 The three jobs of `attention_mask_encoder` (preserves §2.4)

| Job | When | Implementation note |
|---|---|---|
| 1 — guard SID embedding lookup | Inside `_add_repeating_offset_to_rows` | Compute `lookup_idx = mask * (c + self.code_offsets[h])` (broadcast over positions). For real positions (mask=1): yields `c + cumsum_h ∈ [cumsum_h, cumsum_h + codebook_sizes[h])` — the correct row in the compact `sum(codebook_sizes)` embedding table. For padding positions (mask=0): yields `0` — a deterministic in-range row (the first code at hierarchy 0), safe because attention blocks these positions from contributing to softmax (Job 3). The mask multiply collapses padding-vs-real ambiguity into a single safe row regardless of input value. |
| 2 — extend through SEP injection | Inside `_inject_sep_token_between_sids` | Identical logic to GRID. The extended mask is what's returned out of `encoder_forward_pass`. |
| 3 — feed T5 self-attention | At the final `self.encoder_t5(...)` call | T5 turns 0s into additive `-inf` bias on attention logits. Same as GRID. |

3. **Run `encoder_forward_pass`** (logic verbatim from GRID `:584`): apply offset trick → SID embedding lookup → optional SEP injection → optional user-id prepend → T5 encoder. Returns `(encoder_output, attention_mask_for_encoder)`.

4. **Run `decoder_forward_pass`** with `future_ids` (from step 1 — already a SID tuple from `batch.labels["label_sids"]`), `encoder_output`, and the extended encoder mask. Returns the decoder hidden states `(B, num_hierarchies + 1, embed_dim)` (the `+1` is the BOS prepend) — drop the last position to align with the target.

5. **Return a `predictions` dict** with the fields downstream code needs:
   - `decoder_hidden`: the `(B, num_hierarchies, embed_dim)` hidden states (post-BOS-strip). The `loss()` method reads this and applies the per-hierarchy heads.
   - `encoder_output` and `attention_mask_for_encoder`: optional, only included when the engine is going to also call `generate()` (e.g. eval), so we don't recompute. See §7.2.
   - (At inference time, an additional `generated_ids` and `marginal_probs` — see §7.)

### 6.3 Loss (`loss(self, predictions, batch)`)

Implements the GRID §4.2 training objective:

- Let `target_sid = batch.labels["label_sids"].view(B, num_hierarchies).long()`. This is **0-indexed** by the §3.6 convention; the per-hierarchy heads produce 0-indexed logits in the same range. No conversion needed.
- For `h in range(num_hierarchies)`:
  - `logits_h = self.decoder_mlp[h](predictions["decoder_hidden"][:, h, :])` → `(B, codebook_sizes[h])`, indexed `[0, codebook_sizes[h])`.
  - `loss_h = F.cross_entropy(logits_h, target_sid[:, h])`.
- Return `{"tiger_ce": sum(loss_h for h in range(num_hierarchies))}`.

The returned dict key needs to **match exactly one entry** the proto-level `LossConfig` (or our own convention) will be aware of for metric init. Convention recommendation: pick a single string like `"tiger_ce"` so `init_loss/init_metric` can register a `MeanMetric("tiger_ce")`.

`init_loss(self)` — for TIGER we don't actually have any loss modules to register (the criterion is pure functional), so this can be a no-op or it can register a placeholder `nn.Identity` to satisfy any framework introspection. The TrainWrapper calls `init_loss` before training begins regardless.

---

## 7. Inference / beam search (replaces GRID §4.3)

### 7.1 Where beam search runs

Beam search runs in two situations:

1. **During eval** (inside `update_metric`) — we need predictions to score retrieval metrics. `update_metric` is called by the engine inside `_evaluate` (`main.py:_evaluate`) under `torch.no_grad()`. Implementation: call a `self.generate(batch)` method which mirrors GRID's `generate` 1:1 but reads its inputs from the `Batch` (same extraction logic as `predict()` §6.1 steps 1–3).
2. **During predict** (when the engine is `predict` or `predict_checkpoint`) — `predict_checkpoint` will work without further hassle because it runs eagerly via `PredictWrapper.forward → BaseModel.predict`; we just need `predict()` to behave differently when `batch.labels` is empty (i.e. inference): call `self.generate(batch)` instead of the teacher-forced decoder, and put the results into the predictions dict.

### 7.2 `generate(self, batch)` — port of GRID `:738`

Pseudocode-level mapping (no code):

1. Extract sequence + user_id from the batch exactly as §6.1.1.
2. Run `encoder_forward_pass` once → `(encoder_output, attention_mask_for_encoder)`.
3. Initialize `generated_sids=None`, `marginal_log_prob=None`, `step_scores=None` (a `(B, K, num_hierarchies)` accumulator), `past_key_values = EncoderDecoderCache(self_attention_cache=DynamicCache(), cross_attention_cache=DynamicCache())`.
4. For `h in range(num_hierarchies)`:
   - If `h > 0`: reshape and repeat-interleave the encoder outputs / mask to match the current beam width (`beam_width`).
   - Call `decoder_forward_pass` with `use_cache=True` and the accumulating `past_key_values`.
   - Read the last position's hidden state → apply `self.decoder_mlp[h]` → logits `(B*K, codebook_sizes[h])`, naturally 0-indexed.
   - Call `_beam_search_one_step` (logic adapted from GRID `:253`, **with the `_check_valid_prefix` masking branch removed in v1**): softmax → per-step probabilities; sort/top-K; beam expansion at step 0; KV-cache reorder via `past_key_values.reorder_cache(replace_indices)` at later steps; update `generated_sids` (still 0-indexed — matches on-disk convention) and `marginal_log_prob`. **Also record the per-step softmax probability of the selected code at each beam** into `step_scores[:, :, h]`.
5. **No restoration step needed** — `generated_sids` is already 0-indexed throughout, matching the on-disk input convention (§3.6). The model's argmax output and the input data live in the same indexing space end-to-end.
6. Return:

   | Output key | Shape | Dtype | Meaning |
   |---|---|---|---|
   | `generated_sids` | `(B, K, num_hierarchies)` | int64 | Top-K beams as 0-indexed SID tuples (same convention as `label_sids` and `history_sids`), **sorted descending by `marginal_log_prob`**. |
   | `step_scores` | `(B, K, num_hierarchies)` | float32 | Per-hierarchy softmax probability of the selected code at each step, in the same beam order as `generated_sids`. Lets downstream / diagnostics inspect where beams diverge from ground truth. |

The emitted `generated_sids` is **the model's final output** in predict mode (returned from `Tiger.predict()` when called in inference mode; passed through `predict_step` to the writer). Downstream offline retrieval (out of scope) joins each SID tuple against the catalog to map back to item IDs; SIDs that don't correspond to any item are simply skipped by the join, naturally filtering out the hallucinated beams that unconstrained search may produce.

### 7.3 Constrained prefix check — *deferred to v2*

GRID's `_check_valid_prefix` (and its tzrec analogue using a per-item or per-prefix table) is deferred to v2 — see §0.1. v1 beam search emits any SID tuple the decoder predicts; the offline retrieval step is responsible for filtering. See `ai_report/tiger_beam_search_pruning_comparison.md` for the v2 design space.

### 7.4 JIT-scriptability caveat for `tzrec.predict` (the export path)

The fast `tzrec.predict` path (not `predict_checkpoint`) loads a **scripted** model from `tzrec.export`. Beam search code with:

- Python `for` loops over hierarchies,
- dynamic KV-cache reordering with `reorder_cache(...)`,
- `EncoderDecoderCache` objects from HuggingFace,

is **not straightforwardly TorchScript-able**. There are three escape hatches:

1. **Recommended (first milestone)**: export only the encoder. Ship a scripted artifact that produces `(encoder_output, attention_mask_for_encoder)` from a batch dict; do beam search in Python on the predict side using the same `generate` code-path. Mirrors how `MatchModel` exports per-tower in tzrec (see `TZREC_PIPELINE.md` §6). Concretely: special-case TIGER in the `export()` orchestration analogous to the `MatchModel` branch, exporting a `TigerEncoderWrapper` instead of the full model.
2. Use `predict_checkpoint` instead of `predict` — the raw-checkpoint path uses `PredictWrapper` (no JIT), so the full beam search runs unchanged. Slower than the scripted path but simplest.
3. (Long-term) Manually script-friendly the beam search (drop `EncoderDecoderCache` for two parallel `DynamicCache`s, inline the loop bodies, etc.). Defer.

The proto field `is_scripted_inference` (boolean, default `false`) could gate this in a follow-up; not needed for the first milestone.

---

## 8. Eval-time scoring (replaces GRID §4.4)

### 8.1 Metric registration: `init_metric(self)`

Read the proto's eval block (or hard-code, since TIGER's metric set is fixed) and register one torchmetrics module per `(metric_name, top_k)` pair, mirroring GRID's `SIDRetrievalEvaluator`:

- For each `k in [5, 10]` (configurable): register `RetrievalNormalizedDCG(top_k=k)` and `RetrievalRecall(top_k=k)` under `self._metric_modules` with names like `"recall@5"`, `"ndcg@5"`, `"recall@10"`, `"ndcg@10"`. These are stock torchmetrics retrieval modules (already a dependency).
- Also register `self._metric_modules["tiger_ce"] = MeanMetric()` so the engine's average-loss logging works.

### 8.2 Updating metrics: `update_metric(self, predictions, batch, losses=None)`

The engine's `_evaluate` calls this for every batch under `torch.no_grad()`. Implementation:

1. If `losses` is provided (engine passes them), `self._metric_modules["tiger_ce"].update(losses["tiger_ce"], batch_size)`.
2. Call `predictions = self.generate(batch)` → returns the same dict described in §7.2 step 6, with `generated_sids` **0-indexed** (matching the on-disk convention).
3. Read `target_sids = batch.labels["label_sids"].view(B, num_hierarchies)` — also 0-indexed (§3.6 convention). Both sides are in the same indexing space; no conversion needed.
4. Compute the match map: `match = (generated_sids == target_sids.unsqueeze(1)).all(dim=2)` → `(B, K)` bool — does any of the K beams equal the ground-truth SID?
5. Compute the per-beam marginal score by reducing the per-hierarchy step scores: `marginal_probs = step_scores.prod(dim=-1)` → `(B, K)` (the standard product-of-step-probabilities under the chain rule). Beams are already sorted descending by this value coming out of `generate()`.
6. Build the `(preds, target, indexes)` triple that torchmetrics retrieval modules require (identical reshape to GRID `eval_metrics.py:281-313`):
   - `preds = marginal_probs.reshape(-1)`,
   - `target = match.reshape(-1)`,
   - `indexes = repeat([0..B-1], K).reshape(-1)`.
7. Forward this triple into every registered NDCG/Recall metric via `metric.update(preds, target, indexes=indexes)`.

### 8.3 Computing metrics: `compute_metric(self)`

Inherited from `BaseModel.compute_metric` (`tzrec/models/model.py:123`) — walks `self._metric_modules`, calls `.compute()` on each, resets, returns the dict. Engine writes the dict to `train_eval_result.txt` / `eval_result.txt` as one JSON line per evaluation.

### 8.4 Train-time monitoring metric

For the periodic train-step logging (`_log_train`), tzrec exposes `update_train_metric` / `compute_train_metric`. We can register a single lightweight metric here (e.g. step-level CE mean) so tensorboard gets a real-time scalar without running beam search every step. The full retrieval metrics only run at eval-checkpoint cadence.

---

## 9. Train / eval / predict lifecycle integration

Putting §5 – §8 in context of `ai_report/TZREC_PIPELINE.md`:

| Engine step | Where TIGER plugs in |
|---|---|
| `_create_features` | Reads the `feature_configs` declared in §3.2; produces a list with one `SequenceFeature` (history) and zero/one `IdFeature` (user). |
| `create_dataloader` | Loads pre-expanded parquet (§3.4 option 1); no model-specific code. |
| `_create_model` | Picks our `Tiger` class via `BaseModel.create_class("Tiger")` from the proto oneof. |
| `TrainWrapper(model, mixed_precision=...)` | Wraps as usual. **No autocast surprises** expected with `bf16-mixed`; if numerical issues arise in the constrained-beam-search softmax (the GRID warning at §7.4 of `TIGER_implementation.md`), recommend `precision: "fp32"` for the first runs. |
| TorchRec planner | Sees zero `ShardedModule`s (per §5.1) → produces a trivial plan. `DistributedModelParallel` still wraps the model — that's fine. |
| Optimizer assembly | All parameters go to the **dense optimizer** (sparse-parameters override returns empty). Standard Adam-with-weight-decay matches GRID. |
| `create_train_pipeline` | Falls back to `TrainPipelineBase` (no sparse modules). Throughput is bounded by dense compute; pipeline overlap less critical. |
| `_train_and_evaluate` step loop | `pipeline.progress(iterator) → TrainWrapper.forward → Tiger.predict + Tiger.loss → backward`. Identical pattern to every other tzrec model. |
| `_evaluate` step loop | `pipeline.progress(iterator) → TrainWrapper.forward` (still computes loss), then `model.update_metric(predictions, batch, losses)` which runs `generate` internally. |
| `predict_checkpoint` | Loads the checkpoint, calls `PredictWrapper.forward → Tiger.predict` which detects empty labels and routes through `generate`. Writes `(user_id, generated_sids)` rows. |
| `predict` (exported) | Per §7.4: ship the encoder-only artifact in the first milestone; the writer side does Python beam search. |
| `export` | Same caveat: special-case TIGER in `export()` to dump a `TigerEncoderWrapper` (encoder + offset trick + SEP injection only); regular path for the full model is left as future work. |
| Checkpointing | No special handling in v1 (no extra persistent buffers beyond the trainable parameters). Resume semantics from `TZREC_PIPELINE.md` §7.1 apply unchanged. v2 will add the codebooks buffer to the state dict; see §0.1. |
| `on_train_end` | No-op for TIGER (unlike `SidRqkmeans` which uses it for FAISS fit). |

---

## 10. Open decisions to resolve before implementation

These are the things the design intentionally leaves to a follow-up review:

1. **User-embedding source** (§3.2, §5.5): keep TIGER's own remainder-hashed `nn.Embedding` (matches GRID) vs route through tzrec's `EmbeddingGroup` (more idiomatic, but loses the remainder-hashing semantics). Recommendation: keep TIGER's for parity; revisit when user-side features grow beyond `user_id`.
2. **Sharded vs dense SID table** (§5.1, §5.3): `self.sid_embedding` is `sum(codebook_sizes) × embed_dim` ≈ 100K params for typical configs. Plain `nn.Embedding` is correct; no sharding needed. Document why so a future contributor doesn't "fix" this by switching to TorchRec `EmbeddingCollection`.
3. **JIT-scripting beam search** (§7.4): encoder-only export for v1 vs scripting the full beam search. Recommendation: encoder-only.
4. **Loss-key name** (§6.3): `"tiger_ce"` vs `"cross_entropy"` vs `"summed_cross_entropy_per_hierarchy"`. The exact string only affects tensorboard scalar names and the metric-key in `_metric_modules`. Recommendation: `"tiger_ce"`.
(Resolved upstream: contiguous-subsequence augmentation — see §3.4. `weight_tying` flag — removed as a no-op, see `tiger_weight_tying_explanation.md`. `prediction_key_name` / `prediction_value_name` fields — removed in favor of hardcoded output column names plus `--reserved_columns`, see §2.1 "Output naming." Deferred to v2: codebook storage, constrained-beam-search default, mode-detection mechanism — see §0.1.)

---

## 11. File / proto delivery map

What gets created or touched in the migration PR:

| Path | Status | Purpose |
|---|---|---|
| `tzrec/protos/models/tiger.proto` | **new** | Message `Tiger`, all model knobs. |
| `tzrec/protos/model.proto` | edited | Add `Tiger tiger = N;` to the `model` oneof and import. |
| `tzrec/protos/{model,tiger}_pb2.py` & `*_pb2.pyi` | regenerated | Run `bash scripts/gen_proto.sh`. Commit the generated files. |
| `tzrec/models/tiger.py` | **new** | `Tiger(BaseModel)` — sections 5, 6, 7, 8 of this design. |
| `tzrec/models/tiger_test.py` | **new** | Unit tests; use a mock config like `tzrec/tests/configs/tiger_mock.config`. |
| `tzrec/modules/tiger_ff.py` | **new** | `T5MultiLayerFF` ported from `GRID/src/models/modules/semantic_id/tiger_generation_model.py:1080`. |
| `tzrec/models/__init__.py` | edited | Import `Tiger` so the registry sees it. |
| `tzrec/tests/configs/tiger_mock.config` | **new** | Small config for unit tests; mirrors the layout of `dssm_mock.config`. |
| `tzrec/main.py` | edited (only if §7.4 option 1 is taken) | Add a TIGER branch to `export()` that emits an encoder-only artifact. |
| `docs/source/models/tiger.md` | **new** | User-facing model doc; can be a slim version of this design. |
| `examples/tiger_amazon.config` | **new** | End-to-end example config users can copy. |

---

## 12. Acceptance criteria (what "done" looks like)

- `bash scripts/gen_proto.sh` runs clean; new `Tiger` message is reachable from `ModelConfig.tiger`.
- `python -m unittest tzrec.models.tiger_test` passes with a mock config and synthetic SID-encoded data — v1 has no `semantic_id_path` and no codebooks buffer, so tests need only synthesize `history_sids` + `label_sids` parquet rows.
- `torchrun -m tzrec.train_eval --pipeline_config_path examples/tiger_amazon.config --train_input_path .../user_sequences/*.parquet` runs at least one full epoch with a real Amazon Beauty stage-3 SID output (one row per user, no pre-expansion).
- `python -m tzrec.predict_checkpoint --pipeline_config_path examples/tiger_amazon.config --predict_input_path .../predict_in.parquet --predict_output_path .../predict_out.parquet` writes a parquet with `(user_id, semantic_ids)` columns.
- For-eval `recall@5 / recall@10` metrics match the corresponding GRID checkpoint within a small tolerance, when both are run on the same Amazon Beauty data after equal training compute.
- Encoder-only `tzrec.export → tzrec.predict` path produces encoder hidden states matching `tzrec.predict_checkpoint` within numerical precision (covers §7.4 option 1).

This becomes the PR description checklist.
