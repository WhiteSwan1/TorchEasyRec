# What does `weight_tying` actually do in TIGER?

Short answer: in v1 (and in GRID), **`weight_tying` is a vestigial flag — it has no observable effect on the model.** It's a proto field we accept and ignore. The "tying" that actually exists in TIGER is hardcoded regardless of the flag value.

## What weight tying normally means

In a standard encoder–decoder Transformer (e.g. T5), `weight_tying=True` means the **input embedding table is reused as the output projection**:

```
shared.weight  ←————┐
                    ├── encoder.embed_tokens.weight
                    ├── decoder.embed_tokens.weight
                    └── lm_head.weight  (transposed)
```

One `(vocab_size, d_model)` tensor, three roles. The benefit is well-known (Press & Wolf, 2017): smaller parameter count and a strong inductive bias that input and output token spaces coincide. T5-base saves ~25 M params this way (its `lm_head` ties to the 32128×768 `shared` embedding — see `TIGER_vs_T5base_params.md`).

## What `weight_tying` does in TIGER

**Nothing observable.** GRID's `SemanticIDEncoderDecoder` hardcodes two relationships regardless of the flag:

1. **Encoder and decoder share the SID embedding table.** See `tiger_generation_model.py:863-884` — `get_embedding_table("encoder")` and `get_embedding_table("decoder")` both return `self.item_sid_embedding_table_encoder`. The encoder reads from it during `encoder_forward_pass`; the decoder reads from it during `decoder_forward_pass`. Always.
2. **Output heads are NOT tied to the embedding table.** `decoder_mlp[h]` is its own `Linear(embed_dim, codebook_sizes[h], bias=False)` per hierarchy — distinct trainable parameters, never tied to input embedding rows.

So the only "tying" that exists in TIGER is point 1, and it happens whether `weight_tying=True` or `weight_tying=False`. The flag is read by the parent class `TransformerBaseModule` (`get_embedding_table` at `transformer_base_module.py:66-70`), but that code path is for a generic embedding-vs-decoder retrieval scoring — and `SemanticIDEncoderDecoder` overrides `model_step` so the parent's logic that uses the flag is **never reached**.

GRID's own config sets `weight_tying: true` and the model trains fine; flipping it to `false` produces an identical trained model (the embedding-table sharing in point 1 is still there). It is effectively dead config.

## Why we keep it in v1's proto

Two reasons, both documented at §2.1 row #21 and §5.3 of `tiger_migration_design.md`:

1. **Proto-level compatibility.** If a user copies a GRID YAML wholesale into their tzrec pipeline config (with the codebook fields renamed per §2.1), having `weight_tying: true` survive the proto parser without error keeps the migration friction low. Removing the field would force users to grep-and-delete one extra line.
2. **Forward compatibility.** If a future version decides the flag should mean something specific (see next section), we already have the field plumbed.

## What it could mean if we activated it (deferred decision)

§10 decision 5 of the design doc leaves this as an open question. The most natural semantics would be:

> When `weight_tying=True`, tie each `decoder_mlp[h].weight` to a slice of `item_sid_embedding_table_encoder` — specifically, the slice covering rows `[h × max(codebook_sizes), h × max(codebook_sizes) + codebook_sizes[h])`.

That would mirror T5's `shared ↔ lm_head` recipe within each hierarchy: the input embedding for code `c` at hierarchy `h` is reused as the row of the `Linear` weight that produces the logit for code `c` at hierarchy `h`. Conceptually appealing — it forces "the score of predicting code c at level h depends on the input embedding of c at level h."

Parameter savings would be `num_hierarchies × codebook_width × embed_dim`. With the GRID default (4 × 256 × 128 = 131 K params), that's about 1% of the model's ~13 M params — not a meaningful compression. The inductive bias might help recall@K, but that's an empirical question.

**Recommendation in §10 decision 5**: keep the GRID-style ambiguous behavior in v1 (flag accepted, ignored) and revisit only if a future ablation shows the actual T5-style tying improves metrics. Document loudly in the proto comment so users don't think the flag is doing something.

## TL;DR

| Aspect | Value |
|---|---|
| Does the flag change model behavior in v1? | **No** |
| Does the flag change model behavior in GRID? | **No** (despite the GRID YAML defaulting it to `true`) |
| What tying *is* present regardless of the flag? | Encoder and decoder share `item_sid_embedding_table_encoder` — hardcoded in `get_embedding_table` |
| Why is it still in the proto? | Backward compat with GRID configs + forward compat if we decide to give it semantics |
| Could it be removed? | Yes, safely — but the cost of keeping it is one ignored field, and removing it makes copy-pasting GRID configs slightly harder |
| Param savings if v2 activates T5-style tying inside the decoder heads | ~131 K (≈ 1% of total) for the GRID default; not significant |

## Recommendation

Keep the field, accept it silently in v1, document its no-op status in the proto comment. If a future ablation experiment shows tying `decoder_mlp[h]` to the SID embedding slice helps metrics, activate it in v2 with the same field — no proto migration needed.
