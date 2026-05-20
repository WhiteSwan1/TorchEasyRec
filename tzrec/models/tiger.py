# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Tiger: generative recommendation with semantic IDs (port of GRID).

See ai_report/tiger_migration_design.md for the full design rationale. This
file is the v1 implementation: pre-SID-encoded inputs, unconstrained beam
search, no codebooks buffer / constrained-prefix pruning (deferred to v2).

Architectural summary:
  - T5 encoder + T5 decoder (HF), with their internal `embed_tokens` deleted
    and weights re-initialized from scratch (no pretrained T5 load).
  - One compact `nn.Embedding(sum(codebook_sizes), embed_dim)` shared by
    encoder + decoder; per-hierarchy offset shift maps codes into the right
    block of rows.
  - Per-hierarchy `Linear(embed_dim, codebook_sizes[h], bias=False)` heads.
  - Learnable BOS (decoder start) and SEP (encoder inter-item separator).
  - Optional remainder-hashed user embedding prepended to encoder input.
  - When `hidden_dims` has length > 1, every `T5LayerFF` is replaced by
    a `T5MultiLayerFF` (see `tzrec/modules/tiger_ff.py`).
"""

from typing import Any, Dict, List, Optional, Tuple

import torch
import torchmetrics
from torch import nn
from torch.nn import functional as F
from transformers import T5EncoderModel
from transformers.models.t5.configuration_t5 import T5Config
from transformers.models.t5.modeling_t5 import T5LayerFF, T5Stack

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models._sid_helpers import parse_int_list
from tzrec.models.model import BaseModel
from tzrec.modules.tiger_ff import T5MultiLayerFF
from tzrec.protos.model_pb2 import ModelConfig

# Keys used in predictions/losses dicts. Kept as module-level constants so
# users (downstream writer, metric registration, etc.) can reference them
# without worrying about typos.
DECODER_HIDDEN_KEY = "decoder_hidden"  # only present in training-mode predict()
GENERATED_SIDS_KEY = "generated_sids"  # only present in inference-mode predict()
STEP_SCORES_KEY = "step_scores"
LOSS_KEY = "tiger_ce"

# Output column names emitted by predict-mode for the writer.
PREDICT_OUTPUT_COLUMNS = (GENERATED_SIDS_KEY, STEP_SCORES_KEY)


# ----------------------------------------------------------------------------
# Helpers
# ----------------------------------------------------------------------------


def _get_parent_and_attr(root: nn.Module, qualified_name: str) -> Tuple[nn.Module, str]:
    """Look up the parent module and final attribute name for a dotted path."""
    parts = qualified_name.split(".")
    parent = root
    for p in parts[:-1]:
        parent = getattr(parent, p)
    return parent, parts[-1]


def _swap_t5_ff_with_multilayer(
    root: nn.Module, t5_cfg: T5Config, hidden_dims: List[int]
) -> None:
    """Replace every `T5LayerFF` inside `root` with a `T5MultiLayerFF`."""
    targets = [n for n, m in root.named_modules() if isinstance(m, T5LayerFF)]
    for qn in targets:
        parent, attr = _get_parent_and_attr(root, qn)
        setattr(parent, attr, T5MultiLayerFF(config=t5_cfg, hidden_dims=hidden_dims))


def _delete_if_present(module: nn.Module, attr_name: str) -> None:
    if hasattr(module, attr_name):
        delattr(module, attr_name)


# ----------------------------------------------------------------------------
# Main model
# ----------------------------------------------------------------------------


class Tiger(BaseModel):
    """Tiger: generative recommendation model decoding next-item SID tuples.

    Inherits the tzrec model contract:
      - predict(batch) -> dict (training mode returns decoder hidden states;
        inference mode returns generated SIDs + per-step scores).
      - loss(predictions, batch) -> {"tiger_ce": Tensor}.
      - init_loss() / init_metric() / update_metric() / compute_metric().

    Args:
        model_config: ModelConfig containing a `tiger` sub-message.
        features: list of BaseFeatures (consumed only for compatibility with
            BaseModel; this model reads from the batch directly).
        labels: list of label names; for TIGER, exactly ["label_sids"].
        sample_weights: unused.
    """

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        cfg = self._model_config  # Tiger proto sub-message

        # --- Parse proto fields ---
        self._codebook_sizes: List[int] = parse_int_list(cfg.codebook)
        assert len(self._codebook_sizes) >= 1, "codebook must list at least one size"
        self._num_hierarchies: int = len(self._codebook_sizes)
        self._embed_dim: int = cfg.embed_dim
        self._beam_width: int = cfg.beam_width
        self._use_sep_token: bool = cfg.use_sep_token
        self._num_user_bins: int = cfg.num_user_bins
        self._history_label_name: str = labels[0] if labels else "label_sids"

        hidden_dims = parse_int_list(cfg.hidden_dims)
        assert len(hidden_dims) >= 1, "hidden_dims must list at least one size"

        # Precomputed offsets for the per-hierarchy index shift (§3.6 of
        # design): lookup_idx[h] = c[h] + cumsum(codebook_sizes[:h]).
        offsets = [0]
        for s in self._codebook_sizes[:-1]:
            offsets.append(offsets[-1] + s)
        self.register_buffer(
            "code_offsets",
            torch.tensor(offsets, dtype=torch.long),
            persistent=False,
        )

        # --- Build T5 encoder and decoder ---
        # vocab_size=1 is a true no-op: T5Stack uses it once at __init__ to
        # size `embed_tokens`, which we delete immediately.
        enc_cfg = T5Config(
            vocab_size=1,
            d_model=cfg.embed_dim,
            d_kv=cfg.d_kv,
            d_ff=max(hidden_dims),
            num_layers=cfg.num_encoder_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            tie_word_embeddings=False,
        )
        dec_cfg = T5Config(
            vocab_size=1,
            d_model=cfg.embed_dim,
            d_kv=cfg.d_kv,
            d_ff=max(hidden_dims),
            num_layers=cfg.num_decoder_layers,
            num_heads=cfg.num_heads,
            dropout_rate=cfg.dropout_rate,
            is_decoder=True,
            is_encoder_decoder=False,
            tie_word_embeddings=False,
        )

        self.encoder_t5 = T5EncoderModel(config=enc_cfg)
        # T5Stack needs an embed_tokens at construction time; we discard.
        _tmp_embed = nn.Embedding(1, cfg.embed_dim)
        self.decoder_t5 = T5Stack(dec_cfg, embed_tokens=_tmp_embed)

        # Delete the T5 internal embedding tables — we feed `inputs_embeds`
        # exclusively.
        _delete_if_present(self.encoder_t5, "shared")
        _delete_if_present(self.encoder_t5.encoder, "embed_tokens")
        _delete_if_present(self.decoder_t5, "embed_tokens")

        # Multi-layer FF swap (only when length > 1).
        if len(hidden_dims) > 1:
            _swap_t5_ff_with_multilayer(self.encoder_t5, enc_cfg, hidden_dims)
            _swap_t5_ff_with_multilayer(self.decoder_t5, dec_cfg, hidden_dims)

        # --- Compact SID embedding table (§5.3 of design) ---
        # Single shared table of shape (sum(codebook_sizes), embed_dim).
        self.sid_embedding = nn.Embedding(
            num_embeddings=sum(self._codebook_sizes),
            embedding_dim=cfg.embed_dim,
        )

        # --- BOS, SEP, optional user embedding (§5.5 of design) ---
        self.bos_token = nn.Parameter(torch.randn(1, cfg.embed_dim))
        if self._use_sep_token:
            self.sep_token = nn.Parameter(torch.randn(1, cfg.embed_dim))
        else:
            self.register_parameter("sep_token", None)
        if self._num_user_bins > 0:
            self.user_embedding = nn.Embedding(self._num_user_bins, cfg.embed_dim)
        else:
            self.user_embedding = None

        # --- Per-hierarchy decoder heads (§5.4 of design) ---
        self.decoder_mlp = nn.ModuleList(
            [
                nn.Linear(cfg.embed_dim, self._codebook_sizes[h], bias=False)
                for h in range(self._num_hierarchies)
            ]
        )

    # ------------------------------------------------------------------
    # Batch extraction
    # ------------------------------------------------------------------

    def _extract_history_sids_and_mask(
        self, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Densify the history_sids KJT into a (B, L_tokens) tensor + mask.

        Returns:
            history_sids: (B, L_tokens) int64. L_tokens is the max
                sequence length in this batch, rounded to a multiple of
                num_hierarchies. Real positions hold 0-indexed codes;
                padding positions hold 0 (a value that may collide with
                real code-0-at-hierarchy-0; the mask blocks attention).
            attention_mask: (B, L_tokens) int64 (1=real, 0=pad).
        """
        # First sparse_features value is the history group's KJT.
        history_kjt = next(iter(batch.sparse_features.values()))
        feat_dict = history_kjt.to_dict()
        # In the conventional setup the history group has exactly one key.
        jt = next(iter(feat_dict.values()))
        values = jt.values()  # (sum_lengths,) int64
        lengths = jt.lengths()  # (B,) int64

        # Pad to (B, L_tokens) where L_tokens = max(lengths). Use 0 fill —
        # padding correctness is enforced by the attention mask (§6.2 Job 1).
        B = lengths.size(0)
        L = int(lengths.max().item()) if B > 0 else 0
        device = values.device

        # Round L up to a multiple of num_hierarchies so the encoder
        # reshape to (B, items, num_hierarchies) is always well-defined.
        if L % self._num_hierarchies != 0:
            L = ((L // self._num_hierarchies) + 1) * self._num_hierarchies
        if L == 0:
            history_sids = torch.zeros((B, 0), dtype=torch.long, device=device)
            attention_mask = torch.zeros((B, 0), dtype=torch.long, device=device)
            return history_sids, attention_mask

        history_sids = torch.zeros((B, L), dtype=torch.long, device=device)
        attention_mask = torch.zeros((B, L), dtype=torch.long, device=device)
        # Scatter the flat values into the padded tensor row by row.
        # Use cumulative-sum offsets to avoid a Python loop on the hot path.
        offsets = torch.zeros(B + 1, dtype=torch.long, device=device)
        offsets[1:] = torch.cumsum(lengths, dim=0)
        for b in range(B):
            n = int(lengths[b].item())
            if n == 0:
                continue
            history_sids[b, :n] = values[offsets[b] : offsets[b + 1]]
            attention_mask[b, :n] = 1
        return history_sids, attention_mask

    def _extract_user_id(self, batch: Batch) -> Optional[torch.Tensor]:
        """Read user_id from batch.sparse_features['user'] if present."""
        if self.user_embedding is None:
            return None
        if "user" not in batch.sparse_features:
            return None
        user_kjt = batch.sparse_features["user"]
        feat_dict = user_kjt.to_dict()
        if not feat_dict:
            return None
        jt = next(iter(feat_dict.values()))
        # Length 1 per sample by convention; squeeze.
        return jt.values().long()

    def _extract_target_sids(self, batch: Batch) -> Optional[torch.Tensor]:
        """Read label_sids from batch.jagged_labels and reshape to (B, H)."""
        if self._history_label_name not in batch.jagged_labels:
            return None
        jt = batch.jagged_labels[self._history_label_name]
        values = jt.values()
        return values.view(-1, self._num_hierarchies).long()

    # ------------------------------------------------------------------
    # Offset trick and SEP injection (§6.2)
    # ------------------------------------------------------------------

    def _add_repeating_offset(
        self, sids: torch.Tensor, attention_mask: torch.Tensor
    ) -> torch.Tensor:
        """Apply per-hierarchy offset (§3.6), masked.

        For real positions: returns `code + cumsum_h`.
        For padding positions (mask=0): returns `0` (the first row).
        """
        if sids.numel() == 0:
            return sids
        B, L = sids.shape
        # offset pattern of length L: repeat code_offsets ⌈L/H⌉ times.
        offsets = self.code_offsets
        num_repeats = (L + self._num_hierarchies - 1) // self._num_hierarchies
        repeated = offsets.repeat(num_repeats)[:L]
        shifted = sids + repeated.unsqueeze(0)
        return shifted * attention_mask

    def _inject_sep_token(
        self,
        sid_embeds: torch.Tensor,
        attention_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Insert a learnable SEP after every item's SID block.

        Args:
            sid_embeds: (B, L_tokens, d_model). L_tokens = items * H.
            attention_mask: (B, L_tokens).

        Returns:
            (sid_embeds_with_sep, mask_with_sep), both extended to
            (B, items * (H + 1), ...).
        """
        if self.sep_token is None:
            return sid_embeds, attention_mask
        B, L, D = sid_embeds.shape
        H = self._num_hierarchies
        if L == 0:
            return sid_embeds, attention_mask
        items = L // H
        sid_grouped = sid_embeds.view(B, items, H, D)
        mask_grouped = attention_mask.view(B, items, H)
        sep_block = (
            self.sep_token.unsqueeze(0).expand(B, items, -1).unsqueeze(2)
        )  # (B, items, 1, D)
        sep_mask = mask_grouped[:, :, -1:]  # inherit the trailing token's mask
        sid_grouped = torch.cat([sid_grouped, sep_block], dim=2)
        mask_grouped = torch.cat([mask_grouped, sep_mask], dim=2)
        return sid_grouped.reshape(B, items * (H + 1), D), mask_grouped.reshape(
            B, items * (H + 1)
        )

    # ------------------------------------------------------------------
    # Encoder forward (§6.1 + §6.2)
    # ------------------------------------------------------------------

    def _encoder_forward(
        self,
        history_sids: torch.Tensor,
        attention_mask: torch.Tensor,
        user_id: Optional[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        # Job 1: offset shift + mask-multiply.
        shifted = self._add_repeating_offset(history_sids, attention_mask)
        sid_embeds = self.sid_embedding(shifted)
        # Job 2: SEP injection.
        sid_embeds, attention_mask = self._inject_sep_token(sid_embeds, attention_mask)
        # Optional user-id prepend.
        if user_id is not None and self.user_embedding is not None:
            user_idx = torch.remainder(user_id, self.user_embedding.num_embeddings)
            user_emb = self.user_embedding(user_idx).unsqueeze(1)  # (B, 1, D)
            sid_embeds = torch.cat([user_emb, sid_embeds], dim=1)
            user_mask = torch.ones(
                (sid_embeds.size(0), 1), dtype=attention_mask.dtype, device=sid_embeds.device
            )
            attention_mask = torch.cat([user_mask, attention_mask], dim=1)
        # Job 3: hand the extended mask to T5's self-attention.
        out = self.encoder_t5(
            inputs_embeds=sid_embeds,
            attention_mask=attention_mask,
        )
        return out.last_hidden_state, attention_mask

    # ------------------------------------------------------------------
    # Decoder forward (training/teacher-forced eval)
    # ------------------------------------------------------------------

    def _decoder_forward_teacher(
        self,
        future_sids: torch.Tensor,
        encoder_output: torch.Tensor,
        encoder_attention_mask: torch.Tensor,
    ) -> torch.Tensor:
        """Teacher-forced decoder pass — used at training and eval CE.

        Args:
            future_sids: (B, num_hierarchies) int64, 0-indexed target SIDs.
            encoder_output: (B, L_enc, d_model).
            encoder_attention_mask: (B, L_enc).

        Returns:
            hidden_states: (B, num_hierarchies, d_model), with BOS stripped.
        """
        B = future_sids.size(0)
        # Embed targets via the same shared SID table (with offsets).
        # No padding inside targets (always full-length), so mask is all-ones.
        target_mask = torch.ones_like(future_sids)
        shifted_targets = self._add_repeating_offset(future_sids, target_mask)
        target_embeds = self.sid_embedding(shifted_targets)
        # Prepend learned BOS.
        bos_block = self.bos_token.unsqueeze(0).expand(B, 1, -1)  # (B, 1, D)
        decoder_inputs = torch.cat([bos_block, target_embeds], dim=1)
        # Decoder self-attention mask (causal) is built by T5 internally; we
        # pass an all-ones attention mask of the right length.
        decoder_attention_mask = torch.ones(
            (B, decoder_inputs.size(1)), dtype=torch.long, device=decoder_inputs.device
        )
        out = self.decoder_t5(
            inputs_embeds=decoder_inputs,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_output,
            encoder_attention_mask=encoder_attention_mask,
            use_cache=False,
        )
        hidden = out.last_hidden_state  # (B, num_hierarchies + 1, d_model)
        # Strip the BOS-aligned position (it predicts position 0) — keep
        # positions [0, num_hierarchies) which align with target positions
        # [0, num_hierarchies).
        return hidden[:, :-1, :]

    # ------------------------------------------------------------------
    # BaseModel API
    # ------------------------------------------------------------------

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Forward pass.

        Behavior depends on whether labels are present in the batch:
          - Training / teacher-forced eval (labels present): returns
            `{DECODER_HIDDEN_KEY: (B, H, d_model)}`, consumed by `loss()`
            and `update_metric()`.
          - Inference (labels absent): runs `generate()` and returns
            `{GENERATED_SIDS_KEY: (B, K, H), STEP_SCORES_KEY: (B, K, H)}`,
            consumed by the predict-mode writer.
        """
        history_sids, attention_mask = self._extract_history_sids_and_mask(batch)
        user_id = self._extract_user_id(batch)
        target_sids = self._extract_target_sids(batch)
        encoder_output, encoder_mask = self._encoder_forward(
            history_sids, attention_mask, user_id
        )
        if target_sids is not None:
            # Training / teacher-forced eval.
            decoder_hidden = self._decoder_forward_teacher(
                target_sids, encoder_output, encoder_mask
            )
            return {DECODER_HIDDEN_KEY: decoder_hidden}
        # Pure inference.
        generated_sids, step_scores = self._generate_from_encoder(
            encoder_output, encoder_mask
        )
        return {
            GENERATED_SIDS_KEY: generated_sids,
            STEP_SCORES_KEY: step_scores,
        }

    def init_loss(self) -> None:
        """No loss modules to register; the CE is computed functionally."""
        pass

    def loss(
        self, predictions: Dict[str, torch.Tensor], batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Per-hierarchy cross-entropy, summed across hierarchies (§6.3)."""
        decoder_hidden = predictions[DECODER_HIDDEN_KEY]
        target_sids = self._extract_target_sids(batch)
        assert target_sids is not None, "loss() requires label_sids in the batch"
        total = decoder_hidden.new_zeros(())
        for h in range(self._num_hierarchies):
            logits_h = self.decoder_mlp[h](decoder_hidden[:, h, :])
            total = total + F.cross_entropy(logits_h, target_sids[:, h].long())
        return {LOSS_KEY: total}

    def init_metric(self) -> None:
        """Register retrieval metrics + the CE mean."""
        self._metric_modules[LOSS_KEY] = torchmetrics.MeanMetric()
        # Recall and NDCG at K = beam_width and a smaller default of 5.
        topk_list = sorted(set([5, self._beam_width]))
        for k in topk_list:
            self._metric_modules[f"recall@{k}"] = torchmetrics.retrieval.RetrievalRecall(
                top_k=k, sync_on_compute=False, compute_with_cache=False
            )
            self._metric_modules[f"ndcg@{k}"] = torchmetrics.retrieval.RetrievalNormalizedDCG(
                top_k=k, sync_on_compute=False, compute_with_cache=False
            )

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        # CE mean.
        if losses is not None and LOSS_KEY in losses:
            B = predictions[next(iter(predictions))].size(0)
            self._metric_modules[LOSS_KEY].update(
                losses[LOSS_KEY], losses[LOSS_KEY].new_tensor(B)
            )
        # Retrieval metrics require generated_sids.
        with torch.no_grad():
            history_sids, attention_mask = self._extract_history_sids_and_mask(batch)
            user_id = self._extract_user_id(batch)
            encoder_output, encoder_mask = self._encoder_forward(
                history_sids, attention_mask, user_id
            )
            generated_sids, step_scores = self._generate_from_encoder(
                encoder_output, encoder_mask
            )
        target_sids = self._extract_target_sids(batch)
        if target_sids is None:
            return
        # match[b, k] = True iff beam k's SID equals the ground-truth SID.
        match = (generated_sids == target_sids.unsqueeze(1)).all(dim=2)  # (B, K)
        # Marginal score per beam = product of per-step softmax probs.
        marginal_probs = step_scores.prod(dim=-1)  # (B, K)
        B, K = match.shape
        preds = marginal_probs.reshape(-1)
        target = match.reshape(-1)
        indexes = (
            torch.arange(B, device=preds.device).unsqueeze(-1).expand(B, K).reshape(-1)
        )
        for metric_name, metric in self._metric_modules.items():
            if metric_name == LOSS_KEY:
                continue
            metric.update(preds, target.to(preds.device), indexes=indexes.to(preds.device))

    # ------------------------------------------------------------------
    # Beam search generate() (§7.2)
    # ------------------------------------------------------------------

    def _generate_from_encoder(
        self,
        encoder_output: torch.Tensor,
        encoder_mask: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Unconstrained autoregressive beam search.

        Returns:
            generated_sids: (B, K, num_hierarchies) int64, 0-indexed
                codes; rows sorted descending by marginal probability.
            step_scores: (B, K, num_hierarchies) float32, the softmax
                probability of the selected code at each step. Same beam
                ordering as `generated_sids`.
        """
        B = encoder_output.size(0)
        H = self._num_hierarchies
        K = self._beam_width
        device = encoder_output.device

        generated_sids: Optional[torch.Tensor] = None  # (B, K, h)
        step_scores: Optional[torch.Tensor] = None  # (B, K, h)
        marginal_probs: Optional[torch.Tensor] = None  # (B, K)

        for h in range(H):
            if h == 0:
                # First step: single decoder query per sample (BOS only).
                bos_block = self.bos_token.unsqueeze(0).expand(B, 1, -1)  # (B,1,D)
                dec_attn_mask = torch.ones(
                    (B, 1), dtype=torch.long, device=device
                )
                out = self.decoder_t5(
                    inputs_embeds=bos_block,
                    attention_mask=dec_attn_mask,
                    encoder_hidden_states=encoder_output,
                    encoder_attention_mask=encoder_mask,
                    use_cache=False,
                )
                hidden_last = out.last_hidden_state[:, -1, :]  # (B, D)
                logits = self.decoder_mlp[h](hidden_last)  # (B, codebook_sizes[0])
                probs = F.softmax(logits, dim=-1)  # (B, codebook_sizes[0])
                top_probs, top_codes = torch.topk(probs, k=K, dim=-1)  # (B, K)
                generated_sids = top_codes.unsqueeze(-1).long()  # (B, K, 1)
                step_scores = top_probs.unsqueeze(-1)  # (B, K, 1)
                marginal_probs = top_probs  # (B, K)
            else:
                # Subsequent steps: B*K beams each carry their accumulated
                # 0-indexed SID prefix of length h.
                # Build decoder input: BOS + embedded(prefix), shape (B*K, h+1, D).
                prefix = generated_sids.view(B * K, h)  # (B*K, h)
                prefix_mask = torch.ones_like(prefix)
                shifted_prefix = self._add_repeating_offset(prefix, prefix_mask)
                prefix_embeds = self.sid_embedding(shifted_prefix)  # (B*K, h, D)
                bos_block = self.bos_token.unsqueeze(0).expand(B * K, 1, -1)
                decoder_inputs = torch.cat([bos_block, prefix_embeds], dim=1)
                dec_attn_mask = torch.ones(
                    (B * K, h + 1), dtype=torch.long, device=device
                )
                # Repeat encoder outputs per beam.
                enc_out_rep = encoder_output.repeat_interleave(K, dim=0)
                enc_mask_rep = encoder_mask.repeat_interleave(K, dim=0)
                out = self.decoder_t5(
                    inputs_embeds=decoder_inputs,
                    attention_mask=dec_attn_mask,
                    encoder_hidden_states=enc_out_rep,
                    encoder_attention_mask=enc_mask_rep,
                    use_cache=False,
                )
                hidden_last = out.last_hidden_state[:, -1, :]  # (B*K, D)
                logits = self.decoder_mlp[h](hidden_last)  # (B*K, codebook_sizes[h])
                probs = F.softmax(logits, dim=-1)  # (B*K, codebook_sizes[h])
                C = self._codebook_sizes[h]
                # Joint score = marginal_probs[b, k] * probs[b*k, c].
                joint = marginal_probs.view(B * K, 1) * probs  # (B*K, C)
                joint = joint.view(B, K * C)
                top_probs, top_idx = torch.topk(joint, k=K, dim=-1)  # (B, K)
                # Decompose top_idx into (parent_beam, code).
                parent_beam = top_idx // C  # (B, K)
                new_code = top_idx % C  # (B, K)
                # Reorder existing tensors to follow `parent_beam`.
                batch_idx = (
                    torch.arange(B, device=device).unsqueeze(-1).expand(B, K)
                )  # (B, K)
                reordered_sids = generated_sids[batch_idx, parent_beam]  # (B, K, h)
                reordered_step_scores = step_scores[batch_idx, parent_beam]  # (B, K, h)
                # Per-step score for the new code: probs[parent_beam_flat, new_code].
                parent_flat = (batch_idx * K + parent_beam).view(-1)  # (B*K,)
                new_code_flat = new_code.view(-1)  # (B*K,)
                step_score_new = probs[parent_flat, new_code_flat].view(B, K)
                # Update.
                generated_sids = torch.cat(
                    [reordered_sids, new_code.unsqueeze(-1).long()], dim=-1
                )  # (B, K, h+1)
                step_scores = torch.cat(
                    [reordered_step_scores, step_score_new.unsqueeze(-1)], dim=-1
                )  # (B, K, h+1)
                marginal_probs = top_probs

        # Beams are already sorted descending by marginal_probs from torch.topk
        # at each step's top-K selection. Final return shapes:
        #   generated_sids: (B, K, H) int64, 0-indexed (no +1 restoration —
        #     the on-disk convention is also 0-indexed; §3.6).
        #   step_scores  : (B, K, H) float32.
        return generated_sids, step_scores
