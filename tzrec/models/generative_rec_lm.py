# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Generic generative-recommendation language-model base for TorchEasyRec.

Implements the FINAL design (see FINAL_DESIGN_GENERATIVE_REC_LM.md):

  * Per-family subclasses (design §2 / G4): ``GenerativeRecLM`` is the
    abstract base; each LLM family is a concrete subclass implementing the
    ``_build_prompt_tokens`` and ``predict`` hooks (e.g. ``Qwen2RecLM`` in
    ``tzrec/models/qwen2_rec_lm.py``). The pipeline config selects the
    family via ``generative_rec_lm.class_name``; dispatch goes through the
    BaseModel registry (subclasses auto-register by class name).
  * Streaming sample format: each row carries two raw-int64 sequence features,
    ``user_sequence`` (list[int]) and ``label`` (list[int]), both holding raw
    SID indices in ``[1, sum(codebook)]``.
  * The chat template is tokenised ONCE at ``__init__`` and cached as
    ``nn.Module`` non-persistent buffers, so per-batch encoding is purely
    integer arithmetic + tensor concatenation (no HF tokenizer in the hot
    path).
  * SID → token id by integer offset: ``token = sid + base_vocab - 1`` (the
    SID atoms ``C0..C{sum-1}`` are added right after the original vocabulary,
    no [SEP] in between; matches algr's ``add_tokens`` layout).
  * Left padding with ``eos_token_id`` (L7 fix from §11 of the design doc) —
    real content sits at the END of every row so the suffix slice captures
    only ``[response + end_markers]`` and matches algr's pad-side exactly.
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import torch
import torchmetrics
from transformers import AutoModelForCausalLM, AutoTokenizer

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


class GenerativeRecLM(BaseModel):
    """Abstract base for HF-backed generative-recommendation LMs.

    The base owns the architecture-agnostic plumbing: model construction,
    SID vocab extension, the shared sample data-prep (``_sid_token_rows`` /
    ``_tokenize_sids`` — the streaming SID sample contract is the same for all
    families, design §1), loss, and metrics. The two architecture-specific
    pieces are abstract hooks that each family subclass implements (§15/§16):

        _build_prompt_tokens(tokenizer, cfg)  — cache the prompt template
        predict(batch)                        — build inputs + HF forward

    ``Qwen2RecLM`` (``tzrec/models/qwen2_rec_lm.py``) provides the decoder-only
    chat implementation (ChatML splice + ``.model``/``.lm_head`` forward),
    reusable by Llama/Mistral/Gemma/Phi-style families; GPT-NeoX/RWKV/Mamba/T5
    each need their own. The pipeline config selects the family via
    ``generative_rec_lm.class_name`` (resolved through the BaseModel registry).
    """

    def __new__(cls, model_config: ModelConfig, *args: Any, **kwargs: Any):
        """Dispatch to the concrete family subclass.

        ``tzrec.main._create_model`` resolves the proto oneof message name
        (``GenerativeRecLM``) to THIS class; the actual family is selected
        by ``generative_rec_lm.class_name`` and looked up in the BaseModel
        registry (every subclass auto-registers via the metaclass).
        """
        if cls is GenerativeRecLM:
            cfg = getattr(model_config, model_config.WhichOneof("model"))
            class_name = cfg.class_name
            # pyre-ignore [16]
            sub_cls = BaseModel.create_class(class_name)
            if not issubclass(sub_cls, GenerativeRecLM):
                raise ValueError(
                    f"generative_rec_lm.class_name = {class_name!r} resolves "
                    f"to {sub_cls}, which is not a GenerativeRecLM subclass."
                )
            return super().__new__(sub_cls)
        return super().__new__(cls)

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(model_config, features, labels, sample_weights, **kwargs)
        cfg = self._model_config  # populated by BaseModel from WhichOneof

        # proto -> python knobs
        self._input_name: str = cfg.user_sequence_feature_name
        self._label_name: str = cfg.label_feature_name
        self._ignore_index: int = int(cfg.ignore_index)
        codebook = list(cfg.codebook)
        if len(codebook) == 0:
            raise ValueError(
                "GenerativeRecLM: codebook must be non-empty "
                "(see design §3 — required field)"
            )
        # One entry per RQ level: ``len(codebook)`` = SID codes per item (the
        # exact width of every answer), ``sum(codebook)`` = total SID atoms to
        # append to the vocab. e.g. AL-GR is 3 levels x 8192 -> [8192,8192,8192].
        self._num_levels = len(codebook)
        sid_atoms = sum(int(c) for c in codebook)
        pad_mult = int(cfg.vocab_pad_to_multiple_of) or 128

        # backbone + tokenizer
        hf_model_id = cfg.hf_model_id
        if not hf_model_id:
            raise ValueError(
                "GenerativeRecLM v1: hf_model_id is required "
                "(architecture-spec path deferred to v1.x)"
            )
        # torch_dtype="auto" preserves the safetensors-stored dtype (bf16
        # for Qwen2.5-0.5B). The default would silently upcast to fp32.
        # On CPU the same flag is honoured; on GPU it avoids a 2× memory
        # blow-up.
        self.lm = AutoModelForCausalLM.from_pretrained(
            hf_model_id, torch_dtype="auto"
        )
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)

        # vocab extension: base = tokenizer's next free id BEFORE adding C0..
        # (use len(tokenizer), NOT config.vocab_size which counts reserved slots).
        base = len(tokenizer)
        new_atoms = [f"C{i}" for i in range(sid_atoms)]
        added = tokenizer.add_tokens(new_atoms)
        if added != sid_atoms:
            # The tokenizer already had some Cxxx tokens — we expect a fresh
            # base, so this would silently break our offset arithmetic.
            raise RuntimeError(
                f"GenerativeRecLM: tokenizer was expected to grow by "
                f"{sid_atoms} new atoms, only added {added}. "
                f"Aborting to avoid silent SID-token mismatch."
            )
        # SID atoms appended directly after the existing vocab (algr's layout);
        # offset arithmetic is `token = base + (sid - 1)`.
        self.lm.resize_token_embeddings(
            base + sid_atoms, pad_to_multiple_of=pad_mult
        )

        # assert C0 landed at the recorded base (the offset arithmetic relies on it)
        c0_id = tokenizer.convert_tokens_to_ids("C0")
        if c0_id != base:
            raise RuntimeError(
                f"GenerativeRecLM: SID atom layout mismatch — expected "
                f"C0 at token id {base}, got {c0_id}. "
                f"Splice arithmetic would produce wrong token ids."
            )
        self._base_vocab = base

        # pad token for the left-padded splice (fall back to eos)
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self._pad_token_id = int(pad_id)

        self._build_prompt_tokens(tokenizer, cfg)

        # one-shot debug dump of the first spliced batch
        self._smoke_log_once = (os.environ.get("TZREC_GENRECLM_DEBUG", "0") == "1")
        self._first_predict = True

    def _build_prompt_tokens(self, tokenizer, cfg) -> None:
        """Family hook: cache the tokenised prompt template as buffers.

        Called from ``__init__`` after vocab extension; the buffers it
        registers are consumed by the family's ``predict``. Architecture-
        specific — see design §15.1/§15.2. Subclasses MUST implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement _build_prompt_tokens "
            f"(GenerativeRecLM is abstract)."
        )

    def init_input(self) -> None:
        """No-op override.

        The HF backbone owns its own ``embed_tokens``; we don't use TER's
        ``EmbeddingGroup`` at all. Token IDs flow through directly.
        """
        self.embedding_group = None

    @property
    def device(self) -> torch.device:
        """Device the HF backbone runs on — the single source for model I/O."""
        return self.lm.device

    def _tokenize_sids(self, sids: torch.Tensor) -> torch.Tensor:
        """Map raw 1-indexed SID values to extended-vocab token ids.

        Atom ``C{k}`` sits at ``base_vocab + k`` (atoms appended right after the
        original vocab), so ``token_id = sid + base_vocab - 1``. The integer
        counterpart of the HF tokenizer used for text; shape-agnostic.
        """
        return sids + (self._base_vocab - 1)

    def _sid_token_rows(
        self, jt, expected_width: Optional[int] = None
    ) -> List[torch.Tensor]:
        """Read a SID jagged feature -> per-row token-id tensors.

        TER delivers the feature as a JaggedTensor (flat ``values`` +
        ``lengths``); ``values`` may arrive as float / shape ``(N, 1)``. The
        whole batch is tokenized once (``_tokenize_sids``) on the backbone
        device, then split into rows.

        ``expected_width``, when set, enforces the sample contract here at the
        data boundary: every row must have exactly that many codes (e.g. the
        answer = ``num_levels``); a deviation is an anomalous sample.
        """
        values = jt.values()
        lengths = jt.lengths()
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        # host-side split bounds, read before the H2D copy below
        sizes = lengths.long().tolist()
        if expected_width is not None:
            bad = [i for i, n in enumerate(sizes) if n != expected_width]
            if bad:
                raise ValueError(
                    f"{type(self).__name__}: each SID item must be "
                    f"{expected_width} codes (len(codebook)); rows {bad} have "
                    f"{[sizes[i] for i in bad]} — anomalous sample(s)."
                )
        # one vectorized SID->token map over the whole batch, on the backbone device
        values = self._tokenize_sids(values.to(self.device).long())
        return list(torch.split(values, sizes))

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Family hook: build inputs, run the HF forward, return ``{"loss": ...}``.

        Architecture-specific — the decoder-only implementation lives in
        ``Qwen2RecLM``; other families (GPT-NeoX, Mamba, T5, …) need their own.
        See design §15.4/§16. Subclasses MUST implement this.
        """
        raise NotImplementedError(
            f"{type(self).__name__} must implement predict "
            f"(GenerativeRecLM is abstract)."
        )

    def init_loss(self) -> None:
        """No-op: the loss is computed inside ``predict`` (HF loss_function)."""
        return

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        """Surface the CE loss already computed in ``predict``."""
        return {"ce_loss": predictions["loss"]}

    # BaseModel declares only the eval-side metric hooks, but the train loop
    # calls both eval and train hooks, so both are overridden here.
    def init_metric(self) -> None:
        """Register a mean-CE metric for the eval loop."""
        self._metric_modules["ce_loss"] = torchmetrics.MeanMetric()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """Update the mean-CE metric with this batch's loss."""
        self._metric_modules["ce_loss"].update(predictions["loss"].detach())

    def init_train_metric(self) -> None:
        """No-op: no train-time metric beyond the logged CE loss."""
        return

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        """No-op: no train-time metric beyond the logged CE loss."""
        return
