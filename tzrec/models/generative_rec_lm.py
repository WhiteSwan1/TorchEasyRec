# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Generic generative-recommendation language-model base for TorchEasyRec.

Implements the FINAL design (see FINAL_DESIGN_GENERATIVE_REC_LM.md):

  * Per-family subclasses (design §2 / G4): ``GenerativeRecLM`` is the
    abstract base; each LLM family is a concrete subclass declaring a
    ``CHAT_TEMPLATE`` class var (e.g. ``Qwen2RecLM`` in
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

(The old offline-tokenized ``tzrec/models/qwen2.py`` v1 wrapper and its
``Qwen2 qwen2 = 600`` proto entry have been REMOVED — proto field 600 is
reserved. This streaming pipeline is the only generative-rec path.)
"""

from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Tuple

import torch
import torchmetrics
from torch import nn
from transformers import AutoModelForCausalLM, AutoTokenizer

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.model import BaseModel
from tzrec.protos.model_pb2 import ModelConfig


def _encode_no_special(tokenizer, text: str) -> List[int]:
    """Encode a fragment without prepending BOS / appending EOS specials.

    We're building the prompt manually from explicit ``<|im_start|>`` markers,
    so we must NOT let the tokenizer's BOS/EOS handling double-emit them.
    """
    return tokenizer.encode(text, add_special_tokens=False)


class GenerativeRecLM(BaseModel):
    """Abstract base for HF-backed generative-recommendation LMs.

    Subclasses declare ``CHAT_TEMPLATE`` (design §5) and rarely override
    ``predict()`` (e.g. a future ``MixtralRecLM`` must call the full HF
    forward to capture ``aux_loss``). Everything else — model construction,
    SID vocab extension, template caching, splice, algr-aligned forward,
    loss/metrics — lives here.

    ``CHAT_TEMPLATE`` keys (all strings):
        system_prefix / system_suffix   — wrap the system instruction
        user_prefix / user_suffix       — wrap the user message
        asst_prefix / asst_suffix       — wrap the assistant answer
        default_system_instruction      — used when the proto doesn't
                                          override ``system_instruction``
    """

    CHAT_TEMPLATE: Optional[Dict[str, str]] = None

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

        tpl = type(self).CHAT_TEMPLATE
        if tpl is None:
            raise NotImplementedError(
                f"{type(self).__name__} must declare a non-empty CHAT_TEMPLATE "
                f"class var (design §2/G4); GenerativeRecLM itself is abstract "
                f"— set generative_rec_lm.class_name to a concrete family "
                f"(e.g. 'Qwen2RecLM')."
            )

        # --------- proto -> python knobs ------------------------------
        self._input_name: str = cfg.user_sequence_feature_name
        self._label_name: str = cfg.label_feature_name
        self._ignore_index: int = int(cfg.ignore_index)
        codebook = list(cfg.codebook)
        if len(codebook) == 0:
            raise ValueError(
                "GenerativeRecLM: codebook must be non-empty "
                "(see design §3 — required field)"
            )
        sid_atoms = sum(int(c) for c in codebook)
        pad_mult = int(cfg.vocab_pad_to_multiple_of) or 128

        # --------- backbone + tokenizer -------------------------------
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
        # ``use_fast=True`` is the modern default; explicit for clarity.
        tokenizer = AutoTokenizer.from_pretrained(hf_model_id, use_fast=True)

        # --------- vocab extension (codebook required) ----------------
        # The SID-atom base is the tokenizer's next free id BEFORE adding
        # ``C0..``. For Qwen2.5-0.5B that's 151665 = `model.config.vocab_size`
        # (151936, includes ~300 reserved padding slots) minus the unused
        # reserved span — so use ``len(tokenizer)`` directly, NOT
        # ``model.config.vocab_size``.
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
        # Final vocab = base + sid_atoms, padded up to multiple of pad_mult.
        # Matches algr's layout: SID atoms appended directly to the existing
        # tokenizer vocab; offset arithmetic is `token = base + (sid - 1)`.
        self.lm.resize_token_embeddings(
            base + sid_atoms, pad_to_multiple_of=pad_mult
        )

        # L3 safety check: assert C0 lands at the recorded base.
        c0_id = tokenizer.convert_tokens_to_ids("C0")
        if c0_id != base:
            raise RuntimeError(
                f"GenerativeRecLM: SID atom layout mismatch — expected "
                f"C0 at token id {base}, got {c0_id}. "
                f"Splice arithmetic would produce wrong token ids."
            )
        self._base_vocab = base  # used in `_splice_input_ids`

        # L1 + L7 mitigation: pad with eos_token_id on the LEFT side.
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        self._pad_token_id = int(pad_id)

        # --------- cache chat-template buffers ------------------------
        self._build_prompt_tokens(tokenizer, cfg)

        # Diagnostics for the first launch — useful when chasing splice bugs.
        self._smoke_log_once = (os.environ.get("TZREC_GENRECLM_DEBUG", "0") == "1")
        self._first_predict = True

    # ------------------------------------------------------------------ template
    def _build_prompt_tokens(self, tokenizer, cfg) -> None:
        """Tokenise the family chat template once; cache as buffers.

        Composes the proto's optional ``system_instruction`` /
        ``user_prefix_text`` / ``user_suffix_text`` (algr's CN prompt
        wrappers — L2/L4 mitigations) with the family's static fragments:

            tpl_system      = system_prefix + system_instruction + system_suffix
            tpl_user_prefix = user_prefix + user_prefix_text
            tpl_user_suffix = user_suffix_text + user_suffix
            tpl_asst_prefix / tpl_asst_suffix verbatim from the template

        Buffers are non-persistent — they live with the module (move with
        ``model.to(...)``) but stay off the state_dict so HF safetensors
        round-tripping isn't polluted by TER-only state.
        """
        tpl = type(self).CHAT_TEMPLATE
        sys_text = cfg.system_instruction or tpl["default_system_instruction"]
        u_pre = cfg.user_prefix_text or ""
        u_suf = cfg.user_suffix_text or ""
        frags = {
            "system": tpl["system_prefix"] + sys_text + tpl["system_suffix"],
            "user_prefix": tpl["user_prefix"] + u_pre,
            "user_suffix": u_suf + tpl["user_suffix"],
            "asst_prefix": tpl["asst_prefix"],
            "asst_suffix": tpl["asst_suffix"],
        }
        for slot_name, frag_str in frags.items():
            ids = torch.tensor(
                _encode_no_special(tokenizer, frag_str), dtype=torch.long
            )
            self.register_buffer(f"tpl_{slot_name}", ids, persistent=False)
        # algr appends eos to BOTH input_ids and labels at train time
        # (algr/models/qwen2_5/data.py:46-47) — i.e. the trailing eos is a
        # SUPERVISED token. Cache it so the splice can mirror that exactly.
        self.register_buffer(
            "tpl_eos",
            torch.tensor([int(tokenizer.eos_token_id)], dtype=torch.long),
            persistent=False,
        )

    # ------------------------------------------------------------------ init_input
    def init_input(self) -> None:
        """No-op override.

        The HF backbone owns its own ``embed_tokens``; we don't use TER's
        ``EmbeddingGroup`` at all. Token IDs flow through directly.
        """
        self.embedding_group = None

    # ------------------------------------------------------------------ jagged -> rows
    @staticmethod
    def _jagged_to_row_list(jt) -> List[torch.Tensor]:
        """Convert a TER JaggedTensor (values, lengths) to a list of 1-D
        int64 row tensors.

        ``values`` may arrive as float (TER's ``sequence_raw_feature`` reads
        ``list<int64>`` as float — see [[project-tzrec-qwen2-integration]]
        gotcha §B). We cast to long here; SID values fit in float32 mantissa
        for any realistic codebook size (< 2^24).
        """
        values = jt.values() if callable(getattr(jt, "values", None)) else jt.values
        lengths = jt.lengths() if callable(getattr(jt, "lengths", None)) else jt.lengths
        if values.dim() == 2 and values.size(-1) == 1:
            values = values.squeeze(-1)
        values = values.long()
        lengths = lengths.long()
        out: List[torch.Tensor] = []
        start = 0
        for n in lengths.tolist():
            out.append(values[start : start + n])
            start += n
        return out

    # ------------------------------------------------------------------ splice
    def _splice_input_ids(
        self,
        user_seq_rows: List[torch.Tensor],
        label_rows: List[torch.Tensor],
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Build ``(input_ids, labels, attention_mask)``, each ``(B, T_max)``.

        Left-padded with ``eos_token_id``. ``attention_mask`` is essential —
        without it self-attention would let pad positions pollute real
        positions' hidden states. CE is separately protected by ``-100``
        labels at pad slots, but the forward needs the mask too.

        SID → token: ``token = sid + base_vocab - 1`` (SID atoms ``C0..``
        start at position ``base_vocab``; SID indices are 1-indexed).
        """
        assert len(user_seq_rows) == len(label_rows)
        B = len(user_seq_rows)
        dev = self.tpl_system.device
        base = self._base_vocab

        rows_ids: List[torch.Tensor] = []
        rows_lab: List[torch.Tensor] = []
        for i in range(B):
            # SID → token id with int math. Map ``sid`` to its corresponding
            # ``C{sid-1}`` atom: ``token = base + (sid - 1)`` = ``sid + (base-1)``.
            # Cast to long matches the buffer dtype.
            u_tok = (user_seq_rows[i].to(dev) + (base - 1))
            a_tok = (label_rows[i].to(dev) + (base - 1))
            ids = torch.cat([
                self.tpl_system, self.tpl_user_prefix, u_tok,
                self.tpl_user_suffix, self.tpl_asst_prefix, a_tok,
                self.tpl_asst_suffix, self.tpl_eos,
            ])
            ign = torch.full_like(ids, self._ignore_index)
            start = (
                self.tpl_system.numel()
                + self.tpl_user_prefix.numel()
                + u_tok.numel()
                + self.tpl_user_suffix.numel()
                + self.tpl_asst_prefix.numel()
            )
            ign[start : start + a_tok.numel()] = a_tok
            # algr supervises the trailing eos (after the masked
            # ``<|im_end|>\n`` markers) — data.py:46-47. Mirror it.
            ign[-1] = self.tpl_eos[0]
            rows_ids.append(ids)
            rows_lab.append(ign)

        T = max(r.numel() for r in rows_ids)
        input_ids = torch.full(
            (B, T), self._pad_token_id, dtype=torch.long, device=dev
        )
        labels = torch.full(
            (B, T), self._ignore_index, dtype=torch.long, device=dev
        )
        attention_mask = torch.zeros((B, T), dtype=torch.long, device=dev)
        for i, (ids, ign) in enumerate(zip(rows_ids, rows_lab)):
            n = ids.numel()
            # LEFT padding: write rows to the END of each (T,) slot.
            input_ids[i, -n:] = ids
            labels[i, -n:] = ign
            attention_mask[i, -n:] = 1
        return input_ids, labels, attention_mask

    @staticmethod
    def _min_first_non_neg_index(labels: torch.Tensor) -> int:
        """Verbatim port of algr's helper (al_sid/algr/models/qwen2_5/
        modeling_qwen.py:1267-1274).

        Returns the smallest position (across rows in the batch) where the
        first non-(-100) label appears. Used to decide how many trailing
        positions to feed into ``lm_head``.
        """
        tmp = (labels >= 0).cumsum(dim=-1)
        return int((tmp == 1).float().argmax(dim=-1).min().item())

    # ------------------------------------------------------------------ predict
    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        jt_u = batch.sequence_dense_features[self._input_name]
        jt_l = batch.sequence_dense_features[self._label_name]
        u_rows = self._jagged_to_row_list(jt_u)
        l_rows = self._jagged_to_row_list(jt_l)

        input_ids, labels, attention_mask = self._splice_input_ids(u_rows, l_rows)

        if self._smoke_log_once and self._first_predict:
            print(
                f"[GENRECLM_DEBUG] first batch: B={input_ids.shape[0]} "
                f"T={input_ids.shape[1]} pad_id={self._pad_token_id} "
                f"ign={self._ignore_index} dev={input_ids.device} "
                f"input_ids[0, -8:]={input_ids[0, -8:].tolist()} "
                f"labels[0, -8:]={labels[0, -8:].tolist()}",
                flush=True,
            )
            self._first_predict = False

        outputs = self.lm.model(
            input_ids=input_ids, attention_mask=attention_mask
        )
        hidden = outputs.last_hidden_state  # (B, T, D)

        # Suffix slice in BOTH train and eval. algr only slices when
        # training (its eval goes through a separate beam-search predict),
        # but for CE the slice is value-identical (positions outside the
        # suffix all carry -100 labels) and it bounds the logits tensor to
        # (B, T_suffix, V) — without it, eval at bsz=80 would materialise
        # (B, T, 217k) logits plus HF loss_function's fp32 upcast and OOM.
        if (labels >= 0).any():
            keep = labels.shape[1] - self._min_first_non_neg_index(labels) + 1
            sl = slice(-keep, None)
            labels_sl = labels[:, sl]
        else:
            sl = slice(None)
            labels_sl = labels

        logits = self.lm.lm_head(hidden[:, sl, :])

        # ``loss_function`` is the HF ``ForCausalLMLoss`` callable hung off
        # every ``…ForCausalLM`` class; does shift-by-one + CE with -100
        # ignore. Calling it here matches algr's training-step loss exactly.
        loss = self.lm.loss_function(
            logits=logits,
            labels=labels_sl,
            vocab_size=self.lm.config.vocab_size,
        )
        return {"loss": loss, "logits": logits}

    # ------------------------------------------------------------------ loss
    def init_loss(self) -> None:
        return

    def loss(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
    ) -> Dict[str, torch.Tensor]:
        return {"ce_loss": predictions["loss"]}

    # ------------------------------------------------------------------ metrics
    # See [[project-tzrec-qwen2-integration]] gotcha §C: BaseModel only
    # declares the eval-side metric methods; the train loop calls both
    # families, so we must override both.
    def init_metric(self) -> None:
        # Mean CE over the eval set — gives `_evaluate` something to log
        # (BaseModel.compute_metric iterates `_metric_modules` generically;
        # torchmetrics handles the cross-rank sync at compute()).
        self._metric_modules["ce_loss"] = torchmetrics.MeanMetric()

    def update_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        self._metric_modules["ce_loss"].update(predictions["loss"].detach())

    def init_train_metric(self) -> None:
        return

    def update_train_metric(
        self,
        predictions: Dict[str, torch.Tensor],
        batch: Batch,
        losses: Optional[Dict[str, torch.Tensor]] = None,
    ) -> None:
        return
