# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Qwen2/Qwen2.5 family subclass of ``GenerativeRecLM`` (design §5).

Selected from the pipeline config via::

    model_config {
        generative_rec_lm {
            class_name: "Qwen2RecLM"
            ...
        }
    }

This subclass owns the decoder-only-chat implementation: the ChatML prompt
template, the causal-LM splice, and the ``.model``/``.lm_head`` forward
(design §15/§16). The ``GenerativeRecLM`` base owns the architecture-agnostic
plumbing (vocab extension, jagged→row, loss, metrics).

The splice/forward here are generic to decoder-only families sharing Qwen2's
``.model``/``.lm_head`` layout (Llama/Mistral/Gemma/Phi — design §16), not
Qwen2-specific; only ``QWEN2_TEMPLATE`` is. When a second such family lands,
lift ``_splice_input_ids`` / ``_min_first_non_neg_index`` / ``predict`` (and
the ChatML ``_build_prompt_tokens``) into an intermediate
``DecoderOnlyChatRecLM`` base so each family is just its template. Until then
they live here.
"""

from typing import Dict, List, Tuple

import torch
from torch.nn.utils.rnn import pad_sequence

from tzrec.datasets.utils import Batch
from tzrec.models.generative_rec_lm import GenerativeRecLM


def _encode_no_special(tokenizer, text: str) -> List[int]:
    """Encode a fragment without prepending BOS / appending EOS specials.

    We're building the prompt manually from explicit ``<|im_start|>`` markers,
    so we must NOT let the tokenizer's BOS/EOS handling double-emit them.
    """
    return tokenizer.encode(text, add_special_tokens=False)

# Verbatim Qwen2 ChatML fragments. ``default_system_instruction`` matches
# algr/models/qwen2_5/data.py:73 ("default_instruction") bit-for-bit.
QWEN2_TEMPLATE = {
    "system_prefix": "<|im_start|>system\n",
    "system_suffix": "<|im_end|>\n",
    "user_prefix": "<|im_start|>user\n",
    "user_suffix": "<|im_end|>\n",
    "asst_prefix": "<|im_start|>assistant\n",
    "asst_suffix": "<|im_end|>\n",
    "default_system_instruction": (
        "You are Qwen, created by Alibaba Cloud. You are a helpful assistant."
    ),
}


class Qwen2RecLM(GenerativeRecLM):
    """Qwen2 / Qwen2.5 generative-recommendation LM."""

    CHAT_TEMPLATE = QWEN2_TEMPLATE

    def _build_prompt_tokens(self, tokenizer, cfg) -> None:
        """Tokenise the family chat template once; cache as buffers.

        Composes the proto's optional ``system_instruction`` /
        ``user_prefix_text`` / ``user_suffix_text`` (algr's CN prompt
        wrappers) with the family's static fragments:

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

        Every answer is exactly ``self._num_levels`` SID codes (one per codebook
        level — validated at the data boundary in ``_sid_token_rows``), so the
        supervised tail ``[answer | asst_suffix | eos]`` has a FIXED width and,
        after left-padding, lands in the SAME columns for every row: ``labels``
        is built in one vectorized assignment (no per-row label loop).
        ``input_ids`` still varies per row (the user history length differs).

        ``user_seq_rows`` / ``label_rows`` already hold extended-vocab token ids
        on the model device (see ``_sid_token_rows``).
        """
        assert len(user_seq_rows) == len(label_rows)
        A = self._num_levels

        # input_ids: assembled per row (user history length varies), then
        # left-padded into a (B, T) batch (real content right-aligned).
        rows_ids = [
            torch.cat([
                self.tpl_system, self.tpl_user_prefix, user_seq_rows[i],
                self.tpl_user_suffix, self.tpl_asst_prefix, label_rows[i],
                self.tpl_asst_suffix, self.tpl_eos,
            ])
            for i in range(len(user_seq_rows))
        ]
        input_ids = pad_sequence(
            rows_ids, batch_first=True,
            padding_value=self._pad_token_id, padding_side="left",
        )
        attention_mask = pad_sequence(
            [torch.ones_like(r) for r in rows_ids], batch_first=True,
            padding_value=0, padding_side="left",
        )

        # labels: the supervised tail is fixed-width, so left-padding aligns it
        # to the same columns for every row -> one vectorized write.
        # tail layout (from the end): [answer(A) | asst_suffix(s) | eos(1)].
        # ``tail <= T`` always holds: every row already contains those tokens.
        B, T = input_ids.shape
        s = self.tpl_asst_suffix.numel()
        tail = A + s + 1
        labels = torch.full(
            (B, T), self._ignore_index, dtype=torch.long, device=self.device
        )
        labels[:, T - tail : T - tail + A] = torch.stack(label_rows)
        labels[:, -1] = self.tpl_eos[0]  # algr supervises the trailing eos
        return input_ids, labels, attention_mask

    @staticmethod
    def _min_first_non_neg_index(labels: torch.Tensor) -> int:
        """Return the batch-min index of the first non-(-100) label.

        Verbatim port of algr's helper (al_sid/algr/models/qwen2_5/
        modeling_qwen.py:1267-1274) — the smallest position (across rows in
        the batch) where the first non-(-100) label appears, used to decide
        how many trailing positions to feed into ``lm_head``.
        """
        tmp = (labels >= 0).cumsum(dim=-1)
        return int((tmp == 1).float().argmax(dim=-1).min().item())

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Decoder-only forward: splice → ``.model`` → suffix-slice → CE loss."""
        # SID indices -> token ids once, at the data boundary (see
        # _sid_token_rows); the splice then just assembles the prompt.
        u_rows = self._sid_token_rows(batch.sequence_dense_features[self._input_name])
        l_rows = self._sid_token_rows(
            batch.sequence_dense_features[self._label_name],
            expected_width=self._num_levels,  # answer = one item = num_levels codes
        )

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
