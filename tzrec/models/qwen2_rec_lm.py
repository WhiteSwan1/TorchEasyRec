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

The family contributes ONLY its chat-template fragments; the splice,
algr-aligned forward, vocab extension and checkpoint plumbing all live in
the ``GenerativeRecLM`` base. A new LLM family is one file like this one
(see ``FINAL_DESIGN_GENERATIVE_REC_LM.md`` §2/§5 — ``Qwen3RecLM`` should
subclass ``Qwen2RecLM`` and override only what differs).
"""

from tzrec.models.generative_rec_lm import GenerativeRecLM

# Verbatim Qwen2 ChatML fragments. ``default_system_instruction`` matches
# algr/models/qwen2_5/data.py:73 ("default_instruction") bit-for-bit — the
# L2 mitigation in design §11.
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
