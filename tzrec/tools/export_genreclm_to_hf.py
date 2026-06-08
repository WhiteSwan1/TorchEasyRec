# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0

"""Export a `GenerativeRecLM` TER DCP checkpoint to a HF-loadable directory.

Design §6.4 (FINAL_DESIGN_GENERATIVE_REC_LM.md), simplified: instead of
hand-writing safetensors shards, we rebuild the model from the pipeline
config (which re-applies the SID vocab extension so shapes match), overlay
the DCP shards onto it, and let HF's ``save_pretrained`` deal with weight
tying, sharding and config serialisation. The extended tokenizer (C0.. at
``len(tokenizer)`` — TER layout, no [SEP]) is saved alongside so generation
consumers decode SID atoms with the SAME ids the model was trained on.

DCP shard FQNs are ``model.lm.<hf_fqn>`` (TrainWrapper prefix ``model.`` +
wrapper attr ``lm.``); we restore through a TrainWrapper-shaped state dict
so no manual FQN surgery is needed.

Usage (CPU-only; safe to run next to a live training)::

    PYTHONPATH=. python -m tzrec.tools.export_genreclm_to_hf \\
        --pipeline_config_path experiments/<run>/pipeline.config \\
        --checkpoint_path experiments/<run>/model.ckpt-40000 \\
        --export_dir experiments/<run>/export_hf_40000
"""

from __future__ import annotations

import argparse
import os

import torch
from google.protobuf import text_format
from torch.distributed.checkpoint import FileSystemReader, load

from tzrec.models.generative_rec_lm import GenerativeRecLM  # noqa: F401
from tzrec.models.model import BaseModel
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from transformers import AutoTokenizer


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pipeline_config_path", required=True)
    ap.add_argument("--checkpoint_path", required=True)
    ap.add_argument("--export_dir", required=True)
    args = ap.parse_args()

    pipeline_config = EasyRecConfig()
    with open(args.pipeline_config_path) as f:
        text_format.Merge(f.read(), pipeline_config)
    model_config = pipeline_config.model_config
    grl_cfg = getattr(model_config, model_config.WhichOneof("model"))

    # Rebuild the model exactly as training did (from_pretrained backbone +
    # SID vocab extension), CPU-resident.
    print(f"[export] building {grl_cfg.class_name} from {grl_cfg.hf_model_id}")
    # pyre-ignore [16]
    model_cls = BaseModel.create_class("GenerativeRecLM")
    model = model_cls(model_config, features=[], labels=[])
    model.eval()

    # Overlay DCP shards. Shard keys are "model.lm.<hf_fqn>" — present the
    # state dict under the same prefix.
    ckpt_model_dir = os.path.join(args.checkpoint_path, "model")
    print(f"[export] overlaying DCP shards from {ckpt_model_dir}")
    lm_sd = model.lm.state_dict()
    prefixed = {f"model.lm.{k}": v for k, v in lm_sd.items()}
    load(prefixed, storage_reader=FileSystemReader(ckpt_model_dir))
    model.lm.load_state_dict({k[len("model.lm."):]: v for k, v in prefixed.items()})

    # Sanity: SID rows must differ from fresh init → confirm overlay landed.
    with torch.no_grad():
        emb = model.lm.get_input_embeddings().weight
        print(
            f"[export] embed_tokens: shape={tuple(emb.shape)} "
            f"dtype={emb.dtype} mean_abs={emb.abs().mean().item():.6f}"
        )

    os.makedirs(args.export_dir, exist_ok=True)
    print(f"[export] save_pretrained -> {args.export_dir}")
    model.lm.save_pretrained(args.export_dir)

    # Save the EXTENDED tokenizer (TER layout: C0 at len(base tokenizer),
    # no [SEP]) so downstream generation maps SID atoms identically.
    tokenizer = AutoTokenizer.from_pretrained(grl_cfg.hf_model_id, use_fast=True)
    base = len(tokenizer)
    tokenizer.add_tokens([f"C{i}" for i in range(sum(grl_cfg.codebook))])
    assert tokenizer.convert_tokens_to_ids("C0") == base
    tokenizer.save_pretrained(args.export_dir)
    with open(os.path.join(args.export_dir, "TER_EXPORT_INFO.txt"), "w") as f:
        f.write(
            f"source_checkpoint={args.checkpoint_path}\n"
            f"pipeline_config={args.pipeline_config_path}\n"
            f"sid_base_token_id={base}\n"
            f"codebook={list(grl_cfg.codebook)}\n"
            "note=C atoms appended directly after base vocab (NO [SEP]); "
            "token_id = base + (sid - 1) for 1-indexed SIDs / base + k for C{k}.\n"
        )
    print(f"[export] done; sid_base_token_id={base}")
    return 0


if __name__ == "__main__":
    main()
