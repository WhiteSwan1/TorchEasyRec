# Copyright (c) 2026, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""HuggingFace export for HF-backed models."""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, Optional, Set, Tuple

import torch
from safetensors.torch import save_file
from transformers import GenerationConfig, PretrainedConfig

from tzrec.prompt.compile import save_tokenizer_dir
from tzrec.prompt.types import CompiledPrompt, ResolvedSidSpace
from tzrec.utils.filesystem_util import url_to_fs

SERVING_ARCH = "GenRecForCausalLM"
SERVING_MODEL_TYPE = "genrec"


def convert_qwen_mcore_state(
    state: Dict[str, torch.Tensor],
    config: PretrainedConfig,
    to_mcore: bool,
) -> Dict[str, torch.Tensor]:
    """Convert unsharded dense Qwen2/2.5/3 weights between HF and MCore.

    Args:
        state: backbone state without the GenRec model prefix.
        config: HF architecture, including the resized vocabulary.
        to_mcore: pack HF projections when true, unpack them when false.

    Returns:
        Converted weights; Transformer Engine runtime extra state is excluded.
    """
    mapping: Dict[str, Tuple[str, ...]] = {
        "embedding.word_embeddings.weight": ("model.embed_tokens.weight",),
        "output_layer.weight": ("lm_head.weight",),
        "decoder.final_layernorm.weight": ("model.norm.weight",),
    }
    for layer in range(config.num_hidden_layers):
        mcore = f"decoder.layers.{layer}."
        hf = f"model.layers.{layer}."
        for mcore_key, hf_key in (
            ("input_layernorm.weight", "input_layernorm.weight"),
            ("pre_mlp_layernorm.weight", "post_attention_layernorm.weight"),
            ("self_attention.linear_proj.weight", "self_attn.o_proj.weight"),
            ("mlp.linear_fc2.weight", "mlp.down_proj.weight"),
        ):
            mapping[mcore + mcore_key] = (hf + hf_key,)
        if config.model_type == "qwen3":
            for component in ("q", "k"):
                mapping[mcore + f"self_attention.{component}_layernorm.weight"] = (
                    hf + f"self_attn.{component}_norm.weight",
                )
        suffixes = (
            ("weight", "bias")
            if getattr(config, "attention_bias", config.model_type == "qwen2")
            else ("weight",)
        )
        for suffix in suffixes:
            mapping[mcore + "self_attention.linear_qkv." + suffix] = tuple(
                hf + f"self_attn.{projection}_proj.{suffix}"
                for projection in ("q", "k", "v")
            )
        mapping[mcore + "mlp.linear_fc1.weight"] = (
            hf + "mlp.gate_proj.weight",
            hf + "mlp.up_proj.weight",
        )

    groups = config.num_key_value_heads
    head_dim = getattr(
        config, "head_dim", config.hidden_size // config.num_attention_heads
    )
    query_width = config.num_attention_heads // groups * head_dim
    result = {}
    for mcore_key, hf_keys in mapping.items():
        if to_mcore:
            tensors = [state[key] for key in hf_keys]
            if len(tensors) == 3:
                trailing = tensors[0].shape[1:]
                tensors = [tensor.reshape(groups, -1, *trailing) for tensor in tensors]
                result[mcore_key] = torch.cat(tensors, dim=1).reshape(-1, *trailing)
            elif len(tensors) == 2:
                result[mcore_key] = torch.cat(tensors, dim=0)
            else:
                result[mcore_key] = tensors[0]
        else:
            tensor = state[mcore_key]
            if len(hf_keys) == 3:
                trailing = tensor.shape[1:]
                pieces = tensor.reshape(groups, -1, *trailing).split(
                    (query_width, head_dim, head_dim), dim=1
                )
                tensors = [piece.reshape(-1, *trailing) for piece in pieces]
            elif len(hf_keys) == 2:
                tensors = tensor.chunk(2, dim=0)
            else:
                tensors = [tensor]
            result.update(zip(hf_keys, tensors))
    return result


def dcp_to_hf(ckpt_dir: str, out_dir: str, config: PretrainedConfig) -> None:
    """Write the backbone of a checkpoint as a ``from_pretrained`` safetensors file.

    The backbone's keys are found by suffix, under whatever prefix the training
    wrapper gave them. Keys that do not map 1:1 onto ``config`` raise rather
    than write a partial model. No config is written: the caller composes the
    one ``config.json`` the export carries.

    Args:
        ckpt_dir: the checkpoint the weights come from.
        out_dir: where ``model.safetensors`` is written.
        config: the backbone config, read off the live model.
    """
    from torch.distributed.checkpoint.state_dict_loader import (
        _load_state_dict_from_keys,
        _storage_setup,
    )
    from transformers import AutoModelForCausalLM

    model_ckpt_path = os.path.join(ckpt_dir, "model")
    if not os.path.exists(model_ckpt_path):
        raise RuntimeError(f"dcp_to_hf: model DCP dir [{model_ckpt_path}] not exists.")

    with torch.device("meta"):
        empty = AutoModelForCausalLM.from_config(config)
    target_shapes = {k: tuple(v.shape) for k, v in empty.state_dict().items()}
    target_keys: Set[str] = set(target_shapes)
    tied_keys: Set[str] = set(getattr(empty, "_tied_weights_keys", None) or [])

    # the mapping is decided on names alone, so the load below reads the
    # backbone and not the sparse tables beside it
    reader = _storage_setup(None, model_ckpt_path, reader=True)
    metadata = reader.read_metadata().state_dict_metadata
    ckpt_keys: Set[str] = set(metadata)
    is_mcore = any(
        key.endswith("lm.embedding.word_embeddings.weight") for key in ckpt_keys
    )
    if is_mcore:
        target_shapes = {
            key: tuple(value.shape)
            for key, value in convert_qwen_mcore_state(
                empty.state_dict(), config, to_mcore=True
            ).items()
        }
        target_keys = set(target_shapes)
    del empty

    def _derive_by_suffix() -> Optional[Dict[str, str]]:
        """Each target key is a unique suffix of exactly one DCP key; None if not."""
        out: Dict[str, str] = {}
        for tk in target_keys:
            matches = [k for k in ckpt_keys if k == tk or k.endswith("." + tk)]
            if len(matches) != 1:
                return None
            out[tk] = matches[0]
        return out

    key_map = _derive_by_suffix()
    # one backbone under one prefix: a look-alike key elsewhere cannot stand in
    if key_map is not None and len({ck[: -len(tk)] for tk, ck in key_map.items()}) != 1:
        key_map = None
    if key_map is None:
        raise RuntimeError(
            "dcp_to_hf: cannot map the DCP state dict onto the backbone "
            f"architecture. Wanted {len(target_keys)} keys like "
            f"{sorted(target_keys)[:3]}; the checkpoint holds {len(ckpt_keys)} "
            f"like {sorted(ckpt_keys)[:3]}. Refusing to write a partially-loaded "
            "HF model."
        )
    # names alone would let a resized vocabulary or a swapped backbone through
    drifted = [
        (tk, target_shapes[tk], tuple(metadata[ck].size))
        for tk, ck in key_map.items()
        if tuple(metadata[ck].size) != target_shapes[tk]
    ]
    if drifted:
        raise RuntimeError(
            "dcp_to_hf: the checkpoint's tensors do not fit the exported config, "
            f"e.g. {drifted[:3]} as (key, config shape, checkpoint shape); was it "
            "trained under a different sid_space or backbone? Refusing to write "
            "a partially-loaded HF model."
        )

    # non-distributed => full tensors locally
    raw_state: Dict[str, torch.Tensor] = _load_state_dict_from_keys(
        set(key_map.values()), checkpoint_id=model_ckpt_path
    )
    mapped = {tk: raw_state[ck] for tk, ck in key_map.items()}
    if is_mcore:
        mapped = convert_qwen_mcore_state(mapped, config, to_mcore=False)

    # from_pretrained re-ties them.
    if getattr(config, "tie_word_embeddings", False):
        mapped = {k: v for k, v in mapped.items() if k not in tied_keys}
    mapped = {k: v.contiguous() for k, v in mapped.items()}  # save_file rejects views
    del raw_state
    os.makedirs(out_dir, exist_ok=True)
    save_file(mapped, os.path.join(out_dir, "model.safetensors"))


def sid_space_config(space: ResolvedSidSpace) -> Dict[str, Any]:
    """What a serving engine needs to decode against a SID bundle.

    A bundle stores each SID as offset codes (``level_offsets[l] + code``); the
    model generates token ids ``base_vocab_size + offset``. The codebook gives
    the levels, and ``bundle_uuid`` -- present when the prompt named a manifest
    -- is the bundle the model was compiled against, so an engine can refuse a
    bundle whose SIDs mean other items.

    Args:
        space: the resolved SID space the model was compiled with.

    Returns:
        dict: the ``sid_space`` block of the exported ``config.json``.
    """
    config: Dict[str, Any] = {
        "codebook": list(space.codebook),
        "base_vocab_size": space.base_vocab_size,
        "token_format": space.token_format,
    }
    if space.bundle_uuid is not None:
        config["bundle_uuid"] = space.bundle_uuid
    return config


def export_hf_assets(
    checkpoint_path: str,
    export_dir: str,
    compiled_prompt: CompiledPrompt,
    backbone_config: PretrainedConfig,
    generation_config: Optional[GenerationConfig],
) -> None:
    """Write what an LLM engine reads beside the scripted front-end.

    The HuggingFace weights, the extended tokenizer and one composite
    ``config.json``: the backbone sits under ``text_config``, which names it to
    a runtime that composes an arbitrary causal LM behind one registered
    architecture and treats a projected prompt slot as a second modality. A
    remote ``export_dir`` is written locally and uploaded, as ``export_model``
    does for its own files.

    Args:
        checkpoint_path: the checkpoint the weights come from.
        export_dir: the export directory.
        compiled_prompt: the prompt the model was built on; its tokenizer and
            SID space are what the engine decodes with.
        backbone_config: the backbone config, read off the live model.
        generation_config: the backbone's generation config, when it has one.
    """
    fs, local_dir = url_to_fs(export_dir)
    if fs is not None:
        local_dir = tempfile.mkdtemp()
    try:
        dcp_to_hf(checkpoint_path, local_dir, backbone_config)
        if generation_config is not None:
            generation_config.save_pretrained(local_dir)
        backbone = json.loads(backbone_config.to_json_string())
        save_tokenizer_dir(compiled_prompt, local_dir)
        composite: Dict[str, Any] = {
            "architectures": [SERVING_ARCH],
            "model_type": SERVING_MODEL_TYPE,
            "text_config": backbone,
            "eos_token_id": compiled_prompt.sid_space.eos_token_id,
            "pad_token_id": compiled_prompt.sid_space.pad_token_id,
            "sid_space": sid_space_config(compiled_prompt.sid_space),
        }
        # a runtime that reads only the outer config still needs to size its cache
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "dtype"):
            if key in backbone:
                composite[key] = backbone[key]
        with open(os.path.join(local_dir, "config.json"), "w") as f:
            json.dump(composite, f, indent=2)
        if fs is not None:
            fs.upload(
                local_dir, export_dir, recursive=True, file_thread_num=os.cpu_count()
            )
    finally:
        # the staging dir holds the full LM weights; drop it however this ends
        if fs is not None:
            shutil.rmtree(local_dir, ignore_errors=True)
