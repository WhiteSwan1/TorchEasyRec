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

"""HuggingFace export for HF-backed models.

Kept out of ``export_util`` so ``checkpoint_util`` can call it without a
circular import.
"""

import json
import os
import shutil
import tempfile
from typing import Any, Dict, List, Optional, Set

import torch
from safetensors.torch import save_file
from torch import nn

from tzrec.constant import HF_EXPORT_META_FILENAME
from tzrec.features.feature import BaseFeature
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.types import CompiledPrompt
from tzrec.protos.pipeline_pb2 import EasyRecConfig
from tzrec.utils import checkpoint_util
from tzrec.utils.filesystem_util import url_to_fs
from tzrec.utils.logging_util import logger

SERVING_ARCH = "GenRecForCausalLM"
SERVING_MODEL_TYPE = "genrec"

_HF_ASSET_FILES = (
    "generation_config.json",
    "tokenizer.json",
    "tokenizer_config.json",
    "vocab.json",
    "merges.txt",
    "added_tokens.json",
    "special_tokens_map.json",
)

_SID_ABI_META_KEY = "sid_abi"
_SID_ABI_VERSION = 1


def build_sid_abi(compiled_prompt: CompiledPrompt) -> Dict[str, Any]:
    """Build the canonical ordered SID vocabulary ABI for a compiled prompt.

    Args:
        compiled_prompt: prompt whose tokenizer layout is being checkpointed.

    Returns:
        A JSON-serializable ABI dictionary whose list order is significant.
    """
    spaces = [
        {
            "name": space.name,
            "codebook": list(space.codebook),
            "num_levels": space.num_levels,
            "token_format": space.token_format,
            "manifest_sha256": space.manifest_sha256,
            "base_vocab_size": space.base_vocab_size,
            "level_offsets": list(space.level_offsets),
            "band_lo": list(space.band_lo),
            "band_hi": list(space.band_hi),
        }
        for space in compiled_prompt.sid_spaces
    ]
    return {
        "version": _SID_ABI_VERSION,
        "sid_spaces": spaces,
        "target_sid_space_index": compiled_prompt.target_sid_space_index,
        "target_vocab_size": compiled_prompt.target_vocab_size,
        "sentinel_token_id": compiled_prompt.sentinel_token_id,
        "eos_token_id": compiled_prompt.eos_token_id,
        "pad_token_id": compiled_prompt.pad_token_id,
        "tokenizer_sha256": compiled_prompt.tokenizer_sha256,
    }


def validate_checkpoint_sid_abi(
    checkpoint_path: str, compiled_prompt: CompiledPrompt
) -> None:
    """Require a checkpoint SID ABI to exactly match the current prompt.

    Args:
        checkpoint_path: checkpoint directory containing HF export metadata.
        compiled_prompt: prompt compiled from the configuration being restored.

    Raises:
        RuntimeError: if ABI metadata is absent, malformed, or different.
    """
    meta_path = os.path.join(checkpoint_path, HF_EXPORT_META_FILENAME)
    if not os.path.exists(meta_path):
        raise RuntimeError(
            f"checkpoint [{checkpoint_path}] has no SID ABI metadata; it cannot "
            "be restored by the multi-SID runtime."
        )
    try:
        with open(meta_path, "r") as f:
            checkpoint_meta = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        raise RuntimeError(
            f"checkpoint [{checkpoint_path}] has unreadable SID ABI metadata."
        ) from e

    if not isinstance(checkpoint_meta, dict):
        raise RuntimeError(
            f"checkpoint [{checkpoint_path}] has malformed SID ABI metadata."
        )
    if _SID_ABI_META_KEY not in checkpoint_meta:
        raise RuntimeError(
            f"checkpoint [{checkpoint_path}] has no SID ABI metadata; it cannot "
            "be restored by the multi-SID runtime."
        )
    checkpoint_abi = checkpoint_meta[_SID_ABI_META_KEY]
    current_abi = build_sid_abi(compiled_prompt)
    checkpoint_json = json.dumps(
        checkpoint_abi, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    current_json = json.dumps(
        current_abi, ensure_ascii=False, separators=(",", ":"), sort_keys=True
    )
    if checkpoint_json != current_json:
        raise RuntimeError(
            "SID checkpoint ABI mismatch; vocabulary rows cannot be restored "
            "without an explicit migration. "
            f"checkpoint={json.dumps(checkpoint_abi, sort_keys=True)}; "
            f"current={json.dumps(current_abi, sort_keys=True)}"
        )


def write_hf_assets(wrapped_model: nn.Module, save_dir: str) -> None:
    """Write HF config, optional tokenizer, and export metadata beside a checkpoint.

    The backbone's FQN prefix and canonical SID ABI go into
    ``hf_export_meta.json`` so restore and export can reject incompatible
    vocabulary layouts. Rank 0 only.
    """
    if int(os.environ.get("RANK", 0)) != 0:
        return
    inner = checkpoint_util.unwrap_to(wrapped_model, "hf_backbone")
    if inner is None:
        return
    os.makedirs(save_dir, exist_ok=True)

    backbone = inner.hf_backbone()
    backbone.config.save_pretrained(save_dir)
    gen_cfg = getattr(backbone, "generation_config", None)
    if gen_cfg is not None:
        gen_cfg.save_pretrained(save_dir)
    tokenizer = getattr(inner, "hf_tokenizer", None)
    if tokenizer is not None:
        tokenizer().save_pretrained(save_dir)

    # named_modules() FQNs carry the DMP prefix that state_dict() strips.
    raw_prefix = next(
        (n for n, m in wrapped_model.named_modules() if m is backbone), ""
    )
    prefix = checkpoint_util._strip_dmp_prefix(raw_prefix)
    meta: Dict[str, Any] = {
        "backbone_state_dict_prefix": prefix + ("." if prefix else "")
    }
    compiled_prompt = getattr(inner, "compiled_prompt", None)
    if compiled_prompt is not None:
        meta[_SID_ABI_META_KEY] = build_sid_abi(compiled_prompt)
    with open(os.path.join(save_dir, HF_EXPORT_META_FILENAME), "w") as f:
        json.dump(meta, f, indent=2)


def dcp_to_hf(ckpt_dir: str, out_dir: str) -> Dict[str, Any]:
    """Convert a checkpoint with co-located HF assets to a ``from_pretrained`` dir.

    Keys that do not map 1:1 onto the co-located ``config.json`` raise rather
    than write a partial model. The config itself is not written: the caller
    composes the one ``config.json`` the export carries.

    Returns:
        The backbone config as ``config.json`` would hold it.
    """
    from torch.distributed.checkpoint.state_dict_loader import (
        _load_state_dict_from_keys,
        _storage_setup,
    )
    from transformers import AutoConfig, AutoModelForCausalLM

    model_ckpt_path = os.path.join(ckpt_dir, "model")
    if not os.path.exists(model_ckpt_path):
        raise RuntimeError(f"dcp_to_hf: model DCP dir [{model_ckpt_path}] not exists.")

    meta_path = os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME)
    prefix: Optional[str] = None
    if os.path.exists(meta_path):
        with open(meta_path, "r") as f:
            prefix = json.load(f).get("backbone_state_dict_prefix")

    cfg = AutoConfig.from_pretrained(ckpt_dir)
    with torch.device("meta"):
        empty = AutoModelForCausalLM.from_config(cfg)
    target_keys: Set[str] = set(empty.state_dict().keys())
    tied_keys: Set[str] = set(getattr(empty, "_tied_weights_keys", None) or [])
    del empty

    # the mapping is decided on names alone, so the load below reads the
    # backbone and not the sparse tables beside it
    reader = _storage_setup(None, model_ckpt_path, reader=True)
    ckpt_keys: Set[str] = set(reader.read_metadata().state_dict_metadata)

    def _strip_recorded_prefix() -> Optional[Dict[str, str]]:
        """Strip the recorded prefix; None unless it yields an EXACT match."""
        if not prefix:
            return None
        out = {k[len(prefix) :]: k for k in ckpt_keys if k.startswith(prefix)}
        return out if set(out) == target_keys else None

    def _derive_by_suffix() -> Optional[Dict[str, str]]:
        """Each target key is a unique suffix of exactly one DCP key; None if not."""
        out: Dict[str, str] = {}
        for tk in target_keys:
            matches = [k for k in ckpt_keys if k == tk or k.endswith("." + tk)]
            if len(matches) != 1:
                return None
            out[tk] = matches[0]
        return out

    key_map = _strip_recorded_prefix()
    if key_map is None:
        if prefix:
            logger.warning(
                f"dcp_to_hf: recorded prefix [{prefix}] did not map exactly onto "
                "the architecture; deriving the backbone prefix by suffix-matching."
            )
        key_map = _derive_by_suffix()

    if key_map is None:
        raise RuntimeError(
            "dcp_to_hf: cannot map the DCP state dict onto the backbone "
            f"architecture (recorded prefix={prefix!r}). Wanted "
            f"{len(target_keys)} keys like {sorted(target_keys)[:3]}; the "
            f"checkpoint holds {len(ckpt_keys)} like {sorted(ckpt_keys)[:3]}. "
            "Refusing to write a partially-loaded HF model."
        )

    # non-distributed => full tensors locally
    raw_state: Dict[str, torch.Tensor] = _load_state_dict_from_keys(
        set(key_map.values()), checkpoint_id=model_ckpt_path
    )
    mapped = {tk: raw_state[ck] for tk, ck in key_map.items()}

    # from_pretrained re-ties them.
    if getattr(cfg, "tie_word_embeddings", False):
        mapped = {k: v for k, v in mapped.items() if k not in tied_keys}
    mapped = {k: v.contiguous() for k, v in mapped.items()}  # save_file rejects views
    del raw_state
    os.makedirs(out_dir, exist_ok=True)
    save_file(mapped, os.path.join(out_dir, "model.safetensors"))

    for fname in _HF_ASSET_FILES:
        src = os.path.join(ckpt_dir, fname)
        if os.path.exists(src):
            shutil.copy(src, os.path.join(out_dir, fname))
    return json.loads(cfg.to_json_string())


def export_hf_assets(
    pipeline_config: EasyRecConfig,
    features: List[BaseFeature],
    checkpoint_path: str,
    export_dir: str,
) -> None:
    """Write what an LLM engine reads beside the scripted front-end.

    The HuggingFace weights, the extended tokenizer and one composite
    ``config.json``: the backbone sits under ``text_config``, which names it to
    a runtime that composes an arbitrary causal LM behind one registered
    architecture and treats a projected prompt slot as a second modality. A
    remote ``export_dir`` is written locally and uploaded, as ``export_model``
    does for its own files.

    Args:
        pipeline_config: the pipeline being exported.
        features: the created features the prompt compiles against.
        checkpoint_path: the checkpoint the weights come from.
        export_dir: the export directory.
    """
    fs, local_dir = url_to_fs(export_dir)
    if fs is not None:
        local_dir = tempfile.mkdtemp()
    try:
        compiled = compile_prompt(
            pipeline_config.prompt_config,
            features,
            list(pipeline_config.data_config.label_fields),
        )
        validate_checkpoint_sid_abi(checkpoint_path, compiled)
        backbone = dcp_to_hf(checkpoint_path, local_dir)
        exported_prompt = compile_prompt(
            pipeline_config.prompt_config,
            features,
            list(pipeline_config.data_config.label_fields),
            tokenizer_dir=local_dir,
        )
        if exported_prompt != compiled:
            raise RuntimeError(
                "prompt compilation changed while exporting the checkpoint."
            )
        backbone_vocab_size = int(backbone.get("vocab_size", -1))
        if backbone_vocab_size != compiled.target_vocab_size:
            raise RuntimeError(
                f"checkpoint backbone vocab_size [{backbone_vocab_size}] does not "
                f"match the SID ABI target_vocab_size "
                f"[{compiled.target_vocab_size}]."
            )
        sid_abi = build_sid_abi(compiled)
        composite: Dict[str, Any] = {
            "architectures": [SERVING_ARCH],
            "model_type": SERVING_MODEL_TYPE,
            "text_config": backbone,
            "eos_token_id": compiled.eos_token_id,
            "pad_token_id": compiled.pad_token_id,
            "sentinel_token_id": compiled.sentinel_token_id,
            "sid_spaces": sid_abi["sid_spaces"],
            "target_sid_space_index": compiled.target_sid_space_index,
            "target_vocab_size": compiled.target_vocab_size,
            _SID_ABI_META_KEY: sid_abi,
        }
        # a runtime that reads only the outer config still needs to size its cache
        for key in ("vocab_size", "hidden_size", "num_hidden_layers", "torch_dtype"):
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
