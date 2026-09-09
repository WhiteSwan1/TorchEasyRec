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

import json
import os
import shutil
import threading
import unittest
from dataclasses import replace
from unittest import mock

import torch
from safetensors.torch import load_file
from torch import nn

from tzrec.constant import HF_EXPORT_META_FILENAME
from tzrec.prompt.types import (
    CompiledPrompt,
    ProjectionPlan,
    PromptPlan,
    ResolvedSidSpace,
)
from tzrec.utils.checkpoint_util import restore_model, save_model, unwrap_to
from tzrec.utils.hf_export_util import (
    build_sid_abi,
    dcp_to_hf,
    validate_checkpoint_sid_abi,
    write_hf_assets,
)
from tzrec.utils.test_util import create_tiny_causal_lm, make_test_dir


def _tied_lm():
    """The tied-head backbone every case here needs; dcp_to_hf must drop the tie."""
    return create_tiny_causal_lm(64, tie_word_embeddings=True)


def _compiled_prompt(
    label_token_format: str = "<|label_sid_{i}|>",
) -> CompiledPrompt:
    """Build the minimal two-space prompt metadata needed by ABI tests."""
    spaces = (
        ResolvedSidSpace(
            name="history_sid",
            codebook=(3, 3, 3),
            num_levels=3,
            token_format="<|history_sid_{i}|>",
            manifest_sha256=None,
            base_vocab_size=64,
            level_offsets=(0, 3, 6),
            band_lo=(64, 67, 70),
            band_hi=(66, 69, 72),
        ),
        ResolvedSidSpace(
            name="label_sid",
            codebook=(3, 3, 2),
            num_levels=3,
            token_format=label_token_format,
            manifest_sha256="label-manifest-sha256",
            base_vocab_size=73,
            level_offsets=(0, 3, 6),
            band_lo=(73, 76, 79),
            band_hi=(75, 78, 80),
        ),
    )
    return CompiledPrompt(
        sid_spaces=spaces,
        target_sid_space_index=1,
        target_vocab_size=128,
        sentinel_token_id=81,
        eos_token_id=1,
        pad_token_id=0,
        tokenizer_sha256="extended-tokenizer-sha256",
        prompt_plan=PromptPlan(
            segments=(),
            response_segments=(),
            max_length=0,
            max_total_length=0,
            max_holes=0,
            logits_suffix_len=0,
            static_prefix_len=0,
            projected_slots=(),
        ),
        projection_plan=ProjectionPlan(projections={}, slot_to_module={}),
    )


class _FakeTokenizer:
    """Writes the two tokenizer asset files `write_hf_assets` copies."""

    def save_pretrained(self, save_dir):
        for name in ("tokenizer.json", "tokenizer_config.json"):
            with open(os.path.join(save_dir, name), "w") as f:
                f.write("{}")


class _GenRec(nn.Module):
    """Stand-in for an HF-backed model exposing the optional tokenizer protocol."""

    def __init__(self, lm, compiled_prompt=None):
        super().__init__()
        self.lm = lm
        if compiled_prompt is not None:
            self.lm.resize_token_embeddings(
                compiled_prompt.target_vocab_size, mean_resizing=False
            )
        self.other = nn.Linear(4, 4)
        self.compiled_prompt = compiled_prompt

    def hf_backbone(self):
        return self.lm

    def hf_tokenizer(self):
        return _FakeTokenizer()


class _TrainWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model


class _DmpLike(nn.Module):
    def __init__(self, module):
        super().__init__()
        self.module = module


class HfExportUtilTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        # the asset writers are rank-0-gated; pin it without leaking the value.
        patcher = mock.patch.dict(os.environ, {"RANK": "0"})
        patcher.start()
        self.addCleanup(patcher.stop)

    def tearDown(self) -> None:
        shutil.rmtree(self.test_dir, ignore_errors=True)

    def test_unwrap_terminates_on_a_wrapper_cycle(self) -> None:
        """A .model/.module cycle must return None, not spin.

        Every checkpoint save of every model walks this, so an unbounded loop
        here would hang save() and strand the peers waiting on the collective
        that follows. Run on a thread so a regression fails the test instead of
        hanging the suite.
        """
        a, b = nn.Linear(4, 4), nn.Linear(4, 4)
        object.__setattr__(a, "model", b)
        object.__setattr__(b, "model", a)
        out = []
        t = threading.Thread(target=lambda: out.append(unwrap_to(a, "hf_backbone")))
        t.daemon = True
        t.start()
        t.join(timeout=5)
        self.assertFalse(t.is_alive(), "unwrap_to did not terminate")
        self.assertEqual(out, [None])

    def test_write_hf_assets_noop_for_non_hf_model(self) -> None:
        save_dir = os.path.join(self.test_dir, "plain")
        write_hf_assets(_TrainWrapper(nn.Linear(4, 4)), save_dir)
        self.assertFalse(os.path.exists(save_dir))

    def _save_ckpt(self, wrapped):
        ckpt_dir = os.path.join(self.test_dir, "model.ckpt-1")
        with mock.patch("tzrec.utils.checkpoint_util.has_dynamicemb", False):
            save_model(ckpt_dir, wrapped)
        write_hf_assets(wrapped, ckpt_dir)
        return ckpt_dir

    def _write_sid_abi(self, compiled):
        ckpt_dir = os.path.join(self.test_dir, "abi.ckpt")
        os.makedirs(ckpt_dir)
        with open(os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME), "w") as f:
            json.dump({"sid_abi": build_sid_abi(compiled)}, f)
        return ckpt_dir

    def test_write_hf_assets_records_state_dict_prefix(self) -> None:
        lm = _tied_lm()
        wrapped = _TrainWrapper(_GenRec(lm))
        ckpt_dir = self._save_ckpt(wrapped)
        for name in ("config.json", "tokenizer.json", HF_EXPORT_META_FILENAME):
            self.assertTrue(os.path.exists(os.path.join(ckpt_dir, name)), name)
        with open(os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME)) as f:
            prefix = json.load(f)["backbone_state_dict_prefix"]
        self.assertEqual(prefix, "model.lm.")
        # the prefix must reconstruct the exact FQNs save_model wrote
        saved = set(wrapped.state_dict())
        self.assertTrue(all(prefix + k in saved for k in lm.state_dict()))

    def test_write_hf_assets_records_and_validates_sid_abi(self) -> None:
        compiled = _compiled_prompt()
        ckpt_dir = self._save_ckpt(
            _TrainWrapper(_GenRec(_tied_lm(), compiled_prompt=compiled))
        )
        with open(os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME)) as f:
            meta = json.load(f)
        self.assertEqual(meta["sid_abi"], build_sid_abi(compiled))
        validate_checkpoint_sid_abi(ckpt_dir, compiled)

    def test_validate_checkpoint_sid_abi_rejects_prefix_change(self) -> None:
        ckpt_dir = self._write_sid_abi(_compiled_prompt())
        changed = _compiled_prompt(label_token_format="<|collision_sid_{i}|>")
        with self.assertRaisesRegex(RuntimeError, "SID checkpoint ABI mismatch"):
            validate_checkpoint_sid_abi(ckpt_dir, changed)

    def test_validate_checkpoint_sid_abi_rejects_space_reordering(self) -> None:
        compiled = _compiled_prompt()
        ckpt_dir = self._write_sid_abi(compiled)
        changed = replace(
            compiled,
            sid_spaces=tuple(reversed(compiled.sid_spaces)),
            target_sid_space_index=0,
        )
        with self.assertRaisesRegex(RuntimeError, "SID checkpoint ABI mismatch"):
            validate_checkpoint_sid_abi(ckpt_dir, changed)

    def test_validate_checkpoint_sid_abi_rejects_tokenizer_change(self) -> None:
        compiled = _compiled_prompt()
        ckpt_dir = self._write_sid_abi(compiled)
        changed = replace(compiled, tokenizer_sha256="different-tokenizer-sha256")
        with self.assertRaisesRegex(RuntimeError, "SID checkpoint ABI mismatch"):
            validate_checkpoint_sid_abi(ckpt_dir, changed)

    def test_validate_checkpoint_sid_abi_rejects_manifest_change(self) -> None:
        compiled = _compiled_prompt()
        ckpt_dir = self._write_sid_abi(compiled)
        changed_target = replace(
            compiled.sid_spaces[1], manifest_sha256="different-manifest-sha256"
        )
        changed = replace(
            compiled,
            sid_spaces=(compiled.sid_spaces[0], changed_target),
        )
        with self.assertRaisesRegex(RuntimeError, "SID checkpoint ABI mismatch"):
            validate_checkpoint_sid_abi(ckpt_dir, changed)

    def test_validate_checkpoint_sid_abi_rejects_json_type_change(self) -> None:
        compiled = _compiled_prompt()
        ckpt_dir = self._write_sid_abi(compiled)
        meta_path = os.path.join(ckpt_dir, HF_EXPORT_META_FILENAME)
        with open(meta_path) as f:
            meta = json.load(f)
        meta["sid_abi"]["version"] = True
        with open(meta_path, "w") as f:
            json.dump(meta, f)
        with self.assertRaisesRegex(RuntimeError, "SID checkpoint ABI mismatch"):
            validate_checkpoint_sid_abi(ckpt_dir, compiled)

    def test_restore_model_requires_sid_abi_metadata(self) -> None:
        ckpt_dir = os.path.join(self.test_dir, "legacy.ckpt")
        os.makedirs(ckpt_dir)
        model = _TrainWrapper(_GenRec(_tied_lm(), _compiled_prompt()))
        with self.assertRaisesRegex(RuntimeError, "has no SID ABI metadata"):
            restore_model(ckpt_dir, model)

    def test_dcp_to_hf_round_trip_drops_tied_head(self) -> None:
        from transformers import AutoModelForCausalLM

        lm = _tied_lm()
        ckpt_dir = self._save_ckpt(_DmpLike(_TrainWrapper(_GenRec(lm))))
        out_dir = os.path.join(self.test_dir, "hf_out")
        config = dcp_to_hf(ckpt_dir, out_dir)
        # the caller composes config.json; here the backbone's own is enough
        self.assertEqual(config["model_type"], lm.config.model_type)
        with open(os.path.join(out_dir, "config.json"), "w") as f:
            json.dump(config, f)

        st = load_file(os.path.join(out_dir, "model.safetensors"))
        self.assertNotIn("lm_head.weight", st)
        self.assertIn("model.embed_tokens.weight", st)
        back = AutoModelForCausalLM.from_pretrained(out_dir)
        self.assertEqual(
            back.lm_head.weight.data_ptr(), back.model.embed_tokens.weight.data_ptr()
        )
        for k, v in lm.state_dict().items():
            self.assertTrue(torch.equal(back.state_dict()[k], v), k)

    def test_dcp_to_hf_loads_only_the_backbone_keys(self) -> None:
        """The rest of a genrec checkpoint is the sparse tables; never read them."""
        from torch.distributed.checkpoint import state_dict_loader

        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(_tied_lm())))
        original = state_dict_loader._load_state_dict_from_keys
        requested = []

        def _spy(keys=None, **kwargs):
            requested.append(keys)
            return original(keys, **kwargs)

        with mock.patch.object(state_dict_loader, "_load_state_dict_from_keys", _spy):
            dcp_to_hf(ckpt_dir, os.path.join(self.test_dir, "hf_out_keys"))
        self.assertEqual(len(requested), 1)
        self.assertIsNotNone(requested[0])
        self.assertFalse([k for k in requested[0] if ".other." in k])

    def test_dcp_to_hf_refuses_a_mismatched_architecture(self) -> None:
        ckpt_dir = self._save_ckpt(_TrainWrapper(_GenRec(_tied_lm())))
        # widen the recorded architecture so the checkpoint can no longer fill it
        cfg_path = os.path.join(ckpt_dir, "config.json")
        with open(cfg_path) as f:
            cfg = json.load(f)
        cfg["num_hidden_layers"] = 4
        cfg.pop("layer_types", None)
        with open(cfg_path, "w") as f:
            json.dump(cfg, f)
        with self.assertRaisesRegex(RuntimeError, "Refusing to write"):
            dcp_to_hf(ckpt_dir, os.path.join(self.test_dir, "hf_out_bad"))

    def test_dcp_to_hf_missing_dcp_dir(self) -> None:
        empty = os.path.join(self.test_dir, "no_dcp")
        os.makedirs(empty, exist_ok=True)
        with self.assertRaisesRegex(RuntimeError, "not exists"):
            dcp_to_hf(empty, os.path.join(self.test_dir, "hf_out_missing"))


if __name__ == "__main__":
    unittest.main()
