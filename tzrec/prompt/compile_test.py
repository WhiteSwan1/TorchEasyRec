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

import hashlib
import json
import os
import unittest

from google.protobuf import text_format
from tokenizers import Tokenizer

from tzrec.features.feature import FgMode, create_features
from tzrec.prompt.compile import compile_prompt
from tzrec.prompt.types import FillMode, SlotSeg, Static, WidthKind
from tzrec.protos import feature_pb2
from tzrec.protos.prompt_pb2 import PromptConfig
from tzrec.utils.test_util import create_genrec_test_tokenizer, make_test_dir

_WORDS = ["History", "Profile", "Predict", ":", ".", "Histor0", "<unk>", "<|im_end|>"]


_HIST = 'sequence_raw_feature { feature_name: "hist" expression: "user:hist" }'
_PROF = (
    'sequence_id_feature { feature_name: "prof" expression: "user:prof" '
    "num_buckets: 768 embedding_dim: 16 sequence_length: 4 }"
)
_AGE = 'id_feature { feature_name: "age" expression: "user:age" num_buckets: 8 }'


def _feature(text: str):
    config = feature_pb2.FeatureConfig()
    text_format.Merge(text, config)
    return create_features([config], fg_mode=FgMode.FG_NONE)[0]


class CompilePromptTest(unittest.TestCase):
    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.tok_path = create_genrec_test_tokenizer(
            os.path.join(self.test_dir, "tok.json"), _WORDS
        )

    def _config(self, **kwargs) -> PromptConfig:
        kwargs.setdefault("response", "{{answer}}")
        cfg = PromptConfig(tokenizer_path=self.tok_path, **kwargs)
        return cfg

    def _compile(self, cfg, features):
        return compile_prompt(cfg, features, ["answer"])

    def _add_sid_space(self, cfg, name, codebook=(4,), token_format=None, padding=128):
        space = cfg.sid_space.add(name=name)
        space.codebook.extend(codebook)
        space.vocab_pad_to_multiple_of = padding
        space.token_format = token_format or f"<|{name}_sid_{{i}}|>"
        return space

    def _add_hist_and_answer_spaces(self, cfg, codebook=(4,)):
        self._add_sid_space(cfg, "hist", codebook)
        self._add_sid_space(cfg, "answer", codebook)

    def test_sid_space_resolves_offsets_and_bands(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4, 4))
        tokenizer_dir = os.path.join(self.test_dir, "two_spaces")
        compiled = compile_prompt(
            cfg,
            [_feature(_HIST)],
            ["answer"],
            tokenizer_dir=tokenizer_dir,
        )
        raw_space, target_space = compiled.sid_spaces

        base_vocab_size = raw_space.base_vocab_size
        self.assertEqual(
            [space.name for space in compiled.sid_spaces], ["hist", "answer"]
        )
        self.assertEqual(raw_space.token_format, "<|hist_sid_{i}|>")
        self.assertEqual(target_space.token_format, "<|answer_sid_{i}|>")
        self.assertEqual(raw_space.num_levels, 3)
        self.assertEqual(sum(raw_space.codebook), 12)
        self.assertEqual(raw_space.level_offsets, (0, 4, 8))
        self.assertEqual(
            raw_space.band_lo,
            (base_vocab_size, base_vocab_size + 4, base_vocab_size + 8),
        )
        self.assertEqual(
            raw_space.band_hi,
            (base_vocab_size + 3, base_vocab_size + 7, base_vocab_size + 11),
        )
        self.assertEqual(target_space.base_vocab_size, base_vocab_size + 12)
        self.assertEqual(compiled.target_sid_space_index, 1)
        self.assertIsNone(compiled.sentinel_token_id)
        unpadded_vocab_size = target_space.base_vocab_size + sum(target_space.codebook)
        self.assertEqual(
            compiled.target_vocab_size,
            -(-unpadded_vocab_size // 128) * 128,
        )

        body_seg = next(
            seg for seg in compiled.prompt_plan.segments if isinstance(seg, SlotSeg)
        )
        target_seg = next(
            seg
            for seg in compiled.prompt_plan.response_segments
            if isinstance(seg, SlotSeg)
        )
        self.assertEqual(body_seg.sid_space_index, 0)
        self.assertEqual(target_seg.sid_space_index, 1)

        raw_flat = [1, 5, 9]
        resolved_flat = [1, 5, 8]
        tokenizer = Tokenizer.from_file(os.path.join(tokenizer_dir, "tokenizer.json"))
        self.assertEqual(
            [raw_space.base_vocab_size + value for value in raw_flat],
            [base_vocab_size + 1, base_vocab_size + 5, base_vocab_size + 9],
        )
        self.assertEqual(
            [target_space.base_vocab_size + value for value in resolved_flat],
            [base_vocab_size + 13, base_vocab_size + 17, base_vocab_size + 20],
        )
        self.assertEqual(
            [tokenizer.token_to_id(f"<|hist_sid_{value}|>") for value in raw_flat],
            [base_vocab_size + 1, base_vocab_size + 5, base_vocab_size + 9],
        )
        self.assertEqual(
            [
                tokenizer.token_to_id(f"<|answer_sid_{value}|>")
                for value in resolved_flat
            ],
            [base_vocab_size + 13, base_vocab_size + 17, base_vocab_size + 20],
        )

    def test_inline_needs_no_group_projected_gets_one(self) -> None:
        cfg = self._config(prompt="History : {{hist}} . Profile : {{prof}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4, 4))
        compiled = self._compile(cfg, [_feature(_HIST), _feature(_PROF)])

        by_name = {
            s.name: s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg)
        }
        self.assertIs(by_name["hist"].fill, FillMode.INLINE)
        self.assertIs(by_name["prof"].fill, FillMode.PROJECTED)
        # only the projected slot produces a group, and so a hole
        self.assertEqual(
            [s.name for s in compiled.prompt_plan.projected_slots], ["prof"]
        )
        groups = compiled.projection_plan.feature_groups
        self.assertEqual([g.group_name for g in groups], ["prof"])
        self.assertEqual(list(groups[0].feature_names), ["prof"])
        self.assertEqual(compiled.prompt_plan.max_holes, 4)
        self.assertIsNotNone(compiled.sentinel_token_id)
        target_space = compiled.sid_spaces[compiled.target_sid_space_index]
        self.assertEqual(
            compiled.sentinel_token_id,
            target_space.base_vocab_size + sum(target_space.codebook),
        )

    def test_static_runs_are_woven_between_slots(self) -> None:
        cfg = self._config(prompt="History : {{hist}} . Predict :")
        self._add_hist_and_answer_spaces(cfg)
        compiled = self._compile(cfg, [_feature(_HIST)])
        kinds = [
            "static" if isinstance(s, Static) else s.name
            for s in compiled.prompt_plan.segments
        ]
        self.assertEqual(kinds, ["static", "hist", "static"])
        # the leading run is request-invariant; "History :" is two tokens
        self.assertEqual(compiled.prompt_plan.static_prefix_len, 2)

    def test_scalar_slot_is_one_deep_position(self) -> None:
        cfg = self._config(prompt="Profile : {{age}}")
        self._add_sid_space(cfg, "answer")
        compiled = self._compile(cfg, [_feature(_AGE)])
        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.fill, FillMode.PROJECTED)
        self.assertEqual(seg.output_key, "")
        self.assertIs(seg.width.kind, WidthKind.STATIC)
        self.assertEqual(seg.width.num_positions, 1)

    def test_manifest_mismatch_is_fatal(self) -> None:
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": [8, 8, 8]}, f)
        cfg = self._config(prompt="History : {{hist}}")
        hist_space = self._add_sid_space(cfg, "hist", (4, 4, 4))
        hist_space.manifest_path = manifest
        self._add_sid_space(cfg, "answer", (4, 4, 4))
        with self.assertRaisesRegex(ValueError, "does not match the manifest"):
            self._compile(cfg, [_feature(_HIST)])

    def test_manifest_match_compiles(self) -> None:
        manifest = os.path.join(self.test_dir, "manifest.json")
        with open(manifest, "w") as f:
            json.dump({"codebook": [4, 4, 4]}, f)
        cfg = self._config(prompt="History : {{hist}}")
        hist_space = self._add_sid_space(cfg, "hist", (4, 4, 4))
        hist_space.manifest_path = manifest
        self._add_sid_space(cfg, "answer", (4, 4, 4))
        self.assertEqual(
            self._compile(cfg, [_feature(_HIST)]).sid_spaces[0].manifest_sha256,
            hashlib.sha256(b'{"codebook":[4,4,4]}').hexdigest(),
        )

    def test_rejects_a_mixed_kind_slot(self) -> None:
        cfg = self._config(prompt="X : {{both}}")
        self._add_sid_space(cfg, "answer")
        slot = cfg.slots.add(name="both")
        slot.feature_names.extend(["hist", "age"])
        with self.assertRaisesRegex(ValueError, "mixes sequence and scalar"):
            self._compile(cfg, [_feature(_HIST), _feature(_AGE)])

    def test_rejects_unknown_feature_and_unreferenced_slot(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        self._add_sid_space(cfg, "answer")
        slot = cfg.slots.add(name="hist")
        slot.feature_names.append("nope")
        with self.assertRaisesRegex(ValueError, "not in\n?\\s*feature_configs"):
            self._compile(cfg, [_feature(_HIST)])

        cfg2 = self._config(prompt="X : {{hist}}")
        self._add_sid_space(cfg2, "answer")
        cfg2.slots.add(name="ghost").feature_names.append("hist")
        with self.assertRaisesRegex(ValueError, "never referenced"):
            self._compile(cfg2, [_feature(_HIST)])

    def test_rejects_a_projection_on_an_inline_slot(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        self._add_hist_and_answer_spaces(cfg)
        slot = cfg.slots.add(name="hist")
        slot.feature_names.append("hist")
        slot.projection.bias = True
        with self.assertRaisesRegex(ValueError, "is INLINE"):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_tokens_absent_from_the_base_tokenizer(self) -> None:
        cfg = self._config(prompt="X : {{hist}}")
        # renders Histor0..Histor3, and Histor0 is already in the base vocab
        self._add_sid_space(cfg, "hist", token_format="Histor{i}")
        self._add_sid_space(cfg, "answer")
        with self.assertRaisesRegex(ValueError, "already exist in the tokenizer"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_training_compile_persists_nothing(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4))
        before = sorted(os.listdir(self.test_dir))

        compile_prompt(cfg, [_feature(_HIST)], ["answer"])

        self.assertEqual(sorted(os.listdir(self.test_dir)), before)

    def test_extended_tokenizer_is_written(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4))
        out = os.path.join(self.test_dir, "export")
        compiled = compile_prompt(cfg, [_feature(_HIST)], ["answer"], tokenizer_dir=out)
        written = os.path.join(out, "tokenizer.json")
        self.assertTrue(os.path.exists(written))
        # the SID tokens round-trip, which is what serving reloads
        reloaded = Tokenizer.from_file(written)
        self.assertIsNotNone(reloaded.token_to_id("<|hist_sid_0|>"))
        self.assertIsNotNone(reloaded.token_to_id("<|hist_sid_7|>"))
        self.assertIsNotNone(reloaded.token_to_id("<|answer_sid_0|>"))
        self.assertIsNotNone(reloaded.token_to_id("<|answer_sid_7|>"))
        canonical = json.dumps(
            json.loads(reloaded.to_str()),
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
        self.assertEqual(
            compiled.tokenizer_sha256,
            hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        )

    def test_answer_width_comes_from_the_codebook(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        self._add_sid_space(cfg, "hist", (8,))
        self._add_sid_space(cfg, "answer", (4, 4, 4))
        compiled = self._compile(cfg, [_feature(_HIST)])

        seg = next(
            s for s in compiled.prompt_plan.response_segments if isinstance(s, SlotSeg)
        )
        # the answer is one SID item, so its width needs no sequence_length
        self.assertIs(seg.width.kind, WidthKind.STATIC)
        self.assertEqual(seg.width.num_positions, 3)
        # +1 because HF shifts logits: the window opens one column before the
        # first supervised label
        self.assertEqual(compiled.prompt_plan.logits_suffix_len, 4)

    def test_response_must_name_a_label_field(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{prof}}")
        self._add_sid_space(cfg, "hist", (4, 4, 4))
        self._add_sid_space(cfg, "prof", (4, 4, 4))

        with self.assertRaisesRegex(
            ValueError,
            r"\[prof\] names \['prof'\], which are not in "
            r"data_config.label_fields",
        ):
            self._compile(cfg, [_feature(_HIST), _feature(_PROF)])

    def test_response_slot_takes_exactly_one_label_field(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4, 4))
        slot = cfg.slots.add(name="answer")
        slot.feature_names.extend(["sid_a", "sid_b"])

        with self.assertRaisesRegex(ValueError, "is one label field"):
            compile_prompt(cfg, [_feature(_HIST)], ["sid_a", "sid_b"])

    def test_response_slot_may_not_declare_a_projection(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        self._add_hist_and_answer_spaces(cfg, (4, 4, 4))
        slot = cfg.slots.add(name="answer")
        slot.feature_names.append("answer")
        slot.projection.SetInParent()

        with self.assertRaisesRegex(ValueError, "drop its projection"):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_space_matches_the_underlying_field_names(self) -> None:
        cfg = self._config(prompt="History : {{history}}", response="{{target}}")
        cfg.slots.add(name="history").feature_names.append("hist")
        cfg.slots.add(name="target").feature_names.append("answer")
        self._add_hist_and_answer_spaces(cfg)

        compiled = self._compile(cfg, [_feature(_HIST)])
        body_seg = next(
            seg for seg in compiled.prompt_plan.segments if isinstance(seg, SlotSeg)
        )
        self.assertEqual(body_seg.sid_space_index, 0)
        target_seg = next(
            seg
            for seg in compiled.prompt_plan.response_segments
            if isinstance(seg, SlotSeg)
        )
        self.assertEqual(target_seg.sid_space_index, 1)
        self.assertEqual(compiled.target_sid_space_index, 1)

    def test_inline_field_without_a_named_sid_space_is_rejected(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_sid_space(cfg, "answer")

        with self.assertRaisesRegex(ValueError, "no sid_space has that name"):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_space_without_an_inline_field_is_rejected(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_hist_and_answer_spaces(cfg)
        self._add_sid_space(cfg, "ghost")

        with self.assertRaisesRegex(
            ValueError,
            r"sid_space entries \['ghost'\].*do not match any INLINE",
        ):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_space_names_must_be_nonempty_and_unique(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_sid_space(cfg, "", token_format="<|empty_sid_{i}|>")
        with self.assertRaisesRegex(ValueError, "empty name"):
            self._compile(cfg, [_feature(_HIST)])

        cfg = self._config(prompt="History : {{hist}}")
        self._add_sid_space(cfg, "hist", token_format="<|first_sid_{i}|>")
        self._add_sid_space(cfg, "hist", token_format="<|second_sid_{i}|>")
        with self.assertRaisesRegex(ValueError, "names must be unique"):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_spaces_must_share_the_global_padding_multiple(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_sid_space(cfg, "hist", padding=64)
        self._add_sid_space(cfg, "answer", padding=128)

        with self.assertRaisesRegex(ValueError, "same vocab_pad_to_multiple_of"):
            self._compile(cfg, [_feature(_HIST)])

    def test_sid_token_prefixes_must_be_globally_unique(self) -> None:
        cfg = self._config(prompt="History : {{hist}}")
        self._add_sid_space(cfg, "hist", token_format="<|shared_{i}|>")
        self._add_sid_space(cfg, "answer", token_format="<|shared_{i}|>")

        with self.assertRaisesRegex(ValueError, "already exist in the tokenizer"):
            self._compile(cfg, [_feature(_HIST)])

    def test_response_contains_one_sid_label_placeholder(self) -> None:
        cfg = self._config(
            prompt="History : {{hist}}", response="{{answer}}{{other_answer}}"
        )
        self._add_hist_and_answer_spaces(cfg)
        self._add_sid_space(cfg, "other_answer")

        with self.assertRaisesRegex(ValueError, "exactly one SID label placeholder"):
            compile_prompt(
                cfg,
                [_feature(_HIST)],
                ["answer", "other_answer"],
            )

    def test_one_named_space_uses_the_repeated_path(self) -> None:
        cfg = self._config(prompt="Profile : {{age}}")
        self._add_sid_space(cfg, "answer", (4, 4))

        compiled = self._compile(cfg, [_feature(_AGE)])
        self.assertEqual(len(compiled.sid_spaces), 1)
        self.assertEqual(compiled.sid_spaces[0].name, "answer")
        self.assertEqual(compiled.target_sid_space_index, 0)

    def test_missing_sid_space_is_rejected(self) -> None:
        # the response width is codebook-derived, so sid_space must exist
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        with self.assertRaisesRegex(ValueError, "requires at least one"):
            self._compile(cfg, [_feature(_HIST)])

    def test_token_format_without_a_placeholder_is_rejected(self) -> None:
        # without {i} every token renders alike: one row, not sum(codebook)
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        self._add_sid_space(cfg, "hist", (4, 4, 4), token_format="<|hist_sid|>")
        self._add_sid_space(cfg, "answer", (4, 4, 4))
        with self.assertRaisesRegex(ValueError, "has no '{i}' placeholder"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_custom_token_format_with_a_placeholder_compiles(self) -> None:
        cfg = self._config(prompt="History : {{hist}}", response="{{answer}}")
        self._add_sid_space(cfg, "hist", (4, 4, 4), token_format="C{i}")
        self._add_sid_space(cfg, "answer", (4, 4, 4))
        compiled = self._compile(cfg, [_feature(_HIST)])

        space = compiled.sid_spaces[0]
        self.assertEqual(space.band_hi[-1] - space.band_lo[0] + 1, 12)

    def test_missing_response_is_rejected(self) -> None:
        # no response collapses the window to one ignored position: nan loss
        cfg = self._config(prompt="History : {{hist}}", response="")
        self._add_sid_space(cfg, "hist", (4, 4, 4))
        cfg.ClearField("response")
        with self.assertRaisesRegex(ValueError, "response is required"):
            self._compile(cfg, [_feature(_HIST)])

    def test_a_grouped_feature_inherits_the_group_cap(self) -> None:
        # a SequenceFeature member never sets its own sequence_length; the cap
        # comes from the group, so reading .config here would say UNBOUNDED
        fc = feature_pb2.FeatureConfig()
        text_format.Merge(
            """sequence_feature {
                 sequence_name: "clk" sequence_length: 16 sequence_delim: ";"
                 features { id_feature { feature_name: "h" expression: "item:h"
                            num_buckets: 8 embedding_dim: 4 } }
               }""",
            fc,
        )
        grouped = create_features([fc], fg_mode=FgMode.FG_NONE)
        self.assertFalse(grouped[0].config.HasField("sequence_length"))

        cfg = self._config(prompt="History : {{clk__h}}")
        self._add_sid_space(cfg, "answer")
        compiled = self._compile(cfg, grouped)

        seg = next(s for s in compiled.prompt_plan.segments if isinstance(s, SlotSeg))
        self.assertIs(seg.width.kind, WidthKind.BOUNDED)
        self.assertEqual(seg.width.num_positions, 16)


if __name__ == "__main__":
    unittest.main()
