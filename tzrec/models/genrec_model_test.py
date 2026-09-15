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

import os
import unittest

import torch
import torch.fx
from parameterized import parameterized
from torchrec import KeyedJaggedTensor
from transformers import AutoModelForCausalLM

from tzrec.datasets.utils import BASE_DATA_GROUP, Batch
from tzrec.main import _create_model
from tzrec.models.genrec_model import (
    _PARAM_DTYPE,
    SLOT_EMBEDS,
    GenRecFrontEnd,
    project_slots,
)
from tzrec.models.model import ScriptWrapper, TrainWrapper
from tzrec.prompt.assembler import (
    ATTACH_POSITIONS,
    HOLE_POSITIONS,
    HOLE_SLOT_COUNTS,
    INPUT_IDS,
    PromptAssembler,
)
from tzrec.prompt.hole_keys import HOLE_KEYS, HoleKeyBuilder
from tzrec.protos import feature_pb2
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import GenRecModelConfig
from tzrec.protos.prompt_pb2 import PromptSlot
from tzrec.utils.fx_util import symbolic_trace
from tzrec.utils.state_dict_util import init_parameters
from tzrec.utils.test_util import (
    create_genrec_test_model,
    make_test_dir,
    parameterized_name_func,
)

# offset SID codes for the (4, 4, 4) codebook: level_offsets[l] + code
_HIST_CODES = [0, 5, 10]
_LONG_HIST_CODES = [0, 5, 10, 3, 4, 9]
_ANSWER_CODES = [1, 6, 11]


def _hist() -> feature_pb2.FeatureConfig:
    return feature_pb2.FeatureConfig(
        sequence_raw_feature=feature_pb2.RawFeature(
            feature_name="hist", expression="user:hist"
        )
    )


def _projected(name: str, dim: int) -> feature_pb2.FeatureConfig:
    return feature_pb2.FeatureConfig(
        sequence_id_feature=feature_pb2.IdFeature(
            feature_name=name,
            expression=f"user:{name}",
            num_buckets=32,
            embedding_dim=dim,
            sequence_length=2,
        )
    )


def _batch(compiled_prompt, parsed, sparse=None) -> Batch:
    batch = Batch(sparse_features={BASE_DATA_GROUP: sparse} if sparse else {})
    batch.additional_infos.update(
        PromptAssembler(
            compiled_prompt.prompt_plan,
            compiled_prompt.sid_spaces,
            compiled_prompt.sentinel_token_id,
        )(parsed)
    )
    return batch


def _projected_batch(compiled_prompt) -> Batch:
    return _batch(
        compiled_prompt,
        {
            "hist.values": torch.tensor(_HIST_CODES).reshape(-1, 1),
            "hist.lengths": torch.tensor([3]),
            "answer.values": torch.tensor(_ANSWER_CODES),
            "answer.lengths": torch.tensor([3]),
            "prof.values": torch.tensor([5, 9]),
            "prof.lengths": torch.tensor([2]),
        },
        sparse=KeyedJaggedTensor.from_lengths_sync(
            keys=["prof"], values=torch.tensor([5, 9]), lengths=torch.tensor([2])
        ),
    )


class BaseGenRecModelTest(unittest.TestCase):
    """Shared causal-LM behavior, reached through its concrete subclass."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.model, self.compiled_prompt = create_genrec_test_model(self.test_dir)

    def test_tokens_to_local_codes_undoes_shifts_and_groups_beams(self) -> None:
        space = self.compiled_prompt.sid_spaces[
            self.compiled_prompt.target_sid_space_index
        ]
        local_codes = torch.tensor(
            [
                [0, 1, 3],
                [3, 0, 2],
                [1, 3, 0],
                [2, 2, 1],
            ]
        )
        tokens = local_codes + torch.tensor(space.level_offsets) + space.base_vocab_size
        codes = self.model._tokens_to_local_codes(tokens, batch_size=2)

        self.assertEqual(codes.shape, (2, 2, space.num_levels))
        self.assertEqual(codes.tolist(), local_codes.reshape(2, 2, -1).tolist())

    def test_rejects_a_model_built_without_a_prompt(self) -> None:
        model_config = ModelConfig()
        model_config.genrec_causal_lm_model.hf_model_name_or_path = os.path.join(
            self.test_dir, "backbone"
        )
        with self.assertRaisesRegex(ValueError, "needs a compiled prompt"):
            _create_model(model_config, [], ["answer"], compiled_prompt=None)

    def test_shared_projection_name_requires_matching_widths(self) -> None:
        with self.assertRaisesRegex(ValueError, "cannot share a module"):
            create_genrec_test_model(
                self.test_dir,
                feature_configs=[_hist(), _projected("pa", 8), _projected("pb", 16)],
                prompt="History : {{hist}} . {{pa}} {{pb}} Predict :",
                slots=[
                    PromptSlot(
                        name="pa", feature_names=["pa"], projection_name="shared"
                    ),
                    PromptSlot(
                        name="pb", feature_names=["pb"], projection_name="shared"
                    ),
                ],
            )

    def test_projected_slot_overwrites_sentinels_and_backpropagates(self) -> None:
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("prof", 8)],
            prompt="History : {{hist}} . Predict {{prof}} :",
        )
        # the embedding table is built on meta until something materializes it
        init_parameters(model, device=torch.device("cpu"))
        batch = _projected_batch(compiled_prompt)

        embeds = model.build_input(batch)
        raw = model.lm.get_input_embeddings()(batch.additional_infos[INPUT_IDS])
        holes = batch.additional_infos[HOLE_POSITIONS]
        self.assertGreater(holes.numel(), 0)

        changed = ~torch.isclose(embeds, raw).all(dim=-1)
        self.assertEqual(sorted(changed.nonzero().flatten().tolist()), holes.tolist())

        embeds[holes].sum().backward()
        proj = next(iter(model.projections.values()))
        self.assertIsNotNone(proj.head.weight.grad)

    @parameterized.expand(
        [[GenRecModelConfig.BF16], [GenRecModelConfig.FP16]],
        name_func=parameterized_name_func,
    )
    def test_projected_slot_follows_a_narrow_lm_dtype(self, lm_parameter_dtype) -> None:
        model, compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("prof", 8)],
            prompt="History : {{hist}} . Predict {{prof}} :",
            lm_parameter_dtype=lm_parameter_dtype,
        )
        init_parameters(model, device=torch.device("cpu"))
        batch = _projected_batch(compiled_prompt)

        embeds = model.build_input(batch)
        self.assertIs(embeds.dtype, _PARAM_DTYPE[lm_parameter_dtype])

        predictions = model.predict(batch)
        loss = model.loss(predictions, batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()
        # the projection keeps fp32 masters, so only the spliced values convert
        proj = next(iter(model.projections.values()))
        self.assertIs(proj.head.weight.dtype, torch.float32)
        self.assertGreater(float(proj.head.weight.grad.abs().sum()), 0.0)

    def test_metric_averages_the_loss_across_batches(self) -> None:
        self.model.init_metric()
        for value in (1.0, 3.0):
            self.model.update_metric({}, Batch(), {"ce_loss": torch.tensor(value)})

        self.assertAlmostEqual(
            self.model._metric_modules["ce_loss"].compute().item(), 2.0, places=5
        )

    def test_init_from_pretrained_replaces_the_empty_weights(self) -> None:
        base_vocab_size = self.compiled_prompt.sid_spaces[0].base_vocab_size
        embeddings = self.model.lm.get_input_embeddings()
        before = embeddings.weight[:base_vocab_size].clone()
        self.model.init_from_pretrained()
        after = embeddings.weight[:base_vocab_size]

        # the checkpoint rows land verbatim; only the appended SID rows are new
        reference = AutoModelForCausalLM.from_pretrained(
            os.path.join(self.test_dir, "backbone")
        )
        expected = reference.get_input_embeddings().weight[:base_vocab_size]
        self.assertFalse(torch.allclose(before, expected))
        torch.testing.assert_close(after, expected)

    def test_model_resizes_to_target_vocab_size(self) -> None:
        rows = self.model.lm.get_input_embeddings().weight.shape[0]
        self.assertEqual(rows, self.compiled_prompt.target_vocab_size)
        self.assertGreater(
            rows, max(space.band_hi[-1] for space in self.compiled_prompt.sid_spaces)
        )
        self.assertEqual(
            self.model._sid_base_vocabs.tolist(),
            [space.base_vocab_size for space in self.compiled_prompt.sid_spaces],
        )

    def test_loss_is_finite_and_backpropagates_into_the_backbone(self) -> None:
        batch = _batch(
            self.compiled_prompt,
            {
                "hist.values": torch.tensor(_LONG_HIST_CODES),
                "hist.lengths": torch.tensor([6]),
                "answer.values": torch.tensor(_ANSWER_CODES),
                "answer.lengths": torch.tensor([3]),
            },
        )
        predictions = self.model.predict(batch)
        loss = self.model.loss(predictions, batch)["ce_loss"]
        self.assertTrue(bool(torch.isfinite(loss)))
        loss.backward()

        grad = self.model.lm.get_input_embeddings().weight.grad
        self.assertIsNotNone(grad)
        self.assertTrue(bool((grad.abs().sum() > 0)))

    def test_training_forward_survives_fx_tracing(self) -> None:
        torch.fx.symbolic_trace(TrainWrapper(self.model))


class GenRecFrontEndTest(unittest.TestCase):
    """The served half of the model, under the same wrapper every export uses."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.model, self.compiled_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("beh", 8)],
            prompt="History : {{hist}} . {{beh}} Predict :",
        )
        init_parameters(self.model, device=torch.device("cpu"))
        # the parsed dict as the data parser emits it: a dense sequence of codes
        # and a sparse behaviour sequence
        self.data = {
            "hist.values": torch.tensor(_LONG_HIST_CODES, dtype=torch.float32).reshape(
                -1, 1
            ),
            "hist.lengths": torch.tensor([6]),
            "beh.values": torch.tensor([3, 9]),
            "beh.lengths": torch.tensor([2]),
        }
        self.wrapped = ScriptWrapper(GenRecFrontEnd(self.model))

    def test_front_end_returns_the_walk_and_the_projected_slots(self) -> None:
        out = self.wrapped(self.data)
        compiled = self.compiled_prompt
        walk = PromptAssembler(
            compiled.prompt_plan,
            compiled.sid_spaces,
            compiled.sentinel_token_id,
            include_response=False,
        )(self.data)
        for key in (INPUT_IDS, HOLE_POSITIONS):
            self.assertTrue(torch.equal(out[key], walk[key]), key)
        self.assertTrue(
            torch.equal(out[HOLE_KEYS], HoleKeyBuilder(compiled.prompt_plan)(self.data))
        )
        batch = self.wrapped.get_batch(self.data)
        expected = project_slots(
            self.model.embedding_group(batch),
            compiled.prompt_plan,
            self.model._slot_projections,
            int(self.model.lm.config.hidden_size),
        )
        self.assertTrue(torch.allclose(out[SLOT_EMBEDS], expected))
        self.assertEqual(tuple(out[SLOT_EMBEDS].shape), (2, 32))

    def test_front_end_joins_several_projected_slots_in_order(self) -> None:
        """Two slots: ``slot_embeds`` is their projections, occurrence by occurrence."""
        model, compiled = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("beh", 8), _projected("ctx", 8)],
            prompt="History : {{hist}} . {{beh}} then {{ctx}} Predict :",
        )
        init_parameters(model, device=torch.device("cpu"))
        wrapped = ScriptWrapper(GenRecFrontEnd(model))
        data = dict(self.data)
        data["ctx.values"] = torch.tensor([1, 4, 7])
        data["ctx.lengths"] = torch.tensor([3])

        out = wrapped(data)
        counts = out[HOLE_SLOT_COUNTS]
        self.assertEqual(counts.tolist(), [2, 3])
        grouped = model.embedding_group(wrapped.get_batch(data))
        spans = torch.split(out[SLOT_EMBEDS], counts.tolist())
        for seg, proj, span in zip(
            compiled.prompt_plan.projected_slots, model._slot_projections, spans
        ):
            expected = proj(grouped[seg.name + seg.output_key]).reshape(-1, 32)
            self.assertTrue(torch.allclose(span, expected), seg.name)

    def test_front_end_shares_the_checkpoint_names_and_not_the_lm(self) -> None:
        names = set(self.wrapped.state_dict())
        self.assertTrue(any(n.startswith("model.embedding_group.") for n in names))
        self.assertTrue(any(n.startswith("model.projections.") for n in names))
        self.assertFalse(any(".lm." in n for n in names))

    def test_front_end_traces_and_scripts(self) -> None:
        """The assembler is an FX leaf, so the export's trace-then-script works."""
        eager = self.wrapped(self.data)
        scripted = torch.jit.script(symbolic_trace(self.wrapped))
        out = scripted(self.data)
        for key, value in eager.items():
            if value.is_floating_point():
                self.assertTrue(torch.allclose(out[key], value), key)
            else:
                self.assertTrue(torch.equal(out[key], value), key)


class GenRecAttachedSlotTest(unittest.TestCase):
    """A slot added onto the history SID tokens through a zero-initialized map."""

    def setUp(self) -> None:
        self.test_dir = make_test_dir()
        self.plain, plain_prompt = create_genrec_test_model(self.test_dir)
        slot = PromptSlot(
            name="hist_attr", feature_names=["ta", "tb"], attach_to="hist"
        )
        slot.projection.bias = True
        self.attached, attached_prompt = create_genrec_test_model(
            self.test_dir,
            feature_configs=[_hist(), _projected("ta", 8), _projected("tb", 4)],
            slots=[slot],
        )
        init_parameters(self.attached, device=torch.device("cpu"))
        self.attached.lm.load_state_dict(self.plain.lm.state_dict())
        parsed = {
            "hist.values": torch.tensor(_LONG_HIST_CODES),
            "hist.lengths": torch.tensor([6]),
            "answer.values": torch.tensor(_ANSWER_CODES),
            "answer.lengths": torch.tensor([3]),
        }
        self.plain_batch = _batch(plain_prompt, parsed)
        attributes = {
            "ta.values": torch.tensor([3, 9]),
            "ta.lengths": torch.tensor([2]),
            "tb.values": torch.tensor([1, 4]),
            "tb.lengths": torch.tensor([2]),
        }
        self.attached_batch = _batch(
            attached_prompt,
            parsed | attributes,
            sparse=KeyedJaggedTensor.from_lengths_sync(
                keys=["ta", "tb"],
                values=torch.tensor([3, 9, 1, 4]),
                lengths=torch.tensor([2, 2]),
            ),
        )

    def test_attached_projection_is_created_zero_initialized(self) -> None:
        proj = self.attached.projections["hist_attr"]
        self.assertEqual(list(self.attached.projections), ["hist_attr"])
        self.assertIs(self.attached._attach_projections[0], proj)
        self.assertEqual(self.attached._slot_projections, [])
        self.assertEqual(proj.head.in_features, 12)
        self.assertFalse(bool(proj.head.weight.any()))
        self.assertFalse(bool(proj.head.bias.any()))

    def test_input_is_unchanged_until_the_projection_moves(self) -> None:
        expected = self.plain.build_input(self.plain_batch)
        infos = self.attached_batch.additional_infos
        self.assertTrue(
            torch.equal(infos[INPUT_IDS], self.plain_batch.additional_infos[INPUT_IDS])
        )
        self.assertTrue(
            torch.equal(self.attached.build_input(self.attached_batch), expected)
        )

        with torch.no_grad():
            self.attached.projections["hist_attr"].head.weight.fill_(0.5)
        embeds = self.attached.build_input(self.attached_batch)
        changed = ~torch.isclose(embeds, expected).all(dim=-1)
        self.assertEqual(
            changed.nonzero().flatten().tolist(), infos[ATTACH_POSITIONS].tolist()
        )
        self.assertEqual(infos[ATTACH_POSITIONS].numel(), len(_LONG_HIST_CODES))

    def test_training_forward_with_an_attached_slot_survives_fx_tracing(self) -> None:
        torch.fx.symbolic_trace(TrainWrapper(self.attached))

    def test_front_end_rejects_attached_slots(self) -> None:
        with self.assertRaisesRegex(ValueError, "not part of the serving contract"):
            GenRecFrontEnd(self.attached)

    def test_zero_init_keeps_logits_and_trains_the_projection(self) -> None:
        expected = self.plain.predict(self.plain_batch)
        predictions = self.attached.predict(self.attached_batch)
        self.assertTrue(torch.equal(predictions["logits"], expected["logits"]))
        loss = self.attached.loss(predictions, self.attached_batch)["ce_loss"]
        loss.backward()
        head = self.attached.projections["hist_attr"].head
        self.assertGreater(float(head.weight.grad.abs().sum()), 0.0)


if __name__ == "__main__":
    unittest.main()
