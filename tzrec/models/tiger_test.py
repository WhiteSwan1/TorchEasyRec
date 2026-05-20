# Copyright (c) 2024, Alibaba Group;
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#    http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Unit tests for the Tiger model.

These tests construct minimal synthetic batches (KJT for history_sids,
JaggedTensor for label_sids, optional KJT for user_id) and exercise the
model's forward, loss, beam search, and metric update paths.

Standalone, in-process — does not require a parquet dataset on disk.
"""

import unittest
from typing import List, Optional

import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

from tzrec.datasets.utils import Batch
from tzrec.models.tiger import (
    DECODER_HIDDEN_KEY,
    GENERATED_SIDS_KEY,
    LOSS_KEY,
    STEP_SCORES_KEY,
    Tiger,
)
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.tiger_pb2 import Tiger as TigerProto


def _build_proto(
    codebook: str = "8,8,8",
    embed_dim: int = 16,
    num_heads: int = 2,
    d_kv: int = 8,
    hidden_dims: str = "32",
    num_encoder_layers: int = 1,
    num_decoder_layers: int = 1,
    dropout_rate: float = 0.0,
    num_user_bins: int = 0,
    use_sep_token: bool = True,
    beam_width: int = 4,
) -> ModelConfig:
    """Build a minimal ModelConfig with a Tiger sub-message for testing."""
    tiger_msg = TigerProto(
        embed_dim=embed_dim,
        num_heads=num_heads,
        d_kv=d_kv,
        hidden_dims=hidden_dims,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout_rate=dropout_rate,
        codebook=codebook,
        num_user_bins=num_user_bins,
        use_sep_token=use_sep_token,
        beam_width=beam_width,
    )
    mc = ModelConfig()
    mc.tiger.CopyFrom(tiger_msg)
    return mc


def _build_batch(
    history_sids_per_user: List[List[int]],
    label_sids_per_user: List[List[int]],
    user_ids: Optional[List[int]] = None,
) -> Batch:
    """Pack synthetic SID-encoded data into a Batch."""
    flat_history = [c for seq in history_sids_per_user for c in seq]
    history_lengths = [len(seq) for seq in history_sids_per_user]
    history_values = torch.tensor(flat_history, dtype=torch.long)
    history_lengths_t = torch.tensor(history_lengths, dtype=torch.long)
    history_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["history_sids"],
        values=history_values,
        lengths=history_lengths_t,
    )
    sparse = {"history": history_kjt}
    if user_ids is not None:
        user_values = torch.tensor(user_ids, dtype=torch.long)
        user_lengths = torch.ones(len(user_ids), dtype=torch.long)
        user_kjt = KeyedJaggedTensor.from_lengths_sync(
            keys=["user_id"], values=user_values, lengths=user_lengths
        )
        sparse["user"] = user_kjt

    label_values = torch.tensor(
        [c for tup in label_sids_per_user for c in tup], dtype=torch.long
    )
    label_lengths = torch.tensor(
        [len(tup) for tup in label_sids_per_user], dtype=torch.long
    )
    label_jt = JaggedTensor(values=label_values, lengths=label_lengths)
    return Batch(
        sparse_features=sparse,
        jagged_labels={"label_sids": label_jt},
    )


class TigerForwardTest(unittest.TestCase):
    """Test the training-mode forward pass."""

    def test_train_forward_returns_decoder_hidden(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8", embed_dim=16)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.train()
        # Each user has 2 items × 3 hierarchies = 6 history tokens.
        batch = _build_batch(
            history_sids_per_user=[
                [1, 2, 3, 4, 5, 6],
                [7, 0, 1, 2, 3, 4],
            ],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6]],
        )
        out = model.predict(batch)
        self.assertIn(DECODER_HIDDEN_KEY, out)
        self.assertEqual(out[DECODER_HIDDEN_KEY].shape, (2, 3, 16))

    def test_train_forward_with_user_id(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8", embed_dim=16, num_user_bins=100)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.train()
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6]],
            user_ids=[42, 7],
        )
        out = model.predict(batch)
        self.assertEqual(out[DECODER_HIDDEN_KEY].shape, (2, 3, 16))

    def test_train_forward_variable_history_length(self) -> None:
        """Different users with different history lengths should work."""
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8")
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.train()
        batch = _build_batch(
            history_sids_per_user=[
                [1, 2, 3],  # 1 item
                [4, 5, 6, 7, 0, 1],  # 2 items
                [2, 3, 4, 5, 6, 7, 0, 1, 2],  # 3 items
            ],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6], [7, 0, 1]],
        )
        out = model.predict(batch)
        self.assertEqual(out[DECODER_HIDDEN_KEY].shape, (3, 3, 16))


class TigerLossTest(unittest.TestCase):
    """Verify the cross-entropy loss path."""

    def test_loss_returns_single_key_scalar(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8")
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.train()
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6]],
        )
        predictions = model.predict(batch)
        losses = model.loss(predictions, batch)
        self.assertIn(LOSS_KEY, losses)
        self.assertEqual(losses[LOSS_KEY].dim(), 0)
        self.assertFalse(torch.isnan(losses[LOSS_KEY]))
        # Random init → loss ≈ log(8) * 3 ≈ 6.24 (3-hierarchy, 8-way each).
        self.assertGreater(losses[LOSS_KEY].item(), 0.0)
        self.assertLess(losses[LOSS_KEY].item(), 50.0)

    def test_loss_backward_propagates_to_all_params(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8")
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.train()
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6]],
        )
        predictions = model.predict(batch)
        losses = model.loss(predictions, batch)
        total = sum(losses.values())
        total.backward()
        # Spot-check a few parameters got gradients.
        self.assertIsNotNone(model.sid_embedding.weight.grad)
        self.assertIsNotNone(model.bos_token.grad)
        for h in range(model._num_hierarchies):
            self.assertIsNotNone(model.decoder_mlp[h].weight.grad)


class TigerGenerateTest(unittest.TestCase):
    """Verify beam search output shape, dtype, and value ranges."""

    def test_generate_shape_and_dtype(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8", beam_width=4)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.eval()
        # Inference batch: history present, no labels.
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]],
            label_sids_per_user=[[0, 0, 0], [0, 0, 0]],  # placeholder
        )
        # Manually drop the labels to mimic pure inference.
        batch.jagged_labels = {}
        with torch.no_grad():
            out = model.predict(batch)
        self.assertIn(GENERATED_SIDS_KEY, out)
        self.assertIn(STEP_SCORES_KEY, out)
        self.assertEqual(out[GENERATED_SIDS_KEY].shape, (2, 4, 3))
        self.assertEqual(out[GENERATED_SIDS_KEY].dtype, torch.long)
        self.assertEqual(out[STEP_SCORES_KEY].shape, (2, 4, 3))
        self.assertTrue(out[STEP_SCORES_KEY].dtype.is_floating_point)
        # Every emitted code must be in [0, codebook_sizes[h]) for its slot.
        sids = out[GENERATED_SIDS_KEY]
        codebook_sizes = [8, 8, 8]
        for h in range(3):
            self.assertTrue((sids[:, :, h] >= 0).all())
            self.assertTrue((sids[:, :, h] < codebook_sizes[h]).all())
        # Step scores must be probabilities in [0, 1].
        scores = out[STEP_SCORES_KEY]
        self.assertTrue((scores >= 0).all())
        self.assertTrue((scores <= 1).all())

    def test_generate_beams_sorted_descending(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8", beam_width=4)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.eval()
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6]],
            label_sids_per_user=[[0, 0, 0]],
        )
        batch.jagged_labels = {}
        with torch.no_grad():
            out = model.predict(batch)
        marginal = out[STEP_SCORES_KEY].prod(dim=-1)  # (B, K)
        diffs = marginal[:, :-1] - marginal[:, 1:]
        self.assertTrue((diffs >= -1e-6).all(), "beams not sorted descending")


class TigerOffsetTest(unittest.TestCase):
    """Verify the per-hierarchy offset trick.

    With codebook=[8,8,8] the offsets should be [0, 8, 16] and the
    embedding table should have sum(codebook_sizes)=24 rows.
    """

    def test_compact_table_size(self) -> None:
        mc = _build_proto(codebook="8,8,8")
        model = Tiger(mc, features=[], labels=["label_sids"])
        self.assertEqual(model.sid_embedding.weight.shape, (24, 16))

    def test_non_uniform_codebook(self) -> None:
        mc = _build_proto(codebook="4,8,16")
        model = Tiger(mc, features=[], labels=["label_sids"])
        # Table is sum([4, 8, 16]) = 28 rows.
        self.assertEqual(model.sid_embedding.weight.shape, (28, 16))
        # Per-hierarchy head widths.
        self.assertEqual(model.decoder_mlp[0].out_features, 4)
        self.assertEqual(model.decoder_mlp[1].out_features, 8)
        self.assertEqual(model.decoder_mlp[2].out_features, 16)

    def test_offset_tensor_values(self) -> None:
        mc = _build_proto(codebook="4,8,16")
        model = Tiger(mc, features=[], labels=["label_sids"])
        # offsets = [0, 4, 4+8] = [0, 4, 12]
        self.assertEqual(model.code_offsets.tolist(), [0, 4, 12])

    def test_offset_with_mask(self) -> None:
        """The mask-multiply trick must send padding to row 0."""
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8")
        model = Tiger(mc, features=[], labels=["label_sids"])
        # Two samples: full SID tuple, one with a padded tail position.
        sids = torch.tensor([[1, 2, 3], [4, 0, 0]], dtype=torch.long)
        # Mark the last two positions of sample 1 as padding.
        mask = torch.tensor([[1, 1, 1], [1, 0, 0]], dtype=torch.long)
        shifted = model._add_repeating_offset(sids, mask)
        # Sample 0: (1+0, 2+8, 3+16) = (1, 10, 19)
        self.assertEqual(shifted[0].tolist(), [1, 10, 19])
        # Sample 1: (4+0, masked->0, masked->0)
        self.assertEqual(shifted[1].tolist(), [4, 0, 0])


class TigerSepInjectionTest(unittest.TestCase):
    """Verify SEP-token injection extends sequences correctly."""

    def test_sep_injection_extends_length(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="4,4,4", embed_dim=8, use_sep_token=True)
        model = Tiger(mc, features=[], labels=["label_sids"])
        # 2 items × 3 hierarchies = 6 tokens per sample.
        sid_embeds = torch.randn(2, 6, 8)
        attention_mask = torch.ones(2, 6, dtype=torch.long)
        out_embeds, out_mask = model._inject_sep_token(sid_embeds, attention_mask)
        # 2 items × (3 + 1 SEP) = 8 tokens per sample.
        self.assertEqual(out_embeds.shape, (2, 8, 8))
        self.assertEqual(out_mask.shape, (2, 8))
        # Mask should be all-ones since input had no padding.
        self.assertTrue((out_mask == 1).all())

    def test_sep_disabled(self) -> None:
        mc = _build_proto(codebook="4,4,4", use_sep_token=False)
        model = Tiger(mc, features=[], labels=["label_sids"])
        self.assertIsNone(model.sep_token)


class TigerMetricTest(unittest.TestCase):
    """Verify metric initialization and update."""

    def test_metric_init(self) -> None:
        mc = _build_proto(codebook="8,8,8", beam_width=4)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.init_metric()
        self.assertIn(LOSS_KEY, model._metric_modules)
        self.assertIn("recall@5", model._metric_modules)
        self.assertIn("recall@4", model._metric_modules)
        self.assertIn("ndcg@5", model._metric_modules)
        self.assertIn("ndcg@4", model._metric_modules)

    def test_metric_update_runs(self) -> None:
        torch.manual_seed(0)
        mc = _build_proto(codebook="8,8,8", beam_width=4)
        model = Tiger(mc, features=[], labels=["label_sids"])
        model.eval()
        model.init_metric()
        batch = _build_batch(
            history_sids_per_user=[[1, 2, 3, 4, 5, 6], [7, 0, 1, 2, 3, 4]],
            label_sids_per_user=[[1, 2, 3], [4, 5, 6]],
        )
        with torch.no_grad():
            preds = model.predict(batch)
        losses = model.loss(preds, batch)
        # update_metric runs generate() internally.
        model.update_metric(preds, batch, losses=losses)
        result = model.compute_metric()
        self.assertIn(LOSS_KEY, result)
        self.assertIn("recall@5", result)


if __name__ == "__main__":
    unittest.main()
