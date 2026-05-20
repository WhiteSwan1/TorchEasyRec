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

"""Standalone smoke test for the Tiger model.

Runs forward + backward + beam-search on synthetic SID-encoded data and
prints shape/score sanity checks. Does NOT require the full tzrec stack;
specifically, monkey-patches `tzrec.utils.dynamicemb_util` (which currently
fails to import against torchrec < 1.6) with a no-op shim so we can verify
the TIGER model logic in isolation.

Run from the repo root:
    python ft_scripts/tiger_smoke.py
"""

import os
import sys
import types

# ---------------------------------------------------------------------------
# Monkey-patch the broken `dynamicemb_util` import (unrelated to TIGER).
# This module fails on torchrec < 1.6 because it depends on
# `torchrec.distributed.planner.estimator.types.HardwarePerfConfig` which
# was introduced in a newer torchrec. Replace it with a no-op shim before
# tzrec's auto_import runs.
# ---------------------------------------------------------------------------
_shim = types.ModuleType("tzrec.utils.dynamicemb_util")
_shim.__file__ = __file__
_shim.__spec__ = None

def _noop(*args, **kwargs):  # pyre-ignore
    return None

_shim.apply_dynamicemb_patches = _noop
_shim.is_dynamicemb_enabled = lambda: False
sys.modules["tzrec.utils.dynamicemb_util"] = _shim

# Ensure the repo root is on sys.path so the local tzrec wins over installed.
_REPO_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

import torch
from torchrec.sparse.jagged_tensor import JaggedTensor, KeyedJaggedTensor

# Now safe to import tzrec.
from tzrec.datasets.utils import Batch  # noqa: E402
from tzrec.models.tiger import (  # noqa: E402
    DECODER_HIDDEN_KEY,
    GENERATED_SIDS_KEY,
    LOSS_KEY,
    STEP_SCORES_KEY,
    Tiger,
)
from tzrec.protos.model_pb2 import ModelConfig  # noqa: E402
from tzrec.protos.models.tiger_pb2 import Tiger as TigerProto  # noqa: E402


def _build_proto(
    codebook: str = "8192,8192,8192",
    embed_dim: int = 128,
    num_heads: int = 4,
    d_kv: int = 32,
    hidden_dims: str = "256",
    num_encoder_layers: int = 2,
    num_decoder_layers: int = 2,
    beam_width: int = 4,
) -> ModelConfig:
    tiger_msg = TigerProto(
        embed_dim=embed_dim,
        num_heads=num_heads,
        d_kv=d_kv,
        hidden_dims=hidden_dims,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dropout_rate=0.1,
        codebook=codebook,
        num_user_bins=0,
        use_sep_token=True,
        beam_width=beam_width,
        # New: explicit group_name plumbing. Tiger will resolve the
        # actual feature_name via feature_groups below.
        history_group_name="history",
    )
    mc = ModelConfig()
    mc.tiger.CopyFrom(tiger_msg)
    # Declare matching feature_groups so Tiger's resolver can find them.
    hist_fg = mc.feature_groups.add()
    hist_fg.group_name = "history"
    hist_fg.feature_names.append("history_sids")
    return mc


def _build_synth_batch(
    batch_size: int,
    items_per_user: int,
    codebook_sizes,
):
    """Build a synthetic batch with 0-indexed codes."""
    H = len(codebook_sizes)
    history_per_user = []
    label_per_user = []
    for _ in range(batch_size):
        seq = []
        for _ in range(items_per_user):
            for h in range(H):
                seq.append(int(torch.randint(0, codebook_sizes[h], (1,)).item()))
        history_per_user.append(seq)
        label_per_user.append(
            [int(torch.randint(0, codebook_sizes[h], (1,)).item()) for h in range(H)]
        )

    flat_history = [c for seq in history_per_user for c in seq]
    history_lengths = [len(seq) for seq in history_per_user]
    history_kjt = KeyedJaggedTensor.from_lengths_sync(
        keys=["history_sids"],
        values=torch.tensor(flat_history, dtype=torch.long),
        lengths=torch.tensor(history_lengths, dtype=torch.long),
    )

    label_values = torch.tensor(
        [c for tup in label_per_user for c in tup], dtype=torch.long
    )
    label_lengths = torch.tensor(
        [len(tup) for tup in label_per_user], dtype=torch.long
    )
    label_jt = JaggedTensor(values=label_values, lengths=label_lengths)

    return Batch(
        sparse_features={"history": history_kjt},
        jagged_labels={"label_sids": label_jt},
    )


def _check(condition: bool, message: str) -> None:
    status = "PASS" if condition else "FAIL"
    print(f"  [{status}] {message}")
    if not condition:
        raise SystemExit(1)


def main() -> None:
    torch.manual_seed(0)
    print("=" * 68)
    print("TIGER smoke test — synthetic data, codebook = 8192,8192,8192")
    print("=" * 68)

    codebook = "8192,8192,8192"
    codebook_sizes = [8192, 8192, 8192]
    H = len(codebook_sizes)
    embed_dim = 128
    beam_width = 4

    mc = _build_proto(
        codebook=codebook,
        embed_dim=embed_dim,
        beam_width=beam_width,
    )
    model = Tiger(mc, features=[], labels=["label_sids"])
    n_params = sum(p.numel() for p in model.parameters())
    print(f"\nModel constructed: {n_params:,} parameters")
    print(
        f"  sid_embedding: {model.sid_embedding.weight.shape} "
        f"(sum(codebook) = {sum(codebook_sizes)})"
    )
    print(f"  per-hierarchy heads: {[m.out_features for m in model.decoder_mlp]}")
    print(f"  code_offsets: {model.code_offsets.tolist()}")
    _check(
        model.sid_embedding.weight.shape == (sum(codebook_sizes), embed_dim),
        "compact embedding table sized sum(codebook_sizes) × embed_dim",
    )
    _check(
        model.code_offsets.tolist() == [0, 8192, 16384],
        "code_offsets = cumsum prefix",
    )

    # --- Training-mode forward + backward ---
    print("\nTest 1: train-mode forward + backward")
    model.train()
    batch = _build_synth_batch(batch_size=4, items_per_user=3, codebook_sizes=codebook_sizes)
    out = model.predict(batch)
    _check(DECODER_HIDDEN_KEY in out, "predict() returns decoder_hidden in train mode")
    _check(
        out[DECODER_HIDDEN_KEY].shape == (4, H, embed_dim),
        f"decoder_hidden shape = (B={4}, H={H}, D={embed_dim})",
    )
    losses = model.loss(out, batch)
    _check(LOSS_KEY in losses, "loss() returns tiger_ce")
    _check(losses[LOSS_KEY].dim() == 0, "loss is scalar")
    _check(not torch.isnan(losses[LOSS_KEY]), "loss is finite")
    print(f"  initial loss: {losses[LOSS_KEY].item():.4f} (expected ~3*ln(8192) ≈ 27.0)")
    losses[LOSS_KEY].backward()
    _check(model.sid_embedding.weight.grad is not None, "sid_embedding receives gradient")
    _check(model.bos_token.grad is not None, "bos_token receives gradient")
    for h in range(H):
        _check(
            model.decoder_mlp[h].weight.grad is not None,
            f"decoder_mlp[{h}] receives gradient",
        )

    # --- Loss decreases over a few optimizer steps ---
    print("\nTest 2: loss decreases over training steps")
    model.zero_grad()
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    initial_loss = None
    final_loss = None
    for step in range(20):
        optimizer.zero_grad()
        out = model.predict(batch)
        losses = model.loss(out, batch)
        losses[LOSS_KEY].backward()
        optimizer.step()
        if step == 0:
            initial_loss = losses[LOSS_KEY].item()
        if step == 19:
            final_loss = losses[LOSS_KEY].item()
    print(f"  step 0 loss: {initial_loss:.4f}")
    print(f"  step 19 loss: {final_loss:.4f}")
    _check(
        final_loss < initial_loss,
        f"loss decreases (step 19 < step 0)",
    )

    # --- Inference-mode forward (beam search) ---
    print("\nTest 3: inference-mode beam search")
    model.eval()
    inference_batch = _build_synth_batch(
        batch_size=4, items_per_user=3, codebook_sizes=codebook_sizes
    )
    inference_batch.jagged_labels = {}  # mimic pure inference
    with torch.no_grad():
        out = model.predict(inference_batch)
    _check(GENERATED_SIDS_KEY in out, "predict() returns generated_sids in eval mode")
    _check(STEP_SCORES_KEY in out, "predict() returns step_scores in eval mode")
    sids = out[GENERATED_SIDS_KEY]
    scores = out[STEP_SCORES_KEY]
    _check(sids.shape == (4, beam_width, H), f"generated_sids shape = (4, {beam_width}, {H})")
    _check(scores.shape == (4, beam_width, H), f"step_scores shape = (4, {beam_width}, {H})")
    _check(sids.dtype == torch.long, "generated_sids dtype = int64")
    for h in range(H):
        in_range = (sids[:, :, h] >= 0).all() and (sids[:, :, h] < codebook_sizes[h]).all()
        _check(bool(in_range.item() if isinstance(in_range, torch.Tensor) else in_range),
               f"hierarchy {h} codes in [0, {codebook_sizes[h]})")
    _check(((scores >= 0) & (scores <= 1)).all().item(), "step_scores are probabilities")

    # Beams sorted descending by marginal probability.
    marginal = scores.prod(dim=-1)  # (B, K)
    diffs = marginal[:, :-1] - marginal[:, 1:]
    _check((diffs >= -1e-6).all().item(), "beams sorted descending by marginal")

    # --- Metrics ---
    print("\nTest 4: metric init + update")
    model.init_metric()
    _check(LOSS_KEY in model._metric_modules, "tiger_ce metric registered")
    _check("recall@5" in model._metric_modules, "recall@5 registered")
    _check(f"recall@{beam_width}" in model._metric_modules, f"recall@{beam_width} registered")
    # Run update on a teacher-forced batch.
    eval_batch = _build_synth_batch(
        batch_size=4, items_per_user=3, codebook_sizes=codebook_sizes
    )
    with torch.no_grad():
        preds = model.predict(eval_batch)
    losses = model.loss(preds, eval_batch)
    model.update_metric(preds, eval_batch, losses=losses)
    result = model.compute_metric()
    print(f"  metrics computed: {sorted(result.keys())}")
    _check(LOSS_KEY in result, "compute_metric() returns tiger_ce")

    print("\n" + "=" * 68)
    print("All TIGER smoke tests PASSED")
    print("=" * 68)


if __name__ == "__main__":
    main()
