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

"""ResidualQuantizer: abstract base for multi-layer residual quantizers."""

from typing import List, Tuple, Union

import torch
from torch import nn
from torch.nn import functional as F


def normalize_n_embed(n_embed: Union[int, List[int]], n_layers: int) -> List[int]:
    """Broadcast a scalar codebook size to a per-layer list (or validate one).

    Args:
        n_embed (int|List[int]): codebook size, shared or per-layer.
        n_layers (int): number of residual quantization layers.

    Returns:
        List[int]: per-layer codebook sizes, length ``n_layers``.
    """
    if isinstance(n_embed, int):
        return [n_embed] * n_layers
    assert len(n_embed) == n_layers, (
        "length of n_embed and n_layers must be same, "
        f"but got {len(n_embed)} vs {n_layers}"
    )
    return list(n_embed)


class ResidualQuantizer(nn.Module):
    """Abstract base for multi-layer residual quantization.

    Shared contract for the two SID quantizer backends — the VQ-based,
    gradient-trained :class:`ResidualVectorQuantizer` and the K-Means-based,
    offline-FAISS-trained :class:`ResidualKMeansQuantizer`. Both quantize the
    residual of the previous layer:

        residual_0 = input
        for each layer i:
            (optionally) residual_i = L2_normalize(residual_i)
            code_i, quantized_i = layer_i(residual_i)
            residual_{i+1} = residual_i - quantized_i
        output = sum of all quantized_i

    Semantic ID = (code_0, code_1, ..., code_{n_layers-1}).

    This base owns the structural invariants (``embed_dim``, ``n_layers``,
    per-layer codebook sizes, residual normalization toggle) and the shared
    residual walk (:meth:`_residual_pass`, :meth:`get_codes`,
    :meth:`decode_codes`, :meth:`output_dim`). Subclasses build ``self.layers``
    and implement the per-layer primitives :meth:`_quantize_layer` (encode) and
    :meth:`_lookup_code` (decode), plus :meth:`forward` and
    :meth:`get_codebook_embeddings`.

    Args:
        embed_dim (int): feature / codebook dimension.
        n_layers (int): number of residual quantization layers.
        n_embed (int|List[int]): codebook size per layer. Default: 256.
        normalize_residuals (bool): L2-normalize residuals before each
            layer. Default: False.
    """

    def __init__(
        self,
        embed_dim: int,
        n_layers: int,
        n_embed: Union[int, List[int]] = 256,
        normalize_residuals: bool = False,
    ) -> None:
        super().__init__()
        assert n_layers >= 1, f"n_layers must be >= 1, got {n_layers}"
        self.embed_dim = embed_dim
        self.n_layers = n_layers
        self.normalize_residuals = normalize_residuals
        self.n_embed_list = normalize_n_embed(n_embed, n_layers)
        # Subclasses MUST populate this with one quantization layer each.
        self.layers: nn.ModuleList = nn.ModuleList()

    def output_dim(self) -> int:
        """Output dimension of the module."""
        return self.embed_dim

    def forward(self, input: torch.Tensor):  # noqa: ANN201
        """Assign codes per layer and accumulate the quantized output."""
        raise NotImplementedError

    def _quantize_layer(
        self,
        layer_idx: int,
        residual: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Assign one layer's codes and look up its quantized vector.

        Backend primitive behind the residual walk (encode-direction mirror of
        :meth:`_lookup_code`).

        Args:
            layer_idx (int): quantization layer index.
            residual (Tensor): current residual, shape (B, D).

        Returns:
            codes (Tensor): per-layer cluster ids, shape (B,).
            quantized (Tensor): the layer's quantized vector, shape (B, D).
        """
        raise NotImplementedError

    def _residual_pass_with_inputs(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor], List[torch.Tensor]]:
        """Shared residual walk: per-layer assign, subtract, accumulate.

        The quantized vector is subtracted detached (keeps the residual chain
        gradient-free) and accumulated (keeps gradient when the backend
        supplies it, e.g. VQ). Also returns the exact per-layer residual inputs
        used for assignment, which candidate generation uses for last-layer KNN.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            cluster_ids (Tensor): stacked codes, shape (B, n_layers).
            aggregated (Tensor): sum of quantized vectors, shape (B, D).
            cumulative (List[Tensor]): running sum after each layer
                (``cumulative[-1] is aggregated``).
            layer_inputs (List[Tensor]): residual input to each layer after
                optional normalization.
        """
        residual = input
        all_codes: List[torch.Tensor] = []
        cumulative: List[torch.Tensor] = []
        layer_inputs: List[torch.Tensor] = []
        aggregated = torch.zeros_like(input)
        for i in range(self.n_layers):
            if self.normalize_residuals:
                residual = F.normalize(residual, dim=-1)
            layer_inputs.append(residual)
            codes, quantized = self._quantize_layer(i, residual)
            all_codes.append(codes)
            aggregated = aggregated + quantized
            cumulative.append(aggregated)
            residual = residual - quantized.detach()
        cluster_ids = torch.stack(all_codes, dim=-1)  # (B, n_layers)
        return cluster_ids, aggregated, cumulative, layer_inputs

    def _residual_pass(
        self,
        input: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor, List[torch.Tensor]]:
        """Shared residual walk without returning candidate-only state."""
        cluster_ids, aggregated, cumulative, _ = self._residual_pass_with_inputs(input)
        return cluster_ids, aggregated, cumulative

    @torch.no_grad()
    def get_codes(self, input: torch.Tensor) -> torch.Tensor:
        """Assign semantic IDs without updating the codebook.

        Shared encode-direction mirror of :meth:`decode_codes`.

        Args:
            input (Tensor): input embeddings, shape (B, D).

        Returns:
            Tensor: cluster ids, shape (B, n_layers).
        """
        cluster_ids, _, _ = self._residual_pass(input)
        return cluster_ids

    @torch.no_grad()
    def get_code_candidates(
        self,
        input: torch.Tensor,
        topk: int,
        strategy: str = "last_layer_knn",
        target_layer: int = -1,
        include_origin: bool = True,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Generate candidate SID tuples by replacing one quantization layer.

        v1 supports the collision-prevention workflow's last-layer KNN strategy:
        keep the greedy prefix unchanged, rank all codebook entries for the last
        residual, and replace only that layer's code.

        Args:
            input (Tensor): input embeddings, shape (B, D).
            topk (int): number of candidate SID tuples to return.
            strategy (str): candidate strategy. Only ``last_layer_knn`` is
                supported.
            target_layer (int): target layer. Must resolve to the final layer.
            include_origin (bool): force the original SID into slot 0.

        Returns:
            candidate_codes (Tensor): shape (B, topk, n_layers).
            candidate_scores (Tensor): shape (B, topk), lower is better.
        """
        if strategy != "last_layer_knn":
            raise ValueError("v1 supports only strategy='last_layer_knn'.")
        if topk < 1:
            raise ValueError(f"topk must be >= 1, got {topk}")
        if target_layer < 0:
            target_layer += self.n_layers
        if target_layer < 0 or target_layer >= self.n_layers:
            raise ValueError(
                f"target_layer out of range for {self.n_layers} layers: {target_layer}"
            )
        if target_layer != self.n_layers - 1:
            raise ValueError("v1 candidate generation supports only the last layer.")

        n_embed = self.n_embed_list[target_layer]
        if topk > n_embed:
            raise ValueError(
                f"topk ({topk}) must be <= target layer codebook size ({n_embed})."
            )

        cluster_ids, _, _, layer_inputs = self._residual_pass_with_inputs(input)
        residual = layer_inputs[target_layer]
        distances = self.layers[target_layer].compute_distances(residual)

        candidate_codes = cluster_ids.unsqueeze(1).expand(-1, topk, -1).clone()
        candidate_scores = distances.new_empty((input.shape[0], topk))

        offset = 0
        search_distances = distances
        if include_origin:
            origin_codes = cluster_ids[:, target_layer].unsqueeze(1)
            candidate_scores[:, 0] = distances.gather(1, origin_codes).squeeze(1)
            offset = 1
            if topk > 1:
                search_distances = distances.clone()
                search_distances.scatter_(1, origin_codes, float("inf"))

        if offset < topk:
            scores, ids = torch.topk(
                search_distances,
                k=topk - offset,
                dim=1,
                largest=False,
            )
            candidate_codes[:, offset:, target_layer] = ids
            candidate_scores[:, offset:] = scores

        return candidate_codes, candidate_scores

    @torch.no_grad()
    def get_codebook_embeddings(self, layer_idx: int) -> torch.Tensor:
        """Get the codebook (centroid) weights for a specific layer.

        Args:
            layer_idx (int): index of the quantization layer.

        Returns:
            Tensor: codebook weights, shape (n_embed, embed_dim).
        """
        raise NotImplementedError

    def _lookup_code(self, layer_idx: int, code_idx: torch.Tensor) -> torch.Tensor:
        """Look up the codebook vectors for ``code_idx`` at ``layer_idx``.

        The single backend-specific primitive :meth:`decode_codes` builds on
        (VQ reads ``embedding(idx)``, K-Means reads ``centroids[idx]``).

        Args:
            layer_idx (int): index of the quantization layer.
            code_idx (Tensor): codebook indices, shape (B,).

        Returns:
            Tensor: looked-up codebook vectors, shape (B, embed_dim).
        """
        raise NotImplementedError

    @torch.no_grad()
    def decode_codes(self, codes: torch.Tensor) -> torch.Tensor:
        """Reconstruct embeddings from semantic ID codes (centroid sum).

        Args:
            codes (Tensor): cluster ids, shape (B, n_layers).

        Returns:
            Tensor: reconstructed embeddings, shape (B, embed_dim).
        """
        # Seed from the first lookup so device and dtype follow the codebook
        # (avoids pinning the sum to fp32 under mixed precision). n_layers >= 1
        # is guaranteed by the codebook config.
        quantized_sum = self._lookup_code(0, codes[:, 0])
        for i in range(1, self.n_layers):
            quantized_sum = quantized_sum + self._lookup_code(i, codes[:, i])
        return quantized_sum
