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

"""T5MultiLayerFF — multi-hidden-layer feed-forward replacement for T5LayerFF.

When TIGER's proto has `hidden_dims: "1024,1024"` (length > 1), every
T5LayerFF inside the encoder/decoder is replaced by a T5MultiLayerFF whose
MLP body uses those widths. Layout per block:

    LayerNorm -> Linear(d_model, h0) -> ReLU -> Linear(h0, h1) -> ReLU
              -> ... -> Linear(h_{n-1}, d_model) -> Dropout -> residual add

Length-1 hidden_dims (e.g. "1024") keeps the stock T5 single-hidden FF
and does not trigger this swap.
"""

from typing import List

import torch
from torch import nn
from transformers.models.t5.modeling_t5 import T5Config, T5LayerNorm


class _MLP(nn.Module):
    """Local MLP helper.

    Layout: input -> Linear(in, hidden[0]) -> [ReLU + Dropout + Linear -> ...]
    -> Linear(hidden[-1], out). Bias-on by default to match the standard T5
    FF behavior.
    """

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        hidden_dims: List[int],
        dropout: float,
    ) -> None:
        super().__init__()
        assert len(hidden_dims) >= 1
        layers: List[nn.Module] = [nn.Linear(input_dim, hidden_dims[0], bias=True)]
        for i in range(1, len(hidden_dims)):
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(dropout))
            layers.append(nn.Linear(hidden_dims[i - 1], hidden_dims[i], bias=True))
        layers.append(nn.ReLU())
        layers.append(nn.Dropout(dropout))
        layers.append(nn.Linear(hidden_dims[-1], output_dim, bias=True))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.model(x)


class T5MultiLayerFF(nn.Module):
    """Drop-in replacement for `transformers.models.t5.modeling_t5.T5LayerFF`.

    Mirrors GRID's `T5MultiLayerFF` from
    `src/models/modules/semantic_id/tiger_generation_model.py:1080`:
        layer_norm -> mlp -> dropout, with a residual add over the input.

    Args:
        config: A T5Config; reads `d_model`, `dropout_rate`,
            `layer_norm_epsilon`.
        hidden_dims: Per-MLP-layer widths, e.g. `[1024, 1024]` for a
            two-hidden-layer body. The MLP shape is
            `d_model -> hidden_dims[0] -> ... -> hidden_dims[-1] -> d_model`.
    """

    def __init__(self, config: T5Config, hidden_dims: List[int]) -> None:
        super().__init__()
        self.mlp = _MLP(
            input_dim=config.d_model,
            output_dim=config.d_model,
            hidden_dims=hidden_dims,
            dropout=config.dropout_rate,
        )
        self.layer_norm = T5LayerNorm(config.d_model, eps=config.layer_norm_epsilon)
        self.dropout = nn.Dropout(config.dropout_rate)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        forwarded = self.layer_norm(hidden_states)
        forwarded = self.mlp(forwarded)
        hidden_states = hidden_states + self.dropout(forwarded)
        return hidden_states
