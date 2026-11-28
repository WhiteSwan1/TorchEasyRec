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

"""Decoder-only training with HF or MCore, and decode over an HF causal LM.

This subclass asks the backbone's own forward for a suffix window of logits, so
a full (batch, length, vocab) tensor is never materialized, and it decodes by
prefilling once and stepping a self-attention cache.

The MCore branch reuses packed FlashAttention and supervised positions.
For HF, ``init_backbone`` checks the interfaces both paths need. A backbone whose
forward has no ``logits_to_keep``, or which returns a legacy tuple cache, is
not supported; Qwen2.5/Qwen3 are what CI covers.
"""

from dataclasses import dataclass
from math import prod
from typing import Any, Dict, List, Optional, Sequence, Tuple, cast

import torch
from torch.nn.attention import SDPBackend, sdpa_kernel
from transformers import AttentionInterface, AutoConfig
from transformers.integrations.sdpa_attention import sdpa_attention_forward

from tzrec.datasets.utils import Batch
from tzrec.features.feature import BaseFeature
from tzrec.models.genrec_model import _PARAM_DTYPE, BaseGenRecModel
from tzrec.modules.beam_cache import create_beam_cache
from tzrec.modules.dynamic_beam import capped_beam_widths, dynamic_beam_search
from tzrec.prompt.assembler import (
    CU_SEQLENS,
    INPUT_IDS,
    MAX_SEQLEN,
    RESPONSE_LENGTHS,
)
from tzrec.prompt.types import CompiledPrompt, SlotSeg
from tzrec.protos.model_pb2 import ModelConfig
from tzrec.protos.models.genrec_model_pb2 import (
    GenRecCausalLMModel as CausalLMConfig,
)
from tzrec.protos.models.genrec_model_pb2 import (
    GenRecModelConfig,
)


def _enable_mlp_fp8(linear: torch.nn.Module) -> None:
    """Use current-scaling FP8 only inside one MLP linear operation.

    Args:
        linear: MCore Transformer Engine linear with framework-owned parameters.
    """
    from transformer_engine.common.recipe import Float8CurrentScaling, Format
    from transformer_engine.pytorch import fp8_autocast

    recipe = Float8CurrentScaling(fp8_format=Format.E4M3)
    original_forward = linear.forward
    # TorchRec owns updates and does not drive MCore's weight-cache lifecycle.
    linear.disable_parameter_transpose_cache = True

    def fp8_forward(value: torch.Tensor) -> Tuple[torch.Tensor, Any]:
        """Pad token rows for FP8 GEMMs without changing attention boundaries."""
        shape = value.shape
        rows = prod(shape[:-1])
        padding = -rows % 16
        if padding:
            value = torch.nn.functional.pad(
                value.reshape(rows, shape[-1]), (0, 0, 0, padding)
            )
        with fp8_autocast(enabled=True, fp8_recipe=recipe):
            output, bias = original_forward(value)
        if padding:
            output = output[:rows].reshape(*shape[:-1], output.shape[-1])
        return output, bias

    linear.forward = fp8_forward


class _GatherFP32Grad(torch.autograd.Function):
    """Gather BF16 activations while accumulating repeated gradients in FP32."""

    @staticmethod
    def forward(ctx, value, indices):
        """Preserve the original gather values without float intermediate copies."""
        ctx.save_for_backward(indices)
        ctx.shape = value.shape
        return value.index_select(0, indices)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Sum repeated rows before rounding back to the activation dtype."""
        (grad,) = grad_outputs
        (indices,) = ctx.saved_tensors
        result = torch.zeros(ctx.shape, dtype=torch.float32, device=grad.device)
        result.index_add_(0, indices, grad.float())
        return result.to(grad.dtype), None


class _TENormOutput(torch.autograd.Function):
    """Keep native FP32 RMSNorm math while selecting its output storage dtype."""

    @staticmethod
    def forward(
        ctx,
        x,
        weight,
        eps,
        output_dtype,
        forward_margin,
        zero_centered,
    ):
        """Preserve native forward input and weight precision."""
        from transformer_engine.pytorch.constants import TE_DType
        from transformer_engine_torch import rmsnorm_fwd

        flat = x.contiguous().view(-1, weight.numel())
        output, _, rstdev = rmsnorm_fwd(
            flat,
            weight,
            eps,
            None,
            None,
            TE_DType[output_dtype],
            forward_margin,
            zero_centered,
        )
        ctx.save_for_backward(flat, weight, rstdev)
        ctx.shape = x.shape
        ctx.eps = eps

        ctx.zero_centered = zero_centered
        return output.view(x.shape)

    @staticmethod
    def backward(ctx: Any, *grad_outputs: Any) -> Any:
        """Read the upstream gradient directly and accumulate dx/dw in FP32."""
        (grad,) = grad_outputs
        x, weight, rstdev = ctx.saved_tensors
        grad = grad.contiguous().view_as(x)
        from flash_attn.ops.triton.layer_norm import _layer_norm_bwd

        dx, dw = _layer_norm_bwd(
            grad,
            x,
            weight,
            None,
            ctx.eps,
            None,
            rstdev,
            is_rms_norm=True,
            x_dtype=x.dtype,
            zero_centered_weight=ctx.zero_centered,
        )[:2]
        return dx.view(ctx.shape), dw.view_as(weight), None, None, None, None


@dataclass
class _MultiTargetLayout:
    """Training-only SID branches that share each row's packed prompt prefix.

    Tokens keep the assembler's packed order. Every target is one attention
    segment whose keys are its row's prefix followed by its own tokens, so
    causal attention aligned to the last key isolates targets. A row's first
    segment also holds its prefix queries, so only later targets copy keys.

    Args:
        position_ids: packed positions, restarting after the prefix per target.
        cu_seqlens_q: int32 query boundaries of the segments.
        cu_seqlens_k: int32 key boundaries of the same segments.
        kv_indices: packed token behind every segment key/value.
        max_seqlen: upper bound of any segment's query or key length.
        label_indices: packed indices of target tokens, shaped (targets, levels).
        score_indices: packed indices whose logits predict those labels.
        target_weights: inverse target count of each target's row.
        token_targets: target index of every packed token, -1 in a prefix.
    """

    position_ids: torch.Tensor
    cu_seqlens_q: torch.Tensor
    cu_seqlens_k: torch.Tensor
    kv_indices: torch.Tensor
    max_seqlen: int
    label_indices: torch.Tensor
    score_indices: torch.Tensor
    target_weights: torch.Tensor
    token_targets: torch.Tensor


def _shared_prefix_attention(
    module: torch.nn.Module,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: Any,
    scaling: Optional[float] = None,
    dropout: float = 0.0,
    *,
    multi_target_layout: _MultiTargetLayout,
    **kwargs: Any,
) -> Tuple[torch.Tensor, None]:
    """Run one FA2 varlen call over the prefix and target segments.

    Every target segment reads its row's prefix keys before its own, so the
    causal mask, aligned to the last key, hides the row's other targets.

    Args:
        module: the calling attention module; unused.
        query: packed queries shaped (1, query heads, total tokens, head dim).
        key: packed keys, retaining the backbone's grouped KV heads.
        value: packed values with the same layout as keys.
        attention_mask: required by the HF attention interface; always None
            here, the segments carry the visibility.
        scaling: backbone attention scale.
        dropout: attention dropout probability, 0 outside training.
        multi_target_layout: batch-local segments and key gather indices.
        **kwargs: unused backbone attention options.

    Returns:
        Packed attention output and no attention weights.
    """
    from flash_attn import flash_attn_varlen_func

    layout = multi_target_layout
    # rotary embedding may promote q/k to fp32; v keeps the projection dtype
    query, key = query.to(value.dtype), key.to(value.dtype)
    output = flash_attn_varlen_func(
        query[0].transpose(0, 1),
        key[0].transpose(0, 1).index_select(0, layout.kv_indices),
        value[0].transpose(0, 1).index_select(0, layout.kv_indices),
        cu_seqlens_q=layout.cu_seqlens_q,
        cu_seqlens_k=layout.cu_seqlens_k,
        max_seqlen_q=layout.max_seqlen,
        max_seqlen_k=layout.max_seqlen,
        dropout_p=dropout,
        softmax_scale=scaling,
        causal=True,
    )
    return output.unsqueeze(0), None


_SHARED_PREFIX_ATTENTION = "genrec_shared_prefix_fa2"
AttentionInterface.register(_SHARED_PREFIX_ATTENTION, _shared_prefix_attention)


def _padded_sdpa_attention(*args: Any, **kwargs: Any) -> Tuple[torch.Tensor, None]:
    """Run HF's SDPA attention without cuDNN for a left-padded multi-target batch.

    cuDNN's masked SDPA backward returns NaN dq at lengths that are odd
    multiples of 64 (torch 2.14, cuDNN 9.24).

    Args:
        *args: HF attention interface arguments.
        **kwargs: HF attention interface options.

    Returns:
        Attention output and no attention weights.
    """
    with sdpa_kernel([SDPBackend.EFFICIENT_ATTENTION, SDPBackend.MATH]):
        return sdpa_attention_forward(*args, **kwargs)


_PADDED_SDPA_ATTENTION = "genrec_padded_sdpa"
AttentionInterface.register(_PADDED_SDPA_ATTENTION, _padded_sdpa_attention)


class _MCoreFlashAttention(torch.nn.Module):
    """Use packed FlashAttention inside an MCore attention layer.

    Args:
        config: MCore transformer configuration.
        attn_kernel: FA2 or FA3 implementation for the packed attention core.
        fp32_kv_grad: accumulate repeated packed K/V gradients in FP32.
        **kwargs: attention construction arguments supplied by MCore.
    """

    def __init__(
        self,
        config: Any,
        attn_kernel: str = "flash_attention_2",
        fp32_kv_grad: bool = False,
        **kwargs: Any,
    ) -> None:
        super().__init__()
        self.dropout = config.attention_dropout
        self.attn_kernel = attn_kernel
        self.fp32_kv_grad = fp32_kv_grad
        if attn_kernel == "flash_attention_3" and self.dropout != 0:
            raise ValueError("FlashAttention-3 requires attention_dropout=0.")

    def forward(
        self,
        query: torch.Tensor,
        key: torch.Tensor,
        value: torch.Tensor,
        attention_mask: Any,
        *,
        packed_seq_params: Any,
        **kwargs: Any,
    ) -> torch.Tensor:
        """Rotate unique tokens before gathering each target's prefix K/V.

        Args:
            query: packed queries, shaped (tokens, query heads, head dim).
            key: unique packed keys with grouped KV heads.
            value: unique packed values with grouped KV heads.
            attention_mask: unused; the packed segments define visibility.
            packed_seq_params: boundaries, rotary values and optional target layout.
            **kwargs: unused MCore attention options.

        Returns:
            Attention output in the packed query order.
        """
        params = packed_seq_params
        if getattr(params, "genrec_fused_rope_cos", None) is not None:
            from flash_attn.layers.rotary import apply_rotary_emb

            query = apply_rotary_emb(
                query.unsqueeze(0),
                params.genrec_fused_rope_cos,
                params.genrec_fused_rope_sin,
                inplace=False,
            )[0]
            key = apply_rotary_emb(
                key.unsqueeze(0),
                params.genrec_fused_rope_cos,
                params.genrec_fused_rope_sin,
                inplace=False,
            )[0]
        else:
            from transformers.models.qwen2.modeling_qwen2 import apply_rotary_pos_emb

            query, key = apply_rotary_pos_emb(
                query,
                key,
                params.genrec_cos,
                params.genrec_sin,
                unsqueeze_dim=1,
            )
        dropout = self.dropout if self.training else 0.0
        if self.attn_kernel == "flash_attention_3":
            from flash_attn_3.flash_attn_interface import flash_attn_varlen_func

            layout = params.genrec_layout
            self.scored_indices = None
            if getattr(self, "score_final_queries", False) and layout is not None:
                self.scored_indices = layout.score_indices.reshape(-1)
                query = query.index_select(0, self.scored_indices)
                indices = params.genrec_final_kv_indices
                if self.fp32_kv_grad:
                    key = _GatherFP32Grad.apply(key, indices)
                    value = _GatherFP32Grad.apply(value, indices)
                else:
                    key = key.index_select(0, indices)
                    value = value.index_select(0, indices)
                return flash_attn_varlen_func(
                    query.to(value.dtype),
                    key.to(value.dtype),
                    value,
                    cu_seqlens_q=params.genrec_final_cu_q,
                    cu_seqlens_k=params.genrec_final_cu_k,
                    max_seqlen_q=layout.score_indices.shape[1],
                    max_seqlen_k=params.max_seqlen_kv,
                    causal=True,
                    num_splits=1,
                )
            if layout is not None:
                if self.fp32_kv_grad:
                    key = _GatherFP32Grad.apply(key, layout.kv_indices)
                    value = _GatherFP32Grad.apply(value, layout.kv_indices)
                else:
                    key = key.index_select(0, layout.kv_indices)
                    value = value.index_select(0, layout.kv_indices)
            return flash_attn_varlen_func(
                query.to(value.dtype),
                key.to(value.dtype),
                value,
                cu_seqlens_q=params.cu_seqlens_q,
                cu_seqlens_k=params.cu_seqlens_kv,
                max_seqlen_q=params.max_seqlen_q,
                max_seqlen_k=params.max_seqlen_kv,
                causal=True,
                num_splits=1,
            )
        from flash_attn import flash_attn_varlen_func

        if params.genrec_layout is not None:
            output, _ = _shared_prefix_attention(
                self,
                query.transpose(0, 1).unsqueeze(0),
                key.transpose(0, 1).unsqueeze(0),
                value.transpose(0, 1).unsqueeze(0),
                None,
                dropout=dropout,
                multi_target_layout=params.genrec_layout,
            )
            return output[0]
        return flash_attn_varlen_func(
            query.to(value.dtype),
            key.to(value.dtype),
            value,
            cu_seqlens_q=params.cu_seqlens_q,
            cu_seqlens_k=params.cu_seqlens_kv,
            max_seqlen_q=params.max_seqlen_q,
            max_seqlen_k=params.max_seqlen_kv,
            dropout_p=dropout,
            causal=True,
        )


class GenRecCausalLMModel(BaseGenRecModel):
    """A causal LM driven by a compiled prompt and the selected compute backend.

    Args:
        model_config: the model oneof.
        features: every created feature.
        labels: data_config label fields.
        sample_weights: optional sample weight fields.
        compiled_prompt: the compiled prompt; required.
    """

    _use_mcore = False

    def __init__(
        self,
        model_config: ModelConfig,
        features: List[BaseFeature],
        labels: List[str],
        sample_weights: Optional[List[str]] = None,
        compiled_prompt: Optional[CompiledPrompt] = None,
        **kwargs: Any,
    ) -> None:
        super().__init__(
            model_config,
            features,
            labels,
            sample_weights,
            compiled_prompt=compiled_prompt,
            **kwargs,
        )
        common = self._model_config.common
        self._generated_sids_key = common.generated_sids_key
        self._read_beam_config(common)
        response = self._prompt.prompt_plan.response_segments
        self._sid_only_response = len(response) == 1 and isinstance(
            response[0], SlotSeg
        )

    def init_backbone(
        self,
        hf_model_name_or_path: str,
        lm_parameter_dtype: int,
        attn_kernel: str,
    ) -> None:
        """Construct the selected decoder inside the existing GenRec model.

        Args:
            hf_model_name_or_path: local HF architecture and pretrained weights.
            lm_parameter_dtype: parameter precision, independent of autocast.
            attn_kernel: attention implementation; MCore supports FA2 and FA3.
        """
        self._use_mcore = self._model_config.lm_backend == CausalLMConfig.MEGATRON_CORE
        packed_optimizations = (
            self._model_config.mcore_share_batch_prefix
            or self._model_config.mcore_prune_final_layer
        )
        if packed_optimizations and (
            not self._use_mcore or attn_kernel != "flash_attention_3"
        ):
            raise ValueError(
                "Packed MCore optimizations require MEGATRON_CORE with FA3."
            )
        if (
            self._model_config.mcore_fused_norm
            or self._model_config.mcore_fused_residual
            or self._model_config.mcore_mlp_fp8
        ) and not self._use_mcore:
            raise ValueError("MCore compute options require MEGATRON_CORE.")
        if not self._use_mcore:
            super().init_backbone(
                hf_model_name_or_path, lm_parameter_dtype, attn_kernel
            )
            return

        from megatron.core.extensions.transformer_engine import (
            TEColumnParallelLinear,
            TENorm,
        )
        from megatron.core.models.gpt.gpt_layer_specs import (
            get_gpt_layer_with_transformer_engine_spec,
        )
        from megatron.core.process_groups_config import ProcessGroupCollection
        from megatron.core.transformer.spec_utils import ModuleSpec
        from megatron.core.transformer.transformer_block import (
            TransformerBlock,
            TransformerBlockSubmodules,
        )
        from megatron.core.transformer.transformer_config import TransformerConfig

        hf_config = AutoConfig.from_pretrained(hf_model_name_or_path)
        if hf_config.model_type not in ("qwen2", "qwen3") or attn_kernel not in (
            "flash_attention_2",
            "flash_attention_3",
        ):
            raise ValueError(
                "MEGATRON_CORE supports dense Qwen2/2.5/3 with FA2 or FA3."
            )
        is_qwen3 = hf_config.model_type == "qwen3"
        if is_qwen3:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RotaryEmbedding

            rotary_class = Qwen3RotaryEmbedding
        else:
            from transformers.models.qwen2.modeling_qwen2 import Qwen2RotaryEmbedding

            rotary_class = Qwen2RotaryEmbedding
        if getattr(hf_config, "use_sliding_window", False):
            raise ValueError("MEGATRON_CORE does not support sliding-window attention.")
        if not torch.distributed.is_initialized():
            raise RuntimeError("Initialize torch.distributed before MEGATRON_CORE.")

        group = cast(
            torch.distributed.ProcessGroup,
            torch.distributed.new_group(
                ranks=[torch.distributed.get_rank()], use_local_synchronization=True
            ),
        )
        groups = ProcessGroupCollection()
        groups.tp = groups.pp = groups.cp = groups.ep = groups.expt_tp = group
        groups.gtp_remat = groups.expt_gtp_remat = group
        groups.dp = groups.dp_cp = cast(
            torch.distributed.ProcessGroup, torch.distributed.group.WORLD
        )
        config = TransformerConfig(
            num_layers=hf_config.num_hidden_layers,
            hidden_size=hf_config.hidden_size,
            ffn_hidden_size=hf_config.intermediate_size,
            num_attention_heads=hf_config.num_attention_heads,
            num_query_groups=hf_config.num_key_value_heads,
            kv_channels=getattr(
                hf_config,
                "head_dim",
                hf_config.hidden_size // hf_config.num_attention_heads,
            ),
            normalization="RMSNorm",
            layernorm_epsilon=hf_config.rms_norm_eps,
            activation_func=torch.nn.functional.silu,
            gated_linear_unit=True,
            add_bias_linear=False,
            add_qkv_bias=getattr(hf_config, "attention_bias", not is_qwen3),
            qk_layernorm=is_qwen3,
            hidden_dropout=0.0,
            attention_dropout=hf_config.attention_dropout,
            params_dtype=_PARAM_DTYPE[lm_parameter_dtype],
            use_cpu_initialization=True,
            gradient_accumulation_fusion=False,
            bias_activation_fusion=True,
            bias_dropout_fusion=self._model_config.mcore_fused_residual,
        )
        spec = get_gpt_layer_with_transformer_engine_spec()

        def build_norm(config: Any, hidden_size: int, eps: float) -> torch.nn.Module:
            """Preserve Qwen's parameter precision for RMSNorm under autocast.

            Args:
                config: transformer configuration and parameter precision.
                hidden_size: normalized channel count.
                eps: variance stabilizer from the pretrained configuration.

            Returns:
                Transformer Engine RMSNorm preserving input and weight precision.
            """
            norm = cast(torch.nn.Module, TENorm(config, hidden_size, eps))
            if self._model_config.mcore_fused_norm:

                def te_norm_output(x):
                    """Preserve forward statistics and fuse the backward cast."""
                    output_dtype = (
                        torch.get_autocast_dtype("cuda")
                        if torch.is_autocast_enabled()
                        else x.dtype
                    )
                    stage = "forward" if torch.is_grad_enabled() else "inference"
                    return _TENormOutput.apply(
                        x,
                        norm.weight,
                        eps,
                        output_dtype,
                        cast(Dict[str, int], norm._sm_margins)[stage],
                        norm.zero_centered_gamma,
                    )

                norm.forward = te_norm_output
            else:
                norm.forward = torch.autocast("cuda", enabled=False)(norm.forward)
            return norm

        spec.submodules.input_layernorm = build_norm
        spec.submodules.pre_mlp_layernorm = build_norm
        spec.submodules.self_attention.submodules.linear_qkv = TEColumnParallelLinear
        spec.submodules.self_attention.submodules.core_attention = ModuleSpec(
            module=_MCoreFlashAttention,
            params={
                "attn_kernel": attn_kernel,
                "fp32_kv_grad": self._model_config.mcore_share_batch_prefix
                or self._model_config.mcore_prune_final_layer,
            },
        )
        if is_qwen3:
            from transformers.models.qwen3.modeling_qwen3 import Qwen3RMSNorm

            def build_qk_norm(
                config: Any, hidden_size: int, eps: float
            ) -> torch.nn.Module:
                """Keep Qwen3's rounding before its FP32 Q/K norm weight multiply.

                Args:
                    config: transformer configuration and parameter precision.
                    hidden_size: channels per attention head.
                    eps: variance stabilizer from the pretrained configuration.

                Returns:
                    The HF Qwen3 per-head normalization module.
                """
                return Qwen3RMSNorm(hidden_size, eps=eps).to(dtype=config.params_dtype)

            spec.submodules.self_attention.submodules.q_layernorm = build_qk_norm
            spec.submodules.self_attention.submodules.k_layernorm = build_qk_norm
        spec.submodules.mlp.keywords["submodules"].linear_fc1 = TEColumnParallelLinear
        block_spec = TransformerBlockSubmodules(
            layer_specs=[spec] * config.num_layers, layer_norm=build_norm
        )
        hf_config.vocab_size = self._prompt.sid_space.target_vocab_size
        self.lm = torch.nn.Module()
        self.lm.config = hf_config
        self.lm.embedding = torch.nn.Module()
        self.lm.embedding.word_embeddings = torch.nn.Embedding(
            hf_config.vocab_size, hf_config.hidden_size, dtype=config.params_dtype
        )
        self.lm.decoder = TransformerBlock(config, block_spec, pg_collection=groups)
        if self._model_config.mcore_mlp_fp8:
            for layer in self.lm.decoder.layers:
                _enable_mlp_fp8(layer.mlp.linear_fc1)
                _enable_mlp_fp8(layer.mlp.linear_fc2)
        self._prune_final_layer = self._model_config.mcore_prune_final_layer
        if self._prune_final_layer:
            last = self.lm.decoder.layers[-1]
            core = last.self_attention.core_attention
            core.score_final_queries = True
            core.scored_indices = None
            original_bda = last.attention_bda_and_cross_attention

            def scored_attention_residual(output, residual, *args, **kwargs):
                """Align the residual with the supervised attention queries."""
                if core.scored_indices is not None:
                    residual = residual.index_select(0, core.scored_indices)
                return original_bda(output, residual, *args, **kwargs)

            last.attention_bda_and_cross_attention = scored_attention_residual
        self.lm.output_layer = torch.nn.Linear(
            hf_config.hidden_size,
            hf_config.vocab_size,
            bias=False,
            dtype=config.params_dtype,
        )
        if hf_config.tie_word_embeddings:
            self.lm.output_layer.weight = self.lm.embedding.word_embeddings.weight
        self.lm.rotary_emb = rotary_class(config=hf_config)

    def input_embeddings(self) -> torch.nn.Module:
        """Return the same token lookup used by the selected decoder."""
        if self._use_mcore:
            return self.lm.embedding.word_embeddings
        return super().input_embeddings()

    def causal_lm_loss(self, **kwargs: Any) -> torch.Tensor:
        """Retain HF causal shifting for non-SID response windows."""
        if self._use_mcore:
            from transformers.loss.loss_utils import ForCausalLMLoss

            return ForCausalLMLoss(**kwargs)
        return super().causal_lm_loss(**kwargs)

    def load_hf_state_dict(self, state: Dict[str, torch.Tensor]) -> None:
        """Load HF weights using the same layout mapping as export."""
        if not self._use_mcore:
            super().load_hf_state_dict(state)
            return
        from tzrec.utils.hf_export_util import convert_qwen_mcore_state

        converted = convert_qwen_mcore_state(state, self.lm.config, to_mcore=True)
        converted.update(
            {
                key: value
                for key, value in self.lm.state_dict().items()
                if key.endswith("_extra_state")
            }
        )
        self.lm.load_state_dict(converted)

    def _mcore_hidden(
        self,
        embeds: torch.Tensor,
        position_ids: torch.Tensor,
        cu_seqlens: torch.Tensor,
        max_seqlen: int,
        layout: Optional[_MultiTargetLayout] = None,
    ) -> torch.Tensor:
        """Run the decoder with the existing packed positions and target layout.

        Args:
            embeds: unique input tokens, including differentiable projected slots.
            position_ids: rotary position of each unique token.
            cu_seqlens: packed row boundaries for a single-target batch.
            max_seqlen: maximum query/key segment length.
            layout: optional shared-prefix target branches.

        Returns:
            Hidden states in packed order, or supervised order when pruning.
        """
        from megatron.core.packed_seq_params import PackedSeqParams

        params = PackedSeqParams(
            qkv_format="thd",
            cu_seqlens_q=cu_seqlens if layout is None else layout.cu_seqlens_q,
            cu_seqlens_kv=cu_seqlens if layout is None else layout.cu_seqlens_k,
            max_seqlen_q=max_seqlen,
            max_seqlen_kv=max_seqlen,
        )
        cos, sin = self.lm.rotary_emb(embeds.unsqueeze(0), position_ids.reshape(1, -1))
        params.genrec_cos, params.genrec_sin = cos[0], sin[0]
        if self._model_config.mcore_fused_rope:
            half = cos.shape[-1] // 2
            params.genrec_fused_rope_cos = cos[0, :, :half].contiguous()
            params.genrec_fused_rope_sin = sin[0, :, :half].contiguous()
        params.genrec_layout = layout
        if self._prune_final_layer and layout is not None:
            segments = torch.searchsorted(
                layout.cu_seqlens_q[1:].long(),
                layout.score_indices[:, -1].contiguous(),
                right=True,
            )
            key_lengths = layout.cu_seqlens_k.long().diff()[segments] - 1
            total_keys = int(key_lengths.sum())
            cu_keys = torch.nn.functional.pad(key_lengths.cumsum(0), (1, 0))
            key_segments = torch.repeat_interleave(
                torch.arange(segments.numel(), device=embeds.device),
                key_lengths,
                output_size=total_keys,
            )
            offsets = (
                torch.arange(total_keys, device=embeds.device) - cu_keys[key_segments]
            )
            old_positions = layout.cu_seqlens_k.long()[segments[key_segments]] + offsets
            params.genrec_final_kv_indices = layout.kv_indices[old_positions]
            params.genrec_final_cu_k = cu_keys.int()
            params.genrec_final_cu_q = (
                torch.arange(
                    segments.numel() + 1,
                    device=embeds.device,
                    dtype=torch.int32,
                )
                * layout.score_indices.shape[1]
            )
        return self.lm.decoder(
            hidden_states=embeds.unsqueeze(1),
            attention_mask=None,
            packed_seq_params=params,
        )[:, 0]

    def _read_beam_config(self, common: GenRecModelConfig) -> None:
        """Parse the decode knobs; the schedule must match the codebook.

        Args:
            common: the shared model config.
        """
        space = self._prompt.sid_space
        self._bands: List[Tuple[int, int]] = list(zip(space.band_lo, space.band_hi))
        self._capped_widths = resolve_beam_config(common, space.codebook)

    def predict(self, batch: Batch) -> Dict[str, torch.Tensor]:
        """Run teacher-forced loss or inference decode over the assembled stream.

        Args:
            batch: carries the packed prompt in ``additional_infos``.

        Returns:
            The response-window logits and labels when training, the decoded
            SIDs otherwise.
        """
        # outside the leaf on both paths, so the pipeline can prefetch it
        embeds = self.build_input(batch)
        if self.is_inference:
            return {self._generated_sids_key: _fx_wrapped_generate(self, embeds, batch)}
        predictions = _fx_wrapped_forward(self, embeds, batch)
        keys = ("logits", "labels")
        if self._sid_only_response:
            keys += ("target_weights",)
        return {key: predictions[key] for key in keys}

    def _multi_target_predictions(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Dict[str, torch.Tensor]:
        """Score independent SID branches over one prompt per row.

        FA2 keeps the batch packed; other kernels run left-padded SDPA whose
        per-row mask hides each target from its siblings.

        Args:
            embeds: packed embeddings, retaining one prefix per row.
            batch: existing assembler row boundaries and response lengths.

        Returns:
            Per-target logits and labels, and each target's row weight.
        """
        infos = batch.additional_infos
        layout = self._build_multi_target_layout(
            infos[CU_SEQLENS],
            infos[RESPONSE_LENGTHS],
            embeds.shape[0],
        )
        if self._use_mcore:
            if self._model_config.mcore_share_batch_prefix:
                embeds, layout = self._share_static_batch_prefix(embeds, infos, layout)
            hidden = self._mcore_hidden(
                embeds,
                layout.position_ids,
                infos[CU_SEQLENS],
                layout.max_seqlen,
                layout,
            )
            return {
                "logits": self.lm.output_layer(
                    hidden.view(*layout.score_indices.shape, -1)
                    if self._prune_final_layer
                    else hidden[layout.score_indices]
                ),
                "labels": infos[INPUT_IDS][layout.label_indices],
                "target_weights": layout.target_weights,
            }
        if self._attn_kernel == "flash_attention_2":
            attention, attention_mask = _SHARED_PREFIX_ATTENTION, None
            inputs_embeds = embeds.unsqueeze(0)
            position_ids = layout.position_ids.unsqueeze(0)
            token_mask = None
        else:
            attention = _PADDED_SDPA_ATTENTION
            inputs_embeds, padding_mask = _left_pad(
                embeds, infos[CU_SEQLENS], layout.max_seqlen
            )
            token_mask = padding_mask.bool()
            position_ids = torch.zeros_like(padding_mask)
            position_ids[token_mask] = layout.position_ids
            # -2 marks left padding, which causality keeps attending to itself
            targets = torch.full_like(padding_mask, -2)
            targets[token_mask] = layout.token_targets
            key_targets = targets[:, None, :]
            attention_mask = (
                ((key_targets == -1) | (key_targets == targets[:, :, None]))
                .tril()
                .unsqueeze(1)
            )
        previous_attention = self.lm.config._attn_implementation
        self.lm.set_attn_implementation(attention)
        hidden = self.lm.base_model(
            inputs_embeds=inputs_embeds,
            attention_mask={"full_attention": attention_mask},
            position_ids=position_ids,
            use_cache=False,
            multi_target_layout=layout,
        ).last_hidden_state
        self.lm.set_attn_implementation(previous_attention)
        hidden = hidden[0] if token_mask is None else hidden[token_mask]
        return {
            "logits": self.lm.get_output_embeddings()(hidden[layout.score_indices]),
            "labels": infos[INPUT_IDS][layout.label_indices],
            "target_weights": layout.target_weights,
        }

    def _share_static_batch_prefix(
        self, embeds: torch.Tensor, infos: Dict[str, Any], layout: _MultiTargetLayout
    ) -> Tuple[torch.Tensor, _MultiTargetLayout]:
        """Reuse request-invariant causal prefix activations within this batch."""
        prefix = self._prompt.prompt_plan.static_prefix_len
        if prefix == 0:
            return embeds, layout
        cu = infos[CU_SEQLENS].long()
        lengths = cu.diff()
        rows = torch.repeat_interleave(
            torch.arange(lengths.numel(), device=embeds.device),
            lengths,
            output_size=embeds.shape[0],
        )
        offsets = torch.arange(embeds.shape[0], device=embeds.device) - cu[rows]
        keep = ((rows == 0) | (offsets >= prefix)).nonzero().flatten()
        mapping = torch.where(
            offsets < prefix,
            offsets,
            torch.arange(embeds.shape[0], device=embeds.device) - rows * prefix,
        )
        counts = (infos[RESPONSE_LENGTHS] // self._prompt.sid_space.num_levels).clamp(
            min=1
        )
        first_segments = counts.cumsum(0) - counts
        query_lengths = layout.cu_seqlens_q.long().diff()
        query_lengths[first_segments.long()] -= prefix
        query_lengths = torch.cat((query_lengths.new_tensor([prefix]), query_lengths))
        key_lengths = torch.cat(
            (query_lengths.new_tensor([prefix]), layout.cu_seqlens_k.long().diff())
        )
        compact = _MultiTargetLayout(
            position_ids=layout.position_ids[keep],
            cu_seqlens_q=torch.nn.functional.pad(query_lengths.cumsum(0), (1, 0)).int(),
            cu_seqlens_k=torch.nn.functional.pad(key_lengths.cumsum(0), (1, 0)).int(),
            kv_indices=torch.cat(
                (torch.arange(prefix, device=embeds.device), mapping[layout.kv_indices])
            ),
            max_seqlen=layout.max_seqlen,
            label_indices=layout.label_indices,
            score_indices=mapping[layout.score_indices],
            target_weights=layout.target_weights,
            token_targets=layout.token_targets[keep],
        )
        return embeds.index_select(0, keep), compact

    def _build_multi_target_layout(
        self,
        cu_seqlens: torch.Tensor,
        response_lengths: torch.Tensor,
        num_tokens: int,
    ) -> _MultiTargetLayout:
        """Split each packed row into its shared prefix and independent SID targets.

        Args:
            cu_seqlens: packed row boundaries from the assembler.
            response_lengths: supervised SID token count of each row.
            num_tokens: packed token count, equal to ``cu_seqlens[-1]``.

        Returns:
            Positions, attention segments and scoring indices over packed tokens.
        """
        num_levels = self._prompt.sid_space.num_levels
        device = cu_seqlens.device
        cu_seqlens = cu_seqlens.to(torch.int64)
        row_lengths = torch.diff(cu_seqlens)
        prefix_lengths = row_lengths - response_lengths
        target_counts = response_lengths // num_levels
        # the first target shares its row's prefix segment
        segment_counts = target_counts.clamp(min=1)
        # one host read sizes every repeat below
        num_targets, num_segments, num_keys, max_seqlen = torch.stack(
            (
                target_counts.sum(),
                segment_counts.sum(),
                (prefix_lengths * segment_counts + response_lengths).sum(),
                row_lengths.max(),
            )
        ).tolist()
        row_ids = torch.arange(row_lengths.numel(), device=device)

        segment_ids = torch.arange(num_segments, device=device)
        segment_rows = torch.repeat_interleave(
            row_ids, segment_counts, output_size=num_segments
        )
        # i for a row's i-th target; segment 0 also holds the prefix
        segment_ranks = (
            segment_ids - (segment_counts.cumsum(0) - segment_counts)[segment_rows]
        )
        segment_prefix_lengths = prefix_lengths[segment_rows]
        key_lengths = segment_prefix_lengths + num_levels * (
            target_counts[segment_rows] > 0
        )
        query_lengths = torch.where(segment_ranks > 0, num_levels, key_lengths)
        cu_seqlens_q = torch.nn.functional.pad(query_lengths.cumsum(0), (1, 0))
        cu_seqlens_k = torch.nn.functional.pad(key_lengths.cumsum(0), (1, 0))

        token_segments = torch.repeat_interleave(
            segment_ids, query_lengths, output_size=num_tokens
        )
        # a query's position counts the keys before it in its segment
        position_ids = (
            torch.arange(num_tokens, device=device)
            - (cu_seqlens_q[:-1] + query_lengths - key_lengths)[token_segments]
        )
        segment_targets = (target_counts.cumsum(0) - target_counts)[
            segment_rows
        ] + segment_ranks
        token_targets = torch.where(
            position_ids < segment_prefix_lengths[token_segments],
            -1,
            segment_targets[token_segments],
        )

        key_segments = torch.repeat_interleave(
            segment_ids, key_lengths, output_size=num_keys
        )
        key_offsets = torch.arange(num_keys, device=device) - cu_seqlens_k[key_segments]
        key_rows = segment_rows[key_segments]
        # past the prefix, a target's own keys skip the row's earlier targets
        kv_indices = (
            cu_seqlens[key_rows]
            + key_offsets
            + (key_offsets >= prefix_lengths[key_rows])
            * segment_ranks[key_segments]
            * num_levels
        )

        target_rows = torch.repeat_interleave(
            row_ids, target_counts, output_size=num_targets
        )
        # rows without targets still own a prefix segment
        target_segments = (
            torch.arange(num_targets, device=device)
            + (target_counts == 0).cumsum(0)[target_rows]
        )
        # a target's tokens close its segment
        label_indices = cu_seqlens_q[target_segments + 1, None] - torch.arange(
            num_levels, 0, -1, device=device
        )
        score_indices = label_indices - 1
        score_indices[:, 0] = cu_seqlens[target_rows] + prefix_lengths[target_rows] - 1

        return _MultiTargetLayout(
            position_ids=position_ids,
            cu_seqlens_q=cu_seqlens_q.to(torch.int32),
            cu_seqlens_k=cu_seqlens_k.to(torch.int32),
            kv_indices=kv_indices,
            max_seqlen=max_seqlen,
            label_indices=label_indices,
            score_indices=score_indices,
            target_weights=target_counts[target_rows].float().reciprocal(),
            token_targets=token_targets,
        )

    def _forward(
        self, embeds: torch.Tensor, batch: Batch
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Run the LM over the assembled embeddings and cut the response window.

        Body and head are called separately so logits cover the supervised
        window only: a full (batch, length, vocab) upcast does not fit.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.

        Returns:
            Logits and labels over the same window, so the shift ``loss``
            applies lands on the pairs the window was sized for.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[CU_SEQLENS]
        suffix = cast(int, self._prompt.prompt_plan.logits_suffix_len)
        suffix_offsets = torch.arange(-suffix, 0, device=embeds.device)
        # (rows, suffix): the window each row is supervised over, the same
        # absolute positions whichever layout the kernel is given
        keep = cu_seqlens[1:, None] + suffix_offsets
        labels = infos[INPUT_IDS][keep].masked_fill(
            suffix_offsets[None, :] < -infos[RESPONSE_LENGTHS][:, None],
            self._ignore_index,
        )
        if self._use_mcore or self._attn_kernel == "flash_attention_2":
            logits = self._varlen_logits(embeds, batch, keep)
        else:
            logits = self._padded_logits(embeds, batch, suffix)
        return logits, labels

    def _varlen_logits(
        self, embeds: torch.Tensor, batch: Batch, keep: torch.Tensor
    ) -> torch.Tensor:
        """Score every row in one varlen call, with no padding at all.

        Only the flash varlen kernel reads ``cu_seq_lens``; every other kernel
        would rebuild the row boundaries as a dense ``(1, 1, T, T)`` mask and
        pay ``(sum L)^2`` instead of ``sum L^2``.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.
            keep: ``(rows, suffix)`` absolute positions of the response window.

        Returns:
            ``(rows, suffix, vocab)`` logits over the response window.
        """
        infos = batch.additional_infos
        cu_seqlens = infos[CU_SEQLENS]
        row_starts = torch.repeat_interleave(
            cu_seqlens[:-1], torch.diff(cu_seqlens), output_size=embeds.shape[0]
        )
        position_ids = (
            torch.arange(embeds.shape[0], device=embeds.device) - row_starts
        ).unsqueeze(0)
        # varlen flash-attention wants int32 boundaries
        flash_cu_seqlens = cu_seqlens.to(torch.int32)
        max_seqlen = int(infos[MAX_SEQLEN])
        if self._use_mcore:
            hidden = self._mcore_hidden(
                embeds, position_ids, flash_cu_seqlens, max_seqlen
            )
            return self.lm.output_layer(hidden[keep])
        outputs = self.lm(
            inputs_embeds=embeds.unsqueeze(0),
            attention_mask=None,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=keep.reshape(-1),
            cu_seq_lens_q=flash_cu_seqlens,
            cu_seq_lens_k=flash_cu_seqlens,
            max_length_q=max_seqlen,
            max_length_k=max_seqlen,
        )
        return outputs.logits.reshape(keep.shape[0], keep.shape[1], -1)

    def _padded_logits(
        self, embeds: torch.Tensor, batch: Batch, suffix: int
    ) -> torch.Tensor:
        """Score left-padded rows, which every kernel can serve.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the row boundaries and the collator's width.
            suffix: width of the supervised window.

        Returns:
            ``(rows, suffix, vocab)`` logits over the response window.
        """
        infos = batch.additional_infos
        padded, mask = _left_pad(embeds, infos[CU_SEQLENS], int(infos[MAX_SEQLEN]))
        if padded.shape[1] < suffix:
            raise ValueError(
                f"{type(self).__name__}: no sample reaches the supervised "
                f"window -- the padded width is {padded.shape[1]}, the window "
                f"is {suffix}, and logits_to_keep would silently return the "
                f"narrower one."
            )
        # left padding offsets every row, so spell the positions out rather
        # than letting HF assign arange(max_seqlen) over the padding too
        position_ids = (mask.cumsum(-1) - 1).clamp(min=0)
        outputs = self.lm(
            inputs_embeds=padded,
            attention_mask=mask,
            position_ids=position_ids,
            use_cache=False,
            logits_to_keep=suffix,
        )
        return outputs.logits

    def _generate(self, embeds: torch.Tensor, batch: Batch) -> torch.Tensor:
        """Beam-search the SID answer.

        Args:
            embeds: the assembled prompt embeddings, packed.
            batch: carries the packed prompt and the padded width.

        Returns:
            ``(B, self._capped_widths[-1], num_levels)`` local codes, best first.
        """
        if self._use_mcore:
            raise ValueError("Export MEGATRON_CORE weights to HF before generation.")
        infos = batch.additional_infos
        return generate_codes(
            self.lm,
            embeds,
            infos[CU_SEQLENS],
            int(infos[MAX_SEQLEN]),
            self._capped_widths,
            self._bands,
        )


def resolve_beam_config(
    common: GenRecModelConfig, codebook: Sequence[int]
) -> List[int]:
    """Validate the decode knobs against the codebook and cap the beam widths.

    Args:
        common: the shared model config.
        codebook: vocabulary size per SID level.

    Returns:
        The capped global width per SID level.
    """
    beam_widths = list(common.beam_widths)
    if not beam_widths:
        raise ValueError(
            "beam_widths is required; give one width per SID level, e.g. "
            "[50, 50, 50] or [100, 200, 400]."
        )
    if len(beam_widths) != len(codebook):
        raise ValueError(
            f"beam_widths has {len(beam_widths)} entries but the codebook has "
            f"{len(codebook)} levels; give one width per level."
        )
    if any(width < 1 for width in beam_widths):
        raise ValueError(f"beam_widths must be >= 1, got {beam_widths}.")
    return capped_beam_widths(beam_widths, codebook)


def _left_pad(
    embeds: torch.Tensor, cu_seqlens: torch.Tensor, max_seqlen: int
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Pad packed rows into a rectangle, newest token last.

    Args:
        embeds: packed embeddings, ``(total_tokens, hidden)``.
        cu_seqlens: row boundaries, ``(rows + 1,)``.
        max_seqlen: the widest row.

    Returns:
        Padded embeddings and attention mask.
    """
    starts = cu_seqlens[:-1]
    lengths = cu_seqlens[1:] - starts
    batch_size = lengths.numel()
    hidden = embeds.shape[-1]

    columns = torch.arange(max_seqlen, device=embeds.device)
    mask = columns[None, :] >= (max_seqlen - lengths)[:, None]

    padded = embeds.new_zeros((batch_size, max_seqlen, hidden))
    # mask selects row-major, which is how embeds and input_ids are packed
    padded[mask] = embeds
    return padded, mask.long()


def generate_codes(
    lm: torch.nn.Module,
    embeds: torch.Tensor,
    cu_seqlens: torch.Tensor,
    max_seqlen: int,
    capped_widths: Sequence[int],
    bands: Sequence[Tuple[int, int]],
) -> torch.Tensor:
    """Beam-search the SID answer of packed prompts.

    Args:
        lm: an HF causal LM implementing the Cache protocol.
        embeds: the assembled prompt embeddings, packed.
        cu_seqlens: row boundaries, ``(rows + 1,)``.
        max_seqlen: the widest row.
        capped_widths: beam width per SID level, already capped.
        bands: inclusive ``(lo, hi)`` token-id edge per SID level.

    Returns:
        ``(rows, capped_widths[-1], num_levels)`` local codes, best first.
    """
    padded, mask = _left_pad(embeds, cu_seqlens, max_seqlen)
    tokens = dynamic_beam_search(
        lm,
        padded,
        mask,
        capped_widths,
        bands,
        initial_cache=create_beam_cache(lm),
    )
    # a band starts at base_vocab_size + level_offset, so this undoes both shifts
    codes = tokens - tokens.new_tensor([lo for lo, _ in bands])
    codes = codes.view(padded.shape[0], -1, len(bands))
    return codes


@torch.fx.wrap
def _fx_wrapped_forward(
    model: "GenRecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> Dict[str, torch.Tensor]:
    """Hide the packed forward and its per-batch routing from FX.

    ``TrainPipelineSparseDist`` symbolically traces the model whenever a
    sharded module exists, and the LM call reads ``max_seqlen`` as a host int.
    A SID-only response whose batch has a row with several SIDs takes the
    multi-target forward; every other batch keeps the response window.

    Args:
        model: the model whose response window to compute.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch being scored.

    Returns:
        Logits and labels, plus per-target row weights for a SID-only
        response.

    Raises:
        ValueError: no valid SID targets remain after masking incomplete responses.
    """
    if not model._sid_only_response:
        logits, labels = model._forward(embeds, batch)
        return {"logits": logits, "labels": labels}
    num_levels = model._prompt.sid_space.num_levels
    lengths = batch.additional_infos[RESPONSE_LENGTHS]
    lengths = lengths.masked_fill(lengths % num_levels != 0, 0)
    batch.additional_infos[RESPONSE_LENGTHS] = lengths
    max_length = int(lengths.max())
    if max_length == 0:
        raise ValueError("No valid SID targets remain in this batch.")
    if max_length > num_levels:
        return model._multi_target_predictions(embeds, batch)
    logits, labels = model._forward(embeds, batch)
    return {
        "logits": logits,
        "labels": labels,
        "target_weights": (lengths > 0).float(),
    }


@torch.fx.wrap
def _fx_wrapped_generate(
    model: "GenRecCausalLMModel", embeds: torch.Tensor, batch: Batch
) -> torch.Tensor:
    """Hide the decode loop from FX.

    ``PredictPipelineSparseDist`` FX-traces the model, and beam decode reads
    host ints and branches on them. Wrapping an inner helper only moves the
    failure to the next such read, so the whole loop is one leaf.

    Args:
        model: the model whose decode loop to run.
        embeds: the assembled prompt embeddings, packed.
        batch: the batch to decode.

    Returns:
        The decoded local codes.
    """
    return model._generate(embeds, batch)
