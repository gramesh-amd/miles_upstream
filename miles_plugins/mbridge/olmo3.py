"""
OLMo 3 (7B Instruct-DPO / Think) — Megatron-Core layer spec + mbridge.

Architecture specifics handled here:
  - Post-norm only (no pre-attention / pre-MLP layernorm)
  - Full-dimension QK layernorm (over hidden_size, not head_dim)
  - Per-layer RoPE: default for sliding-window layers, YaRN for full-attention
  - MHA with 32 heads, kv_channels=128
  - Sliding window attention: 3 SWA + 1 full, repeating
"""

from dataclasses import dataclass
from typing import Optional, Union

import torch
from megatron.core import __version__
from megatron.core.fusions.fused_bias_dropout import get_bias_dropout_add
from megatron.core.models.common.embeddings.yarn_rotary_pos_embedding import (
    _yarn_find_correction_range,
    _yarn_linear_ramp_mask,
)
from megatron.core.transformer.attention import SelfAttention, SelfAttentionSubmodules
from megatron.core.transformer.enums import AttnMaskType
from megatron.core.transformer.identity_op import IdentityOp
from megatron.core.transformer.mlp import MLP, MLPSubmodules
from megatron.core.transformer.spec_utils import ModuleSpec, build_module
from megatron.core.transformer.transformer_layer import (
    TransformerLayer,
    TransformerLayerSubmodules,
)
from megatron.core.utils import make_viewless_tensor
from packaging import version

try:
    from megatron.core.extensions.transformer_engine import (
        TEColumnParallelLinear,
        TEDotProductAttention,
        TENorm,
        TERowParallelLinear,
    )

    HAVE_TE = True
except ImportError:
    HAVE_TE = False

from mbridge.core import LLMBridge, register_model

_SENTINEL = object()

# YaRN config attributes temporarily set on TransformerConfig for full-attention
# layers so _yarn_get_concentration_factor_from_config returns the correct
# mscale (1.2079) during RoPE application.
_OLMO3_YARN_CONFIG_ATTRS = {
    "yarn_rotary_scaling_factor": 8.0,
    "yarn_mscale": 1.0,
    "yarn_mscale_all_dim": 0.0,
}

# OLMo 3 YaRN parameters (from HF config rope_scaling)
_OLMO3_ROTARY_BASE = 500000.0
_OLMO3_YARN_FACTOR = 8.0
_OLMO3_YARN_ORIG_MAX_POS = 8192
_OLMO3_YARN_BETA_FAST = 32.0
_OLMO3_YARN_BETA_SLOW = 1.0


# ---------------------------------------------------------------------------
# Custom SelfAttention: full-dim QK norm (OLMo 3 style)
# ---------------------------------------------------------------------------


class OLMo3SelfAttention(SelfAttention):
    """
    OLMo 3 applies QK layernorm over the full hidden dimension (e.g. 4096)
    BEFORE reshaping into attention heads. Megatron's default applies it
    per-head (e.g. 128 dims). This subclass replaces per-head norms with
    full-dim norms and overrides get_query_key_value_tensors accordingly.
    """

    def __init__(self, config, submodules, layer_number, **kwargs):
        super().__init__(config, submodules, layer_number, **kwargs)

        # Replace with full-dim norms over the TP-local Q/K dimensions.
        # TODO: For TP > 1, mbridge needs to split q/k layernorm weights
        # across TP ranks (column-parallel) instead of replicating them.
        if self.q_layernorm is not None:
            q_hidden = self.num_attention_heads_per_partition * self.hidden_size_per_attention_head
            self.q_layernorm = build_module(
                submodules.q_layernorm,
                config=config,
                hidden_size=q_hidden,
                eps=config.layernorm_epsilon,
            )
        if self.k_layernorm is not None:
            kv_hidden = self.num_query_groups_per_partition * self.hidden_size_per_attention_head
            self.k_layernorm = build_module(
                submodules.k_layernorm,
                config=config,
                hidden_size=kv_hidden,
                eps=config.layernorm_epsilon,
            )

    def get_query_key_value_tensors(self, hidden_states, key_value_states=None, split_qkv=True):
        mixed_qkv, _ = self.linear_qkv(hidden_states)

        if not split_qkv:
            return super().get_query_key_value_tensors(hidden_states, key_value_states, split_qkv=False)

        # Megatron stores QKV in INTERLEAVED format per query-group:
        #   [Q0, K0, V0, Q1, K1, V1, ..., Q_{ng-1}, K_{ng-1}, V_{ng-1}]
        heads_per_group = self.num_attention_heads_per_partition // self.num_query_groups_per_partition
        group_size = (heads_per_group + 2) * self.hidden_size_per_attention_head

        new_shape = mixed_qkv.size()[:-1] + (self.num_query_groups_per_partition, group_size)
        mixed_qkv = mixed_qkv.view(*new_shape)

        split_sizes = [
            heads_per_group * self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
            self.hidden_size_per_attention_head,
        ]
        query, key, value = torch.split(mixed_qkv, split_sizes, dim=-1)

        # Flatten Q and K across all heads for full-dim QK norm (OLMo 3 architecture).
        # HF applies: q_norm(q_proj(x)) over the full projected dim before head reshape.
        if self.q_layernorm is not None:
            q_flat = query.reshape(*query.shape[:-2], -1)
            q_flat = self.q_layernorm(q_flat)
            query = q_flat.reshape(*query.shape[:-2],
                                   self.num_attention_heads_per_partition,
                                   self.hidden_size_per_attention_head)
        else:
            query = query.reshape(*query.shape[:-2],
                                  self.num_attention_heads_per_partition,
                                  self.hidden_size_per_attention_head)

        if self.k_layernorm is not None:
            k_flat = key.reshape(*key.shape[:-2], -1)
            k_flat = self.k_layernorm(k_flat)
            key = k_flat.reshape(*key.shape[:-2],
                                 self.num_query_groups_per_partition,
                                 self.hidden_size_per_attention_head)

        value = value.reshape(*value.shape[:-2],
                              self.num_query_groups_per_partition,
                              self.hidden_size_per_attention_head)

        return query, key, value


# ---------------------------------------------------------------------------
# Custom TransformerLayer with post-norm + per-layer RoPE (OLMo 3)
# ---------------------------------------------------------------------------


@dataclass
class OLMo3TransformerLayerSubmodules(TransformerLayerSubmodules):
    post_attention_layernorm: Union[ModuleSpec, type] = IdentityOp
    post_feedforward_layernorm: Union[ModuleSpec, type] = IdentityOp


class OLMo3TransformerLayer(TransformerLayer):
    """
    OLMo 3 post-norm transformer layer.

    Computation order per block:
      attn_out = self_attention(hidden)          # no pre-norm
      hidden = residual + post_attn_norm(attn_out)
      mlp_out = mlp(hidden)                      # no pre-norm
      hidden = residual + post_ffn_norm(mlp_out)
    """

    def __init__(
        self,
        config,
        submodules: OLMo3TransformerLayerSubmodules,
        layer_number: int = 1,
        hidden_dropout: float = None,
        **kwargs,
    ):
        super().__init__(
            config=config,
            submodules=submodules,
            layer_number=layer_number,
            hidden_dropout=hidden_dropout,
            **kwargs,
        )

        self.post_attention_layernorm = build_module(
            submodules.post_attention_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )
        self.post_feedforward_layernorm = build_module(
            submodules.post_feedforward_layernorm,
            config=self.config,
            hidden_size=self.config.hidden_size,
            eps=self.config.layernorm_epsilon,
        )

        # Per-layer RoPE: OLMo 3 uses default RoPE for sliding-attention layers
        # and YaRN RoPE for full-attention layers (pattern: 3 sliding + 1 full).
        # Full attention at 0-indexed layers 3,7,11,...,31 -> 1-indexed 4,8,...,32.
        layer_idx = self.layer_number - 1
        self.is_full_attention = (layer_idx % 4 == 3)

        if self.is_full_attention:
            dim = self.config.kv_channels
            inv_freq_extra = 1.0 / (
                _OLMO3_ROTARY_BASE
                ** (torch.arange(0, dim, 2, dtype=torch.float32) / dim)
            )
            inv_freq_inter = inv_freq_extra / _OLMO3_YARN_FACTOR

            low, high = _yarn_find_correction_range(
                _OLMO3_YARN_BETA_FAST,
                _OLMO3_YARN_BETA_SLOW,
                dim,
                _OLMO3_ROTARY_BASE,
                _OLMO3_YARN_ORIG_MAX_POS,
            )
            inv_freq_mask = 1.0 - _yarn_linear_ramp_mask(low, high, dim // 2)
            yarn_inv_freq = (
                inv_freq_inter * (1 - inv_freq_mask)
                + inv_freq_extra * inv_freq_mask
            )
            self.register_buffer("_yarn_inv_freq", yarn_inv_freq, persistent=False)

    def _compute_yarn_rotary_pos_emb(self, rotary_pos_emb):
        """Replace default RoPE freqs with YaRN-modified freqs."""
        is_tuple = isinstance(rotary_pos_emb, tuple)
        ref = rotary_pos_emb[0] if is_tuple else rotary_pos_emb
        seq_len = ref.shape[0]

        positions = torch.arange(
            seq_len, device=self._yarn_inv_freq.device, dtype=self._yarn_inv_freq.dtype
        )
        freqs = torch.outer(positions, self._yarn_inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)[:, None, None, :]

        return (emb, emb) if is_tuple else emb

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        context=None,
        context_mask=None,
        rotary_pos_emb=None,
        rotary_pos_cos=None,
        rotary_pos_sin=None,
        rotary_pos_cos_sin=None,
        attention_bias=None,
        inference_context=None,
        inference_params=None,
        packed_seq_params=None,
        sequence_len_offset=None,
        **kwargs,
    ):
        # Per-layer RoPE: full-attention layers use YaRN, sliding layers keep
        # the default RoPE passed from the model.  We also temporarily set
        # YaRN config attributes so the attention module applies the correct
        # mscale (1.2079) during apply_rotary_pos_emb.
        saved_config = {}
        if self.is_full_attention and rotary_pos_emb is not None:
            rotary_pos_emb = self._compute_yarn_rotary_pos_emb(rotary_pos_emb)
            for attr, val in _OLMO3_YARN_CONFIG_ATTRS.items():
                saved_config[attr] = getattr(self.config, attr, _SENTINEL)
                setattr(self.config, attr, val)

        try:
            residual = hidden_states

            extra_kwargs = {}
            if version.parse(__version__) >= version.parse("0.12.0"):
                extra_kwargs["inference_context"] = inference_context
            else:
                extra_kwargs["inference_params"] = inference_params

            input_layernorm_output = self.input_layernorm(hidden_states)

            hidden_states, hidden_states_bias = self.self_attention(
                input_layernorm_output,
                attention_mask=attention_mask,
                rotary_pos_emb=rotary_pos_emb,
                rotary_pos_cos=rotary_pos_cos,
                rotary_pos_sin=rotary_pos_sin,
                rotary_pos_cos_sin=rotary_pos_cos_sin,
                attention_bias=attention_bias,
                packed_seq_params=packed_seq_params,
                sequence_len_offset=sequence_len_offset,
                **extra_kwargs,
            )

            if hidden_states_bias is not None:
                hidden_states = hidden_states + hidden_states_bias

            hidden_states = self.post_attention_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            residual = hidden_states

            pre_mlp_layernorm_output = self.pre_mlp_layernorm(hidden_states)

            hidden_states, hidden_states_bias = self.mlp(pre_mlp_layernorm_output)

            if hidden_states_bias is not None:
                hidden_states = hidden_states + hidden_states_bias

            hidden_states = self.post_feedforward_layernorm(hidden_states)
            hidden_states = residual + hidden_states

            output = make_viewless_tensor(
                inp=hidden_states,
                requires_grad=hidden_states.requires_grad,
                keep_graph=True,
            )

            if self.config.external_cuda_graph and self.training:
                return output
            return output, context
        finally:
            for attr, val in saved_config.items():
                if val is _SENTINEL:
                    if hasattr(self.config, attr):
                        delattr(self.config, attr)
                else:
                    setattr(self.config, attr, val)


def get_olmo3_layer_spec_te(args=None, config=None, vp_stage=None) -> ModuleSpec:
    """
    OLMo 3 layer spec using Transformer Engine.

    Uses TEColumnParallelLinear (not fused with layernorm) for post-norm arch,
    and OLMo3SelfAttention for full-dim QK norm.
    """
    return ModuleSpec(
        module=OLMo3TransformerLayer,
        submodules=OLMo3TransformerLayerSubmodules(
            self_attention=ModuleSpec(
                module=OLMo3SelfAttention,
                params={"attn_mask_type": AttnMaskType.causal},
                submodules=SelfAttentionSubmodules(
                    linear_qkv=TEColumnParallelLinear,
                    core_attention=TEDotProductAttention,
                    linear_proj=TERowParallelLinear,
                    q_layernorm=TENorm,
                    k_layernorm=TENorm,
                ),
            ),
            self_attn_bda=get_bias_dropout_add,
            pre_mlp_layernorm=IdentityOp,
            mlp=ModuleSpec(
                module=MLP,
                submodules=MLPSubmodules(
                    linear_fc1=TEColumnParallelLinear,
                    linear_fc2=TERowParallelLinear,
                ),
            ),
            mlp_bda=get_bias_dropout_add,
            post_attention_layernorm=TENorm,
            post_feedforward_layernorm=TENorm,
        ),
    )


# ---------------------------------------------------------------------------
# OLMo3 Bridge
# ---------------------------------------------------------------------------


@register_model("olmo3")
class OLMo3Bridge(LLMBridge):
    """
    Bridge for OLMo 3 models (7B Instruct-DPO, Think).

    Key architectural differences from Llama:
      - Post-norm only (no input_layernorm, no pre-MLP layernorm)
      - Has post_attention_layernorm and post_feedforward_layernorm
      - Full-dim QK layernorm (q_norm / k_norm over hidden_size, not head_dim)
      - MHA with 32 heads (not GQA)
    """

    _DIRECT_MAPPING = {
        "embedding.word_embeddings.weight": "model.embed_tokens.weight",
        "decoder.final_layernorm.weight": "model.norm.weight",
        "output_layer.weight": "lm_head.weight",
    }

    _ATTENTION_MAPPING = {
        "self_attention.linear_qkv.weight": [
            "model.layers.{layer_number}.self_attn.q_proj.weight",
            "model.layers.{layer_number}.self_attn.k_proj.weight",
            "model.layers.{layer_number}.self_attn.v_proj.weight",
        ],
        "self_attention.linear_proj.weight": [
            "model.layers.{layer_number}.self_attn.o_proj.weight",
        ],
        "self_attention.q_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.q_norm.weight",
        ],
        "self_attention.k_layernorm.weight": [
            "model.layers.{layer_number}.self_attn.k_norm.weight",
        ],
    }

    _MLP_MAPPING = {
        "mlp.linear_fc1.weight": [
            "model.layers.{layer_number}.mlp.gate_proj.weight",
            "model.layers.{layer_number}.mlp.up_proj.weight",
        ],
        "mlp.linear_fc2.weight": [
            "model.layers.{layer_number}.mlp.down_proj.weight",
        ],
    }

    _OTHER_MAPPING = {
        "post_attention_layernorm.weight": [
            "model.layers.{layer_number}.post_attention_layernorm.weight",
        ],
        "post_feedforward_layernorm.weight": [
            "model.layers.{layer_number}.post_feedforward_layernorm.weight",
        ],
    }

    def _build_config(self):
        return self._build_base_config(
            add_qkv_bias=False,
            qk_layernorm=True,
            normalization="RMSNorm",
        )

    def _get_gptmodel_args(self) -> dict:
        return dict(
            vocab_size=self.hf_config.vocab_size,
            max_sequence_length=self.hf_config.max_position_embeddings,
            position_embedding_type="rope",
            rotary_base=self.hf_config.rope_theta,
        )

    def _get_transformer_layer_spec(self, vp_stage=None):
        return get_olmo3_layer_spec_te()
