# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Fused DeBERTa-v2 attention for the Transformers modeling backend."""

import torch
from torch import Tensor
from transformers.models.deberta_v2.modeling_deberta_v2 import (
    DisentangledSelfAttention,
)

from vllm.model_executor.layers.attention.flash_deberta_attention import (
    flash_deberta_attention,
)


class FusedDebertaV2Attention(DisentangledSelfAttention):
    """Preserve Transformers' projections while fusing relative attention."""

    def forward(
        self,
        hidden_states: Tensor,
        attention_mask: Tensor,
        output_attentions: bool = False,
        query_states: Tensor | None = None,
        relative_pos: Tensor | None = None,
        rel_embeddings: Tensor | None = None,
    ) -> tuple[Tensor, Tensor | None]:
        if (
            not hidden_states.is_cuda
            or output_attentions
            or query_states is not None
            or not self.relative_attention
            or rel_embeddings is None
        ):
            return super().forward(
                hidden_states,
                attention_mask,
                output_attentions=output_attentions,
                query_states=query_states,
                relative_pos=relative_pos,
                rel_embeddings=rel_embeddings,
            )

        batch, length, _ = hidden_states.shape

        def heads(x: Tensor) -> Tensor:
            return x.view(batch, length, self.num_attention_heads, -1).transpose(1, 2)

        query = heads(self.query_proj(hidden_states))
        key = heads(self.key_proj(hidden_states))
        value = heads(self.value_proj(hidden_states))
        span = self.pos_ebd_size
        relative = self.pos_dropout(rel_embeddings[: span * 2])[None, ...]
        factor = 1 + int("c2p" in self.pos_att_type) + int("p2c" in self.pos_att_type)
        scale = (self.attention_head_size * factor) ** -0.5

        pos_key = None
        if "c2p" in self.pos_att_type:
            projection = self.key_proj if self.share_att_key else self.pos_key_proj
            projected = projection(relative)
            position_key = projected.view(
                1, 2 * span, self.num_attention_heads, -1
            ).transpose(1, 2)
            pos_key = query @ position_key.transpose(-1, -2)

        pos_query = None
        if "p2c" in self.pos_att_type:
            projection = self.query_proj if self.share_att_key else self.pos_query_proj
            position_query = (
                projection(relative)
                .view(1, 2 * span, self.num_attention_heads, -1)
                .transpose(1, 2)
            )
            pos_query = key @ position_query.transpose(-1, -2)

        lengths = (
            attention_mask[:, 0, 0, :].sum(-1)
            if attention_mask.ndim == 4
            else attention_mask.sum(-1)
        ).to(torch.int32)
        context = flash_deberta_attention(
            query,
            key,
            value,
            seq_lengths=lengths,
            pos_key=pos_key,
            pos_query=pos_query,
            sm_scale=scale,
            position_buckets=self.position_buckets,
            max_relative_distance=self.max_relative_positions,
        )
        return context.transpose(1, 2).reshape(batch, length, -1), None
