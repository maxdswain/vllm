# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fused DeBERTa attention against independent PyTorch tensor operations."""

import pytest
import torch
from transformers.models.deberta_v2.modeling_deberta_v2 import build_relative_position

from vllm.model_executor.layers.attention.flash_deberta_attention import (
    flash_deberta_attention,
)
from vllm.triton_utils import HAS_TRITON


def reference_attention(q, k, v, lengths, pos_key, pos_query, span, distance, scale):
    relative = build_relative_position(q, k, bucket_size=span, max_position=distance)
    scores = q @ k.transpose(-1, -2)
    index = (relative + span).clamp(0, 2 * span - 1).expand_as(scores)
    scores += pos_key.gather(-1, index)
    # HF gathers in key/query order, then transposes back to query/key order.
    index = (-relative + span).clamp(0, 2 * span - 1).expand_as(scores)
    scores += pos_query.gather(-1, index).transpose(-1, -2)
    valid = torch.arange(q.shape[2], device=q.device)[None, :] < lengths[:, None]
    scores = (scores * scale).masked_fill(~valid[:, None, None, :], -torch.inf)
    return scores.softmax(-1) @ v


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON, reason="Triton CUDA is required"
)
@pytest.mark.parametrize("dtype, atol", [(torch.float32, 3e-3), (torch.bfloat16, 3e-2)])
@pytest.mark.parametrize("length, span", [(1, 256), (19, 8), (257, 256)])
def test_fused_deberta_relative_attention_matches_pytorch(dtype, atol, length, span):
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires Ampere or newer")
    torch.manual_seed(42)
    batch, heads, width, distance = 2, 4, 64, 512
    q, k, v = (
        torch.randn(batch, heads, length, width, device="cuda", dtype=dtype)
        for _ in range(3)
    )
    pos_key, pos_query = (
        torch.randn(batch, heads, length, 2 * span, device="cuda", dtype=dtype)
        for _ in range(2)
    )
    lengths = torch.tensor([length, max(1, length - 5)], device="cuda")
    scale = (3 * width) ** -0.5
    actual = flash_deberta_attention(
        q, k, v, lengths, pos_key, pos_query, scale, span, distance
    )
    expected = reference_attention(
        q.float(),
        k.float(),
        v.float(),
        lengths,
        pos_key.float(),
        pos_query.float(),
        span,
        distance,
        scale,
    ).to(dtype)
    valid = torch.arange(length, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(
        actual.transpose(1, 2)[valid],
        expected.transpose(1, 2)[valid],
        atol=atol,
        rtol=atol,
    )
