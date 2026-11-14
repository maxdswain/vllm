# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Check fused DeBERTa relative attention against its PyTorch fallback."""

import pytest
import torch

from vllm.model_executor.layers.attention.flash_deberta_attention import (
    HAS_TRITON,
    flash_deberta_attention,
)


@pytest.mark.skipif(
    not torch.cuda.is_available() or not HAS_TRITON, reason="Triton CUDA is required"
)
@pytest.mark.parametrize("dtype, atol", [(torch.float32, 3e-3), (torch.bfloat16, 3e-2)])
def test_fused_deberta_relative_attention_matches_pytorch(dtype, atol):
    if dtype is torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires Ampere or newer")
    torch.manual_seed(42)
    batch, heads, length, width, span = 2, 4, 16, 64, 8
    q, k, v = (
        torch.randn(batch, heads, length, width, device="cuda", dtype=dtype)
        for _ in range(3)
    )
    pos_key, pos_query = (
        torch.randn(batch, heads, length, 2 * span, device="cuda", dtype=dtype)
        for _ in range(2)
    )
    lengths = torch.tensor([length - 1, length - 5], device="cuda")
    kwargs = dict(
        seq_lengths=lengths,
        pos_key=pos_key,
        pos_query=pos_query,
        position_buckets=span,
        max_relative_distance=32,
        use_log_bucket=True,
    )
    actual = flash_deberta_attention(q, k, v, **kwargs)
    expected = flash_deberta_attention(
        q.cpu().float(),
        k.cpu().float(),
        v.cpu().float(),
        seq_lengths=lengths.cpu(),
        pos_key=pos_key.cpu().float(),
        pos_query=pos_query.cpu().float(),
        position_buckets=span,
        max_relative_distance=32,
        use_log_bucket=True,
    ).to(device="cuda", dtype=dtype)
    valid = torch.arange(length, device="cuda")[None, :] < lengths[:, None]
    torch.testing.assert_close(actual[valid], expected[valid], atol=atol, rtol=atol)
