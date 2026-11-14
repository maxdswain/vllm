# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Bidirectional DeBERTa attention, adapted from maxdswain/vllm-gliner
and Knowledgator/FlashDeBERTa (Apache-2.0).
"""

import torch

from vllm.platforms import current_platform
from vllm.triton_utils import tl, triton


@triton.jit
def _flash_deberta_fwd_kernel(
    Q,
    K,
    V,
    POS_KEY,
    POS_QUERY,
    OUT,
    LENGTHS,
    q_stride: tl.constexpr,
    k_stride: tl.constexpr,
    v_stride: tl.constexpr,
    pk_stride: tl.constexpr,
    pq_stride: tl.constexpr,
    out_stride: tl.constexpr,
    scale: tl.constexpr,
    D: tl.constexpr,
    SPAN: tl.constexpr,
    BUCKETS: tl.constexpr,
    MAX_DISTANCE: tl.constexpr,
    HAS_C2P: tl.constexpr,
    HAS_P2C: tl.constexpr,
    BLOCK_D: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    DOT_PRECISION: tl.constexpr,
):
    batch = tl.program_id(2).to(tl.int64)
    head = tl.program_id(1).to(tl.int64)
    rows = tl.program_id(0) * BLOCK_M + tl.arange(0, BLOCK_M)
    cols = tl.arange(0, BLOCK_N)
    dims = tl.arange(0, BLOCK_D)
    length = tl.load(LENGTHS + batch)
    Q += batch * q_stride[0] + head * q_stride[1]
    K += batch * k_stride[0] + head * k_stride[1]
    V += batch * v_stride[0] + head * v_stride[1]
    OUT += batch * out_stride[0] + head * out_stride[1]
    if HAS_C2P:
        POS_KEY += batch * pk_stride[0] + head * pk_stride[1]
    if HAS_P2C:
        POS_QUERY += batch * pq_stride[0] + head * pq_stride[1]

    q = tl.load(
        Q + rows[:, None] * q_stride[2] + dims[None, :] * q_stride[3],
        mask=(rows[:, None] < length) & (dims[None, :] < D),
        other=0,
    )
    maximum = tl.full((BLOCK_M,), -float("inf"), tl.float32)
    denominator = tl.zeros((BLOCK_M,), tl.float32)
    accumulator = tl.zeros((BLOCK_M, BLOCK_D), tl.float32)
    for start in range(0, tl.cdiv(length, BLOCK_N) * BLOCK_N, BLOCK_N):
        keys = start + cols
        k = tl.load(
            K + dims[:, None] * k_stride[3] + keys[None, :] * k_stride[2],
            mask=(keys[None, :] < length) & (dims[:, None] < D),
            other=0,
        )
        v = tl.load(
            V + keys[:, None] * v_stride[2] + dims[None, :] * v_stride[3],
            mask=(keys[:, None] < length) & (dims[None, :] < D),
            other=0,
        )
        scores = tl.dot(q, k, input_precision=DOT_PRECISION) * scale
        relative = rows[:, None] - keys[None, :]
        if BUCKETS > 0:
            mid: tl.constexpr = BUCKETS // 2
            absolute = tl.where(tl.abs(relative) < mid, mid - 1, tl.abs(relative))
            log_pos = (
                tl.ceil(
                    tl.log(absolute / mid)
                    / tl.log((MAX_DISTANCE - 1) / mid)
                    * (mid - 1)
                )
                + mid
            )
            relative = tl.where(
                absolute <= mid, relative, log_pos * tl.where(relative > 0, 1, -1)
            )
        index = tl.minimum(tl.maximum(relative + SPAN, 0), 2 * SPAN - 1).to(tl.int32)
        if HAS_C2P:
            bias = tl.load(
                POS_KEY + rows[:, None] * pk_stride[2] + index * pk_stride[3],
                mask=rows[:, None] < length,
                other=0,
            )
            scores += bias * scale
        if HAS_P2C:
            # HF gathers with the opposite distance, then transposes the result.
            bias = tl.load(
                POS_QUERY + keys[None, :] * pq_stride[2] + index * pq_stride[3],
                mask=keys[None, :] < length,
                other=0,
            )
            scores += bias * scale
        scores = tl.where(keys[None, :] < length, scores, -float("inf"))
        next_maximum = tl.maximum(maximum, tl.max(scores, 1))
        correction = tl.exp2((maximum - next_maximum) * 1.4426950408889634)
        probabilities = tl.exp2((scores - next_maximum[:, None]) * 1.4426950408889634)
        accumulator *= correction[:, None]
        accumulator += tl.dot(
            probabilities.to(q.dtype), v, input_precision=DOT_PRECISION
        )
        denominator = denominator * correction + tl.sum(probabilities, 1)
        maximum = next_maximum

    result = accumulator / denominator[:, None]
    tl.store(
        OUT + rows[:, None] * out_stride[2] + dims[None, :] * out_stride[3],
        result,
        mask=(rows[:, None] < length) & (dims[None, :] < D),
    )


def flash_deberta_attention(
    q: torch.Tensor,
    k: torch.Tensor,
    v: torch.Tensor,
    seq_lengths: torch.Tensor,
    pos_key: torch.Tensor | None,
    pos_query: torch.Tensor | None,
    sm_scale: float,
    position_buckets: int,
    max_relative_distance: int,
) -> torch.Tensor:
    """Attend within right-padded sequences, with optional c2p/p2c scores.

    Q/K/V have shape `(batch, heads, length, head_dim)`; position scores have
    shape `(batch, heads, length, 2 * span)`. CPU models use Transformers instead.
    """
    batch, heads, length, width = q.shape
    assert k.shape == v.shape == q.shape
    span = position_buckets if position_buckets > 0 else max_relative_distance
    output = torch.zeros_like(q)
    precision = (
        "tf32x3"
        if not current_platform.is_rocm()
        and torch.cuda.get_device_capability(q.device.index)[0] >= 8
        else "ieee"
    )
    with torch.accelerator.device_index(q.device.index):
        _flash_deberta_fwd_kernel[(triton.cdiv(length, 32), heads, batch)](
            q,
            k,
            v,
            pos_key,
            pos_query,
            output,
            seq_lengths,
            q.stride(),
            k.stride(),
            v.stride(),
            pos_key.stride() if pos_key is not None else (0, 0, 0, 0),
            pos_query.stride() if pos_query is not None else (0, 0, 0, 0),
            output.stride(),
            sm_scale,
            width,
            span,
            position_buckets,
            max_relative_distance,
            HAS_C2P=pos_key is not None,
            HAS_P2C=pos_query is not None,
            BLOCK_D=max(16, triton.next_power_of_2(width)),
            BLOCK_M=32,
            BLOCK_N=32,
            DOT_PRECISION=precision,
            num_warps=4,
            num_stages=1,
        )
    return output
