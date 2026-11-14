# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the GLiNER2.5 Decide classification path."""

import pytest
import requests
import torch
from transformers import DebertaV2Config, DebertaV2Model

from tests.utils import RemoteOpenAIServer
from vllm.config import ModelConfig, VllmConfig, set_current_vllm_config
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.models.gliner2 import Gliner2ForSchemaExtraction
from vllm.model_executor.models.transformers.deberta import FusedDebertaV2Attention
from vllm.pooling_params import PoolingParams
from vllm.transformers_utils.configs.gliner2 import GLiNER2Config
from vllm.v1.pool.metadata import PoolingMetadata, PoolingStates

pytestmark = pytest.mark.core_model


def test_decide_cpu_batch_keeps_bidirectional_sequences_separate(tmp_path, dist_init):
    encoder_config = DebertaV2Config(
        vocab_size=32,
        hidden_size=16,
        num_attention_heads=4,
        num_hidden_layers=1,
        intermediate_size=32,
        max_position_embeddings=32,
        relative_attention=False,
        position_biased_input=False,
        type_vocab_size=0,
    )
    config = GLiNER2Config(
        encoder_config=encoder_config.to_dict(),
        architectures=["Gliner2ForSchemaExtraction"],
        architecture="span",
        counting_layer="count_lstm",
    )
    config.save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), dtype="float32", runner="pooling")
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        model = Gliner2ForSchemaExtraction(vllm_config=vllm_config)
    for param in model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    process_weights_after_loading(model, model_config, torch.device("cpu"))
    model.eval()
    assert model.pooler.get_supported_tasks() == {"classify"}

    first = torch.tensor([1, 2, 3, 4, 5])
    second = torch.tensor([6, 7, 8, 9])
    # The trailing zero-position slot is runner padding, not another sequence.
    ids = torch.cat([first, second, torch.zeros(1, dtype=torch.long)])
    positions = torch.tensor([0, 1, 2, 3, 4, 0, 1, 2, 3, 0])
    with torch.no_grad():
        hidden = model(input_ids=ids, positions=positions)
        expected = torch.cat(
            [
                model.model(input_ids=row[None], return_dict=False)[0][0]
                for row in (first, second)
            ]
        )
    torch.testing.assert_close(hidden[:9], expected)
    assert torch.count_nonzero(hidden[9]) == 0

    metadata = PoolingMetadata(
        prompt_lens=torch.tensor([5, 4]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[
            PoolingParams(task="classify", extra_kwargs={"label_positions": [1, 3]}),
            PoolingParams(task="classify", extra_kwargs={"label_positions": [0, 2]}),
        ],
        pooling_states=[PoolingStates(), PoolingStates()],
    )
    outputs = model.pooler(hidden, metadata)
    for output, row in zip(outputs, (expected[:5], expected[5:]), strict=True):
        positions = [1, 3] if row.shape[0] == 5 else [0, 2]
        torch.testing.assert_close(
            output, model.pooler.classifier(row[positions]).flatten()
        )

    warmup = PoolingMetadata(
        prompt_lens=torch.tensor([5]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[PoolingParams(task="classify")],
        pooling_states=[PoolingStates()],
    )
    assert model.pooler(hidden[:5], warmup)[0].tolist() == [0.0]

    mixed = PoolingMetadata(
        prompt_lens=torch.tensor([5, 4]),
        prompt_token_ids=None,
        prompt_token_ids_cpu=None,
        pooling_params=[warmup.pooling_params[0], metadata.pooling_params[1]],
        pooling_states=[PoolingStates(), PoolingStates()],
    )
    empty, scored = model.pooler(hidden, mixed)
    assert empty.tolist() == [0.0]
    torch.testing.assert_close(
        scored, model.pooler.classifier(expected[5:][[0, 2]]).flatten()
    )


@pytest.mark.parametrize("conv_kernel_size", [0, 3])
def test_backend_deberta_relative_attention_matches_transformers_on_cpu(
    tmp_path, dist_init, conv_kernel_size
):
    config = DebertaV2Config(
        vocab_size=32,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_buckets=8,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        position_biased_input=False,
        type_vocab_size=0,
        conv_kernel_size=conv_kernel_size,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    gliner_config = GLiNER2Config(
        encoder_config=config.to_dict(),
        architectures=["Gliner2ForSchemaExtraction"],
        architecture="span",
        counting_layer="count_lstm",
    )
    gliner_config.save_pretrained(tmp_path)
    model_config = ModelConfig(model=str(tmp_path), dtype="float32", runner="pooling")
    vllm_config = VllmConfig(model_config=model_config)
    with set_current_vllm_config(vllm_config):
        model = Gliner2ForSchemaExtraction(vllm_config=vllm_config).eval()
    for param in model.model.parameters():
        torch.nn.init.normal_(param, std=0.02)
    assert all(
        isinstance(layer.attention.self, FusedDebertaV2Attention)
        for layer in model.model.encoder.layer
    )
    reference = DebertaV2Model(config).eval()
    weights = model.model.state_dict()
    weights["embeddings.word_embeddings.weight"] = weights[
        "embeddings.word_embeddings.weight"
    ][: config.vocab_size]
    reference.load_state_dict(weights)
    process_weights_after_loading(model, model_config, torch.device("cpu"))

    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    valid = ids.ne(0)
    mask = valid.long()
    with torch.no_grad():
        actual = model.model(input_ids=ids, attention_mask=mask).last_hidden_state
        expected = reference(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-5, rtol=1e-5)


@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is required")
@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_fused_transformers_deberta_matches_reference_on_gpu(dtype):
    if dtype == torch.bfloat16 and torch.cuda.get_device_capability()[0] < 8:
        pytest.skip("bfloat16 requires Ampere or newer")
    config = DebertaV2Config(
        vocab_size=32,
        hidden_size=32,
        num_attention_heads=4,
        num_hidden_layers=2,
        intermediate_size=64,
        max_position_embeddings=32,
        relative_attention=True,
        pos_att_type=["p2c", "c2p"],
        position_buckets=8,
        norm_rel_ebd="layer_norm",
        share_att_key=True,
        position_biased_input=False,
        type_vocab_size=0,
        hidden_dropout_prob=0.0,
        attention_probs_dropout_prob=0.0,
    )
    reference = DebertaV2Model(config).to(device="cuda", dtype=dtype).eval()
    model = DebertaV2Model(config).to(device="cuda", dtype=dtype).eval()
    model.load_state_dict(reference.state_dict())
    for layer in model.encoder.layer:
        layer.attention.self.__class__ = FusedDebertaV2Attention

    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]], device="cuda")
    valid = ids.ne(0)
    with torch.inference_mode():
        actual = model(input_ids=ids, attention_mask=valid).last_hidden_state
        expected = reference(input_ids=ids, attention_mask=valid).last_hidden_state
    atol = 3e-2 if dtype == torch.bfloat16 else 3e-3
    torch.testing.assert_close(actual[valid], expected[valid], atol=atol, rtol=atol)


def test_decide_scores_a_real_checkpoint(vllm_runner):
    model = "fastino/GLiNER2.5-Decide"
    with vllm_runner(
        model,
        max_model_len=128,
        dtype="float32",
    ) as vllm_model:
        tokenizer = vllm_model.llm.get_tokenizer()
        pieces = [
            "(",
            "[P]",
            "route: Which action?",
            "(",
            "[L]",
            "cancel",
            "[L]",
            "refund",
            ")",
            ")",
            "[SEP_TEXT]",
            "please",
            "cancel",
            "the",
            "order",
            ".",
        ]
        encoded = tokenizer(pieces, add_special_tokens=False)["input_ids"]
        positions = [sum(map(len, encoded[:index])) for index in (4, 6)]
        ids = [token_id for chunk in encoded for token_id in chunk]
        output = vllm_model.llm.classify(
            [{"prompt_token_ids": ids}],
            pooling_params=PoolingParams(
                task="classify",
                use_activation=False,
                extra_kwargs={"label_positions": positions},
            ),
            use_tqdm=False,
        )[0]
    logits = torch.tensor(output.outputs.probs)
    assert logits.shape == (2,)
    assert logits[0] > logits[1]


def test_decide_serves_both_endpoints_with_default_flags():
    """Default serving must load the checkpoint and score request-specific labels."""
    model = "fastino/GLiNER2.5-Decide"
    state = "Please cancel my subscription."
    question = {"type": "choice", "instructions": "What does the customer want?"}
    with RemoteOpenAIServer(model, []) as server:
        for endpoint, payload in (
            (
                "decisions",
                {
                    "input": state,
                    "questions": [
                        {
                            **question,
                            "name": "intent",
                            "choices": [{"value": "cancel"}, {"value": "refund"}],
                        }
                    ],
                },
            ),
            (
                "systemone",
                {
                    "state": state,
                    "questions": {
                        "intent": {
                            **question,
                            "criteria": {"cancel": None, "refund": None},
                        }
                    },
                },
            ),
        ):
            response = requests.post(
                server.url_for(f"v1/{endpoint}"),
                json={"model": model, **payload},
                timeout=120,
            )
            assert response.status_code == 200, response.text
            body = response.json()
            answer = body["answers"][0 if endpoint == "decisions" else "intent"]
            assert answer["choice"] == "cancel"
            assert 0 <= answer["confidence"] <= 1
            assert body["usage"]["input_tokens"] > 0
            assert body["usage"]["output_tokens"] == 0
