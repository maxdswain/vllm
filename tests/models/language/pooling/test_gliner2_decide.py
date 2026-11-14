# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Regression coverage for the GLiNER2.5 Decide classification path."""

import pytest
import torch
from transformers import DebertaV2Config, DebertaV2Model

from vllm.config import ModelConfig, VllmConfig
from vllm.model_executor.model_loader.utils import process_weights_after_loading
from vllm.model_executor.model_loader.weight_utils import default_weight_loader
from vllm.model_executor.models.deberta_v2 import DebertaV2EncoderModel
from vllm.model_executor.models.gliner2 import Gliner2ForSchemaExtraction
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
    model = Gliner2ForSchemaExtraction(
        vllm_config=VllmConfig(model_config=model_config)
    )
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
            [model.encoder(input_ids=row[None])[0] for row in (first, second)]
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
@pytest.mark.parametrize("use_flash_kernel", [True, False])
def test_native_deberta_relative_attention_matches_transformers_on_cpu(
    dist_init, conv_kernel_size, use_flash_kernel
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
    reference = DebertaV2Model(config).eval()
    model = DebertaV2EncoderModel(config).eval()
    params = dict(model.named_parameters())
    for name, weight in reference.named_parameters():
        param = params[name]
        getattr(param, "weight_loader", default_weight_loader)(param, weight)
    for module in model.modules():
        if hasattr(module, "quant_method"):
            module.quant_method.process_weights_after_loading(module)
    for layer in model.encoder.layer:
        layer.attention.self.use_flash_kernel = use_flash_kernel

    ids = torch.tensor([[1, 2, 3, 4, 5], [6, 7, 8, 0, 0]])
    valid = ids.ne(0)
    mask = valid.long()
    with torch.no_grad():
        actual = model(input_ids=ids, attention_mask=mask)
        expected = reference(input_ids=ids, attention_mask=mask).last_hidden_state
    torch.testing.assert_close(actual[valid], expected[valid], atol=1e-5, rtol=1e-5)


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
