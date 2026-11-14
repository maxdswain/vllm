# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""GLiNER2.5 Decide's DeBERTa encoder and classification head."""

from collections.abc import Iterable, Set
from functools import cached_property

import torch
from torch import Tensor, nn
from transformers import DebertaV2Config
from transformers.models.deberta_v2.modeling_deberta_v2 import DebertaV2Model

from vllm.config import VllmConfig
from vllm.model_executor.layers.pooler.abstract import Pooler
from vllm.model_executor.models.interfaces_base import VllmModelForPooling
from vllm.model_executor.models.transformers.base import Base
from vllm.model_executor.models.transformers.deberta import FusedDebertaV2Attention
from vllm.sequence import IntermediateTensors
from vllm.tasks import PoolingTask
from vllm.v1.pool.metadata import PoolingMetadata


class GLiNER2DecidePooler(Pooler):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, 2 * hidden_size),
            nn.ReLU(),
            nn.Linear(2 * hidden_size, 1),
        )

    def get_supported_tasks(self) -> Set[PoolingTask]:
        return {"classify"}

    def forward(
        self, hidden_states: Tensor, pooling_metadata: PoolingMetadata
    ) -> list[Tensor]:
        outputs = []
        offset = 0
        for length, params in zip(
            pooling_metadata.prompt_lens.tolist(),
            pooling_metadata.pooling_params,
            strict=True,
        ):
            row = hidden_states[offset : offset + length]
            # The runner calls the pooler with a synthetic prompt at startup.
            if params.extra_kwargs is None:
                outputs.append(row.new_zeros(1))
            else:
                labels = params.extra_kwargs.get("label_positions")
                if not labels:
                    raise ValueError("GLiNER2.5 Decide requires label_positions")
                outputs.append(self.classifier(row[labels]).flatten())
            offset += length
        return outputs


class Gliner2ForSchemaExtraction(VllmModelForPooling, Base):
    """Serve GLiNER2.5 Decide with the Transformers modeling backend."""

    def __init__(self, *, vllm_config: VllmConfig, prefix: str = "") -> None:
        config = vllm_config.model_config.hf_config
        if config.architecture != "span" or config.counting_layer != "count_lstm":
            raise ValueError("Only the GLiNER2.5 Decide span checkpoint is supported")
        if (
            vllm_config.parallel_config.tensor_parallel_size != 1
            or vllm_config.parallel_config.pipeline_parallel_size != 1
        ):
            raise ValueError(
                "GLiNER2.5 Decide requires tensor and pipeline parallel size 1"
            )
        super().__init__(vllm_config=vllm_config, prefix=prefix)
        self.pooler = GLiNER2DecidePooler(self.model.config.hidden_size)
        self.pad_token_id = self.model.config.pad_token_id or 0

    @cached_property
    def _from_config_kwargs(self) -> dict:
        encoder_config = DebertaV2Config(**self.config.encoder_config)
        encoder_config.hidden_dropout_prob = 0.0
        encoder_config.attention_probs_dropout_prob = 0.0
        return {"config": encoder_config, "dtype": self.model_config.dtype}

    def _decorate_for_torch_compile(self) -> None:
        # DeBERTa's embedding mask specializes the padded sequence length.
        return

    def _create_model(self) -> DebertaV2Model:
        model = super()._create_model()
        if not isinstance(model, DebertaV2Model):
            raise TypeError("Expected Transformers' DebertaV2Model")
        for layer in model.encoder.layer:
            layer.attention.self.__class__ = FusedDebertaV2Attention
        return model

    def forward(
        self,
        input_ids: Tensor | None,
        positions: Tensor,
        intermediate_tensors: IntermediateTensors | None = None,
        inputs_embeds: Tensor | None = None,
        **kwargs,
    ) -> Tensor:
        if input_ids is None or inputs_embeds is not None:
            raise ValueError(
                "GLiNER2.5 Decide requires input_ids without inputs_embeds"
            )
        flat = input_ids.flatten()
        pos = positions.flatten()
        # Positions restart at zero for each sequence; the runner can append
        # zero-position padding after the last real sequence.
        nonzero = pos.nonzero().flatten()
        real = int(nonzero[-1]) + 1 if nonzero.numel() else flat.numel()
        starts = (pos[:real] == 0).nonzero().flatten().tolist()
        if len(starts) == 1:
            packed = self.model(input_ids=flat[:real][None], return_dict=False)[0][0]
        else:
            lengths = [b - a for a, b in zip(starts, [*starts[1:], real])]
            rows = (pos[:real] == 0).cumsum(0) - 1
            cols = pos[:real]
            ids = flat.new_full((len(lengths), max(lengths)), self.pad_token_id)
            ids[rows, cols] = flat[:real]
            mask = torch.zeros_like(ids)
            mask[rows, cols] = 1
            hidden = self.model(input_ids=ids, attention_mask=mask, return_dict=False)[
                0
            ]
            packed = hidden[rows, cols]
        if packed.shape[0] == flat.shape[0]:
            return packed
        return nn.functional.pad(packed, (0, 0, 0, flat.shape[0] - packed.shape[0]))

    def load_weights(self, weights: Iterable[tuple[str, Tensor]]) -> set[str]:
        def used_weights():
            for name, tensor in weights:
                if name.startswith("encoder."):
                    yield name.removeprefix("encoder."), tensor
                elif name.startswith("classifier."):
                    yield f"pooler.{name}", tensor

        loaded = super().load_weights(used_weights())
        missing = set(dict(self.named_parameters())) - loaded
        if missing:
            raise ValueError(f"Missing GLiNER2 Decide weights: {sorted(missing)}")
        return loaded
