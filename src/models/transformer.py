"""MuRIL (or any BERT-style encoder) with multi-task heads on the [CLS] state."""
from __future__ import annotations

import torch
from torch import nn
from transformers import AutoModel, BertConfig, BertModel


class TransformerMultiTask(nn.Module):
    def __init__(self, model_name: str, n_intent: int, n_urgency: int = 5, dropout: float = 0.1,
                 pretrained: bool = True, encoder_config: dict | None = None):
        super().__init__()
        if pretrained:
            self.encoder = AutoModel.from_pretrained(model_name, add_pooling_layer=False)
        else:  # randomly initialised small encoder (unit tests only; never used for results)
            self.encoder = BertModel(BertConfig(**encoder_config), add_pooling_layer=False)
        hid = self.encoder.config.hidden_size
        self.cfg = dict(model_name=model_name, n_intent=n_intent, n_urgency=n_urgency, dropout=dropout,
                        encoder_config=self.encoder.config.to_dict())
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(hid, n_intent)
        self.urgency_head = nn.Linear(hid, n_urgency)
        self.escalation_head = nn.Linear(hid, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        h = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state[:, 0]
        h = self.drop(h)
        return {"intent": self.intent_head(h), "urgency": self.urgency_head(h),
                "escalation": self.escalation_head(h).squeeze(-1)}
