"""BiLSTM multi-task baseline."""
from __future__ import annotations

import torch
from torch import nn


class BiLSTMMultiTask(nn.Module):
    def __init__(self, vocab_size: int, n_intent: int, n_urgency: int = 5, emb_dim: int = 128,
                 hidden: int = 128, layers: int = 1, dropout: float = 0.3, shared_dim: int = 256):
        super().__init__()
        self.cfg = dict(vocab_size=vocab_size, n_intent=n_intent, n_urgency=n_urgency, emb_dim=emb_dim,
                        hidden=hidden, layers=layers, dropout=dropout, shared_dim=shared_dim)
        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=0)
        self.lstm = nn.LSTM(emb_dim, hidden, num_layers=layers, batch_first=True, bidirectional=True,
                            dropout=dropout if layers > 1 else 0.0)
        self.shared = nn.Sequential(nn.Linear(4 * hidden, shared_dim), nn.ReLU(), nn.Dropout(dropout))
        self.drop = nn.Dropout(dropout)
        self.intent_head = nn.Linear(shared_dim, n_intent)
        self.urgency_head = nn.Linear(shared_dim, n_urgency)
        self.escalation_head = nn.Linear(shared_dim, 1)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> dict[str, torch.Tensor]:
        x = self.drop(self.embedding(input_ids))
        lengths = attention_mask.sum(1).clamp(min=1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(x, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.lstm(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=input_ids.size(1))
        m = attention_mask.unsqueeze(-1).to(out.dtype)
        mean = (out * m).sum(1) / m.sum(1).clamp(min=1)
        mx = out.masked_fill(m == 0, -1e9).max(1).values
        h = self.shared(torch.cat([mean, mx], dim=-1))
        return {"intent": self.intent_head(h), "urgency": self.urgency_head(h),
                "escalation": self.escalation_head(h).squeeze(-1)}
