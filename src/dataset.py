"""Torch dataset / collation for the multi-task data.

Missing labels (a row only has the labels its source dataset provides) are encoded as -1
and are masked out of the loss.
"""
from __future__ import annotations

from typing import Callable

import torch
from torch.utils.data import Dataset

MISSING = -1


class ComplaintDataset(Dataset):
    def __init__(self, rows: list[dict], label_maps: dict):
        self.rows = rows
        self.intent_idx = {l: i for i, l in enumerate(label_maps["intent"])}

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, i: int) -> dict:
        r = self.rows[i]
        return {
            "text": r["text"],
            "intent": self.intent_idx[r["intent"]] if r.get("intent") is not None else MISSING,
            "urgency": int(r["urgency"]) if r.get("urgency") is not None else MISSING,
            "escalation": int(r["escalation"]) if r.get("escalation") is not None else MISSING,
            "index": i,
        }


def make_collate(tokenizer, max_length: int, id_map: dict[int, int] | None = None) -> Callable:
    """Collate with dynamic padding. ``id_map`` remaps tokenizer ids to a compact vocab (BiLSTM)."""
    pad = 0

    def collate(batch: list[dict]) -> dict:
        enc = tokenizer([b["text"] for b in batch], truncation=True, max_length=max_length,
                        padding=True, return_tensors="pt")
        ids = enc["input_ids"]
        if id_map is not None:
            unk = id_map.get(tokenizer.unk_token_id, 1)
            flat = [id_map.get(int(t), unk) for t in ids.flatten().tolist()]
            ids = torch.tensor(flat, dtype=torch.long).view_as(ids)
            ids = torch.where(enc["attention_mask"] == 1, ids, torch.full_like(ids, pad))
        return {
            "input_ids": ids,
            "attention_mask": enc["attention_mask"],
            "intent": torch.tensor([b["intent"] for b in batch], dtype=torch.long),
            "urgency": torch.tensor([b["urgency"] for b in batch], dtype=torch.long),
            "escalation": torch.tensor([b["escalation"] for b in batch], dtype=torch.long),
            "index": torch.tensor([b["index"] for b in batch], dtype=torch.long),
        }

    return collate


def build_compact_vocab(tokenizer, texts: list[str], max_length: int) -> dict[int, int]:
    """Map tokenizer ids seen in *training* text to 0..V-1 (0=PAD, 1=UNK). Fit on train only."""
    id_map = {tokenizer.pad_token_id: 0, tokenizer.unk_token_id: 1}
    for ids in tokenizer(texts, truncation=True, max_length=max_length)["input_ids"]:
        for t in ids:
            if t not in id_map:
                id_map[t] = len(id_map)
    return id_map
