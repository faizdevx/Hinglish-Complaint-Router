"""Raw / processed data access and label vocabularies."""
from __future__ import annotations

import json
from pathlib import Path

from .config import DATA_PROCESSED, DATA_RAW, URGENCY_LABELS


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError as e:  # malformed record: report, don't hide
                raise ValueError(f"{path}:{i}: malformed JSON ({e})") from e
    return rows


def write_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def load_raw(subset: str) -> list[dict]:
    return read_jsonl(DATA_RAW / "v1" / f"{subset}.jsonl")


def item_label(row: dict) -> str:
    """Resolve the gold option string of a choice item."""
    return row["question"]["options"][row["expected"]]


def load_split(split: str) -> list[dict]:
    path = DATA_PROCESSED / f"{split}.jsonl"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_dataset.py")
    return read_jsonl(path)


def load_label_maps() -> dict:
    path = DATA_PROCESSED / "label_maps.json"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run: python scripts/prepare_dataset.py")
    return json.loads(path.read_text(encoding="utf-8"))


def default_label_maps(intents: list[str]) -> dict:
    return {"intent": intents, "urgency": URGENCY_LABELS, "escalation": ["no", "yes"]}
