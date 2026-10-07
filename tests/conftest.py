"""Shared fixtures. Unit tests use tiny randomly-initialised models + a tiny WordPiece vocab so they run
offline and quickly. These are test fixtures only; no reported metric ever comes from them."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest
import torch

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from transformers import BertTokenizerFast  # noqa: E402

from src.models import BiLSTMMultiTask, TransformerMultiTask  # noqa: E402

INTENTS = ["card_arrival", "Refund_not_showing_up", "compromised_card", "country_support", "pin_blocked"]
LABELS = {"intent": INTENTS, "urgency": ["very_low", "low", "medium", "high", "critical"], "escalation": ["no", "yes"]}
WORDS = ["mera", "card", "hai", "refund", "paani", "nahi", "aaya", "block", "ho", "gaya", "the", "my", "i", "want", "to"]
TINY_ENC = dict(vocab_size=64, hidden_size=16, num_hidden_layers=1, num_attention_heads=2, intermediate_size=32,
                max_position_embeddings=64)


@pytest.fixture(scope="session")
def tiny_tokenizer(tmp_path_factory):
    d = tmp_path_factory.mktemp("vocab")
    vocab = ["[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]"] + WORDS + list(".,?!0123456789")
    vocab += [chr(c) for c in range(ord("a"), ord("z") + 1)] + ["##" + chr(c) for c in range(ord("a"), ord("z") + 1)]
    return BertTokenizerFast(vocab={t: i for i, t in enumerate(vocab)}, do_lower_case=False)


def write_tiny_model(path: Path, key: str, tok) -> Path:
    torch.manual_seed(0)
    path.mkdir(parents=True, exist_ok=True)
    if key == "lstm":
        id_map = {tok.pad_token_id: 0, tok.unk_token_id: 1}
        for t in range(len(tok)):
            id_map.setdefault(t, len(id_map))
        model = BiLSTMMultiTask(len(id_map), len(INTENTS), emb_dim=8, hidden=8, shared_dim=16)
        (path / "vocab.json").write_text(json.dumps({str(k): v for k, v in id_map.items()}))
    else:
        enc = dict(TINY_ENC, vocab_size=len(tok))
        model = TransformerMultiTask("tiny", len(INTENTS), pretrained=False, encoder_config=enc)
    torch.save({"state_dict": model.state_dict(), "epoch": 1}, path / "best.pt")
    tok.save_pretrained(path / "tokenizer")
    (path / "labels.json").write_text(json.dumps(LABELS))
    (path / "config.json").write_text(json.dumps({
        "model_key": key, "arch": model.cfg, "parameters": sum(p.numel() for p in model.parameters()),
        "tokenizer_id": "tiny-test", "model_id": "tiny-test", "max_length": 32, "best_epoch": 1, "epochs_run": 1,
        "seed": 0, "learning_rate": 0.0, "optimizer": "none", "trained_at": "test"}))
    return path


@pytest.fixture(scope="session")
def tiny_models(tmp_path_factory, tiny_tokenizer):
    root = tmp_path_factory.mktemp("models")
    for key in ("lstm", "transformer"):
        write_tiny_model(root / key, key, tiny_tokenizer)
    return root


@pytest.fixture()
def client(tiny_models, monkeypatch):
    from fastapi.testclient import TestClient
    import app.main as m
    from app.services.prediction_service import PredictionService
    monkeypatch.setattr(m, "PredictionService", lambda: PredictionService(models_dir=tiny_models))
    with TestClient(m.app) as c:
        yield c


@pytest.fixture()
def client_no_transformer(tiny_models, tmp_path, monkeypatch):
    """Only the LSTM checkpoint exists."""
    import shutil
    from fastapi.testclient import TestClient
    import app.main as m
    from app.services.prediction_service import PredictionService
    shutil.copytree(tiny_models / "lstm", tmp_path / "lstm")
    monkeypatch.setattr(m, "PredictionService", lambda: PredictionService(models_dir=tmp_path))
    with TestClient(m.app) as c:
        yield c
