"""Model loading and prediction (shared by API, UI, scripts and tests)."""
from __future__ import annotations

import json
import time
from pathlib import Path

import numpy as np
import torch
from transformers import AutoTokenizer

from .config import (ESCALATION_DECISION_THRESHOLD, MODEL_DISPLAY, MODELS_DIR, URGENCY_LABELS, escalation_band)
from .dataset import make_collate
from .evaluation import sigmoid
from .langid import detect_language
from .models import BiLSTMMultiTask, TransformerMultiTask
from .preprocessing import normalize_text
from .routing import route_for


def default_device() -> torch.device:
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class ModelUnavailable(RuntimeError):
    pass


def softmax(x: np.ndarray) -> np.ndarray:
    e = np.exp(x - x.max(-1, keepdims=True))
    return e / e.sum(-1, keepdims=True)


class Predictor:
    """One loaded model + tokenizer + labels. Load once, reuse for every request."""

    def __init__(self, key: str, model, tokenizer, config: dict, labels: dict, id_map: dict | None,
                 device: torch.device, model_dir: Path, temperature: float | None):
        self.key, self.model, self.tokenizer, self.config, self.labels = key, model, tokenizer, config, labels
        self.id_map, self.device, self.model_dir, self.temperature = id_map, device, model_dir, temperature
        self.collate = make_collate(tokenizer, config["max_length"], id_map)
        self.model.eval()

    @classmethod
    def load(cls, model_dir: Path | str, key: str, device: torch.device | None = None) -> "Predictor":
        model_dir = Path(model_dir)
        for f in ("config.json", "labels.json", "best.pt", "tokenizer"):
            if not (model_dir / f).exists():
                raise ModelUnavailable(f"{model_dir / f} is missing. Run the training pipeline first "
                                       f"(python scripts/train_{key}.py).")
        device = device or default_device()
        config = json.loads((model_dir / "config.json").read_text())
        labels = json.loads((model_dir / "labels.json").read_text())
        arch = config["arch"]
        id_map = None
        if key == "lstm":
            model = BiLSTMMultiTask(**arch)
            id_map = {int(k): v for k, v in json.loads((model_dir / "vocab.json").read_text()).items()}
        else:
            model = TransformerMultiTask(arch["model_name"], arch["n_intent"], arch["n_urgency"], arch["dropout"],
                                         pretrained=False, encoder_config=arch["encoder_config"])
        ckpt = torch.load(model_dir / "best.pt", map_location="cpu", weights_only=True)
        model.load_state_dict(ckpt["state_dict"])
        model.to(device)
        tok = AutoTokenizer.from_pretrained(model_dir / "tokenizer")
        T = None
        cal = model_dir / "calibration.json"
        if cal.exists():
            T = json.loads(cal.read_text()).get("temperature")
        return cls(key, model, tok, config, labels, id_map, device, model_dir, T)

    # ------------------------------------------------------------------
    @torch.no_grad()
    def predict_batch(self, texts: list[str], top_k: int = 5, batch_size: int = 32) -> list[dict]:
        clean = [normalize_text(t) for t in texts]  # raises InvalidInput
        results: list[dict] = []
        for s in range(0, len(clean), batch_size):
            chunk = clean[s:s + batch_size]
            t0 = time.perf_counter()
            batch = self.collate([{"text": t, "intent": -1, "urgency": -1, "escalation": -1, "index": i}
                                  for i, t in enumerate(chunk)])
            out = self.model(batch["input_ids"].to(self.device), batch["attention_mask"].to(self.device))
            p_int = softmax(out["intent"].float().cpu().numpy())
            p_urg = softmax(out["urgency"].float().cpu().numpy())
            z = out["escalation"].float().cpu().numpy()
            elapsed_ms = (time.perf_counter() - t0) * 1000 / len(chunk)
            for i, text in enumerate(chunk):
                order = np.argsort(-p_int[i])[:top_k]
                intent = self.labels["intent"][int(order[0])]
                lang, script = detect_language(text)
                p_raw = float(sigmoid(z[i]))
                p_esc = float(sigmoid(z[i] / self.temperature)) if self.temperature else p_raw
                u = int(p_urg[i].argmax())
                results.append({
                    "text": text,
                    "intent": intent,
                    "intent_scores": [{"label": self.labels["intent"][int(j)], "score": float(p_int[i][j])}
                                      for j in order],
                    "urgency": URGENCY_LABELS[u],
                    "urgency_scores": {URGENCY_LABELS[k]: float(p_urg[i][k]) for k in range(len(URGENCY_LABELS))},
                    "escalation_probability": p_esc,
                    "escalation_probability_uncalibrated": p_raw,
                    "escalation_band": escalation_band(p_esc),
                    "escalate": p_esc >= ESCALATION_DECISION_THRESHOLD,
                    "calibrated": bool(self.temperature),
                    "language": lang,
                    "script": script,
                    "route": route_for(intent),
                    "model": self.key,
                    "model_name": MODEL_DISPLAY.get(self.key, self.key),
                    "inference_ms": elapsed_ms,  # normalised batch time: tokenise+forward+softmax, excl. langid/HTTP
                })
        return results

    def predict(self, text: str, top_k: int = 5) -> dict:
        return self.predict_batch([text], top_k=top_k)[0]

    def metadata(self) -> dict:
        c = self.config
        return {
            "key": self.key, "name": MODEL_DISPLAY.get(self.key, self.key), "status": "ready",
            "architecture": ("BiLSTM (embedding→BiLSTM→mean+max pool→shared MLP→3 heads)" if self.key == "lstm"
                             else f"{c.get('model_id')} encoder + [CLS]→3 linear heads"),
            "parameters": c.get("parameters"), "checkpoint": str(self.model_dir.relative_to(MODELS_DIR.parent))
            if self.model_dir.is_relative_to(MODELS_DIR.parent) else str(self.model_dir),
            "device": str(self.device), "trained_at": c.get("trained_at"), "max_length": c.get("max_length"),
            "best_epoch": c.get("best_epoch"), "epochs_run": c.get("epochs_run"), "seed": c.get("seed"),
            "learning_rate": c.get("learning_rate"), "optimizer": c.get("optimizer"),
            "temperature": self.temperature, "tokenizer_id": c.get("tokenizer_id"),
            "smoke_test": c.get("smoke_test", False),
        }
