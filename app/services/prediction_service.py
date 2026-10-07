"""Owns the loaded models. Models are loaded once (app startup) and reused."""
from __future__ import annotations

import threading
import time

from src.config import MODEL_DISPLAY, MODEL_KEYS, MODELS_DIR
from src.inference import ModelUnavailable, Predictor

UNAVAILABLE_MSG = "{name} checkpoint is not available. Run the training pipeline first (python scripts/train_{key}.py)."


class PredictionService:
    def __init__(self, models_dir=MODELS_DIR):
        self.models_dir = models_dir
        self.predictors: dict[str, Predictor] = {}
        self.errors: dict[str, str] = {}
        self._lock = threading.Lock()

    def load(self) -> None:
        for key in MODEL_KEYS:
            try:
                self.predictors[key] = Predictor.load(self.models_dir / key, key)
            except ModelUnavailable:
                self.errors[key] = UNAVAILABLE_MSG.format(name=MODEL_DISPLAY[key], key=key)
            except Exception as e:  # corrupt checkpoint etc.: report, keep app alive
                self.errors[key] = f"{MODEL_DISPLAY[key]} failed to load: {type(e).__name__}: {e}"

    def available(self, key: str) -> bool:
        return key in self.predictors

    def default_model(self) -> str | None:
        for k in ("transformer", "lstm"):
            if k in self.predictors:
                return k
        return None

    def status(self) -> dict:
        out = {}
        for key in MODEL_KEYS:
            if key in self.predictors:
                out[key] = self.predictors[key].metadata()
            else:
                out[key] = {"key": key, "name": MODEL_DISPLAY[key], "status": "unavailable",
                            "message": self.errors.get(key, UNAVAILABLE_MSG.format(name=MODEL_DISPLAY[key], key=key))}
        return out

    def predict(self, text: str, model: str | None = None, language: str | None = None, top_k: int = 5) -> dict:
        key = model or self.default_model()
        if key is None or key not in self.predictors:
            name = MODEL_DISPLAY.get(key or "transformer", "Model")
            raise ModelUnavailable(self.errors.get(key or "", UNAVAILABLE_MSG.format(name=name, key=key or "transformer")))
        t0 = time.perf_counter()
        with self._lock:  # torch modules are not guaranteed thread-safe for concurrent forward in eval
            res = self.predictors[key].predict(text, top_k=top_k)
        res["service_ms"] = (time.perf_counter() - t0) * 1000
        res["language_source"] = "model/script-rule"
        if language and language != "auto":
            res["language_detected"] = res["language"]
            res["language"] = language
            res["language_source"] = "user-selected (not used by the model)"
        res["checkpoint"] = self.predictors[key].metadata()["checkpoint"]
        res["device"] = str(self.predictors[key].device)
        return res

    def compare(self, text: str, language: str | None = None) -> dict:
        out = {}
        for key in MODEL_KEYS:
            try:
                out[key] = {"ok": True, **self.predict(text, key, language)}
            except ModelUnavailable as e:
                out[key] = {"ok": False, "error": str(e), "model_name": MODEL_DISPLAY[key]}
        return out
