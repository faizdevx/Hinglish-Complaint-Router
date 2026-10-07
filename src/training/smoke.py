"""Post-training smoke verification: reload the saved checkpoint and run real inference."""
from __future__ import annotations

from ..config import MODELS_DIR
from ..data import load_split
from ..inference import Predictor


def smoke_check(model_key: str) -> dict:
    p = Predictor.load(MODELS_DIR / f"smoke_{model_key}", model_key)
    rows = [r for r in load_split("val") if r["intent"] is not None][:3] + \
           [r for r in load_split("val") if r["intent"] is None][:2]
    results = p.predict_batch([r["text"] for r in rows])
    for r, res in zip(rows, results):
        assert 0.0 <= res["escalation_probability"] <= 1.0
    print(f"[smoke:{model_key}] checkpoint reload + inference OK on {len(rows)} real val rows; "
          f"sample: {results[0]['intent']} / {results[0]['urgency']} / p_esc={results[0]['escalation_probability']:.3f}")
    return results[0]
