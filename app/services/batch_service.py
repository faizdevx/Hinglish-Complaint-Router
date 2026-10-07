from __future__ import annotations

from src.batch import BatchError, parse_csv, run_batch, to_csv
from src.inference import ModelUnavailable

MAX_BYTES = 2 * 1024 * 1024
MAX_ROWS = 1000


def process_upload(prediction_service, raw: bytes, model: str | None) -> dict:
    if len(raw) > MAX_BYTES:
        raise BatchError(f"File too large (limit {MAX_BYTES // 1024 // 1024} MB).")
    rows = parse_csv(raw, max_rows=MAX_ROWS)
    key = model or prediction_service.default_model()
    if key is None or not prediction_service.available(key):
        raise ModelUnavailable(prediction_service.errors.get(key or "", "No model is available. Run the training pipeline first."))
    res = run_batch(prediction_service.predictors[key], rows)
    res["model"] = key
    res["csv"] = to_csv(res["results"])
    return res
