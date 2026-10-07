"""CSV batch inference shared by the CLI script and the web app."""
from __future__ import annotations

import csv
import io
import time

from .preprocessing import InvalidInput, normalize_text

OUT_COLUMNS = ["id", "text", "language", "intent", "urgency", "escalation_probability", "route"]


class BatchError(ValueError):
    """Unusable CSV (missing column, undecodable, too large...)."""


def parse_csv(raw: bytes, max_rows: int = 5000) -> list[dict]:
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError as e:
        raise BatchError("File is not valid UTF-8 text.") from e
    if not text.strip():
        raise BatchError("CSV is empty.")
    try:
        reader = csv.DictReader(io.StringIO(text))
        fields = [f.strip().lower() for f in (reader.fieldnames or [])]
        if "text" not in fields:
            raise BatchError("CSV must contain a 'text' column (optional: 'id').")
        rows = []
        for n, rec in enumerate(reader, start=1):
            rec = {(k or "").strip().lower(): v for k, v in rec.items()}
            rows.append({"id": (rec.get("id") or str(n)).strip() or str(n), "text": rec.get("text")})
            if len(rows) > max_rows:
                raise BatchError(f"Too many rows (limit {max_rows}).")
    except csv.Error as e:
        raise BatchError(f"Malformed CSV: {e}") from e
    if not rows:
        raise BatchError("CSV has a header but no rows.")
    return rows


def run_batch(predictor, rows: list[dict], batch_size: int = 32) -> dict:
    """Predict every valid row; invalid rows are reported, not fatal."""
    t0 = time.perf_counter()
    valid, failed = [], []
    for r in rows:
        try:
            valid.append({**r, "text": normalize_text(r["text"] if r["text"] is not None else "")})
        except InvalidInput as e:
            failed.append({"id": r["id"], "text": r["text"] or "", "error": str(e)})
    results = []
    for s in range(0, len(valid), batch_size):
        chunk = valid[s:s + batch_size]
        preds = predictor.predict_batch([c["text"] for c in chunk], top_k=1)
        for c, p in zip(chunk, preds):
            results.append({"id": c["id"], "text": c["text"], "language": p["language"], "intent": p["intent"],
                            "urgency": p["urgency"], "escalation_probability": round(p["escalation_probability"], 4),
                            "route": p["route"], "inference_ms": p["inference_ms"]})
    total_s = time.perf_counter() - t0
    return {"n_inputs": len(rows), "n_success": len(results), "n_failed": len(failed), "failed": failed,
            "results": results, "total_seconds": total_s,
            "mean_inference_ms": (sum(r["inference_ms"] for r in results) / len(results)) if results else None}


def to_csv(results: list[dict]) -> str:
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=OUT_COLUMNS, extrasaction="ignore", lineterminator="\n")
    w.writeheader()
    w.writerows(results)
    return buf.getvalue()
