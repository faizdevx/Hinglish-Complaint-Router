"""Batch inference: CSV with a `text` column (optional `id`) -> CSV of predictions."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.batch import BatchError, parse_csv, run_batch, to_csv  # noqa: E402
from src.config import MODELS_DIR  # noqa: E402
from src.inference import ModelUnavailable, Predictor  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("input")
    ap.add_argument("output")
    ap.add_argument("--model", choices=["lstm", "transformer"], default="transformer")
    a = ap.parse_args()
    try:
        rows = parse_csv(Path(a.input).read_bytes())
        pred = Predictor.load(MODELS_DIR / a.model, a.model)
    except (BatchError, ModelUnavailable, FileNotFoundError) as e:
        print("error:", e, file=sys.stderr)
        return 1
    res = run_batch(pred, rows)
    Path(a.output).write_text(to_csv(res["results"]), encoding="utf-8")
    print(f"{res['n_success']}/{res['n_inputs']} predicted ({res['n_failed']} failed) in "
          f"{res['total_seconds']:.2f}s -> {a.output}")
    for f in res["failed"][:5]:
        print("  failed row", f["id"], f["error"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
