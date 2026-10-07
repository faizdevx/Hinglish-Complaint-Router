"""Reads saved artifacts only (metrics JSON, audit JSON, error CSV, figures). Never computes metrics."""
from __future__ import annotations

import csv
import json
from functools import lru_cache
from pathlib import Path

from src.config import DATA_PROCESSED, FIGURES, METRICS, MODEL_DISPLAY, MODEL_KEYS, REPORTS
from src.data import read_jsonl

METRIC_FILES = {"lstm": "cnn_or_lstm_metrics.json", "transformer": "transformer_metrics.json"}
HOWTO_EVAL = "Run: python scripts/evaluate.py"

# Manually written demonstration inputs. NOT dataset samples; the UI labels them "Demo example".
DEMO_EXAMPLES = [
    {"group": "English", "text": "Water supply has stopped since morning."},
    {"group": "Hindi (Devanagari)", "text": "सुबह से पानी नहीं आया है।"},
    {"group": "Hinglish / Romanised Hindi", "text": "Subah se paani nahi aaya."},
    {"group": "Hinglish / Romanised Hindi", "text": "Mera card block ho gaya hai aur paise kat gaye"},
    {"group": "English", "text": "I was charged twice for the same card payment."},
]


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None


class MetricsService:
    def __init__(self):
        self._err_cache: tuple[float, list[dict]] | None = None

    # -- metrics
    def comparison(self):
        return _read_json(METRICS / "comparison.json")

    def model_metrics(self, key: str):
        return _read_json(METRICS / METRIC_FILES[key])

    def audit(self):
        return _read_json(METRICS / "data_audit.json")

    def langid(self):
        return _read_json(METRICS / "langid_metrics.json")

    def all_metrics(self) -> dict:
        return {"comparison": self.comparison(), "models": {k: self.model_metrics(k) for k in MODEL_KEYS},
                "hint": None if self.comparison() else HOWTO_EVAL}

    # -- dataset metadata
    def dataset_summary(self) -> dict | None:
        summ = _read_json(DATA_PROCESSED / "summary.json")
        if not summ:
            return None
        labels = _read_json(DATA_PROCESSED / "label_maps.json") or {}
        audit = self.audit() or {}
        langs = sorted({l for s in (audit.get("splits") or {}).values() for l in s.get("by_language", {})})
        return {"name": "IndicJevBench subsets: fintech_banking77 (intent) + synthetic_enterprise (urgency/escalation)",
                "examples": sum(s["rows"] for s in summ["splits"].values()), "splits": summ["splits"],
                "languages": langs, "n_intents": len(labels.get("intent", [])), "n_urgency": len(labels.get("urgency", []))}

    # -- figures
    def figures(self) -> dict[str, bool]:
        names = [f"confusion_intent_{k}.png" for k in MODEL_KEYS] + [f"confusion_urgency_{k}.png" for k in MODEL_KEYS] + \
                ["escalation_pr_curve.png", "intent_distribution.png", "urgency_distribution.png", "escalation_distribution.png",
                 "language_distribution.png", "text_length_distribution.png"]
        return {n: (FIGURES / n).exists() for n in names}

    # -- errors
    def _errors(self) -> list[dict]:
        p = REPORTS / "error_analysis.csv"
        if not p.exists():
            return []
        mtime = p.stat().st_mtime
        if self._err_cache and self._err_cache[0] == mtime:
            return self._err_cache[1]
        with open(p, newline="", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))
        self._err_cache = (mtime, rows)
        return rows

    def errors_available(self) -> bool:
        return (REPORTS / "error_analysis.csv").exists()

    def query_errors(self, model: str | None = None, q: str | None = None, language: str | None = None,
                     intent: str | None = None, limit: int = 100, offset: int = 0) -> dict:
        rows = self._errors()
        facets = {"languages": sorted({r["language"] for r in rows}),
                  "intents": sorted({r["true_intent"] for r in rows if r["true_intent"]})}
        if model:
            rows = [r for r in rows if r["model"] == model]
        if language:
            rows = [r for r in rows if r["language"] == language]
        if intent:
            rows = [r for r in rows if r["true_intent"] == intent or r["pred_intent"] == intent]
        if q:
            ql = q.lower()
            rows = [r for r in rows if ql in r["text"].lower() or ql in r["true_intent"].lower() or ql in r["pred_intent"].lower()]
        limit = max(1, min(limit, 500))
        return {"total": len(rows), "limit": limit, "offset": offset, "rows": rows[offset:offset + limit], "facets": facets}

    # -- examples
    def dataset_examples(self, per_group: int = 2) -> list[dict]:
        """Real test-split rows (deterministic: first rows per language) for UI quick tests."""
        p = DATA_PROCESSED / "test.jsonl"
        if not p.exists():
            return []
        groups = {"en-Latn": "English", "hi-Deva": "Hindi (Devanagari)", "hi-Latn": "Hinglish / Romanised Hindi"}
        counts: dict[str, int] = {}
        out = []
        for r in read_jsonl(p):
            lang = r["lang"]
            g = groups.get(lang, "Other supported languages")
            key = lang if g == "Other supported languages" else g
            if counts.get(key, 0) >= (1 if g == "Other supported languages" else per_group):
                continue
            if len(r["text"]) > 140:
                continue
            counts[key] = counts.get(key, 0) + 1
            out.append({"group": g, "language": lang, "text": r["text"], "true_intent": r["intent"],
                        "true_urgency": r["urgency"], "true_escalation": r["escalation"],
                        "provenance": r["provenance"]})
        return out

    def demo_examples(self) -> list[dict]:
        return DEMO_EXAMPLES

    def model_comparison_rows(self) -> list[dict] | None:
        comp = self.comparison()
        if not comp:
            return None
        spec = [("Intent macro-F1", "intent_macro_f1", "{:.3f}", True), ("Urgency macro-F1", "urgency_macro_f1", "{:.3f}", True),
                ("Escalation F1", "escalation_f1", "{:.3f}", True), ("Escalation PR-AUC", "escalation_pr_auc", "{:.3f}", True),
                ("Parameters", "parameters", "{:,}", False), ("CPU latency (median, ms)", "cpu_latency_median_ms", "{:.1f}", False)]
        rows = []
        for label, field, fmt, higher_better in spec:
            vals = {k: comp["models"].get(k, {}).get(field) for k in MODEL_KEYS}
            present = [v for v in vals.values() if v is not None]
            bar_max = max(present) if present else None
            rows.append({"label": label, "field": field, "higher_better": higher_better, "bar": field in (
                "intent_macro_f1", "urgency_macro_f1", "escalation_f1", "escalation_pr_auc"),
                "cells": {k: {"value": v, "text": fmt.format(v) if v is not None else "Not available",
                              "pct": (100 * v / bar_max) if (v is not None and bar_max) else 0} for k, v in vals.items()}})
        return rows

    # -- chart data (values straight from comparison.json; bar width = value*100 for 0..1 metrics)
    def charts(self) -> dict | None:
        comp = self.comparison()
        if not comp:
            return None

        def cell(v, n=None):
            return {"text": "n/a" if v is None else f"{v:.3f}", "pct": 0 if v is None else 100 * v, "n": n}

        overall = {}
        for title, field in (("Intent macro-F1", "intent_macro_f1"), ("Urgency macro-F1", "urgency_macro_f1"),
                             ("Escalation F1", "escalation_f1")):
            overall[title] = [{"label": MODEL_DISPLAY[k], "cls": "b" if i else "", **cell(comp["models"][k].get(field))}
                              for i, k in enumerate(MODEL_KEYS) if k in comp["models"]]
        langs = {}
        for title, field, nfield in (("Intent macro-F1 by language", "intent_macro_f1", "intent_n"),
                                     ("Urgency macro-F1 by language", "urgency_macro_f1", "urgency_n"),
                                     ("Escalation F1 by language", "escalation_f1", "escalation_n")):
            rows = []
            for lang, per in comp["language_table"].items():
                if not any(per.get(k, {}).get(nfield) for k in MODEL_KEYS):
                    continue
                rows.append({"label": lang, "bars": [{"label": MODEL_DISPLAY[k], "cls": "b" if i else "",
                             **cell(per[k][field], per[k][nfield])} for i, k in enumerate(MODEL_KEYS) if k in per]})
            langs[title] = rows
        return {"overall": overall, "languages": langs}
