"""Metrics, calibration, language slicing and bootstrap intervals."""
from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (accuracy_score, average_precision_score, brier_score_loss, confusion_matrix,
                             f1_score, precision_recall_fscore_support, roc_auc_score)

from .config import ESCALATION_DECISION_THRESHOLD


@torch.no_grad()
def collect_outputs(model, loader, device) -> dict[str, np.ndarray]:
    model.eval()
    acc: dict[str, list] = {k: [] for k in ("intent_logits", "urgency_logits", "escalation_logit",
                                            "y_intent", "y_urgency", "y_escalation", "index")}
    for b in loader:
        out = model(b["input_ids"].to(device), b["attention_mask"].to(device))
        acc["intent_logits"].append(out["intent"].float().cpu().numpy())
        acc["urgency_logits"].append(out["urgency"].float().cpu().numpy())
        acc["escalation_logit"].append(out["escalation"].float().cpu().numpy())
        acc["y_intent"].append(b["intent"].numpy())
        acc["y_urgency"].append(b["urgency"].numpy())
        acc["y_escalation"].append(b["escalation"].numpy())
        acc["index"].append(b["index"].numpy())
    return {k: np.concatenate(v) for k, v in acc.items()}


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.asarray(x, dtype=np.float64)))


def expected_calibration_error(y, p, n_bins: int = 10) -> float:
    y, p = np.asarray(y, float), np.asarray(p, float)
    bins = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(bins[:-1], bins[1:]):
        m = (p >= lo) & ((p < hi) | (hi == 1.0) & (p <= hi))
        if m.any():
            ece += m.mean() * abs(y[m].mean() - p[m].mean())
    return float(ece)


def intent_metrics(y, pred) -> dict | None:
    if len(y) == 0:
        return None
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="macro", zero_division=0)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)), "macro_precision": float(p),
            "macro_recall": float(r), "macro_f1": float(f),
            "weighted_f1": float(f1_score(y, pred, average="weighted", zero_division=0))}


def urgency_metrics(y, pred, n_classes: int = 5) -> dict | None:
    if len(y) == 0:
        return None
    y, pred = np.asarray(y), np.asarray(pred)
    return {"n": int(len(y)), "accuracy": float(accuracy_score(y, pred)),
            "macro_f1": float(f1_score(y, pred, average="macro", zero_division=0)),
            "mae_levels": float(np.abs(y - pred).mean()),
            "within_one_accuracy": float((np.abs(y - pred) <= 1).mean()),
            "confusion_matrix": confusion_matrix(y, pred, labels=list(range(n_classes))).tolist()}


def escalation_metrics(y, prob, threshold: float = ESCALATION_DECISION_THRESHOLD) -> dict | None:
    if len(y) == 0:
        return None
    y, prob = np.asarray(y), np.asarray(prob, dtype=np.float64)
    pred = (prob >= threshold).astype(int)
    p, r, f, _ = precision_recall_fscore_support(y, pred, average="binary", zero_division=0)
    two = len(set(y.tolist())) == 2
    return {"n": int(len(y)), "n_positive": int(y.sum()), "threshold": threshold,
            "roc_auc": float(roc_auc_score(y, prob)) if two else None,
            "pr_auc": float(average_precision_score(y, prob)) if two else None,
            "precision": float(p), "recall": float(r), "f1": float(f),
            "brier": float(brier_score_loss(y, prob)), "ece_10bin": expected_calibration_error(y, prob),
            "positive_rate": float(y.mean())}


def temperature_nll(logits, y, T: float) -> float:
    z = np.asarray(logits, float) / T
    return float(np.mean(np.logaddexp(0, z) - np.asarray(y) * z))


def fit_temperature(logits, y) -> float | None:
    """1-parameter temperature scaling on validation logits (grid search on NLL)."""
    if len(y) < 10 or len(set(np.asarray(y).tolist())) < 2:
        return None
    grid = np.exp(np.linspace(np.log(0.25), np.log(8.0), 200))
    return float(grid[int(np.argmin([temperature_nll(logits, y, t) for t in grid]))])


def summarize(out: dict, rows: list[dict], temperature: float | None = None, by_lang: bool = True) -> dict:
    """Compute all task metrics (+ per-language slices) from collected outputs."""
    order = out["index"]
    langs = np.array([rows[i]["lang"] for i in order])
    yi, yu, ye = out["y_intent"], out["y_urgency"], out["y_escalation"]
    pi, pu = out["intent_logits"].argmax(1), out["urgency_logits"].argmax(1)
    z = out["escalation_logit"]
    pe = sigmoid(z)
    pe_cal = sigmoid(z / temperature) if temperature else None

    def block(mask_fn):
        mi, mu, me = (yi >= 0) & mask_fn, (yu >= 0) & mask_fn, (ye >= 0) & mask_fn
        res = {"intent": intent_metrics(yi[mi], pi[mi]), "urgency": urgency_metrics(yu[mu], pu[mu]),
               "escalation": escalation_metrics(ye[me], pe[me])}
        if pe_cal is not None:
            res["escalation_calibrated"] = escalation_metrics(ye[me], pe_cal[me])
        return res

    res = {"overall": block(np.ones(len(order), bool)), "temperature": temperature}
    if by_lang:
        res["by_language"] = {l: block(langs == l) for l in sorted(set(langs.tolist()))}
    return res


def bootstrap_ci(fn, *arrays, n_boot: int = 1000, seed: int = 0, alpha: float = 0.05):
    rng = np.random.default_rng(seed)
    n = len(arrays[0])
    vals = []
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        try:
            v = fn(*[a[idx] for a in arrays])
        except ValueError:
            continue
        if v is not None and not np.isnan(v):
            vals.append(v)
    if len(vals) < 20:
        return None
    return [float(np.percentile(vals, 100 * alpha / 2)), float(np.percentile(vals, 100 * (1 - alpha / 2)))]
