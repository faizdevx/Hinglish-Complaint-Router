"""Evaluate trained models on the held-out TEST split.

Writes reports/metrics/*.json, reports/figures/*.png and reports/error_analysis.csv.
Temperature scaling for the escalation logit is fitted on the VALIDATION split only.
"""
from __future__ import annotations

import argparse
import csv
import json
import statistics
import sys
import time
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import torch  # noqa: E402
from sklearn.metrics import f1_score, precision_recall_curve  # noqa: E402
from torch.utils.data import DataLoader  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import (FIGURES, METRICS, MODEL_DISPLAY, MODEL_KEYS, MODELS_DIR, REPORTS, URGENCY_LABELS)  # noqa: E402
from src.dataset import ComplaintDataset  # noqa: E402
from src.data import load_label_maps, load_split  # noqa: E402
from src.evaluation import (bootstrap_ci, collect_outputs, escalation_metrics, fit_temperature, intent_metrics,  # noqa: E402
                            sigmoid, summarize, urgency_metrics)
from src.inference import Predictor  # noqa: E402

METRIC_FILES = {"lstm": "cnn_or_lstm_metrics.json", "transformer": "transformer_metrics.json"}


def outputs_for(pred: Predictor, rows, label_maps):
    dl = DataLoader(ComplaintDataset(rows, label_maps), batch_size=64, shuffle=False, collate_fn=pred.collate)
    return collect_outputs(pred.model, dl, pred.device)


def cpu_latency(pred: Predictor, texts: list[str], warmup: int = 5, runs: int = 60) -> dict:
    torch.set_num_threads(torch.get_num_threads())
    for t in texts[:warmup]:
        pred.predict(t)
    times = []
    for i in range(runs):
        t0 = time.perf_counter()
        pred.predict(texts[i % len(texts)])
        times.append((time.perf_counter() - t0) * 1000)
    times.sort()
    return {"device": "cpu", "threads": torch.get_num_threads(), "batch_size": 1, "runs": runs,
            "median_ms": statistics.median(times), "p95_ms": times[int(0.95 * (len(times) - 1))],
            "scope": "end-to-end predict(): normalise+tokenise+forward+softmax+language-id+routing"}


def by_length(out, rows, tokenizer):
    idx = out["index"]
    ntok = np.array([len(tokenizer(rows[i]["text"])["input_ids"]) for i in idx])
    yi, pi = out["y_intent"], out["intent_logits"].argmax(1)
    res = {}
    for name, m in (("<=8 tokens", ntok <= 8), ("9-16 tokens", (ntok > 8) & (ntok <= 16)), (">16 tokens", ntok > 16)):
        mm = m & (yi >= 0)
        res[name] = {"n": int(mm.sum()), "intent_accuracy": float((yi[mm] == pi[mm]).mean()) if mm.any() else None}
    return res


def majority_baseline(train_rows, test_rows, label_maps):
    intents = [r["intent"] for r in train_rows if r["intent"] is not None]
    maj_i = max(set(intents), key=intents.count)
    urg = [r["urgency"] for r in train_rows if r["urgency"] is not None]
    maj_u = max(set(urg), key=urg.count)
    esc = [r["escalation"] for r in train_rows if r["escalation"] is not None]
    p_pos = float(np.mean(esc))
    ti = [r for r in test_rows if r["intent"] is not None]
    tu = [r for r in test_rows if r["urgency"] is not None]
    te = [r for r in test_rows if r["escalation"] is not None]
    return {
        "intent": intent_metrics([r["intent"] for r in ti], [maj_i] * len(ti)) | {"predicts": maj_i},
        "urgency": urgency_metrics([r["urgency"] for r in tu], [maj_u] * len(tu)) | {"predicts": URGENCY_LABELS[maj_u]},
        "escalation_constant_prior": escalation_metrics(np.array([r["escalation"] for r in te]),
                                                        np.full(len(te), p_pos)) | {"prior_p": p_pos},
    }


def plot_confusion(cm, labels, title, path, annotate=True, figsize=(6, 5)):
    cm = np.asarray(cm)
    fig, ax = plt.subplots(figsize=figsize, dpi=110)
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks(range(len(labels)))
    ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=90 if len(labels) > 8 else 30, ha="center" if len(labels) > 8 else "right",
                       fontsize=4 if len(labels) > 30 else 9)
    ax.set_yticklabels(labels, fontsize=4 if len(labels) > 30 else 9)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    if annotate:
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, int(cm[i, j]), ha="center", va="center", fontsize=8,
                        color="white" if cm[i, j] > cm.max() / 2 else "black")
    fig.colorbar(im, fraction=0.046)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", nargs="*", default=list(MODEL_KEYS))
    a = ap.parse_args()
    METRICS.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    label_maps = load_label_maps()
    intents = label_maps["intent"]
    train_rows, val_rows, test_rows = load_split("train"), load_split("val"), load_split("test")
    results, err_rows, pr_curves = {}, [], {}
    for key in a.models:
        mdir = MODELS_DIR / key
        if not (mdir / "best.pt").exists():
            print(f"[{key}] checkpoint missing; skipping")
            continue
        (mdir / "calibration.json").unlink(missing_ok=True)
        pred = Predictor.load(mdir, key, device=torch.device("cpu"))
        # ---- calibration fitted on validation only
        vout = outputs_for(pred, val_rows, label_maps)
        vm = vout["y_escalation"] >= 0
        T = fit_temperature(vout["escalation_logit"][vm], vout["y_escalation"][vm])
        if T:
            (mdir / "calibration.json").write_text(json.dumps({
                "temperature": T, "fit_on": "validation split escalation logits", "n_val": int(vm.sum()),
                "note": "single-parameter temperature scaling; validation set is tiny so treat as indicative"}))
        pred.temperature = T
        out = outputs_for(pred, test_rows, label_maps)
        summ = summarize(out, test_rows, temperature=T)
        # bootstrap CIs on overall test metrics
        yi, pi = out["y_intent"], out["intent_logits"].argmax(1)
        mi = yi >= 0
        yu, pu = out["y_urgency"], out["urgency_logits"].argmax(1)
        mu = yu >= 0
        ye, pe = out["y_escalation"], sigmoid(out["escalation_logit"])
        me = ye >= 0
        from sklearn.metrics import average_precision_score
        summ["overall"]["intent"]["macro_f1_ci95"] = bootstrap_ci(lambda y, p: f1_score(y, p, average="macro", zero_division=0), yi[mi], pi[mi], n_boot=300)
        summ["overall"]["urgency"]["macro_f1_ci95"] = bootstrap_ci(lambda y, p: f1_score(y, p, average="macro", zero_division=0), yu[mu], pu[mu])
        summ["overall"]["escalation"]["pr_auc_ci95"] = bootstrap_ci(lambda y, p: average_precision_score(y, p), ye[me], pe[me])
        summ["by_length_intent"] = by_length(out, test_rows, pred.tokenizer)
        summ["latency_cpu"] = cpu_latency(pred, [r["text"] for r in test_rows if r["intent"] is not None][:20])
        summ["parameters"] = pred.config["parameters"]
        summ["model"] = {k: pred.config.get(k) for k in ("model_id", "tokenizer_id", "max_length", "batch_size", "learning_rate",
                                                          "optimizer", "epochs_run", "best_epoch", "seed", "device", "python",
                                                          "torch", "transformers", "trained_at", "synthetic_repeat", "lambdas")}
        summ["test_sizes"] = {"rows": len(test_rows), "intent": int(mi.sum()), "urgency": int(mu.sum()), "escalation": int(me.sum())}
        summ["intent_labels"] = intents
        # confusion matrix intents (counts)
        from sklearn.metrics import confusion_matrix
        cm_i = confusion_matrix(yi[mi], pi[mi], labels=list(range(len(intents))))
        summ["intent_confusion_top_pairs"] = sorted(
            [{"true": intents[i], "pred": intents[j], "count": int(cm_i[i, j])}
             for i in range(len(intents)) for j in range(len(intents)) if i != j and cm_i[i, j] > 0],
            key=lambda d: -d["count"])[:15]
        name = MODEL_DISPLAY[key]
        plot_confusion(cm_i, intents, f"Intent confusion ({name}, test, n={int(mi.sum())})",
                       FIGURES / f"confusion_intent_{key}.png", annotate=False, figsize=(11, 10))
        plot_confusion(summ["overall"]["urgency"]["confusion_matrix"], URGENCY_LABELS,
                       f"Urgency confusion ({name}, test, n={int(mu.sum())})", FIGURES / f"confusion_urgency_{key}.png")
        prec, rec, _ = precision_recall_curve(ye[me], pe[me])
        pr_curves[key] = (rec, prec, summ["overall"]["escalation"]["pr_auc"], float(ye[me].mean()))
        results[key] = summ
        # ---- error rows
        idx = out["index"]
        lens = [len(pred.tokenizer(test_rows[i]["text"])["input_ids"]) for i in idx]
        for n, i in enumerate(idx):
            r = test_rows[i]
            wi = r["intent"] is not None and int(pi[n]) != intents.index(r["intent"])
            wu = r["urgency"] is not None and int(pu[n]) != int(r["urgency"])
            esc_pred = int(pe[n] >= 0.5)
            we = r["escalation"] is not None and esc_pred != int(r["escalation"])
            if not (wi or wu or we):
                continue
            kinds = [k for k, w in (("intent", wi), ("urgency", wu), ("escalation", we)) if w]
            if we:
                kinds.append("missed_escalation" if r["escalation"] == 1 else "false_escalation")
            if wu and r["urgency"] is not None and int(pu[n]) > int(r["urgency"]):
                kinds.append("urgency_overpredicted")
            err_rows.append({
                "model": key, "id": r["id"], "text": r["text"], "language": r["lang"], "source": r["provenance"],
                "true_intent": r["intent"] or "", "pred_intent": intents[int(pi[n])] if r["intent"] is not None else "",
                "true_urgency": URGENCY_LABELS[r["urgency"]] if r["urgency"] is not None else "",
                "pred_urgency": URGENCY_LABELS[int(pu[n])] if r["urgency"] is not None else "",
                "true_escalation": r["escalation"] if r["escalation"] is not None else "",
                "pred_escalation": esc_pred if r["escalation"] is not None else "",
                "escalation_probability": round(float(pe[n]), 4) if r["escalation"] is not None else "",
                "error_types": "|".join(kinds), "n_tokens": lens[n], "n_chars": len(r["text"])})
        print(f"[{key}] test intent F1={summ['overall']['intent']['macro_f1']:.4f} "
              f"urgency F1={summ['overall']['urgency']['macro_f1']:.4f} esc PR-AUC={summ['overall']['escalation']['pr_auc']}")
    # ---- write
    base = majority_baseline(train_rows, test_rows, label_maps)
    for key, summ in results.items():
        (METRICS / METRIC_FILES[key]).write_text(json.dumps(summ, indent=2, ensure_ascii=False))
    if err_rows:
        with open(REPORTS / "error_analysis.csv", "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=list(err_rows[0].keys()))
            w.writeheader()
            w.writerows(err_rows)
    comp = {"models": {}, "majority_baseline": base,
            "note": "All values measured on the held-out test split. Urgency/escalation test sets are tiny (see n); intervals are bootstrap 95%."}
    langs = sorted({l for s in results.values() for l in s["by_language"]})
    comp["language_table"] = {}
    for key, s in results.items():
        o = s["overall"]
        comp["models"][key] = {
            "name": MODEL_DISPLAY[key], "intent_macro_f1": o["intent"]["macro_f1"], "intent_macro_f1_ci95": o["intent"]["macro_f1_ci95"],
            "intent_accuracy": o["intent"]["accuracy"], "urgency_macro_f1": o["urgency"]["macro_f1"],
            "urgency_macro_f1_ci95": o["urgency"]["macro_f1_ci95"], "escalation_f1": o["escalation"]["f1"],
            "escalation_pr_auc": o["escalation"]["pr_auc"], "escalation_pr_auc_ci95": o["escalation"]["pr_auc_ci95"],
            "escalation_roc_auc": o["escalation"]["roc_auc"], "escalation_brier": o["escalation"]["brier"],
            "escalation_ece": o["escalation"]["ece_10bin"],
            "escalation_brier_calibrated": (s["overall"].get("escalation_calibrated") or {}).get("brier"),
            "escalation_ece_calibrated": (s["overall"].get("escalation_calibrated") or {}).get("ece_10bin"),
            "parameters": s["parameters"], "cpu_latency_median_ms": s["latency_cpu"]["median_ms"],
            "cpu_latency_p95_ms": s["latency_cpu"]["p95_ms"]}
        for l in langs:
            b = s["by_language"][l]
            comp["language_table"].setdefault(l, {})[key] = {
                "intent_macro_f1": (b["intent"] or {}).get("macro_f1"), "intent_n": (b["intent"] or {}).get("n", 0),
                "urgency_macro_f1": (b["urgency"] or {}).get("macro_f1"), "urgency_n": (b["urgency"] or {}).get("n", 0),
                "escalation_f1": (b["escalation"] or {}).get("f1"), "escalation_n": (b["escalation"] or {}).get("n", 0)}
    (METRICS / "comparison.json").write_text(json.dumps(comp, indent=2, ensure_ascii=False))
    # ---- PR curve figure
    if pr_curves:
        fig, ax = plt.subplots(figsize=(5.5, 4.5), dpi=110)
        for key, (rec, prec, ap, prev) in pr_curves.items():
            ax.step(rec, prec, where="post", label=f"{MODEL_DISPLAY[key]} (AP={ap:.3f})")
        ax.axhline(prev, ls="--", c="gray", label=f"prevalence={prev:.2f}")
        ax.set_xlabel("Recall")
        ax.set_ylabel("Precision")
        ax.set_title(f"Escalation PR curve (test, n={sum(1 for r in test_rows if r['escalation'] is not None)})")
        ax.legend()
        fig.tight_layout()
        fig.savefig(FIGURES / "escalation_pr_curve.png")
        plt.close(fig)
    print("wrote metrics/figures/error_analysis")


if __name__ == "__main__":
    main()
