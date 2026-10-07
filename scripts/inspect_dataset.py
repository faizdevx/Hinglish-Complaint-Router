"""Inspect raw + processed data and audit for leakage. Writes reports/metrics/data_audit.json."""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.metrics import average_precision_score, f1_score

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, METRICS, TRANSFORMER_ID  # noqa: E402
from src.data import load_raw, load_split  # noqa: E402
from src.preprocessing import detect_script  # noqa: E402

RAW = ["fintech_banking77", "synthetic_enterprise", "hinglish_lid", "intent_massive"]


def raw_overview():
    out = {}
    for s in RAW:
        rows = load_raw(s)
        keys = collections.Counter()
        for r in rows:
            for k in ("id", "family", "lang", "state", "question", "expected"):
                if r.get(k) in (None, ""):
                    keys[k] += 1
        out[s] = {"n": len(rows), "languages": dict(collections.Counter(r["lang"] for r in rows)),
                  "families": dict(collections.Counter(r["family"] for r in rows)),
                  "origins": dict(collections.Counter(r["provenance"]["origin"] for r in rows)),
                  "provenance_notes": sorted({r["provenance"]["notes"] for r in rows}),
                  "licenses": sorted({r["license"] for r in rows}), "splits_field": dict(collections.Counter(r["split"] for r in rows)),
                  "missing_fields": dict(keys), "duplicate_ids": len(rows) - len({r["id"] for r in rows}),
                  "duplicate_state_texts": len(rows) - len({r["state"] for r in rows})}
    return out


def near_dups(train, test, thr=0.85):
    """Fraction of test texts whose most similar train text (char 3-5gram TF-IDF cosine) is >= thr."""
    if not test or not train:
        return None
    v = TfidfVectorizer(analyzer="char_wb", ngram_range=(3, 5), sublinear_tf=True).fit([r["text"] for r in train + test])
    A, B = v.transform([r["text"] for r in test]), v.transform([r["text"] for r in train])
    sims = (A @ B.T).toarray()
    best = sims.max(1)
    arg = sims.argmax(1)
    hit = best >= thr
    agree = None
    return {"n_test": len(test), "n_near_dup": int(hit.sum()), "fraction": float(hit.mean()),
            "threshold": thr, "median_max_similarity": float(np.median(best)), "max_sim_p90": float(np.percentile(best, 90)),
            "_pairs": [(int(i), int(arg[i])) for i in np.where(hit)[0]]}


def main() -> None:
    METRICS.mkdir(parents=True, exist_ok=True)
    splits = {s: load_split(s) for s in ("train", "val", "test")}
    audit = {"raw": raw_overview(), "tokenizer": TRANSFORMER_ID}
    # --- splits
    audit["splits"] = {}
    for s, rows in splits.items():
        audit["splits"][s] = {
            "rows": len(rows), "by_language": dict(collections.Counter(r["lang"] for r in rows)),
            "intent_rows": sum(r["intent"] is not None for r in rows),
            "urgency_dist": dict(collections.Counter(r["urgency"] for r in rows if r["urgency"] is not None)),
            "escalation_dist": dict(collections.Counter(r["escalation"] for r in rows if r["escalation"] is not None))}
    # --- imbalance
    ic = collections.Counter(r["intent"] for r in splits["train"] if r["intent"])
    uc = collections.Counter(r["urgency"] for r in splits["train"] if r["urgency"] is not None)
    audit["imbalance"] = {"intent_train_classes": len(ic), "intent_min": min(ic.values()), "intent_max": max(ic.values()),
                          "intent_max_over_min": max(ic.values()) / min(ic.values()),
                          "urgency_train": {int(k): v for k, v in sorted(uc.items())},
                          "escalation_train": dict(collections.Counter(r["escalation"] for r in splits["train"] if r["escalation"] is not None)),
                          "decision": f"No class weighting / resampling applied: largest intent class is {max(ic.values()) / min(ic.values()):.1f}x the smallest "
                                      "(mild); urgency class 'very_low' has only 1 example in the whole synthetic set so it cannot be rebalanced meaningfully."}
    # --- exact dup / id overlap across splits
    texts = {s: {(r["lang"], r["text"]) for r in rows} for s, rows in splits.items()}
    ids = {s: {r["id"] for r in rows} for s, rows in splits.items()}
    audit["leakage"] = {"exact_text_overlap": {f"{a}-{b}": len(texts[a] & texts[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))},
                        "id_overlap": {f"{a}-{b}": len(ids[a] & ids[b]) for a, b in (("train", "val"), ("train", "test"), ("val", "test"))}}
    # --- near duplicates test->train per source
    nd = {}
    for src_name, flt in (("banking77", lambda r: r["intent"] is not None), ("synthetic_enterprise", lambda r: r["intent"] is None)):
        tr = [r for r in splits["train"] if flt(r)]
        te = [r for r in splits["test"] if flt(r)]
        res = near_dups(tr, te)
        if res and src_name == "banking77":
            pairs = res.pop("_pairs")
            same = sum(te[i]["intent"] == tr[j]["intent"] for i, j in pairs)
            res["near_dup_pairs_with_same_label"] = same
        elif res:
            pairs = res.pop("_pairs")
            same = [(te[i], tr[j]) for i, j in pairs]
            res["escalation_label_agreement"] = (sum(a["escalation"] == b["escalation"] for a, b in same
                                                     if a["escalation"] is not None and b["escalation"] is not None),
                                                 sum(a["escalation"] is not None and b["escalation"] is not None for a, b in same))
            res["urgency_label_agreement"] = (sum(a["urgency"] == b["urgency"] for a, b in same
                                                  if a["urgency"] is not None and b["urgency"] is not None),
                                              sum(a["urgency"] is not None and b["urgency"] is not None for a, b in same))
        nd[src_name] = res
    audit["leakage"]["near_duplicates_test_vs_train"] = nd
    audit["leakage"]["translated_copies"] = ("Each Banking77 item appears in exactly one language (no id appears in two languages), "
                                             "so per-item translated copies across splits are not identifiable from this data; "
                                             "translations of the same original query MAY exist across languages but cannot be linked.")
    audit["leakage"]["metadata_in_features"] = "Model input is only the normalised `text`; ids, language tags, provenance never enter the model."
    # --- sanity baseline: bag-of-words LR for escalation (5-fold CV over all labelled synthetic texts)
    syn = [r for s in splits.values() for r in s if r["escalation"] is not None]
    X = [r["text"] for r in syn]
    y = np.array([r["escalation"] for r in syn])
    vec = TfidfVectorizer(analyzer="char_wb", ngram_range=(2, 4), sublinear_tf=True)
    from sklearn.pipeline import make_pipeline
    p = cross_val_predict(make_pipeline(vec, LogisticRegression(max_iter=1000)), X, y,
                          cv=StratifiedKFold(5, shuffle=True, random_state=0), method="predict_proba")[:, 1]
    audit["sanity_baseline_escalation_char_tfidf_lr_5fold_cv"] = {
        "n": len(y), "prevalence": float(y.mean()), "pr_auc": float(average_precision_score(y, p)),
        "f1": float(f1_score(y, p >= .5)),
        "note": "Non-neural baseline; shows how learnable the synthetic escalation label is from surface text. "
                "CV folds are not grouped by near-duplicate templates, so this is optimistic."}
    # --- lengths, scripts, code-mixing
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(TRANSFORMER_ID)
    allrows = [r for s in splits.values() for r in s]
    ntok = np.array([len(tok(r["text"])["input_ids"]) for r in allrows])
    nch = np.array([len(r["text"]) for r in allrows])
    audit["lengths"] = {"chars_mean": float(nch.mean()), "chars_median": float(np.median(nch)),
                        "tokens_mean": float(ntok.mean()), "tokens_percentiles": {str(q): float(np.percentile(ntok, q)) for q in (50, 90, 95, 99, 100)},
                        "tokens_over_64": int((ntok > 64).sum()), "chosen_max_length": 64}
    sc = collections.Counter((r["lang"], detect_script(r["text"])) for r in allrows)
    audit["script_by_language"] = {f"{l}|{s}": n for (l, s), n in sorted(sc.items())}
    audit["code_mixed_script_rows"] = {"Deva+Latn": sum(detect_script(r["text"]) == "Deva+Latn" for r in allrows)}
    audit["not_measurable"] = ["Intent labels for Romanised Hindi / Hinglish: no such labelled data in the selected subsets",
                               "Intent+urgency+escalation on the same text: no text carries all labels"]
    (METRICS / "data_audit.json").write_text(json.dumps(audit, indent=2, ensure_ascii=False))
    print(json.dumps({k: audit[k] for k in ("leakage", "imbalance", "sanity_baseline_escalation_char_tfidf_lr_5fold_cv", "lengths")}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
