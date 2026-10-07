"""Train the Latin-script en-Latn vs hi-Latn classifier on the TRAIN split and evaluate on val/test."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report
from sklearn.pipeline import make_pipeline

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import METRICS, SEED  # noqa: E402
from src.data import load_raw, load_split  # noqa: E402
from src.langid import LANGID_PATH  # noqa: E402
from src.preprocessing import detect_script, normalize_text  # noqa: E402


def latin_rows(split):
    return [(r["text"], r["lang"]) for r in load_split(split) if r["lang"] in ("en-Latn", "hi-Latn")]


def main() -> None:
    Xtr, ytr = zip(*latin_rows("train"))
    pipe = make_pipeline(TfidfVectorizer(analyzer="char_wb", ngram_range=(1, 4), lowercase=True, sublinear_tf=True),
                         LogisticRegression(max_iter=1000, class_weight="balanced", random_state=SEED))
    pipe.fit(Xtr, ytr)
    LANGID_PATH.parent.mkdir(exist_ok=True)
    joblib.dump(pipe, LANGID_PATH)
    report = {"note": "in-domain only: en-Latn=Banking77 English, hi-Latn=synthetic_enterprise. Optimistic.",
              "train_counts": {l: ytr.count(l) for l in set(ytr)}}
    for split in ("val", "test"):
        X, y = zip(*latin_rows(split))
        report[split] = classification_report(y, pipe.predict(X), output_dict=True, zero_division=0)
    # Script rule validation on hinglish_lid (gold derived from script presence => near-tautological)
    h = load_raw("hinglish_lid")
    names = {0: "Deva", 1: "Latn", 2: "Deva+Latn"}
    ok = sum(detect_script(normalize_text(r["state"])) == names[r["expected"]] for r in h)
    report["script_rule_vs_hinglish_lid_gold"] = {"n": len(h), "agree": ok, "accuracy": ok / len(h),
        "caveat": "gold label is itself derived from Unicode script presence, so this checks implementation consistency, not real-world LID quality"}
    lat = [normalize_text(r["state"]) for r in h if names[r["expected"]] == "Latn"]
    pred = pipe.predict(lat)
    report["hinglish_lid_latin_only_predicted_distribution_unlabeled"] = {l: int((pred == l).sum()) for l in set(pred)}
    METRICS.mkdir(parents=True, exist_ok=True)
    (METRICS / "langid_metrics.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2)[:1800])


if __name__ == "__main__":
    main()
