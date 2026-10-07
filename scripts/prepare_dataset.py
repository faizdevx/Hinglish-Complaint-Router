"""Build processed train/val/test splits from the raw IndicJevBench files.

* Intent  <- fintech_banking77 (Banking77 + NLLB-200 translations). One row per item.
* Urgency / escalation <- synthetic_enterprise (LLM-generated, Hindi + Hinglish only).
  Rows are merged per unique text so splitting is grouped by text.
* No official splits exist (everything is tagged "v1"), so splits are created here with
  a fixed seed: 70/15/15, stratified, with duplicate texts kept in the same split.
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

from sklearn.model_selection import train_test_split

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_PROCESSED, SEED, URGENCY_LABELS  # noqa: E402
from src.data import item_label, load_raw, write_jsonl  # noqa: E402
from src.preprocessing import normalize_text  # noqa: E402


def split_groups(keys: list[str], strata: list[str], seed: int):
    """70/15/15 over groups."""
    idx = list(range(len(keys)))
    tr, rest = train_test_split(idx, test_size=0.30, random_state=seed, stratify=strata)
    rest_strata = [strata[i] for i in rest]
    va, te = train_test_split(rest, test_size=0.5, random_state=seed, stratify=rest_strata)
    return set(tr), set(va), set(te)


def main() -> None:
    DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
    # ---- intent rows (banking77) ----
    groups: dict[tuple, dict] = {}
    skipped = 0
    for r in load_raw("fintech_banking77"):
        try:
            text = normalize_text(r["state"])
        except ValueError:
            skipped += 1
            continue
        key = (r["lang"], text)
        label = item_label(r)
        g = groups.setdefault(key, {"id": r["id"], "text": text, "lang": r["lang"], "source": r["source"],
                                    "intent": label, "n": 0})
        g["n"] += 1
    intent_rows = list(groups.values())
    conflicts = 0  # same text, different label
    intents = sorted({g["intent"] for g in intent_rows})
    keys = [f"{g['lang']}|{g['text']}" for g in intent_rows]
    tr, va, te = split_groups(keys, [g["intent"] for g in intent_rows], SEED)

    out = collections.defaultdict(list)
    for i, g in enumerate(intent_rows):
        split = "train" if i in tr else "val" if i in va else "test"
        out[split].append({
            "id": g["id"], "text": g["text"], "lang": g["lang"], "source": "banking77",
            "provenance": "banking77_english" if g["lang"] == "en-Latn" else "banking77_nllb200_translation",
            "intent": g["intent"], "urgency": None, "escalation": None, "route_gold": None,
        })

    # ---- urgency / escalation / routing (synthetic_enterprise) ----
    by_text: dict[str, dict] = {}
    for r in load_raw("synthetic_enterprise"):
        text = normalize_text(r["state"])
        d = by_text.setdefault(text, {"id": r["id"].rsplit("-q", 1)[0], "text": text, "lang": r["lang"],
                                      "urgency": None, "escalation": None, "route_gold": None})
        fam = r["family"]
        if fam == "urgency":
            d["urgency"] = int(r["expected"])
        elif fam == "escalation":
            d["escalation"] = int(r["expected"])
        elif fam == "routing":
            d["route_gold"] = item_label(r)
    syn = list(by_text.values())
    skeys = [s["text"] for s in syn]
    s_tr, s_va, s_te = split_groups(skeys, [s["lang"] for s in syn], SEED)
    for i, s in enumerate(syn):
        split = "train" if i in s_tr else "val" if i in s_va else "test"
        out[split].append({**s, "source": "synthetic_enterprise", "provenance": "synthetic_llm_generated",
                           "intent": None})

    for split in ("train", "val", "test"):
        write_jsonl(out[split], DATA_PROCESSED / f"{split}.jsonl")
    (DATA_PROCESSED / "label_maps.json").write_text(json.dumps(
        {"intent": intents, "urgency": URGENCY_LABELS, "escalation": ["no", "yes"]}, indent=2, ensure_ascii=False))
    summary = {
        "seed": SEED, "skipped_malformed": skipped, "intent_label_conflicts": conflicts,
        "banking_unique_texts": len(intent_rows), "synthetic_unique_texts": len(syn),
        "splits": {s: {"rows": len(out[s]),
                       "intent_rows": sum(r["intent"] is not None for r in out[s]),
                       "urgency_rows": sum(r["urgency"] is not None for r in out[s]),
                       "escalation_rows": sum(r["escalation"] is not None for r in out[s])}
                   for s in out},
    }
    (DATA_PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
