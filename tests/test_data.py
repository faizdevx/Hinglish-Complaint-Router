import pytest

from src.config import DATA_PROCESSED, URGENCY_LABELS
from src.data import load_label_maps, load_split
from src.dataset import MISSING, ComplaintDataset, build_compact_vocab, make_collate
from src.routing import INTENT_TO_ROUTE, ROUTES, UNMAPPED, route_for
from tests.conftest import LABELS

needs_data = pytest.mark.skipif(not (DATA_PROCESSED / "train.jsonl").exists(), reason="run scripts/prepare_dataset.py first")


@needs_data
def test_splits_have_no_overlap():
    seen = {}
    for s in ("train", "val", "test"):
        for r in load_split(s):
            key = (r["lang"], r["text"])
            assert key not in seen, f"{key} appears in {seen[key]} and {s}"
            seen[key] = s


@needs_data
def test_rows_have_expected_labels_and_provenance():
    for r in load_split("train")[:2000]:
        assert r["provenance"] in {"banking77_english", "banking77_nllb200_translation", "synthetic_llm_generated"}
        if r["urgency"] is not None:
            assert 0 <= r["urgency"] < len(URGENCY_LABELS)
        if r["escalation"] is not None:
            assert r["escalation"] in (0, 1)
        assert r["intent"] is not None or r["urgency"] is not None or r["escalation"] is not None


@needs_data
def test_label_maps_and_routing_coverage():
    lm = load_label_maps()
    assert len(lm["intent"]) == 77 and lm["urgency"] == URGENCY_LABELS
    missing = [i for i in lm["intent"] if route_for(i) == UNMAPPED]
    assert not missing, missing
    assert set(INTENT_TO_ROUTE.values()) <= set(ROUTES)


def test_route_for_unknown():
    assert route_for("not_a_label") == UNMAPPED
    assert route_for("compromised_card") == "fraud_and_disputes"


def test_dataset_missing_labels_and_collate_shapes(tiny_tokenizer):
    rows = [{"text": "mera card hai", "intent": "card_arrival", "urgency": None, "escalation": None},
            {"text": "my refund", "intent": None, "urgency": 3, "escalation": 1}]
    ds = ComplaintDataset(rows, LABELS)
    assert ds[0]["urgency"] == MISSING and ds[1]["intent"] == MISSING
    batch = make_collate(tiny_tokenizer, 16)([ds[0], ds[1]])
    assert batch["input_ids"].shape == batch["attention_mask"].shape and batch["input_ids"].shape[0] == 2
    assert batch["intent"].tolist() == [0, MISSING] and batch["urgency"].tolist() == [MISSING, 3]


def test_compact_vocab_fit_on_given_texts_only(tiny_tokenizer):
    m = build_compact_vocab(tiny_tokenizer, ["mera card"], 16)
    assert m[tiny_tokenizer.pad_token_id] == 0 and m[tiny_tokenizer.unk_token_id] == 1
    assert tiny_tokenizer.convert_tokens_to_ids("paani") not in m  # unseen word not in vocab


def test_tokenizer_handles_indic_text_via_unk(tiny_tokenizer):
    ids = tiny_tokenizer("सुबह")["input_ids"]
    assert ids[0] == tiny_tokenizer.cls_token_id and ids[-1] == tiny_tokenizer.sep_token_id
