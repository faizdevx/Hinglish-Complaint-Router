import pytest

from src.config import MODELS_DIR
from src.inference import ModelUnavailable, Predictor
from src.preprocessing import InvalidInput


@pytest.mark.parametrize("key", ["lstm", "transformer"])
def test_checkpoint_load_and_predict_contract(tiny_models, key):
    p = Predictor.load(tiny_models / key, key)
    r = p.predict("Mera card block ho gaya hai")
    assert set(["intent", "urgency", "escalation_probability", "language", "route", "inference_ms"]) <= set(r)
    assert 0.0 <= r["escalation_probability"] <= 1.0
    assert r["intent"] in p.labels["intent"] and r["urgency"] in p.labels["urgency"]
    assert abs(sum(s["score"] for s in r["intent_scores"]) - 1) < 1.0 + 1e-6 and r["intent_scores"][0]["label"] == r["intent"]
    assert abs(sum(r["urgency_scores"].values()) - 1) < 1e-5
    assert r["inference_ms"] > 0 and r["model"] == key


def test_batch_matches_single(tiny_models):
    p = Predictor.load(tiny_models / "lstm", "lstm")
    texts = ["my card", "mera refund nahi aaya", "सुबह से पानी नहीं आया"]
    batch = p.predict_batch(texts, batch_size=2)
    for t, b in zip(texts, batch):
        s = p.predict(t)
        assert s["intent"] == b["intent"] and abs(s["escalation_probability"] - b["escalation_probability"]) < 1e-5


def test_invalid_input_raises(tiny_models):
    p = Predictor.load(tiny_models / "lstm", "lstm")
    with pytest.raises(InvalidInput):
        p.predict("   ")


def test_missing_checkpoint_raises(tmp_path):
    with pytest.raises(ModelUnavailable):
        Predictor.load(tmp_path / "nope", "transformer")


def test_temperature_changes_probability(tiny_models, tmp_path):
    import shutil, json
    d = tmp_path / "lstm"; shutil.copytree(tiny_models / "lstm", d)
    p0 = Predictor.load(d, "lstm").predict("mera card")["escalation_probability"]
    (d / "calibration.json").write_text(json.dumps({"temperature": 100.0}))
    r = Predictor.load(d, "lstm").predict("mera card")
    assert abs(r["escalation_probability"] - 0.5) < abs(p0 - 0.5) + 1e-9 and r["calibrated"]


@pytest.mark.skipif(not (MODELS_DIR / "lstm" / "best.pt").exists(), reason="trained BiLSTM checkpoint not present")
def test_real_trained_lstm_predicts():
    p = Predictor.load(MODELS_DIR / "lstm", "lstm")
    r = p.predict("My card has not arrived yet")
    assert 0 <= r["escalation_probability"] <= 1 and r["route"] != "unmapped"
