import pytest

from src.batch import BatchError, parse_csv, run_batch, to_csv
from src.integrations import sarvam
from src.inference import Predictor


def test_sarvam_requires_key(monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    assert not sarvam.is_configured()
    with pytest.raises(sarvam.SarvamUnavailable):
        sarvam.transcribe(b"x")


def test_sarvam_mocked_success(monkeypatch):
    monkeypatch.setenv("SARVAM_API_KEY", "test-key")
    captured = {}

    class R:
        def __enter__(self): return self
        def __exit__(self, *a): return False
        def read(self): return b'{"transcript": "mera card", "language_code": "hi-IN"}'

    def fake_open(req, timeout=None):
        captured["headers"] = dict(req.header_items())
        return R()

    monkeypatch.setattr(sarvam.urllib.request, "urlopen", fake_open)
    out = sarvam.transcribe(b"audio")
    assert out == {"transcript": "mera card", "language_code": "hi-IN"}
    assert captured["headers"]["Api-subscription-key"] == "test-key"


def test_parse_csv_variants():
    rows = parse_csv("﻿Text,ID\nhello,7\nworld,\n".encode("utf-8"))
    assert rows[0] == {"id": "7", "text": "hello"} and rows[1]["id"] == "2"
    with pytest.raises(BatchError):
        parse_csv(b"a,b\n1,2")


def test_run_batch_and_csv_roundtrip(tiny_models):
    p = Predictor.load(tiny_models / "lstm", "lstm")
    res = run_batch(p, parse_csv(b"id,text\n1,mera card\n2,   \n3,my refund\n"))
    assert (res["n_inputs"], res["n_success"], res["n_failed"]) == (3, 2, 1)
    out = to_csv(res["results"]).splitlines()
    assert out[0] == "id,text,language,intent,urgency,escalation_probability,route" and len(out) == 3
