import io

PAGES = {"/": "Dashboard", "/classify": "Classify message", "/playground": "Playground", "/batch": "Batch testing",
         "/models": "Models", "/evaluation": "Evaluation", "/errors": "Error analysis", "/health-ui": "System health",
         "/about": "About", "/demo": "Demo"}


def test_health_and_root_json(client):
    r = client.get("/health")
    assert r.status_code == 200 and r.json()["status"] == "ok" and sorted(r.json()["models_loaded"]) == ["lstm", "transformer"]


def test_ui_pages_render_html_css_js(client):
    for path, marker in PAGES.items():
        r = client.get(path)
        assert r.status_code == 200, path
        assert "text/html" in r.headers["content-type"] and marker in r.text, path
        assert "/static/css/app.css" in r.text and "/static/js/app.js" in r.text
    assert client.get("/static/css/app.css").status_code == 200
    assert client.get("/static/js/app.js").status_code == 200
    assert client.get("/docs").status_code == 200


def test_sidebar_status_is_dynamic_all_loaded(client):
    assert "2 Loaded" in client.get("/classify").text


def test_sidebar_status_is_dynamic_one_missing(client_no_transformer):
    r = client_no_transformer.get("/classify")
    assert r.status_code == 200 and "1 Loaded" in r.text and "unavailable" in r.text.lower()


def test_predict_valid(client):
    r = client.post("/predict", json={"text": "bhai road ki light 3 din se band hai"})
    assert r.status_code == 200
    j = r.json()
    assert {"intent", "urgency", "escalation_probability", "language", "route"} <= set(j)
    assert 0.0 <= j["escalation_probability"] <= 1.0 and j["language"] in {"hi-Latn", "en-Latn", "und-Latn"}


def test_predict_validation(client):
    for body in ({"text": ""}, {"text": "   "}, {}, {"text": "x", "model": "gpt"}):
        r = client.post("/predict", json=body)
        assert r.status_code == 422 and isinstance(r.json()["detail"], str)
    assert client.post("/predict", json={"text": "!!!"}).status_code == 422


def test_predict_missing_model_is_503_with_message(client_no_transformer):
    r = client_no_transformer.post("/predict", json={"text": "my card", "model": "transformer"})
    assert r.status_code == 503 and "not available" in r.json()["detail"]
    assert client_no_transformer.post("/predict", json={"text": "my card"}).status_code == 200  # falls back to lstm


def test_api_predict_full_payload_and_language_override(client):
    r = client.post("/api/predict", json={"text": "mera card", "model": "lstm", "language": "en-Latn"})
    j = r.json()
    assert r.status_code == 200 and j["language"] == "en-Latn" and "user-selected" in j["language_source"]
    assert j["escalation_band"] in {"LOW", "MEDIUM", "HIGH"} and "checkpoint" in j and j["device"]


def test_api_models_metrics_health(client):
    m = client.get("/api/models").json()
    assert set(m["models"]) == {"lstm", "transformer"} and m["models"]["lstm"]["status"] == "ready"
    assert m["models"]["lstm"]["parameters"] > 0
    r = client.get("/api/metrics")
    assert r.status_code == 200 and set(r.json()) >= {"comparison", "models"}
    h = client.get("/api/health").json()
    assert "system" in h and "torch" in h["system"] and "SARVAM" not in str(h).upper().replace("SARVAM_CONFIGURED", "")


def test_compare_both(client):
    j = client.post("/api/compare", json={"text": "Mera refund nahi aaya"}).json()
    assert set(j["results"]) == {"lstm", "transformer"} and all(v["ok"] for v in j["results"].values())
    assert client.post("/api/compare", json={"text": " "}).status_code == 422


def test_compare_one_missing(client_no_transformer):
    j = client_no_transformer.post("/api/compare", json={"text": "Mera refund nahi aaya"}).json()
    assert j["results"]["lstm"]["ok"] and not j["results"]["transformer"]["ok"]


def _csv(content: bytes, name="x.csv"):
    return {"file": (name, io.BytesIO(content), "text/csv")}


def test_batch_ok_and_partial_failure(client):
    data = "id,text\na1,mera card block ho gaya\na2,\na3,my refund has not arrived\n".encode()
    r = client.post("/api/batch", files=_csv(data), data={"model": "lstm"})
    j = r.json()
    assert r.status_code == 200 and j["n_inputs"] == 3 and j["n_success"] == 2 and j["n_failed"] == 1
    assert j["csv"].splitlines()[0] == "id,text,language,intent,urgency,escalation_probability,route" and "a1" in j["csv"]
    assert all(0 <= x["escalation_probability"] <= 1 for x in j["results"])


def test_batch_malformed(client):
    assert client.post("/api/batch", files=_csv(b"foo,bar\n1,2\n")).status_code == 400
    assert client.post("/api/batch", files=_csv(b"")).status_code == 400
    assert client.post("/api/batch", files=_csv(b"\xff\xfe\x00bad")).status_code == 400
    assert client.post("/api/batch", files=_csv(b"text\n")).status_code == 400
    assert client.post("/api/batch", files=_csv(b"text\nhi\n"), data={"model": "bogus"}).status_code == 422


def test_batch_model_unavailable(client_no_transformer):
    r = client_no_transformer.post("/api/batch", files=_csv(b"text\nhello\n"), data={"model": "transformer"})
    assert r.status_code == 503


def test_errors_api_shape(client):
    j = client.get("/api/errors?limit=5").json()
    assert "total" in j and "rows" in j and ("available" in j)


def test_voice_disabled_without_key(client, monkeypatch):
    monkeypatch.delenv("SARVAM_API_KEY", raising=False)
    r = client.post("/voice/predict", files={"file": ("a.wav", io.BytesIO(b"RIFF"), "audio/wav")})
    assert r.status_code == 501 and "SARVAM_API_KEY" in r.json()["detail"]
