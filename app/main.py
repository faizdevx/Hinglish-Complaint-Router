"""FastAPI app: JSON API + server-rendered dashboard (Jinja2, vanilla JS)."""
from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, Form, Query, Request, UploadFile
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from src.batch import BatchError
from src.config import (ESCALATION_BANDS, ESCALATION_DECISION_THRESHOLD, FIGURES, MODEL_DISPLAY, MODEL_KEYS, ROOT)
from src.inference import ModelUnavailable
from src.integrations import sarvam
from src.preprocessing import InvalidInput

from .schemas import CompareRequest, PredictRequest, PredictResponse
from .services import batch_service, health_service
from .services.metrics_service import MetricsService
from .services.prediction_service import PredictionService

log = logging.getLogger("app")
APP_DIR = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=str(APP_DIR / "templates"))
UNAVAILABLE_PREDICTION = "Prediction service is unavailable."


@asynccontextmanager
async def lifespan(app: FastAPI):
    app.state.predictions = PredictionService()
    app.state.predictions.load()  # models are loaded once here
    app.state.metrics = MetricsService()
    yield


app = FastAPI(title="Hinglish Complaint Router", version="1.0.0", lifespan=lifespan,
              description="Multilingual multi-task message router (intent / urgency / escalation / route). "
                          "Research & portfolio project; not for autonomous high-stakes decisions.")
app.mount("/static", StaticFiles(directory=str(APP_DIR / "static")), name="static")
FIGURES.mkdir(parents=True, exist_ok=True)
app.mount("/figures", StaticFiles(directory=str(FIGURES)), name="figures")


def preds(request: Request) -> PredictionService:
    return request.app.state.predictions


def mets(request: Request) -> MetricsService:
    return request.app.state.metrics


def page(request: Request, name: str, active: str, title: str, **ctx):
    p, m = preds(request), mets(request)
    status = p.status()
    loaded = sum(1 for s in status.values() if s["status"] == "ready")
    base = {"active": active, "title": title, "models_status": status,
            "sidebar": {"online": True, "models_loaded": loaded, "models_total": len(status),
                        "dataset": health_service.artifact_status()["dataset_metadata"]},
            "bands": ESCALATION_BANDS, "threshold": ESCALATION_DECISION_THRESHOLD, "model_display": MODEL_DISPLAY,
            "model_keys": MODEL_KEYS, "default_model": p.default_model()}
    try:
        return templates.TemplateResponse(request, name, {**base, **ctx})
    except Exception:  # never serve a blank page
        log.exception("template render failed: %s", name)
        return templates.TemplateResponse(request, "error.html", {**base, "message": "This page failed to render. See server logs."},
                                          status_code=500)


# ------------------------------------------------------------------ error handling (JSON routes)
@app.exception_handler(RequestValidationError)
async def validation_handler(request: Request, exc: RequestValidationError):
    msgs = []
    for e in exc.errors():
        m = str(e.get("msg", "Invalid input")).replace("Value error, ", "")
        msgs.append(m)
    return JSONResponse(status_code=422, content={"detail": "; ".join(dict.fromkeys(msgs)) or "Invalid input."})


@app.exception_handler(InvalidInput)
async def invalid_input_handler(request: Request, exc: InvalidInput):
    return JSONResponse(status_code=422, content={"detail": str(exc)})


@app.exception_handler(ModelUnavailable)
async def unavailable_handler(request: Request, exc: ModelUnavailable):
    return JSONResponse(status_code=503, content={"detail": str(exc)})


@app.exception_handler(BatchError)
async def batch_handler(request: Request, exc: BatchError):
    return JSONResponse(status_code=400, content={"detail": str(exc)})


@app.exception_handler(Exception)
async def unhandled(request: Request, exc: Exception):
    log.exception("unhandled error on %s", request.url.path)
    return JSONResponse(status_code=500, content={"detail": UNAVAILABLE_PREDICTION})


# ------------------------------------------------------------------ pages
@app.get("/", include_in_schema=False)
def dashboard(request: Request):
    m = mets(request)
    comp = m.comparison()
    return page(request, "dashboard.html", "dashboard", "Dashboard", dataset=m.dataset_summary(), comparison=comp,
                system=health_service.system_info(), health=health_service.health(preds(request).status()))


@app.get("/classify", include_in_schema=False)
def classify_page(request: Request):
    m = mets(request)
    return page(request, "classify.html", "classify", "Classify", dataset_examples=m.dataset_examples(),
                demo_examples=m.demo_examples())


@app.get("/playground", include_in_schema=False)
def playground_page(request: Request):
    m = mets(request)
    return page(request, "playground.html", "playground", "Playground", dataset_examples=m.dataset_examples(1),
                demo_examples=m.demo_examples())


@app.get("/batch", include_in_schema=False)
def batch_page(request: Request):
    return page(request, "batch.html", "batch", "Batch")


@app.get("/models", include_in_schema=False)
def models_page(request: Request):
    m = mets(request)
    return page(request, "models.html", "models", "Models", rows=m.model_comparison_rows(), comparison=m.comparison(), charts=m.charts(),
                model_metrics={k: m.model_metrics(k) for k in MODEL_KEYS})


@app.get("/evaluation", include_in_schema=False)
def evaluation_page(request: Request):
    m = mets(request)
    comp = m.comparison()
    return page(request, "evaluation.html", "evaluation", "Evaluation", comparison=comp,
                rows=m.model_comparison_rows(), model_metrics={k: m.model_metrics(k) for k in MODEL_KEYS},
                audit=m.audit(), langid=m.langid(), figures=m.figures(), charts=m.charts())


@app.get("/errors", include_in_schema=False)
def errors_page(request: Request, model: str | None = None):
    m = mets(request)
    if model not in MODEL_KEYS:  # default: first model that actually has saved metrics
        model = next((k for k in ("transformer", "lstm") if m.model_metrics(k)), "transformer")
    mm = m.model_metrics(model)
    return page(request, "errors.html", "errors", "Errors", sel_model=model, metrics=mm, figures=m.figures(),
                errors_available=m.errors_available(), facets=m.query_errors(limit=1)["facets"])


@app.get("/health-ui", include_in_schema=False)
def health_page(request: Request):
    p = preds(request)
    status = p.status()
    return page(request, "health.html", "health", "Health", health=health_service.health(status),
                system=health_service.system_info(), sarvam_configured=sarvam.is_configured())


@app.get("/demo", include_in_schema=False)
def demo_page(request: Request):
    p, m = preds(request), mets(request)
    examples = m.dataset_examples(2)[:8]
    for ex in examples:
        ex["predictions"] = {}
        for key in MODEL_KEYS:
            try:
                r = p.predict(ex["text"], key, top_k=1)
                ex["predictions"][key] = r
            except (ModelUnavailable, InvalidInput):
                ex["predictions"][key] = None
    return page(request, "demo.html", "demo", "Demo", examples=examples)


@app.get("/about", include_in_schema=False)
def about_page(request: Request):
    m = mets(request)
    return page(request, "about.html", "about", "About", dataset=m.dataset_summary(), audit=m.audit())


# ------------------------------------------------------------------ JSON API
@app.get("/health")
def health(request: Request):
    return health_service.health(preds(request).status())


@app.get("/api/health")
def api_health(request: Request):
    return {**health_service.health(preds(request).status()), "system": health_service.system_info(),
            "sarvam_configured": sarvam.is_configured()}


@app.get("/api/models")
def api_models(request: Request):
    return {"models": preds(request).status(), "default": preds(request).default_model(),
            "escalation_bands": ESCALATION_BANDS, "escalation_decision_threshold": ESCALATION_DECISION_THRESHOLD}


@app.get("/api/metrics")
def api_metrics(request: Request):
    return mets(request).all_metrics()


@app.get("/api/errors")
def api_errors(request: Request, model: str | None = None, q: str | None = None, language: str | None = None,
               intent: str | None = None, limit: int = Query(100, ge=1, le=500), offset: int = Query(0, ge=0)):
    m = mets(request)
    if not m.errors_available():
        return {"available": False, "message": "No error analysis has been generated yet. Run: python scripts/evaluate.py",
                "total": 0, "rows": []}
    return {"available": True, **m.query_errors(model, q, language, intent, limit, offset)}


def _predict(request: Request, req: PredictRequest) -> dict:
    return preds(request).predict(req.text, req.model, req.language)


@app.post("/predict", response_model=PredictResponse)
def predict(request: Request, req: PredictRequest):
    r = _predict(request, req)
    return {"intent": r["intent"], "urgency": r["urgency"], "escalation_probability": r["escalation_probability"],
            "language": r["language"], "route": r["route"], "model": r["model"], "inference_ms": r["inference_ms"]}


@app.post("/api/predict")
def api_predict(request: Request, req: PredictRequest):
    r = _predict(request, req)
    r["escalation_threshold"] = ESCALATION_DECISION_THRESHOLD
    return r


@app.post("/api/compare")
def api_compare(request: Request, req: CompareRequest):
    return {"text": req.text, "results": preds(request).compare(req.text, req.language)}


@app.post("/api/batch")
async def api_batch(request: Request, file: UploadFile = File(...), model: str | None = Form(None)):
    if model not in (None, "", *MODEL_KEYS):
        return JSONResponse(status_code=422, content={"detail": f"Unknown model '{model}'."})
    raw = await file.read(batch_service.MAX_BYTES + 1)
    return batch_service.process_upload(preds(request), raw, model or None)


@app.post("/voice/predict")
async def voice_predict(request: Request, file: UploadFile = File(...), model: str | None = Form(None)):
    """Optional: Sarvam speech-to-text -> classifier. Disabled without SARVAM_API_KEY. Untested against the live API."""
    if not sarvam.is_configured():
        return JSONResponse(status_code=501, content={"detail": "Voice input is disabled: SARVAM_API_KEY is not set."})
    audio = await file.read(sarvam.MAX_AUDIO_BYTES + 1)
    try:
        stt = sarvam.transcribe(audio, file.filename or "audio.wav", file.content_type or "audio/wav")
    except sarvam.SarvamUnavailable as e:
        return JSONResponse(status_code=502, content={"detail": str(e)})
    except ValueError as e:
        return JSONResponse(status_code=422, content={"detail": str(e)})
    result = preds(request).predict(stt["transcript"], model or None)
    return {"transcript": stt["transcript"], "stt_language": stt["language_code"], "result": result}
