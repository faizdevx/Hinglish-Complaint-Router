"""Language / script identification.

* Non-Latin scripts: deterministic Unicode-script rule (script -> language tag).
* Latin script: trained char n-gram logistic regression, English (en-Latn) vs Romanised Hindi
  (hi-Latn). It is trained on Banking77 English (real) vs synthetic_enterprise hi-Latn, so it
  can pick up domain/style artefacts; see MODEL_CARD.md. Without a trained model, Latin text is
  reported as ``und-Latn``.
"""
from __future__ import annotations

from pathlib import Path

from .config import MODELS_DIR
from .preprocessing import SCRIPT_TO_LANG, detect_script

LANGID_PATH = MODELS_DIR / "langid.joblib"
_clf = None
_loaded = False


def _load():
    global _clf, _loaded
    if not _loaded:
        _loaded = True
        if LANGID_PATH.exists():
            import joblib
            _clf = joblib.load(LANGID_PATH)
    return _clf


def detect_language(text: str) -> tuple[str, str]:
    """Return (language tag, script)."""
    script = detect_script(text)
    if script in SCRIPT_TO_LANG:
        return SCRIPT_TO_LANG[script], script
    if script == "Latn":
        clf = _load()
        return (str(clf.predict([text])[0]) if clf is not None else "und-Latn"), script
    if script == "Deva+Latn":
        return "hi-mixed", script
    return "und", script
