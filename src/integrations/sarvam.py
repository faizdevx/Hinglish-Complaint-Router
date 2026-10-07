"""OPTIONAL Sarvam speech-to-text adapter.

STATUS: written against the public REST docs (POST https://api.sarvam.ai/speech-to-text,
header ``api-subscription-key``, multipart ``file``/``model``/``mode``). It has NOT been
tested against the live API in this repository because no API key was available. Tests only
exercise the no-key and mocked paths. The text classifier never depends on this module.
"""
from __future__ import annotations

import json
import os
import uuid
import urllib.error
import urllib.request

STT_URL = "https://api.sarvam.ai/speech-to-text"
MAX_AUDIO_BYTES = 10 * 1024 * 1024


class SarvamUnavailable(RuntimeError):
    """No API key configured (or the service could not be reached)."""


def api_key() -> str | None:
    return os.environ.get("SARVAM_API_KEY") or None


def is_configured() -> bool:
    return api_key() is not None


def _multipart(fields: dict[str, str], file_field: str, filename: str, data: bytes, ctype: str) -> tuple[bytes, str]:
    boundary = uuid.uuid4().hex
    parts = []
    for k, v in fields.items():
        parts.append(f'--{boundary}\r\nContent-Disposition: form-data; name="{k}"\r\n\r\n{v}\r\n'.encode())
    parts.append((f'--{boundary}\r\nContent-Disposition: form-data; name="{file_field}"; filename="{filename}"\r\n'
                  f"Content-Type: {ctype}\r\n\r\n").encode() + data + b"\r\n")
    parts.append(f"--{boundary}--\r\n".encode())
    return b"".join(parts), f"multipart/form-data; boundary={boundary}"


def transcribe(audio: bytes, filename: str = "audio.wav", content_type: str = "audio/wav",
               mode: str = "codemix", timeout: float = 60.0) -> dict:
    """Return {"transcript": str, "language_code": str|None}. Raises SarvamUnavailable without a key."""
    key = api_key()
    if not key:
        raise SarvamUnavailable("SARVAM_API_KEY is not set; voice input is disabled.")
    if len(audio) > MAX_AUDIO_BYTES:
        raise ValueError("Audio file too large (max 10 MB).")
    body, ctype = _multipart({"model": "saaras:v3", "mode": mode, "language_code": "unknown"},
                             "file", filename, audio, content_type)
    req = urllib.request.Request(STT_URL, data=body, method="POST",
                                 headers={"api-subscription-key": key, "Content-Type": ctype})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            payload = json.loads(r.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, json.JSONDecodeError) as e:
        raise SarvamUnavailable(f"Sarvam request failed: {type(e).__name__}") from e
    return {"transcript": payload.get("transcript", ""), "language_code": payload.get("language_code")}
