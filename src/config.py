"""Central configuration: paths, label vocabularies, thresholds and defaults."""
from __future__ import annotations

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = ROOT / "data" / "raw"
DATA_PROCESSED = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS = ROOT / "reports"
FIGURES = REPORTS / "figures"
METRICS = REPORTS / "metrics"

HF_DATASET = "cmul8-hf/IndicJevBench"
HF_DATASET_FILES = [
    "manifest.json",
    "README.md",
    "v1/fintech_banking77.jsonl",
    "v1/synthetic_enterprise.jsonl",
    "v1/hinglish_lid.jsonl",
    "v1/intent_massive.jsonl",
]
# Subsets used for training/evaluation (intent_massive is inspected but not used; see DATASET.md)
INTENT_SUBSET = "fintech_banking77"
SYNTHETIC_SUBSET = "synthetic_enterprise"

TRANSFORMER_ID = "google/muril-base-cased"
SEED = 42

# Urgency labels in the dataset are the 1..5 scale; index = score - 1
URGENCY_LABELS = ["very_low", "low", "medium", "high", "critical"]

# Display thresholds for the escalation probability. These are UI bands, not
# calibrated decision thresholds, and were not tuned on any data.
ESCALATION_BANDS = [(0.33, "LOW"), (0.66, "MEDIUM"), (1.01, "HIGH")]
# Probability at/above which the API flags `escalate=true`. Default 0.5, untuned.
ESCALATION_DECISION_THRESHOLD = 0.5

MAX_LENGTH_DEFAULT = 64
MAX_INPUT_CHARS = 2000
TASKS = ("intent", "urgency", "escalation")

MODEL_KEYS = ("lstm", "transformer")
MODEL_DISPLAY = {"lstm": "BiLSTM", "transformer": "MuRIL"}


def model_dir(key: str) -> Path:
    return MODELS_DIR / key


def escalation_band(p: float) -> str:
    for upper, name in ESCALATION_BANDS:
        if p < upper:
            return name
    return ESCALATION_BANDS[-1][1]
