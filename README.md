# Hinglish Complaint Router

A multilingual, multi-task NLP classifier that reads a short user message in **English, Hindi in Devanagari, Romanised Hindi / Hinglish, and five other Indian languages** and predicts:

- **Intent**
- **Urgency**
- **Escalation probability**
- **Route**

The project also includes a **FastAPI service** and an **internal-style web dashboard** to inspect and compare two trained models:

- **BiLSTM baseline**
- **Fine-tuned MuRIL encoder**

> **Project type:** multilingual message-intent routing study / research & portfolio prototype

---

## Tech Stack

<div align="center">

<table>
<tr>
<td align="center" width="180">

**Language**

<img src="https://img.shields.io/badge/Python-3.13-3776AB?style=for-the-badge&logo=python&logoColor=white" alt="Python">

</td>
<td align="center" width="180">

**Deep Learning**

<img src="https://img.shields.io/badge/PyTorch-2.14.1-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white" alt="PyTorch">

</td>
<td align="center" width="180">

**NLP Model**

<img src="https://img.shields.io/badge/MuRIL-Google-4285F4?style=for-the-badge&logo=google&logoColor=white" alt="MuRIL">

</td>
<td align="center" width="180">

**API**

<img src="https://img.shields.io/badge/FastAPI-API-009688?style=for-the-badge&logo=fastapi&logoColor=white" alt="FastAPI">

</td>
</tr>
<tr>
<td align="center">

**Server**

<img src="https://img.shields.io/badge/Uvicorn-ASGI-499848?style=for-the-badge" alt="Uvicorn">

</td>
<td align="center">

**NLP**

<img src="https://img.shields.io/badge/WordPiece-Tokenizer-6C3483?style=for-the-badge" alt="WordPiece">

</td>
<td align="center">

**Experimentation**

<img src="https://img.shields.io/badge/Jupyter-Notebook-F37626?style=for-the-badge&logo=jupyter&logoColor=white" alt="Jupyter">

</td>
<td align="center">

**Testing**

<img src="https://img.shields.io/badge/pytest-Tests-0A9EDC?style=for-the-badge&logo=pytest&logoColor=white" alt="pytest">

</td>
</tr>
<tr>
<td align="center">

**Model Hub**

<img src="https://img.shields.io/badge/Hugging%20Face-Checkpoints-FFD21E?style=for-the-badge&logo=huggingface&logoColor=black" alt="Hugging Face">

</td>
<td align="center">

**Frontend**

<img src="https://img.shields.io/badge/HTML%20%2F%20CSS%20%2F%20JS-Web%20UI-E34F26?style=for-the-badge" alt="HTML CSS JS">

</td>
<td align="center">

**Optional Speech**

<img src="https://img.shields.io/badge/Sarvam-Speech%20API-111111?style=for-the-badge" alt="Sarvam">

</td>
<td align="center">

**Runtime**

<img src="https://img.shields.io/badge/Linux-CPU%20Only-FCC624?style=for-the-badge&logo=linux&logoColor=black" alt="Linux">

</td>
</tr>
</table>

</div>

---

## Read This First: What the Data Can and Cannot Support

There is **no dataset of real labelled customer complaints** here.

- Intent labels come from **Banking77** (banking support queries; English original, the other six languages machine-translated with **NLLB-200**).
- Urgency/escalation labels come from a **synthetic, LLM-generated Hindi/Hinglish set (222 texts)**.
- The project is therefore a **multilingual message-intent routing study**, not a model trained on real complaints.
- **No labelled Hinglish intent data exists in the chosen subsets**, so Hinglish intent accuracy is not measured.
- Urgency/escalation test sets are tiny (**27 / 26 texts**). Treat those numbers as anecdotal.
- The MuRIL model is **under-trained (4 epochs on CPU, loss still falling)** and loses badly to the BiLSTM here. That is reported as measured.

---

## Problem

Support-style messages arrive in many languages and scripts.

A router must decide:

1. What the message is about
2. How urgent it is
3. Whether a human should take over
4. Which queue should get it

### Example flow

```text
User message
     │
     ▼
Normalize + tokenize
     │
     ▼
BiLSTM / MuRIL
     │
     ├──────────────┬───────────────┬─────────────────┐
     ▼              ▼               ▼
  Intent         Urgency       Escalation
77 classes       5 levels       probability
     │
     ▼
Route = f(intent)
     │
     ▼
Destination queue
```

---

## Why Hinglish?

Romanised Hindi is Hindi written in Latin characters, often mixed with English words:

> `Transaction failed hui par service charge kat li`

Spelling is unstandardised, the same word has many romanisations, and script-based tools (Unicode ranges, Devanagari-trained tokenizers) behave differently from either English or Devanagari Hindi.

This repo keeps the **original text** (no transliteration to English) and measures per-script behaviour wherever labels allow.

Only urgency/escalation have Hinglish labels, and those are synthetic.

---

## Dataset

**Source:** `cmul8-hf/IndicJevBench` (CC BY 4.0), retrieved **2026-10-07**. Full provenance table in [`DATASET.md`](DATASET.md).

| Subset | Used for | Nature |
|---|---|---|
| `fintech_banking77` (7,816 items, 7 langs) | intent (77 classes) | English = public benchmark; hi/bn/ta/te/kn/ml = machine-translated |
| `synthetic_enterprise` (444 rows → 222 texts; hi-Deva, hi-Latn) | urgency (5 levels), escalation (yes/no) | synthetic / constructed |
| `hinglish_lid`, `intent_massive` | not trained on (script-rule sanity check only / unrelated voice-assistant domain) | benchmark-derived |

### Data split and leakage audit

No official splits exist, so the project created:

- **70/15/15 splits**
- **seed 42**
- stratified
- duplicates grouped

Leakage audit (`reports/metrics/data_audit.json`):

- **0** exact-text or ID overlap across splits
- **30/1173** Banking77 test items have a near-duplicate (cosine ≥ 0.85) in train
- **29** of those near-duplicates have the same label
- **0/34** for the synthetic set

A plain char-n-gram logistic regression reaches **PR-AUC 0.955** on synthetic escalation (5-fold CV), so that label is easy to learn from surface text.

---

## Tasks

| Task | Labels | Trained on | Notes |
|---|---|---|---|
| **intent** | 77 banking intents | Banking77 (7 languages) | |
| **urgency** | `very_low` … `critical` (5) | synthetic (hi-Deva/hi-Latn) | only 1 `very_low` example exists |
| **escalation** | probability that a human is needed | synthetic | sigmoid output; temperature-scaled on validation |
| **language/script** | script rule (Unicode) + char-n-gram LR for en-Latn vs hi-Latn | LR trained on Banking77-en vs synthetic hi-Latn | in-domain accuracy 100% on val/test: optimistic (domain artefacts possible) |
| **route** | 6 destination queues | not learned | deterministic map from predicted intent (`src/routing.py`), my own taxonomy, unvalidated |

Because no text carries all labels, training uses **masked multi-task loss**: each row contributes only the losses it has labels for.

Urgency/escalation behaviour on non-Hindi inputs is **zero-shot transfer and unmeasured**.

---

## Models

### 1. BiLSTM

**1.61 M parameters**

```text
MuRIL WordPiece ids
        ↓
compact train-only vocabulary
        ↓
embedding(128)
        ↓
BiLSTM(128)
        ↓
mean + max pooling
        ↓
shared MLP
        ↓
3 heads
```

Training:

- Adam `2e-3`
- early stopped
- best epoch: **13**

### 2. MuRIL

**`google/muril-base-cased`**

**237 M parameters**

```text
MuRIL encoder
     ↓
   [CLS]
     ↓
  dropout
     ↓
3 linear heads
```

Training:

- AdamW
- encoder LR: `3e-5`
- heads LR: `1e-3`
- 10% warm-up
- 4 epochs
- CPU

### Multi-task loss

```text
L = λ₁L_intent + λ₂L_urgency + λ₃L_escalation
```

with:

```text
λ = 1
```

Synthetic training rows are repeated **×4 per epoch**.

Maximum length is **64 tokens**, which covers **>99% of texts**.

No class weighting: intent classes differ by **≤3.1×**.

---

## Architecture

```text
text
  │
  ▼
normalise (NFC, whitespace)
  │
  ▼
WordPiece tokenizer
  │
  ▼
shared encoder (BiLSTM | MuRIL)
  │
  ├──────────────┬───────────────┬─────────────────┐
  ▼              ▼               ▼
intent head    urgency head   escalation head
77-way         5-way           sigmoid → probability
softmax        softmax

language/script:
Unicode rule + char-n-gram LR for Latin text

route:
f(intent)
```

---

## Results

All values are measured on the **held-out test split** by `scripts/evaluate.py` (`reports/metrics/*.json`).

CPU latency = median single-message end-to-end `predict()` on **4 CPU threads**.

| Model | Intent macro-F1 | Urgency macro-F1 | Escalation F1 | Escalation PR-AUC | Params | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| **BiLSTM** | **0.516** [0.480, 0.536] | 0.328 [0.153, 0.574] | **0.889** | **0.953** [0.852, 1.000] | 1.61 M | **4.8 ms** |
| **MuRIL (4 epochs)** | 0.242 [0.222, 0.256] | 0.279 [0.176, 0.548] | 0.865 | 0.931 [0.789, 1.000] | 237 M | 86.1 ms |
| **Majority / prior baseline** | 0.000 | 0.116 | 0.791 | 0.654 | – | – |

Intervals are **95% bootstrap CIs**.

- Intent `n = 1,173`
- Urgency `n = 27`
- Escalation `n = 26` (**17 positive**)
- Intent accuracy: **BiLSTM 0.529**, **MuRIL 0.327**

The urgency and escalation differences between models are within noise; the intent gap is not.

MuRIL's validation loss/score were still improving at epoch 4:

```text
0.37 → 0.49 → 0.49 → 0.52
```

So this is a statement about **this compute budget**, not about MuRIL's ceiling.

Always predicting **“escalate”** already gets F1 **0.79** on this synthetic test set.

---

## Escalation Calibration

Calibration of the escalation probability on the test set (`n=26`; temperature fitted on the equally tiny validation split):

| Model | Brier before → after | ECE before → after | Temperature |
|---|---|---|---:|
| BiLSTM | 0.125 → **0.102** | 0.145 → **0.126** | 3.78 |
| MuRIL | 0.123 → 0.120 | 0.133 → **0.125** | 1.10 |

The raw BiLSTM model is very over-confident.

With only **26 examples**, these calibration numbers are **indicative only**.

The probability is a **model output**, not a statement about correctness.

---

## Language Breakdown

Test-split macro-F1. `–` = not measured (no labels).

| Language | n (intent / urg / esc) | BiLSTM intent | MuRIL intent | BiLSTM urgency | MuRIL urgency | BiLSTM esc F1 | MuRIL esc F1 |
|---|---:|---:|---:|---:|---:|---:|---:|
| `en-Latn` | 199 / – / – | 0.449 | 0.209 | – | – | – | – |
| `hi-Deva` | 163 / 13 / 13 | 0.430 | 0.210 | 0.725 | 0.363 | 0.947 | 0.842 |
| `hi-Latn` (Hinglish) | 0 / 14 / 13 | – | – | 0.123 | 0.327 | 0.824 | 0.889 |
| `bn-Beng` | 167 / – / – | 0.478 | 0.287 | – | – | – | – |
| `kn-Knda` | 164 / – / – | 0.457 | 0.223 | – | – | – | – |
| `ml-Mlym` | 140 / – / – | 0.362 | 0.219 | – | – | – | – |
| `ta-Taml` | 165 / – / – | 0.441 | 0.254 | – | – | – | – |
| `te-Telu` | 175 / – / – | 0.507 | 0.197 | – | – | – | – |

Per-language intent differences of a few points are noise (`n≈170` over up to 77 classes).

**Hinglish intent and code-mixed performance cannot be measured with this data.**

Intent accuracy by length (tokens):

| Token length | BiLSTM | MuRIL |
|---|---:|---:|
| ≤8 | 0.47 | 0.40 |
| 9–16 | 0.56 | 0.33 |
| >16 | 0.47 | 0.32 |

---

## Error Analysis

Real failed test rows:

- `reports/error_analysis.csv`
  - BiLSTM: **571 rows** with ≥1 wrong task
  - MuRIL: **805 rows** with ≥1 wrong task
- 24 sampled failures per model with all labels and computed patterns in `reports/error_examples.md`
- Confusion matrices:
  - `reports/figures/confusion_intent_{lstm,transformer}.png`
  - `reports/figures/confusion_urgency_*.png`
  - `reports/figures/escalation_pr_curve.png`

### Observed error patterns

Top confusions are between semantically adjacent banking intents, for example:

```text
wrong_exchange_rate_for_cash_withdrawal
                ↓
card_payment_wrong_exchange_rate
```

BiLSTM: **6**, MuRIL: **16**

```text
pending_transfer
      ↓
failed_transfer
```

BiLSTM: **6**

```text
extra_charge_on_statement
      ↓
card_payment_fee_charged
```

MuRIL: **17**

```text
verify_my_identity
      ↓
why_verify_identity
```

MuRIL: **14**

Several MuRIL errors collapse different cash-withdrawal intents into:

```text
balance_not_updated_after_cheque_or_cash_deposit
```

with **14 each**, which is consistent with under-training.

Urgency errors:

- BiLSTM: **15**
- MuRIL: **14 texts**
- mostly off by one level
- BiLSTM within-±1 accuracy: **0.93**

Escalation on 26 examples:

- BiLSTM: **1 missed / 3 false**
- MuRIL: **1 missed / 4 false**

---

## Pretrained Checkpoints (Hugging Face)

The trained weights are published (currently private) at:

`Faizdevx/hinglish-complaint-router`

Repository layout:

```text
lstm/
transformer/
langid.joblib
```

`transformer/` contains the **MuRIL `best.pt` only**.

They are git-ignored here because MuRIL alone is approximately **950 MB**.

To use the checkpoints instead of retraining:

```python
from huggingface_hub import snapshot_download

snapshot_download(
    "Faizdevx/hinglish-complaint-router",
    local_dir="models",
    allow_patterns=["lstm/*", "transformer/*", "langid.joblib"],
)
```

This needs Hugging Face access to the repo (token while it is private).

The expected layout is:

```text
models/
├── lstm/
├── transformer/
└── langid.joblib
```

The download path has **not been run end-to-end**; the upload itself was verified by listing the repo files.

Note: the MuRIL checkpoint there is the **under-trained 4-epoch run** described in Results.

---

## Web Dashboard

Start the server:

```bash
uvicorn app.main:app --reload
```

Then open:

```text
http://127.0.0.1:8000
```

| Page | What it shows |
|---|---|
| `/` | Dashboard, model status, dataset facts, latest test metrics, system info (all read from saved artifacts / live state; “Not available” otherwise) |
| `/classify` | One message → intent, urgency, escalation meter (LOW <0.33, MEDIUM <0.66, HIGH; fixed display thresholds), language, route, top-5 intent scores, latency, checkpoint, “Explain result” (derived only from model scores) |
| `/playground` | Same input through both models side by side |
| `/batch` | CSV upload (`text`, optional `id`), summary, table, Download results CSV |
| `/models`, `/evaluation` | Comparison tables and bars, per-language tables, calibration, audit highlights (from `reports/metrics/*.json`) |
| `/errors` | Confusion matrices, top confusions, searchable table of real failed test examples (model / language / intent filters) |
| `/demo` | Held-out dataset examples next to labels and live predictions |
| `/health-ui`, `/about`, `/docs` | Health, methodology & limits, Swagger |

Example inputs labelled **Demo example** are hand-written; those labelled **Dataset example** are real test-split rows.

The UI uses **system fonts and no CDN**, so it works offline.

Light/dark theme persists in `localStorage`.

---

## API

Example request:

```bash
curl -X POST localhost:8000/predict \
  -H 'content-type: application/json' \
  -d '{"text": "Mera card block ho gaya hai aur paise kat gaye", "model": "transformer"}'
```

Actual response from this repo's MuRIL checkpoint:

```json
{
  "intent": "transaction_charged_twice",
  "urgency": "critical",
  "escalation_probability": 0.957,
  "language": "hi-Latn",
  "route": "fraud_and_disputes",
  "model": "transformer",
  "inference_ms": 164.4
}
```

That intent is **wrong for this message**.

A BiLSTM run gives `cash_withdrawal_not_recognised`; MuRIL is under-trained.

Other routes:

```text
GET  /
GET  /health
GET  /api/health
GET  /api/models
GET  /api/metrics
GET  /api/errors

POST /api/predict
POST /api/compare
POST /api/batch
POST /voice/predict
```

Invalid input → **422**

Missing checkpoint → **503** with an actionable message.

---

## Batch Inference

```bash
python scripts/predict_file.py input.csv output.csv --model lstm
```

Output columns:

```text
id
text
language
intent
urgency
escalation_probability
route
```

---

## Optional Sarvam Speech Integration

`src/integrations/sarvam.py` + `POST /voice/predict` can:

```text
audio
  ↓
Sarvam REST speech-to-text
  ↓
transcript
  ↓
text classifier
```

Requires:

```text
SARVAM_API_KEY
```

Without the key, the endpoint returns **501**.

The integration was written from Sarvam's public API docs and is **untested against the live API** (no key available).

Unit tests cover only the **no-key and mocked paths**.

The text classifier never depends on the speech integration.

---

## Installation

Tested on:

- **Python 3.13**
- **Linux**
- **CPU only**

Install PyTorch CPU build:

```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
```

Then:

```bash
pip install -r requirements.txt
```

---

## Training

Prepare the dataset:

```bash
python scripts/download_data.py && \
python scripts/prepare_dataset.py && \
python scripts/inspect_dataset.py
```

Train the language detector:

```bash
python scripts/train_langid.py
```

Train the BiLSTM:

```bash
python scripts/train_lstm.py
```

Approximate runtime:

```text
~6 min on 4 CPU cores
```

Available flags include:

```text
--epochs
--batch-size
--max-length
--lr
--device
```

Train MuRIL:

```bash
python scripts/train_transformer.py
```

Approximate runtime:

```text
~76 min on 4 CPU cores for 4 epochs
```

### Smoke tests

```bash
python scripts/train_lstm.py --smoke-test
python scripts/train_transformer.py --smoke-test
```

These perform **3 steps on real samples** and check:

- backprop
- checkpoint
- reload
- inference

Each run stores:

```text
config.json
history.json
labels
tokenizer
```

under:

```text
models/<name>/
```

The config records seed, versions, learning rate, optimiser, epochs, device, max length, and model/tokenizer ID.

The EDA notebook is:

```text
notebooks/01_eda.ipynb
```

and is executed.

---

## Evaluation

Run:

```bash
python scripts/evaluate.py
```

This produces:

- metrics
- figures
- `error_analysis.csv`
- temperature scaling (fit on validation)

Then:

```bash
python scripts/summarize_errors.py
```

Run tests:

```bash
python -m pytest -q
```

The project currently has **56 tests**, offline, using tiny random models.

Run the UI check:

```bash
python scripts/ui_check.py
```

This is a headless-browser check of a running server and needs Playwright.

---

## Project Structure

```text
Hinglish-Complaint-Router/
├── app/                  # FastAPI app + dashboard
├── data/                 # dataset/data artifacts
├── models/               # trained checkpoints (git-ignored)
├── notebooks/            # EDA notebook
├── reports/              # evaluation, metrics, figures, error analysis
├── scripts/              # data preparation, training, inference, evaluation
├── src/                  # core model / routing / integrations
├── tests/                # automated tests
├── DATASET.md            # dataset provenance
├── MODEL_CARD.md         # model details and responsible-use notes
├── Dockerfile
├── Makefile
├── pyproject.toml
├── requirements.txt
└── README.md
```

---

## Limitations

- Synthetic urgency/escalation labels and ~120 training texts; tiny test sets; no human validation of labels.
- Banking-only intent taxonomy; other domains (street lights, water) get a wrong banking intent.
- Non-English Banking77 text is machine-translated; quality unchecked.
- Hinglish intent and code-mixed performance unmeasured; the Latin-script en/hi classifier is trained on different sources per class (style leakage likely).
- MuRIL is under-trained; conclusions about MuRIL vs BiLSTM hold only for this budget. Test-set size prevents conclusions on urgency/escalation.
- Route is a hand-written mapping, not validated.

---

## Responsible Use

This is a **research/portfolio routing classifier**.

Predictions, especially **urgency and escalation**, must not be the sole basis for consequential decisions about people (for example, deprioritising a fraud report).

It should not act autonomously without:

- domain-specific validation on real data
- human review
- monitoring across languages

See [`MODEL_CARD.md`](MODEL_CARD.md).

---

## About

**Multilingual complaint intelligence system for Indian users.**
