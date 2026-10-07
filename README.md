# Hinglish Complaint Router

A multilingual, multi-task NLP classifier that reads a short user message (English, Hindi in Devanagari, Romanised Hindi / Hinglish, and five other Indian languages) and predicts **intent**, **urgency**, **escalation probability** and a **route**, plus a FastAPI service and an internal-style web dashboard to inspect and compare two trained models: a **BiLSTM** baseline and a fine-tuned **MuRIL** encoder.

> **Read this first: what the data can and cannot support**
> * There is **no dataset of real labelled customer complaints** here. Intent labels come from **Banking77** (banking support queries; English original, the other six languages **machine-translated** with NLLB-200). Urgency/escalation labels come from a **synthetic, LLM-generated** Hindi/Hinglish set (222 texts). The project is therefore a *multilingual message-intent routing* study, **not** a model trained on real complaints.
> * **No labelled Hinglish intent data exists** in the chosen subsets, so Hinglish intent accuracy is **not measured**.
> * Urgency/escalation test sets are tiny (27 / 26 texts). Treat those numbers as anecdotal.
> * The MuRIL model is **under-trained** (4 epochs on CPU, loss still falling) and **loses badly to the BiLSTM** here. That is reported as measured.

## Problem

Support-style messages arrive in many languages and scripts. A router must decide what the message is about, how urgent it is, whether a human should take over, and which queue should get it.

## Why Hinglish?

Romanised Hindi is Hindi written in Latin letters, often mixed with English words (“Transaction failed hui par service charge kat li”). Spelling is unstandardised, the same word has many romanisations, and script-based tools (Unicode ranges, Devanagari-trained tokenizers) behave differently from either English or Devanagari Hindi. This repo keeps the original text (no transliteration to English) and measures per-script behaviour wherever labels allow. Only urgency/escalation have Hinglish labels, and those are synthetic.

## Dataset

Source: `cmul8-hf/IndicJevBench` (CC BY 4.0), retrieved 2026-10-07. Full provenance table in [DATASET.md](DATASET.md).

| Subset | Used for | Nature |
|---|---|---|
| `fintech_banking77` (7,816 items, 7 langs) | intent (77 classes) | English = public benchmark; hi/bn/ta/te/kn/ml = **machine-translated** |
| `synthetic_enterprise` (444 rows → 222 texts; hi-Deva, hi-Latn) | urgency (5 levels), escalation (yes/no) | **synthetic / constructed** |
| `hinglish_lid`, `intent_massive` | not trained on (script-rule sanity check only / unrelated voice-assistant domain) | benchmark-derived |

No official splits exist; I created 70/15/15 splits (seed 42, stratified, duplicates grouped). Leakage audit (`reports/metrics/data_audit.json`): 0 exact-text or ID overlap across splits; 30/1173 Banking77 test items have a near-duplicate (cosine ≥ 0.85) in train (29 with the same label); 0/34 for the synthetic set. A plain char-n-gram logistic regression reaches PR-AUC 0.955 on synthetic escalation (5-fold CV), so that label is easy to learn from surface text.

## Tasks

| Task | Labels | Trained on | Notes |
|---|---|---|---|
| intent | 77 banking intents | Banking77 (7 languages) | |
| urgency | very_low … critical (5) | synthetic (hi-Deva/hi-Latn) | only 1 `very_low` example exists |
| escalation | probability that a human is needed | synthetic | sigmoid output; temperature-scaled on validation |
| language/script | script rule (Unicode) + char-n-gram LR for en-Latn vs hi-Latn | LR trained on Banking77-en vs synthetic hi-Latn | in-domain accuracy 100% on val/test: **optimistic** (domain artefacts possible) |
| route | 6 destination queues | **not learned** | deterministic map from predicted intent (`src/routing.py`), my own taxonomy, unvalidated |

Because no text carries all labels, training uses **masked multi-task loss**: each row contributes only the losses it has labels for. Urgency/escalation behaviour on non-Hindi inputs is zero-shot transfer and unmeasured.

## Models

* **BiLSTM** (1.61 M params): MuRIL WordPiece ids → compact train-only vocabulary → embedding(128) → BiLSTM(128) → mean+max pooling → shared MLP → 3 heads. Adam 2e-3, early stopped (best epoch 13).
* **MuRIL** (`google/muril-base-cased`, Apache-2.0, 237 M params): `[CLS]` → dropout → 3 linear heads. AdamW (encoder 3e-5, heads 1e-3), 10% warm-up, 4 epochs, CPU.
* `L = λ₁L_intent + λ₂L_urgency + λ₃L_escalation`, λ = 1. Synthetic training rows are repeated ×4 per epoch. Max length 64 tokens (covers >99% of texts). No class weighting: intent classes differ ≤3.1×.

## Architecture

```
text ─► normalise (NFC, whitespace) ─► WordPiece tokenizer ─► shared encoder (BiLSTM | MuRIL)
                                                         ├─► intent head     (77-way softmax)
                                                         ├─► urgency head    (5-way softmax)
                                                         └─► escalation head (sigmoid → probability)
language/script: Unicode rule (+ char-n-gram LR for Latin text)        route = f(intent)
```

## Results

All values measured on the held-out **test split** by `scripts/evaluate.py` (`reports/metrics/*.json`). CPU latency = median single-message end-to-end `predict()` on 4 CPU threads.

| Model | Intent macro-F1 | Urgency macro-F1 | Escalation F1 | Escalation PR-AUC | Params | CPU latency |
|---|---:|---:|---:|---:|---:|---:|
| BiLSTM | **0.516** [0.480, 0.536] | **0.328** [0.153, 0.574] | **0.889** | **0.953** [0.852, 1.000] | 1.61 M | **4.8 ms** |
| MuRIL (4 epochs) | 0.242 [0.222, 0.256] | 0.279 [0.176, 0.548] | 0.865 | 0.931 [0.789, 1.000] | 237 M | 86.1 ms |
| Majority / prior baseline | 0.000 | 0.116 | 0.791 | 0.654 | – | – |

Intervals are 95% bootstrap CIs. Intent n=1,173; urgency n=27; escalation n=26 (17 positive). Intent accuracy: BiLSTM 0.529, MuRIL 0.327. The urgency and escalation differences between models are **within noise**; the intent gap is not. MuRIL's validation loss/score were still improving at epoch 4 (val score 0.37 → 0.49 → 0.49 → 0.52), so this is a statement about this compute budget, not about MuRIL's ceiling. Always predicting “escalate” already gets F1 0.79 on this synthetic test set.

**Calibration of the escalation probability** (test, n=26; temperature fitted on the equally tiny validation split): BiLSTM Brier 0.125 → 0.102, ECE 0.145 → 0.126 (T=3.78; the raw model is very over-confident); MuRIL Brier 0.123 → 0.120, ECE 0.133 → 0.125 (T=1.10). With 26 examples these are indicative only. The probability is a model output, not a statement about correctness.

## Language breakdown

Test-split macro-F1; “–” = **not measured** (no labels).

| Language | n (intent / urg / esc) | BiLSTM intent | MuRIL intent | BiLSTM urgency | MuRIL urgency | BiLSTM esc F1 | MuRIL esc F1 |
|---|---|---:|---:|---:|---:|---:|---:|
| en-Latn | 199 / – / – | 0.449 | 0.209 | – | – | – | – |
| hi-Deva | 163 / 13 / 13 | 0.430 | 0.210 | 0.725 | 0.363 | 0.947 | 0.842 |
| hi-Latn (Hinglish) | **0 / 14 / 13** | **–** | **–** | 0.123 | 0.327 | 0.824 | 0.889 |
| bn-Beng | 167 / – / – | 0.478 | 0.287 | – | – | – | – |
| kn-Knda | 164 / – / – | 0.457 | 0.223 | – | – | – | – |
| ml-Mlym | 140 / – / – | 0.362 | 0.219 | – | – | – | – |
| ta-Taml | 165 / – / – | 0.441 | 0.254 | – | – | – | – |
| te-Telu | 175 / – / – | 0.507 | 0.197 | – | – | – | – |

Per-language intent differences of a few points are noise (n≈170 over up to 77 classes). Hinglish intent and code-mixed performance cannot be measured with this data. Intent accuracy by length (tokens): BiLSTM 0.47 (≤8) / 0.56 (9–16) / 0.47 (>16); MuRIL 0.40 / 0.33 / 0.32.

## Error analysis

Real failed test rows: `reports/error_analysis.csv` (BiLSTM 571, MuRIL 805 rows with ≥1 wrong task); 24 sampled failures per model with all labels and computed patterns in [`reports/error_examples.md`](reports/error_examples.md); confusion matrices `reports/figures/confusion_intent_{lstm,transformer}.png`, `confusion_urgency_*.png`, `escalation_pr_curve.png`.

Patterns (counted, not interpreted by a model): the top confusions are between *semantically adjacent* banking intents, e.g. `wrong_exchange_rate_for_cash_withdrawal → card_payment_wrong_exchange_rate` (BiLSTM 6, MuRIL 16), `pending_transfer → failed_transfer` (BiLSTM 6), `extra_charge_on_statement → card_payment_fee_charged` (MuRIL 17), `verify_my_identity → why_verify_identity` (MuRIL 14). Several MuRIL errors collapse different cash-withdrawal intents into `balance_not_updated_after_cheque_or_cash_deposit` (14 each), which is consistent with under-training. Urgency errors (BiLSTM 15, MuRIL 14 texts) are mostly off by one level (BiLSTM within-±1 accuracy 0.93). Escalation: BiLSTM 1 missed / 3 false, MuRIL 1 missed / 4 false escalations out of 26.

## Web dashboard

```bash
uvicorn app.main:app --reload       # then open http://127.0.0.1:8000
```

| Page | What it shows |
|---|---|
| `/` Dashboard | model status, dataset facts, latest test metrics, system info (all read from saved artifacts / live state; “Not available” otherwise) |
| `/classify` | one message → intent, urgency, escalation meter (bands LOW <0.33, MEDIUM <0.66, HIGH; fixed display thresholds), language, route, top-5 intent scores, latency, checkpoint, “Explain result” (derived only from model scores) |
| `/playground` | same input through both models side by side |
| `/batch` | CSV upload (`text`, optional `id`), summary, table, **Download results CSV** |
| `/models`, `/evaluation` | comparison tables and bars, per-language tables, calibration, audit highlights (from `reports/metrics/*.json`) |
| `/errors` | confusion matrices, top confusions, searchable table of real failed test examples (model / language / intent filters) |
| `/demo` | held-out dataset examples next to labels and live predictions |
| `/health-ui`, `/about`, `/docs` | health, methodology & limits, Swagger |

Example inputs labelled **Demo example** are hand-written; those labelled **Dataset example** are real test-split rows. The UI uses system fonts and no CDN, so it works offline. Light/dark theme persists in `localStorage`.

## API

```bash
curl -X POST localhost:8000/predict -H 'content-type: application/json' \
  -d '{"text": "Mera card block ho gaya hai aur paise kat gaye", "model": "transformer"}'
```
Actual response from this repo's MuRIL checkpoint:
```json
{"intent":"transaction_charged_twice","urgency":"critical","escalation_probability":0.957,"language":"hi-Latn","route":"fraud_and_disputes","model":"transformer","inference_ms":164.4}
```
(That intent is wrong for this message, and a BiLSTM run gives `cash_withdrawal_not_recognised`: MuRIL is under-trained.) Other routes: `GET /`, `GET /health`, `GET /api/health`, `GET /api/models`, `GET /api/metrics`, `GET /api/errors`, `POST /api/predict` (full payload), `POST /api/compare`, `POST /api/batch`, `POST /voice/predict` (optional, below). Invalid input → 422, missing checkpoint → 503 with an actionable message.

## Batch inference

```bash
python scripts/predict_file.py input.csv output.csv --model lstm   # columns: id,text,language,intent,urgency,escalation_probability,route
```

## Optional Sarvam speech integration

`src/integrations/sarvam.py` + `POST /voice/predict` send audio to Sarvam's REST speech-to-text endpoint, then classify the transcript. Requires `SARVAM_API_KEY`; returns 501 without it. It was written from Sarvam's public API docs and is **untested against the live API** (no key available); unit tests cover only the no-key and mocked paths. The text classifier never depends on it.

## Installation

Tested on Python 3.13, Linux, CPU only.
```bash
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements.txt
```

## Training

```bash
python scripts/download_data.py && python scripts/prepare_dataset.py && python scripts/inspect_dataset.py
python scripts/train_langid.py
python scripts/train_lstm.py                  # ~6 min on 4 CPU cores; --epochs --batch-size --max-length --lr --device
python scripts/train_transformer.py           # ~76 min on 4 CPU cores for 4 epochs
python scripts/train_lstm.py --smoke-test     # 3 steps on real samples; checks backprop, checkpoint, reload, inference
python scripts/train_transformer.py --smoke-test
```
Each run stores `config.json` (seed 42, versions, lr, optimiser, epochs, device, max length, model/tokenizer id), `history.json`, labels and tokenizer under `models/<name>/` (git-ignored). The EDA notebook is `notebooks/01_eda.ipynb` (executed).

## Evaluation

```bash
python scripts/evaluate.py        # metrics, figures, error_analysis.csv, temperature scaling (fit on val)
python scripts/summarize_errors.py
python -m pytest -q               # 56 tests, offline, tiny random models
python scripts/ui_check.py        # headless-browser check of a running server (needs playwright)
```

## Limitations

* Synthetic urgency/escalation labels and ~120 training texts; tiny test sets; no human validation of labels.
* Banking-only intent taxonomy; other domains (street lights, water) get a wrong banking intent.
* Non-English Banking77 text is machine-translated; quality unchecked.
* Hinglish intent and code-mixed performance unmeasured; the Latin-script en/hi classifier is trained on different sources per class (style leakage likely).
* MuRIL is under-trained; conclusions about MuRIL vs BiLSTM hold only for this budget. Test-set size prevents conclusions on urgency/escalation.
* Route is a hand-written mapping, not validated.

## Responsible use

This is a research/portfolio routing classifier. Predictions (especially urgency and escalation) must not be the sole basis for consequential decisions about people (e.g. deprioritising a fraud report). It should not act autonomously without domain-specific validation on real data, human review, and monitoring across languages. See [MODEL_CARD.md](MODEL_CARD.md).
