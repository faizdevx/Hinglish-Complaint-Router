# Model card: Hinglish Complaint Router

**Purpose.** Research / portfolio multi-task classifier that predicts intent (77 banking intents), urgency (5 levels), escalation probability and a derived route for short multilingual messages, and compares a BiLSTM with a fine-tuned MuRIL. It should not be treated as an autonomous high-stakes decision-maker without domain-specific validation.

**Models.** (1) BiLSTM, 1.61 M parameters, trained from scratch. (2) `google/muril-base-cased` (Apache-2.0) + 3 linear heads, 237 M parameters, 4 epochs. Seed 42, max length 64, CPU training. Exact hyperparameters: `models/*/config.json` (git-ignored; regenerate with the training scripts).

**Data and provenance.** See [DATASET.md](DATASET.md). Intent: Banking77 via IndicJevBench (English public benchmark; six Indic languages **machine-translated** with NLLB-200). Urgency/escalation: `synthetic_enterprise`, **LLM-generated Hindi/Hinglish**, 222 texts. Not real customer data. Licence of the data: CC BY 4.0.

**Languages / scripts.** English, Hindi (Devanagari), Bengali, Tamil, Telugu, Kannada, Malayalam for intent; Hindi Devanagari and Romanised Hindi for urgency/escalation. Romanised Hindi has **no intent labels**.

**Intended use.** Studying multilingual/code-mixed classification, comparing small vs large encoders, demonstrating an ML dashboard.

**Not intended for.** Production triage of real complaints; safety-, fraud-, legal- or finance-critical decisions; any domain outside banking support; inferring anything about individuals.

**Evaluation (test split, measured).**

| Model | Intent macro-F1 | Urgency macro-F1 (n=27) | Escalation F1 / PR-AUC (n=26) | Escalation Brier / ECE after temperature scaling |
|---|---:|---:|---:|---:|
| BiLSTM | 0.516 | 0.328 | 0.889 / 0.953 | 0.102 / 0.126 |
| MuRIL | 0.242 | 0.279 | 0.865 / 0.931 | 0.120 / 0.125 |

Per-language and length slices, confidence intervals and baselines: README and `reports/metrics/`.

**Known errors.** Confusions among adjacent banking intents (exchange-rate, fee, transfer-status, identity-verification families); MuRIL additionally collapses several cash-withdrawal intents. Urgency errors are mostly off by one level. Real failed examples: `reports/error_analysis.csv`, `reports/error_examples.md`.

**Class imbalance.** Intent: largest class 3.1× the smallest, no reweighting. Urgency: 1 `very_low`, 12 `low`, 37 `medium`, 57 `high`, 68 `critical` over the 175 labelled texts; escalation 67% positive. No resampling was applied.

**Script limitations.** Script detection is a Unicode-range rule; Devanagari is assumed to be Hindi (Marathi/Nepali would be mislabelled). The en-Latn vs hi-Latn classifier is trained on Banking77-English vs synthetic Hinglish and reports 100% in-domain, which likely reflects style differences between sources.

**Code-mixing limitations.** Only 39 of 8,038 rows mix Devanagari and Latin letters, and none carry intent labels, so code-mixed behaviour is unmeasured. Romanised-Hindi intent is unmeasured.

**Calibration limitations.** The escalation probability is a sigmoid output with temperature scaling fitted on 28 validation examples and checked on 26 test examples; calibration estimates are very noisy. A high probability is not a confidence judgement about correctness. The synthetic escalation label is largely predictable from surface text (non-neural baseline PR-AUC 0.955), so it says little about real-world difficulty. Intent scores are uncalibrated softmax outputs.

**Ethical considerations.** Wrongly low urgency/escalation can delay help for people with genuine problems; wrongly high values can waste staff time. Performance differs across languages and scripts and is unmeasured for several, so it must be validated per language before any deployment. Machine-translated and LLM-generated training text can carry translation and generation artefacts and biases. Do not use for decisions about individuals without human review.
