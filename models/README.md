Trained checkpoints live here and are git-ignored (the MuRIL checkpoint is ~950 MB).
Regenerate: `python scripts/train_lstm.py && python scripts/train_transformer.py && python scripts/evaluate.py`.
Layout per model: `best.pt final.pt config.json labels.json tokenizer/ history.json calibration.json` (+ `vocab.json` for the BiLSTM).
