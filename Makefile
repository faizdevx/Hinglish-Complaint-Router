.PHONY: data prepare inspect langid train-lstm train-transformer evaluate test smoke serve all
data:            ; python scripts/download_data.py
prepare:         ; python scripts/prepare_dataset.py
inspect:         ; python scripts/inspect_dataset.py
langid:          ; python scripts/train_langid.py
train-lstm:      ; python scripts/train_lstm.py
train-transformer: ; python scripts/train_transformer.py
evaluate:        ; python scripts/evaluate.py
test:            ; python -m pytest -q
smoke:           ; python scripts/train_lstm.py --smoke-test && python scripts/train_transformer.py --smoke-test
serve:           ; uvicorn app.main:app --reload
all: data prepare inspect langid train-lstm train-transformer evaluate test
