# data/

* `raw/`: files downloaded by `python scripts/download_data.py` from `cmul8-hf/IndicJevBench` (git-ignored; ~50 MB). `download_info.json` records SHA-256 and retrieval date.
* `processed/`: output of `python scripts/prepare_dataset.py` (train/val/test JSONL, `label_maps.json`, `summary.json`). Small; committed so the app can show dataset facts and examples.

See `../DATASET.md` for provenance and licences of every subset.
