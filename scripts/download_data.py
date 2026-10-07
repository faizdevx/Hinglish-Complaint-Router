"""Download the IndicJevBench files used by this project into data/raw/."""
from __future__ import annotations

import argparse
import datetime as dt
import hashlib
import json
import sys
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import DATA_RAW, HF_DATASET, HF_DATASET_FILES  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--files", nargs="*", default=HF_DATASET_FILES)
    args = ap.parse_args()
    provenance = {}
    for rel in args.files:
        dest = DATA_RAW / rel
        dest.parent.mkdir(parents=True, exist_ok=True)
        url = f"https://huggingface.co/datasets/{HF_DATASET}/resolve/main/{rel}"
        print("GET", url)
        with urllib.request.urlopen(url, timeout=300) as r:
            dest.write_bytes(r.read())
        provenance[rel] = {"sha256": hashlib.sha256(dest.read_bytes()).hexdigest(), "bytes": dest.stat().st_size}
    out = DATA_RAW / "download_info.json"
    out.write_text(json.dumps({"dataset": HF_DATASET, "retrieved": dt.date.today().isoformat(), "files": provenance}, indent=2))
    print("wrote", out)


if __name__ == "__main__":
    main()
