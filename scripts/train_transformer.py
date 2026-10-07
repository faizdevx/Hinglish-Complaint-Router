"""Train the MuRIL multi-task model."""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from src.config import MAX_LENGTH_DEFAULT, SEED  # noqa: E402
from src.training.smoke import smoke_check  # noqa: E402
from src.training.trainer import run_training  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--epochs", type=int)
    ap.add_argument("--batch-size", type=int)
    ap.add_argument("--max-length", type=int, default=MAX_LENGTH_DEFAULT)
    ap.add_argument("--lr", type=float)
    ap.add_argument("--patience", type=int)
    ap.add_argument("--seed", type=int, default=SEED)
    ap.add_argument("--synthetic-repeat", type=int, default=4)
    ap.add_argument("--device")
    ap.add_argument("--smoke-test", action="store_true", help="3 steps on real dataset samples; saves to models/smoke_transformer")
    a = ap.parse_args()
    cfg = run_training("transformer", epochs=a.epochs, batch_size=a.batch_size, max_length=a.max_length, lr=a.lr,
                       patience=a.patience, seed=a.seed, smoke_test=a.smoke_test,
                       synthetic_repeat=a.synthetic_repeat, device=a.device)
    if a.smoke_test:
        smoke_check("transformer")
    print("done: best epoch", cfg["best_epoch"], "val_score", round(cfg["best_val_score"], 4))


if __name__ == "__main__":
    main()
