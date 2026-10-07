"""Reusable multi-task training loop (BiLSTM and transformer share it)."""
from __future__ import annotations

import copy
import datetime as dt
import json
import platform
import random
import time
from pathlib import Path

import numpy as np
import torch
import transformers
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from ..config import MAX_LENGTH_DEFAULT, MODELS_DIR, SEED, TRANSFORMER_ID, TASKS
from ..data import load_label_maps, load_split
from ..dataset import ComplaintDataset, build_compact_vocab, make_collate
from ..evaluation import collect_outputs, summarize
from ..models import BiLSTMMultiTask, TransformerMultiTask

DEFAULTS = {
    "lstm": dict(epochs=25, batch_size=32, lr=2e-3, patience=5),
    "transformer": dict(epochs=4, batch_size=32, lr=3e-5, patience=2),
}


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def get_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def count_params(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters())


def multitask_loss(out: dict, batch: dict, lambdas: dict, device) -> tuple[torch.Tensor, dict]:
    """L = l1*L_intent + l2*L_urgency + l3*L_escalation, each averaged over rows that HAVE that label."""
    parts = {}
    zero = out["intent"].sum() * 0.0
    yi, yu, ye = (batch[k].to(device) for k in ("intent", "urgency", "escalation"))
    mi, mu, me = yi >= 0, yu >= 0, ye >= 0
    parts["intent"] = F.cross_entropy(out["intent"][mi], yi[mi]) if mi.any() else zero
    parts["urgency"] = F.cross_entropy(out["urgency"][mu], yu[mu]) if mu.any() else zero
    parts["escalation"] = (F.binary_cross_entropy_with_logits(out["escalation"][me], ye[me].float())
                           if me.any() else zero)
    total = sum(lambdas[t] * parts[t] for t in TASKS)
    return total, parts


def _batch_hits(out, batch, device, hits):
    yi, yu, ye = (batch[k].to(device) for k in ("intent", "urgency", "escalation"))
    for name, y, pred in (("intent", yi, out["intent"].argmax(1)), ("urgency", yu, out["urgency"].argmax(1)),
                          ("escalation", ye, (out["escalation"] > 0).long())):
        m = y >= 0
        h = hits[name]
        h[0] += int((pred[m] == y[m]).sum())
        h[1] += int(m.sum())


def val_score(summary: dict) -> float:
    """Early-stopping score: mean of intent macro-F1, urgency macro-F1, escalation PR-AUC (those available)."""
    o = summary["overall"]
    vals = []
    if o["intent"]:
        vals.append(o["intent"]["macro_f1"])
    if o["urgency"]:
        vals.append(o["urgency"]["macro_f1"])
    if o["escalation"] and o["escalation"]["pr_auc"] is not None:
        vals.append(o["escalation"]["pr_auc"])
    return float(np.mean(vals)) if vals else 0.0


def _smoke_rows(rows: list[dict], n_intent: int, n_syn: int) -> list[dict]:
    intent = [r for r in rows if r["intent"] is not None][:n_intent]
    syn = [r for r in rows if r["intent"] is None][:n_syn]
    return intent + syn


def run_training(model_key: str, *, epochs: int | None = None, batch_size: int | None = None,
                 max_length: int = MAX_LENGTH_DEFAULT, lr: float | None = None, patience: int | None = None,
                 seed: int = SEED, smoke_test: bool = False, synthetic_repeat: int = 4,
                 lambdas: dict | None = None, device: str | None = None, out_dir: Path | None = None,
                 head_lr: float = 1e-3, log=print) -> dict:
    d = DEFAULTS[model_key]
    epochs = epochs or d["epochs"]
    batch_size = batch_size or d["batch_size"]
    lr = lr or d["lr"]
    patience = patience or d["patience"]
    lambdas = lambdas or {"intent": 1.0, "urgency": 1.0, "escalation": 1.0}
    seed_everything(seed)
    dev = get_device(device)
    out_dir = Path(out_dir) if out_dir else MODELS_DIR / (f"smoke_{model_key}" if smoke_test else model_key)
    out_dir.mkdir(parents=True, exist_ok=True)

    label_maps = load_label_maps()
    train_rows, val_rows = load_split("train"), load_split("val")
    if smoke_test:
        train_rows, val_rows = _smoke_rows(train_rows, 40, 16), _smoke_rows(val_rows, 24, 8)
        epochs, synthetic_repeat = 1, 1
    n_syn_train = sum(r["intent"] is None for r in train_rows)
    train_fit = train_rows + [r for r in train_rows if r["intent"] is None] * (synthetic_repeat - 1)

    tokenizer = AutoTokenizer.from_pretrained(TRANSFORMER_ID)
    n_intent = len(label_maps["intent"])
    id_map = None
    if model_key == "lstm":
        id_map = build_compact_vocab(tokenizer, [r["text"] for r in train_rows], max_length)  # train only
        model = BiLSTMMultiTask(len(id_map), n_intent)
    else:
        model = TransformerMultiTask(TRANSFORMER_ID, n_intent)
    model.to(dev)
    collate = make_collate(tokenizer, max_length, id_map)
    g = torch.Generator().manual_seed(seed)
    train_dl = DataLoader(ComplaintDataset(train_fit, label_maps), batch_size=batch_size, shuffle=True,
                          collate_fn=collate, generator=g)
    val_ds = ComplaintDataset(val_rows, label_maps)
    val_dl = DataLoader(val_ds, batch_size=64, shuffle=False, collate_fn=collate)

    if model_key == "transformer":
        heads = [p for n, p in model.named_parameters() if n.endswith(("head.weight", "head.bias"))]
        enc = [p for n, p in model.named_parameters() if not n.endswith(("head.weight", "head.bias"))]
        opt = torch.optim.AdamW([{"params": enc, "lr": lr}, {"params": heads, "lr": head_lr}], weight_decay=0.01)
        optimizer_name = f"AdamW(encoder lr={lr}, heads lr={head_lr}, wd=0.01) + linear warmup/decay"
    else:
        opt = torch.optim.Adam(model.parameters(), lr=lr)
        optimizer_name = f"Adam(lr={lr})"
    total_steps = max(1, epochs * len(train_dl))
    sched = None
    if model_key == "transformer":
        warm = max(1, int(0.1 * total_steps))
        sched = torch.optim.lr_scheduler.LambdaLR(
            opt, lambda s: (s + 1) / warm if s < warm else max(0.0, (total_steps - s) / max(1, total_steps - warm)))

    init_state = copy.deepcopy({k: v.detach().cpu().clone() for k, v in model.state_dict().items()}) if smoke_test else None
    history, best, best_epoch, bad = [], -1.0, -1, 0
    max_steps = 3 if smoke_test else None
    step = 0
    t0 = time.time()
    log(f"[{model_key}] device={dev} params={count_params(model):,} train_rows={len(train_fit)} "
        f"(synthetic rows x{synthetic_repeat}: {n_syn_train}) val_rows={len(val_rows)} steps/epoch={len(train_dl)}")
    for epoch in range(1, epochs + 1):
        model.train()
        sums = {"total": 0.0, **{t: 0.0 for t in TASKS}}
        counts = {t: 0 for t in TASKS}
        hits = {t: [0, 0] for t in TASKS}
        nb = 0
        for batch in train_dl:
            out = model(batch["input_ids"].to(dev), batch["attention_mask"].to(dev))
            loss, parts = multitask_loss(out, batch, lambdas, dev)
            if not torch.isfinite(loss):
                raise RuntimeError("non-finite loss")
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            if sched:
                sched.step()
            sums["total"] += loss.item()
            for t in TASKS:
                if (batch[t] >= 0).any():
                    sums[t] += parts[t].item()
                    counts[t] += 1
            _batch_hits(out, batch, dev, hits)
            nb += 1
            step += 1
            if nb % 25 == 0:
                log(f"  epoch {epoch} step {nb}/{len(train_dl)} loss={sums['total'] / nb:.4f} "
                    f"elapsed={time.time() - t0:.0f}s")
            if max_steps and step >= max_steps:
                break
        out_val = collect_outputs(model, val_dl, dev)
        summ = summarize(out_val, val_rows, by_lang=False)
        score = val_score(summ)
        rec = {"epoch": epoch, "train_loss_total": sums["total"] / max(nb, 1),
               **{f"train_loss_{t}": sums[t] / max(counts[t], 1) for t in TASKS},
               **{f"train_acc_{t}": (hits[t][0] / hits[t][1] if hits[t][1] else None) for t in TASKS},
               "val": summ["overall"], "val_score": score, "elapsed_s": time.time() - t0}
        history.append(rec)
        log(f"[{model_key}] epoch {epoch}: loss={rec['train_loss_total']:.4f} val_score={score:.4f} "
            f"intent_f1={(summ['overall']['intent'] or {}).get('macro_f1')} "
            f"urg_f1={(summ['overall']['urgency'] or {}).get('macro_f1')} "
            f"esc_prauc={(summ['overall']['escalation'] or {}).get('pr_auc')}")
        if score > best:
            best, best_epoch, bad = score, epoch, 0
            torch.save({"state_dict": model.state_dict(), "epoch": epoch, "val_score": score}, out_dir / "best.pt")
        else:
            bad += 1
        if bad >= patience:
            log(f"[{model_key}] early stopping at epoch {epoch}")
            break
    torch.save({"state_dict": model.state_dict(), "epoch": epoch}, out_dir / "final.pt")
    tokenizer.save_pretrained(out_dir / "tokenizer")
    (out_dir / "labels.json").write_text(json.dumps(label_maps, indent=2, ensure_ascii=False))
    if id_map is not None:
        (out_dir / "vocab.json").write_text(json.dumps({str(k): v for k, v in id_map.items()}))
    config = {
        "model_key": model_key, "arch": model.cfg, "parameters": count_params(model),
        "tokenizer_id": TRANSFORMER_ID, "model_id": TRANSFORMER_ID if model_key == "transformer" else "BiLSTM (from scratch)",
        "max_length": max_length, "batch_size": batch_size, "learning_rate": lr, "optimizer": optimizer_name,
        "epochs_requested": epochs, "epochs_run": len(history), "best_epoch": best_epoch, "best_val_score": best,
        "early_stopping_patience": patience, "seed": seed, "device": str(dev), "lambdas": lambdas,
        "synthetic_repeat": synthetic_repeat, "smoke_test": smoke_test,
        "train_rows_unique": len(train_rows), "val_rows": len(val_rows),
        "python": platform.python_version(), "torch": torch.__version__, "transformers": transformers.__version__,
        "trained_at": dt.datetime.now(dt.timezone.utc).isoformat(timespec="seconds"),
        "train_seconds": time.time() - t0,
        "val_score_definition": "mean(intent macro-F1, urgency macro-F1, escalation PR-AUC) on validation split",
    }
    (out_dir / "config.json").write_text(json.dumps(config, indent=2))
    (out_dir / "history.json").write_text(json.dumps(history, indent=2))
    if smoke_test:
        changed = any(not torch.equal(init_state[k], v.detach().cpu()) for k, v in model.state_dict().items())
        config["backprop_changed_params"] = changed
        if not changed:
            raise RuntimeError("smoke test: parameters did not change after optimizer steps")
    return config
