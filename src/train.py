"""تدريب النموذج مع: transfer learning، class weights، label smoothing، early stopping،
معايرة الثقة بـ temperature scaling، واختيار عتبة القبول التلقائي.

الاستخدام:
  python -m src.train --splits data/processed/splits.csv --arch efficientnet_b0 --epochs 30
"""

from __future__ import annotations

import argparse
import json
import random
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from .dataset import load_rows, make_loader
from .evaluate import collect_logits, evaluate_checkpoint, pick_device
from .metrics import compute_metrics, select_threshold, softmax
from .model import SUPPORTED_ARCHS, build_model


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def fit_temperature(logits: np.ndarray, y: np.ndarray) -> float:
    """Temperature scaling: يجد T يقلل NLL على تقسيم التحقق، فتصبح الاحتمالات أصدق."""
    log_t = torch.zeros(1, requires_grad=True)
    lt, yt = torch.tensor(logits), torch.tensor(y)
    opt = torch.optim.LBFGS([log_t], lr=0.1, max_iter=100)

    def closure():
        opt.zero_grad()
        loss = F.cross_entropy(lt / log_t.exp(), yt)
        loss.backward()
        return loss

    opt.step(closure)
    return float(log_t.exp().item())


def run_training(cfg: dict) -> dict:
    """ينفذ التدريب كاملًا ويكتب model.pt و metrics.json داخل cfg['out_dir']."""
    set_seed(cfg["seed"])
    device = pick_device()
    splits_csv = Path(cfg["splits_csv"])
    out_dir = Path(cfg["out_dir"])
    out_dir.mkdir(parents=True, exist_ok=True)

    classes = json.loads((splits_csv.parent / "classes.json").read_text(encoding="utf-8"))
    train_rows = load_rows(splits_csv, "train")
    val_rows = load_rows(splits_csv, "val")

    kw = dict(batch_size=cfg["batch_size"], num_workers=cfg["num_workers"])
    train_loader = make_loader(train_rows, classes, cfg["img_size"], train=True, **kw)
    val_loader = make_loader(val_rows, classes, cfg["img_size"], train=False, shuffle=False, **kw)

    # أوزان الفئات: تعوّض الفئات الأقل عددًا.
    counts = Counter(r["label"] for r in train_rows)
    total = sum(counts.values())
    weights = torch.tensor(
        [total / (len(classes) * counts[c]) if counts[c] else 0.0 for c in classes], dtype=torch.float32
    ).to(device)
    criterion = nn.CrossEntropyLoss(weight=weights, label_smoothing=0.05)

    model = build_model(cfg["arch"], len(classes), pretrained=cfg["pretrained"]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg["lr"], weight_decay=cfg["weight_decay"])
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg["epochs"])

    best_f1, best_loss, bad_epochs = -1.0, float("inf"), 0
    best_path = out_dir / "best_state.pt"
    history = []
    for epoch in range(1, cfg["epochs"] + 1):
        t0 = time.time()
        model.train()
        running, seen = 0.0, 0
        for x, y in train_loader:
            x, y = x.to(device), y.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()
            running += loss.item() * len(y)
            seen += len(y)
        scheduler.step()

        val_logits, val_y = collect_logits(model, val_loader, device)
        val_loss = F.cross_entropy(torch.tensor(val_logits), torch.tensor(val_y)).item()
        vm = compute_metrics(softmax(val_logits), val_y, classes)
        rec = {
            "epoch": epoch,
            "train_loss": running / max(seen, 1),
            "val_loss": val_loss,
            "val_accuracy": vm["accuracy"],
            "val_macro_f1": vm["macro_f1"],
            "seconds": round(time.time() - t0, 1),
        }
        history.append(rec)
        print(json.dumps(rec, ensure_ascii=False))

        improved = vm["macro_f1"] > best_f1 or (vm["macro_f1"] == best_f1 and val_loss < best_loss)
        if improved:
            best_f1, best_loss, bad_epochs = vm["macro_f1"], val_loss, 0
            torch.save(model.state_dict(), best_path)
        else:
            bad_epochs += 1
            if bad_epochs >= cfg["patience"]:
                print(f"early stopping at epoch {epoch}")
                break

    # تحميل أفضل وزن، ثم معايرة الثقة وتحديد عتبة القبول على تقسيم التحقق فقط.
    model.load_state_dict(torch.load(best_path, map_location=device, weights_only=True))
    val_logits, val_y = collect_logits(model, val_loader, device)
    temperature = fit_temperature(val_logits, val_y)
    val_probs = softmax(val_logits / temperature)
    thr = select_threshold(val_probs, val_y, target_accuracy=cfg["target_accuracy"])

    meta = {
        "arch": cfg["arch"],
        "img_size": cfg["img_size"],
        "classes": classes,
        "temperature": temperature,
        "threshold": thr["threshold"],
        "target_accuracy": cfg["target_accuracy"],
    }
    ckpt_path = out_dir / "model.pt"
    torch.save({"state_dict": model.state_dict(), "meta": meta}, ckpt_path)
    best_path.unlink(missing_ok=True)

    val_metrics = compute_metrics(val_probs, val_y, classes)
    test_metrics = evaluate_checkpoint(str(ckpt_path), str(splits_csv), split="test")
    summary = {
        "config": cfg,
        "temperature": temperature,
        "threshold": thr,
        "history": history,
        "val": val_metrics,
        "test": test_metrics,
    }
    (out_dir / "metrics.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"test accuracy={test_metrics['accuracy']:.4f} macro_f1={test_metrics['macro_f1']:.4f}")
    return summary


def parse_args(argv=None) -> dict:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--splits", default="data/processed/splits.csv")
    ap.add_argument("--out-dir", default="runs/baseline")
    ap.add_argument("--arch", default="efficientnet_b0", choices=SUPPORTED_ARCHS)
    ap.add_argument("--img-size", type=int, default=224)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--lr", type=float, default=3e-4)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--target-accuracy", type=float, default=0.98,
                    help="دقة الصور المقبولة تلقائيًا المطلوبة عند اختيار العتبة")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--num-workers", type=int, default=2)
    ap.add_argument("--no-pretrained", action="store_true", help="تدريب من الصفر (بدون أوزان ImageNet)")
    a = ap.parse_args(argv)
    return {
        "splits_csv": a.splits, "out_dir": a.out_dir, "arch": a.arch, "img_size": a.img_size,
        "epochs": a.epochs, "batch_size": a.batch_size, "lr": a.lr, "weight_decay": a.weight_decay,
        "patience": a.patience, "target_accuracy": a.target_accuracy, "seed": a.seed,
        "num_workers": a.num_workers, "pretrained": not a.no_pretrained,
    }


if __name__ == "__main__":
    run_training(parse_args())
