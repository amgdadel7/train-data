"""تحميل نقاط الحفظ (checkpoints) وتقييمها على أي تقسيم (val / test)."""

from __future__ import annotations

import numpy as np
import torch

from .dataset import load_rows, make_loader
from .metrics import compute_metrics, decision_stats, softmax
from .model import build_model


def pick_device() -> torch.device:
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def load_checkpoint(path: str, device: torch.device | None = None):
    """يعيد (model, meta, device). meta تحوي: arch, img_size, classes, temperature, threshold."""
    device = device or pick_device()
    ck = torch.load(path, map_location=device, weights_only=True)
    meta = ck["meta"]
    model = build_model(meta["arch"], len(meta["classes"]), pretrained=False)
    model.load_state_dict(ck["state_dict"])
    model.to(device).eval()
    return model, meta, device


@torch.no_grad()
def collect_logits(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    logits, ys = [], []
    for x, y in loader:
        logits.append(model(x.to(device)).float().cpu())
        ys.append(y)
    return torch.cat(logits).numpy(), torch.cat(ys).numpy()


def evaluate_checkpoint(ckpt_path: str, splits_csv: str, split: str = "test", batch_size: int = 32) -> dict:
    model, meta, device = load_checkpoint(ckpt_path)
    rows = load_rows(splits_csv, split)
    loader = make_loader(rows, meta["classes"], meta["img_size"], train=False,
                         batch_size=batch_size, num_workers=0, shuffle=False)
    logits, y = collect_logits(model, loader, device)
    probs = softmax(logits / meta.get("temperature", 1.0))
    m = compute_metrics(probs, y, meta["classes"])
    if meta.get("threshold") is not None:
        m["decision"] = decision_stats(probs, y, meta["threshold"])
    return m
