"""تحميل نقاط الحفظ (checkpoints) وتقييمها على أي تقسيم (val / test).

الاستخدام:
  python -m src.evaluate --checkpoint runs/baseline/model.pt --split test
  python -m src.evaluate --checkpoint runs/baseline/model.pt --split all --out report.json
  python -m src.evaluate --checkpoint runs/baseline/model.pt --split val --sweep
"""

from __future__ import annotations

from pathlib import Path

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


def describe_checkpoint(ckpt_path: str) -> dict:
    """يعرض محتوى نقطة الحفظ (المعمارية، الفئات، العتبة، حرارة المعايرة) بدون صور."""
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    meta = ck.get("meta", {})
    return {
        "checkpoint": ckpt_path,
        "arch": meta.get("arch"),
        "img_size": meta.get("img_size"),
        "classes": meta.get("classes"),
        "temperature": meta.get("temperature"),
        "threshold": meta.get("threshold"),
        "target_accuracy": meta.get("target_accuracy"),
        "tensors": len(ck.get("state_dict", {})),
    }


def main() -> None:
    """تقييم نقطة حفظ على أي تقسيم: python -m src.evaluate --checkpoint runs/baseline/model.pt --split test"""
    import argparse
    import json

    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--splits", default="data/processed/splits.csv")
    ap.add_argument("--split", default="test", choices=["train", "val", "test", "all"])
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--out", default=None, help="احفظ التقرير JSON في هذا المسار")
    ap.add_argument("--info", action="store_true", help="اعرض بيانات نقطة الحفظ فقط دون تقييم")
    ap.add_argument("--sweep", action="store_true",
                    help="اعرض دقة الصور المقبولة ونسبة التحويل للمراجعة عند كل عتبة من 0.50 إلى 0.99")
    args = ap.parse_args()

    if args.info:
        print(json.dumps(describe_checkpoint(args.checkpoint), ensure_ascii=False, indent=2))
        return

    if args.sweep:
        if args.split == "all":
            ap.error("--sweep يحتاج تقسيمًا واحدًا: استخدم --split val أو --split test")
        sw = threshold_sweep(args.checkpoint, args.splits, args.split, args.batch_size)
        print(f"التقسيم: {sw['split']} | عدد الصور: {sw['n']} | العتبة الحالية: {sw['current_threshold']}\n")
        print(f"{'العتبة':>6}  {'نسبة القبول':>12}  {'دقة المقبولات':>14}  {'نسبة المراجعة':>13}")
        for r in sw["sweep"]:
            acc = "-" if r["accepted_accuracy"] is None else f"{r['accepted_accuracy']:.1%}"
            print(f"{r['threshold']:>6.2f}  {r['accept_rate']:>12.1%}  {acc:>14}  {r['review_rate']:>13.1%}")
        if args.out:
            out = Path(args.out)
            out.parent.mkdir(parents=True, exist_ok=True)
            out.write_text(json.dumps(sw, ensure_ascii=False, indent=2), encoding="utf-8")
            print(f"\nحُفظ التقرير في {out}")
        return

    if args.split == "all":
        report = {s: evaluate_checkpoint(args.checkpoint, args.splits, s, args.batch_size)
                  for s in ("train", "val", "test")}
    else:
        report = evaluate_checkpoint(args.checkpoint, args.splits, args.split, args.batch_size)

    text = json.dumps(report, ensure_ascii=False, indent=2)
    print(text)
    if args.out:
        out = Path(args.out)
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(text, encoding="utf-8")
        print(f"\nحُفظ التقرير في {out}")


@torch.no_grad()
def collect_probs(ckpt_path: str, splits_csv: str, split: str = "test",
                  batch_size: int = 32) -> tuple[np.ndarray, np.ndarray, dict]:
    """يُعيد (الاحتمالات بعد المعايرة، التسميات الحقيقية، بيانات نقطة الحفظ)."""
    model, meta, device = load_checkpoint(ckpt_path)
    rows = load_rows(splits_csv, split)
    loader = make_loader(rows, meta["classes"], meta["img_size"], train=False,
                         batch_size=batch_size, num_workers=0, shuffle=False)
    logits, y = collect_logits(model, loader, device)
    return softmax(logits / meta.get("temperature", 1.0)), y, meta


def evaluate_checkpoint(ckpt_path: str, splits_csv: str, split: str = "test", batch_size: int = 32) -> dict:
    probs, y, meta = collect_probs(ckpt_path, splits_csv, split, batch_size)
    m = compute_metrics(probs, y, meta["classes"])
    if meta.get("threshold") is not None:
        m["decision"] = decision_stats(probs, y, meta["threshold"])
    return m


def threshold_sweep(ckpt_path: str, splits_csv: str, split: str = "val",
                    batch_size: int = 32) -> dict:
    """كم تصبح دقة المقبولات ونسبة المراجعة عند كل عتبة ثقة؟

    يُستخدم لاختيار نقطة تشغيل آمنة: كلما رفعت العتبة قلّ ما يُقبل تلقائيًا
    وارتفعت دقة المقبولات.
    """
    probs, y, meta = collect_probs(ckpt_path, splits_csv, split, batch_size)
    rows = []
    for t in [round(float(x), 2) for x in np.arange(0.50, 1.00, 0.01)]:
        d = decision_stats(probs, y, t)
        rows.append({"threshold": t, "accept_rate": d["auto_accept_rate"],
                     "accepted_accuracy": d["accepted_accuracy"], "review_rate": d["sent_to_review_rate"]})
    return {"split": split, "n": int(len(y)), "current_threshold": meta.get("threshold"), "sweep": rows}


if __name__ == "__main__":
    main()
