"""التنبؤ بصور جديدة. كل صورة تحصل على: الفئة، الثقة، وقرار (accept أو review).

الصور ذات الثقة الأقل من العتبة تُحوَّل للمراجعة البشرية بدلًا من قبولها بتخمين.

الاستخدام:
  python -m src.predict --checkpoint models/production/model.pt new_images/ --out predictions.csv
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path

import torch
from PIL import Image

from .dataset import build_transforms
from .evaluate import load_checkpoint
from .paths import project_root
from .metrics import softmax
from .prepare_data import IMG_EXT


def collect_images(inputs: list[str]) -> list[Path]:
    paths: list[Path] = []
    for s in inputs:
        p = Path(s)
        if p.is_dir():
            paths += sorted(q for q in p.rglob("*") if q.suffix.lower() in IMG_EXT)
        else:
            paths.append(p)
    return paths


@torch.no_grad()
def predict_images(checkpoint: str, paths: list[Path], batch_size: int = 32,
                   threshold: float | None = None, model=None, meta=None, device=None) -> list[dict]:
    """يصنّف الصور. threshold=None تعني استخدام العتبة المحفوظة داخل نقطة الحفظ.

    يُمكن تمرير model/meta/device مُحمّلة مسبقًا (تفعلها واجهة الويب) لتسريع الطلبات.
    """
    if model is None or meta is None or device is None:
        model, meta, device = load_checkpoint(checkpoint)
    tf = build_transforms(meta["img_size"], train=False)
    classes = meta["classes"]
    temperature = meta.get("temperature", 1.0)
    threshold = meta.get("threshold", 1.01) if threshold is None else threshold

    results: list[dict] = []
    for i in range(0, len(paths), batch_size):
        batch_paths, tensors = [], []
        for p in paths[i:i + batch_size]:
            try:
                img = Image.open(p).convert("RGB")
            except Exception as e:  # ملف تالف أو ليس صورة
                results.append({"path": str(p), "predicted": None, "confidence": None,
                                "decision": "error", "error": str(e), "probs": None})
                continue
            batch_paths.append(p)
            tensors.append(tf(img))
        if not tensors:
            continue
        logits = model(torch.stack(tensors).to(device)).float().cpu().numpy()
        probs = softmax(logits / temperature)
        for p, pr in zip(batch_paths, probs):
            k = int(pr.argmax())
            conf = float(pr[k])
            results.append({
                "path": str(p),
                "predicted": classes[k],
                "confidence": conf,
                "decision": "accept" if conf >= threshold else "review",
                "error": "",
                "probs": {c: float(v) for c, v in zip(classes, pr)},
            })
    return results


def detect_model_version(checkpoint: str) -> str:
    """اسم الإصدار من سجل النماذج إن كان الملف هو نموذج الإنتاج، وإلا اسم مجلده."""
    from .feedback import load_registry
    reg = load_registry()
    prod = reg.get("production")
    if prod and Path(checkpoint).resolve() == (project_root(Path.cwd()) / "models" / "production" / "model.pt").resolve():
        return str(prod)
    return Path(checkpoint).parent.name


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("inputs", nargs="+", help="ملفات صور أو مجلدات")
    ap.add_argument("--out", default="predictions.csv")
    ap.add_argument("--threshold", type=float, default=None,
                    help="تجاوز عتبة القبول المحفوظة في النموذج (مثل 0.95 لصرامة أعلى)")
    ap.add_argument("--feedback-db", default=None,
                    help="إذا حُدد، تُسجَّل التنبؤات في قاعدة المراجعة لتحسين النموذج لاحقًا")
    ap.add_argument("--model-version", default=None)
    args = ap.parse_args()

    paths = collect_images(args.inputs)
    results = predict_images(args.checkpoint, paths, threshold=args.threshold)

    with open(args.out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "predicted", "confidence", "decision", "error"])
        w.writeheader()
        for r in results:
            w.writerow({k: r[k] for k in w.fieldnames})
    for r in results:
        conf = "-" if r["confidence"] is None else f"{r['confidence']:.3f}"
        print(f"{r['decision']:7s} {conf:>6s} {r['predicted']}  {r['path']}")
    print(f"\nتم حفظ النتائج في {args.out}")

    if args.feedback_db:
        from .feedback import record_predictions
        version = args.model_version or detect_model_version(args.checkpoint)
        n = record_predictions(args.feedback_db, results, version)
        print(f"سُجّل {n} تنبؤ في {args.feedback_db}")


if __name__ == "__main__":
    main()
