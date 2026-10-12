"""حلقة التحسين الذاتي المُراقَبة:

1. كل تنبؤ يُسجَّل في قاعدة SQLite مع نسخة من الصورة (predict --feedback-db).
2. الصور التي يحوّلها النموذج للمراجعة (ثقة منخفضة) يصنّفها إنسان: `pending` ثم `label`.
3. `status` يعرض دقة كل إصدار من النموذج على الصور الحقيقية التي تمت مراجعتها.
4. `register` يسجل نموذجًا كإصدار إنتاجي مع مقاييسه على تقسيم الاختبار الثابت.
5. `retrain` يدرّب مرشحًا جديدًا على التدريب الأصلي + التصنيفات المؤكدة، ثم:
   - يقيّمه ومعه النموذج الحالي على تقسيم الاختبار الثابت نفسه.
   - لا يُرقّى إلا إذا تحسن macro-F1 ولم تنخفض recall لأي فئة بأكثر من الحد المسموح.

قاعدة مهمة: صور الاختبار والتحقق لا تدخل أبدًا في التدريب (حتى لو أصبحت مُصنّفة)،
وإلا لن يكون التقييم موثوقًا.

الأوامر:
  python -m src.feedback register --checkpoint runs/baseline/model.pt --splits data/processed/splits.csv --version v1
  python -m src.feedback pending --limit 20
  python -m src.feedback label --id 5 --label "من هاتف"
  python -m src.feedback status
  python -m src.feedback retrain --min-new 10
"""

from __future__ import annotations

import argparse
import csv
import json
import shutil
import sqlite3
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path

from .prepare_data import sha256_file

DEFAULT_DB = "feedback/feedback.db"
DEFAULT_IMAGES = "feedback/images"
REGISTRY = "models/registry.json"
SCHEMA = """
CREATE TABLE IF NOT EXISTS predictions (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at    TEXT NOT NULL,
    sha256        TEXT NOT NULL,
    image_path    TEXT NOT NULL,
    model_version TEXT NOT NULL,
    predicted     TEXT NOT NULL,
    confidence    REAL NOT NULL,
    decision      TEXT NOT NULL,
    true_label    TEXT,
    reviewed_at   TEXT,
    reviewer      TEXT
);
CREATE INDEX IF NOT EXISTS idx_pred_sha ON predictions(sha256);
"""


def now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def connect(db_path: str) -> sqlite3.Connection:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(SCHEMA)
    return conn


def record_predictions(db_path: str, results: list[dict], model_version: str,
                       images_dir: str = DEFAULT_IMAGES) -> int:
    conn = connect(db_path)
    img_dir = Path(images_dir)
    img_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in results:
        if r["decision"] == "error":
            continue
        src = Path(r["path"])
        sha = sha256_file(src)
        dst = img_dir / f"{sha}{src.suffix.lower()}"
        if not dst.exists():
            shutil.copy2(src, dst)
        conn.execute(
            "INSERT INTO predictions (created_at, sha256, image_path, model_version, predicted, "
            "confidence, decision) VALUES (?,?,?,?,?,?,?)",
            (now_iso(), sha, str(dst.resolve()), model_version, r["predicted"], r["confidence"], r["decision"]),
        )
        n += 1
    conn.commit()
    conn.close()
    return n


def pending_reviews(db_path: str, limit: int = 20) -> list[sqlite3.Row]:
    conn = connect(db_path)
    rows = conn.execute(
        "SELECT * FROM predictions WHERE decision='review' AND true_label IS NULL "
        "ORDER BY confidence ASC LIMIT ?", (limit,)
    ).fetchall()
    conn.close()
    return rows


def submit_label(db_path: str, row_id: int, label: str, classes: list[str], reviewer: str = "") -> None:
    if label not in classes:
        raise ValueError(f"التسمية {label!r} غير موجودة. الفئات: {classes}")
    conn = connect(db_path)
    cur = conn.execute(
        "UPDATE predictions SET true_label=?, reviewed_at=?, reviewer=? WHERE id=?",
        (label, now_iso(), reviewer, row_id),
    )
    conn.commit()
    conn.close()
    if cur.rowcount == 0:
        raise ValueError(f"لا يوجد تنبؤ بالمعرف {row_id}")


def status(db_path: str) -> list[dict]:
    """لكل إصدار: عدد التنبؤات، نسبة الإحالة للمراجعة، ودقته على الصور المراجعة بشريًا."""
    conn = connect(db_path)
    rows = conn.execute("SELECT * FROM predictions").fetchall()
    conn.close()
    by_ver: dict[str, dict] = defaultdict(lambda: {"total": 0, "review": 0, "reviewed": 0, "correct": 0})
    for r in rows:
        s = by_ver[r["model_version"]]
        s["total"] += 1
        s["review"] += r["decision"] == "review"
        if r["true_label"] is not None:
            s["reviewed"] += 1
            s["correct"] += r["predicted"] == r["true_label"]
    out = []
    for ver, s in by_ver.items():
        out.append({
            "version": ver,
            "total": s["total"],
            "review_rate": s["review"] / s["total"],
            "reviewed": s["reviewed"],
            "accuracy_on_reviewed": (s["correct"] / s["reviewed"]) if s["reviewed"] else None,
        })
    return out


def build_feedback_splits(db_path: str, base_splits_csv: str, out_csv: str) -> dict:
    """ينشئ splits جديدًا: التدريب الأصلي + التصنيفات البشرية المؤكدة. val/test لا تتغير."""
    base_csv = Path(base_splits_csv)
    with open(base_csv, newline="", encoding="utf-8") as f:
        base = list(csv.DictReader(f))
    locked = {r["sha256"] for r in base if r["split"] in ("val", "test")}
    train_hashes = {r["sha256"] for r in base if r["split"] == "train"}

    conn = connect(db_path)
    labeled = conn.execute(
        "SELECT * FROM predictions WHERE true_label IS NOT NULL ORDER BY reviewed_at ASC"
    ).fetchall()
    conn.close()

    new_rows: dict[str, dict] = {}
    skipped_locked = 0
    for r in labeled:
        sha = r["sha256"]
        if sha in locked:
            skipped_locked += 1
            continue
        if sha in train_hashes:
            continue
        new_rows[sha] = {"path": r["image_path"], "label": r["true_label"], "sha256": sha, "split": "train"}

    out = Path(out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "sha256", "split"])
        w.writeheader()
        for r in base:
            w.writerow({k: r[k] for k in w.fieldnames})
        w.writerows(new_rows.values())
    shutil.copy2(base_csv.parent / "classes.json", out.parent / "classes.json")
    return {"new_labeled": len(new_rows), "skipped_locked": skipped_locked}


def decide_promotion(prod: dict, cand: dict, min_gain: float = 0.0, max_recall_drop: float = 0.05) -> tuple[bool, str]:
    if cand["macro_f1"] < prod["macro_f1"] + min_gain:
        return False, f"macro_f1 {cand['macro_f1']:.4f} < {prod['macro_f1']:.4f} + {min_gain}"
    for cls, pc in prod["per_class"].items():
        cr = cand["per_class"][cls]["recall"]
        if cr < pc["recall"] - max_recall_drop:
            return False, f"recall انخفض لفئة {cls}: {cr:.3f} < {pc['recall']:.3f}"
    return True, "المرشح تفوق أو تساوى مع الإنتاجي دون تراجع في أي فئة"


# ---------- سجل الإصدارات ----------

def load_registry(path: str = REGISTRY) -> dict:
    p = Path(path)
    if not p.exists():
        return {"production": None, "versions": {}}
    return json.loads(p.read_text(encoding="utf-8"))


def save_registry(reg: dict, path: str = REGISTRY) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(reg, ensure_ascii=False, indent=2), encoding="utf-8")


def _summary(m: dict) -> dict:
    return {"accuracy": m["accuracy"], "macro_f1": m["macro_f1"], "ece": m["ece"],
            "per_class": m["per_class"], "decision": m.get("decision")}


def _install_version(src_ckpt: Path, version: str) -> Path:
    dest = Path("models") / "versions" / version
    dest.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, dest / "model.pt")
    prod = Path("models") / "production"
    prod.mkdir(parents=True, exist_ok=True)
    shutil.copy2(src_ckpt, prod / "model.pt")
    return dest / "model.pt"


def cmd_register(args) -> None:
    from .evaluate import evaluate_checkpoint
    reg = load_registry()
    if args.version in reg["versions"]:
        raise SystemExit(f"الإصدار {args.version} موجود مسبقًا")
    m = evaluate_checkpoint(args.checkpoint, args.splits, "test")
    _install_version(Path(args.checkpoint), args.version)
    reg["versions"][args.version] = {
        "splits_csv": str(Path(args.splits).resolve()), "created": now_iso(),
        "source": args.checkpoint, "test": _summary(m), "promoted": True, "reason": "baseline",
    }
    reg["production"] = args.version
    save_registry(reg)
    print(f"سُجل {args.version} كإصدار إنتاجي | test accuracy={m['accuracy']:.4f} macro_f1={m['macro_f1']:.4f}")


def cmd_retrain(args) -> None:
    import torch
    from .evaluate import evaluate_checkpoint
    from .train import run_training

    reg = load_registry()
    if not reg["production"]:
        raise SystemExit("لا يوجد إصدار إنتاجي. شغّل register أولًا.")
    prod_name = reg["production"]
    prod = reg["versions"][prod_name]
    prod_ckpt = Path("models") / "production" / "model.pt"
    prod_meta = torch.load(prod_ckpt, map_location="cpu", weights_only=True)["meta"]

    run_dir = Path("runs") / f"retrain_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    run_dir.mkdir(parents=True, exist_ok=True)
    info = build_feedback_splits(args.db, prod["splits_csv"], str(run_dir / "splits.csv"))
    print(f"تصنيفات بشرية جديدة للتدريب: {info['new_labeled']} (مستبعد لأنه في val/test: {info['skipped_locked']})")
    if info["new_labeled"] < args.min_new:
        print(f"أقل من الحد الأدنى ({args.min_new}). لا يوجد إعادة تدريب الآن.")
        return

    cfg = {
        "splits_csv": str(run_dir / "splits.csv"), "out_dir": str(run_dir),
        "arch": prod_meta["arch"], "img_size": prod_meta["img_size"], "epochs": args.epochs,
        "batch_size": args.batch_size, "lr": args.lr, "weight_decay": 1e-4, "patience": args.patience,
        "target_accuracy": prod_meta.get("target_accuracy", 0.98), "seed": args.seed,
        "num_workers": args.num_workers, "pretrained": not args.no_pretrained,
    }
    run_training(cfg)

    base_csv = prod["splits_csv"]
    cand_m = evaluate_checkpoint(str(run_dir / "model.pt"), base_csv, "test")
    prod_m = evaluate_checkpoint(str(prod_ckpt), base_csv, "test")
    ok, reason = decide_promotion(prod_m, cand_m, args.min_gain, args.max_recall_drop)

    version = f"v{len(reg['versions']) + 1}"
    if ok:
        _install_version(run_dir / "model.pt", version)
        reg["production"] = version
    reg["versions"][version] = {
        "splits_csv": str(run_dir / "splits.csv"), "created": now_iso(), "source": str(run_dir),
        "test": _summary(cand_m), "production_before": _summary(prod_m),
        "promoted": ok, "reason": reason, "new_labeled": info["new_labeled"],
    }
    save_registry(reg)
    verdict = "تمت ترقيته للإنتاج" if ok else "لم يُرقَّ"
    print(f"{version}: {verdict} | {reason}")
    print(f"macro_f1: إنتاجي={prod_m['macro_f1']:.4f} مرشح={cand_m['macro_f1']:.4f}")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--db", default=DEFAULT_DB)
    sub = ap.add_subparsers(dest="cmd", required=True)

    p = sub.add_parser("register")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--splits", default="data/processed/splits.csv")
    p.add_argument("--version", default="v1")

    p = sub.add_parser("pending")
    p.add_argument("--limit", type=int, default=20)

    p = sub.add_parser("label")
    p.add_argument("--id", type=int, required=True)
    p.add_argument("--label", required=True)
    p.add_argument("--reviewer", default="")

    sub.add_parser("status")

    p = sub.add_parser("retrain")
    p.add_argument("--min-new", type=int, default=10)
    p.add_argument("--epochs", type=int, default=30)
    p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--patience", type=int, default=6)
    p.add_argument("--min-gain", type=float, default=0.0)
    p.add_argument("--max-recall-drop", type=float, default=0.05)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num-workers", type=int, default=2)
    p.add_argument("--no-pretrained", action="store_true")

    args = ap.parse_args()
    if args.cmd == "register":
        cmd_register(args)
    elif args.cmd == "pending":
        for r in pending_reviews(args.db, args.limit):
            print(f"id={r['id']} conf={r['confidence']:.3f} predicted={r['predicted']} image={r['image_path']}")
    elif args.cmd == "label":
        reg = load_registry()
        prod = reg["versions"].get(reg["production"] or "", {})
        classes = json.loads((Path(prod["splits_csv"]).parent / "classes.json").read_text(encoding="utf-8")) if prod else []
        submit_label(args.db, args.id, args.label, classes, args.reviewer)
        print(f"تم تسجيل التصنيف للمعرف {args.id}")
    elif args.cmd == "status":
        for s in status(args.db):
            acc = "-" if s["accuracy_on_reviewed"] is None else f"{s['accuracy_on_reviewed']:.3f}"
            print(f"{s['version']}: total={s['total']} review_rate={s['review_rate']:.2%} "
                  f"reviewed={s['reviewed']} accuracy_on_reviewed={acc}")
    elif args.cmd == "retrain":
        cmd_retrain(args)


if __name__ == "__main__":
    main()
