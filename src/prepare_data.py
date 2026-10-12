"""تجهيز البيانات: استخراج الأرشيف (اختياري)، التحقق من الصور، إزالة التكرار، والتقسيم الطبقي.

المخرجات داخل data/processed/:
  - splits.csv          : path, label, sha256, split (train / val / test)
  - classes.json        : قائمة أسماء الفئات بالترتيب
  - duplicates_report.csv : الصور المكررة أو المتعارضة التي تم استبعادها

الاستخدام:
  python -m src.prepare_data --archive "من 100 الى 300.rar"
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import random
import sys
from collections import defaultdict
from pathlib import Path

from PIL import Image

from .paths import project_root, rel_to_root

IMG_EXT = {".jpg", ".jpeg", ".png", ".webp", ".bmp"}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def is_valid_image(path: Path) -> bool:
    try:
        with Image.open(path) as im:
            im.verify()
        return True
    except Exception:
        return False


def extract_archive(archive: Path, dest: Path) -> None:
    """يستخرج ملف RAR (يتطلب مكتبة rarfile وأداة unrar أو bsdtar أو 7z على النظام)."""
    try:
        import rarfile  # type: ignore
    except ImportError:
        sys.exit("يلزم تثبيت rarfile: pip install rarfile + أداة unrar/7z على النظام")
    dest.mkdir(parents=True, exist_ok=True)
    with rarfile.RarFile(str(archive)) as rf:
        rf.extractall(str(dest))


def collect_items(raw_dir: Path) -> list[tuple[Path, str]]:
    """كل صورة تأخذ اسم مجلدها المباشر كتسمية (label)."""
    items = []
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            items.append((p, p.parent.name))
    return items


def dedupe_rows(rows: list[dict]) -> tuple[list[dict], list[dict]]:
    """يزيل الصور المكررة بالتجزئة (hash). إذا ظهرت الصورة بتسميتين مختلفتين تُستبعد كليًا."""
    by_hash: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_hash[r["sha256"]].append(r)

    kept: list[dict] = []
    report: list[dict] = []
    for h, group in by_hash.items():
        if len({r["label"] for r in group}) > 1:
            for r in group:
                report.append({"sha256": h, "action": "dropped_label_conflict", "path": r["path"]})
            continue
        kept.append(group[0])
        for r in group[1:]:
            report.append({"sha256": h, "action": "dropped_duplicate", "path": r["path"]})
    return kept, report


def stratified_split(rows: list[dict], val_frac: float, test_frac: float, seed: int) -> list[dict]:
    """تقسيم طبقي: كل فئة تُقسم بالنسب نفسها تقريبًا."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by_label[r["label"]].append(r)

    out: list[dict] = []
    for label in sorted(by_label):
        items = by_label[label][:]
        rng.shuffle(items)
        n = len(items)
        n_test = max(1, round(n * test_frac)) if n >= 3 else 0
        n_val = max(1, round(n * val_frac)) if n >= 3 else 0
        for i, r in enumerate(items):
            if i < n_test:
                split = "test"
            elif i < n_test + n_val:
                split = "val"
            else:
                split = "train"
            out.append({**r, "split": split})
    return out


def index_by_hash(raw_dir: Path) -> dict[str, Path]:
    """فهرس الصور الموجودة فعليًا حسب بصمتها sha256."""
    index: dict[str, Path] = {}
    for p in sorted(raw_dir.rglob("*")):
        if p.is_file() and p.suffix.lower() in IMG_EXT:
            index.setdefault(sha256_file(p), p)
    return index


def rebase_splits(splits_csv: Path, raw_dir: Path, report_csv: Path | None = None) -> dict:
    """يعيد كتابة مسارات splits.csv لتصبح نسبية عن جذر المشروع.

    المطابقة تتم بالبصمة sha256 (أو باسم الملف للصور غير الصالحة)، لذلك:
      * لا يتغيّر التقسيم (train/val/test) ولا التسميات إطلاقًا،
      * تعمل نفس الملفات بعد نقل المشروع إلى جهاز أو نظام آخر.
    """
    with open(splits_csv, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    index = index_by_hash(raw_dir)
    by_name = {p.name: p for p in index.values()}

    def rebase(r: dict) -> bool:
        hit = index.get(r.get("sha256", "")) or (by_name.get(Path(r["path"]).name) if r.get("path") else None)
        if hit is None:
            return False
        r["path"] = rel_to_root(hit)
        return True

    updated = sum(rebase(r) for r in rows)
    with open(splits_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "sha256", "split"])
        w.writeheader()
        w.writerows(rows)

    report_updated = 0
    if report_csv is None:
        report_csv = splits_csv.parent / "duplicates_report.csv"
    if report_csv.exists():
        with open(report_csv, newline="", encoding="utf-8") as f:
            rep = list(csv.DictReader(f))
        for r in rep:
            if rebase(r):
                report_updated += 1
        with open(report_csv, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=["sha256", "action", "path"])
            w.writeheader()
            w.writerows(rep)

    return {"rows": len(rows), "rebased": updated, "missing": len(rows) - updated,
            "report_rebased": report_updated, "root": str(project_root(Path.cwd()))}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw-dir", default="data/raw")
    ap.add_argument("--rebase", action="store_true",
                    help="أعد كتابة مسارات splits.csv الحالية لتصبح نسبية (بدون إعادة تقسيم)")
    ap.add_argument("--archive", default=None, help="ملف RAR/ZIP يُستخرج داخل raw-dir أولاً")
    ap.add_argument("--out-dir", default="data/processed")
    ap.add_argument("--val-frac", type=float, default=0.15)
    ap.add_argument("--test-frac", type=float, default=0.15)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    raw = Path(args.raw_dir)
    out = Path(args.out_dir)
    if args.rebase:
        splits_csv = out / "splits.csv"
        if not splits_csv.exists():
            sys.exit(f"لا يوجد ملف {splits_csv} لإعادة ضبط مساراته")
        info = rebase_splits(splits_csv, raw)
        print(f"أُعيد ضبط {info['rebased']}/{info['rows']} مسارًا بالنسبة لجذر المشروع: {info['root']}")
        if info["missing"]:
            print(f"تحذير: {info['missing']} صف لم تُعثر على صورته داخل {raw}")
        return

    if args.archive:
        extract_archive(Path(args.archive), raw)

    items = collect_items(raw)
    if not items:
        sys.exit(f"لا توجد صور داخل {raw}")

    rows, invalid = [], []
    for p, label in items:
        if not is_valid_image(p):
            invalid.append(rel_to_root(p))
            continue
        rows.append({"path": rel_to_root(p), "label": label, "sha256": sha256_file(p)})

    rows, report = dedupe_rows(rows)
    for p in invalid:
        report.append({"sha256": "", "action": "dropped_invalid_image", "path": p})

    classes = sorted({r["label"] for r in rows})
    splits = stratified_split(rows, args.val_frac, args.test_frac, args.seed)

    out.mkdir(parents=True, exist_ok=True)
    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["path", "label", "sha256", "split"])
        w.writeheader()
        w.writerows(splits)
    with open(out / "duplicates_report.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["sha256", "action", "path"])
        w.writeheader()
        w.writerows(report)
    (out / "classes.json").write_text(json.dumps(classes, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"الصور الصالحة: {len(rows)} | المستبعدة: {len(report)}")
    for split in ("train", "val", "test"):
        counts = defaultdict(int)
        for r in splits:
            if r["split"] == split:
                counts[r["label"]] += 1
        print(f"{split:5s}: {dict(sorted(counts.items()))}")


if __name__ == "__main__":
    main()
