"""واجهة ويب للتنبؤ والمراجعة البشرية.

تشغيل:
  python -m src.app --port 7860

الصفحات:
  /           رفع صور + التنبؤ بها + بطاقة النموذج وإحصاءات المراجعة
  /review     طابور المراجعة: الصور غير المصنّفة (الأقل ثقة أولًا) مع أزرار التصنيف
  /image/<id> عرض صورة مسجّلة في قاعدة المراجعة

واجهة برمجية (JSON):
  POST /api/predict  يرفع صورًا ويُعيد النتائج (يمكن تمرير threshold في النموذج)
  GET  /api/status   ملخّص الإصدار الإنتاجي وإحصاءات المراجعة
"""

from __future__ import annotations

import argparse
import threading
import uuid
from pathlib import Path

import torch
from flask import Flask, jsonify, redirect, render_template, request, send_file, url_for
from werkzeug.utils import secure_filename

from .evaluate import load_checkpoint
from .feedback import (DEFAULT_DB, DEFAULT_IMAGES, load_registry, pending_reviews,
                       record_predictions_ids, row_by_id, status, submit_label, unreviewed)
from .paths import project_root
from .predict import predict_images
from .prepare_data import IMG_EXT

ROOT = project_root(Path(__file__).resolve())
TEMPLATES = ROOT / "templates"
DEFAULT_CKPT = ROOT / "models" / "production" / "model.pt"
UPLOADS = ROOT / "feedback" / "uploads"

app = Flask(__name__, template_folder=str(TEMPLATES))
app.config["MAX_CONTENT_LENGTH"] = 64 * 1024 * 1024  # 64 ميغابايت لكل طلب

STATE: dict = {"ready": False, "error": None, "checkpoint": None, "version": None,
               "model": None, "meta": None, "device": None, "db": DEFAULT_DB}
LOCK = threading.Lock()


# ---------- تهيئة النموذج ----------

def load_state(checkpoint: str | None = None, db: str = DEFAULT_DB) -> dict:
    """يحمّل نموذج الإنتاج مرة واحدة ويحتفظ به في الذاكرة."""
    ck = Path(checkpoint) if checkpoint else DEFAULT_CKPT
    STATE["db"] = db
    if not ck.exists():
        STATE.update(ready=False, error=f"لا توجد نقطة حفظ في {ck}. سجّل نموذجًا أولًا: "
                                        f"python -m src.feedback register --checkpoint runs/baseline/model.pt --version v1")
        return STATE
    model, meta, device = load_checkpoint(str(ck))
    reg = load_registry()
    STATE.update(ready=True, error=None, checkpoint=str(ck), model=model, meta=meta,
                 device=device, version=reg.get("production") or ck.parent.name)
    return STATE


def _meta_view() -> dict:
    m = STATE.get("meta") or {}
    return {
        "arch": m.get("arch"), "img_size": m.get("img_size"), "classes": m.get("classes", []),
        "temperature": m.get("temperature"), "threshold": m.get("threshold"),
        "target_accuracy": m.get("target_accuracy"),
    }


def _registry_view() -> dict:
    reg = load_registry()
    prod = reg.get("production")
    info = reg.get("versions", {}).get(prod) if prod else None
    return {"production": prod, "info": info, "versions": list(reg.get("versions", {}))}


# ---------- التنبؤ ----------

def _save_uploads(files) -> list[Path]:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    saved: list[Path] = []
    for f in files:
        if not f or not f.filename:
            continue
        name = f"{uuid.uuid4().hex}_{secure_filename(f.filename)}"
        p = UPLOADS / name
        f.save(p)
        if p.suffix.lower() in IMG_EXT:
            saved.append(p)
        else:
            p.unlink(missing_ok=True)
    return saved


def _run(files, threshold: float | None):
    if not STATE["ready"]:
        raise RuntimeError(STATE["error"] or "النموذج غير محمّل")
    paths = _save_uploads(files)
    if not paths:
        return []
    with LOCK:  # الاستدلال غير آمن مع الخيوط المتعددة
        results = predict_images(STATE["checkpoint"], paths, threshold=threshold,
                                 model=STATE["model"], meta=STATE["meta"], device=STATE["device"])
    ids = record_predictions_ids(STATE["db"], results, STATE["version"], images_dir=DEFAULT_IMAGES)
    return [{"id": i, **r} for i, r in zip(ids, results)]


# ---------- الصفحات ----------

@app.get("/")
def index():
    return render_template("index.html", state=STATE, meta=_meta_view(), reg=_registry_view(),
                           stats=status(STATE["db"]) if Path(STATE["db"]).exists() else [],
                           pending=len(pending_reviews(STATE["db"], limit=1000)) if Path(STATE["db"]).exists() else 0,
                           results=None, threshold=None)


@app.post("/predict")
def predict():
    threshold = request.form.get("threshold", type=float)
    try:
        results = _run(request.files.getlist("images"), threshold)
        err = None
    except Exception as e:  # ملف تالف أو نموذج غير محمّل
        results, err = [], str(e)
    return render_template("index.html", state=STATE, meta=_meta_view(), reg=_registry_view(),
                           stats=status(STATE["db"]) if Path(STATE["db"]).exists() else [],
                           pending=len(pending_reviews(STATE["db"], limit=1000)) if Path(STATE["db"]).exists() else 0,
                           results=results, threshold=threshold, error=err)


@app.get("/review")
def review():
    db = STATE["db"]
    limit = request.args.get("limit", default=12, type=int)
    only_review = request.args.get("only_review", default="1") == "1"
    rows = unreviewed(db, limit=limit, only_review=only_review) if Path(db).exists() else []
    remaining = len(unreviewed(db, limit=10000)) if Path(db).exists() else 0
    return render_template("review.html", rows=rows, classes=(STATE["meta"] or {}).get("classes", []),
                           limit=limit, only_review=only_review, remaining=remaining,
                           total_pending=len(pending_reviews(db, limit=10000)) if Path(db).exists() else 0)


@app.post("/review/<int:row_id>")
def label_row(row_id: int):
    label = request.form.get("label", "")
    reviewer = request.form.get("reviewer", "web") or "web"
    classes = (STATE["meta"] or {}).get("classes", [])
    if label in classes:
        submit_label(STATE["db"], row_id, label, classes, reviewer=reviewer)
    return redirect(url_for("review", limit=request.form.get("limit", 12),
                            only_review="1" if request.form.get("only_review", "1") == "1" else "0"))


@app.get("/image/<int:row_id>")
def image(row_id: int):
    r = row_by_id(STATE["db"], row_id)
    if r is None:
        return "الصورة غير موجودة", 404
    p = Path(r["image_path"])
    if not p.exists():
        return "ملف الصورة مفقود على القرص", 404
    return send_file(p)


# ---------- واجهة برمجية ----------

@app.post("/api/predict")
def api_predict():
    threshold = request.form.get("threshold", type=float)
    try:
        results = _run(request.files.getlist("images"), threshold)
    except Exception as e:
        return jsonify({"error": str(e)}), 400
    return jsonify({"model_version": STATE["version"], "threshold": threshold,
                    "results": [{k: v for k, v in r.items() if k != "probs"} | {"probs": r.get("probs")}
                                for r in results]})


@app.get("/api/status")
def api_status():
    return jsonify({"ready": STATE["ready"], "error": STATE["error"],
                    "checkpoint": STATE["checkpoint"], "model_version": STATE["version"],
                    "meta": _meta_view(), "registry": _registry_view(),
                    "feedback": status(STATE["db"]) if Path(STATE["db"]).exists() else []})


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--checkpoint", default=None, help="نقطة الحفظ (افتراضيًا models/production/model.pt)")
    ap.add_argument("--db", default=DEFAULT_DB)
    ap.add_argument("--host", default="0.0.0.0")
    ap.add_argument("--port", type=int, default=7860)
    ap.add_argument("--debug", action="store_true")
    args = ap.parse_args()

    st = load_state(args.checkpoint, args.db)
    device_name = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"النموذج: {st['checkpoint']} | الإصدار: {st.get('version')} | الجهاز: {device_name}")
    if not st["ready"]:
        print(f"تحذير: {st['error']}")
    app.run(host=args.host, port=args.port, debug=args.debug, threaded=True)


if __name__ == "__main__":
    main()
