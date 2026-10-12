import csv
import json

from src.feedback import (
    build_feedback_splits,
    decide_promotion,
    record_predictions,
    status,
    submit_label,
    pending_reviews,
)


def _write_base(tmp_path):
    data = tmp_path / "data"
    data.mkdir()
    (data / "classes.json").write_text(json.dumps(["A", "B"]), encoding="utf-8")
    imgs = []
    for i, split in enumerate(["train", "train", "val", "test"]):
        p = tmp_path / f"base{i}.jpg"
        p.write_bytes(b"x")
        imgs.append((str(p), "A" if i % 2 == 0 else "B", f"sha{i}", split))
    with open(data / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["path", "label", "sha256", "split"])
        w.writerows(imgs)
    return data / "splits.csv"


def _insert(db, sha, predicted, decision, img):
    from src.feedback import connect, now_iso
    conn = connect(db)
    conn.execute(
        "INSERT INTO predictions (created_at, sha256, image_path, model_version, predicted, confidence, decision)"
        " VALUES (?,?,?,?,?,?,?)",
        (now_iso(), sha, str(img), "v1", predicted, 0.5, decision),
    )
    conn.commit()
    conn.close()


def test_record_and_review_flow(tmp_path):
    db = str(tmp_path / "fb.db")
    img = tmp_path / "new.jpg"
    img.write_bytes(b"new-image-bytes")
    results = [{"path": str(img), "predicted": "A", "confidence": 0.4, "decision": "review", "error": ""}]
    assert record_predictions(db, results, "v1", images_dir=str(tmp_path / "imgs")) == 1

    pend = pending_reviews(db)
    assert len(pend) == 1
    submit_label(db, pend[0]["id"], "B", ["A", "B"], reviewer="me")
    assert pending_reviews(db) == []

    st = status(db)
    assert st[0]["reviewed"] == 1
    assert st[0]["accuracy_on_reviewed"] == 0.0


def test_submit_rejects_unknown_label(tmp_path):
    db = str(tmp_path / "fb.db")
    img = tmp_path / "x.jpg"
    img.write_bytes(b"1")
    record_predictions(db, [{"path": str(img), "predicted": "A", "confidence": 0.3,
                             "decision": "review", "error": ""}], "v1", images_dir=str(tmp_path / "i"))
    try:
        submit_label(db, 1, "Z", ["A", "B"])
    except ValueError:
        return
    raise AssertionError("expected ValueError")


def test_feedback_never_leaks_into_val_or_test(tmp_path):
    base = _write_base(tmp_path)
    db = str(tmp_path / "fb.db")
    # صورة في test (sha3) وصورة جديدة (sha9)
    _insert(db, "sha3", "A", "review", tmp_path / "t.jpg")
    _insert(db, "sha9", "A", "review", tmp_path / "n.jpg")
    conn_rows = __import__("src.feedback", fromlist=["connect"]).connect(db)
    conn_rows.execute("UPDATE predictions SET true_label='B' WHERE sha256 IN ('sha3','sha9')")
    conn_rows.commit()
    conn_rows.close()

    out = tmp_path / "run" / "splits.csv"
    info = build_feedback_splits(db, str(base), str(out))
    assert info["skipped_locked"] == 1
    assert info["new_labeled"] == 1
    with open(out, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 5
    assert [r["sha256"] for r in rows if r["split"] == "test"] == ["sha3"]
    assert (out.parent / "classes.json").exists()


def test_promotion_rules():
    prod = {"macro_f1": 0.80, "per_class": {"A": {"recall": 0.9}, "B": {"recall": 0.7}}}
    better = {"macro_f1": 0.85, "per_class": {"A": {"recall": 0.92}, "B": {"recall": 0.75}}}
    worse_f1 = {"macro_f1": 0.79, "per_class": {"A": {"recall": 0.99}, "B": {"recall": 0.99}}}
    recall_drop = {"macro_f1": 0.90, "per_class": {"A": {"recall": 0.80}, "B": {"recall": 0.99}}}

    assert decide_promotion(prod, better)[0] is True
    assert decide_promotion(prod, worse_f1)[0] is False
    ok, reason = decide_promotion(prod, recall_drop)
    assert ok is False and "A" in reason
