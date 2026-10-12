from collections import Counter

from src.prepare_data import dedupe_rows, stratified_split


def _rows(label, n, prefix):
    return [{"path": f"/x/{label}/{prefix}{i}.jpg", "label": label, "sha256": f"{prefix}{i}"} for i in range(n)]


def test_dedupe_removes_duplicates_and_drops_label_conflicts():
    rows = [
        {"path": "a/1.jpg", "label": "A", "sha256": "h1"},
        {"path": "a/2.jpg", "label": "A", "sha256": "h1"},  # تكرار
        {"path": "b/3.jpg", "label": "A", "sha256": "h2"},
        {"path": "c/4.jpg", "label": "B", "sha256": "h2"},  # تعارض تسمية
    ]
    kept, report = dedupe_rows(rows)
    assert [r["sha256"] for r in kept] == ["h1"]
    actions = Counter(r["action"] for r in report)
    assert actions == {"dropped_duplicate": 1, "dropped_label_conflict": 2}


def test_stratified_split_keeps_class_proportions_and_no_overlap():
    rows = _rows("A", 100, "a") + _rows("B", 40, "b") + _rows("C", 20, "c")
    out = stratified_split(rows, val_frac=0.15, test_frac=0.15, seed=1)

    assert len({r["sha256"] for r in out}) == len(out)  # لا تداخل
    for label, n in [("A", 100), ("B", 40), ("C", 20)]:
        splits = Counter(r["split"] for r in out if r["label"] == label)
        assert splits["test"] == round(n * 0.15)
        assert splits["val"] == round(n * 0.15)
        assert splits["train"] == n - splits["test"] - splits["val"]


def test_split_is_reproducible():
    rows = _rows("A", 30, "a") + _rows("B", 30, "b")
    s1 = stratified_split(rows, 0.2, 0.2, seed=7)
    s2 = stratified_split(rows, 0.2, 0.2, seed=7)
    assert s1 == s2
