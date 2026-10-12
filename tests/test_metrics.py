import numpy as np

from src.metrics import compute_metrics, decision_stats, expected_calibration_error, select_threshold, softmax


def test_perfect_predictions_have_zero_ece_and_full_accuracy():
    probs = np.eye(3)[[0, 1, 2, 0]] * 0.999 + 0.0005
    y = np.array([0, 1, 2, 0])
    m = compute_metrics(probs, y, ["a", "b", "c"])
    assert m["accuracy"] == 1.0
    assert m["macro_f1"] == 1.0
    assert m["ece"] < 0.01


def test_overconfident_wrong_predictions_raise_ece():
    probs = np.array([[0.99, 0.01], [0.99, 0.01]])
    y = np.array([1, 1])
    assert expected_calibration_error(probs, y) > 0.9


def test_softmax_rows_sum_to_one():
    p = softmax(np.array([[1.0, 2.0, 3.0], [1000.0, 0.0, -1000.0]]))
    assert np.allclose(p.sum(axis=1), 1.0)
    assert np.isfinite(p).all()


def test_select_threshold_meets_target_accuracy():
    # الصور عالية الثقة صحيحة دائمًا، والمنخفضة أحيانًا خاطئة.
    probs = np.array([[0.95, 0.05]] * 10 + [[0.6, 0.4]] * 10)
    y = np.array([0] * 10 + [1] * 5 + [0] * 5)
    thr = select_threshold(probs, y, target_accuracy=0.98)
    assert thr["accepted_accuracy"] >= 0.98
    assert 0.5 < thr["threshold"] <= 0.95


def test_select_threshold_sends_everything_to_review_when_impossible():
    probs = np.array([[0.6, 0.4], [0.6, 0.4]])
    y = np.array([1, 1])
    assert select_threshold(probs, y, target_accuracy=0.98)["threshold"] > 1.0


def test_decision_stats_splits_accept_and_review():
    probs = np.array([[0.9, 0.1], [0.5, 0.5]])
    y = np.array([0, 0])
    s = decision_stats(probs, y, threshold=0.8)
    assert s["auto_accept_rate"] == 0.5
    assert s["sent_to_review_rate"] == 0.5
    assert s["accepted_accuracy"] == 1.0
