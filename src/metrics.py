"""مقاييس التقييم: الدقة، F1، المصفوفة الارتباكية، معايرة الثقة، وعتبة القبول التلقائي.

هذه الوحدة لا تعتمد على torch لتكون سهلة الاختبار.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support


def softmax(logits: np.ndarray) -> np.ndarray:
    z = logits - logits.max(axis=1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=1, keepdims=True)


def expected_calibration_error(probs: np.ndarray, y: np.ndarray, n_bins: int = 10) -> float:
    """الفرق بين الثقة المعلنة والدقة الفعلية. كلما قل كان النموذج أصدق في ثقته."""
    conf = probs.max(axis=1)
    correct = (probs.argmax(axis=1) == y).astype(float)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    ece = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (conf > lo) & (conf <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - conf[mask].mean())
    return float(ece)


def compute_metrics(probs: np.ndarray, y: np.ndarray, classes: list[str]) -> dict:
    pred = probs.argmax(axis=1)
    labels = list(range(len(classes)))
    p, r, f, s = precision_recall_fscore_support(y, pred, labels=labels, zero_division=0)
    per_class = {
        classes[i]: {"precision": float(p[i]), "recall": float(r[i]), "f1": float(f[i]), "support": int(s[i])}
        for i in labels
    }
    return {
        "n": int(len(y)),
        "accuracy": float(accuracy_score(y, pred)),
        "macro_f1": float(np.mean(f)),
        "per_class": per_class,
        "confusion_matrix": confusion_matrix(y, pred, labels=labels).tolist(),
        "ece": expected_calibration_error(probs, y),
    }


def decision_stats(probs: np.ndarray, y: np.ndarray, threshold: float) -> dict:
    """ما الذي يُقبل تلقائيًا عند عتبة معينة، وما الذي يُحوَّل للمراجعة البشرية."""
    conf = probs.max(axis=1)
    accepted = conf >= threshold
    correct = probs.argmax(axis=1) == y
    return {
        "threshold": float(threshold),
        "auto_accept_rate": float(accepted.mean()),
        "accepted_accuracy": float(correct[accepted].mean()) if accepted.any() else None,
        "sent_to_review_rate": float((~accepted).mean()),
    }


def select_threshold(probs: np.ndarray, y: np.ndarray, target_accuracy: float = 0.98) -> dict:
    """أدنى عتبة ثقة تجعل الصور المقبولة تلقائيًا دقتها >= target_accuracy.

    إذا لم توجد عتبة تحقق الهدف، تُعاد عتبة 1.01 أي أن كل شيء يذهب للمراجعة.
    """
    conf = probs.max(axis=1)
    correct = probs.argmax(axis=1) == y
    for t in np.arange(0.50, 1.00, 0.01):
        mask = conf >= t
        if not mask.any():
            break
        acc = float(correct[mask].mean())
        if acc >= target_accuracy:
            return {"threshold": round(float(t), 2), "accepted_accuracy": acc, "coverage": float(mask.mean())}
    return {"threshold": 1.01, "accepted_accuracy": None, "coverage": 0.0}
