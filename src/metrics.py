from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score, average_precision_score


def multilabel_scores(y_true: np.ndarray, y_prob: np.ndarray, thresholds=None) -> dict:
    if thresholds is None:
        thresholds = np.full(y_true.shape[1], 0.5)

    y_pred = (y_prob >= thresholds.reshape(1, -1)).astype(int)

    return {
        "macro_f1": f1_score(y_true, y_pred, average="macro", zero_division=0),
        "micro_f1": f1_score(y_true, y_pred, average="micro", zero_division=0),
        "per_class_f1": f1_score(y_true, y_pred, average=None, zero_division=0),
        "mAP": average_precision_score(y_true, y_prob, average="macro"),
    }


def optimize_thresholds(y_true: np.ndarray, y_prob: np.ndarray) -> np.ndarray:
    thresholds = []
    grid = np.linspace(0.05, 0.70, 66)

    for c in range(y_true.shape[1]):
        best_t = 0.5
        best_f1 = -1.0
        for t in grid:
            pred = (y_prob[:, c] >= t).astype(int)
            score = f1_score(y_true[:, c], pred, zero_division=0)
            if score > best_f1:
                best_f1 = score
                best_t = t
        thresholds.append(best_t)

    return np.array(thresholds, dtype=np.float32)
