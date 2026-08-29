"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/metrics/prediction.py
Phase    : Phase 10
Status   : IMPLEMENTED

Accuracy, precision, recall, Macro-F1, MCC, and pre/post-drift metrics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
)


def compute_classification_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Compute primary classification metrics on binary/multiclass predictions.

    Metrics computed:
    - accuracy: Fraction of correct predictions.
    - precision: Macro-averaged precision across classes.
    - recall: Macro-averaged recall across classes.
    - f1: Macro-averaged F1-score across classes.
    - mcc: Matthews Correlation Coefficient (-1.0 to +1.0).
    - confusion_matrix: 2x2 dictionary with tp, fp, fn, tn (for binary).
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    if len(y_t) == 0:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "f1": 0.0,
            "mcc": 0.0,
            "sample_count": 0,
            "confusion_matrix": {"tp": 0, "fp": 0, "fn": 0, "tn": 0},
        }

    acc = float(accuracy_score(y_t, y_p))
    prec = float(precision_score(y_t, y_p, average="macro", zero_division=0))
    rec = float(recall_score(y_t, y_p, average="macro", zero_division=0))
    f1 = float(f1_score(y_t, y_p, average="macro", zero_division=0))

    try:
        mcc = float(matthews_corrcoef(y_t, y_p))
    except Exception:
        mcc = 0.0

    cm = confusion_matrix(y_t, y_p, labels=[0, 1])
    tn, fp, fn, tp = (int(cm[0, 0]), int(cm[0, 1]), int(cm[1, 0]), int(cm[1, 1]))

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "mcc": mcc,
        "sample_count": int(len(y_t)),
        "confusion_matrix": {"tp": tp, "fp": fp, "fn": fn, "tn": tn},
    }


def compute_pre_post_metrics(
    y_true: Sequence[int] | np.ndarray,
    y_pred: Sequence[int] | np.ndarray,
    drift_onset_index: int,
) -> dict[str, Any]:
    """Compute and contrast classification metrics strictly separated by drift onset.

    Parameters:
    - y_true: Ground-truth target labels.
    - y_pred: Model or system predictions.
    - drift_onset_index: Stream index where controlled drift began.

    Returns:
    - pre_drift: Metrics for index < drift_onset_index.
    - post_drift: Metrics for index >= drift_onset_index.
    - delta: post_drift - pre_drift for each metric.
    - percentage_change: Relative change in percentage.
    """
    y_t = np.asarray(y_true, dtype=int)
    y_p = np.asarray(y_pred, dtype=int)

    pre_mask = np.arange(len(y_t)) < drift_onset_index
    post_mask = np.arange(len(y_t)) >= drift_onset_index

    pre_metrics = compute_classification_metrics(y_t[pre_mask], y_p[pre_mask])
    post_metrics = compute_classification_metrics(y_t[post_mask], y_p[post_mask])

    delta = {}
    pct_change = {}
    for key in ("accuracy", "precision", "recall", "f1", "mcc"):
        d = post_metrics[key] - pre_metrics[key]
        delta[f"delta_{key}"] = float(d)
        if abs(pre_metrics[key]) > 1e-9:
            pct_change[f"pct_change_{key}"] = float((d / pre_metrics[key]) * 100.0)
        else:
            pct_change[f"pct_change_{key}"] = 0.0

    return {
        "pre_drift": pre_metrics,
        "post_drift": post_metrics,
        "delta": delta,
        "percentage_change": pct_change,
        "drift_onset_index": int(drift_onset_index),
        "pre_drift_samples": int(pre_mask.sum()),
        "post_drift_samples": int(post_mask.sum()),
    }
