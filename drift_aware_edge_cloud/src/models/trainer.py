"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/models/trainer.py
Phase    : Phase 2
Status   : IMPLEMENTED

Causal training pipeline, partition data loaders, and model evaluation utilities.
Strictly guards against target leakage, future information, and acausal fitting.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from src.data import loader, preprocessing
from src.data.loader import BaselineProfile, CausalityError, ConfigError, LoadedFile, SchemaError
from src.data.preprocessing import BaselineStatistics
from src.models.base import BaseModel
from src.models.cloud_model import CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree


# Excluded columns that MUST NOT enter model feature matrices under any circumstance
LEAKAGE_COLUMNS = (
    "Target",
    "Traffic",
    "StartTime",
    "LastTime",
    "RunTime",
    "SrcAddr",
    "DstAddr",
    "Sport",
    "Dport",
    "Proto",
    "sIpId",
    "dIpId",
)


def get_key_by_role(config: Mapping[str, Any], role: str) -> str:
    """Resolve the dataset file key for a specific role (e.g. baseline_train)."""
    files = config.get("dataset", {}).get("files", {})
    for key, spec in files.items():
        if spec.get("role") == role:
            return str(key)
    raise ConfigError(f"No file declared with role '{role}' in dataset.files")


def extract_partition_labels(
    config: Mapping[str, Any],
    role_or_key: str,
    *,
    root: Path | str = ".",
    max_rows: int | None = None,
) -> np.ndarray:
    """Causally extract ground truth Target labels for a specific partition.

    Labels are read from the raw file following the exact chronological and
    tie-breaker ordering contract of Phase 1, never manufactured or altered.
    """
    ds = config.get("dataset") or {}
    files = ds.get("files") or {}

    # Determine key from role or direct key
    if role_or_key in files:
        key = role_or_key
    else:
        key = get_key_by_role(config, role_or_key)

    specs = loader.file_specs(config, root)
    if key not in specs:
        raise ConfigError(f"Key {key!r} not found in file specs")
    spec = specs[key]

    entry = files.get(key) or {}
    sel_range = entry.get("selection_time_range")

    raw = loader._read_frame(
        spec, ds, nrows=max_rows if not sel_range else None, selection_time_range=sel_range
    )
    raw = loader._order_and_select(raw, ds, key)
    if max_rows is not None:
        raw = raw.iloc[:max_rows].reset_index(drop=True)

    target_col = loader.resolve_target(config)
    if target_col not in raw.columns:
        raise SchemaError(f"Target column {target_col!r} not found in raw partition {key!r}")

    y = pd.to_numeric(raw[target_col], errors="coerce").fillna(0).to_numpy(dtype=int)
    return y


_CACHED_BASELINE_STATS: tuple[Any, Any] | None = None


def _get_or_fit_baseline_stats(config: Mapping[str, Any], root: Path | str) -> tuple[Any, Any]:
    global _CACHED_BASELINE_STATS
    if _CACHED_BASELINE_STATS is None:
        baseline = loader.load_baseline(config, root=root)
        profile = loader.profile_baseline(config, baseline)
        stats = preprocessing.fit(config, baseline, profile)
        _CACHED_BASELINE_STATS = (profile, stats)
    return _CACHED_BASELINE_STATS


def load_causal_train_data(
    config: Mapping[str, Any],
    *,
    root: Path | str = ".",
    max_rows: int | None = None,
    profile: Any | None = None,
    stats: Any | None = None,
) -> tuple[pd.DataFrame, np.ndarray, Any, Any]:
    """Causally load and preprocess baseline training data for model fitting.

    Guarantees:
    - Training data strictly comes from partition with role 'baseline_train'.
    - Features are normalized against frozen baseline statistics.
    - Target is strictly excluded from the feature matrix X.
    - Ordering tie-breakers and metadata columns are strictly excluded.
    """
    train_key = get_key_by_role(config, "baseline_train")

    if profile is None or stats is None:
        profile, stats = _get_or_fit_baseline_stats(config, root=root)

    train_file = loader.load_file(config, train_key, root=root, max_rows=max_rows)

    if train_file.role != "baseline_train":
        raise CausalityError(
            f"Training data role must be 'baseline_train', got '{train_file.role}'"
        )

    prep = preprocessing.transform(config, train_file, stats)
    X_train = prep.frame
    y_train = extract_partition_labels(config, "baseline_train", root=root, max_rows=max_rows)

    if max_rows is not None:
        X_train = X_train.iloc[:max_rows].reset_index(drop=True)
        y_train = y_train[:max_rows]

    if len(X_train) != len(y_train):
        raise ValueError(
            f"Row count mismatch in baseline_train: X has {len(X_train)} rows, y has {len(y_train)}"
        )

    _assert_no_leakage(X_train)

    return X_train, y_train, stats, profile


def load_causal_eval_data(
    config: Mapping[str, Any],
    role: str,
    stats: BaselineStatistics,
    *,
    root: Path | str = ".",
    max_rows: int | None = None,
) -> tuple[pd.DataFrame, np.ndarray]:
    """Load evaluation features and labels for baseline_validation or inference_stream.

    LEAKAGE GUARD:
    - Reuses FROZEN baseline statistics fitted on baseline_train.
    - Never refits statistics on evaluation data.
    - Refuses to load baseline_train (use load_causal_train_data instead).
    """
    if role == "baseline_train":
        raise CausalityError(
            "Use load_causal_train_data to obtain baseline_train and fit statistics."
        )

    if stats is None:
        raise CausalityError(
            "Evaluation data cannot be loaded without frozen BaselineStatistics fitted on baseline_train."
        )

    key = get_key_by_role(config, role)
    loaded_file = loader.load_file(config, key, root=root, max_rows=max_rows)

    prep = preprocessing.transform(config, loaded_file, stats)
    X_eval = prep.frame
    y_eval = extract_partition_labels(config, role, root=root, max_rows=max_rows)

    if len(X_eval) != len(y_eval):
        raise ValueError(
            f"Row count mismatch in {role}: X has {len(X_eval)} rows, y has {len(y_eval)}"
        )

    _assert_no_leakage(X_eval)

    return X_eval, y_eval


def _assert_no_leakage(X: pd.DataFrame) -> None:
    """Enforce that no target, metadata, or tie-breaker columns entered feature matrix."""
    cols = set(X.columns)
    for forbidden in LEAKAGE_COLUMNS:
        if forbidden in cols:
            raise CausalityError(
                f"Leakage detected: forbidden column {forbidden!r} present in feature matrix"
            )


def train_edge_model(
    config: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    **kwargs: Any,
) -> EdgeHoeffdingTree:
    """Train the River Hoeffding Tree Edge model on causal baseline_train data."""
    _assert_no_leakage(X_train)
    model = EdgeHoeffdingTree(config=config, **kwargs)
    model.fit(X_train, y_train)
    return model


def train_cloud_model(
    config: Mapping[str, Any],
    X_train: pd.DataFrame,
    y_train: np.ndarray,
    **kwargs: Any,
) -> CloudXGBoost:
    """Train the XGBoost Cloud model on causal baseline_train data."""
    _assert_no_leakage(X_train)
    model = CloudXGBoost(config=config, **kwargs)
    model.fit(X_train, y_train)
    return model


def evaluate_predictions(y_true: Sequence[int] | np.ndarray, y_pred: Sequence[int] | np.ndarray) -> dict[str, Any]:
    """Calculate classification performance metrics.

    Macro-F1 is the primary research metric under class imbalance.
    """
    yt = np.asarray(y_true, dtype=int)
    yp = np.asarray(y_pred, dtype=int)

    macro_f1 = float(f1_score(yt, yp, average="macro", zero_division=0))
    acc = float(accuracy_score(yt, yp))
    macro_prec = float(precision_score(yt, yp, average="macro", zero_division=0))
    macro_rec = float(recall_score(yt, yp, average="macro", zero_division=0))
    cm = confusion_matrix(yt, yp, labels=[0, 1])

    return {
        "macro_f1": macro_f1,
        "accuracy": acc,
        "macro_precision": macro_prec,
        "macro_recall": macro_rec,
        "confusion_matrix": cm.tolist(),
        "n_samples": len(yt),
    }


def evaluate_model(
    model: BaseModel,
    X: pd.DataFrame | np.ndarray,
    y: Sequence[int] | np.ndarray,
) -> dict[str, Any]:
    """Evaluate a BaseModel on a dataset and return performance + timing metrics."""
    _assert_no_leakage(X if isinstance(X, pd.DataFrame) else pd.DataFrame(X))
    y_pred = model.predict(X)
    metrics = evaluate_predictions(y, y_pred)
    metrics["inference_time_s"] = model.last_inference_time_s
    metrics["model_name"] = model.model_name
    return metrics
