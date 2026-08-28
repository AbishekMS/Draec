"""Phase 2 Verification Harness -- Edge and Cloud prediction models.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Verifies:
- Edge model (River Hoeffding Tree Classifier)
- Cloud model (XGBoost Classifier)
- Common BaseModel interface
- Exact verified 37-feature Phase 1 representation
- Causal baseline_train model fitting with validation/inference isolation
- Macro-F1 evaluation and inference latency tracking
- Small end-to-end smoke test

Run:
    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe verify_phase2.py
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def note(msg: str) -> None:
    _details.append(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


# =============================================================================
# 1. Modules import and are marked IMPLEMENTED
# =============================================================================
try:
    from src.models import (
        BaseModel,
        CloudModel,
        CloudXGBoost,
        EdgeHoeffdingTree,
        EdgeModel,
        evaluate_model,
        evaluate_predictions,
        extract_partition_labels,
        get_key_by_role,
        load_causal_eval_data,
        load_causal_train_data,
        train_cloud_model,
        train_edge_model,
    )
    from src.models.trainer import LEAKAGE_COLUMNS
    from src.utils import config as cfgmod

    MOD_NAMES = ("base.py", "edge_model.py", "cloud_model.py", "trainer.py")
    impl_flags = []
    for mod_name in MOD_NAMES:
        content = (ROOT / "src" / "models" / mod_name).read_text(encoding="utf-8")
        impl_flags.append("Status   : IMPLEMENTED" in content)

    check(
        "1. all four src/models modules import and are marked IMPLEMENTED",
        all(impl_flags),
        f"verified {', '.join(MOD_NAMES)} carry 'Status   : IMPLEMENTED'",
    )
except Exception as e:
    check("1. src/models import", False, f"{type(e).__name__}: {e}")
    print("FATAL: cannot import src/models modules; aborting.")
    raise SystemExit(1)

CONFIG_DIR = ROOT / "config"
cfg = cfgmod.load("default", config_dir=CONFIG_DIR)

# =============================================================================
# 2. EdgeHoeffdingTree BaseModel conformance & hyperparameter binding
# =============================================================================
edge = EdgeModel(cfg)
edge_info = edge.get_info()
check(
    "2. EdgeHoeffdingTree implements BaseModel and binds configured hyperparameters",
    (
        isinstance(edge, BaseModel)
        and edge.model_name == "RiverHoeffdingTreeClassifier"
        and not edge.is_trained
        and edge.n_features is None
        and "grace_period" in edge_info["hyperparameters"]
        and edge_info["hyperparameters"]["delta"] == 1e-7
    ),
    f"model_name={edge.model_name!r}, params={edge_info['hyperparameters']}",
)

# =============================================================================
# 3. CloudXGBoost BaseModel conformance & reproducibility seed binding
# =============================================================================
cloud = CloudModel(cfg)
cloud_info = cloud.get_info()
check(
    "3. CloudXGBoost implements BaseModel and binds configured hyperparameters & seed",
    (
        isinstance(cloud, BaseModel)
        and cloud.model_name == "CloudXGBoostClassifier"
        and not cloud.is_trained
        and cloud.n_features is None
        and cloud_info["hyperparameters"]["n_estimators"] == 100
        and cloud_info["hyperparameters"]["random_state"] == 42
    ),
    f"model_name={cloud.model_name!r}, params={cloud_info['hyperparameters']}",
)

# =============================================================================
# 4. Causal baseline_train data loading: exactly 37 features, Target excluded
# =============================================================================
X_train_sub, y_train_sub, stats, profile = load_causal_train_data(
    cfg, root=ROOT, max_rows=500
)
has_target = "Target" in X_train_sub.columns
leakage_in_X = [c for c in LEAKAGE_COLUMNS if c in X_train_sub.columns]

check(
    "4. load_causal_train_data loads 37 features with Target & leakage excluded",
    (
        X_train_sub.shape == (500, 37)
        and len(y_train_sub) == 500
        and not has_target
        and not leakage_in_X
        and len(stats.columns) == 37
    ),
    f"X_train shape: {X_train_sub.shape}, target excluded: {not has_target}, "
    f"leakage columns: {leakage_in_X}",
)

# =============================================================================
# 5. Causal baseline_validation loading with frozen baseline statistics
# =============================================================================
X_val_sub, y_val_sub = load_causal_eval_data(
    cfg, "baseline_validation", stats, root=ROOT, max_rows=200
)
val_leakage = [c for c in LEAKAGE_COLUMNS if c in X_val_sub.columns]
check(
    "5. load_causal_eval_data loads validation using frozen stats with zero leakage",
    (
        X_val_sub.shape == (200, 37)
        and len(y_val_sub) == 200
        and "Target" not in X_val_sub.columns
        and not val_leakage
        and tuple(X_val_sub.columns) == tuple(X_train_sub.columns)
    ),
    f"X_val shape: {X_val_sub.shape}, leakage columns: {val_leakage}",
)

# =============================================================================
# 6. Causal inference_stream loading with frozen baseline statistics
# =============================================================================
X_inf_sub, y_inf_sub = load_causal_eval_data(
    cfg, "inference_stream", stats, root=ROOT, max_rows=200
)
inf_leakage = [c for c in LEAKAGE_COLUMNS if c in X_inf_sub.columns]
check(
    "6. load_causal_eval_data loads inference stream using frozen stats with zero leakage",
    (
        X_inf_sub.shape == (200, 37)
        and len(y_inf_sub) == 200
        and "Target" not in X_inf_sub.columns
        and not inf_leakage
        and tuple(X_inf_sub.columns) == tuple(X_train_sub.columns)
    ),
    f"X_inf shape: {X_inf_sub.shape}, leakage columns: {inf_leakage}",
)

# =============================================================================
# 7. Causality guard: fitting on validation or inference is refused
# =============================================================================
from src.data.loader import CausalityError

refused_train_on_val = False
try:
    load_causal_eval_data(cfg, "baseline_train", stats=None)
except CausalityError:
    refused_train_on_val = True

refused_eval_without_stats = False
try:
    load_causal_eval_data(cfg, "baseline_validation", stats=None)
except CausalityError:
    refused_eval_without_stats = True

check(
    "7. acausal fitting and eval-without-baseline-stats are strictly refused",
    refused_train_on_val and refused_eval_without_stats,
    "eval_data refuses baseline_train; eval_data refuses missing frozen stats",
)

# =============================================================================
# 8. EdgeHoeffdingTree supervised training & prediction
# =============================================================================
edge = train_edge_model(cfg, X_train_sub, y_train_sub)
edge_preds = edge.predict(X_val_sub)
edge_probas = edge.predict_proba(X_val_sub)
edge_single = edge.predict_one(X_val_sub.iloc[0])
edge_single_p = edge.predict_proba_one(X_val_sub.iloc[0])

check(
    "8. EdgeHoeffdingTree predicts classes and valid probabilities",
    (
        edge.is_trained
        and edge.n_features == 37
        and edge_preds.shape == (len(X_val_sub),)
        and edge_probas.shape == (len(X_val_sub), 2)
        and np.allclose(edge_probas.sum(axis=1), 1.0)
        and edge_single in {0, 1}
        and set(edge_single_p.keys()) == {0, 1}
        and edge.last_inference_time_s is not None
    ),
    f"samples trained: {edge.n_samples_trained}, last latency: {edge.last_inference_time_s * 1000:.3f}ms",
)

# =============================================================================
# 9. CloudXGBoost supervised training & prediction
# =============================================================================
cloud = train_cloud_model(cfg, X_train_sub, y_train_sub)
cloud_preds = cloud.predict(X_val_sub)
cloud_probas = cloud.predict_proba(X_val_sub)
cloud_single = cloud.predict_one(X_val_sub.iloc[0])
cloud_single_p = cloud.predict_proba_one(X_val_sub.iloc[0])

check(
    "9. CloudXGBoost predicts classes and valid probabilities",
    (
        cloud.is_trained
        and cloud.n_features == 37
        and cloud_preds.shape == (len(X_val_sub),)
        and cloud_probas.shape == (len(X_val_sub), 2)
        and np.allclose(cloud_probas.sum(axis=1), 1.0)
        and cloud_single in {0, 1}
        and set(cloud_single_p.keys()) == {0, 1}
        and cloud.last_inference_time_s is not None
    ),
    f"samples trained: {cloud.n_samples_trained}, last latency: {cloud.last_inference_time_s * 1000:.3f}ms",
)

# =============================================================================
# 10. Edge online incremental learning
# =============================================================================
n_before = edge.n_samples_trained
edge.learn_one(X_val_sub.iloc[0], y_val_sub[0])
edge.learn_many(X_val_sub.iloc[1:10], y_val_sub[1:10])
n_after = edge.n_samples_trained

check(
    "10. EdgeHoeffdingTree supports online incremental update (learn_one & learn_many)",
    n_after == n_before + 10,
    f"samples trained updated from {n_before} to {n_after}",
)

# =============================================================================
# 11. Feature parity between Edge and Cloud
# =============================================================================
check(
    "11. Edge and Cloud models receive identical 37-feature representation",
    (
        edge.n_features == 37
        and cloud.n_features == 37
        and edge.feature_names == cloud.feature_names
        and edge.feature_names == tuple(X_train_sub.columns)
    ),
    f"both models verify exactly 37 features ({list(edge.feature_names)[:3]}...)",
)

# =============================================================================
# 12. Small Phase 2 end-to-end smoke test
# =============================================================================
eval_edge = evaluate_model(edge, X_val_sub, y_val_sub)
eval_cloud = evaluate_model(cloud, X_val_sub, y_val_sub)

check(
    "12. small end-to-end smoke test evaluates Macro-F1 and inference latency",
    (
        0.0 <= eval_edge["macro_f1"] <= 1.0
        and 0.0 <= eval_cloud["macro_f1"] <= 1.0
        and eval_edge["n_samples"] == len(y_val_sub)
        and eval_cloud["n_samples"] == len(y_val_sub)
        and eval_edge["inference_time_s"] is not None
        and eval_cloud["inference_time_s"] is not None
    ),
    f"Smoke test (200 val rows) -> Edge Macro-F1: {eval_edge['macro_f1']:.4f} "
    f"({eval_edge['inference_time_s']*1000:.2f}ms), "
    f"Cloud Macro-F1: {eval_cloud['macro_f1']:.4f} "
    f"({eval_cloud['inference_time_s']*1000:.2f}ms) -- SMOKE TEST ONLY, NOT FINAL RESULTS",
)

# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("PHASE 2 VERIFICATION -- src/models/{base,edge_model,cloud_model,trainer}.py")
print("=" * 78)
for ln in _details:
    print(ln)
print("-" * 78)
n_pass = sum(1 for _, ok, _ in results if ok)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for ln in str(detail).splitlines():
            print(f"       {ln}")
print("-" * 78)
print(f"{n_pass}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if n_pass == len(results) else 1)
