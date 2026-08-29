#!/usr/bin/env python3
"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Script   : verify_phase9.py
Phase    : Phase 9 Verification Harness
Status   : IMPLEMENTED

Verifies the implementation of Phase 9: DRAEC Model Adaptation & Retraining Layer.
Runs 27 automated checks covering API contracts, anti-forgetting, test1 quarantine,
causal delayed feedback, candidate validation, atomic deployment with rollback,
monitoring telemetry integration, and regression checks across Phases 1-8.
"""

from __future__ import annotations

import ast
import copy
import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

passed = 0
failed = 0
results: list[tuple[str, bool, str]] = []


def record_check(name: str, condition: bool, details: str = "") -> None:
    global passed, failed
    if condition:
        passed += 1
        print(f"[PASS] {name} -- {details}")
    else:
        failed += 1
        print(f"[FAIL] {name} -- {details}")
    results.append((name, condition, details))


# =============================================================================
# Check 01: Public API imports
# =============================================================================
try:
    from src.adaptation import (
        AdaptationManager,
        AdaptationResult,
        AdaptationState,
        AtomicModelDeployer,
        CandidateValidator,
        CloudRetrainer,
        FeedbackQueue,
        FeedbackRecord,
        ModelVersionRecord,
        ValidationResult,
    )
    c01_ok = True
    record_check("Check 01: Public API imports", c01_ok, "All Phase 9 symbols imported successfully")
except Exception as e:
    record_check("Check 01: Public API imports", False, str(e))


# =============================================================================
# Check 02: Module file headers
# =============================================================================
try:
    adaptation_modules = [
        "src/adaptation/__init__.py",
        "src/adaptation/base.py",
        "src/adaptation/feedback.py",
        "src/adaptation/retrainer.py",
        "src/adaptation/validator.py",
        "src/adaptation/deployment.py",
        "src/adaptation/manager.py",
    ]
    c02_ok = True
    for mod in adaptation_modules:
        content = (ROOT / mod).read_text(encoding="utf-8")
        if "Phase    : Phase 9" not in content or "Status   : IMPLEMENTED" not in content:
            c02_ok = False
            break
    record_check("Check 02: Module file headers", c02_ok, "All 7 adaptation modules marked Phase 9 IMPLEMENTED")
except Exception as e:
    record_check("Check 02: Module file headers", False, str(e))


# =============================================================================
# Check 03: Configuration parsing
# =============================================================================
try:
    from src.utils.config import load
    cfg = load("default")
    adapt_cfg = cfg.get("adaptation", {})
    c03_ok = (
        adapt_cfg.get("enabled") is True
        and "trigger" in adapt_cfg
        and "training" in adapt_cfg
        and "validation" in adapt_cfg
        and "cooldown" in adapt_cfg
        and "deployment" in adapt_cfg
        and adapt_cfg["trigger"]["min_severity"] == 0.30
        and adapt_cfg["trigger"]["min_feedback_samples"] == 50
    )
    record_check("Check 03: Adaptation configuration", c03_ok, "top-level adaptation: section verified with valid defaults")
except Exception as e:
    record_check("Check 03: Adaptation configuration", False, str(e))


# =============================================================================
# Check 04: FeedbackQueue bounded memory and FIFO eviction
# =============================================================================
try:
    q = FeedbackQueue(max_size=20)
    for i in range(35):
        q.record_prediction(i, [0.1 * i], 0, {0: 1.0, 1: 0.0}, "v1")
    stats = q.get_stats()
    c04_ok = (
        stats["current_buffer_size"] == 20
        and stats["total_recorded"] == 35
        and stats["max_size"] == 20
    )
    record_check("Check 04: FeedbackQueue bounded capacity", c04_ok, f"Buffer size capped at {stats['current_buffer_size']} (total={stats['total_recorded']})")
except Exception as e:
    record_check("Check 04: FeedbackQueue bounded capacity", False, str(e))


# =============================================================================
# Check 05: FeedbackQueue acausal arrival prevention
# =============================================================================
try:
    q = FeedbackQueue(max_size=100)
    q.record_prediction(10, [0.5], 0, {0: 1.0, 1: 0.0}, "v1")
    acausal_caught = False
    try:
        q.provide_feedback(observation_index=10, label=1, arrival_index=9)
    except ValueError as ve:
        if "Acausal feedback arrival" in str(ve):
            acausal_caught = True
    record_check("Check 05: Acausal feedback rejection", acausal_caught, "Feedback arrival before observation rejected")
except Exception as e:
    record_check("Check 05: Acausal feedback rejection", False, str(e))


# =============================================================================
# Check 06: FeedbackQueue future feedback quarantine
# =============================================================================
try:
    q = FeedbackQueue(max_size=100)
    q.record_prediction(5, [0.5], 0, {0: 1.0, 1: 0.0}, "v1")
    q.provide_feedback(observation_index=5, label=1, arrival_index=20)

    # At current_index=15, feedback arriving at 20 must NOT be eligible
    el_at_15 = q.count_eligible(current_index=15)
    el_at_25 = q.count_eligible(current_index=25)
    c06_ok = (el_at_15 == 0 and el_at_25 == 1)
    record_check("Check 06: Future feedback quarantine", c06_ok, f"Eligible at t=15: {el_at_15}, Eligible at t=25: {el_at_25}")
except Exception as e:
    record_check("Check 06: Future feedback quarantine", False, str(e))


# =============================================================================
# Check 07: Strict test1 quarantine in FeedbackQueue
# =============================================================================
try:
    q = FeedbackQueue(max_size=100)
    test1_rejected = False
    try:
        q.record_prediction(1, [0.1], 0, {0: 1.0, 1: 0.0}, "v1", source="test1")
    except ValueError as ve:
        if "test1 evaluation data is strictly quarantined" in str(ve):
            test1_rejected = True
    record_check("Check 07: Strict test1 quarantine", test1_rejected, "test1 evaluation stream strictly prevented from entering feedback queue")
except Exception as e:
    record_check("Check 07: Strict test1 quarantine", False, str(e))


# =============================================================================
# Check 08: CloudRetrainer representative baseline caching
# =============================================================================
try:
    retrainer = CloudRetrainer(max_baseline_samples=50, random_seed=42)
    X_toy = np.random.randn(200, 5)
    y_toy = np.random.randint(0, 2, size=200)
    retrainer.set_baseline_data(X_toy, y_toy)
    stats = retrainer.get_stats()
    c08_ok = (
        retrainer.has_baseline_data
        and stats["baseline_samples_cached"] == 50
    )
    record_check("Check 08: Representative baseline caching", c08_ok, f"Baseline sample cached and bounded to {stats['baseline_samples_cached']} samples")
except Exception as e:
    record_check("Check 08: Representative baseline caching", False, str(e))


# =============================================================================
# Check 09: CloudRetrainer hybrid dataset combination (Anti-Forgetting)
# =============================================================================
try:
    retrainer = CloudRetrainer(min_feedback_samples=10, max_baseline_samples=30, random_seed=42)
    X_toy = np.random.randn(100, 4)
    y_toy = np.random.randint(0, 2, size=100)
    retrainer.set_baseline_data(X_toy, y_toy)

    feedback = [
        FeedbackRecord(
            observation_index=i,
            features=X_toy[i],
            prediction=0,
            probabilities={0: 0.8, 1: 0.2},
            model_version="v1",
            label=int(y_toy[i]),
            arrival_index=i + 1,
            is_labeled=True,
        )
        for i in range(15)
    ]
    cand, meta = retrainer.retrain(feedback, parent_version="v1", candidate_version="v2")
    from src.models.cloud_model import CloudXGBoost
    c09_ok = (
        isinstance(cand, CloudXGBoost)
        and meta["baseline_samples_used"] == 30
        and meta["feedback_samples_used"] == 15
        and meta["total_samples_trained"] == 45
    )
    record_check("Check 09: Hybrid anti-forgetting retraining", c09_ok, f"Candidate trained on {meta['total_samples_trained']} samples ({meta['baseline_samples_used']} base + {meta['feedback_samples_used']} feedback)")
except Exception as e:
    record_check("Check 09: Hybrid anti-forgetting retraining", False, str(e))


# =============================================================================
# Check 10: CloudRetrainer deterministic training
# =============================================================================
try:
    r1 = CloudRetrainer(min_feedback_samples=5, max_baseline_samples=20, random_seed=42)
    r2 = CloudRetrainer(min_feedback_samples=5, max_baseline_samples=20, random_seed=42)
    r1.set_baseline_data(X_toy, y_toy)
    r2.set_baseline_data(X_toy, y_toy)

    c1, _ = r1.retrain(feedback[:10])
    c2, _ = r2.retrain(feedback[:10])

    p1 = c1.predict(X_toy[:20])
    p2 = c2.predict(X_toy[:20])
    c10_ok = np.array_equal(p1, p2)
    record_check("Check 10: Deterministic retraining", c10_ok, "Identical seeds produced identical candidate predictions")
except Exception as e:
    record_check("Check 10: Deterministic retraining", False, str(e))


# =============================================================================
# Check 11: CandidateValidator clean validation on train2
# =============================================================================
try:
    val = CandidateValidator(minimum_metric=0.50, max_regression_margin=0.20)
    X_val = np.random.randn(50, 4)
    y_val = (X_val[:, 0] > 0).astype(int)
    val.set_validation_data(X_val, y_val, source="train2")

    from src.models.base import BaseModel
    class SimpleModel(BaseModel):
        def __init__(self, pred: int, name: str = "simple"):
            super().__init__(model_name=name)
            self._p = int(pred)
        def fit(self, X, y): return self
        def predict(self, X): return np.full(len(X), self._p, dtype=int)
        def predict_proba(self, X): return np.tile([0.5, 0.5], (len(X), 1))
        def predict_one(self, x): return self._p
        def predict_proba_one(self, x): return {0: 0.5, 1: 0.5}
        def get_info(self): return {"name": self.model_name}

    res_val = val.validate(SimpleModel(1), SimpleModel(0))
    c11_ok = (
        isinstance(res_val, ValidationResult)
        and res_val.metric_name == "macro_f1"
        and val.has_validation_data
    )
    record_check("Check 11: Candidate validation on train2", c11_ok, f"Validation result: status={res_val.status}, metric={res_val.candidate_metric:.3f}")
except Exception as e:
    record_check("Check 11: Candidate validation on train2", False, str(e))


# =============================================================================
# Check 12: CandidateValidator test1 quarantine
# =============================================================================
try:
    val = CandidateValidator()
    test1_rejected_val = False
    try:
        val.set_validation_data([[1, 2]], [0], source="test1")
    except ValueError as ve:
        if "test1 is reserved strictly" in str(ve):
            test1_rejected_val = True
    record_check("Check 12: Validator test1 quarantine", test1_rejected_val, "test1 partition strictly rejected as validation source")
except Exception as e:
    record_check("Check 12: Validator test1 quarantine", False, str(e))


# =============================================================================
# Check 13: CandidateValidator minimum metric threshold enforcement
# =============================================================================
try:
    val_strict = CandidateValidator(minimum_metric=0.99)
    val_strict.set_validation_data(X_val, y_val)
    res_strict = val_strict.validate(SimpleModel(0), SimpleModel(0))
    c13_ok = (
        res_strict.candidate_valid is False
        and res_strict.status == "REJECTED"
        and "below minimum threshold" in (res_strict.reason or "")
    )
    record_check("Check 13: Minimum metric threshold", c13_ok, "Low-quality candidate rejected as invalid")
except Exception as e:
    record_check("Check 13: Minimum metric threshold", False, str(e))


# =============================================================================
# Check 14: CandidateValidator regression bound enforcement
# =============================================================================
try:
    val_reg = CandidateValidator(minimum_metric=0.10, max_regression_margin=0.01)
    val_reg.set_validation_data(X_val, y_val)
    # Active model matches ground truth perfectly
    class PerfectModel(BaseModel):
        def __init__(self, y_true):
            super().__init__(model_name="perfect")
            self.y_true = y_true
        def fit(self, X, y): return self
        def predict(self, X): return self.y_true
        def predict_proba(self, X): return np.tile([0.5, 0.5], (len(X), 1))
        def predict_one(self, x): return int(self.y_true[0])
        def predict_proba_one(self, x): return {0: 0.5, 1: 0.5}
        def get_info(self): return {"name": "perfect"}

    res_reg = val_reg.validate(SimpleModel(0), PerfectModel(y_val))
    c14_ok = (
        res_reg.candidate_valid is False
        and res_reg.status == "REJECTED"
        and "regresses" in (res_reg.reason or "")
    )
    record_check("Check 14: Regression bound enforcement", c14_ok, f"Regressing candidate rejected (delta={res_reg.metric_delta:.3f})")
except Exception as e:
    record_check("Check 14: Regression bound enforcement", False, str(e))


# =============================================================================
# Check 15: AtomicModelDeployer 4-way version tracking
# =============================================================================
try:
    from src.deployment.runtimes import CloudRuntime, EdgeRuntime
    deployer = AtomicModelDeployer(CloudRuntime(SimpleModel(0)), EdgeRuntime(SimpleModel(0)), initial_version="v1")
    c15_ok = (
        deployer.candidate_version is None
        and deployer.cloud_version == "v1"
        and deployer.edge_version == "v1"
        and deployer.active_system_version == "v1"
    )
    record_check("Check 15: 4-way version tracking", c15_ok, f"candidate={deployer.candidate_version}, cloud={deployer.cloud_version}, edge={deployer.edge_version}, system={deployer.active_system_version}")
except Exception as e:
    record_check("Check 15: 4-way version tracking", False, str(e))


# =============================================================================
# Check 16: AtomicModelDeployer successful atomic commit
# =============================================================================
try:
    cand_c = SimpleModel(1)
    cand_e = SimpleModel(1)
    succ, rb, err = deployer.deploy(cand_c, cand_e, candidate_version="v2")
    c16_ok = (
        succ is True
        and rb is False
        and deployer.cloud_version == "v2"
        and deployer.edge_version == "v2"
        and deployer.active_system_version == "v2"
        and deployer.cloud_runtime.model is cand_c
        and deployer.edge_runtime.model is cand_e
    )
    record_check("Check 16: Successful atomic commit", c16_ok, f"Both Cloud and Edge updated to v2; active_system_version={deployer.active_system_version}")
except Exception as e:
    record_check("Check 16: Successful atomic commit", False, str(e))


# =============================================================================
# Check 17: AtomicModelDeployer Cloud rollback on Edge failure
# =============================================================================
try:
    old_c = deployer.cloud_runtime.model
    old_e = deployer.edge_runtime.model
    succ_fail, rb_fail, err_fail = deployer.deploy(
        SimpleModel(2), SimpleModel(2), candidate_version="v3", force_edge_failure=True,
    )
    c17_ok = (
        succ_fail is False
        and rb_fail is True
        and deployer.active_system_version == "v2"
        and deployer.cloud_version == "v2"
        and deployer.edge_version == "v2"
        and deployer.cloud_runtime.model is old_c
        and deployer.edge_runtime.model is old_e
    )
    record_check("Check 17: Atomic rollback on Edge failure", c17_ok, f"Rollback executed: Cloud restored to v2, active={deployer.active_system_version}")
except Exception as e:
    record_check("Check 17: Atomic rollback on Edge failure", False, str(e))


# =============================================================================
# Check 18: AtomicModelDeployer ModelRegistry integration
# =============================================================================
try:
    from src.monitoring.registry import ModelRegistry
    reg = ModelRegistry()
    reg.register_model(SimpleModel(0), "cloud", "cloud", version="v1")
    reg.register_model(SimpleModel(0), "edge", "edge", version="v1")

    dep_reg = AtomicModelDeployer(
        CloudRuntime(SimpleModel(0)), EdgeRuntime(SimpleModel(0)),
        model_registry=reg, initial_version="v1",
    )
    dep_reg.deploy(SimpleModel(1), SimpleModel(1), candidate_version="v2")
    c18_ok = (
        reg.get_metadata("cloud").model_version == "v2"
        and reg.get_metadata("edge").model_version == "v2"
    )
    record_check("Check 18: ModelRegistry synchronization", c18_ok, "Phase 7 ModelRegistry metadata synchronized on atomic deployment")
except Exception as e:
    record_check("Check 18: ModelRegistry synchronization", False, str(e))


# =============================================================================
# Check 19: AdaptationManager persistent drift gating
# =============================================================================
try:
    queue = FeedbackQueue()
    ret = CloudRetrainer(min_feedback_samples=5)
    ret.set_baseline_data(X_val[:20], y_val[:20])
    val = CandidateValidator(minimum_metric=0.10)
    val.set_validation_data(X_val[20:], y_val[20:])
    dep = AtomicModelDeployer(CloudRuntime(SimpleModel(0)), EdgeRuntime(SimpleModel(0)))
    mgr = AdaptationManager(queue, ret, val, dep, min_feedback_samples=5)

    # Fill 10 labeled feedback samples
    for i in range(10):
        queue.record_prediction(i, X_val[i % 20], 0, {0: 1.0, 1: 0.0}, "v1")
        queue.provide_feedback(i, y_val[i % 20], arrival_index=i)

    res_trans = mgr.step(
        observation_index=15, x=X_val[0], prediction=0, probabilities={0: 1.0, 1: 0.0},
        model_version="v1", is_persistent_drift=False, drift_severity=0.80,
    )
    c19_ok = (res_trans.triggered is False and res_trans.state == AdaptationState.IDLE)
    record_check("Check 19: Persistent drift gating", c19_ok, "Transient drift (is_persistent=False) safely ignored")
except Exception as e:
    record_check("Check 19: Persistent drift gating", False, str(e))


# =============================================================================
# Check 20: AdaptationManager drift severity gating
# =============================================================================
try:
    res_low = mgr.step(
        observation_index=16, x=X_val[0], prediction=0, probabilities={0: 1.0, 1: 0.0},
        model_version="v1", is_persistent_drift=True, drift_severity=0.15,
    )
    c20_ok = (res_low.triggered is False and res_low.state == AdaptationState.IDLE)
    record_check("Check 20: Drift severity gating", c20_ok, "Low drift severity (0.15 < 0.30) safely ignored")
except Exception as e:
    record_check("Check 20: Drift severity gating", False, str(e))


# =============================================================================
# Check 21: AdaptationManager feedback sample count gating
# =============================================================================
try:
    empty_q = FeedbackQueue()
    mgr_empty = AdaptationManager(empty_q, ret, val, dep, min_feedback_samples=50)
    res_insuf = mgr_empty.step(
        observation_index=1, x=X_val[0], prediction=0, probabilities={0: 1.0, 1: 0.0},
        model_version="v1", is_persistent_drift=True, drift_severity=0.85,
    )
    c21_ok = (res_insuf.triggered is False and res_insuf.state == AdaptationState.IDLE)
    record_check("Check 21: Feedback sample count gating", c21_ok, "Insufficient feedback (< 50) prevented trigger")
except Exception as e:
    record_check("Check 21: Feedback sample count gating", False, str(e))


# =============================================================================
# Check 22: AdaptationManager cooldown enforcement
# =============================================================================
try:
    # Trigger adaptation
    res_trig = mgr.step(
        observation_index=20, x=X_val[0], prediction=0, probabilities={0: 1.0, 1: 0.0},
        model_version="v1", is_persistent_drift=True, drift_severity=0.50,
    )
    # Next step during cooldown
    res_cool = mgr.step(
        observation_index=25, x=X_val[0], prediction=0, probabilities={0: 1.0, 1: 0.0},
        model_version="v2", is_persistent_drift=True, drift_severity=0.50,
    )
    c22_ok = (
        res_trig.triggered is True
        and res_cool.triggered is False
        and res_cool.state == AdaptationState.COOLDOWN
    )
    record_check("Check 22: Cooldown enforcement", c22_ok, "Cooldown active: repeated immediate retraining prevented")
except Exception as e:
    record_check("Check 22: Cooldown enforcement", False, str(e))


# =============================================================================
# Check 23: End-to-end adaptation lifecycle
# =============================================================================
try:
    c23_ok = (
        res_trig.state == AdaptationState.ACCEPTED
        and res_trig.deployment_success is True
        and res_trig.active_version == "v2"
        and dep.active_system_version == "v2"
    )
    record_check("Check 23: End-to-end adaptation lifecycle", c23_ok, f"Lifecycle completed: active_version={res_trig.active_version}, samples={res_trig.samples_used}")
except Exception as e:
    record_check("Check 23: End-to-end adaptation lifecycle", False, str(e))


# =============================================================================
# Check 24: Phase 5 DecisionEngine compatibility
# =============================================================================
try:
    from src.decision.engine import AdaptiveController, DecisionEngine
    from src.decision.base import DecisionInputs, DecisionAction
    ctrl = AdaptiveController()
    d_eng = DecisionEngine(controller=ctrl, edge_model=SimpleModel(0), cloud_model=SimpleModel(1))
    d_res = d_eng.decide(DecisionInputs(reliability=0.85, observation_index=1))
    c24_ok = (
        d_res.selected_action == DecisionAction.EDGE
        and ctrl.critical_cloud_threshold == 0.30
        and ctrl.cloud_threshold == 0.50
        and ctrl.edge_return_threshold == 0.70
    )
    record_check("Check 24: Phase 5 routing compatibility", c24_ok, "Phase 5 hysteresis thresholds and actions strictly preserved")
except Exception as e:
    record_check("Check 24: Phase 5 routing compatibility", False, str(e))


# =============================================================================
# Check 25: Phase 4 ReliabilityEstimator compatibility
# =============================================================================
try:
    from src.reliability.estimator import ReliabilityEstimator
    rel_est = ReliabilityEstimator()
    rel_score = rel_est.update(probs={0: 0.90, 1: 0.10}, drift_severity=0.05, quality=0.98)
    c25_ok = (
        0.0 <= rel_score.reliability <= 1.0
        and hasattr(rel_score.inputs, "confidence")
        and hasattr(rel_score.inputs, "drift")
    )
    record_check("Check 25: Phase 4 reliability compatibility", c25_ok, f"Harmonic mean reliability R_t={rel_score.reliability:.4f} computed unaltered")
except Exception as e:
    record_check("Check 25: Phase 4 reliability compatibility", False, str(e))


# =============================================================================
# Check 26: Phase 8 DeploymentEnvironment compatibility
# =============================================================================
try:
    from src.deployment.environment import DeploymentEnvironment
    from src.deployment.network import NetworkSimulator
    env = DeploymentEnvironment(dep.edge_runtime, dep.cloud_runtime, NetworkSimulator())
    e_res = env.execute_edge(X_val[0])
    c_res = env.execute_cloud(X_val[0])
    h_res = env.execute_hybrid(X_val[0])
    c26_ok = (e_res.success and c_res.success and h_res.success)
    record_check("Check 26: Phase 8 deployment compatibility", c26_ok, "DeploymentEnvironment executes adapted Edge and Cloud models seamlessly")
except Exception as e:
    record_check("Check 26: Phase 8 deployment compatibility", False, str(e))


# =============================================================================
# Check 27: Scope boundary enforcement (Phase 10 quarantined)
# =============================================================================
try:
    phase10_files = [
        ROOT / "src" / "simulation" / "__init__.py",
        ROOT / "src" / "simulation" / "environment.py",
        ROOT / "src" / "metrics" / "__init__.py",
        ROOT / "src" / "metrics" / "system.py",
    ]
    p10_clean = True
    for p10 in phase10_files:
        if p10.exists():
            tree = ast.parse(p10.read_text(encoding="utf-8"))
            stmts = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if stmts:
                p10_clean = False
                break
    record_check("Check 27: Phase 10 scope boundary", p10_clean, "Phase 10 modules remain pure un-implemented stubs")
except Exception as e:
    record_check("Check 27: Phase 10 scope boundary", False, str(e))


# =============================================================================
# Summary
# =============================================================================
print("\n" + "=" * 60)
print(f"PHASE 9 VERIFICATION SUMMARY: {passed} / {passed + failed} CHECKS PASSED")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
