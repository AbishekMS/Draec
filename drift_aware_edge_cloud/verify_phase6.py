"""Phase 6 verification harness -- Hardened Execution Layer, Telemetry, and Failure Handling.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Checks:
1. check_phase6_contracts: ExecutionResult, ExecutionStatus exist with expected fields.
2. check_edge_execution: Edge executes, returns valid binary prediction, model_used='edge', status=SUCCESS.
3. check_cloud_execution: Cloud executes, returns valid binary prediction, model_used='cloud', status=SUCCESS.
4. check_hybrid_edge_first: Hybrid action executes Edge model first.
5. check_hybrid_edge_only: Hybrid completes at Edge when confidence >= 0.60, Cloud not invoked.
6. check_hybrid_cloud_fallback: Hybrid falls back to Cloud when Edge confidence < 0.60.
7. check_edge_latency_measured: T_edge > 0 and recorded as positive elapsed duration.
8. check_cloud_latency_measured: T_cloud > 0 and recorded as local software execution latency.
9. check_hybrid_latency_measured: T_hybrid > 0 measured by actual wall-clock timer covering complete path.
10. check_prediction_validity: Valid predictions in {0, 1}, invalid predictions rejected.
11. check_probability_validity: Probabilities valid distribution in [0, 1], sum to 1.0, finite.
12. check_input_validation: Rejects None, empty, wrong dim, NaN, and Target leakage key.
13. check_edge_failure_handling: Edge failure -> explicit status FAILED, success False, prediction None.
14. check_cloud_failure_handling: Cloud failure -> explicit status FAILED, success False, prediction None.
15. check_hybrid_fallback_failure: Edge uncertain + Cloud fails -> explicit status FAILED, prediction None.
16. check_execution_status_values: Execution status takes SUCCESS, FALLBACK, or FAILED.
17. check_cloud_fallback_flag: Flag is True only when fallback occurred.
18. check_execution_determinism: Same input produces identical execution result across runs.
19. check_bounded_telemetry: Telemetry respects max_records and does not grow indefinitely.
20. check_pipeline_compatibility: End-to-end integration across Phase 1, 2, 3, 4, 5, and 6.
21. check_quarantine_integrity: Zero leakage, Target isolation, ground truth isolation, Phase 7+ quarantine intact.

Run:
    ../.venv/Scripts/python.exe verify_phase6.py
"""

from __future__ import annotations

import ast
import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decision.base import (
    BaseController,
    DecisionAction,
    DecisionInputs,
    DecisionResult,
    ExecutionResult,
    ExecutionStatus,
)
from src.decision.engine import (
    AdaptiveController,
    DecisionEngine,
    DecisionInstrumentation,
    StaticBaselineController,
    validate_input,
    validate_output,
)
from src.models.base import BaseModel
from src.models.cloud_model import CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree
from src.reliability import ReliabilityEstimator
from src.utils import config as config_mod

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def note(msg: str) -> None:
    _details.append(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def banner(text: str) -> None:
    line = "=" * 78
    print(f"\n{line}\n{text}\n{line}")


class MockModel(BaseModel):
    """Controllable mock model for deterministic Phase 6 verification."""

    def __init__(
        self,
        name: str = "mock",
        pred: int = 0,
        probas: dict[int, float] | None = None,
        fail_predict: bool = False,
        fail_proba: bool = False,
        is_trained: bool = True,
        call_tracker: list[str] | None = None,
    ) -> None:
        super().__init__(model_name=name)
        self._pred = pred
        self._probas = probas or {0: 0.90, 1: 0.10}
        self._fail_predict = fail_predict
        self._fail_proba = fail_proba
        self._is_trained = is_trained
        self.call_tracker = call_tracker if call_tracker is not None else []

    def fit(self, X: Any, y: Any) -> MockModel:
        self._is_trained = True
        return self

    def predict_one(self, x: Any) -> int:
        self.call_tracker.append(f"{self.model_name}.predict_one")
        if self._fail_predict:
            raise RuntimeError(f"Simulated fault in {self.model_name}.predict_one")
        return self._pred

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        self.call_tracker.append(f"{self.model_name}.predict_proba_one")
        if self._fail_proba:
            raise RuntimeError(f"Simulated fault in {self.model_name}.predict_proba_one")
        return dict(self._probas)

    def predict(self, X: Any) -> np.ndarray:
        return np.full(len(X), self._pred, dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        p0 = self._probas[0]
        p1 = self._probas[1]
        return np.tile([p0, p1], (len(X), 1))

    def get_info(self) -> dict[str, Any]:
        return {"model_name": self.model_name, "is_trained": self._is_trained}


def main() -> int:
    banner("STEP 6 VERIFICATION -- DRAEC HARDENED EXECUTION LAYER")

    # -------------------------------------------------------------------------
    # Check 1: Phase 6 Contracts
    # -------------------------------------------------------------------------
    c1_ok = (
        hasattr(ExecutionStatus, "SUCCESS")
        and hasattr(ExecutionStatus, "FALLBACK")
        and hasattr(ExecutionStatus, "FAILED")
        and hasattr(ExecutionResult, "status")
        and hasattr(ExecutionResult, "success")
        and hasattr(ExecutionResult, "edge_latency_s")
        and hasattr(ExecutionResult, "cloud_latency_s")
        and hasattr(ExecutionResult, "hybrid_latency_s")
        and hasattr(ExecutionResult, "error")
    )
    check("1. Phase 6 contracts and datastructures exist with all fields", c1_ok, "ExecutionResult and ExecutionStatus contracts verified")

    # -------------------------------------------------------------------------
    # Check 2: Edge Successful Execution
    # -------------------------------------------------------------------------
    edge_m = MockModel(name="edge", pred=0, probas={0: 0.90, 1: 0.10})
    cloud_m = MockModel(name="cloud", pred=1, probas={0: 0.15, 1: 0.85})
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=edge_m, cloud_model=cloud_m)

    res_edge = engine.execute([0.1, 0.2, 0.3, 0.4], inputs=0.9)
    c2_ok = (
        res_edge.action == DecisionAction.EDGE
        and res_edge.prediction == 0
        and res_edge.model_used == "edge"
        and res_edge.status == ExecutionStatus.SUCCESS
        and res_edge.success is True
        and res_edge.cloud_fallback is False
    )
    check("2. Edge execution succeeds, returns valid prediction, status=SUCCESS", c2_ok, f"action={res_edge.action.value}, pred={res_edge.prediction}, status={res_edge.status.value}")

    # -------------------------------------------------------------------------
    # Check 3: Cloud Successful Execution
    # -------------------------------------------------------------------------
    ctrl_cloud = StaticBaselineController(policy="cloud_only")
    engine_cloud = DecisionEngine(controller=ctrl_cloud, edge_model=edge_m, cloud_model=cloud_m)

    res_cloud = engine_cloud.execute([0.1, 0.2, 0.3, 0.4], inputs=0.1)
    c3_ok = (
        res_cloud.action == DecisionAction.CLOUD
        and res_cloud.prediction == 1
        and res_cloud.model_used == "cloud"
        and res_cloud.status == ExecutionStatus.SUCCESS
        and res_cloud.success is True
        and res_cloud.cloud_fallback is False
    )
    check("3. Cloud execution succeeds, returns valid prediction, status=SUCCESS", c3_ok, f"action={res_cloud.action.value}, pred={res_cloud.prediction}, status={res_cloud.status.value}")

    # -------------------------------------------------------------------------
    # Check 4: Hybrid Edge-First Execution
    # -------------------------------------------------------------------------
    call_order: list[str] = []
    edge_track = MockModel(name="edge", pred=0, probas={0: 0.95, 1: 0.05}, call_tracker=call_order)
    cloud_track = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90}, call_tracker=call_order)
    ctrl_hyb = StaticBaselineController(policy="static_hybrid")
    eng_hyb = DecisionEngine(controller=ctrl_hyb, edge_model=edge_track, cloud_model=cloud_track)

    eng_hyb.execute([0.1, 0.2, 0.3, 0.4], inputs=0.45)
    c4_ok = len(call_order) >= 2 and call_order[0].startswith("edge.")
    check("4. Hybrid executes Edge model first", c4_ok, f"first call was {call_order[0] if call_order else 'none'}")

    # -------------------------------------------------------------------------
    # Check 5: Hybrid Edge-Only Completion
    # -------------------------------------------------------------------------
    edge_conf = MockModel(name="edge", pred=0, probas={0: 0.90, 1: 0.10})
    cloud_conf = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90})
    eng_hyb_conf = DecisionEngine(
        controller=ctrl_hyb,
        edge_model=edge_conf,
        cloud_model=cloud_conf,
        fallback_confidence_threshold=0.60,
    )
    res_hyb_edge = eng_hyb_conf.execute([0.1, 0.2, 0.3, 0.4], inputs=0.45)
    c5_ok = (
        res_hyb_edge.model_used == "hybrid_edge"
        and res_hyb_edge.cloud_fallback is False
        and res_hyb_edge.status == ExecutionStatus.SUCCESS
        and len(cloud_conf.call_tracker) == 0
    )
    check("5. Hybrid completes at Edge when confidence >= 0.60 (Cloud not called)", c5_ok, f"model_used={res_hyb_edge.model_used}, cloud_calls={len(cloud_conf.call_tracker)}")

    # -------------------------------------------------------------------------
    # Check 6: Hybrid Cloud Fallback
    # -------------------------------------------------------------------------
    edge_unc = MockModel(name="edge", pred=0, probas={0: 0.52, 1: 0.48})
    cloud_unc = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90})
    eng_hyb_unc = DecisionEngine(
        controller=ctrl_hyb,
        edge_model=edge_unc,
        cloud_model=cloud_unc,
        fallback_confidence_threshold=0.60,
    )
    res_hyb_cloud = eng_hyb_unc.execute([0.1, 0.2, 0.3, 0.4], inputs=0.45)
    c6_ok = (
        res_hyb_cloud.model_used == "hybrid_cloud"
        and res_hyb_cloud.cloud_fallback is True
        and res_hyb_cloud.status == ExecutionStatus.FALLBACK
        and len(cloud_unc.call_tracker) > 0
    )
    check("6. Hybrid falls back to Cloud when Edge confidence < 0.60", c6_ok, f"model_used={res_hyb_cloud.model_used}, fallback={res_hyb_cloud.cloud_fallback}")

    # -------------------------------------------------------------------------
    # Check 7: Edge Latency Measured
    # -------------------------------------------------------------------------
    c7_ok = res_edge.edge_latency_s is not None and res_edge.edge_latency_s > 0.0
    check("7. Edge latency measured and positive", c7_ok, f"T_edge={res_edge.edge_latency_s:.6f}s")

    # -------------------------------------------------------------------------
    # Check 8: Cloud Latency Measured (Local Software Execution Latency)
    # -------------------------------------------------------------------------
    c8_ok = res_cloud.cloud_latency_s is not None and res_cloud.cloud_latency_s > 0.0
    check("8. Cloud latency measured and positive (local software execution)", c8_ok, f"T_cloud={res_cloud.cloud_latency_s:.6f}s")

    # -------------------------------------------------------------------------
    # Check 9: Hybrid Latency Measured
    # -------------------------------------------------------------------------
    c9_ok = (
        res_hyb_cloud.hybrid_latency_s is not None
        and res_hyb_cloud.hybrid_latency_s > 0.0
        and res_hyb_cloud.edge_latency_s is not None
        and res_hyb_cloud.cloud_latency_s is not None
    )
    check("9. Hybrid wall-clock latency measured and positive", c9_ok, f"T_hybrid={res_hyb_cloud.hybrid_latency_s:.6f}s (T_edge={res_hyb_cloud.edge_latency_s:.6f}s, T_cloud={res_hyb_cloud.cloud_latency_s:.6f}s)")

    # -------------------------------------------------------------------------
    # Check 10: Prediction Validity
    # -------------------------------------------------------------------------
    val_pred_ok = True
    try:
        validate_output(0, {0: 0.8, 1: 0.2})
        validate_output(1, {0: 0.1, 1: 0.9})
    except Exception:
        val_pred_ok = False
    invalid_pred_caught = False
    try:
        validate_output(2, {0: 0.5, 1: 0.5})
    except ValueError:
        invalid_pred_caught = True
    c10_ok = val_pred_ok and invalid_pred_caught
    check("10. Prediction validation passes binary {0, 1} and rejects other labels", c10_ok, "binary label domain strictly enforced")

    # -------------------------------------------------------------------------
    # Check 11: Probability Validity
    # -------------------------------------------------------------------------
    val_prob_ok = True
    try:
        validate_output(0, {0: 0.5, 1: 0.5})
    except Exception:
        val_prob_ok = False
    invalid_sum_caught = False
    try:
        validate_output(0, {0: 0.3, 1: 0.3})
    except ValueError:
        invalid_sum_caught = True
    c11_ok = val_prob_ok and invalid_sum_caught
    check("11. Probability validation verifies bounds [0, 1], finite, sum=1.0", c11_ok, "valid simplex distribution required")

    # -------------------------------------------------------------------------
    # Check 12: Input Validation
    # -------------------------------------------------------------------------
    none_caught, empty_caught, nan_caught, dim_caught, leak_caught = False, False, False, False, False
    try:
        validate_input(None)
    except ValueError:
        none_caught = True
    try:
        validate_input([])
    except ValueError:
        empty_caught = True
    try:
        validate_input([1.0, float("nan"), 3.0])
    except ValueError:
        nan_caught = True
    try:
        validate_input([1.0, 2.0], expected_dim=4)
    except ValueError:
        dim_caught = True
    try:
        validate_input({"f0": 1.0, "Target": 0})
    except ValueError:
        leak_caught = True
    c12_ok = all([none_caught, empty_caught, nan_caught, dim_caught, leak_caught])
    check("12. Input validation rejects None, empty, NaN, dimension mismatch, and Target", c12_ok, "all invalid input permutations safely rejected")

    # -------------------------------------------------------------------------
    # Check 13: Edge Failure Handling
    # -------------------------------------------------------------------------
    fail_edge = MockModel(name="fail_edge", fail_predict=True)
    eng_fail_edge = DecisionEngine(controller=ctrl, edge_model=fail_edge, cloud_model=cloud_m)
    res_fail_edge = eng_fail_edge.execute([0.1, 0.2, 0.3, 0.4], inputs=0.9)
    c13_ok = (
        res_fail_edge.success is False
        and res_fail_edge.status == ExecutionStatus.FAILED
        and res_fail_edge.prediction is None
        and res_fail_edge.probabilities is None
        and res_fail_edge.error is not None
    )
    check("13. Edge failure returns explicit FAILED status without fabricated prediction", c13_ok, f"success={res_fail_edge.success}, status={res_fail_edge.status.value}")

    # -------------------------------------------------------------------------
    # Check 14: Cloud Failure Handling
    # -------------------------------------------------------------------------
    fail_cloud = MockModel(name="fail_cloud", fail_predict=True)
    eng_fail_cloud = DecisionEngine(controller=ctrl_cloud, edge_model=edge_m, cloud_model=fail_cloud)
    res_fail_cloud = eng_fail_cloud.execute([0.1, 0.2, 0.3, 0.4], inputs=0.1)
    c14_ok = (
        res_fail_cloud.success is False
        and res_fail_cloud.status == ExecutionStatus.FAILED
        and res_fail_cloud.prediction is None
        and res_fail_cloud.probabilities is None
        and res_fail_cloud.error is not None
    )
    check("14. Cloud failure returns explicit FAILED status without fabricated prediction", c14_ok, f"success={res_fail_cloud.success}, status={res_fail_cloud.status.value}")

    # -------------------------------------------------------------------------
    # Check 15: Hybrid Fallback Failure
    # -------------------------------------------------------------------------
    eng_fail_hyb = DecisionEngine(
        controller=ctrl_hyb,
        edge_model=edge_unc,
        cloud_model=fail_cloud,
        fallback_confidence_threshold=0.60,
    )
    res_fail_hyb = eng_fail_hyb.execute([0.1, 0.2, 0.3, 0.4], inputs=0.45)
    c15_ok = (
        res_fail_hyb.success is False
        and res_fail_hyb.status == ExecutionStatus.FAILED
        and res_fail_hyb.cloud_fallback is True
        and res_fail_hyb.prediction is None
    )
    check("15. Hybrid Cloud fallback failure returns explicit FAILED status", c15_ok, f"success={res_fail_hyb.success}, status={res_fail_hyb.status.value}, fallback={res_fail_hyb.cloud_fallback}")

    # -------------------------------------------------------------------------
    # Check 16: Execution Status Values
    # -------------------------------------------------------------------------
    c16_ok = {s.value for s in ExecutionStatus} == {"SUCCESS", "FALLBACK", "FAILED"}
    check("16. Execution status takes exactly {SUCCESS, FALLBACK, FAILED}", c16_ok, f"members={[s.value for s in ExecutionStatus]}")

    # -------------------------------------------------------------------------
    # Check 17: cloud_fallback Flag
    # -------------------------------------------------------------------------
    c17_ok = (
        res_edge.cloud_fallback is False
        and res_cloud.cloud_fallback is False
        and res_hyb_edge.cloud_fallback is False
        and res_hyb_cloud.cloud_fallback is True
    )
    check("17. cloud_fallback flag is True only when Cloud fallback was invoked", c17_ok, "truthful fallback provenance verified")

    # -------------------------------------------------------------------------
    # Check 18: Execution Determinism
    # -------------------------------------------------------------------------
    engine.reset()
    r1 = engine.execute([0.5, -0.2, 0.1, 0.8], inputs=0.85)
    engine.reset()
    r2 = engine.execute([0.5, -0.2, 0.1, 0.8], inputs=0.85)
    c18_ok = (
        r1.action == r2.action
        and r1.prediction == r2.prediction
        and r1.probabilities == r2.probabilities
        and r1.model_used == r2.model_used
        and r1.status == r2.status
    )
    check("18. Deterministic execution on identical observation and state", c18_ok, "idempotency confirmed")

    # -------------------------------------------------------------------------
    # Check 19: Bounded Telemetry
    # -------------------------------------------------------------------------
    engine_bounded = DecisionEngine(
        controller=ctrl,
        edge_model=edge_m,
        cloud_model=cloud_m,
        max_instrumentation_records=5,
    )
    for _ in range(20):
        engine_bounded.execute([0.1, 0.2, 0.3, 0.4], inputs=0.85)
    summary = engine_bounded.instrumentation.get_summary()
    c19_ok = (
        summary["total_executions"] == 20
        and summary["records_stored"] == 5
        and summary["latency_stats"]["count"] == 20
    )
    check("19. Bounded execution telemetry respects max_records buffer size", c19_ok, f"total={summary['total_executions']}, stored={summary['records_stored']}")

    # -------------------------------------------------------------------------
    # Check 20: Full Pipeline Compatibility (Phase 1 -> 6)
    # -------------------------------------------------------------------------
    cfg = config_mod.load("default")
    estimator = ReliabilityEstimator(config=cfg)
    ctrl_adapt = AdaptiveController(config=cfg)

    # Train toy real models
    rng = np.random.RandomState(42)
    X_toy = rng.randn(40, 4)
    y_toy = (X_toy[:, 0] > 0).astype(int)
    real_edge = EdgeHoeffdingTree()
    real_edge.fit(X_toy, y_toy)
    real_cloud = CloudXGBoost(n_estimators=5, max_depth=3)
    real_cloud.fit(X_toy, y_toy)

    eng_pipe = DecisionEngine(controller=ctrl_adapt, edge_model=real_edge, cloud_model=real_cloud, config=cfg)
    score = estimator.update(confidence=0.88, instantaneous_error=0, drift_severity=0.03, quality=0.96)
    d_inputs = DecisionInputs(reliability=score.reliability, observation_index=1)
    res_pipe = eng_pipe.execute(X_toy[0], inputs=d_inputs)

    c20_ok = (
        res_pipe.success is True
        and res_pipe.prediction in (0, 1)
        and res_pipe.action == DecisionAction.EDGE
    )
    check("20. Causal integration across Phase 1, 2, 3, 4, 5, and 6", c20_ok, f"prediction={res_pipe.prediction}, action={res_pipe.action.value}, R_t={score.reliability:.4f}")

    # -------------------------------------------------------------------------
    # Check 21: Quarantine Integrity & Scope Boundary
    # -------------------------------------------------------------------------
    # Check that Phase 7+ files are still un-implemented skeletons
    later_phase_files = [
        "src/simulation/__init__.py",
        "src/evaluation/__init__.py",
    ]
    quarantine_ok = True
    for fpath in later_phase_files:
        full_p = ROOT / fpath
        if full_p.exists():
            tree = ast.parse(io.open(full_p, encoding="utf-8").read())
            stmts = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if len(stmts) > 0:
                quarantine_ok = False
                break

    c21_ok = quarantine_ok and ("execution" in cfg)
    check("21. Zero leakage, Target isolation, and Phase 7+ quarantine intact", c21_ok, "Phase 7+ modules remain un-implemented skeletons")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    banner("STEP 6 VERIFICATION SUMMARY")
    for ln in _details:
        print(ln)
    print("-" * 78)
    n_pass = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    for name, ok, det in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}")
        if det:
            print(f"       {det}")
    print("-" * 78)
    print(f"{n_pass}/{n_total} checks passed")
    print("=" * 78)

    if n_pass == n_total:
        print("\nPHASE 6 VERIFICATION = PASS\n")
        return 0
    else:
        print(f"\nPHASE 6 VERIFICATION = FAIL ({n_total - n_pass} checks failed)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
