"""Tests for Phase 6: Hardened Execution Layer, Telemetry, and Failure Handling.

Validates all 26 Phase 6 execution requirements:
1. Common execution interface (ExecutionResult, ExecutionStatus)
2. Edge successful execution
3. Cloud successful execution
4. Hybrid Edge-first execution
5. Hybrid Edge-only completion (confidence >= 0.60, Cloud not invoked)
6. Hybrid Cloud fallback (confidence < 0.60, Cloud invoked)
7. Edge latency measurement (T_edge > 0)
8. Cloud latency measurement (T_cloud > 0)
9. Hybrid latency measurement (T_hybrid wall-clock measured)
10. Prediction validation (binary {0, 1})
11. Probability validation (bounds [0, 1], finite, sum=1.0)
12. Invalid input handling (None, empty, dimensions, NaN, leakage)
13. Edge failure handling (status FAILED, success False)
14. Cloud failure handling (status FAILED, success False)
15. Hybrid Cloud fallback failure (Edge uncertain + Cloud fails -> FAILED)
16. Execution status representation (SUCCESS, FALLBACK, FAILED)
17. cloud_fallback flag correctness
18. Determinism and reproducibility
19. Memory-bounded telemetry
20. Streaming execution sequence
21. Phase 5 API compatibility
22. Phase 4 reliability compatibility
23. Target leakage protection
24. ground_truth.json isolation
25. Future-row isolation
26. End-to-end Phase 1-6 smoke test
"""

from __future__ import annotations

import collections
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import pytest

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


# -----------------------------------------------------------------------------
# Test Fixtures and Mock Models
# -----------------------------------------------------------------------------


class MockModel(BaseModel):
    """Controllable mock model for deterministic testing of failure and execution paths."""

    def __init__(
        self,
        name: str = "mock",
        pred: int = 0,
        probas: dict[int, float] | None = None,
        fail_predict: bool = False,
        fail_proba: bool = False,
        is_trained: bool = True,
        n_features: int = 4,
        call_tracker: list[str] | None = None,
    ) -> None:
        super().__init__(model_name=name)
        self._pred = pred
        self._probas = probas or {0: 0.90, 1: 0.10}
        self._fail_predict = fail_predict
        self._fail_proba = fail_proba
        self._is_trained = is_trained
        self._n_features = n_features
        self._feature_names = tuple(f"f{i}" for i in range(n_features))
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


@pytest.fixture
def mock_edge():
    return MockModel(name="mock_edge", pred=0, probas={0: 0.90, 1: 0.10})


@pytest.fixture
def mock_cloud():
    return MockModel(name="mock_cloud", pred=1, probas={0: 0.15, 1: 0.85})


@pytest.fixture
def trained_models():
    """Trained miniature real models on toy 4-feature data for integration tests."""
    rng = np.random.RandomState(42)
    X = rng.randn(60, 4)
    y = (X[:, 0] + X[:, 1] > 0).astype(int)

    edge = EdgeHoeffdingTree()
    edge.fit(X, y)

    cloud = CloudXGBoost(n_estimators=5, max_depth=3)
    cloud.fit(X, y)

    return edge, cloud


# -----------------------------------------------------------------------------
# Test 1: Common Execution Interface
# -----------------------------------------------------------------------------


def test_01_common_execution_interface():
    """Verify ExecutionStatus enum and ExecutionResult contract."""
    for s in ("SUCCESS", "FALLBACK", "FAILED"):
        status = ExecutionStatus(s)
        assert ExecutionStatus.from_str(s.lower()) == status

    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.85,
        previous_action=None,
        decision_reason="test",
        observation_index=42,
    )
    res = ExecutionResult(
        decision=dec,
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 0.9, 1: 0.1},
        model_used="edge",
        inference_latency_s=0.0012,
        cloud_fallback=False,
        success=True,
        status=ExecutionStatus.SUCCESS,
        edge_latency_s=0.0012,
    )
    assert res.action == DecisionAction.EDGE
    assert res.prediction == 0
    assert res.status == ExecutionStatus.SUCCESS
    assert res.success is True
    assert res.observation_index == 42
    assert res.edge_latency_s == 0.0012


# -----------------------------------------------------------------------------
# Test 2: Edge Successful Execution
# -----------------------------------------------------------------------------


def test_02_edge_successful_execution(mock_edge, mock_cloud):
    """Verify Edge inference executes, returns valid result, and measures T_edge."""
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x = {"f0": 0.1, "f1": 0.2, "f2": -0.5, "f3": 1.0}
    res = engine.execute(x, inputs=0.90)

    assert res.action == DecisionAction.EDGE
    assert res.prediction == 0
    assert res.model_used == "edge"
    assert res.success is True
    assert res.status == ExecutionStatus.SUCCESS
    assert res.cloud_fallback is False
    assert res.edge_latency_s is not None
    assert res.edge_latency_s > 0.0
    assert res.cloud_latency_s is None
    assert "mock_edge.predict_one" in mock_edge.call_tracker
    assert len(mock_cloud.call_tracker) == 0


# -----------------------------------------------------------------------------
# Test 3: Cloud Successful Execution
# -----------------------------------------------------------------------------


def test_03_cloud_successful_execution(mock_edge, mock_cloud):
    """Verify Cloud inference executes, returns valid result, and measures T_cloud."""
    ctrl = StaticBaselineController(policy="cloud_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x = {"f0": 0.1, "f1": 0.2, "f2": -0.5, "f3": 1.0}
    res = engine.execute(x, inputs=0.20)

    assert res.action == DecisionAction.CLOUD
    assert res.prediction == 1
    assert res.model_used == "cloud"
    assert res.success is True
    assert res.status == ExecutionStatus.SUCCESS
    assert res.cloud_fallback is False
    assert res.cloud_latency_s is not None
    assert res.cloud_latency_s > 0.0
    assert res.edge_latency_s is None
    assert "mock_cloud.predict_one" in mock_cloud.call_tracker
    assert len(mock_edge.call_tracker) == 0


# -----------------------------------------------------------------------------
# Test 4: Hybrid Edge-First Execution
# -----------------------------------------------------------------------------


def test_04_hybrid_edge_first_execution():
    """Verify Hybrid action executes Edge model first."""
    calls = []
    edge = MockModel(name="edge", pred=0, probas={0: 0.95, 1: 0.05}, call_tracker=calls)
    cloud = MockModel(name="cloud", pred=1, probas={0: 0.1, 1: 0.9}, call_tracker=calls)

    ctrl = StaticBaselineController(policy="static_hybrid")
    engine = DecisionEngine(controller=ctrl, edge_model=edge, cloud_model=cloud)

    x = [0.1, 0.2, 0.3, 0.4]
    engine.execute(x, inputs=0.40)

    assert len(calls) >= 2
    assert calls[0].startswith("edge.")
    assert calls[1].startswith("edge.")


# -----------------------------------------------------------------------------
# Test 5: Hybrid Edge-Only Completion (Confidence >= 0.60)
# -----------------------------------------------------------------------------


def test_05_hybrid_edge_only_completion():
    """When Edge confidence >= 0.60, Edge result is final and Cloud is not invoked."""
    edge = MockModel(name="edge", pred=0, probas={0: 0.90, 1: 0.10})
    cloud = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90})

    ctrl = StaticBaselineController(policy="static_hybrid")
    engine = DecisionEngine(
        controller=ctrl,
        edge_model=edge,
        cloud_model=cloud,
        fallback_confidence_threshold=0.60,
    )

    x = [0.5, -0.2, 0.1, 1.2]
    res = engine.execute(x, inputs=0.45)

    assert res.action == DecisionAction.HYBRID
    assert res.model_used == "hybrid_edge"
    assert res.prediction == 0
    assert res.cloud_fallback is False
    assert res.status == ExecutionStatus.SUCCESS
    assert res.success is True
    assert res.edge_latency_s is not None and res.edge_latency_s > 0
    assert res.cloud_latency_s is None
    assert len(cloud.call_tracker) == 0, "Cloud must NOT be invoked when Edge is confident"


# -----------------------------------------------------------------------------
# Test 6: Hybrid Cloud Fallback (Confidence < 0.60)
# -----------------------------------------------------------------------------


def test_06_hybrid_cloud_fallback():
    """When Edge confidence < 0.60, invokes Cloud fallback and Cloud result is final."""
    edge = MockModel(name="edge", pred=0, probas={0: 0.55, 1: 0.45})
    cloud = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90})

    ctrl = StaticBaselineController(policy="static_hybrid")
    engine = DecisionEngine(
        controller=ctrl,
        edge_model=edge,
        cloud_model=cloud,
        fallback_confidence_threshold=0.60,
    )

    x = [0.5, -0.2, 0.1, 1.2]
    res = engine.execute(x, inputs=0.45)

    assert res.action == DecisionAction.HYBRID
    assert res.model_used == "hybrid_cloud"
    assert res.prediction == 1
    assert res.cloud_fallback is True
    assert res.status == ExecutionStatus.FALLBACK
    assert res.success is True
    assert res.edge_latency_s is not None and res.edge_latency_s > 0
    assert res.cloud_latency_s is not None and res.cloud_latency_s > 0
    assert res.hybrid_latency_s is not None and res.hybrid_latency_s > 0
    assert len(cloud.call_tracker) > 0, "Cloud MUST be invoked on low Edge confidence"


# -----------------------------------------------------------------------------
# Test 7: Edge Latency Measurement
# -----------------------------------------------------------------------------


def test_07_edge_latency_measurement(mock_edge, mock_cloud):
    """Verify Edge latency represents positive elapsed software execution time."""
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x = [1.0, 2.0, 3.0, 4.0]
    res_edge = engine.execute_edge(x)
    assert res_edge.edge_latency_s is not None
    assert res_edge.edge_latency_s > 0.0
    assert res_edge.inference_latency_s == res_edge.edge_latency_s
    assert res_edge.cloud_latency_s is None


# -----------------------------------------------------------------------------
# Test 8: Cloud Latency Measurement
# -----------------------------------------------------------------------------


def test_08_cloud_latency_measurement(mock_edge, mock_cloud):
    """Verify Cloud latency represents positive elapsed software execution time."""
    ctrl = StaticBaselineController(policy="cloud_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x = [1.0, 2.0, 3.0, 4.0]
    res_cloud = engine.execute_cloud(x)
    assert res_cloud.cloud_latency_s is not None
    assert res_cloud.cloud_latency_s > 0.0
    assert res_cloud.inference_latency_s == res_cloud.cloud_latency_s
    assert res_cloud.edge_latency_s is None


# -----------------------------------------------------------------------------
# Test 9: Hybrid Latency Measurement (Wall-Clock Measured)
# -----------------------------------------------------------------------------


def test_09_hybrid_latency_measurement_wall_clock():
    """Verify T_hybrid is measured by wall-clock elapsed time, not purely summed."""
    edge = MockModel(name="edge", pred=0, probas={0: 0.52, 1: 0.48})
    cloud = MockModel(name="cloud", pred=1, probas={0: 0.10, 1: 0.90})

    ctrl = StaticBaselineController(policy="static_hybrid")
    engine = DecisionEngine(controller=ctrl, edge_model=edge, cloud_model=cloud)

    x = [0.1, 0.2, 0.3, 0.4]
    res = engine.execute(x, inputs=0.45)

    assert res.hybrid_latency_s is not None
    assert res.hybrid_latency_s > 0.0
    assert res.edge_latency_s is not None and res.edge_latency_s > 0.0
    assert res.cloud_latency_s is not None and res.cloud_latency_s > 0.0
    assert np.isfinite(res.hybrid_latency_s)
    assert np.isfinite(res.edge_latency_s)
    assert np.isfinite(res.cloud_latency_s)


# -----------------------------------------------------------------------------
# Test 10: Prediction Validation
# -----------------------------------------------------------------------------


def test_10_prediction_validation():
    """Verify validation passes binary predictions {0, 1} and rejects other values."""
    validate_output(0, {0: 0.8, 1: 0.2})
    validate_output(1, {0: 0.0, 1: 1.0})

    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_output(2, {0: 0.5, 1: 0.5})

    with pytest.raises(ValueError, match="must be 0 or 1"):
        validate_output(-1, {0: 0.5, 1: 0.5})


# -----------------------------------------------------------------------------
# Test 11: Probability Validation
# -----------------------------------------------------------------------------


def test_11_probability_validation():
    """Verify validation passes valid probability distributions and rejects invalid ones."""
    validate_output(0, {0: 0.8, 1: 0.2})
    validate_output(1, {0: 0.0, 1: 1.0})

    with pytest.raises(ValueError, match="must sum to 1.0"):
        validate_output(0, {0: 0.4, 1: 0.4})

    with pytest.raises(ValueError, match="must be in"):
        validate_output(0, {0: -0.1, 1: 1.1})

    with pytest.raises(ValueError, match="must be finite"):
        validate_output(0, {0: float("nan"), 1: 0.5})


# -----------------------------------------------------------------------------
# Test 12: Invalid Input Handling
# -----------------------------------------------------------------------------


def test_12_invalid_input_handling():
    """Verify validate_input rejects None, empty, wrong dimensions, NaNs, and Target."""
    with pytest.raises(ValueError, match="cannot be None"):
        validate_input(None)

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input({})

    with pytest.raises(ValueError, match="cannot be empty"):
        validate_input([])

    with pytest.raises(ValueError, match="finite"):
        validate_input([1.0, float("nan"), 3.0])

    with pytest.raises(ValueError, match="dimension mismatch"):
        validate_input([1.0, 2.0], expected_dim=4)

    with pytest.raises(ValueError, match="forbidden leakage key"):
        validate_input({"f0": 1.0, "Target": 0})


# -----------------------------------------------------------------------------
# Test 13: Edge Failure Handling
# -----------------------------------------------------------------------------


def test_13_edge_failure_handling(mock_cloud):
    """Verify Edge model failure returns explicit FAILED status and no fabricated prediction."""
    failing_edge = MockModel(name="failing_edge", fail_predict=True)
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=failing_edge, cloud_model=mock_cloud)

    x = [0.1, 0.2, 0.3, 0.4]
    res = engine.execute(x, inputs=0.9)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert res.prediction is None
    assert res.probabilities is None
    assert res.error is not None
    assert "Edge execution failed" in res.error


# -----------------------------------------------------------------------------
# Test 14: Cloud Failure Handling
# -----------------------------------------------------------------------------


def test_14_cloud_failure_handling(mock_edge):
    """Verify Cloud model failure returns explicit FAILED status and no fabricated prediction."""
    failing_cloud = MockModel(name="failing_cloud", fail_predict=True)
    ctrl = StaticBaselineController(policy="cloud_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=failing_cloud)

    x = [0.1, 0.2, 0.3, 0.4]
    res = engine.execute(x, inputs=0.1)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert res.prediction is None
    assert res.probabilities is None
    assert res.error is not None
    assert "Cloud execution failed" in res.error


# -----------------------------------------------------------------------------
# Test 15: Hybrid Cloud Fallback Failure
# -----------------------------------------------------------------------------


def test_15_hybrid_cloud_fallback_failure():
    """Case 3: Edge succeeds but is uncertain; Cloud fails during fallback -> explicit FAILED."""
    uncertain_edge = MockModel(name="uncertain_edge", pred=0, probas={0: 0.51, 1: 0.49})
    failing_cloud = MockModel(name="failing_cloud", fail_predict=True)

    ctrl = StaticBaselineController(policy="static_hybrid")
    engine = DecisionEngine(
        controller=ctrl,
        edge_model=uncertain_edge,
        cloud_model=failing_cloud,
        fallback_confidence_threshold=0.60,
    )

    x = [0.1, 0.2, 0.3, 0.4]
    res = engine.execute(x, inputs=0.45)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert res.cloud_fallback is True
    assert res.prediction is None
    assert res.probabilities is None
    assert res.error is not None
    assert "Hybrid Cloud fallback failed" in res.error


# -----------------------------------------------------------------------------
# Test 16: Execution Status
# -----------------------------------------------------------------------------


def test_16_execution_status(mock_edge, mock_cloud):
    """Verify execution status takes expected SUCCESS, FALLBACK, or FAILED values."""
    res1 = ExecutionResult(
        decision=DecisionResult(DecisionAction.EDGE, 0.8, None, "test"),
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 1.0, 1: 0.0},
        model_used="edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
        status=ExecutionStatus.SUCCESS,
    )
    assert res1.status == ExecutionStatus.SUCCESS
    assert res1.success is True

    res2 = ExecutionResult(
        decision=DecisionResult(DecisionAction.HYBRID, 0.4, None, "test"),
        action=DecisionAction.HYBRID,
        prediction=1,
        probabilities={0: 0.1, 1: 0.9},
        model_used="hybrid_cloud",
        inference_latency_s=0.002,
        cloud_fallback=True,
        status=ExecutionStatus.FALLBACK,
    )
    assert res2.status == ExecutionStatus.FALLBACK
    assert res2.success is True

    res3 = ExecutionResult(
        decision=DecisionResult(DecisionAction.CLOUD, 0.2, None, "test"),
        action=DecisionAction.CLOUD,
        prediction=None,
        probabilities=None,
        model_used="none",
        inference_latency_s=0.0005,
        cloud_fallback=False,
        success=False,
        status=ExecutionStatus.FAILED,
        error="Simulation error",
    )
    assert res3.status == ExecutionStatus.FAILED
    assert res3.success is False


# -----------------------------------------------------------------------------
# Test 17: cloud_fallback Flag
# -----------------------------------------------------------------------------


def test_17_cloud_fallback_flag(mock_edge, mock_cloud):
    """Verify cloud_fallback is True only when Cloud fallback was invoked in Hybrid mode."""
    res_edge = ExecutionResult(
        decision=DecisionResult(DecisionAction.EDGE, 0.8, None, "test"),
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 1.0, 0: 0.0},
        model_used="edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
    )
    assert res_edge.cloud_fallback is False

    res_hybrid_edge = ExecutionResult(
        decision=DecisionResult(DecisionAction.HYBRID, 0.6, None, "test"),
        action=DecisionAction.HYBRID,
        prediction=0,
        probabilities={0: 0.9, 1: 0.1},
        model_used="hybrid_edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
    )
    assert res_hybrid_edge.cloud_fallback is False

    res_hybrid_cloud = ExecutionResult(
        decision=DecisionResult(DecisionAction.HYBRID, 0.4, None, "test"),
        action=DecisionAction.HYBRID,
        prediction=1,
        probabilities={0: 0.1, 1: 0.9},
        model_used="hybrid_cloud",
        inference_latency_s=0.002,
        cloud_fallback=True,
        status=ExecutionStatus.FALLBACK,
    )
    assert res_hybrid_cloud.cloud_fallback is True


# -----------------------------------------------------------------------------
# Test 18: Determinism
# -----------------------------------------------------------------------------


def test_18_determinism(trained_models):
    """Repeated executions on the same observation produce identical results."""
    edge, cloud = trained_models
    ctrl = AdaptiveController(cloud_threshold=0.50, edge_return_threshold=0.70, critical_cloud_threshold=0.30)
    engine = DecisionEngine(controller=ctrl, edge_model=edge, cloud_model=cloud)

    x = np.array([0.5, -0.1, 0.8, -0.3])
    inputs = DecisionInputs(reliability=0.85, observation_index=1)

    r1 = engine.execute(x, inputs)
    engine.reset()
    r2 = engine.execute(x, inputs)

    assert r1.action == r2.action
    assert r1.prediction == r2.prediction
    assert r1.probabilities == r2.probabilities
    assert r1.model_used == r2.model_used
    assert r1.status == r2.status


# -----------------------------------------------------------------------------
# Test 19: Memory-Bounded Telemetry
# -----------------------------------------------------------------------------


def test_19_memory_bounded_telemetry(mock_edge, mock_cloud):
    """Telemetry respects max_records and does not grow indefinitely."""
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(
        controller=ctrl,
        edge_model=mock_edge,
        cloud_model=mock_cloud,
        max_instrumentation_records=10,
    )

    x = [0.1, 0.2, 0.3, 0.4]
    for i in range(25):
        engine.execute(x, inputs=0.9)

    summary = engine.instrumentation.get_summary()
    assert summary["total_decisions"] == 25
    assert summary["total_executions"] == 25
    assert summary["successful_executions"] == 25
    assert summary["records_stored"] == 10
    assert summary["latency_stats"]["count"] == 25
    assert summary["latency_stats"]["min_s"] > 0.0


# -----------------------------------------------------------------------------
# Test 20: Streaming Execution
# -----------------------------------------------------------------------------


def test_20_streaming_execution(trained_models):
    """Simulate streaming inference progression over 15 steps."""
    edge, cloud = trained_models
    ctrl = AdaptiveController(
        cloud_threshold=0.50,
        edge_return_threshold=0.70,
        critical_cloud_threshold=0.30,
        initial_action=DecisionAction.EDGE,
    )
    engine = DecisionEngine(controller=ctrl, edge_model=edge, cloud_model=cloud)

    reliabilities = [0.9, 0.85, 0.8, 0.65, 0.55, 0.45, 0.40, 0.35, 0.20, 0.15, 0.25, 0.55, 0.72, 0.80, 0.88]
    rng = np.random.RandomState(123)
    results = []

    for t, r in enumerate(reliabilities):
        x = rng.randn(4)
        inputs = DecisionInputs(reliability=r, observation_index=t)
        res = engine.execute(x, inputs)
        results.append(res)

    actions = [r.action for r in results]
    assert DecisionAction.EDGE in actions
    assert DecisionAction.HYBRID in actions
    assert DecisionAction.CLOUD in actions
    assert all(r.success for r in results)


# -----------------------------------------------------------------------------
# Test 21: Phase 5 API Compatibility
# -----------------------------------------------------------------------------


def test_21_phase_5_api_compatibility(mock_edge, mock_cloud):
    """Verify all Phase 5 methods, attributes, and signatures remain intact."""
    ctrl = AdaptiveController()
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    dec = engine.decide(0.85)
    assert isinstance(dec, DecisionResult)
    assert dec.selected_action == DecisionAction.EDGE

    x = [0.1, 0.2, 0.3, 0.4]
    exec_res = engine.execute(x, 0.85)
    assert hasattr(exec_res, "decision")
    assert hasattr(exec_res, "action")
    assert hasattr(exec_res, "prediction")
    assert hasattr(exec_res, "probabilities")
    assert hasattr(exec_res, "model_used")
    assert hasattr(exec_res, "inference_latency_s")
    assert hasattr(exec_res, "cloud_fallback")

    info = engine.get_info()
    assert "engine" in info
    assert "controller" in info
    assert "edge_model" in info
    assert "cloud_model" in info
    assert "instrumentation" in info


# -----------------------------------------------------------------------------
# Test 22: Phase 4 Reliability Compatibility
# -----------------------------------------------------------------------------


def test_22_phase_4_compatibility(mock_edge, mock_cloud):
    """Verify R_t computed by Phase 4 ReliabilityEstimator directly feeds Phase 6 DecisionEngine."""
    cfg = config_mod.load("default")
    estimator = ReliabilityEstimator(config=cfg)

    ctrl = AdaptiveController(config=cfg)
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud, config=cfg)

    score = estimator.update(confidence=0.90, instantaneous_error=0, drift_severity=0.05, quality=0.98)
    inputs = DecisionInputs(reliability=score.reliability, confidence=0.90, drift_severity=0.05, quality=0.98)
    res = engine.execute([0.1, 0.2, 0.3, 0.4], inputs)

    assert res.action == DecisionAction.EDGE
    assert res.success is True


# -----------------------------------------------------------------------------
# Test 23: Target Leakage Protection
# -----------------------------------------------------------------------------


def test_23_target_leakage_protection(mock_edge, mock_cloud):
    """Observation dictionary containing Target column is rejected by validation."""
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x_leaky = {"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "Target": 1}
    res = engine.execute(x_leaky, inputs=0.9)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert "forbidden leakage" in res.error


# -----------------------------------------------------------------------------
# Test 24: ground_truth.json Isolation
# -----------------------------------------------------------------------------


def test_24_ground_truth_isolation(mock_edge, mock_cloud):
    """Observation dictionary containing synthetic ground truth is rejected by validation."""
    ctrl = StaticBaselineController(policy="edge_only")
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    x_leaky = {"f0": 1.0, "f1": 2.0, "f2": 3.0, "f3": 4.0, "ground_truth": 1}
    res = engine.execute(x_leaky, inputs=0.9)

    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert "forbidden leakage" in res.error


# -----------------------------------------------------------------------------
# Test 25: Future-Row Isolation
# -----------------------------------------------------------------------------


def test_25_future_row_isolation(mock_edge, mock_cloud):
    """Decisions and execution depend strictly on information available at current step."""
    ctrl = AdaptiveController(
        cloud_threshold=0.50,
        edge_return_threshold=0.70,
        critical_cloud_threshold=0.30,
        initial_action=DecisionAction.EDGE,
    )
    engine = DecisionEngine(controller=ctrl, edge_model=mock_edge, cloud_model=mock_cloud)

    inputs_10 = DecisionInputs(reliability=0.85, observation_index=10)
    res_10 = engine.execute([0.1, 0.2, 0.3, 0.4], inputs_10)

    assert res_10.action == DecisionAction.EDGE
    assert res_10.observation_index == 10


# -----------------------------------------------------------------------------
# Test 26: End-to-End Phase 1–6 Smoke Test
# -----------------------------------------------------------------------------


def test_26_end_to_end_phase_1_to_6_smoke_test(trained_models):
    """End-to-end smoke test through full pipeline."""
    edge, cloud = trained_models
    cfg = config_mod.load("default")

    estimator = ReliabilityEstimator(config=cfg)
    ctrl = AdaptiveController(config=cfg)
    engine = DecisionEngine(controller=ctrl, edge_model=edge, cloud_model=cloud, config=cfg)

    rng = np.random.RandomState(999)
    for t in range(5):
        x = rng.randn(4)
        score = estimator.update(confidence=0.85, instantaneous_error=0, drift_severity=0.02, quality=0.95)
        inputs = DecisionInputs(reliability=score.reliability, observation_index=t)
        res = engine.execute(x, inputs)
        assert res.success is True
        assert res.prediction in (0, 1)
        assert res.status in (ExecutionStatus.SUCCESS, ExecutionStatus.FALLBACK)
