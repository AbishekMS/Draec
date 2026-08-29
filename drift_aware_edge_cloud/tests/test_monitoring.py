"""Tests for Phase 7: DRAEC Model Management & Monitoring.

Verifies:
1. Model registry creation and metadata tracking (Edge & Cloud)
2. Version tracking and feature contract inspection
3. Causal monitoring of Phase 4 Reliability, Phase 3 Drift, Phase 5 Decisions, and Phase 6 Execution
4. Bounded history (ring buffer) vs global streaming statistics
5. Routing distributions, hybrid fallback tracking, and streaming latency statistics
6. Observational health states and non-actionable alert flags
7. Phase 10 data-readiness (stable DataFrame schema)
8. Causal isolation, zero target leakage, and zero ground truth dependency
"""

import time
import numpy as np
import pandas as pd
import pytest

from src.decision.base import DecisionAction, DecisionInputs, DecisionResult, ExecutionResult, ExecutionStatus
from src.decision.engine import DecisionEngine
from src.models.base import BaseModel
from src.monitoring.base import (
    ModelHealthStatus,
    ModelMetadata,
    MonitoringRecord,
    MonitoringSnapshot,
    StreamStatistics,
)
from src.monitoring.monitor import DRAECMonitor, SystemMonitor
from src.monitoring.registry import ModelRegistry
from src.reliability.base import ReliabilityFactors, ReliabilityInputs, ReliabilityScore


class MockEdgeModel(BaseModel):
    """Deterministic mock edge model adhering to BaseModel."""

    def __init__(self, name: str = "edge_hoeffding_test") -> None:
        super().__init__(model_name=name)
        self._is_trained = True
        self._n_features = 37
        self._feature_names = tuple(f"feat_{i}" for i in range(37))
        self._classes = (0, 1)

    def fit(self, X: Any, y: Any) -> None:
        pass

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.ones(n) * 0.9, np.ones(n) * 0.1])

    def predict_one(self, x: Any) -> int:
        return 0

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        return {0: 0.9, 1: 0.1}

    def get_info(self) -> dict[str, Any]:
        return {"name": self.model_name, "is_trained": True}


class MockCloudModel(BaseModel):
    """Deterministic mock cloud model adhering to BaseModel."""

    def __init__(self, name: str = "cloud_xgboost_test") -> None:
        super().__init__(model_name=name)
        self._is_trained = True
        self._n_features = 37
        self._feature_names = tuple(f"feat_{i}" for i in range(37))
        self._classes = (0, 1)

    def fit(self, X: Any, y: Any) -> None:
        pass

    def predict(self, X: Any) -> np.ndarray:
        return np.zeros(len(X), dtype=int)

    def predict_proba(self, X: Any) -> np.ndarray:
        n = len(X)
        return np.column_stack([np.ones(n) * 0.95, np.ones(n) * 0.05])

    def predict_one(self, x: Any) -> int:
        return 0

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        return {0: 0.95, 1: 0.05}

    def get_info(self) -> dict[str, Any]:
        return {"name": self.model_name, "is_trained": True}


def make_dummy_reliability_score(r: float, c: float = 0.8, e: float = 0.05, d: float = 0.1, q: float = 1.0) -> ReliabilityScore:
    inp = ReliabilityInputs(confidence=c, error=e, drift=d, quality=q)
    fac = ReliabilityFactors(r_C=c, r_E=1.0 - e, r_D=1.0 - d, r_Q=q)
    return ReliabilityScore(
        reliability=r,
        inputs=inp,
        factors=fac,
        weights={"confidence": 0.25, "error": 0.25, "drift": 0.25, "quality": 0.25},
        epsilon=1e-8,
    )


# -----------------------------------------------------------------------------
# Test 01-06: Model Management & Registry
# -----------------------------------------------------------------------------


def test_01_monitoring_component_imports():
    """Verify all required monitoring classes import cleanly and alias works."""
    assert ModelHealthStatus is not None
    assert ModelMetadata is not None
    assert MonitoringRecord is not None
    assert MonitoringSnapshot is not None
    assert StreamStatistics is not None
    assert ModelRegistry is not None
    assert DRAECMonitor is not None
    assert SystemMonitor is DRAECMonitor


def test_02_model_registry_creation():
    """Verify ModelRegistry initializes empty and handles missing models cleanly."""
    reg = ModelRegistry()
    assert len(reg.list_models()) == 0
    assert not reg.has_model("edge")
    with pytest.raises(KeyError):
        reg.get_model("edge")
    with pytest.raises(KeyError):
        reg.get_metadata("edge")


def test_03_edge_model_registration():
    """Verify registration of Edge model with feature contract and execution location."""
    reg = ModelRegistry()
    edge_m = MockEdgeModel()
    meta = reg.register_model(edge_m, model_id="edge", execution_location="edge", version="1.0.0")

    assert meta.model_id == "edge"
    assert meta.execution_location == "edge"
    assert meta.model_version == "1.0.0"
    assert meta.n_features == 37
    assert len(meta.feature_names) == 37
    assert meta.status == ModelHealthStatus.HEALTHY
    assert reg.has_model("edge")
    assert reg.get_model("edge") is edge_m


def test_04_cloud_model_registration():
    """Verify registration of Cloud model with execution location 'cloud'."""
    reg = ModelRegistry()
    cloud_m = MockCloudModel()
    meta = reg.register_model(cloud_m, model_id="cloud", execution_location="cloud", version="2.1.0")

    assert meta.model_id == "cloud"
    assert meta.execution_location == "cloud"
    assert meta.model_version == "2.1.0"
    assert reg.get_metadata("cloud").status == ModelHealthStatus.HEALTHY


def test_05_model_state_tracking():
    """Verify updating model status, active state, and recording executions."""
    reg = ModelRegistry()
    edge_m = MockEdgeModel()
    reg.register_model(edge_m, model_id="edge", execution_location="edge")

    reg.record_execution("edge", success=True, latency_s=0.0012, status="SUCCESS")
    meta = reg.get_metadata("edge")
    assert meta.total_executions == 1
    assert meta.successful_executions == 1
    assert meta.failed_executions == 0
    assert meta.last_latency_s == 0.0012

    reg.record_execution("edge", success=False, latency_s=0.0020, status="FAILED", error="Memory error")
    assert meta.total_executions == 2
    assert meta.successful_executions == 1
    assert meta.failed_executions == 1
    assert meta.last_error == "Memory error"

    reg.update_status("edge", ModelHealthStatus.DEGRADED)
    assert reg.get_metadata("edge").status == ModelHealthStatus.DEGRADED
    reg.set_active("edge", False)
    assert reg.get_metadata("edge").active is False


def test_06_model_version_tracking():
    """Verify model registry tracks distinct version strings without mutation."""
    reg = ModelRegistry()
    m1 = MockEdgeModel("edge_v1")
    m2 = MockEdgeModel("edge_v2")

    reg.register_model(m1, model_id="edge_v1", execution_location="edge", version="1.0.0")
    reg.register_model(m2, model_id="edge_v2", execution_location="edge", version="1.1.0")

    assert reg.get_metadata("edge_v1").model_version == "1.0.0"
    assert reg.get_metadata("edge_v2").model_version == "1.1.0"


# -----------------------------------------------------------------------------
# Test 07-16: Signals, Routing & Execution Monitoring
# -----------------------------------------------------------------------------


def test_07_reliability_monitoring():
    """Verify Phase 4 ReliabilityScore is recorded unchanged without recomputation."""
    monitor = DRAECMonitor()
    score = make_dummy_reliability_score(r=0.785, c=0.92, e=0.04, d=0.15, q=0.98)

    rec = monitor.observe_step(observation_index=0, reliability_score=score)
    assert rec.reliability == pytest.approx(0.785)
    assert rec.confidence == pytest.approx(0.92)
    assert rec.error_ema == pytest.approx(0.04)
    assert rec.drift_severity == pytest.approx(0.15)
    assert rec.quality == pytest.approx(0.98)


def test_08_drift_monitoring():
    """Verify Phase 3 drift outputs are observed and stored correctly."""
    monitor = DRAECMonitor()
    drift_info = {
        "drift_detected": True,
        "is_persistent": True,
        "raw_severity": 0.45,
        "smoothed_severity": 0.38,
    }
    rec = monitor.observe_step(observation_index=1, drift_status=drift_info)
    assert rec.drift_detected is True
    assert rec.is_persistent is True
    assert rec.raw_severity == pytest.approx(0.45)
    assert rec.smoothed_severity == pytest.approx(0.38)


def test_09_decision_monitoring():
    """Verify Phase 5 DecisionResult is observed and recorded without modifying decisions."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.82,
        previous_action=DecisionAction.CLOUD,
        decision_reason="R_t >= return_threshold",
        switch_count=2,
        observation_index=10,
        timestamp=time.time(),
    )
    rec = monitor.observe_step(observation_index=10, decision_result=dec)
    assert rec.selected_action == "EDGE"
    assert rec.previous_action == "CLOUD"
    assert rec.decision_reason == "R_t >= return_threshold"


def test_10_edge_routing_count():
    """Verify edge action increments edge counter and updates routing distribution."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.85,
        previous_action=None,
        decision_reason="Normal",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    monitor.observe_step(observation_index=0, decision_result=dec)
    snap = monitor.get_snapshot()
    assert snap.routing_counts["EDGE"] == 1
    assert snap.routing_counts["total"] == 1
    assert snap.routing_distribution["EDGE"] == pytest.approx(100.0)


def test_11_cloud_routing_count():
    """Verify cloud action increments cloud counter and updates routing distribution."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.CLOUD,
        reliability=0.25,
        previous_action=None,
        decision_reason="Severe drift",
        switch_count=1,
        observation_index=1,
        timestamp=time.time(),
    )
    monitor.observe_step(observation_index=1, decision_result=dec)
    snap = monitor.get_snapshot()
    assert snap.routing_counts["CLOUD"] == 1
    assert snap.routing_counts["total"] == 1
    assert snap.routing_distribution["CLOUD"] == pytest.approx(100.0)


def test_12_hybrid_routing_count():
    """Verify hybrid action increments hybrid counter and calculates distribution."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.HYBRID,
        reliability=0.45,
        previous_action=None,
        decision_reason="Uncertainty",
        switch_count=1,
        observation_index=2,
        timestamp=time.time(),
    )
    monitor.observe_step(observation_index=2, decision_result=dec)
    snap = monitor.get_snapshot()
    assert snap.routing_counts["HYBRID"] == 1
    assert snap.routing_counts["total"] == 1
    assert snap.routing_distribution["HYBRID"] == pytest.approx(100.0)


def test_13_hybrid_fallback_tracking():
    """Verify hybrid fallback count and rate are correctly tracked."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.HYBRID,
        reliability=0.45,
        previous_action=None,
        decision_reason="Uncertainty",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    # Execution 1: Hybrid without fallback
    res1 = ExecutionResult(
        decision=dec,
        action=DecisionAction.HYBRID,
        prediction=0,
        probabilities={0: 0.9, 1: 0.1},
        model_used="hybrid_edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
        success=True,
        status=ExecutionStatus.SUCCESS,
        edge_latency_s=0.001,
        hybrid_latency_s=0.0011,
    )
    monitor.observe_step(observation_index=0, execution_result=res1)

    # Execution 2: Hybrid with fallback
    res2 = ExecutionResult(
        decision=dec,
        action=DecisionAction.HYBRID,
        prediction=1,
        probabilities={0: 0.1, 1: 0.9},
        model_used="hybrid_cloud",
        inference_latency_s=0.005,
        cloud_fallback=True,
        success=True,
        status=ExecutionStatus.FALLBACK,
        edge_latency_s=0.001,
        cloud_latency_s=0.004,
        hybrid_latency_s=0.0052,
    )
    monitor.observe_step(observation_index=1, execution_result=res2)

    snap = monitor.get_snapshot()
    assert snap.hybrid_stats["executions"] == 2
    assert snap.hybrid_stats["fallbacks"] == 1
    assert snap.hybrid_stats["fallback_rate"] == pytest.approx(0.50)


def test_14_execution_success_tracking():
    """Verify successful executions update total and success counters."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.8,
        previous_action=None,
        decision_reason="Normal",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    res = ExecutionResult(
        decision=dec,
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 0.8, 1: 0.2},
        model_used="edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
        success=True,
        status=ExecutionStatus.SUCCESS,
        edge_latency_s=0.001,
    )
    monitor.observe_step(observation_index=0, execution_result=res)

    snap = monitor.get_snapshot()
    assert snap.execution_stats["total"] == 1
    assert snap.execution_stats["successful"] == 1
    assert snap.execution_stats["failed"] == 0
    assert snap.execution_stats["success_rate"] == pytest.approx(1.0)


def test_15_execution_failure_tracking():
    """Verify execution failures are tracked and provenance preserved."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.CLOUD,
        reliability=0.2,
        previous_action=None,
        decision_reason="Drift",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    res = ExecutionResult(
        decision=dec,
        action=DecisionAction.CLOUD,
        prediction=None,
        probabilities=None,
        model_used="cloud",
        inference_latency_s=0.0,
        cloud_fallback=False,
        success=False,
        status=ExecutionStatus.FAILED,
        cloud_latency_s=None,
        error="Cloud model connection refused",
    )
    rec = monitor.observe_step(observation_index=0, execution_result=res)

    assert rec.execution_status == "FAILED"
    assert "execution_failure_detected" in rec.alerts
    snap = monitor.get_snapshot()
    assert snap.execution_stats["failed"] == 1
    assert snap.execution_stats["cloud_failures"] == 1
    assert snap.execution_stats["success_rate"] == pytest.approx(0.0)


def test_16_latency_statistics():
    """Verify fine-grained streaming latency statistics without fabricating missing latencies."""
    monitor = DRAECMonitor()
    dec_edge = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.8,
        previous_action=None,
        decision_reason="Normal",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    res_edge = ExecutionResult(
        decision=dec_edge,
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 0.8, 1: 0.2},
        model_used="edge",
        inference_latency_s=0.001,
        cloud_fallback=False,
        success=True,
        status=ExecutionStatus.SUCCESS,
        edge_latency_s=0.002,
        cloud_latency_s=None,
        hybrid_latency_s=None,
    )
    rec = monitor.observe_step(observation_index=0, execution_result=res_edge)
    assert rec.edge_latency_s == 0.002
    assert rec.cloud_latency_s is None
    assert rec.hybrid_latency_s is None

    snap = monitor.get_snapshot()
    assert snap.latency_stats["edge"]["count"] == 1
    assert snap.latency_stats["edge"]["mean"] == pytest.approx(0.002)
    assert snap.latency_stats["cloud"]["count"] == 0
    assert snap.latency_stats["cloud"]["mean"] is None


# -----------------------------------------------------------------------------
# Test 17-20: Bounded History & Snapshot Integrity
# -----------------------------------------------------------------------------


def test_17_bounded_history():
    """Verify buffer stays strictly bounded by max_records while global statistics persist."""
    max_recs = 5
    monitor = DRAECMonitor(max_records=max_recs)

    # Ingest 15 observations
    for i in range(15):
        score = make_dummy_reliability_score(r=0.5 + (i * 0.01))
        dec = DecisionResult(
            selected_action=DecisionAction.EDGE,
            reliability=0.5 + (i * 0.01),
            previous_action=None,
            decision_reason="Test",
            switch_count=0,
            observation_index=i,
            timestamp=float(i),
        )
        monitor.observe_step(observation_index=i, reliability_score=score, decision_result=dec)

    # History buffer must contain exactly max_recs
    records = monitor.get_records()
    assert len(records) == max_recs
    # History contains only the most recent records: indices 10 to 14
    assert [r.observation_index for r in records] == [10, 11, 12, 13, 14]

    # Global cumulative stats must reflect all 15 observations
    snap = monitor.get_snapshot()
    assert snap.total_observations == 15
    assert snap.routing_counts["EDGE"] == 15
    assert snap.routing_counts["total"] == 15
    assert snap.reliability_stats["count"] == 15


def test_18_snapshot_generation():
    """Verify MonitoringSnapshot contains all required observability keys."""
    monitor = DRAECMonitor()
    score = make_dummy_reliability_score(r=0.75)
    monitor.observe_step(observation_index=0, reliability_score=score)

    snap = monitor.get_snapshot()
    snap_dict = snap.to_dict()

    required_keys = [
        "timestamp",
        "total_observations",
        "current_reliability",
        "current_drift_severity",
        "current_action",
        "current_policy",
        "routing_counts",
        "routing_distribution",
        "hybrid_stats",
        "execution_stats",
        "latency_stats",
        "drift_stats",
        "reliability_stats",
        "model_health",
        "active_alerts",
    ]
    for key in required_keys:
        assert key in snap_dict, f"Missing snapshot key: {key}"


def test_19_deterministic_monitoring():
    """Verify identical input observations produce bit-identical monitoring outputs."""
    mon1 = DRAECMonitor(max_records=10)
    mon2 = DRAECMonitor(max_records=10)

    for i in range(5):
        score = make_dummy_reliability_score(r=0.7 + i * 0.02)
        dec = DecisionResult(
            selected_action=DecisionAction.EDGE if i % 2 == 0 else DecisionAction.CLOUD,
            reliability=0.7 + i * 0.02,
            previous_action=None,
            decision_reason="Deterministic test",
            switch_count=i,
            observation_index=i,
            timestamp=1000.0 + i,
        )
        rec1 = mon1.observe_step(observation_index=i, reliability_score=score, decision_result=dec, timestamp=1000.0 + i)
        rec2 = mon2.observe_step(observation_index=i, reliability_score=score, decision_result=dec, timestamp=1000.0 + i)
        assert rec1 == rec2

    df1 = mon1.get_records_dataframe()
    df2 = mon2.get_records_dataframe()
    pd.testing.assert_frame_equal(df1, df2)


def test_20_causality():
    """Verify that observation index t depends strictly on data at step t."""
    monitor = DRAECMonitor()
    score_t = make_dummy_reliability_score(r=0.65)
    rec = monitor.observe_step(observation_index=42, reliability_score=score_t, timestamp=1042.0)

    assert rec.observation_index == 42
    assert rec.timestamp == 1042.0
    assert rec.reliability == pytest.approx(0.65)


# -----------------------------------------------------------------------------
# Test 21-23: Leakage & Isolation Protection
# -----------------------------------------------------------------------------


def test_21_target_leakage_protection():
    """Verify monitor does not accept or process ground truth 'Target' column in observations."""
    monitor = DRAECMonitor()
    # Ensure observe_step has no target parameter and does not leak it
    rec = monitor.observe_step(observation_index=0)
    assert not hasattr(rec, "Target")
    assert not hasattr(rec, "target")


def test_22_ground_truth_isolation():
    """Verify monitor does not access or read synthetic ground_truth.json."""
    monitor = DRAECMonitor()
    # Check that monitor configuration has no reference to ground truth sidecars
    assert "ground_truth" not in monitor.config.get("monitoring", {})
    snap = monitor.get_snapshot()
    assert "ground_truth" not in snap.to_dict()


def test_23_future_observation_isolation():
    """Verify streaming stats at step t do not include future steps."""
    monitor = DRAECMonitor()
    monitor.observe_step(observation_index=0, reliability_score=make_dummy_reliability_score(r=0.8))
    snap_t0 = monitor.get_snapshot()
    assert snap_t0.reliability_stats["count"] == 1
    assert snap_t0.reliability_stats["mean"] == pytest.approx(0.8)

    monitor.observe_step(observation_index=1, reliability_score=make_dummy_reliability_score(r=0.4))
    snap_t1 = monitor.get_snapshot()
    assert snap_t1.reliability_stats["count"] == 2
    assert snap_t1.reliability_stats["mean"] == pytest.approx(0.6)


# -----------------------------------------------------------------------------
# Test 24-27: Integration & Phase Compatibility
# -----------------------------------------------------------------------------


def test_24_phase_6_compatibility():
    """Verify seamless ingestion of Phase 6 ExecutionResult."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.75,
        previous_action=None,
        decision_reason="Normal",
        switch_count=0,
        observation_index=0,
        timestamp=time.time(),
    )
    res = ExecutionResult(
        decision=dec,
        action=DecisionAction.EDGE,
        prediction=0,
        probabilities={0: 0.9, 1: 0.1},
        model_used="edge",
        inference_latency_s=0.0015,
        cloud_fallback=False,
        success=True,
        status=ExecutionStatus.SUCCESS,
        edge_latency_s=0.0015,
    )
    rec = monitor.observe_step(observation_index=0, execution_result=res)
    assert rec.execution_status == "SUCCESS"
    assert rec.prediction == 0
    assert rec.edge_latency_s == 0.0015


def test_25_phase_5_compatibility():
    """Verify seamless ingestion of Phase 5 DecisionResult."""
    monitor = DRAECMonitor()
    dec = DecisionResult(
        selected_action=DecisionAction.CLOUD,
        reliability=0.25,
        previous_action=DecisionAction.HYBRID,
        decision_reason="Low reliability",
        switch_count=3,
        observation_index=5,
        timestamp=time.time(),
    )
    rec = monitor.observe_step(observation_index=5, decision_result=dec)
    assert rec.selected_action == "CLOUD"
    assert rec.previous_action == "HYBRID"
    assert rec.decision_reason == "Low reliability"


def test_26_phase_4_compatibility():
    """Verify seamless ingestion of Phase 4 ReliabilityScore."""
    monitor = DRAECMonitor()
    score = make_dummy_reliability_score(r=0.62, c=0.75, e=0.1, d=0.2, q=0.95)
    rec = monitor.observe_step(observation_index=1, reliability_score=score)
    assert rec.reliability == pytest.approx(0.62)
    assert rec.confidence == pytest.approx(0.75)


def test_27_streaming_smoke_test():
    """Multi-step causal streaming smoke test exporting stable DataFrame for Phase 10."""
    reg = ModelRegistry()
    edge_m = MockEdgeModel()
    cloud_m = MockCloudModel()
    reg.register_model(edge_m, "edge", "edge", version="1.0.0")
    reg.register_model(cloud_m, "cloud", "cloud", version="1.0.0")

    monitor = DRAECMonitor(registry=reg, max_records=20)

    for i in range(10):
        action = DecisionAction.EDGE if i < 5 else DecisionAction.HYBRID
        dec = DecisionResult(
            selected_action=action,
            reliability=0.8 - i * 0.05,
            previous_action=None if i == 0 else DecisionAction.EDGE,
            decision_reason=f"Step {i}",
            switch_count=1 if i == 5 else 0,
            observation_index=i,
            timestamp=float(i),
        )
        res = ExecutionResult(
            decision=dec,
            action=action,
            prediction=0,
            probabilities={0: 0.85, 1: 0.15},
            model_used="edge" if action == DecisionAction.EDGE else "hybrid_edge",
            inference_latency_s=0.001,
            cloud_fallback=False,
            success=True,
            status=ExecutionStatus.SUCCESS,
            edge_latency_s=0.001,
            hybrid_latency_s=0.0012 if action == DecisionAction.HYBRID else None,
        )
        score = make_dummy_reliability_score(r=0.8 - i * 0.05)
        drift = {"drift_detected": i >= 5, "is_persistent": i >= 7, "raw_severity": 0.1 * i, "smoothed_severity": 0.08 * i}

        monitor.observe_step(
            observation_index=i,
            execution_result=res,
            decision_result=dec,
            reliability_score=score,
            drift_status=drift,
            controller_policy="draec_adaptive",
            timestamp=float(i),
        )

    # Export to DataFrame
    df = monitor.get_records_dataframe()
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 10

    # Verify fixed, stable schema
    expected_cols = [
        "observation_index",
        "timestamp",
        "reliability",
        "confidence",
        "error_ema",
        "drift_severity",
        "quality",
        "selected_action",
        "previous_action",
        "decision_reason",
        "prediction",
        "model_used",
        "execution_status",
        "cloud_fallback",
        "edge_latency_s",
        "cloud_latency_s",
        "hybrid_latency_s",
        "model_version",
        "drift_detected",
        "is_persistent",
        "raw_severity",
        "smoothed_severity",
        "controller_policy",
    ]
    for col in expected_cols:
        assert col in df.columns, f"Missing schema column: {col}"

    assert (df["controller_policy"] == "draec_adaptive").all()
    assert df["selected_action"].iloc[0] == "EDGE"
    assert df["selected_action"].iloc[9] == "HYBRID"
