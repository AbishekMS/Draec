"""Unit and integration tests for Phase 5 DRAEC Decision Engine and Minimal Execution."""

from __future__ import annotations

import numpy as np
import pytest

from src.decision import (
    AdaptiveController,
    BaseController,
    BaseDecisionEngine,
    DecisionAction,
    DecisionEngine,
    DecisionInputs,
    DecisionInstrumentation,
    DecisionResult,
    ExecutionResult,
    StaticBaselineController,
)
from src.models.base import BaseModel
from src.models.cloud_model import CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree
from src.reliability import ReliabilityEstimator


class DummyModel(BaseModel):
    """Deterministic dummy model for unit testing execution routing."""

    def __init__(self, name: str, fixed_pred: int = 0, fixed_probs: dict[int, float] | None = None) -> None:
        super().__init__(name)
        self._is_trained = True
        self._n_features = 2
        self._fixed_pred = fixed_pred
        self._fixed_probs = fixed_probs or {0: 0.9, 1: 0.1}
        self.call_count = 0

    def fit(self, X, y):
        return self

    def predict(self, X):
        self.call_count += len(X)
        return np.full(len(X), self._fixed_pred, dtype=int)

    def predict_proba(self, X):
        self.call_count += len(X)
        return np.tile([self._fixed_probs[0], self._fixed_probs[1]], (len(X), 1))

    def predict_one(self, x):
        self.call_count += 1
        return self._fixed_pred

    def predict_proba_one(self, x):
        self.call_count += 1
        return dict(self._fixed_probs)

    def get_info(self):
        return {"name": self._model_name, "call_count": self.call_count}


@pytest.fixture
def dummy_edge():
    return DummyModel("dummy_edge", fixed_pred=0, fixed_probs={0: 0.90, 1: 0.10})


@pytest.fixture
def dummy_cloud():
    return DummyModel("dummy_cloud", fixed_pred=1, fixed_probs={0: 0.05, 1: 0.95})


@pytest.fixture
def trained_models():
    X = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [0.5, 0.2]])
    y = np.array([0, 1, 0, 1])
    edge = EdgeHoeffdingTree(grace_period=10).fit(X, y)
    cloud = CloudXGBoost(n_estimators=5, max_depth=2, random_state=42).fit(X, y)
    return edge, cloud


def test_01_decision_engine_creation(dummy_edge, dummy_cloud):
    controller = AdaptiveController()
    engine = DecisionEngine(controller, dummy_edge, dummy_cloud)
    assert isinstance(engine, BaseDecisionEngine)
    assert engine.controller is controller
    assert engine.edge_model is dummy_edge
    assert engine.cloud_model is dummy_cloud
    assert engine.fallback_confidence_threshold == 0.60


def test_02_adaptive_controller_creation():
    ctrl = AdaptiveController()
    assert isinstance(ctrl, BaseController)
    assert ctrl.current_action == DecisionAction.EDGE
    assert ctrl.cloud_threshold == 0.50
    assert ctrl.edge_return_threshold == 0.70
    assert ctrl.critical_cloud_threshold == 0.30
    assert ctrl.switch_count == 0


def test_03_static_baseline_creation():
    base = StaticBaselineController(policy="edge_only")
    assert isinstance(base, BaseController)
    assert base.policy == "edge_only"
    res = base.decide(0.1)  # Low reliability should not change static decision
    assert res.selected_action == DecisionAction.EDGE


def test_04_action_space():
    actions = {a for a in DecisionAction}
    assert actions == {DecisionAction.EDGE, DecisionAction.CLOUD, DecisionAction.HYBRID}
    assert DecisionAction.from_str("edge") == DecisionAction.EDGE
    assert DecisionAction.from_str("CLOUD") == DecisionAction.CLOUD
    assert DecisionAction.from_str("Hybrid") == DecisionAction.HYBRID


def test_05_configuration_loading(cfg):
    ctrl = AdaptiveController(config=cfg)
    assert ctrl.cloud_threshold == 0.50
    assert ctrl.edge_return_threshold == 0.70
    assert ctrl.critical_cloud_threshold == 0.30


def test_06_high_reliability_edge_routing():
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    res = ctrl.decide(0.85)
    assert res.selected_action == DecisionAction.EDGE
    assert res.previous_action == DecisionAction.EDGE
    assert ctrl.switch_count == 0


def test_07_low_reliability_cloud_routing():
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    res = ctrl.decide(0.20)
    assert res.selected_action == DecisionAction.CLOUD
    assert ctrl.switch_count == 1


def test_08_hysteresis_deadband():
    # Between 0.50 and 0.70, previous action is maintained
    # Case A: Coming from EDGE
    ctrl_edge = AdaptiveController(initial_action=DecisionAction.EDGE)
    res_edge = ctrl_edge.decide(0.60)
    assert res_edge.selected_action == DecisionAction.EDGE

    # Case B: Coming from CLOUD
    ctrl_cloud = AdaptiveController(initial_action=DecisionAction.CLOUD)
    res_cloud = ctrl_cloud.decide(0.60)
    assert res_cloud.selected_action == DecisionAction.CLOUD


def test_09_recovery_to_edge():
    ctrl = AdaptiveController(initial_action=DecisionAction.CLOUD)
    # Below return threshold: stay CLOUD
    res1 = ctrl.decide(0.65)
    assert res1.selected_action == DecisionAction.CLOUD
    # At or above return threshold (0.70): recover to EDGE
    res2 = ctrl.decide(0.72)
    assert res2.selected_action == DecisionAction.EDGE
    assert ctrl.switch_count == 1


def test_10_no_rapid_oscillation():
    # Fluctuation around 0.50 after being in EDGE
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    # Start at 0.8 -> EDGE
    ctrl.decide(0.80)
    assert ctrl.current_action == DecisionAction.EDGE

    # Small fluctuation: 0.52 -> EDGE (no switch)
    ctrl.decide(0.52)
    assert ctrl.current_action == DecisionAction.EDGE
    assert ctrl.switch_count == 0

    # Drop into hybrid zone: 0.45 -> HYBRID (1 switch)
    ctrl.decide(0.45)
    assert ctrl.current_action == DecisionAction.HYBRID
    assert ctrl.switch_count == 1

    # Bounce back to 0.55: within hybrid deadband [0.30, 0.70), remains HYBRID (no chatter back to EDGE)
    ctrl.decide(0.55)
    assert ctrl.current_action == DecisionAction.HYBRID
    assert ctrl.switch_count == 1

    # Fluctuate back to 0.45: remains HYBRID
    ctrl.decide(0.45)
    assert ctrl.current_action == DecisionAction.HYBRID
    assert ctrl.switch_count == 1


def test_11_hybrid_selection():
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    # Reliability drops into [0.30, 0.50) -> selects HYBRID
    res = ctrl.decide(0.40)
    assert res.selected_action == DecisionAction.HYBRID


def test_12_hybrid_edge_first_execution(dummy_cloud):
    # High confidence edge model (P = {0: 0.95, 1: 0.05} -> C = 0.90 >= 0.60)
    edge_high = DummyModel("edge_high", fixed_pred=0, fixed_probs={0: 0.95, 1: 0.05})
    ctrl = AdaptiveController(initial_action=DecisionAction.HYBRID)
    engine = DecisionEngine(ctrl, edge_high, dummy_cloud, fallback_confidence_threshold=0.60)

    res = engine.execute([1.0, 2.0], 0.40)
    assert res.action == DecisionAction.HYBRID
    assert res.model_used == "hybrid_edge"
    assert not res.cloud_fallback
    assert res.prediction == 0
    # Cloud was not called
    assert dummy_cloud.call_count == 0
    assert edge_high.call_count > 0


def test_13_hybrid_cloud_fallback(dummy_cloud):
    # Uncertain edge model (P = {0: 0.55, 1: 0.45} -> C = 0.10 < 0.60)
    edge_low = DummyModel("edge_low", fixed_pred=0, fixed_probs={0: 0.55, 1: 0.45})
    ctrl = AdaptiveController(initial_action=DecisionAction.HYBRID)
    engine = DecisionEngine(ctrl, edge_low, dummy_cloud, fallback_confidence_threshold=0.60)

    res = engine.execute([1.0, 2.0], 0.40)
    assert res.action == DecisionAction.HYBRID
    assert res.model_used == "hybrid_cloud"
    assert res.cloud_fallback
    # Cloud prediction provides the final result
    assert res.prediction == dummy_cloud._fixed_pred
    assert dummy_cloud.call_count > 0


def test_14_deterministic_decisions():
    seq = [0.8, 0.6, 0.4, 0.2, 0.35, 0.65, 0.75]
    ctrl1 = AdaptiveController()
    ctrl2 = AdaptiveController()

    out1 = [ctrl1.decide(r).selected_action for r in seq]
    out2 = [ctrl2.decide(r).selected_action for r in seq]
    assert out1 == out2


def test_15_reset_behavior():
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    ctrl.decide(0.20)
    assert ctrl.current_action == DecisionAction.CLOUD
    assert ctrl.switch_count == 1

    ctrl.reset()
    assert ctrl.current_action == DecisionAction.EDGE
    assert ctrl.switch_count == 0
    assert ctrl.decision_count == 0


def test_16_switch_counting():
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    ctrl.decide(0.9)  # EDGE -> EDGE (0)
    ctrl.decide(0.4)  # EDGE -> HYBRID (1)
    ctrl.decide(0.4)  # HYBRID -> HYBRID (1)
    ctrl.decide(0.1)  # HYBRID -> CLOUD (2)
    ctrl.decide(0.1)  # CLOUD -> CLOUD (2)
    ctrl.decide(0.8)  # CLOUD -> EDGE (3)
    assert ctrl.switch_count == 3


def test_17_edge_execution(dummy_edge, dummy_cloud):
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    engine = DecisionEngine(ctrl, dummy_edge, dummy_cloud)
    res = engine.execute([1.0, 2.0], 0.9)
    assert res.action == DecisionAction.EDGE
    assert res.model_used == "edge"
    assert not res.cloud_fallback
    assert dummy_edge.call_count > 0
    assert dummy_cloud.call_count == 0


def test_18_cloud_execution(dummy_edge, dummy_cloud):
    ctrl = AdaptiveController(initial_action=DecisionAction.CLOUD)
    engine = DecisionEngine(ctrl, dummy_edge, dummy_cloud)
    res = engine.execute([1.0, 2.0], 0.2)
    assert res.action == DecisionAction.CLOUD
    assert res.model_used == "cloud"
    assert not res.cloud_fallback
    assert dummy_cloud.call_count > 0
    assert dummy_edge.call_count == 0


def test_19_decision_result_structure():
    inp = DecisionInputs(reliability=0.75, confidence=0.8, drift_severity=0.1, quality=0.9, observation_index=42)
    ctrl = AdaptiveController()
    res = ctrl.decide(inp)
    assert isinstance(res, DecisionResult)
    assert res.selected_action == DecisionAction.EDGE
    assert res.reliability == 0.75
    assert res.observation_index == 42
    assert res.decision_inputs is inp
    assert "maintain EDGE" in res.decision_reason


def test_20_instrumentation(dummy_edge, dummy_cloud):
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    engine = DecisionEngine(ctrl, dummy_edge, dummy_cloud)

    engine.execute([1.0, 2.0], 0.9)  # EDGE
    engine.execute([1.0, 2.0], 0.2)  # CLOUD
    summary = engine.instrumentation.get_summary()

    assert summary["total_decisions"] == 2
    assert summary["edge_count"] == 1
    assert summary["cloud_count"] == 1
    assert summary["switch_count"] == 1
    assert summary["total_latency_s"] > 0.0


def test_21_baseline_independence_from_rt():
    base = StaticBaselineController(policy="edge_only")
    for r in [0.0, 0.2, 0.5, 0.8, 1.0]:
        res = base.decide(r)
        assert res.selected_action == DecisionAction.EDGE

    base_cloud = StaticBaselineController(policy="cloud_only")
    for r in [0.0, 0.2, 0.5, 0.8, 1.0]:
        res = base_cloud.decide(r)
        assert res.selected_action == DecisionAction.CLOUD


def test_22_adaptive_dependence_on_rt():
    ctrl = AdaptiveController()
    res_high = ctrl.decide(0.9)
    ctrl.reset()
    res_low = ctrl.decide(0.1)
    assert res_high.selected_action != res_low.selected_action


def test_23_target_leakage_prevention(dummy_edge, dummy_cloud):
    # Verify execution does not require or accept a true target label
    ctrl = AdaptiveController()
    engine = DecisionEngine(ctrl, dummy_edge, dummy_cloud)
    inp = DecisionInputs(reliability=0.85)
    # Valid execution without any y or target parameter
    res = engine.execute([1.0, 2.0], inp)
    assert res.prediction in (0, 1)


def test_24_ground_truth_isolation():
    ctrl = AdaptiveController()
    info = ctrl.get_info()
    assert "scenario" not in info
    assert "drift_start_index" not in info
    assert "ground_truth" not in info


def test_25_future_data_prevention():
    # A single causal step only consumes index t
    ctrl = AdaptiveController()
    inp_t = DecisionInputs(reliability=0.65, observation_index=10)
    res = ctrl.decide(inp_t)
    assert res.observation_index == 10


def test_26_phase4_reliability_compatibility(dummy_edge, dummy_cloud):
    # Feed output of Phase 4 ReliabilityEstimator directly into Phase 5 DecisionEngine
    rel_est = ReliabilityEstimator()
    rel_score = rel_est.update(probs={0: 0.8, 1: 0.2}, drift_severity=0.05, quality=1.0)
    assert 0.0 <= rel_score.reliability <= 1.0

    ctrl = AdaptiveController()
    engine = DecisionEngine(ctrl, dummy_edge, dummy_cloud)
    inp = DecisionInputs(
        reliability=rel_score.reliability,
        confidence=rel_score.inputs.confidence,
        drift_severity=rel_score.inputs.drift,
        quality=rel_score.inputs.quality,
    )
    exec_res = engine.execute([1.0, 2.0], inp)
    assert exec_res.action == DecisionAction.EDGE
    assert exec_res.prediction in (0, 1)


def test_27_phase2_model_compatibility(trained_models):
    edge_model, cloud_model = trained_models
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    engine = DecisionEngine(ctrl, edge_model, cloud_model)

    x_obs = [1.2, 1.9]
    res_edge = engine.execute(x_obs, 0.85)
    assert res_edge.action == DecisionAction.EDGE
    assert res_edge.model_used == "edge"

    res_cloud = engine.execute(x_obs, 0.15)
    assert res_cloud.action == DecisionAction.CLOUD
    assert res_cloud.model_used == "cloud"


def test_28_causal_streaming_smoke_test(trained_models):
    edge_model, cloud_model = trained_models
    ctrl = AdaptiveController(initial_action=DecisionAction.EDGE)
    engine = DecisionEngine(ctrl, edge_model, cloud_model)
    rel_est = ReliabilityEstimator()

    # Stream 10 observations with evolving reliability
    reliabilities = [0.9, 0.85, 0.8, 0.65, 0.45, 0.40, 0.20, 0.25, 0.75, 0.85]
    for i, r in enumerate(reliabilities):
        x = [1.0 + 0.1 * i, 2.0 - 0.1 * i]
        inp = DecisionInputs(reliability=r, observation_index=i)
        res = engine.execute(x, inp)
        assert res.prediction in (0, 1)

    summary = engine.instrumentation.get_summary()
    assert summary["total_decisions"] == 10
    assert summary["edge_count"] > 0
    assert summary["cloud_count"] > 0
    assert summary["hybrid_count"] > 0
    assert summary["switch_count"] >= 3
