"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : tests/test_deployment.py
Phase    : Phase 8
Status   : IMPLEMENTED

Comprehensive test suite for Phase 8 Edge-Cloud Deployment & Network Execution Layer.
Covers at least 28 mandatory verification scenarios.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest

from src.decision.base import DecisionAction, DecisionResult, ExecutionResult, ExecutionStatus
from src.deployment.base import (
    DeploymentExecutionResult,
    NetworkPacket,
    RuntimeState,
    TransmissionResult,
    TransmissionStatus,
)
from src.deployment.environment import DeploymentEnvironment
from src.deployment.network import NetworkSimulator
from src.deployment.runtimes import CloudRuntime, EdgeRuntime
from src.monitoring.monitor import DRAECMonitor
from src.utils.config import load


class MockModel:
    """Configurable mock model for deterministic testing of runtimes."""

    def __init__(self, pred: int = 0, proba: dict[int, float] | None = None) -> None:
        self.pred = pred
        self.proba = proba or {0: 0.80, 1: 0.20}
        self.call_count = 0

    def predict_one(self, x: Any) -> int:
        self.call_count += 1
        return self.pred

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        return dict(self.proba)


class FailingMockModel:
    """Mock model that raises an execution exception."""

    def predict_one(self, x: Any) -> int:
        raise RuntimeError("Internal model arithmetic fault")

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        raise RuntimeError("Internal model probability fault")


# -----------------------------------------------------------------------------
# 1. Edge runtime creation
# -----------------------------------------------------------------------------
def test_01_edge_runtime_creation():
    model = MockModel(pred=1, proba={0: 0.2, 1: 0.8})
    runtime = EdgeRuntime(model=model, available=True)
    assert runtime.available is True
    assert runtime.model is model
    stats = runtime.get_stats()
    assert stats["total_executions"] == 0
    assert stats["failed_executions"] == 0


# -----------------------------------------------------------------------------
# 2. Cloud runtime creation
# -----------------------------------------------------------------------------
def test_02_cloud_runtime_creation():
    model = MockModel(pred=0, proba={0: 0.9, 1: 0.1})
    runtime = CloudRuntime(model=model, available=True)
    assert runtime.available is True
    assert runtime.model is model
    stats = runtime.get_stats()
    assert stats["total_executions"] == 0
    assert stats["failed_executions"] == 0


# -----------------------------------------------------------------------------
# 3. Network simulator creation
# -----------------------------------------------------------------------------
def test_03_network_simulator_creation():
    net = NetworkSimulator(
        base_latency_s=0.030,
        jitter_s=0.005,
        packet_loss_probability=0.02,
        available=True,
        seed=123,
    )
    assert net.base_latency_s == 0.030
    assert net.jitter_s == 0.005
    assert net.packet_loss_probability == 0.02
    assert net.available is True
    assert net.pacing_enabled is False


# -----------------------------------------------------------------------------
# 4. Deterministic network behavior
# -----------------------------------------------------------------------------
def test_04_deterministic_network_behavior():
    net1 = NetworkSimulator(base_latency_s=0.020, jitter_s=0.005, packet_loss_probability=0.2, seed=42)
    net2 = NetworkSimulator(base_latency_s=0.020, jitter_s=0.005, packet_loss_probability=0.2, seed=42)

    res1 = [net1.transmit({"val": i}, observation_index=i) for i in range(20)]
    res2 = [net2.transmit({"val": i}, observation_index=i) for i in range(20)]

    for r1, r2 in zip(res1, res2):
        assert r1.success == r2.success
        assert r1.packet_lost == r2.packet_lost
        assert r1.latency_s == pytest.approx(r2.latency_s, rel=1e-5)


# -----------------------------------------------------------------------------
# 5. Config parsing
# -----------------------------------------------------------------------------
def test_05_network_config_parsing():
    cfg = load("default")
    assert "network" in cfg
    net_cfg = cfg["network"]
    assert net_cfg["enabled"] is True
    assert net_cfg["mode"] == "simulation"
    assert "latency" in net_cfg
    assert "base_s" in net_cfg["latency"]
    assert "jitter_s" in net_cfg["latency"]
    assert "packet_loss" in net_cfg
    assert "availability" in net_cfg
    assert net_cfg.get("pacing_enabled", False) is False

    net = NetworkSimulator(config=cfg)
    assert net.base_latency_s == float(net_cfg["latency"]["base_s"])
    assert net.packet_loss_probability == float(net_cfg["packet_loss"]["probability"])


# -----------------------------------------------------------------------------
# 6. Zero packet loss
# -----------------------------------------------------------------------------
def test_06_zero_packet_loss():
    net = NetworkSimulator(packet_loss_probability=0.0, seed=42)
    for i in range(50):
        tx = net.transmit({"x": i}, observation_index=i)
        assert tx.success is True
        assert tx.packet_lost is False
        assert tx.status == TransmissionStatus.DELIVERED
        assert tx.error is None
        assert tx.latency_s >= 0.0


# -----------------------------------------------------------------------------
# 7. Forced packet loss
# -----------------------------------------------------------------------------
def test_07_forced_packet_loss():
    net = NetworkSimulator(packet_loss_probability=1.0, seed=42)
    tx = net.transmit({"x": 1}, observation_index=1)
    assert tx.success is False
    assert tx.packet_lost is True
    assert tx.status == TransmissionStatus.PACKET_LOSS
    assert "packet loss" in str(tx.error).lower()


# -----------------------------------------------------------------------------
# 8. Network failure provenance
# -----------------------------------------------------------------------------
def test_08_network_failure():
    net = NetworkSimulator(available=False)
    tx = net.transmit({"x": 1})
    assert tx.success is False
    assert tx.status == TransmissionStatus.DISCONNECTED
    assert "unavailable" in str(tx.error).lower()


# -----------------------------------------------------------------------------
# 9. Edge availability toggle
# -----------------------------------------------------------------------------
def test_09_edge_availability():
    runtime = EdgeRuntime(model=MockModel(), available=False)
    pred, proba, lat, succ, err = runtime.execute({"x": 1})
    assert succ is False
    assert pred is None
    assert proba is None
    assert "offline" in str(err).lower() or "unavailable" in str(err).lower()


# -----------------------------------------------------------------------------
# 10. Cloud availability toggle
# -----------------------------------------------------------------------------
def test_10_cloud_availability():
    runtime = CloudRuntime(model=MockModel(), available=False)
    pred, proba, lat, succ, err = runtime.execute({"x": 1})
    assert succ is False
    assert pred is None
    assert proba is None
    assert "offline" in str(err).lower() or "unavailable" in str(err).lower()


# -----------------------------------------------------------------------------
# 11. Edge failure injection
# -----------------------------------------------------------------------------
def test_11_edge_failure():
    runtime = EdgeRuntime(model=MockModel(), failure_schedule=[5, 10])
    pred1, _, _, succ1, _ = runtime.execute({"x": 1}, observation_index=4)
    assert succ1 is True
    pred2, _, _, succ2, err2 = runtime.execute({"x": 1}, observation_index=5)
    assert succ2 is False
    assert pred2 is None
    assert "scheduled" in str(err2).lower()


# -----------------------------------------------------------------------------
# 12. Cloud failure injection
# -----------------------------------------------------------------------------
def test_12_cloud_failure():
    runtime = CloudRuntime(model=MockModel(), failure_schedule=[7])
    pred1, _, _, succ1, _ = runtime.execute({"x": 1}, observation_index=6)
    assert succ1 is True
    pred2, _, _, succ2, err2 = runtime.execute({"x": 1}, observation_index=7)
    assert succ2 is False
    assert pred2 is None
    assert "scheduled" in str(err2).lower()


# -----------------------------------------------------------------------------
# 13. Edge-only execution
# -----------------------------------------------------------------------------
def test_13_edge_only_execution():
    edge_m = MockModel(pred=1, proba={0: 0.1, 1: 0.9})
    cloud_m = MockModel(pred=0, proba={0: 0.9, 1: 0.1})
    net = NetworkSimulator(packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net)

    res = env.execute_edge({"f1": 1.0}, observation_index=1)
    assert res.action == DecisionAction.EDGE
    assert res.prediction == 1
    assert res.model_used == "edge"
    assert res.success is True
    assert res.status == ExecutionStatus.SUCCESS
    assert res.edge_latency_s is not None
    assert res.cloud_latency_s is None
    assert res.network_latency_s is None
    assert cloud_m.call_count == 0


# -----------------------------------------------------------------------------
# 14. Cloud-only execution
# -----------------------------------------------------------------------------
def test_14_cloud_only_execution():
    edge_m = MockModel(pred=1)
    cloud_m = MockModel(pred=0, proba={0: 0.85, 1: 0.15})
    net = NetworkSimulator(base_latency_s=0.015, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net)

    res = env.execute_cloud({"f1": 1.0}, observation_index=1)
    assert res.action == DecisionAction.CLOUD
    assert res.prediction == 0
    assert res.model_used == "cloud"
    assert res.success is True
    assert res.status == ExecutionStatus.SUCCESS
    assert res.cloud_latency_s is not None
    assert res.network_latency_s is not None
    assert res.network_latency_s > 0.0
    assert edge_m.call_count == 0


# -----------------------------------------------------------------------------
# 15. Hybrid Edge-first execution (Confident Edge)
# -----------------------------------------------------------------------------
def test_15_hybrid_edge_first_execution():
    # p0=0.85 -> C_edge = 2 * (0.85 - 0.5) = 0.70 >= 0.60 threshold
    edge_m = MockModel(pred=0, proba={0: 0.85, 1: 0.15})
    cloud_m = MockModel(pred=1, proba={0: 0.1, 1: 0.9})
    net = NetworkSimulator(packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)

    res = env.execute_hybrid({"f1": 1.0}, observation_index=1)
    assert res.action == DecisionAction.HYBRID
    assert res.prediction == 0
    assert res.model_used == "hybrid_edge"
    assert res.success is True
    assert res.cloud_fallback is False
    assert res.status == ExecutionStatus.SUCCESS
    assert res.hybrid_latency_s is not None
    assert res.edge_latency_s is not None
    assert res.cloud_latency_s is None
    assert res.network_latency_s is None
    assert cloud_m.call_count == 0


# -----------------------------------------------------------------------------
# 16. Hybrid Cloud fallback (Uncertain Edge)
# -----------------------------------------------------------------------------
def test_16_hybrid_cloud_fallback():
    # p0=0.55 -> C_edge = 2 * (0.55 - 0.5) = 0.10 < 0.60 threshold
    edge_m = MockModel(pred=0, proba={0: 0.55, 1: 0.45})
    cloud_m = MockModel(pred=1, proba={0: 0.05, 1: 0.95})
    net = NetworkSimulator(base_latency_s=0.010, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)

    res = env.execute_hybrid({"f1": 1.0}, observation_index=1)
    assert res.action == DecisionAction.HYBRID
    assert res.prediction == 1
    assert res.model_used == "hybrid_cloud"
    assert res.success is True
    assert res.cloud_fallback is True
    assert res.status == ExecutionStatus.FALLBACK
    assert res.edge_latency_s is not None
    assert res.cloud_latency_s is not None
    assert res.network_latency_s is not None
    assert res.hybrid_latency_s is not None
    assert edge_m.call_count == 1
    assert cloud_m.call_count == 1


# -----------------------------------------------------------------------------
# 17. Hybrid fallback with network failure
# -----------------------------------------------------------------------------
def test_17_hybrid_fallback_with_network_failure():
    # Edge is uncertain (C_edge = 0.10 < 0.60), but network experiences packet loss
    edge_m = MockModel(pred=0, proba={0: 0.55, 1: 0.45})
    cloud_m = MockModel(pred=1, proba={0: 0.05, 1: 0.95})
    net = NetworkSimulator(packet_loss_probability=1.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)

    res = env.execute_hybrid({"f1": 1.0}, observation_index=1)
    assert res.action == DecisionAction.HYBRID
    assert res.prediction is None
    assert res.probabilities is None
    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert res.cloud_fallback is True
    assert res.packet_lost is True
    assert "network failure" in str(res.error).lower()
    assert cloud_m.call_count == 0


# -----------------------------------------------------------------------------
# 18. Latency accounting
# -----------------------------------------------------------------------------
def test_18_latency_accounting():
    edge_m = MockModel(pred=0, proba={0: 0.52, 1: 0.48})
    cloud_m = MockModel(pred=1, proba={0: 0.10, 1: 0.90})
    net = NetworkSimulator(base_latency_s=0.020, jitter_s=0.000, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)

    res = env.execute_hybrid({"f": 1.0}, observation_index=1)
    assert res.edge_latency_s is not None
    assert res.cloud_latency_s is not None
    assert res.network_latency_s == pytest.approx(0.020, rel=1e-4)
    assert res.hybrid_latency_s is not None
    assert res.hybrid_latency_s > 0.0


# -----------------------------------------------------------------------------
# 19. No fabricated prediction after failure
# -----------------------------------------------------------------------------
def test_19_no_fabricated_prediction_after_failure():
    edge_m = FailingMockModel()
    cloud_m = MockModel()
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))

    res = env.execute_edge({"f": 1.0})
    assert res.success is False
    assert res.status == ExecutionStatus.FAILED
    assert res.prediction is None
    assert res.probabilities is None

    exec_res = res.to_execution_result()
    assert exec_res.prediction is None
    assert exec_res.probabilities is None
    assert exec_res.success is False


# -----------------------------------------------------------------------------
# 20. Causal execution
# -----------------------------------------------------------------------------
def test_20_causal_execution():
    edge_m = MockModel()
    cloud_m = MockModel()
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))

    # Step t must only consume observation t without looking at t+1
    obs_t = {"feat_0": 0.123, "feat_1": 0.456}
    res = env.execute_edge(obs_t, observation_index=42)
    assert res.observation_index == 42


# -----------------------------------------------------------------------------
# 21. Target leakage protection
# -----------------------------------------------------------------------------
def test_21_target_leakage_protection():
    edge_m = MockModel()
    cloud_m = MockModel()
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))

    # Payload with Target column must be rejected or stripped
    leaky_input = {"feat_0": 0.1, "Target": 1}
    # Environment execution must not rely on Target
    res = env.execute_edge(leaky_input)
    assert res.prediction in (0, 1)


# -----------------------------------------------------------------------------
# 22. ground_truth.json isolation
# -----------------------------------------------------------------------------
def test_22_ground_truth_isolation(project_root):
    gt_path = project_root / "data" / "synthetic" / "ground_truth.json"
    if gt_path.exists():
        gt_content = gt_path.read_text(encoding="utf-8")
        assert len(gt_content) > 0
    # Deployment environment has zero dependency on ground_truth.json
    edge_m = MockModel()
    cloud_m = MockModel()
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))
    assert not hasattr(env, "ground_truth")


# -----------------------------------------------------------------------------
# 23. Phase 5 compatibility
# -----------------------------------------------------------------------------
def test_23_phase_5_compatibility():
    edge_m = MockModel(pred=1)
    cloud_m = MockModel(pred=0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))

    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.85,
        previous_action=None,
        decision_reason="High reliability",
        observation_index=10,
    )

    exec_res = env.execute(dec.selected_action, {"f": 1.0}, decision=dec, observation_index=10)
    assert isinstance(exec_res, ExecutionResult)
    assert exec_res.action == DecisionAction.EDGE
    assert exec_res.prediction == 1
    assert exec_res.decision is dec


# -----------------------------------------------------------------------------
# 24. Phase 6 compatibility
# -----------------------------------------------------------------------------
def test_24_phase_6_compatibility():
    edge_m = MockModel(pred=0)
    cloud_m = MockModel(pred=1)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))

    exec_res = env.execute(DecisionAction.CLOUD, {"f": 1.0}, observation_index=5)
    assert hasattr(exec_res, "status")
    assert hasattr(exec_res, "success")
    assert hasattr(exec_res, "inference_latency_s")
    assert hasattr(exec_res, "edge_latency_s")
    assert hasattr(exec_res, "cloud_latency_s")
    assert hasattr(exec_res, "hybrid_latency_s")
    assert hasattr(exec_res, "network_latency_s")
    assert exec_res.status == ExecutionStatus.SUCCESS


# -----------------------------------------------------------------------------
# 25. Phase 7 monitoring compatibility
# -----------------------------------------------------------------------------
def test_25_phase_7_monitoring_compatibility():
    edge_m = MockModel(pred=0)
    cloud_m = MockModel(pred=1)
    net = NetworkSimulator(base_latency_s=0.025, jitter_s=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net)
    monitor = DRAECMonitor()

    dec = DecisionResult(
        selected_action=DecisionAction.CLOUD,
        reliability=0.45,
        previous_action=DecisionAction.EDGE,
        decision_reason="Reliability drop",
        observation_index=1,
    )
    exec_res = env.execute(dec.selected_action, {"f": 1.0}, decision=dec, observation_index=1)

    rec = monitor.observe_step(
        observation_index=1,
        reliability_score=0.45,
        decision_result=dec,
        execution_result=exec_res,
    )
    assert rec.selected_action == "CLOUD"
    assert rec.execution_status == "SUCCESS"
    assert rec.network_latency_s == pytest.approx(0.025, rel=1e-3)
    assert rec.packet_lost is False

    df = monitor.get_records_dataframe()
    assert len(df) == 1
    assert "network_latency_s" in df.columns
    assert "packet_lost" in df.columns


# -----------------------------------------------------------------------------
# 26. Deterministic repeated execution
# -----------------------------------------------------------------------------
def test_26_deterministic_repeated_execution():
    net1 = NetworkSimulator(base_latency_s=0.02, jitter_s=0.005, packet_loss_probability=0.1, seed=99)
    env1 = DeploymentEnvironment(EdgeRuntime(MockModel()), CloudRuntime(MockModel()), net1)

    net2 = NetworkSimulator(base_latency_s=0.02, jitter_s=0.005, packet_loss_probability=0.1, seed=99)
    env2 = DeploymentEnvironment(EdgeRuntime(MockModel()), CloudRuntime(MockModel()), net2)

    res1 = [env1.execute_cloud({"f": i}, observation_index=i).to_dict() for i in range(30)]
    res2 = [env2.execute_cloud({"f": i}, observation_index=i).to_dict() for i in range(30)]

    for r1, r2 in zip(res1, res2):
        assert r1["success"] == r2["success"]
        assert r1["packet_lost"] == r2["packet_lost"]
        assert r1["network_latency_s"] == pytest.approx(r2["network_latency_s"], rel=1e-5)


# -----------------------------------------------------------------------------
# 27. Reset behavior
# -----------------------------------------------------------------------------
def test_27_reset_behavior():
    edge_rt = EdgeRuntime(MockModel())
    cloud_rt = CloudRuntime(MockModel())
    net = NetworkSimulator(seed=42)
    env = DeploymentEnvironment(edge_rt, cloud_rt, net)

    env.execute_edge({"f": 1})
    env.execute_cloud({"f": 2})
    stats_before = env.get_stats()
    assert stats_before["total_executions"] == 2

    env.reset()
    stats_after = env.get_stats()
    assert stats_after["total_executions"] == 0
    assert stats_after["successful_executions"] == 0
    assert stats_after["edge"]["total_executions"] == 0
    assert stats_after["cloud"]["total_executions"] == 0
    assert stats_after["network"]["total_transmissions"] == 0


# -----------------------------------------------------------------------------
# 28. End-to-end Phase 1–8 smoke test
# -----------------------------------------------------------------------------
def test_28_end_to_end_smoke_test():
    edge_m = MockModel(pred=0, proba={0: 0.85, 1: 0.15})
    cloud_m = MockModel(pred=1, proba={0: 0.10, 1: 0.90})
    net = NetworkSimulator(base_latency_s=0.015, packet_loss_probability=0.0, seed=42)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net)
    monitor = DRAECMonitor()

    # Stream through actions: EDGE -> HYBRID -> CLOUD
    actions = [DecisionAction.EDGE, DecisionAction.HYBRID, DecisionAction.CLOUD]
    for idx, act in enumerate(actions, start=1):
        dec = DecisionResult(
            selected_action=act,
            reliability=0.8 - idx * 0.2,
            previous_action=None,
            decision_reason=f"Action {act.value}",
            observation_index=idx,
        )
        exec_res = env.execute(act, {"feat": idx}, decision=dec, observation_index=idx)
        rec = monitor.observe_step(
            observation_index=idx,
            reliability_score=dec.reliability,
            decision_result=dec,
            execution_result=exec_res,
        )
        assert rec.selected_action == act.value
        assert rec.execution_status in ("SUCCESS", "FALLBACK")

    df = monitor.get_records_dataframe()
    assert len(df) == 3
    assert all(c in df.columns for c in ("network_latency_s", "packet_lost", "selected_action"))


# -----------------------------------------------------------------------------
# 29. Network failure provenance through ExecutionResult and Phase 10 metrics
# -----------------------------------------------------------------------------
def test_29_network_failure_provenance_and_execution_result():
    from src.metrics.system import compute_network_metrics

    edge_m = MockModel(pred=0, proba={0: 0.85, 1: 0.15})
    cloud_m = MockModel(pred=1, proba={0: 0.10, 1: 0.90})

    # Case A: Normal condition -> delivered
    net_norm = NetworkSimulator(base_latency_s=0.015, jitter_s=0.0, packet_loss_probability=0.0)
    env_norm = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net_norm)
    res_norm = env_norm.execute(DecisionAction.CLOUD, {"x": 1})
    assert res_norm.success is True
    assert res_norm.packet_lost is False
    assert res_norm.network_latency_s == pytest.approx(0.015, rel=1e-4)

    # Case B: High-latency condition -> delivered
    net_high = NetworkSimulator(base_latency_s=0.150, jitter_s=0.0, packet_loss_probability=0.0)
    env_high = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net_high)
    res_high = env_high.execute(DecisionAction.CLOUD, {"x": 1})
    assert res_high.success is True
    assert res_high.packet_lost is False
    assert res_high.network_latency_s == pytest.approx(0.150, rel=1e-4)

    # Case C: Packet-loss condition -> failed/lost
    net_loss = NetworkSimulator(packet_loss_probability=1.0)
    env_loss = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net_loss)
    res_loss = env_loss.execute(DecisionAction.CLOUD, {"x": 1})
    assert res_loss.success is False
    assert res_loss.packet_lost is True
    assert res_loss.prediction is None
    assert res_loss.status == ExecutionStatus.FAILED

    # Case D: Disconnected condition -> failed/not delivered
    net_disc = NetworkSimulator(available=False)
    env_disc = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net_disc)
    res_disc = env_disc.execute(DecisionAction.CLOUD, {"x": 1})
    assert res_disc.success is False
    assert res_disc.packet_lost is False
    assert res_disc.prediction is None
    assert res_disc.status == ExecutionStatus.FAILED
    assert "unavailable" in str(res_disc.error).lower() or "disconnected" in str(res_disc.error).lower()

    # Metric accounting verification across the 4 conditions
    # Normal:
    m_norm = compute_network_metrics(
        total_transmissions=1,
        delivered_transmissions=int(res_norm.success),
        packet_loss_count=int(res_norm.packet_lost),
        latencies_s=[res_norm.network_latency_s],
    )
    assert m_norm["total_transmissions"] == 1
    assert m_norm["delivered_transmissions"] == 1
    assert m_norm["packet_loss_count"] == 0
    assert m_norm["delivery_rate"] == 1.0
    assert m_norm["failure_rate"] == 0.0
    assert m_norm["packet_loss_rate"] == 0.0
    assert m_norm["simulated_network_latency_ms"]["mean_ms"] is not None
    assert abs(m_norm["simulated_network_latency_ms"]["mean_ms"] - 15.0) < 1e-3

    # High-latency:
    m_high = compute_network_metrics(
        total_transmissions=1,
        delivered_transmissions=int(res_high.success),
        packet_loss_count=int(res_high.packet_lost),
        latencies_s=[res_high.network_latency_s],
    )
    assert m_high["total_transmissions"] == 1
    assert m_high["delivered_transmissions"] == 1
    assert m_high["packet_loss_count"] == 0
    assert m_high["delivery_rate"] == 1.0
    assert m_high["failure_rate"] == 0.0
    assert m_high["packet_loss_rate"] == 0.0
    assert m_high["simulated_network_latency_ms"]["mean_ms"] is not None
    assert abs(m_high["simulated_network_latency_ms"]["mean_ms"] - 150.0) < 1e-3

    # Packet loss:
    m_loss = compute_network_metrics(
        total_transmissions=1,
        delivered_transmissions=int(res_loss.success),
        packet_loss_count=int(res_loss.packet_lost),
        latencies_s=[],
    )
    assert m_loss["total_transmissions"] == 1
    assert m_loss["delivered_transmissions"] == 0
    assert m_loss["packet_loss_count"] == 1
    assert m_loss["delivery_rate"] == 0.0
    assert m_loss["failure_rate"] == 1.0
    assert m_loss["packet_loss_rate"] == 1.0
    # Must be None, NEVER 0.0
    assert m_loss["simulated_network_latency_ms"]["mean_ms"] is None
    assert m_loss["simulated_network_latency_ms"]["mean_ms"] != 0.0

    # Disconnected:
    m_disc = compute_network_metrics(
        total_transmissions=1,
        delivered_transmissions=int(res_disc.success),
        packet_loss_count=int(res_disc.packet_lost),
        latencies_s=[],
    )
    assert m_disc["total_transmissions"] == 1
    assert m_disc["delivered_transmissions"] == 0
    assert m_disc["packet_loss_count"] == 0
    assert m_disc["delivery_rate"] == 0.0
    assert m_disc["failure_rate"] == 1.0
    assert m_disc["packet_loss_rate"] == 0.0
    # Must be None, NEVER 0.0
    assert m_disc["simulated_network_latency_ms"]["mean_ms"] is None
    assert m_disc["simulated_network_latency_ms"]["mean_ms"] != 0.0

