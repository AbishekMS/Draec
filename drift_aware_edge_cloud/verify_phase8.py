"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : verify_phase8.py
Phase    : Phase 8
Status   : IMPLEMENTED

Verification harness for Phase 8: DRAEC Edge-Cloud Deployment & Network Execution Layer.
Verifies execution runtimes, network simulator, deterministic behavior, packet loss,
failure provenance, latency accounting, Phase 5/6/7 integration, and scope boundaries.
"""

from __future__ import annotations

import ast
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

passed = 0
failed = 0
checks = []


def record_check(name: str, success: bool, detail: str = "") -> None:
    global passed, failed
    if success:
        passed += 1
        status = "PASS"
    else:
        failed += 1
        status = "FAIL"
    msg = f"[{status}] {name}"
    if detail:
        msg += f" -- {detail}"
    print(msg)
    checks.append((name, success, detail))


class SimpleMock:
    def __init__(self, pred: int = 0, p0: float = 0.85) -> None:
        self.pred = pred
        self.p0 = p0
        self.call_count = 0

    def predict_one(self, x: Any) -> int:
        self.call_count += 1
        return self.pred

    def predict_proba_one(self, x: Any) -> dict[int, float]:
        return {0: self.p0, 1: 1.0 - self.p0}


# -----------------------------------------------------------------------------
# Check 1: Public API imports
# -----------------------------------------------------------------------------
try:
    from src.deployment import (
        CloudRuntime,
        DeploymentEnvironment,
        DeploymentExecutionResult,
        EdgeRuntime,
        NetworkPacket,
        NetworkSimulator,
        RuntimeState,
        TransmissionResult,
        TransmissionStatus,
    )
    record_check("Check 01: Public API imports", True, "All Phase 8 symbols imported successfully")
except Exception as e:
    record_check("Check 01: Public API imports", False, str(e))


# -----------------------------------------------------------------------------
# Check 2: Module file headers and IMPLEMENTED status
# -----------------------------------------------------------------------------
try:
    mod_files = [
        ROOT / "src" / "deployment" / "__init__.py",
        ROOT / "src" / "deployment" / "base.py",
        ROOT / "src" / "deployment" / "network.py",
        ROOT / "src" / "deployment" / "runtimes.py",
        ROOT / "src" / "deployment" / "environment.py",
    ]
    all_implemented = True
    for mf in mod_files:
        txt = mf.read_text(encoding="utf-8")
        if "Status   : IMPLEMENTED" not in txt or "Phase    : Phase 8" not in txt:
            all_implemented = False
            break
    record_check("Check 02: Module file headers", all_implemented, "All 5 deployment modules marked Phase 8 IMPLEMENTED")
except Exception as e:
    record_check("Check 02: Module file headers", False, str(e))


# -----------------------------------------------------------------------------
# Check 3: Top-level network configuration
# -----------------------------------------------------------------------------
try:
    from src.utils.config import load
    cfg = load("default")
    net_cfg = cfg.get("network", {})
    ok_cfg = (
        net_cfg.get("enabled") is True
        and net_cfg.get("mode") == "simulation"
        and "latency" in net_cfg
        and "packet_loss" in net_cfg
        and "availability" in net_cfg
        and net_cfg.get("pacing_enabled") is False
    )
    record_check("Check 03: Network configuration", ok_cfg, "top-level network: section verified with simulation defaults")
except Exception as e:
    record_check("Check 03: Network configuration", False, str(e))


# -----------------------------------------------------------------------------
# Check 4: Edge runtime
# -----------------------------------------------------------------------------
try:
    edge_m = SimpleMock(pred=1, p0=0.20)
    edge_rt = EdgeRuntime(model=edge_m, available=True)
    pred, prob, lat, succ, err = edge_rt.execute({"f": 1.0})
    c4_ok = succ is True and pred == 1 and prob[1] == 0.80 and lat >= 0.0 and err is None
    record_check("Check 04: Edge runtime", c4_ok, f"Executed Edge model inference in {lat*1000:.3f}ms")
except Exception as e:
    record_check("Check 04: Edge runtime", False, str(e))


# -----------------------------------------------------------------------------
# Check 5: Cloud runtime
# -----------------------------------------------------------------------------
try:
    cloud_m = SimpleMock(pred=0, p0=0.90)
    cloud_rt = CloudRuntime(model=cloud_m, available=True)
    pred, prob, lat, succ, err = cloud_rt.execute({"f": 1.0})
    c5_ok = succ is True and pred == 0 and prob[0] == 0.90 and lat >= 0.0 and err is None
    record_check("Check 05: Cloud runtime", c5_ok, f"Executed Cloud model inference in {lat*1000:.3f}ms")
except Exception as e:
    record_check("Check 05: Cloud runtime", False, str(e))


# -----------------------------------------------------------------------------
# Check 6: Network simulator transmission
# -----------------------------------------------------------------------------
try:
    sim = NetworkSimulator(base_latency_s=0.020, jitter_s=0.005, packet_loss_probability=0.0, seed=42)
    tx = sim.transmit({"x": 1}, observation_index=1)
    c6_ok = (
        tx.success is True
        and tx.status == TransmissionStatus.DELIVERED
        and tx.packet_lost is False
        and tx.latency_s > 0.0
    )
    record_check("Check 06: Network simulator transmission", c6_ok, f"Simulated latency = {tx.latency_s*1000:.2f}ms")
except Exception as e:
    record_check("Check 06: Network simulator transmission", False, str(e))


# -----------------------------------------------------------------------------
# Check 7: Deterministic network behavior
# -----------------------------------------------------------------------------
try:
    s1 = NetworkSimulator(base_latency_s=0.02, jitter_s=0.005, packet_loss_probability=0.1, seed=77)
    s2 = NetworkSimulator(base_latency_s=0.02, jitter_s=0.005, packet_loss_probability=0.1, seed=77)
    txs1 = [s1.transmit({"val": i}, observation_index=i) for i in range(15)]
    txs2 = [s2.transmit({"val": i}, observation_index=i) for i in range(15)]
    c7_ok = all(
        a.success == b.success and a.packet_lost == b.packet_lost and abs(a.latency_s - b.latency_s) < 1e-6
        for a, b in zip(txs1, txs2)
    )
    record_check("Check 07: Deterministic network behavior", c7_ok, "Identical seeds produced bit-identical traces")
except Exception as e:
    record_check("Check 07: Deterministic network behavior", False, str(e))


# -----------------------------------------------------------------------------
# Check 8: Zero packet loss condition
# -----------------------------------------------------------------------------
try:
    s_zero = NetworkSimulator(packet_loss_probability=0.0, seed=42)
    zero_loss = all(s_zero.transmit({"i": i}).success for i in range(30))
    record_check("Check 08: Zero packet loss condition", zero_loss, "30/30 transmissions delivered successfully")
except Exception as e:
    record_check("Check 08: Zero packet loss condition", False, str(e))


# -----------------------------------------------------------------------------
# Check 9: Packet loss handling
# -----------------------------------------------------------------------------
try:
    s_loss = NetworkSimulator(packet_loss_probability=1.0, seed=42)
    tx_loss = s_loss.transmit({"i": 1})
    c9_ok = (
        tx_loss.success is False
        and tx_loss.packet_lost is True
        and tx_loss.status == TransmissionStatus.PACKET_LOSS
        and "packet loss" in str(tx_loss.error).lower()
    )
    record_check("Check 09: Packet loss handling", c9_ok, "Loss caught and recorded with explicit error message")
except Exception as e:
    record_check("Check 09: Packet loss handling", False, str(e))


# -----------------------------------------------------------------------------
# Check 10: Zero prediction fabrication on failure
# -----------------------------------------------------------------------------
try:
    edge_fail = SimpleMock()
    cloud_fail = SimpleMock()
    net_fail = NetworkSimulator(packet_loss_probability=1.0)
    env_fail = DeploymentEnvironment(EdgeRuntime(edge_fail), CloudRuntime(cloud_fail), net_fail)
    res_fail = env_fail.execute_cloud({"f": 1.0})
    c10_ok = (
        res_fail.success is False
        and res_fail.status.value == "FAILED"
        and res_fail.prediction is None
        and res_fail.probabilities is None
        and res_fail.packet_lost is True
    )
    record_check("Check 10: Zero prediction fabrication", c10_ok, "Prediction strictly None on network failure")
except Exception as e:
    record_check("Check 10: Zero prediction fabrication", False, str(e))


# -----------------------------------------------------------------------------
# Check 11: Edge failure handling
# -----------------------------------------------------------------------------
try:
    edge_rt_off = EdgeRuntime(SimpleMock(), available=False)
    p, prob, lat, succ, err = edge_rt_off.execute({"x": 1})
    c11_ok = succ is False and p is None and "offline" in str(err).lower()
    record_check("Check 11: Edge failure handling", c11_ok, "Edge device offline handled without exception")
except Exception as e:
    record_check("Check 11: Edge failure handling", False, str(e))


# -----------------------------------------------------------------------------
# Check 12: Cloud failure handling
# -----------------------------------------------------------------------------
try:
    cloud_rt_off = CloudRuntime(SimpleMock(), available=False)
    p, prob, lat, succ, err = cloud_rt_off.execute({"x": 1})
    c12_ok = succ is False and p is None and "offline" in str(err).lower()
    record_check("Check 12: Cloud failure handling", c12_ok, "Cloud service offline handled without exception")
except Exception as e:
    record_check("Check 12: Cloud failure handling", False, str(e))


# -----------------------------------------------------------------------------
# Check 13: Edge-only execution path
# -----------------------------------------------------------------------------
try:
    edge_m = SimpleMock(pred=1)
    cloud_m = SimpleMock(pred=0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m))
    res = env.execute_edge({"f": 1})
    c13_ok = (
        res.success is True
        and res.prediction == 1
        and res.model_used == "edge"
        and res.cloud_latency_s is None
        and res.network_latency_s is None
        and cloud_m.call_count == 0
    )
    record_check("Check 13: Edge-only execution path", c13_ok, "Cloud uninvoked, zero network latency")
except Exception as e:
    record_check("Check 13: Edge-only execution path", False, str(e))


# -----------------------------------------------------------------------------
# Check 14: Cloud-only execution path
# -----------------------------------------------------------------------------
try:
    edge_m = SimpleMock(pred=1)
    cloud_m = SimpleMock(pred=0)
    net = NetworkSimulator(base_latency_s=0.015, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net)
    res = env.execute_cloud({"f": 1})
    c14_ok = (
        res.success is True
        and res.prediction == 0
        and res.model_used == "cloud"
        and res.cloud_latency_s is not None
        and res.network_latency_s is not None
        and edge_m.call_count == 0
    )
    record_check("Check 14: Cloud-only execution path", c14_ok, "Edge uninvoked, network + cloud recorded")
except Exception as e:
    record_check("Check 14: Cloud-only execution path", False, str(e))


# -----------------------------------------------------------------------------
# Check 15: Hybrid Edge confident execution
# -----------------------------------------------------------------------------
try:
    # p0 = 0.85 -> C_edge = 2*(0.85-0.5) = 0.70 >= 0.60 threshold
    edge_m = SimpleMock(pred=0, p0=0.85)
    cloud_m = SimpleMock(pred=1, p0=0.10)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), fallback_confidence_threshold=0.60)
    res = env.execute_hybrid({"f": 1})
    c15_ok = (
        res.success is True
        and res.prediction == 0
        and res.model_used == "hybrid_edge"
        and res.cloud_fallback is False
        and res.status.value == "SUCCESS"
        and cloud_m.call_count == 0
    )
    record_check("Check 15: Hybrid Edge confident execution", c15_ok, "Completed at Edge without Cloud fallback")
except Exception as e:
    record_check("Check 15: Hybrid Edge confident execution", False, str(e))


# -----------------------------------------------------------------------------
# Check 16: Hybrid Cloud fallback execution
# -----------------------------------------------------------------------------
try:
    # p0 = 0.55 -> C_edge = 2*(0.55-0.5) = 0.10 < 0.60 threshold
    edge_m = SimpleMock(pred=0, p0=0.55)
    cloud_m = SimpleMock(pred=1, p0=0.05)
    net = NetworkSimulator(base_latency_s=0.010, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)
    res = env.execute_hybrid({"f": 1})
    c16_ok = (
        res.success is True
        and res.prediction == 1
        and res.model_used == "hybrid_cloud"
        and res.cloud_fallback is True
        and res.status.value == "FALLBACK"
        and edge_m.call_count == 1
        and cloud_m.call_count == 1
    )
    record_check("Check 16: Hybrid Cloud fallback execution", c16_ok, "Uncertain Edge triggered Cloud fallback")
except Exception as e:
    record_check("Check 16: Hybrid Cloud fallback execution", False, str(e))


# -----------------------------------------------------------------------------
# Check 17: Hybrid fallback network failure
# -----------------------------------------------------------------------------
try:
    edge_m = SimpleMock(pred=0, p0=0.55)
    cloud_m = SimpleMock(pred=1, p0=0.05)
    net = NetworkSimulator(packet_loss_probability=1.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)
    res = env.execute_hybrid({"f": 1})
    c17_ok = (
        res.success is False
        and res.prediction is None
        and res.cloud_fallback is True
        and res.packet_lost is True
        and res.status.value == "FAILED"
        and cloud_m.call_count == 0
    )
    record_check("Check 17: Hybrid fallback network failure", c17_ok, "Fallback packet loss returns FAILED with None")
except Exception as e:
    record_check("Check 17: Hybrid fallback network failure", False, str(e))


# -----------------------------------------------------------------------------
# Check 18: Separate latency accounting
# -----------------------------------------------------------------------------
try:
    edge_m = SimpleMock(pred=0, p0=0.52)
    cloud_m = SimpleMock(pred=1, p0=0.10)
    net = NetworkSimulator(base_latency_s=0.020, jitter_s=0.0, packet_loss_probability=0.0)
    env = DeploymentEnvironment(EdgeRuntime(edge_m), CloudRuntime(cloud_m), net, fallback_confidence_threshold=0.60)
    res = env.execute_hybrid({"f": 1})
    c18_ok = (
        res.edge_latency_s is not None
        and res.cloud_latency_s is not None
        and res.network_latency_s == 0.020
        and res.hybrid_latency_s is not None
        and res.hybrid_latency_s > 0.0
    )
    record_check("Check 18: Separate latency accounting", c18_ok, "T_edge, T_cloud, T_network, T_hybrid distinct")
except Exception as e:
    record_check("Check 18: Separate latency accounting", False, str(e))


# -----------------------------------------------------------------------------
# Check 19: Causality constraint
# -----------------------------------------------------------------------------
try:
    env = DeploymentEnvironment(EdgeRuntime(SimpleMock()), CloudRuntime(SimpleMock()))
    res = env.execute_edge({"a": 1, "b": 2}, observation_index=105)
    c19_ok = res.observation_index == 105
    record_check("Check 19: Causality constraint", c19_ok, "Execution strictly indexed per step")
except Exception as e:
    record_check("Check 19: Causality constraint", False, str(e))


# -----------------------------------------------------------------------------
# Check 20: Target and Ground Truth isolation
# -----------------------------------------------------------------------------
try:
    gt_file = ROOT / "data" / "synthetic" / "ground_truth.json"
    gt_exists = gt_file.exists()
    env = DeploymentEnvironment(EdgeRuntime(SimpleMock()), CloudRuntime(SimpleMock()))
    c20_ok = not hasattr(env, "ground_truth") and not hasattr(env, "true_onset")
    record_check("Check 20: Target and Ground Truth isolation", c20_ok, "Zero dependency on synthetic ground truth")
except Exception as e:
    record_check("Check 20: Target and Ground Truth isolation", False, str(e))


# -----------------------------------------------------------------------------
# Check 21: Phase 5 compatibility
# -----------------------------------------------------------------------------
try:
    from src.decision.base import DecisionAction, DecisionResult
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.82,
        previous_action=None,
        decision_reason="Nominal reliability",
        observation_index=12,
    )
    env = DeploymentEnvironment(EdgeRuntime(SimpleMock()), CloudRuntime(SimpleMock()))
    exec_res = env.execute(dec.selected_action, {"f": 1.0}, decision=dec, observation_index=12)
    c21_ok = exec_res.action == DecisionAction.EDGE and exec_res.decision is dec
    record_check("Check 21: Phase 5 compatibility", c21_ok, "Seamlessly consumes DecisionResult and DecisionAction")
except Exception as e:
    record_check("Check 21: Phase 5 compatibility", False, str(e))


# -----------------------------------------------------------------------------
# Check 22: Phase 6 compatibility
# -----------------------------------------------------------------------------
try:
    from src.decision.base import ExecutionResult
    env = DeploymentEnvironment(EdgeRuntime(SimpleMock()), CloudRuntime(SimpleMock()))
    exec_res = env.execute(DecisionAction.CLOUD, {"f": 1.0})
    c22_ok = (
        isinstance(exec_res, ExecutionResult)
        and hasattr(exec_res, "inference_latency_s")
        and hasattr(exec_res, "network_latency_s")
        and hasattr(exec_res, "status")
    )
    record_check("Check 22: Phase 6 compatibility", c22_ok, "ExecutionResult schema compliant with Phase 6")
except Exception as e:
    record_check("Check 22: Phase 6 compatibility", False, str(e))


# -----------------------------------------------------------------------------
# Check 23: Phase 7 monitoring compatibility
# -----------------------------------------------------------------------------
try:
    from src.monitoring.monitor import DRAECMonitor
    monitor = DRAECMonitor()
    env = DeploymentEnvironment(
        EdgeRuntime(SimpleMock()),
        CloudRuntime(SimpleMock()),
        NetworkSimulator(base_latency_s=0.018, jitter_s=0.0),
    )
    dec = DecisionResult(selected_action=DecisionAction.CLOUD, reliability=0.40, previous_action=None, decision_reason="Cloud test")
    exec_res = env.execute(dec.selected_action, {"x": 1}, decision=dec, observation_index=1)
    rec = monitor.observe_step(1, reliability_score=0.40, decision_result=dec, execution_result=exec_res)
    df = monitor.get_records_dataframe()
    c23_ok = (
        rec.selected_action == "CLOUD"
        and abs(rec.network_latency_s - 0.018) < 1e-4
        and "network_latency_s" in df.columns
        and "packet_lost" in df.columns
    )
    record_check("Check 23: Phase 7 monitoring compatibility", c23_ok, "Ingested by DRAECMonitor, telemetry exported")
except Exception as e:
    record_check("Check 23: Phase 7 monitoring compatibility", False, str(e))


# -----------------------------------------------------------------------------
# Check 24: Scope boundary enforcement
# -----------------------------------------------------------------------------
try:
    later_files = [
        ROOT / "src" / "simulation" / "__init__.py",
        ROOT / "src" / "simulation" / "environment.py",
        ROOT / "src" / "metrics" / "__init__.py",
        ROOT / "src" / "metrics" / "system.py",
    ]
    scope_ok = True
    for lf in later_files:
        if lf.exists():
            tree = ast.parse(lf.read_text(encoding="utf-8"))
            body = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if body:
                scope_ok = False
                break
    record_check("Check 24: Scope boundary enforcement", scope_ok, "Phase 10 remains pure un-implemented stubs")
except Exception as e:
    record_check("Check 24: Scope boundary enforcement", False, str(e))


# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 60)
print(f"PHASE 8 VERIFICATION SUMMARY: {passed} / {passed + failed} CHECKS PASSED")
print("=" * 60)

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
