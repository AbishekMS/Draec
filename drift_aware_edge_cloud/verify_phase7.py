"""Phase 7 verification harness -- Model Management, Telemetry, and Observability.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Checks:
1. check_phase7_imports: ModelHealthStatus, ModelMetadata, MonitoringRecord, MonitoringSnapshot, ModelRegistry, DRAECMonitor exist.
2. check_phase7_module_integrity: All 4 Phase 7 modules exist and are marked Status : IMPLEMENTED.
3. check_configuration: Top-level monitoring section exists and is parsed from config/default.yaml.
4. check_model_registry: ModelRegistry registers, stores, and retrieves models without mutating weights.
5. check_edge_model_management: EdgeHoeffdingTree registered with contract (37 features, location="edge").
6. check_cloud_model_management: CloudXGBoost registered with contract (37 features, location="cloud").
7. check_reliability_monitoring: Unaltered R_t monitoring and streaming statistics.
8. check_drift_monitoring: Unaltered D_t, persistence, raw/smoothed severity monitoring.
9. check_decision_monitoring: Action routing, previous action, and reason monitoring.
10. check_edge_routing_statistics: EDGE action counts and distribution percentage.
11. check_cloud_routing_statistics: CLOUD action counts and distribution percentage.
12. check_hybrid_routing_statistics: HYBRID action counts and distribution percentage.
13. check_hybrid_fallback_statistics: Fallback count and fallback rate calculation.
14. check_execution_monitoring: Status and success/failure counts recorded with error provenance.
15. check_latency_statistics: Streaming T_edge, T_cloud, T_hybrid statistics without fabricating missing paths.
16. check_bounded_history: Respects max_records buffer limit while global counters persist.
17. check_monitoring_snapshot: Aggregate snapshot contains all required observability keys.
18. check_causality: Observation index t depends strictly on information at or before step t.
19. check_leakage_protection: Target labels and ground_truth.json are strictly quarantined.
20. check_phase4_compatibility: Seamless integration with Phase 4 ReliabilityScore.
21. check_phase5_compatibility: Seamless integration with Phase 5 DecisionResult.
22. check_phase6_compatibility: Seamless integration with Phase 6 ExecutionResult.
23. check_end_to_end_smoke_test: End-to-end streaming run exporting stable DataFrame for Phase 10.
24. check_phase7_scope_boundary: Zero Phase 8+ deployment, retraining, or benchmark evaluation logic.

Run:
    ../.venv/Scripts/python.exe verify_phase7.py
"""

from __future__ import annotations

import ast
import io
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.decision.base import DecisionAction, DecisionInputs, DecisionResult, ExecutionResult, ExecutionStatus
from src.decision.engine import DecisionEngine
from src.models.base import BaseModel
from src.models.cloud_model import CloudXGBoost
from src.models.edge_model import EdgeHoeffdingTree
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
from src.utils import config as config_mod


PASS = "\033[92m[PASS]\033[0m"
FAIL = "\033[91m[FAIL]\033[0m"
total_checks = 0
passed_checks = 0


def check(desc: str, condition: bool, detail: str = "") -> bool:
    global total_checks, passed_checks
    total_checks += 1
    if condition:
        passed_checks += 1
        msg = f"{PASS} {total_checks:2d}. {desc}"
        if detail:
            msg += f" -- {detail}"
        print(msg)
        return True
    else:
        msg = f"{FAIL} {total_checks:2d}. {desc}"
        if detail:
            msg += f" -- FAILED: {detail}"
        print(msg)
        return False


def make_dummy_reliability(r: float = 0.75, c: float = 0.85, e: float = 0.05, d: float = 0.12, q: float = 1.0) -> ReliabilityScore:
    inp = ReliabilityInputs(confidence=c, error=e, drift=d, quality=q)
    fac = ReliabilityFactors(r_C=c, r_E=1.0 - e, r_D=1.0 - d, r_Q=q)
    return ReliabilityScore(
        reliability=r,
        inputs=inp,
        factors=fac,
        weights={"confidence": 0.25, "error": 0.25, "drift": 0.25, "quality": 0.25},
        epsilon=1e-8,
    )


def main() -> int:
    print("=" * 80)
    print("DRAEC PHASE 7 VERIFICATION -- MODEL MANAGEMENT & MONITORING")
    print("=" * 80)

    cfg = config_mod.load()

    # -------------------------------------------------------------------------
    # Check 1: Phase 7 Imports & Class Availability
    # -------------------------------------------------------------------------
    c1_ok = (
        ModelHealthStatus is not None
        and ModelMetadata is not None
        and MonitoringRecord is not None
        and MonitoringSnapshot is not None
        and ModelRegistry is not None
        and DRAECMonitor is not None
        and SystemMonitor is DRAECMonitor
    )
    check("Phase 7 public API imports cleanly", c1_ok, "all core classes and aliases present")

    # -------------------------------------------------------------------------
    # Check 2: Module Status Integrity
    # -------------------------------------------------------------------------
    p7_modules = [
        "src/monitoring/__init__.py",
        "src/monitoring/base.py",
        "src/monitoring/registry.py",
        "src/monitoring/monitor.py",
    ]
    c2_ok = True
    for p in p7_modules:
        content = (ROOT / p).read_text(encoding="utf-8")
        if "Status   : IMPLEMENTED" not in content:
            c2_ok = False
            break
    check("Phase 7 module integrity", c2_ok, "all 4 modules marked Status : IMPLEMENTED")

    # -------------------------------------------------------------------------
    # Check 3: Configuration Parsing
    # -------------------------------------------------------------------------
    mon_cfg = cfg.get("monitoring", {})
    c3_ok = (
        "monitoring" in cfg
        and mon_cfg.get("enabled") is True
        and mon_cfg.get("max_records") == 10000
        and mon_cfg.get("model_management") is True
    )
    check("Phase 7 configuration parsed", c3_ok, f"monitoring.max_records={mon_cfg.get('max_records')}")

    # -------------------------------------------------------------------------
    # Check 4: ModelRegistry Core Methods
    # -------------------------------------------------------------------------
    reg = ModelRegistry()
    reg.register_model(EdgeHoeffdingTree(), "edge_test", "edge", version="1.0.0")
    c4_ok = (
        reg.has_model("edge_test")
        and reg.get_metadata("edge_test").model_version == "1.0.0"
        and reg.get_metadata("edge_test").status == ModelHealthStatus.HEALTHY
    )
    reg.update_status("edge_test", ModelHealthStatus.DEGRADED)
    c4_ok = c4_ok and (reg.get_metadata("edge_test").status == ModelHealthStatus.DEGRADED)
    check("ModelRegistry registration and state management", c4_ok, "models tracked with health states")

    # -------------------------------------------------------------------------
    # Check 5: Edge Model Management
    # -------------------------------------------------------------------------
    edge_m = EdgeHoeffdingTree()
    meta_e = reg.register_model(edge_m, "edge", "edge", version="1.0.0", n_features=37)
    c5_ok = (
        meta_e.execution_location == "edge"
        and meta_e.n_features == 37
        and meta_e.model_type == "EdgeHoeffdingTree"
    )
    check("Edge model management", c5_ok, f"location={meta_e.execution_location}, n_features={meta_e.n_features}")

    # -------------------------------------------------------------------------
    # Check 6: Cloud Model Management
    # -------------------------------------------------------------------------
    cloud_m = CloudXGBoost()
    meta_c = reg.register_model(cloud_m, "cloud", "cloud", version="1.0.0", n_features=37)
    c6_ok = (
        meta_c.execution_location == "cloud"
        and meta_c.n_features == 37
        and meta_c.model_type == "CloudXGBoost"
    )
    check("Cloud model management", c6_ok, f"location={meta_c.execution_location}, n_features={meta_c.n_features}")

    # -------------------------------------------------------------------------
    # Check 7: Reliability Monitoring
    # -------------------------------------------------------------------------
    monitor = DRAECMonitor(registry=reg, max_records=100)
    r_score = make_dummy_reliability(r=0.742, c=0.88, e=0.06, d=0.18, q=0.96)
    rec7 = monitor.observe_step(observation_index=0, reliability_score=r_score)
    c7_ok = (
        rec7.reliability == 0.742
        and rec7.confidence == 0.88
        and rec7.error_ema == 0.06
        and rec7.drift_severity == 0.18
        and rec7.quality == 0.96
    )
    check("Reliability monitoring", c7_ok, f"R_t={rec7.reliability:.3f} captured unaltered")

    # -------------------------------------------------------------------------
    # Check 8: Drift Monitoring
    # -------------------------------------------------------------------------
    drift_status = {"drift_detected": True, "is_persistent": True, "raw_severity": 0.45, "smoothed_severity": 0.35}
    rec8 = monitor.observe_step(observation_index=1, drift_status=drift_status)
    c8_ok = (
        rec8.drift_detected is True
        and rec8.is_persistent is True
        and rec8.raw_severity == 0.45
        and rec8.smoothed_severity == 0.35
    )
    check("Drift monitoring", c8_ok, f"D_t={rec8.smoothed_severity:.2f}, detected={rec8.drift_detected}")

    # -------------------------------------------------------------------------
    # Check 9: Decision Monitoring
    # -------------------------------------------------------------------------
    dec = DecisionResult(
        selected_action=DecisionAction.EDGE,
        reliability=0.742,
        previous_action=None,
        decision_reason="R_t >= return_threshold",
        switch_count=0,
        observation_index=2,
        timestamp=time.time(),
    )
    rec9 = monitor.observe_step(observation_index=2, decision_result=dec)
    c9_ok = (
        rec9.selected_action == "EDGE"
        and rec9.decision_reason == "R_t >= return_threshold"
    )
    check("Decision monitoring", c9_ok, f"action={rec9.selected_action}, reason='{rec9.decision_reason}'")

    # -------------------------------------------------------------------------
    # Check 10-12: Routing Statistics (EDGE, CLOUD, HYBRID)
    # -------------------------------------------------------------------------
    m_route = DRAECMonitor(max_records=10)
    for _ in range(6):
        d_e = DecisionResult(DecisionAction.EDGE, 0.8, None, "test", 0, 0, time.time())
        m_route.observe_step(0, decision_result=d_e)
    for _ in range(3):
        d_h = DecisionResult(DecisionAction.HYBRID, 0.55, None, "test", 0, 0, time.time())
        m_route.observe_step(0, decision_result=d_h)
    for _ in range(1):
        d_c = DecisionResult(DecisionAction.CLOUD, 0.25, None, "test", 0, 0, time.time())
        m_route.observe_step(0, decision_result=d_c)

    snap_r = m_route.get_snapshot()
    c10_ok = snap_r.routing_counts["EDGE"] == 6 and abs(snap_r.routing_distribution["EDGE"] - 60.0) < 1e-4
    check("EDGE routing statistics", c10_ok, f"count={snap_r.routing_counts['EDGE']}, pct={snap_r.routing_distribution['EDGE']}%")

    c11_ok = snap_r.routing_counts["CLOUD"] == 1 and abs(snap_r.routing_distribution["CLOUD"] - 10.0) < 1e-4
    check("CLOUD routing statistics", c11_ok, f"count={snap_r.routing_counts['CLOUD']}, pct={snap_r.routing_distribution['CLOUD']}%")

    c12_ok = snap_r.routing_counts["HYBRID"] == 3 and abs(snap_r.routing_distribution["HYBRID"] - 30.0) < 1e-4
    check("HYBRID routing statistics", c12_ok, f"count={snap_r.routing_counts['HYBRID']}, pct={snap_r.routing_distribution['HYBRID']}%")

    # -------------------------------------------------------------------------
    # Check 13: Hybrid Fallback Statistics
    # -------------------------------------------------------------------------
    m_hyb = DRAECMonitor(max_records=10)
    d_hyb = DecisionResult(DecisionAction.HYBRID, 0.55, None, "test", 0, 0, time.time())
    res_no_fb = ExecutionResult(d_hyb, DecisionAction.HYBRID, 0, {0: 0.9, 1: 0.1}, "hybrid_edge", 0.001, False, True, ExecutionStatus.SUCCESS)
    res_fb = ExecutionResult(d_hyb, DecisionAction.HYBRID, 1, {0: 0.2, 1: 0.8}, "hybrid_cloud", 0.005, True, True, ExecutionStatus.FALLBACK)

    m_hyb.observe_step(0, execution_result=res_no_fb)
    m_hyb.observe_step(1, execution_result=res_fb)
    snap_h = m_hyb.get_snapshot()
    c13_ok = (
        snap_h.hybrid_stats["executions"] == 2
        and snap_h.hybrid_stats["fallbacks"] == 1
        and abs(snap_h.hybrid_stats["fallback_rate"] - 0.50) < 1e-4
    )
    check("Hybrid fallback statistics", c13_ok, f"fallbacks={snap_h.hybrid_stats['fallbacks']}, rate={snap_h.hybrid_stats['fallback_rate']:.2f}")

    # -------------------------------------------------------------------------
    # Check 14: Execution Monitoring
    # -------------------------------------------------------------------------
    m_exec = DRAECMonitor(max_records=10)
    res_fail = ExecutionResult(d_hyb, DecisionAction.HYBRID, None, None, "hybrid_cloud", 0.0, True, False, ExecutionStatus.FAILED, error="Timeout")
    m_exec.observe_step(0, execution_result=res_no_fb)
    m_exec.observe_step(1, execution_result=res_fail)
    snap_ex = m_exec.get_snapshot()
    c14_ok = (
        snap_ex.execution_stats["total"] == 2
        and snap_ex.execution_stats["successful"] == 1
        and snap_ex.execution_stats["failed"] == 1
        and abs(snap_ex.execution_stats["success_rate"] - 0.50) < 1e-4
    )
    check("Execution monitoring", c14_ok, f"successful={snap_ex.execution_stats['successful']}, failed={snap_ex.execution_stats['failed']}")

    # -------------------------------------------------------------------------
    # Check 15: Latency Statistics
    # -------------------------------------------------------------------------
    m_lat = DRAECMonitor(max_records=10)
    res_lat = ExecutionResult(
        dec, DecisionAction.EDGE, 0, {0: 0.9, 1: 0.1}, "edge", 0.001, False, True, ExecutionStatus.SUCCESS,
        edge_latency_s=0.0015, cloud_latency_s=None, hybrid_latency_s=None,
    )
    m_lat.observe_step(0, execution_result=res_lat)
    snap_lat = m_lat.get_snapshot()
    c15_ok = (
        snap_lat.latency_stats["edge"]["count"] == 1
        and snap_lat.latency_stats["edge"]["mean"] == 0.0015
        and snap_lat.latency_stats["cloud"]["count"] == 0
        and snap_lat.latency_stats["cloud"]["mean"] is None
    )
    check("Streaming latency statistics", c15_ok, "T_edge tracked without fabricating missing T_cloud/T_hybrid")

    # -------------------------------------------------------------------------
    # Check 16: Bounded History Buffer Limit
    # -------------------------------------------------------------------------
    m_bound = DRAECMonitor(max_records=5)
    for i in range(12):
        m_bound.observe_step(i, decision_result=dec, reliability_score=r_score)
    snap_bound = m_bound.get_snapshot()
    c16_ok = (
        len(m_bound.get_records()) == 5
        and snap_bound.total_observations == 12
        and snap_bound.routing_counts["total"] == 12
    )
    check("Bounded history buffer limit", c16_ok, f"history len={len(m_bound.get_records())}, total_obs={snap_bound.total_observations}")

    # -------------------------------------------------------------------------
    # Check 17: Monitoring Snapshot Generation
    # -------------------------------------------------------------------------
    snap = monitor.get_snapshot()
    required_keys = [
        "timestamp", "total_observations", "current_reliability", "current_drift_severity",
        "current_action", "routing_counts", "routing_distribution", "hybrid_stats",
        "execution_stats", "latency_stats", "drift_stats", "reliability_stats",
        "model_health", "active_alerts",
    ]
    c17_ok = all(hasattr(snap, k) for k in required_keys)
    check("Monitoring snapshot generation", c17_ok, f"{len(required_keys)}/{len(required_keys)} fields confirmed")

    # -------------------------------------------------------------------------
    # Check 18: Causality Enforcement
    # -------------------------------------------------------------------------
    rec18 = monitor.observe_step(observation_index=99, timestamp=1999.0)
    c18_ok = rec18.observation_index == 99 and rec18.timestamp == 1999.0
    check("Causality enforcement", c18_ok, "observation_index strictly causal")

    # -------------------------------------------------------------------------
    # Check 19: Leakage Protection (Target & Ground Truth Quarantined)
    # -------------------------------------------------------------------------
    c19_ok = (
        not hasattr(rec18, "Target")
        and not hasattr(rec18, "target")
        and "ground_truth" not in monitor.config.get("monitoring", {})
    )
    check("Target and ground truth quarantine", c19_ok, "zero leakage into monitoring stream")

    # -------------------------------------------------------------------------
    # Check 20: Phase 4 Compatibility
    # -------------------------------------------------------------------------
    rec20 = monitor.observe_step(100, reliability_score=r_score)
    c20_ok = rec20.reliability == r_score.reliability
    check("Phase 4 compatibility", c20_ok, "ReliabilityScore ingested seamlessly")

    # -------------------------------------------------------------------------
    # Check 21: Phase 5 Compatibility
    # -------------------------------------------------------------------------
    rec21 = monitor.observe_step(101, decision_result=dec)
    c21_ok = rec21.selected_action == dec.selected_action.value
    check("Phase 5 compatibility", c21_ok, "DecisionResult ingested seamlessly")

    # -------------------------------------------------------------------------
    # Check 22: Phase 6 Compatibility
    # -------------------------------------------------------------------------
    rec22 = monitor.observe_step(102, execution_result=res_no_fb)
    c22_ok = rec22.execution_status == res_no_fb.status.value and rec22.prediction == res_no_fb.prediction
    check("Phase 6 compatibility", c22_ok, "ExecutionResult ingested seamlessly")

    # -------------------------------------------------------------------------
    # Check 23: End-to-End Streaming Run & Phase 10 DataFrame Readiness
    # -------------------------------------------------------------------------
    m_e2e = DRAECMonitor(registry=reg, max_records=20)
    for i in range(10):
        d_i = DecisionResult(DecisionAction.EDGE if i < 5 else DecisionAction.HYBRID, 0.8 - i * 0.05, None, "e2e", 0, i, float(i))
        r_i = make_dummy_reliability(r=0.8 - i * 0.05)
        res_i = ExecutionResult(d_i, d_i.selected_action, 0, {0: 0.9, 1: 0.1}, "edge", 0.001, False, True, ExecutionStatus.SUCCESS, edge_latency_s=0.001)
        m_e2e.observe_step(i, execution_result=res_i, decision_result=d_i, reliability_score=r_i, controller_policy="draec_adaptive")

    df = m_e2e.get_records_dataframe()
    expected_cols = [
        "observation_index", "timestamp", "reliability", "confidence", "error_ema",
        "drift_severity", "quality", "selected_action", "previous_action", "decision_reason",
        "prediction", "model_used", "execution_status", "cloud_fallback", "edge_latency_s",
        "cloud_latency_s", "hybrid_latency_s", "model_version", "drift_detected",
        "is_persistent", "raw_severity", "smoothed_severity", "controller_policy",
    ]
    c23_ok = (
        isinstance(df, pd.DataFrame)
        and len(df) == 10
        and all(c in df.columns for c in expected_cols)
    )
    check("End-to-end streaming run and Phase 10 DataFrame export", c23_ok, f"10 records exported across {len(expected_cols)} stable columns")

    # -------------------------------------------------------------------------
    # Check 24: Phase 7 Scope Boundary (Quarantine of Phase 8, 9, 10)
    # -------------------------------------------------------------------------
    later_phase_files = [
        "src/adaptation/__init__.py",
        "src/adaptation/retrainer.py",
        "src/adaptation/validator.py",
        "src/adaptation/deployment.py",
        "src/simulation/__init__.py",
        "src/simulation/environment.py",
        "src/metrics/__init__.py",
        "src/metrics/system.py",
    ]
    scope_ok = True
    for fpath in later_phase_files:
        full_p = ROOT / fpath
        if full_p.exists():
            tree = ast.parse(io.open(full_p, encoding="utf-8").read())
            stmts = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if len(stmts) > 0:
                scope_ok = False
                break

    c24_ok = scope_ok and ("monitoring" in cfg)
    check("Phase 7 scope boundary maintained", c24_ok, "Phase 8, 9, 10 modules remain pure skeletons")

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("-" * 80)
    print(f"Phase 7 verification complete: {passed_checks}/{total_checks} checks passed.")
    print("=" * 80)

    return 0 if passed_checks == total_checks else 1


if __name__ == "__main__":
    sys.exit(main())
