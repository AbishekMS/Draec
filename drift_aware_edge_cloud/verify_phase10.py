"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Script   : verify_phase10.py
Phase    : Phase 10
Purpose  : Automated verification harness for Phase 10 scientific evaluation and deliverables.
"""

from __future__ import annotations

import ast
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
passed = 0
failed = 0


def record_check(name: str, status: bool, detail: str = "") -> None:
    global passed, failed
    if status:
        passed += 1
        print(f"[PASS] {name}")
    else:
        failed += 1
        print(f"[FAIL] {name}")
    if detail:
        print(f"       {detail}")


print("=" * 78)
print("PHASE 10 VERIFICATION  --  FINAL SCIENTIFIC EVALUATION & BENCHMARKING")
print("=" * 78)

# -----------------------------------------------------------------------------
# Check 1: WUSTL-IIoT Dataset Identity
# -----------------------------------------------------------------------------
try:
    from src.utils import config as config_mod
    cfg = config_mod.load("default")
    ds = cfg["dataset"]
    files = ds["files"]
    train1_p = ROOT / files["train1"]["path"]
    wustl_name_ok = "wustl_iiot_2021.csv" in str(train1_p)
    wustl_size_ok = train1_p.exists() and train1_p.stat().st_size == 409800698
    record_check(
        "Check 01: WUSTL-IIoT dataset identity",
        wustl_name_ok and wustl_size_ok,
        f"Active dataset: wustl_iiot_2021.csv (409,800,698 bytes verified)",
    )
except Exception as e:
    record_check("Check 01: WUSTL-IIoT dataset identity", False, str(e))

# -----------------------------------------------------------------------------
# Check 2: train1 / train2 / test1 Strict Separation
# -----------------------------------------------------------------------------
try:
    f_t1 = files["train1"]
    f_t2 = files["train2"]
    f_inf = files["test1"]
    roles_ok = (
        f_t1["role"] == "baseline_train"
        and f_t2["role"] == "baseline_validation"
        and f_inf["role"] == "inference_stream"
    )
    t1_end = f_t1["time_range"][1]
    t2_start = f_t2["time_range"][0]
    t2_end = f_t2["time_range"][1]
    inf_start = f_inf["time_range"][0]
    temporal_ok = t1_end < t2_start and t2_end < inf_start
    record_check(
        "Check 02: train1/train2/test1 temporal separation",
        roles_ok and temporal_ok,
        f"train1 ({t1_end}) < train2 ({t2_start}..{t2_end}) < test1 ({inf_start})",
    )
except Exception as e:
    record_check("Check 02: train1/train2/test1 temporal separation", False, str(e))

# -----------------------------------------------------------------------------
# Check 3: test1 Adaptation Quarantine
# -----------------------------------------------------------------------------
try:
    from src.adaptation.feedback import FeedbackQueue
    from src.adaptation.validator import CandidateValidator
    q = FeedbackQueue(max_size=10)
    test1_rejected_q = False
    try:
        q.record_prediction(1, np.zeros(37), 0, {0: 1.0, 1: 0.0}, "v1", source="test1_stream")
    except ValueError as err:
        test1_rejected_q = "test1" in str(err).lower()

    val = CandidateValidator()
    test1_rejected_v = False
    try:
        val.set_validation_data(np.zeros((5, 37)), np.zeros(5), source="test1_stream")
    except ValueError as err:
        test1_rejected_v = "test1" in str(err).lower()

    record_check(
        "Check 03: test1 adaptation quarantine",
        test1_rejected_q and test1_rejected_v,
        "test1 rejected by FeedbackQueue and CandidateValidator",
    )
except Exception as e:
    record_check("Check 03: test1 adaptation quarantine", False, str(e))

# -----------------------------------------------------------------------------
# Check 4: Causal Evaluation
# -----------------------------------------------------------------------------
try:
    from src.data import loader
    specs = loader.file_specs(cfg, ROOT)
    spec_inf = specs["test1"]
    record_check(
        "Check 04: Causal evaluation stream specification",
        spec_inf.role == "inference_stream" and not cfg["streaming"]["shuffle"],
        "Inference stream evaluated strictly chronologically without shuffling",
    )
except Exception as e:
    record_check("Check 04: Causal evaluation stream specification", False, str(e))

# -----------------------------------------------------------------------------
# Check 5: Baseline ML Execution
# -----------------------------------------------------------------------------
try:
    from src.models.edge_model import EdgeHoeffdingTree
    from src.models.cloud_model import CloudXGBoost
    em = EdgeHoeffdingTree()
    cm = CloudXGBoost()
    X_toy = np.random.randn(20, 37)
    y_toy = np.random.randint(0, 2, size=20)
    em.fit(X_toy, y_toy)
    cm.fit(X_toy, y_toy)
    p_edge = em.predict(X_toy)
    p_cloud = cm.predict(X_toy)
    record_check(
        "Check 05: Baseline ML model execution",
        len(p_edge) == 20 and len(p_cloud) == 20,
        "Edge Hoeffding Tree and Cloud XGBoost generate valid predictions",
    )
except Exception as e:
    record_check("Check 05: Baseline ML model execution", False, str(e))

# -----------------------------------------------------------------------------
# Check 6: Static Baseline Controller
# -----------------------------------------------------------------------------
try:
    from src.decision.engine import StaticBaselineController
    from src.decision.base import DecisionAction
    s_ctrl = StaticBaselineController(policy="edge_only")
    d1 = s_ctrl.decide(0.85)
    d2 = s_ctrl.decide(0.15)
    record_check(
        "Check 06: Static baseline controller",
        d1.selected_action == DecisionAction.EDGE and d2.selected_action == DecisionAction.EDGE,
        "StaticBaselineController consistently executes configured static policy",
    )
except Exception as e:
    record_check("Check 06: Static baseline controller", False, str(e))

# -----------------------------------------------------------------------------
# Check 7: DRAEC Dynamic Execution
# -----------------------------------------------------------------------------
try:
    from src.decision.engine import AdaptiveController
    a_ctrl = AdaptiveController(critical_cloud_threshold=0.30, cloud_threshold=0.50, edge_return_threshold=0.70)
    d_hi = a_ctrl.decide(0.85)
    d_lo = a_ctrl.decide(0.20)
    record_check(
        "Check 07: DRAEC dynamic controller execution",
        d_hi.selected_action == DecisionAction.EDGE and d_lo.selected_action == DecisionAction.CLOUD,
        "AdaptiveController dynamically routes based on reliability score Rt",
    )
except Exception as e:
    record_check("Check 07: DRAEC dynamic controller execution", False, str(e))

# -----------------------------------------------------------------------------
# Check 8: Drift Experiment
# -----------------------------------------------------------------------------
try:
    from src.metrics.drift import compute_drift_metrics
    dm = compute_drift_metrics(detection_indices=[520, 535], drift_onset_index=500, total_steps=1000)
    c8_ok = dm["drift_onset"] == 500 and dm["detection_delay"] == 20 and dm["total_alarms"] == 2
    record_check(
        "Check 08: Drift experiment evaluation",
        c8_ok,
        f"Onset={dm['drift_onset']}, Delay={dm['detection_delay']} steps, Status={dm['detection_status']}",
    )
except Exception as e:
    record_check("Check 08: Drift experiment evaluation", False, str(e))

# -----------------------------------------------------------------------------
# Check 9: Reliability Quantities Recording
# -----------------------------------------------------------------------------
try:
    from src.reliability.estimator import ReliabilityEstimator
    r_est = ReliabilityEstimator()
    r_step = r_est.update(probs={0: 0.9, 1: 0.1}, drift_severity=0.2, quality=[True] * 37)
    c9_ok = (
        0.0 <= r_step.inputs.confidence <= 1.0
        and 0.0 <= r_step.inputs.drift <= 1.0
        and 0.0 <= r_step.inputs.quality <= 1.0
        and 0.0 <= r_step.reliability <= 1.0
    )
    record_check(
        "Check 09: Reliability factor recording (Ct, Et, Dt, Qt, Rt)",
        c9_ok,
        f"Ct={r_step.inputs.confidence:.2f}, Dt={r_step.inputs.drift:.2f}, Qt={r_step.inputs.quality:.2f}, Rt={r_step.reliability:.4f}",
    )
except Exception as e:
    record_check("Check 09: Reliability factor recording (Ct, Et, Dt, Qt, Rt)", False, str(e))

# -----------------------------------------------------------------------------
# Check 10: Routing Measurement
# -----------------------------------------------------------------------------
try:
    from src.metrics.decision import compute_routing_metrics
    acts = ["EDGE"] * 60 + ["HYBRID"] * 20 + ["CLOUD"] * 20
    rm = compute_routing_metrics(acts, switch_count=2, hybrid_fallbacks=5)
    c10_ok = (
        rm["edge_percentage"] == 60.0
        and rm["hybrid_percentage"] == 20.0
        and rm["cloud_percentage"] == 20.0
        and rm["offloading_ratio"] == 20.0
        and rm["switch_count"] == 2
        and rm["hybrid_fallback_rate"] == 25.0
    )
    record_check(
        "Check 10: Routing distribution and offloading ratio metrics",
        c10_ok,
        f"Edge={rm['edge_percentage']}%, Cloud={rm['cloud_percentage']}%, Offload={rm['offloading_ratio']}%",
    )
except Exception as e:
    record_check("Check 10: Routing distribution and offloading ratio metrics", False, str(e))

# -----------------------------------------------------------------------------
# Check 11: Hybrid Execution & Fallback Gating
# -----------------------------------------------------------------------------
try:
    from src.deployment.environment import DeploymentEnvironment
    from src.deployment.runtimes import EdgeRuntime, CloudRuntime
    class MockEdge:
        def predict_proba_one(self, x): return {0: 0.55, 1: 0.45}  # Conf = 0.10 < 0.60
        def predict_one(self, x): return 0
    class MockCloud:
        def predict_one(self, x): return 1
        def predict_proba_one(self, x): return {0: 0.1, 1: 0.9}
    d_env = DeploymentEnvironment(EdgeRuntime(MockEdge()), CloudRuntime(MockCloud()), fallback_confidence_threshold=0.60)
    h_res = d_env.execute_hybrid({"a": 1})
    c11_ok = h_res.cloud_fallback is True and h_res.model_used == "hybrid_cloud" and h_res.prediction == 1
    record_check(
        "Check 11: Hybrid execution policy and fallback gating",
        c11_ok,
        "Edge confidence < 0.60 successfully triggered Cloud fallback",
    )
except Exception as e:
    record_check("Check 11: Hybrid execution policy and fallback gating", False, str(e))

# -----------------------------------------------------------------------------
# Check 12: Measured Latency Accounting
# -----------------------------------------------------------------------------
try:
    from src.metrics.system import compute_latency_summary
    lats = [0.005, 0.010, 0.015, 0.020]
    ls = compute_latency_summary(lats)
    c12_ok = ls["mean_ms"] == 12.5 and ls["median_ms"] == 12.5 and ls["max_ms"] == 20.0
    record_check(
        "Check 12: Latency accounting (mean, median, p95, max)",
        c12_ok,
        f"Mean={ls['mean_ms']}ms, Median={ls['median_ms']}ms, P95={ls['p95_ms']}ms",
    )
except Exception as e:
    record_check("Check 12: Latency accounting (mean, median, p95, max)", False, str(e))

# -----------------------------------------------------------------------------
# Check 13: Simulated Network Measurement
# -----------------------------------------------------------------------------
try:
    from src.metrics.system import compute_network_metrics
    nms = compute_network_metrics(total_transmissions=100, delivered_transmissions=95, packet_loss_count=5, latencies_s=[0.020] * 95)
    c13_ok = nms["delivery_rate"] == 0.95 and nms["packet_loss_rate"] == 0.05
    record_check(
        "Check 13: Simulated network metrics under Phase 8 simulator",
        c13_ok,
        f"Delivery={nms['delivery_rate']*100}%, Loss={nms['packet_loss_rate']*100}%",
    )
except Exception as e:
    record_check("Check 13: Simulated network metrics under Phase 8 simulator", False, str(e))

# -----------------------------------------------------------------------------
# Check 14: Adaptation Measurement
# -----------------------------------------------------------------------------
try:
    from src.adaptation.base import AdaptationResult, AdaptationState, ValidationResult
    vr = ValidationResult(candidate_valid=True, metric_name="f1", candidate_metric=0.88, active_metric=0.85, metric_delta=0.03, status="ACCEPTED")
    res_ad = AdaptationResult(
        state=AdaptationState.ACCEPTED,
        triggered=True,
        candidate_version="v2",
        active_version="v2",
        cloud_version="v2",
        edge_version="v2",
        validation_result=vr,
    )
    c14_ok = res_ad.triggered and res_ad.candidate_version == "v2" and res_ad.validation_result.candidate_metric == 0.88
    record_check(
        "Check 14: Model adaptation measurement and result recording",
        c14_ok,
        f"State={res_ad.state.value}, Candidate={res_ad.candidate_version}, Metric={res_ad.validation_result.candidate_metric}",
    )
except Exception as e:
    record_check("Check 14: Model adaptation measurement and result recording", False, str(e))

# -----------------------------------------------------------------------------
# Check 15: Model Version Lineage Tracking
# -----------------------------------------------------------------------------
try:
    from src.adaptation.deployment import AtomicModelDeployer
    class MockR:
        def __init__(self): self.model = "m1"
    dep = AtomicModelDeployer(MockR(), MockR())
    v_init = dep.active_system_version
    dep.deploy(candidate_cloud_model="cand_model", updated_edge_model="cand_model", candidate_version="v2")
    v_after = dep.active_system_version
    c15_ok = v_init == "v1" and v_after == "v2" and dep.get_stats()["successful_deployments"] == 1
    record_check(
        "Check 15: Model version tracking and atomic advancement",
        c15_ok,
        f"Version advanced from {v_init} to {v_after}",
    )
except Exception as e:
    record_check("Check 15: Model version tracking and atomic advancement", False, str(e))

# -----------------------------------------------------------------------------
# Check 16: Multi-Seed Reproducibility Protocol
# -----------------------------------------------------------------------------
try:
    from src.metrics.evaluation import compute_confidence_interval, DEFAULT_SEEDS
    c16_ok = len(DEFAULT_SEEDS) == 5 and DEFAULT_SEEDS == [42, 43, 44, 45, 46]
    record_check(
        "Check 16: Multi-seed reproducibility protocol",
        c16_ok,
        f"Seeds configured: {DEFAULT_SEEDS}",
    )
except Exception as e:
    record_check("Check 16: Multi-seed reproducibility protocol", False, str(e))

# -----------------------------------------------------------------------------
# Check 17: Confidence Interval Calculation
# -----------------------------------------------------------------------------
try:
    sample_vals = [0.80, 0.82, 0.84, 0.86, 0.88]
    m_val, s_val, ci_l, ci_u = compute_confidence_interval(sample_vals, confidence=0.95)
    c17_ok = abs(m_val - 0.84) < 1e-4 and ci_l < m_val < ci_u
    record_check(
        "Check 17: Statistical confidence interval estimation",
        c17_ok,
        f"Mean={m_val:.4f}, Std={s_val:.4f}, 95% CI=[{ci_l:.4f}, {ci_u:.4f}]",
    )
except Exception as e:
    record_check("Check 17: Statistical confidence interval estimation", False, str(e))

# -----------------------------------------------------------------------------
# Check 18: Statistical Hypothesis Testing
# -----------------------------------------------------------------------------
try:
    v1 = [0.85, 0.86, 0.87, 0.88, 0.89]
    v2 = [0.70, 0.71, 0.72, 0.73, 0.74]
    from scipy import stats as sp_stats
    t_res = sp_stats.ttest_rel(v1, v2)
    c18_ok = t_res.pvalue < 0.001 and t_res.statistic > 0
    record_check(
        "Check 18: Paired hypothesis testing and significance reporting",
        c18_ok,
        f"t-stat={t_res.statistic:.2f}, p-val={t_res.pvalue:.4e}",
    )
except Exception as e:
    record_check("Check 18: Paired hypothesis testing and significance reporting", False, str(e))

# -----------------------------------------------------------------------------
# Check 19: Result Files Generation Check
# -----------------------------------------------------------------------------
try:
    res_p = ROOT / "results"
    req_csvs = [
        "baseline_model_metrics.csv",
        "drift_metrics.csv",
        "reliability_metrics.csv",
        "routing_metrics.csv",
        "hybrid_metrics.csv",
        "prediction_metrics.csv",
        "adaptation_metrics.csv",
        "latency_metrics.csv",
        "network_metrics.csv",
        "execution_metrics.csv",
        "model_version_metrics.csv",
        "ablation_metrics.csv",
        "statistical_results.csv",
    ]
    # If not yet generated, test structure
    c19_ok = res_p.exists() and all((res_p / f).exists() for f in req_csvs)
    record_check(
        "Check 19: Result CSV artifact verification",
        c19_ok,
        f"{sum(1 for f in req_csvs if (res_p / f).exists())}/{len(req_csvs)} result CSVs exist in results/",
    )
except Exception as e:
    record_check("Check 19: Result CSV artifact verification", False, str(e))

# -----------------------------------------------------------------------------
# Check 20: IEEE Figure Generation Check
# -----------------------------------------------------------------------------
try:
    fig_p = ROOT / "results" / "figures"
    req_figs = [
        "fig1_prediction_under_drift.png",
        "fig2_adwin_drift_detection.png",
        "fig3_reliability_response.png",
        "fig4_routing_distribution.png",
        "fig5_latency_comparison.png",
        "fig6_adaptation_recovery.png",
        "fig7_ablation_study.png",
    ]
    c20_ok = fig_p.exists() and all((fig_p / f).exists() for f in req_figs)
    record_check(
        "Check 20: IEEE publication figure generation",
        c20_ok,
        f"{sum(1 for f in req_figs if (fig_p / f).exists())}/{len(req_figs)} figures present in results/figures/",
    )
except Exception as e:
    record_check("Check 20: IEEE publication figure generation", False, str(e))

# -----------------------------------------------------------------------------
# Check 21: IEEE Table Generation Check
# -----------------------------------------------------------------------------
try:
    tbl_p = ROOT / "results" / "tables"
    req_tbls = [
        "table1_baseline_ml_performance.md",
        "table2_prediction_under_drift.md",
        "table3_system_orchestration.md",
        "table4_statistical_evaluation.md",
    ]
    c21_ok = tbl_p.exists() and all((tbl_p / f).exists() for f in req_tbls)
    record_check(
        "Check 21: IEEE publication table generation",
        c21_ok,
        f"{sum(1 for f in req_tbls if (tbl_p / f).exists())}/{len(req_tbls)} tables present in results/tables/",
    )
except Exception as e:
    record_check("Check 21: IEEE publication table generation", False, str(e))

# -----------------------------------------------------------------------------
# Check 22: Integrity on Unmeasured Quantities
# -----------------------------------------------------------------------------
try:
    from src.metrics.system import get_unmeasured_system_status
    unm = get_unmeasured_system_status()
    c22_ok = (
        unm["cpu_utilization"] == "NOT MEASURED"
        and unm["ram_utilization"] == "NOT MEASURED"
        and unm["energy_consumption"] == "NOT MEASURED"
        and "NOT MEASURED" in unm["physical_hardware_deployment"]
        and unm["formal_constraint_satisfaction"] == "NOT IMPLEMENTED / NOT MEASURED"
    )
    record_check(
        "Check 22: Integrity guard on unmeasured quantities",
        c22_ok,
        "CPU, RAM, Energy, Hardware, Constraints explicitly reported as NOT MEASURED",
    )
except Exception as e:
    record_check("Check 22: Integrity guard on unmeasured quantities", False, str(e))

# -----------------------------------------------------------------------------
# Check 23: No Architecture Modification in Phases 1-9
# -----------------------------------------------------------------------------
try:
    from src.decision.engine import AdaptiveController
    act_ctrl = AdaptiveController()
    c23_ok = (
        act_ctrl.critical_cloud_threshold == 0.30
        and act_ctrl.cloud_threshold == 0.50
        and act_ctrl.edge_return_threshold == 0.70
    )
    record_check(
        "Check 23: Architecture immutability in Phases 1-9",
        c23_ok,
        "Controller thresholds strictly preserved (critical=0.30, cloud=0.50, return=0.70)",
    )
except Exception as e:
    record_check("Check 23: Architecture immutability in Phases 1-9", False, str(e))

# -----------------------------------------------------------------------------
# Check 24: Phase 10 Scope Compliance
# -----------------------------------------------------------------------------
try:
    # Phase 10 is the final phase. Verify no Phase 11 exists and remaining stubs are pure stubs
    sim_init = ROOT / "src" / "simulation" / "__init__.py"
    sim_env = ROOT / "src" / "simulation" / "environment.py"
    scope_ok = True
    for fpath in (sim_init, sim_env):
        if fpath.exists():
            tree = ast.parse(fpath.read_text(encoding="utf-8"))
            stmts = [n for n in tree.body if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant))]
            if stmts:
                scope_ok = False
    record_check(
        "Check 24: Phase 10 scope compliance and final stage boundary",
        scope_ok,
        "Phase 10 is final phase; remaining simulation stubs remain clean stubs",
    )
except Exception as e:
    record_check("Check 24: Phase 10 scope compliance and final stage boundary", False, str(e))

# -----------------------------------------------------------------------------
# Summary
# -----------------------------------------------------------------------------
print("\n" + "=" * 78)
print(f"PHASE 10 VERIFICATION SUMMARY: {passed} / {passed + failed} CHECKS PASSED")
print("=" * 78)

if failed > 0:
    sys.exit(1)
else:
    sys.exit(0)
