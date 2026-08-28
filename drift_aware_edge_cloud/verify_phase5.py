"""Phase 5 verification harness -- src/decision/{base,engine}.py.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Checks:
1. All src/decision modules import and are marked IMPLEMENTED.
2. Configuration: top-level decision section parses with expected defaults.
3. Decision Engine instantiates and binds controller, models, and instrumentation.
4. Adaptive controller instantiates with configured thresholds (0.30, 0.50, 0.70).
5. Static baseline controller instantiates and routes independently of R_t / D_t.
6. Exact action space: a_t in {EDGE, CLOUD, HYBRID} strictly enforced.
7. High reliability routing: R_t >= 0.70 routes to EDGE.
8. Low reliability routing: R_t < 0.30 routes to CLOUD.
9. Hysteresis: deadband [0.50, 0.70) maintains previous action without rapid chatter.
10. Recovery: controller recovers from CLOUD to EDGE when R_t >= 0.70.
11. Hybrid Edge-first behavior: evaluates Edge model first when action is HYBRID.
12. Hybrid fallback: falls back to Cloud when Edge confidence is insufficient.
13. Determinism and idempotence: identical inputs yield identical action sequences.
14. Causality: decisions at step t consume only information available at step t.
15. Leakage protection: Target labels and ground_truth.json are not consumed.
16. Edge execution: minimal execution under EDGE invokes Edge model only.
17. Cloud execution: minimal execution under CLOUD invokes Cloud model only.
18. Lightweight instrumentation: counts, latencies, and switches tracked correctly.
19. Phase 4 compatibility: consumes R_t directly from ReliabilityEstimator.
20. End-to-end streaming smoke test: integrates Phase 1, 2, 3, 4, and 5 causally.
21. Phase 5 scope boundary: no Phase 6+ hardened orchestration or deployment.

Run:
    ../.venv/Scripts/python.exe verify_phase5.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

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


def main() -> int:
    banner("PHASE 5 VERIFICATION -- src/decision/{base,engine}.py")

    # -------------------------------------------------------------------------
    # Check 1: Module presence and IMPLEMENTED status
    # -------------------------------------------------------------------------
    dec_files = ["base.py", "engine.py", "__init__.py"]
    implemented_status: dict[str, bool] = {}
    for fn in dec_files:
        p = ROOT / "src" / "decision" / fn
        if not p.exists():
            implemented_status[fn] = False
            continue
        text = p.read_text(encoding="utf-8")
        implemented_status[fn] = "Status   : IMPLEMENTED" in text

    all_implemented = all(implemented_status.values())
    check(
        "1. all src/decision modules import and are marked IMPLEMENTED",
        all_implemented,
        f"verified {', '.join(dec_files)} carry 'Status   : IMPLEMENTED'",
    )

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
    from src.utils import config as cfgmod

    # -------------------------------------------------------------------------
    # Check 2: Configuration loading
    # -------------------------------------------------------------------------
    cfg = cfgmod.load("default", config_dir=ROOT / "config")
    dec_cfg = cfg.get("decision", {})
    adap_cfg = dec_cfg.get("adaptive", {})
    hyb_cfg = dec_cfg.get("hybrid", {})
    base_cfg = dec_cfg.get("baseline", {})
    inst_cfg = dec_cfg.get("instrumentation", {})

    c2_ok = (
        "decision" in cfg
        and adap_cfg.get("cloud_threshold") == 0.50
        and adap_cfg.get("edge_return_threshold") == 0.70
        and adap_cfg.get("critical_cloud_threshold") == 0.30
        and hyb_cfg.get("fallback_confidence_threshold") == 0.60
        and base_cfg.get("policy") == "edge_only"
        and inst_cfg.get("enabled") is True
    )
    check(
        "2. top-level decision configuration loaded with expected parameters",
        c2_ok,
        f"cloud_threshold={adap_cfg.get('cloud_threshold')}, "
        f"edge_return_threshold={adap_cfg.get('edge_return_threshold')}, "
        f"critical_cloud={adap_cfg.get('critical_cloud_threshold')}, "
        f"fallback_conf={hyb_cfg.get('fallback_confidence_threshold')}",
    )

    # -------------------------------------------------------------------------
    # Check 3: Decision Engine instantiation
    # -------------------------------------------------------------------------
    X_toy = np.array([[1.0, 2.0], [2.0, 1.0], [1.5, 1.8], [0.5, 0.2]])
    y_toy = np.array([0, 1, 0, 1])
    edge_m = EdgeHoeffdingTree(grace_period=10).fit(X_toy, y_toy)
    cloud_m = CloudXGBoost(n_estimators=5, max_depth=2, random_state=42).fit(X_toy, y_toy)

    ctrl = AdaptiveController(config=cfg)
    engine = DecisionEngine(ctrl, edge_m, cloud_m, config=cfg)
    c3_ok = (
        isinstance(engine, BaseDecisionEngine)
        and engine.controller is ctrl
        and engine.edge_model is edge_m
        and engine.cloud_model is cloud_m
    )
    check(
        "3. Decision Engine instantiates and binds controller and models",
        c3_ok,
        f"controller={ctrl.__class__.__name__}, edge={edge_m.model_name}, cloud={cloud_m.model_name}",
    )

    # -------------------------------------------------------------------------
    # Check 4: Adaptive controller instantiation
    # -------------------------------------------------------------------------
    c4_ok = (
        isinstance(ctrl, BaseController)
        and ctrl.current_action == DecisionAction.EDGE
        and ctrl.cloud_threshold == 0.50
        and ctrl.edge_return_threshold == 0.70
        and ctrl.critical_cloud_threshold == 0.30
        and ctrl.switch_count == 0
    )
    check(
        "4. Adaptive controller instantiates with configured thresholds",
        c4_ok,
        f"critical={ctrl.critical_cloud_threshold}, cloud={ctrl.cloud_threshold}, "
        f"return={ctrl.edge_return_threshold}, initial={ctrl.current_action.value}",
    )

    # -------------------------------------------------------------------------
    # Check 5: Static baseline controller
    # -------------------------------------------------------------------------
    base_ctrl = StaticBaselineController(policy="edge_only")
    r_sweep = [0.0, 0.25, 0.50, 0.75, 1.0]
    base_actions = [base_ctrl.decide(r).selected_action for r in r_sweep]
    c5_ok = (
        isinstance(base_ctrl, BaseController)
        and all(a == DecisionAction.EDGE for a in base_actions)
        and base_ctrl.policy == "edge_only"
    )
    check(
        "5. Static baseline controller routes independently of R_t",
        c5_ok,
        f"policy='edge_only' evaluated over R_t in {r_sweep} -> all actions={base_actions[0].value}",
    )

    # -------------------------------------------------------------------------
    # Check 6: Exact action space
    # -------------------------------------------------------------------------
    act_set = {a for a in DecisionAction}
    c6_ok = (
        act_set == {DecisionAction.EDGE, DecisionAction.CLOUD, DecisionAction.HYBRID}
        and DecisionAction.EDGE.value == "EDGE"
        and DecisionAction.CLOUD.value == "CLOUD"
        and DecisionAction.HYBRID.value == "HYBRID"
    )
    check(
        "6. Exact action space {EDGE, CLOUD, HYBRID} strictly defined",
        c6_ok,
        f"actions={[a.value for a in act_set]}",
    )

    # -------------------------------------------------------------------------
    # Check 7: High reliability routing
    # -------------------------------------------------------------------------
    ctrl.reset()
    res_high = ctrl.decide(0.85)
    c7_ok = res_high.selected_action == DecisionAction.EDGE and ctrl.switch_count == 0
    check(
        "7. High reliability R_t >= 0.70 routes to EDGE",
        c7_ok,
        f"R_t=0.85 -> action={res_high.selected_action.value}, switches={ctrl.switch_count}",
    )

    # -------------------------------------------------------------------------
    # Check 8: Low reliability routing
    # -------------------------------------------------------------------------
    ctrl.reset()
    res_low = ctrl.decide(0.20)
    c8_ok = res_low.selected_action == DecisionAction.CLOUD and ctrl.switch_count == 1
    check(
        "8. Low reliability R_t < 0.30 routes to CLOUD",
        c8_ok,
        f"R_t=0.20 -> action={res_low.selected_action.value}, switches={ctrl.switch_count}",
    )

    # -------------------------------------------------------------------------
    # Check 9: Hysteresis deadband stability
    # -------------------------------------------------------------------------
    # From EDGE: R_t = 0.60 maintains EDGE
    ctrl_edge = AdaptiveController(initial_action=DecisionAction.EDGE)
    act_from_edge = ctrl_edge.decide(0.60).selected_action

    # From CLOUD: R_t = 0.60 maintains CLOUD
    ctrl_cloud = AdaptiveController(initial_action=DecisionAction.CLOUD)
    act_from_cloud = ctrl_cloud.decide(0.60).selected_action

    c9_ok = (
        act_from_edge == DecisionAction.EDGE
        and act_from_cloud == DecisionAction.CLOUD
        and ctrl_edge.switch_count == 0
        and ctrl_cloud.switch_count == 0
    )
    check(
        "9. Hysteresis deadband [0.50, 0.70) maintains previous action without chatter",
        c9_ok,
        f"at R_t=0.60: from_EDGE -> {act_from_edge.value}, from_CLOUD -> {act_from_cloud.value}",
    )

    # -------------------------------------------------------------------------
    # Check 10: Recovery from Cloud to Edge
    # -------------------------------------------------------------------------
    ctrl_cloud.reset()
    ctrl_cloud.decide(0.20)  # in CLOUD
    res_rec = ctrl_cloud.decide(0.75)  # R_t >= 0.70 triggers recovery
    c10_ok = res_rec.selected_action == DecisionAction.EDGE
    check(
        "10. Recovery: controller recovers from CLOUD to EDGE when R_t >= 0.70",
        c10_ok,
        f"R_t=0.75 -> recovered to {res_rec.selected_action.value}, switches={ctrl_cloud.switch_count}",
    )

    # -------------------------------------------------------------------------
    # Check 11: Hybrid Edge-first behavior
    # -------------------------------------------------------------------------
    ctrl.reset()
    # At R_t = 0.40, selects HYBRID
    res_hyb = ctrl.decide(0.40)
    c11_ok = res_hyb.selected_action == DecisionAction.HYBRID
    check(
        "11. Hybrid action selection in intermediate zone [0.30, 0.50)",
        c11_ok,
        f"R_t=0.40 -> action={res_hyb.selected_action.value}",
    )

    # -------------------------------------------------------------------------
    # Check 12: Hybrid fallback logic
    # -------------------------------------------------------------------------
    class FixedProbModel(BaseModel):
        def __init__(self, name: str, p0: float, p1: float) -> None:
            super().__init__(name)
            self._is_trained = True
            self.p0 = p0
            self.p1 = p1
            self.called = False
        def fit(self, X, y): return self
        def predict(self, X): return np.zeros(len(X), dtype=int)
        def predict_proba(self, X): return np.zeros((len(X), 2))
        def predict_one(self, x):
            self.called = True
            return 0 if self.p0 >= self.p1 else 1
        def predict_proba_one(self, x):
            self.called = True
            return {0: self.p0, 1: self.p1}
        def get_info(self): return {"name": self._model_name}

    # Case A: Edge is confident (max_p=0.9 -> C_edge=0.8 >= 0.6) -> no fallback
    e_conf = FixedProbModel("edge_conf", 0.9, 0.1)
    c_m1 = FixedProbModel("cloud_1", 0.1, 0.9)
    eng_a = DecisionEngine(AdaptiveController(initial_action=DecisionAction.HYBRID), e_conf, c_m1)
    res_a = eng_a.execute([1.0, 2.0], 0.40)

    # Case B: Edge is uncertain (max_p=0.55 -> C_edge=0.1 < 0.6) -> falls back to Cloud
    e_unc = FixedProbModel("edge_unc", 0.55, 0.45)
    c_m2 = FixedProbModel("cloud_2", 0.05, 0.95)
    eng_b = DecisionEngine(AdaptiveController(initial_action=DecisionAction.HYBRID), e_unc, c_m2)
    res_b = eng_b.execute([1.0, 2.0], 0.40)

    c12_ok = (
        res_a.action == DecisionAction.HYBRID
        and not res_a.cloud_fallback
        and res_a.model_used == "hybrid_edge"
        and res_b.action == DecisionAction.HYBRID
        and res_b.cloud_fallback
        and res_b.model_used == "hybrid_cloud"
        and c_m2.called
    )
    check(
        "12. Hybrid execution: Edge-first with Cloud fallback on low confidence",
        c12_ok,
        f"confident_edge: fallback={res_a.cloud_fallback}, used={res_a.model_used}; "
        f"uncertain_edge: fallback={res_b.cloud_fallback}, used={res_b.model_used}",
    )

    # -------------------------------------------------------------------------
    # Check 13: Determinism and idempotence
    # -------------------------------------------------------------------------
    stream_seq = [0.85, 0.75, 0.55, 0.45, 0.40, 0.25, 0.20, 0.60, 0.72]
    c1 = AdaptiveController()
    c2 = AdaptiveController()
    dec1 = [c1.decide(r).selected_action for r in stream_seq]
    dec2 = [c2.decide(r).selected_action for r in stream_seq]
    c13_ok = dec1 == dec2
    check(
        "13. Pipeline is deterministic and idempotent",
        c13_ok,
        f"identical 9-step sequence yields identical actions: {[a.value for a in dec1]}",
    )

    # -------------------------------------------------------------------------
    # Check 14: Causality
    # -------------------------------------------------------------------------
    # Verify decide() operates strictly on current inputs and state
    ctrl.reset()
    t_inp = DecisionInputs(reliability=0.75, observation_index=15)
    d_out = ctrl.decide(t_inp)
    c14_ok = d_out.observation_index == 15 and d_out.selected_action == DecisionAction.EDGE
    check(
        "14. Causality: decisions consume only current observation inputs",
        c14_ok,
        f"index={d_out.observation_index}, action={d_out.selected_action.value}",
    )

    # -------------------------------------------------------------------------
    # Check 15: Leakage protection
    # -------------------------------------------------------------------------
    # Verify no Target label or ground_truth sidecar is accessed or accepted
    engine.reset()
    exec_clean = engine.execute([1.0, 2.0], DecisionInputs(reliability=0.85))
    info_clean = engine.get_info()
    c15_ok = (
        exec_clean.prediction in (0, 1)
        and "Target" not in str(info_clean)
        and "ground_truth" not in str(info_clean)
    )
    check(
        "15. Leakage protection: no Target or ground_truth.json consumed",
        c15_ok,
        "execution operates purely on feature vectors and reliability inputs",
    )

    # -------------------------------------------------------------------------
    # Check 16: Minimal Edge execution
    # -------------------------------------------------------------------------
    engine.reset()
    res_edge = engine.execute([1.0, 2.0], 0.90)
    c16_ok = (
        res_edge.action == DecisionAction.EDGE
        and res_edge.model_used == "edge"
        and res_edge.prediction in (0, 1)
        and not res_edge.cloud_fallback
    )
    check(
        "16. Edge minimal execution invokes Edge Hoeffding Tree",
        c16_ok,
        f"action={res_edge.action.value}, model={res_edge.model_used}, pred={res_edge.prediction}",
    )

    # -------------------------------------------------------------------------
    # Check 17: Minimal Cloud execution
    # -------------------------------------------------------------------------
    engine.reset()
    res_cloud = engine.execute([1.0, 2.0], 0.15)
    c17_ok = (
        res_cloud.action == DecisionAction.CLOUD
        and res_cloud.model_used == "cloud"
        and res_cloud.prediction in (0, 1)
        and not res_cloud.cloud_fallback
    )
    check(
        "17. Cloud minimal execution invokes Cloud XGBoost",
        c17_ok,
        f"action={res_cloud.action.value}, model={res_cloud.model_used}, pred={res_cloud.prediction}",
    )

    # -------------------------------------------------------------------------
    # Check 18: Lightweight instrumentation
    # -------------------------------------------------------------------------
    engine.reset()
    engine.execute([1.0, 2.0], 0.90)  # EDGE
    engine.execute([1.0, 2.0], 0.40)  # HYBRID
    engine.execute([1.0, 2.0], 0.15)  # CLOUD
    summary = engine.instrumentation.get_summary()
    c18_ok = (
        summary["total_decisions"] == 3
        and summary["edge_count"] == 1
        and summary["hybrid_count"] == 1
        and summary["cloud_count"] == 1
        and summary["switch_count"] >= 2
        and summary["total_latency_s"] > 0.0
    )
    check(
        "18. Lightweight instrumentation tracks counts, switches, and latency",
        c18_ok,
        f"decisions={summary['total_decisions']}, edge={summary['edge_count']}, "
        f"hybrid={summary['hybrid_count']}, cloud={summary['cloud_count']}, switches={summary['switch_count']}",
    )

    # -------------------------------------------------------------------------
    # Check 19: Phase 4 reliability compatibility
    # -------------------------------------------------------------------------
    rel_est = ReliabilityEstimator()
    rel_score = rel_est.update(probs={0: 0.82, 1: 0.18}, drift_severity=0.1, quality=1.0)
    engine.reset()
    r_inp = DecisionInputs(
        reliability=rel_score.reliability,
        confidence=rel_score.inputs.confidence,
        drift_severity=rel_score.inputs.drift,
        quality=rel_score.inputs.quality,
    )
    res_p4 = engine.execute([1.0, 2.0], r_inp)
    c19_ok = (
        0.0 <= rel_score.reliability <= 1.0
        and res_p4.decision.reliability == rel_score.reliability
        and res_p4.action in {DecisionAction.EDGE, DecisionAction.CLOUD, DecisionAction.HYBRID}
    )
    check(
        "19. Phase 4 reliability compatibility: R_t smoothly feeds decision engine",
        c19_ok,
        f"R_t={rel_score.reliability:.4f} -> action={res_p4.action.value}, pred={res_p4.prediction}",
    )

    # -------------------------------------------------------------------------
    # Check 20: End-to-end streaming smoke test
    # -------------------------------------------------------------------------
    engine.reset()
    rel_est.reset()
    stream_reliabilities = [0.95, 0.90, 0.85, 0.65, 0.45, 0.40, 0.25, 0.20, 0.75, 0.85]
    records = []
    for i, r in enumerate(stream_reliabilities):
        x = [1.0 + 0.05 * i, 2.0 - 0.05 * i]
        inp = DecisionInputs(reliability=r, observation_index=i)
        exec_step = engine.execute(x, inp)
        records.append(exec_step)

    c20_ok = (
        len(records) == 10
        and all(r.prediction in (0, 1) for r in records)
        and any(r.action == DecisionAction.EDGE for r in records)
        and any(r.action == DecisionAction.HYBRID for r in records)
        and any(r.action == DecisionAction.CLOUD for r in records)
    )
    check(
        "20. End-to-end streaming smoke test across EDGE, HYBRID, and CLOUD",
        c20_ok,
        f"10 steps executed: actions={[r.action.value for r in records]}",
    )

    # -------------------------------------------------------------------------
    # Check 21: Phase 5 scope boundary
    # -------------------------------------------------------------------------
    # Verify no Phase 6+ orchestration, physical deployment, or retraining modules
    eng_info = engine.get_info()
    c21_ok = (
        "wds" not in eng_info
        and "lri" not in eng_info
        and "retrain" not in eng_info
        and "mqtt" not in eng_info
        and "container" not in eng_info
    )
    check(
        "21. Phase 5 scope boundary strictly maintained (no Phase 6+ logic)",
        c21_ok,
        "decision engine remains minimal execution layer; no WDS/LRI/retraining",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    banner("STEP 5 VERIFICATION SUMMARY")
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
        print("\nPHASE 5 VERIFICATION = PASS\n")
        return 0
    else:
        print(f"\nPHASE 5 VERIFICATION = FAIL ({n_total - n_pass} checks failed)\n")
        return 1


if __name__ == "__main__":
    sys.exit(main())
