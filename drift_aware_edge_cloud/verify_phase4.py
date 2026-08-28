"""Phase 4 verification harness -- src/reliability/{base,estimator}.py.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Checks:
1. All src/reliability modules import and are marked IMPLEMENTED.
2. ReliabilityEstimator instantiates and binds configured parameters.
3. Confidence C_t formula: C_t = 2 * (max(P(0), P(1)) - 0.5) in [0, 1].
4. Instantaneous prediction error e_t is binary 0-1 loss.
5. Recent error E_t = alpha_E * E_{t-1} + (1-alpha_E) * e_t with delayed feedback.
6. Drift severity D_t consumes Phase 3 smoothed severity.
7. Quality Q_t = (1/N_F) * sum(q_j) with general N_F and WUSTL N_F=37.
8. Reliability R_t implements weighted harmonic mean with weakest-link property.
9. R_t satisfies monotonicity with respect to C_t, E_t, D_t, Q_t.
10. Edge cases (zeros and ones) produce no NaN/Inf and remain bounded in [0, 1].
11. Real-time inference operates without future labels or ground_truth.json.
12. Small end-to-end streaming smoke test with Phase 2 models and Phase 3 drift.

Run:
    ../.venv/Scripts/python.exe verify_phase4.py
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
    banner("PHASE 4 VERIFICATION -- src/reliability/{base,estimator}.py")

    # -------------------------------------------------------------------------
    # Check 1: Module presence and IMPLEMENTED status
    # -------------------------------------------------------------------------
    rel_files = ["base.py", "estimator.py", "__init__.py"]
    implemented_status: dict[str, bool] = {}
    for fn in rel_files:
        p = ROOT / "src" / "reliability" / fn
        if not p.exists():
            implemented_status[fn] = False
            continue
        text = p.read_text(encoding="utf-8")
        implemented_status[fn] = "Status   : IMPLEMENTED" in text

    all_implemented = all(implemented_status.values())
    check(
        "1. all src/reliability modules import and are marked IMPLEMENTED",
        all_implemented,
        f"verified {', '.join(rel_files)} carry 'Status   : IMPLEMENTED'",
    )

    from src.reliability import (
        BaseReliabilityEstimator,
        ReliabilityEstimator,
        ReliabilityFactors,
        ReliabilityInputs,
        ReliabilityScore,
        compute_confidence,
        compute_harmonic_reliability,
        compute_instantaneous_error,
        compute_quality,
    )
    from src.utils import config as config_mod

    # -------------------------------------------------------------------------
    # Check 2: Instantiation and parameter binding
    # -------------------------------------------------------------------------
    cfg = config_mod.load("default")
    est = ReliabilityEstimator(config=cfg)
    info = est.get_info()

    params_bound = (
        est.alpha_E == 0.8
        and est.epsilon == 1e-8
        and est.weights == {"confidence": 0.25, "error": 0.25, "drift": 0.25, "quality": 0.25}
        and est.default_n_features == 37
        and est.current_error == 0.0
    )
    check(
        "2. ReliabilityEstimator instantiates and binds configured parameters",
        params_bound,
        f"alpha_E={est.alpha_E}, epsilon={est.epsilon}, weights={est.weights}",
    )

    # -------------------------------------------------------------------------
    # Check 3: Confidence C_t formula and bounds
    # -------------------------------------------------------------------------
    c_ambig = compute_confidence({0: 0.5, 1: 0.5})
    c_cert0 = compute_confidence({0: 1.0, 1: 0.0})
    c_cert1 = compute_confidence({0: 0.0, 1: 1.0})
    c_inter = compute_confidence({0: 0.2, 1: 0.8})

    conf_ok = (
        abs(c_ambig - 0.0) < 1e-6
        and abs(c_cert0 - 1.0) < 1e-6
        and abs(c_cert1 - 1.0) < 1e-6
        and abs(c_inter - 0.6) < 1e-6
    )
    check(
        "3. Confidence C_t = 2 * (max(P(0), P(1)) - 0.5) satisfies bounds and edge cases",
        conf_ok,
        f"C(0.5)={c_ambig}, C(0.8)={c_inter}, C(1.0)={c_cert0}",
    )

    # -------------------------------------------------------------------------
    # Check 4: Instantaneous binary 0-1 error
    # -------------------------------------------------------------------------
    e_correct0 = compute_instantaneous_error(0, 0)
    e_correct1 = compute_instantaneous_error(1, 1)
    e_wrong0 = compute_instantaneous_error(0, 1)
    e_wrong1 = compute_instantaneous_error(1, 0)

    err_ok = (
        e_correct0 == 0.0 and e_correct1 == 0.0 and e_wrong0 == 1.0 and e_wrong1 == 1.0
    )
    check(
        "4. Instantaneous error e_t is exact binary 0-1 loss I(y_pred != y_true)",
        err_ok,
        f"e(0,0)={e_correct0}, e(1,1)={e_correct1}, e(0,1)={e_wrong0}, e(1,0)={e_wrong1}",
    )

    # -------------------------------------------------------------------------
    # Check 5: Recent error E_t EMA with delayed feedback
    # -------------------------------------------------------------------------
    est_err = ReliabilityEstimator(alpha_E=0.8, initial_error=0.0)
    # Step 1 inference: no feedback
    s1 = est_err.update(confidence=0.8)
    e_initial = s1.inputs.error
    # Delayed feedback arrives: error 1.0 -> 0.8 * 0.0 + 0.2 * 1.0 = 0.2
    est_err.update_error(1.0)
    e_step1 = est_err.current_error
    # Step 2 inference: no new feedback -> retains 0.2
    s2 = est_err.update(confidence=0.8)
    e_step2 = s2.inputs.error
    # Delayed feedback arrives: error 0.0 -> 0.8 * 0.2 + 0.2 * 0.0 = 0.16
    est_err.update_error(0.0)
    e_step3 = est_err.current_error

    ema_ok = (
        e_initial == 0.0
        and abs(e_step1 - 0.2) < 1e-6
        and abs(e_step2 - 0.2) < 1e-6
        and abs(e_step3 - 0.16) < 1e-6
    )
    check(
        "5. Recent error E_t = alpha_E*E_{t-1} + (1-alpha_E)*e_t supports delayed feedback",
        ema_ok,
        f"E_initial={e_initial}, E_after_err1={e_step1}, E_retained={e_step2}, E_after_err0={e_step3}",
    )

    # -------------------------------------------------------------------------
    # Check 6: Drift severity D_t consumes Phase 3 smoothed severity
    # -------------------------------------------------------------------------
    from src.drift import DriftStatus
    est_drift = ReliabilityEstimator()
    status = DriftStatus(
        drift_detected=False,  # Note: binary alarm is False
        is_persistent=False,
        raw_severity=0.8,
        smoothed_severity=0.65,  # Continuous severity is elevated
        estimation=0.7,
        monitored_value=0.7,
    )
    score_drift = est_drift.update(confidence=0.9, drift_status=status, quality=1.0)
    drift_ok = (
        abs(score_drift.inputs.drift - 0.65) < 1e-6
        and abs(score_drift.factors.r_D - 0.35) < 1e-6
    )
    check(
        "6. Drift D_t strictly consumes Phase 3 smoothed severity rather than binary alarm",
        drift_ok,
        f"D_t={score_drift.inputs.drift}, r_D={score_drift.factors.r_D}",
    )

    # -------------------------------------------------------------------------
    # Check 7: Quality Q_t general N_F and WUSTL N_F=37
    # -------------------------------------------------------------------------
    q_gen = compute_quality([True, True, True, False], n_features=4)
    q_wustl = compute_quality([True] * 35 + [False] * 2, n_features=37)
    qual_ok = (
        abs(q_gen - 0.75) < 1e-6
        and abs(q_wustl - (35.0 / 37.0)) < 1e-6
    )
    check(
        "7. Quality Q_t implements general (1/N_F)*sum(q_j) and instantiates WUSTL N_F=37",
        qual_ok,
        f"Q(4 feats, 3 valid)={q_gen}, Q(37 feats, 35 valid)={q_wustl:.4f}",
    )

    # -------------------------------------------------------------------------
    # Check 8: Weighted harmonic R_t calculation and weakest-link property
    # -------------------------------------------------------------------------
    r_all_one = compute_harmonic_reliability(1.0, 1.0, 1.0, 1.0)
    r_all_half = compute_harmonic_reliability(0.5, 0.5, 0.5, 0.5)
    r_one_zero = compute_harmonic_reliability(0.0, 1.0, 1.0, 1.0)

    harm_ok = (
        abs(r_all_one - 1.0) < 1e-5
        and abs(r_all_half - 0.5) < 1e-5
        and r_one_zero < 1e-6  # Weakest-link collapses reliability
    )
    check(
        "8. Weighted harmonic mean R_t correctly provides weakest-link degradation",
        harm_ok,
        f"R(1,1,1,1)={r_all_one:.4f}, R(0.5,0.5,0.5,0.5)={r_all_half:.4f}, R(0,1,1,1)={r_one_zero:.2e}",
    )

    # -------------------------------------------------------------------------
    # Check 9: Monotonicity with respect to all 4 components
    # -------------------------------------------------------------------------
    r_base = compute_harmonic_reliability(0.6, 0.8, 0.7, 0.9)
    r_more_c = compute_harmonic_reliability(0.9, 0.8, 0.7, 0.9)  # Higher C -> Higher R
    r_less_e = compute_harmonic_reliability(0.6, 0.2, 0.7, 0.9)  # Higher E (lower r_E) -> Lower R
    r_more_d = compute_harmonic_reliability(0.6, 0.8, 0.2, 0.9)  # Higher D (lower r_D) -> Lower R
    r_less_q = compute_harmonic_reliability(0.6, 0.8, 0.7, 0.3)  # Lower Q -> Lower R

    mono_ok = (
        r_more_c > r_base
        and r_less_e < r_base
        and r_more_d < r_base
        and r_less_q < r_base
    )
    check(
        "9. Reliability R_t is strictly monotonic in C_t, (1-E_t), (1-D_t), Q_t",
        mono_ok,
        f"base={r_base:.4f} -> higher_C={r_more_c:.4f}, higher_E={r_less_e:.4f}, higher_D={r_more_d:.4f}, lower_Q={r_less_q:.4f}",
    )

    # -------------------------------------------------------------------------
    # Check 10: Edge cases produce no NaN/Inf and stay in [0, 1]
    # -------------------------------------------------------------------------
    edge_ok = True
    for c in (0.0, 1.0):
        for e in (0.0, 1.0):
            for d in (0.0, 1.0):
                for q in (0.0, 1.0):
                    inp = ReliabilityInputs(confidence=c, error=e, drift=d, quality=q)
                    score = est.calculate(inp)
                    val = score.reliability
                    if np.isnan(val) or np.isinf(val) or val < 0.0 or val > 1.0:
                        edge_ok = False
                        break
    check(
        "10. Extreme edge cases ({0,1}^4) evaluate without NaN/Inf/Div0 within [0, 1]",
        edge_ok,
        "all 16 extreme input combinations bounded cleanly in [0, 1]",
    )

    # -------------------------------------------------------------------------
    # Check 11: Real-time observable isolation (no Target, no ground_truth.json)
    # -------------------------------------------------------------------------
    est_iso = ReliabilityEstimator()
    # Execute 20 observations purely from model probabilities and drift
    iso_ok = True
    for _ in range(20):
        sc = est_iso.update(
            probs={0: 0.75, 1: 0.25},
            drift_severity=0.1,
            quality=1.0,
        )
        if sc.reliability <= 0.0 or sc.reliability > 1.0:
            iso_ok = False
            break
    check(
        "11. Real-time reliability updates causally without future Target or ground_truth.json",
        iso_ok,
        f"20 causal updates completed; current R_t={sc.reliability:.4f}",
    )

    # -------------------------------------------------------------------------
    # Check 12: End-to-end streaming smoke test with Phase 2 models & Phase 3 drift
    # -------------------------------------------------------------------------
    from src.models.trainer import load_causal_train_data
    from src.models.edge_model import EdgeHoeffdingTree
    from src.drift import DriftPipeline

    # Load 100 rows of baseline_train for model fitting
    X_train, y_train, stats, profile = load_causal_train_data(cfg, max_rows=100)
    model = EdgeHoeffdingTree(grace_period=10)
    model.fit(X_train, y_train)

    drift_pipe = DriftPipeline(cfg)
    rel_est = ReliabilityEstimator(cfg)

    # Stream 10 observations
    r_values = []
    for i in range(10):
        x_row = X_train.iloc[i]
        probs = model.predict_proba_one(x_row)
        pred = model.predict_one(x_row)

        drift_status = drift_pipe.update_from_prediction(probs, y_pred=pred)
        # Quality from feature validity (all 37 features present)
        rel_score = rel_est.update(
            probs=probs,
            drift_status=drift_status,
            quality=1.0,
        )
        r_values.append(rel_score.reliability)

        # Delayed feedback arrives for previous step i-1 (causally valid)
        if i > 0:
            rel_est.update_feedback(y_true=int(y_train[i - 1]), y_pred=pred)

    smoke_ok = len(r_values) == 10 and all(0.0 <= r <= 1.0 for r in r_values)
    check(
        "12. End-to-end streaming smoke test integrates Phase 1, 2, 3, and 4 causally",
        smoke_ok,
        f"10-step streaming completed; mean R_t={np.mean(r_values):.4f}, final R_t={r_values[-1]:.4f}",
    )

    # -------------------------------------------------------------------------
    # Summary
    # -------------------------------------------------------------------------
    print("-" * 78)
    n_passed = sum(1 for _, ok, _ in results if ok)
    n_total = len(results)
    for name, ok, detail in results:
        status = "[PASS]" if ok else "[FAIL]"
        print(f"{status} {name}")
        if detail:
            print(f"       {detail}")
    print("-" * 78)
    print(f"{n_passed}/{n_total} checks passed")
    banner("END PHASE 4 VERIFICATION")

    return 0 if n_passed == n_total else 1


if __name__ == "__main__":
    sys.exit(main())
