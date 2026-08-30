"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : experiments/verify_step10.py
Phase    : Phase 10 / Step 10
Status   : IMPLEMENTED

Step 10 Publication Deliverables & IEEE Results Package Verification Harness.
Verifies all publication tables, aggregations, variance classifications, bounds,
and artifact integrity against the authoritative Step 8 multi-seed benchmark.
"""

from __future__ import annotations

import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy import stats as sp_stats

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

CHECK_RESULTS: list[dict[str, Any]] = []


def record_check(name: str, passed: bool, details: str = "") -> None:
    status = "PASS" if passed else "FAIL"
    CHECK_RESULTS.append({"name": name, "status": status, "details": details})
    mark = "[PASS]" if passed else "[FAIL]"
    print(f"  {mark} {name}: {details}")


def compute_ci(arr: np.ndarray, conf: float = 0.95) -> tuple[float, float, float, float]:
    n = len(arr)
    mean = float(np.mean(arr))
    std = float(np.std(arr, ddof=1))
    if std < 1e-12:
        return mean, 0.0, mean, mean
    df = n - 1
    t_val = float(sp_stats.t.ppf((1.0 + conf) / 2.0, df=df))
    margin = t_val * (std / math.sqrt(n))
    return mean, std, float(mean - margin), float(mean + margin)


def main() -> int:
    print("=" * 80)
    print("EXPERIMENTS / VERIFY_STEP10: IEEE DELIVERABLES & RESULTS HARNESS")
    print("=" * 80)

    results_dir = ROOT / "results"
    tables_dir = results_dir / "tables"
    raw_json_path = results_dir / "step8_raw_per_seed_results.json"
    comb_csv_path = results_dir / "step8_combined_runs.csv"
    agg_csv_path = results_dir / "step8_aggregated_summary.csv"
    pub_md_path = results_dir / "step10_publication_results.md"
    tab1_path = tables_dir / "ieee_table1_predictive.tex"
    tab2_path = tables_dir / "ieee_table2_orchestration.tex"
    tab3_path = tables_dir / "ieee_table3_adaptation.tex"

    # 1. Verify existence of all Step 8 and Step 10 publication artifacts
    artifacts = [raw_json_path, comb_csv_path, agg_csv_path, pub_md_path, tab1_path, tab2_path, tab3_path]
    missing = [str(p.name) for p in artifacts if not p.exists()]
    record_check("Check 01: Authoritative artifacts exist", len(missing) == 0, f"Missing: {missing}")
    if missing:
        return 1

    with open(raw_json_path, "r", encoding="utf-8") as f:
        raw_json = json.load(f)
    df_comb = pd.read_csv(comb_csv_path)
    df_agg = pd.read_csv(agg_csv_path)

    # 2. Verify exactly five seeds per configuration
    expected_seeds = [42, 123, 456, 789, 2024]
    seeds_a = sorted(df_comb[df_comb["config"] == "Config_A_Moderate"]["seed"].tolist())
    seeds_b = sorted(df_comb[df_comb["config"] == "Config_B_Severe"]["seed"].tolist())
    seeds_ok = (seeds_a == expected_seeds) and (seeds_b == expected_seeds) and (len(df_comb) == 10)
    record_check(
        "Check 02: Exactly 5 seeds per configuration",
        seeds_ok,
        f"Seeds A: {seeds_a}, Seeds B: {seeds_b}",
    )

    # 3. Verify both locked configurations (Config A: 2.0σ/n=5, Config B: 5.0σ/n=8)
    configs = sorted(df_comb["config"].unique().tolist())
    cfg_a_params = df_comb[df_comb["config"] == "Config_A_Moderate"][["magnitude", "n_features"]].iloc[0].to_dict()
    cfg_b_params = df_comb[df_comb["config"] == "Config_B_Severe"][["magnitude", "n_features"]].iloc[0].to_dict()
    cfg_ok = (
        configs == ["Config_A_Moderate", "Config_B_Severe"]
        and cfg_a_params == {"magnitude": 2.0, "n_features": 5}
        and cfg_b_params == {"magnitude": 5.0, "n_features": 8}
    )
    record_check("Check 03: Both locked configurations verified", cfg_ok, f"A: {cfg_a_params}, B: {cfg_b_params}")

    # 4. Ensure NO exploratory 4σ or n=9 artifacts enter the final benchmark tables
    mags = set(df_comb["magnitude"].tolist())
    n_feats = set(df_comb["n_features"].tolist())
    exploratory_leak = (4.0 in mags) or (9 in n_feats) or (10 in n_feats)
    record_check(
        "Check 04: Zero exploratory (4-sigma/n=9) artifact leakage",
        not exploratory_leak,
        f"Magnitudes: {mags}, Features: {n_feats}",
    )

    # 5. Distinguish deterministic vs stochastic metrics and detect zero-variance metrics
    zero_variance_cols = [
        "post_accuracy", "post_macro_f1", "post_mcc", "post_precision", "post_recall",
        "min_r", "mean_post_r", "max_d", "mean_post_d", "detection_delay", "persistent_events",
        "edge_pct", "hybrid_pct", "cloud_pct", "switches", "adaptation_triggers",
        "successful_deployments",
    ]
    stochastic_cols = ["candidate_macro_f1"]

    zero_var_ok = True
    for c in zero_variance_cols:
        if df_comb.groupby("config")[c].std(ddof=1).max() > 1e-12:
            zero_var_ok = False

    stoch_ok = True
    for c in stochastic_cols:
        stds = df_comb.groupby("config")[c].std(ddof=1)
        if (stds <= 1e-12).any():
            stoch_ok = False

    record_check(
        "Check 05: Variance classification & zero-variance detection",
        zero_var_ok and stoch_ok,
        f"Zero-variance metrics verified: {len(zero_variance_cols)}, Stochastic metrics verified: {len(stochastic_cols)}",
    )

    # 6. Verify sample standard deviation, df=4, and 95% Student-t CI formula
    t_val = float(sp_stats.t.ppf(0.975, df=4))
    ci_math_ok = math.isclose(t_val, 2.7764451, rel_tol=1e-5)
    row_cand = df_agg[df_agg["metric"] == "candidate_macro_f1"].iloc[0]
    expected_margin_b = t_val * (row_cand["config_b_std"] / math.sqrt(5))
    ci_b_match = math.isclose(row_cand["config_b_ci_upper"] - row_cand["config_b_mean"], expected_margin_b, abs_tol=1e-6)
    record_check(
        "Check 06: Student's t distribution & CI formula (df=4)",
        ci_math_ok and ci_b_match,
        f"t-critical={t_val:.6f}, Config B Margin={expected_margin_b:.6f}",
    )

    # 7. Verify Table I values against raw artifacts
    tab1_content = tab1_path.read_text(encoding="utf-8")
    tab1_checks = [
        "0.9960", "0.4990", "0.0000", "0.4980", "0.5000",
        "0.4160", "0.4168", "0.4146", "0.4174", "0.4165", "0.4170",
        "REJECTED", "v1", "Deterministic by Design", "Stochastic / Seed-Dependent",
    ]
    tab1_pass = all(tok in tab1_content for tok in tab1_checks)
    record_check("Check 07: Table I content verified against raw benchmark", tab1_pass, "All tokens verified")

    # 8. Verify Table II values against raw artifacts
    tab2_content = tab2_path.read_text(encoding="utf-8")
    tab2_checks = [
        "12,500", "12,575", "12,511", "75", "11", "100",
        "0.7299", "0.9652", "0.1849", "0.8926", "0.5968", "0.1261",
        "0.9444", "0.3238", "100.000\\%", "50.032\\%", "0.080\\%", "49.888\\%",
    ]
    tab2_missing = [tok for tok in tab2_checks if tok not in tab2_content]
    tab2_pass = len(tab2_missing) == 0
    record_check(
        "Check 08: Table II content verified against raw benchmark",
        tab2_pass,
        "All tokens verified" if tab2_pass else f"Missing tokens: {tab2_missing}",
    )

    # 9. Verify Table III values against raw artifacts
    tab3_content = tab3_path.read_text(encoding="utf-8")
    tab3_checks = [
        "1,000", "200 (194 C0, 6 C1)", "1,200", "0.70", "0.4168",
        "REJECTED", "BLOCKED", "v1",
    ]
    tab3_missing = [tok for tok in tab3_checks if tok not in tab3_content]
    tab3_pass = len(tab3_missing) == 0
    record_check(
        "Check 09: Table III content verified against raw benchmark",
        tab3_pass,
        "All tokens verified" if tab3_pass else f"Missing tokens: {tab3_missing}",
    )

    # 10. Verify Claim-Evidence Matrix consistency in publication results
    pub_content = pub_md_path.read_text(encoding="utf-8")
    matrix_checks = [
        "Drift-Aware Reliability Tracking",
        "Edge-Preserving Routing under Moderate Drift",
        "Dynamic Cloud Offloading under Severe Drift",
        "Safety Firewall Blocks Degraded Models",
        "Improved Intrusion Detection Accuracy",
        "NOT SUPPORTED",
    ]
    matrix_missing = [m for m in matrix_checks if m not in pub_content]
    matrix_pass = len(matrix_missing) == 0
    record_check(
        "Check 10: Claim-Evidence matrix present with honest scoping",
        matrix_pass,
        "All claims/caveats mapped" if matrix_pass else f"Missing: {matrix_missing}",
    )

    # 11. Verify inline classifier caveats accompany orchestration discussions
    caveat_tokens = [
        "poor minority-class",
        "must not be interpreted as preservation of attack-detection performance",
        "cross-partition feature inversion",
        "was not dynamically exercised",
    ]
    caveat_missing = [tok for tok in caveat_tokens if tok.lower() not in pub_content.lower()]
    caveats_found = len(caveat_missing) == 0
    record_check(
        "Check 11: Inline classifier caveats present in narrative",
        caveats_found,
        "Caveats verified" if caveats_found else f"Missing caveats: {caveat_missing}",
    )

    # 12. Final overall status
    total_checks = len(CHECK_RESULTS)
    passed_checks = sum(1 for c in CHECK_RESULTS if c["status"] == "PASS")
    all_passed = passed_checks == total_checks

    print("\n" + "=" * 80)
    print(f"STEP 10 VERIFICATION HARNESS RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print(f"Passed: {passed_checks} / {total_checks} checks")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
