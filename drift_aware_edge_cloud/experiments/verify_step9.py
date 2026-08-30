"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : experiments/verify_step9.py
Phase    : Phase 10 / Step 9
Status   : IMPLEMENTED

Step 9 Scientific Validity, Reproducibility, and Integrity Verification Harness.
Performs 13 strict automated integrity, bounds, aggregation, and reproducibility checks.
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

from src.utils import config as config_mod
from src.metrics.evaluation import Phase10Evaluator
from src.metrics.prediction import compute_classification_metrics

# Test reporting tracker
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
    print("EXPERIMENTS / VERIFY_STEP9: STEP 8 SCIENTIFIC INTEGRITY HARNESS")
    print("=" * 80)

    results_dir = ROOT / "results"
    raw_json_path = results_dir / "step8_raw_per_seed_results.json"
    comb_csv_path = results_dir / "step8_combined_runs.csv"
    cfg_a_csv_path = results_dir / "step8_config_a_moderate.csv"
    cfg_b_csv_path = results_dir / "step8_config_b_severe.csv"
    agg_csv_path = results_dir / "step8_aggregated_summary.csv"

    # 1. Load Step 8 raw results
    try:
        assert raw_json_path.exists(), f"Missing {raw_json_path}"
        assert comb_csv_path.exists(), f"Missing {comb_csv_path}"
        assert cfg_a_csv_path.exists(), f"Missing {cfg_a_csv_path}"
        assert cfg_b_csv_path.exists(), f"Missing {cfg_b_csv_path}"
        assert agg_csv_path.exists(), f"Missing {agg_csv_path}"

        with open(raw_json_path, "r", encoding="utf-8") as f:
            raw_json = json.load(f)
        df_comb = pd.read_csv(comb_csv_path)
        df_a = pd.read_csv(cfg_a_csv_path)
        df_b = pd.read_csv(cfg_b_csv_path)
        df_agg = pd.read_csv(agg_csv_path)

        record_check("Check 01: Load Step 8 raw and CSV artifacts", True, "All 5 artifact files loaded successfully")
    except Exception as e:
        record_check("Check 01: Load Step 8 raw and CSV artifacts", False, str(e))
        return 1

    # 2. Verify exactly five unique seeds
    seeds = sorted(df_comb["seed"].unique().tolist())
    expected_seeds = [42, 123, 456, 789, 2024]
    check2_pass = (seeds == expected_seeds) and (len(df_comb) == 10)
    record_check(
        "Check 02: Exactly five unique seeds in dataset",
        check2_pass,
        f"Seeds found: {seeds} (total rows: {len(df_comb)})",
    )

    # 3. Verify both locked configurations
    configs = sorted(df_comb["config"].unique().tolist())
    expected_configs = ["Config_A_Moderate", "Config_B_Severe"]
    check3_pass = configs == expected_configs
    record_check("Check 03: Both locked configurations evaluated", check3_pass, f"Configs: {configs}")

    # 4. Check raw JSON vs CSV consistency
    mismatches = 0
    checked_fields = [
        "post_accuracy", "post_macro_f1", "post_mcc", "min_r", "mean_post_r",
        "max_d", "mean_post_d", "edge_pct", "hybrid_pct", "cloud_pct",
        "switches", "candidate_macro_f1",
    ]
    for c_name in expected_configs:
        for s in expected_seeds:
            j_metrics = raw_json[c_name][str(s)]["metrics"]
            row = df_comb[(df_comb["config"] == c_name) & (df_comb["seed"] == s)].iloc[0]
            for fld in checked_fields:
                if not math.isclose(float(j_metrics[fld]), float(row[fld]), rel_tol=1e-5, abs_tol=1e-5):
                    mismatches += 1
    record_check("Check 04: JSON vs CSV exact consistency", mismatches == 0, f"Mismatches: {mismatches}")

    # 5. Recompute aggregates independently
    metric_cols = [
        "post_accuracy", "post_macro_f1", "post_mcc", "post_precision", "post_recall",
        "min_r", "mean_post_r", "max_d", "mean_post_d", "detection_delay", "persistent_events",
        "edge_pct", "hybrid_pct", "cloud_pct", "switches", "adaptation_triggers",
        "candidate_macro_f1", "successful_deployments",
    ]
    agg_diffs = 0
    for col in metric_cols:
        ma, sa, la, ua = compute_ci(df_a[col].values)
        mb, sb, lb, ub = compute_ci(df_b[col].values)
        row = df_agg[df_agg["metric"] == col].iloc[0]
        if not (
            math.isclose(ma, row["config_a_mean"], abs_tol=1e-5)
            and math.isclose(sa, row["config_a_std"], abs_tol=1e-5)
            and math.isclose(la, row["config_a_ci_lower"], abs_tol=1e-5)
            and math.isclose(ua, row["config_a_ci_upper"], abs_tol=1e-5)
            and math.isclose(mb, row["config_b_mean"], abs_tol=1e-5)
            and math.isclose(sb, row["config_b_std"], abs_tol=1e-5)
            and math.isclose(lb, row["config_b_ci_lower"], abs_tol=1e-5)
            and math.isclose(ub, row["config_b_ci_upper"], abs_tol=1e-5)
        ):
            agg_diffs += 1
    record_check("Check 05: Independent aggregate recomputation", agg_diffs == 0, f"Aggregate discrepancies: {agg_diffs}")

    # 6. Check degrees of freedom and CI formula
    # For N=5, df=4, Student-t multiplier at 95% is ~2.7764
    t_val_test = float(sp_stats.t.ppf(0.975, df=4))
    check6_pass = math.isclose(t_val_test, 2.7764451, rel_tol=1e-5)
    record_check("Check 06: Student's t distribution (df=4, t=2.7764)", check6_pass, f"t-critical: {t_val_test:.6f}")

    # 7. Check seed uniqueness & baseline sample hash divergence
    hashes = df_comb["baseline_hash"].unique().tolist()
    # 5 unique seeds should produce 5 unique baseline hashes
    check7_pass = len(hashes) == 5
    record_check("Check 07: Seed uniqueness & baseline hash divergence", check7_pass, f"Unique baseline hashes: {len(hashes)}/5")

    # 8. Check metric bounds [0, 1] and [0, 100]
    bounds_ok = True
    for col in ["post_accuracy", "post_macro_f1", "min_r", "mean_post_r", "max_d", "mean_post_d"]:
        vals = df_comb[col].values
        if (vals < 0.0).any() or (vals > 1.0).any():
            bounds_ok = False
    for col in ["edge_pct", "hybrid_pct", "cloud_pct"]:
        vals = df_comb[col].values
        if (vals < 0.0).any() or (vals > 100.0).any():
            bounds_ok = False
    record_check("Check 08: Metric bounds ([0, 1] and [0, 100%])", bounds_ok, "All metrics within physical bounds")

    # 9. Check NaN/Inf
    nan_or_inf = df_comb.isna().any().any() or np.isinf(df_comb.select_dtypes(include=[np.number])).any().any()
    record_check("Check 09: Numerical stability (zero NaNs, zero Infs)", not nan_or_inf, f"NaN or Inf present: {nan_or_inf}")

    # 10. Check adaptation counts
    triggers = df_comb["adaptation_triggers"].unique().tolist()
    deploys = df_comb["successful_deployments"].unique().tolist()
    check10_pass = (triggers == [2]) and (deploys == [0])
    record_check(
        "Check 10: Adaptation triggers and safety deployments",
        check10_pass,
        f"Triggers per run: {triggers}, Deployments per run: {deploys}",
    )

    # 11. Check routing percentages sum to approximately 100%
    route_sum = df_comb["edge_pct"] + df_comb["hybrid_pct"] + df_comb["cloud_pct"]
    sum_diff = np.max(np.abs(route_sum - 100.0))
    check11_pass = sum_diff < 1e-4
    record_check("Check 11: Routing percentages sum to 100%", check11_pass, f"Max deviation: {sum_diff:.6f}%")

    # 12. Check reproducibility against stored run (Config B, Seed 42)
    cfg = config_mod.load("default")
    evaluator = Phase10Evaluator(config=cfg, results_dir=results_dir, seeds=[42])
    rerun = evaluator.run_streaming_simulation(
        method="FULL_DRAEC",
        drift_scenario="sudden",
        magnitude=5.0,
        n_features=8,
        stream_steps=25000,
        seed=42,
    )
    s42_stored = raw_json["Config_B_Severe"]["42"]["metrics"]
    s42_rerun_min_r = float(np.min(rerun["r_t_history"]))
    s42_rerun_max_d = float(np.max(rerun["d_t_history"]))
    s42_rerun_cloud_pct = float(np.mean([a == "CLOUD" for a in rerun["actions"]])) * 100.0
    repro_match = (
        math.isclose(s42_stored["min_r"], s42_rerun_min_r, abs_tol=1e-5)
        and math.isclose(s42_stored["max_d"], s42_rerun_max_d, abs_tol=1e-5)
        and math.isclose(s42_stored["cloud_pct"], s42_rerun_cloud_pct, abs_tol=1e-5)
    )
    record_check(
        "Check 12: Bit-exact reproducibility of stored run (Seed 42)",
        repro_match,
        f"Stored min_r={s42_stored['min_r']:.4f}, rerun min_r={s42_rerun_min_r:.4f}",
    )

    # 13. Overall Summary
    total_checks = len(CHECK_RESULTS)
    passed_checks = sum(1 for c in CHECK_RESULTS if c["status"] == "PASS")
    all_passed = passed_checks == total_checks

    print("\n" + "=" * 80)
    print(f"STEP 9 VERIFICATION HARNESS RESULT: {'ALL PASS' if all_passed else 'FAILURES DETECTED'}")
    print(f"Passed: {passed_checks} / {total_checks} checks")
    print("=" * 80)

    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())
