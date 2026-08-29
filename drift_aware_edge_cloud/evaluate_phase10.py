"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Script   : evaluate_phase10.py
Phase    : Phase 10
Purpose  : Execute the full scientific evaluation, benchmarking, and IEEE deliverable generation.
"""

from __future__ import annotations

import argparse
import sys
import time

from src.metrics.evaluation import Phase10Evaluator
from src.utils import config as config_mod


def main() -> int:
    parser = argparse.ArgumentParser(description="DRAEC Phase 10 Scientific Evaluation & Benchmarking")
    parser.add_argument("--steps", type=int, default=1000, help="Stream steps per evaluation run (default: 1000)")
    parser.add_argument("--seeds", nargs="+", type=int, default=[42, 43, 44, 45, 46], help="Random seeds for multi-seed protocol")
    parser.add_argument("--output-dir", type=str, default="results", help="Directory to save experimental results")
    args = parser.parse_args()

    print("=" * 78)
    print("DRAEC PHASE 10 — FINAL SCIENTIFIC EVALUATION & BENCHMARKING")
    print("=" * 78)
    print(f"Dataset             : WUSTL-IIoT-2021 (wustl_iiot_2021.csv)")
    print(f"Evaluation Stream   : test1 (inference_stream)")
    print(f"Seeds Evaluated     : {args.seeds}")
    print(f"Steps Per Run       : {args.steps:,}")
    print(f"Results Output Dir  : {args.output_dir}")
    print("-" * 78)

    t0 = time.time()
    cfg = config_mod.load("default")
    evaluator = Phase10Evaluator(config=cfg, results_dir=args.output_dir, seeds=args.seeds)

    print("[1/3] Executing multi-seed streaming simulation runs across all benchmark methods...")
    all_runs = evaluator.evaluate_multi_seed(steps_per_run=args.steps)
    print(f"      Completed {len(args.seeds)} seeds across 6 configurations.")

    print("[2/3] Generating experimental metrics, statistical tests, and IEEE deliverables...")
    deliverables = evaluator.generate_all_deliverables(all_runs)

    print("[3/3] Verifying output artifacts...")
    for key, df in deliverables.items():
        print(f"      - {key:<18}: {len(df):>4} records generated")

    elapsed = time.time() - t0
    print("-" * 78)
    print(f"PHASE 10 EVALUATION COMPLETED IN {elapsed:.2f}s")
    print("=" * 78)
    return 0


if __name__ == "__main__":
    sys.exit(main())
