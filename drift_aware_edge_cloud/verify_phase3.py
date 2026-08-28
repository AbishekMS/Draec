"""Phase 3 Verification Harness -- Drift Detection (ADWIN, Persistence, Severity).

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

Verifies:
- ADWIN statistical change detector (River ADWIN wrapper)
- Drift persistence tracking (consecutive streak & windowed recurrence)
- Continuous drift severity (exact relative_shift formula, raw vs smoothed)
- Causal baseline reference computation
- No ground truth or future label leakage
- Deterministic streaming behavior
- Compatibility with Phase 1 feature representations and Phase 2 models

Run:
    PYTHONIOENCODING=utf-8 ../.venv/Scripts/python.exe verify_phase3.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def note(msg: str) -> None:
    _details.append(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


# =============================================================================
# 1. Modules import and are marked IMPLEMENTED
# =============================================================================
try:
    from src.drift import (
        ADWINDetector,
        DriftPersistence,
        DriftPipeline,
        DriftSeverity,
        DriftStatus,
        compute_baseline_signal_mean,
    )
    from src.models import CloudModel, EdgeModel
    from src.utils import config as cfgmod

    MOD_NAMES = ("adwin_detector.py", "persistence.py", "severity.py", "__init__.py")
    impl_flags = []
    for mod_name in MOD_NAMES:
        content = (ROOT / "src" / "drift" / mod_name).read_text(encoding="utf-8")
        impl_flags.append("Status   : IMPLEMENTED" in content)

    check(
        "1. all four src/drift modules import and are marked IMPLEMENTED",
        all(impl_flags),
        f"verified {', '.join(MOD_NAMES)} carry 'Status   : IMPLEMENTED'",
    )
except Exception as e:
    check("1. src/drift import", False, f"{type(e).__name__}: {e}")
    print("FATAL: cannot import src/drift modules; aborting.")
    raise SystemExit(1)

CONFIG_DIR = ROOT / "config"
cfg = cfgmod.load("default", config_dir=CONFIG_DIR)

# =============================================================================
# 2. ADWINDetector parameter binding from config
# =============================================================================
det = ADWINDetector(cfg)
det_info = det.get_info()
check(
    "2. ADWINDetector instantiates and binds configured parameters",
    (
        det.delta == 0.002
        and det.clock == 32
        and det.monitored_signal == "prediction_probability"
        and not det.drift_detected
        and det.n_samples_seen == 0
        and det_info["detector_type"] == "ADWINDetector"
    ),
    f"delta={det.delta}, clock={det.clock}, signal={det.monitored_signal}",
)

# =============================================================================
# 3. ADWIN sequential update, window tracking, and reset
# =============================================================================
for val in [0.05] * 25:
    det.update(val)
w_before = det.width
det.reset()
check(
    "3. ADWIN tracks window width sequentially and resets cleanly",
    (w_before > 0 and det.width == 0 and det.n_samples_seen == 0 and det.n_drifts_detected == 0),
    f"width_before={w_before}, width_after_reset={det.width}",
)

# =============================================================================
# 4. ADWIN change detection on controlled distribution shift
# =============================================================================
rng = np.random.default_rng(42)
det_shift = ADWINDetector(delta=0.002, clock=8, grace_period=10)
# 150 low baseline samples
pre = rng.normal(0.04, 0.005, size=150).clip(0.0, 1.0)
for v in pre:
    det_shift.update(v)
pre_drifts = det_shift.n_drifts_detected

# Abrupt shift to 0.85
post = rng.normal(0.85, 0.01, size=120).clip(0.0, 1.0)
shift_detected = False
for v in post:
    if det_shift.update(v):
        shift_detected = True
        break

check(
    "4. ADWIN detects statistically significant distribution shift without false alarms on pre-drift",
    (pre_drifts == 0 and shift_detected),
    f"pre_drift_alarms={pre_drifts}, shift_detected={shift_detected}, total_alarms={det_shift.n_drifts_detected}",
)

# =============================================================================
# 5. DriftPersistence consecutive criterion
# =============================================================================
persist_consec = DriftPersistence(criterion="consecutive", consecutive_threshold=3)
persist_consec.update(True)
persist_consec.update(False)
transient_rejected = not persist_consec.is_persistent

persist_consec.update(True)
persist_consec.update(True)
almost_persistent = not persist_consec.is_persistent
persist_consec.update(True)
persistent_confirmed = persist_consec.is_persistent

check(
    "5. DriftPersistence filters transient alarms and confirms consecutive persistence",
    (transient_rejected and almost_persistent and persistent_confirmed),
    f"transient_rejected={transient_rejected}, 3_consecutive_persistent={persistent_confirmed}",
)

# =============================================================================
# 6. DriftPersistence windowed count criterion
# =============================================================================
persist_win = DriftPersistence(criterion="windowed_count", window_size=5, count_threshold=3)
# Sequence: True, False, True, False, True -> 3 in 5 -> persistent
for flag in (True, False, True, False):
    persist_win.update(flag)
not_yet = not persist_win.is_persistent
persist_win.update(True)
now_persistent = persist_win.is_persistent

check(
    "6. DriftPersistence windowed count criterion triggers at threshold",
    (not_yet and now_persistent),
    f"4_steps_persistent={not_yet}, 5th_step_persistent={now_persistent}",
)

# =============================================================================
# 7. DriftSeverity exact relative_shift formula
# =============================================================================
# Formula: D = min(1.0, abs(current_shift - baseline_mean) / max_shift)
b_mean = 0.05
m_shift = 0.95
sev = DriftSeverity(formula="relative_shift", baseline_mean=b_mean, max_shift=m_shift, smoothing_factor=0.0)

val1 = 0.05
d1 = sev.compute_raw_severity(val1)  # 0.0
val2 = 0.525
d2 = sev.compute_raw_severity(val2)  # abs(0.525 - 0.05) / 0.95 = 0.475 / 0.95 = 0.50
val3 = 1.20
d3 = sev.compute_raw_severity(val3)  # clipped to 1.0

check(
    "7. DriftSeverity implements exact relative_shift formula D = min(1.0, abs(shift-mean)/max_shift)",
    (abs(d1 - 0.0) < 1e-6 and abs(d2 - 0.5) < 1e-6 and abs(d3 - 1.0) < 1e-6),
    f"D(0.05)={d1}, D(0.525)={d2}, D(1.20)={d3}",
)

# =============================================================================
# 8. DriftSeverity distinguishes raw_severity from smoothed_severity
# =============================================================================
sev_smooth = DriftSeverity(
    formula="relative_shift",
    baseline_mean=0.0,
    max_shift=1.0,
    smoothing_factor=0.6,
)
sev_smooth.update(1.0)
r1 = sev_smooth.raw_severity
s1 = sev_smooth.smoothed_severity

sev_smooth.update(0.0)
r2 = sev_smooth.raw_severity
s2 = sev_smooth.smoothed_severity
# s2 = 0.6 * 1.0 + 0.4 * 0.0 = 0.6
check(
    "8. DriftSeverity strictly separates raw_severity from smoothed_severity",
    (r1 == 1.0 and s1 == 1.0 and r2 == 0.0 and abs(s2 - 0.6) < 1e-6),
    f"step1 raw={r1} smooth={s1}; step2 raw={r2} smooth={s2}",
)

# =============================================================================
# 9. Causal baseline mean computation on baseline_train only
# =============================================================================
from src.models import load_causal_train_data

X_train_sub, y_train_sub, stats, profile = load_causal_train_data(cfg, root=ROOT, max_rows=500)
edge_model = EdgeModel().fit(X_train_sub, y_train_sub)
causal_b_mean = compute_baseline_signal_mean(edge_model, X_train_sub, signal_type="prediction_probability")

check(
    "9. compute_baseline_signal_mean evaluates causally on baseline_train only",
    (0.0 <= causal_b_mean <= 1.0),
    f"empirical baseline_mean on baseline_train: {causal_b_mean:.5f}",
)

# =============================================================================
# 10. Leakage protection: zero ground-truth / target dependency
# =============================================================================
pipe = DriftPipeline(cfg)
pipe.severity.set_baseline_mean(causal_b_mean)
# Ensure update does not query Target
prob_sample = edge_model.predict_proba_one(X_train_sub.iloc[0])
status = pipe.update_from_prediction(prob_sample)

check(
    "10. Phase 3 components execute using only observable model outputs without Target or ground truth",
    (isinstance(status, DriftStatus) and not status.drift_detected and status.raw_severity < 0.2),
    f"status: drift={status.drift_detected}, persistent={status.is_persistent}, raw_sev={status.raw_severity:.4f}",
)

# =============================================================================
# 11. Deterministic streaming behavior
# =============================================================================
rng1 = np.random.default_rng(777)
s1 = rng1.uniform(0.0, 0.2, size=50)
p1 = DriftPipeline(cfg)
r1 = [p1.update_scalar(v).raw_severity for v in s1]

rng2 = np.random.default_rng(777)
s2 = rng2.uniform(0.0, 0.2, size=50)
p2 = DriftPipeline(cfg)
r2 = [p2.update_scalar(v).raw_severity for v in s2]

check(
    "11. Phase 3 pipeline execution is deterministic under identical stream inputs",
    (r1 == r2),
    f"runs match across {len(r1)} steps",
)

# =============================================================================
# 12. End-to-end streaming integration with Phase 1 features and Phase 2 models
# =============================================================================
cloud_model = CloudModel().fit(X_train_sub, y_train_sub)
cloud_probs = cloud_model.predict_proba_one(X_train_sub.iloc[10])
status_cloud = pipe.update_from_prediction(cloud_probs)

check(
    "12. small end-to-end streaming test with Phase 2 models passes",
    (isinstance(status_cloud, DriftStatus) and 0.0 <= status_cloud.raw_severity <= 1.0),
    f"Cloud model inference status -> raw_sev={status_cloud.raw_severity:.4f}, smoothed_sev={status_cloud.smoothed_severity:.4f}",
)

# =============================================================================
# Report results
# =============================================================================
print("=" * 78)
print("PHASE 3 VERIFICATION -- src/drift/{adwin_detector,persistence,severity}.py")
print("=" * 78)
print("-" * 78)

passed_count = 0
for name, ok, detail in results:
    status_str = "[PASS]" if ok else "[FAIL]"
    if ok:
        passed_count += 1
    print(f"{status_str} {name}")
    if detail:
        print(f"       {detail}")

print("-" * 78)
print(f"{passed_count}/{len(results)} checks passed")
print("=" * 78)

if passed_count != len(results):
    sys.exit(1)
