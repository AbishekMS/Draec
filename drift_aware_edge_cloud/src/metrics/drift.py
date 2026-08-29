"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/metrics/drift.py
Phase    : Phase 10
Status   : IMPLEMENTED

Detection delay, false-positive / missed drift rates, and severity trajectory metrics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def compute_drift_metrics(
    detection_indices: Sequence[int],
    drift_onset_index: int | None,
    total_steps: int,
    transient_alarms: int = 0,
    persistent_events: int = 0,
    severity_history: Sequence[float] | None = None,
) -> dict[str, Any]:
    """Compute empirical drift detection performance metrics.

    Parameters:
    - detection_indices: Step indices where ADWIN detected a change.
    - drift_onset_index: Ground-truth drift start index (or None for control/no drift).
    - total_steps: Total steps in the evaluation stream.
    - transient_alarms: Number of isolated alarms rejected by persistence filter.
    - persistent_events: Number of confirmed persistent drift events.
    - severity_history: Sequence of smoothed severity scores D_t over time.

    Returns:
    Dictionary containing detection delay, alarm counts, false alarm rate, and severity metrics.
    """
    dets = sorted(list(detection_indices))
    sev = np.asarray(severity_history, dtype=float) if severity_history is not None else np.array([])

    total_alarms = len(dets)

    if drift_onset_index is None:
        # Control / clean stream scenario (no injected drift)
        return {
            "drift_scenario": "no_drift",
            "drift_onset": None,
            "first_detection_point": dets[0] if dets else None,
            "detection_delay": None,
            "detection_status": "NO_DRIFT",
            "total_alarms": total_alarms,
            "false_alarms_pre_drift": total_alarms,
            "post_drift_alarms": 0,
            "transient_alarms": int(transient_alarms),
            "persistent_events": int(persistent_events),
            "mean_severity": float(np.mean(sev)) if len(sev) > 0 else 0.0,
            "max_severity": float(np.max(sev)) if len(sev) > 0 else 0.0,
        }

    onset = int(drift_onset_index)
    pre_alarms = [d for d in dets if d < onset]
    post_alarms = [d for d in dets if d >= onset]

    first_post = post_alarms[0] if post_alarms else None
    if first_post is not None:
        delay = int(first_post - onset)
        status = "DETECTED"
    else:
        delay = None
        status = "NOT DETECTED"

    return {
        "drift_scenario": "drift_injected",
        "drift_onset": onset,
        "first_detection_point": first_post,
        "detection_delay": delay,
        "detection_status": status,
        "total_alarms": total_alarms,
        "false_alarms_pre_drift": len(pre_alarms),
        "post_drift_alarms": len(post_alarms),
        "transient_alarms": int(transient_alarms),
        "persistent_events": int(persistent_events),
        "mean_severity": float(np.mean(sev)) if len(sev) > 0 else 0.0,
        "max_severity": float(np.max(sev)) if len(sev) > 0 else 0.0,
        "post_drift_mean_severity": float(np.mean(sev[onset:])) if len(sev) > onset else 0.0,
    }
