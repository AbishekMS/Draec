"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/metrics/system.py
Phase    : Phase 10
Status   : IMPLEMENTED

Latency, execution reliability, simulated network conditions, and unmeasured hardware guards.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def compute_latency_summary(
    latencies_s: Sequence[float],
    empty_as_none: bool = False,
) -> dict[str, Any]:
    """Compute empirical latency summary statistics from measured wall-clock / simulated quantities.

    Parameters:
    - latencies_s: Sequence of measured latency values in seconds.
    - empty_as_none: If True, return None for statistics when valid latencies are empty.
                     If False (default), returns 0.0 for backward compatibility.

    Returns:
    Dictionary containing mean, median, P95, and max in milliseconds and seconds.
    """
    valid = [float(v) for v in latencies_s if v is not None and not np.isnan(v)]
    if not valid:
        default_val = None if empty_as_none else 0.0
        return {
            "count": 0,
            "mean_ms": default_val,
            "median_ms": default_val,
            "p95_ms": default_val,
            "max_ms": default_val,
            "mean_s": default_val,
            "median_s": default_val,
            "p95_s": default_val,
            "max_s": default_val,
        }

    arr = np.asarray(valid, dtype=float)
    arr_ms = arr * 1000.0

    return {
        "count": int(len(arr)),
        "mean_ms": float(np.mean(arr_ms)),
        "median_ms": float(np.median(arr_ms)),
        "p95_ms": float(np.percentile(arr_ms, 95)),
        "max_ms": float(np.max(arr_ms)),
        "mean_s": float(np.mean(arr)),
        "median_s": float(np.median(arr)),
        "p95_s": float(np.percentile(arr, 95)),
        "max_s": float(np.max(arr)),
    }


def compute_execution_reliability(
    total_executions: int,
    successful_executions: int,
    edge_failures: int = 0,
    cloud_failures: int = 0,
    packet_loss_failures: int = 0,
    fallback_failures: int = 0,
) -> dict[str, Any]:
    """Compute execution reliability metrics without fabricated predictions.

    Parameters:
    - total_executions: Total execution attempts.
    - successful_executions: Successfully completed inference executions.
    - edge_failures: Local edge runtime failures.
    - cloud_failures: Cloud service failures.
    - packet_loss_failures: Failures caused by network packet loss.
    - fallback_failures: Failures occurring during hybrid cloud fallback.

    Returns:
    Dictionary containing success rate, failure rate, and failure breakdown.
    """
    total = max(1, int(total_executions))
    succ = int(successful_executions)
    failed = total - succ

    return {
        "total_executions": int(total_executions),
        "successful_executions": succ,
        "failed_executions": failed,
        "success_rate": float(succ / total),
        "failure_rate": float(failed / total),
        "edge_failures": int(edge_failures),
        "cloud_failures": int(cloud_failures),
        "packet_loss_failures": int(packet_loss_failures),
        "fallback_failures": int(fallback_failures),
    }


def compute_network_metrics(
    total_transmissions: int,
    delivered_transmissions: int,
    packet_loss_count: int,
    latencies_s: Sequence[float],
    total_bytes_transmitted: int | None = None,
) -> dict[str, Any]:
    """Compute simulated network metrics under Phase 8 NetworkSimulator conditions.

    Parameters:
    - total_transmissions: Total packet transmission attempts.
    - delivered_transmissions: Successfully delivered transmissions.
    - packet_loss_count: Packets dropped by simulator.
    - latencies_s: Simulated network latency measurements in seconds.
    - total_bytes_transmitted: Sum of packet byte sizes if instrumented.

    Returns:
    Dictionary containing delivery rate, failure rate, loss rate, simulated latency summary, and bandwidth status.
    """
    tot = max(1, int(total_transmissions))
    delivered = int(delivered_transmissions)
    lost = int(packet_loss_count)
    failed = max(0, tot - delivered)

    # When no transmissions were delivered, latency statistics must be None (never 0.0)
    lat_summary = compute_latency_summary(latencies_s, empty_as_none=True)

    bw_status = f"{total_bytes_transmitted:,} bytes" if total_bytes_transmitted is not None else "NOT MEASURED"

    return {
        "total_transmissions": int(total_transmissions),
        "delivered_transmissions": delivered,
        "failed_transmissions": failed,
        "packet_loss_count": lost,
        "delivery_rate": float(delivered / tot),
        "failure_rate": float(failed / tot),
        "packet_loss_rate": float(lost / tot),
        "simulated_network_latency_ms": lat_summary,
        "bandwidth_usage": bw_status,
        "network_simulation_note": "SIMULATED NETWORK ONLY — NOT PHYSICAL HARDWARE OR INTERNET MEASUREMENTS",
    }


def get_unmeasured_system_status(bytes_measured: int | None = None) -> dict[str, str]:
    """Return explicit integrity status for unmeasured or simulated hardware quantities.

    Enforces that CPU, RAM, Energy, Physical Hardware, and Formal Constraints are never
    fabricated or estimated when uninstrumented.
    """
    return {
        "cpu_utilization": "NOT MEASURED",
        "ram_utilization": "NOT MEASURED",
        "energy_consumption": "NOT MEASURED",
        "physical_hardware_deployment": "NOT MEASURED / SIMULATED RUNTIME ONLY",
        "bandwidth": f"{bytes_measured:,} bytes" if bytes_measured is not None else "NOT MEASURED",
        "formal_constraint_satisfaction": "NOT IMPLEMENTED / NOT MEASURED",
    }


def get_metric_completeness_matrix(observed_flags: Mapping[str, bool] | None = None) -> dict[str, str]:
    """Generate the complete IEEE results completeness matrix required by Section 31.I.

    Each metric maps strictly to its scientific status:
    - MEASURED
    - NOT MEASURED
    - NOT OBSERVED
    - NOT TRIGGERED
    """
    flags = dict(observed_flags or {})

    matrix = {
        "Accuracy": "MEASURED",
        "Precision": "MEASURED",
        "Recall": "MEASURED",
        "F1": "MEASURED",
        "MCC": "MEASURED",
        "Drift delay": "MEASURED" if flags.get("drift_detected", True) else "NOT OBSERVED",
        "Reliability": "MEASURED",
        "Routing": "MEASURED",
        "Hybrid fallback": "MEASURED" if flags.get("hybrid_fallback_observed", False) else "NOT OBSERVED",
        "Edge latency": "MEASURED",
        "Cloud latency": "MEASURED",
        "Network latency": "MEASURED (SIMULATED)",
        "Hybrid latency": "MEASURED" if flags.get("hybrid_observed", True) else "NOT OBSERVED",
        "Offloading ratio": "MEASURED",
        "Packet loss": "MEASURED" if flags.get("packet_loss_observed", False) else "NOT OBSERVED",
        "Adaptation": "MEASURED" if flags.get("adaptation_triggered", False) else "NOT TRIGGERED",
        "CPU": "NOT MEASURED",
        "RAM": "NOT MEASURED",
        "Energy": "NOT MEASURED",
        "Physical hardware": "NOT MEASURED / SIMULATED ONLY",
        "Bandwidth": "MEASURED" if flags.get("bytes_recorded", False) else "NOT MEASURED",
        "Constraint satisfaction": "NOT IMPLEMENTED / NOT MEASURED",
    }
    return matrix
