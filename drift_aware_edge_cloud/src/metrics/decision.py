"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/metrics/decision.py
Phase    : Phase 10
Status   : IMPLEMENTED

Routing distribution, offloading ratio, controller switching, and Hybrid execution metrics.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np


def compute_routing_metrics(
    actions: Sequence[str],
    switch_count: int | None = None,
    hybrid_fallbacks: int = 0,
) -> dict[str, Any]:
    """Compute routing distribution and orchestration metrics.

    Parameters:
    - actions: Sequence of selected routing actions ('EDGE', 'CLOUD', 'HYBRID').
    - switch_count: Count of controller switching events (or computed from transitions if None).
    - hybrid_fallbacks: Count of hybrid decisions that triggered Cloud fallback.

    Returns:
    Dictionary containing counts, percentages, offloading ratio, and hybrid fallback rate.
    """
    total = len(actions)
    if total == 0:
        return {
            "total_decisions": 0,
            "edge_count": 0,
            "cloud_count": 0,
            "hybrid_count": 0,
            "edge_percentage": 0.0,
            "cloud_percentage": 0.0,
            "hybrid_percentage": 0.0,
            "offloading_ratio": 0.0,
            "switch_count": 0,
            "hybrid_fallback_count": 0,
            "edge_completed_hybrid_count": 0,
            "hybrid_fallback_rate": 0.0,
            "hybrid_status": "NOT OBSERVED",
        }

    acts = [str(a).upper() for a in actions]
    n_edge = sum(1 for a in acts if a == "EDGE")
    n_cloud = sum(1 for a in acts if a == "CLOUD")
    n_hybrid = sum(1 for a in acts if a == "HYBRID")

    pct_edge = float((n_edge / total) * 100.0)
    pct_cloud = float((n_cloud / total) * 100.0)
    pct_hybrid = float((n_hybrid / total) * 100.0)

    # Offloading ratio: Cloud-only decisions divided by total
    offload_ratio = float((n_cloud / total) * 100.0)

    # Compute switches if not explicitly provided
    if switch_count is None:
        switches = sum(1 for i in range(1, total) if acts[i] != acts[i - 1])
    else:
        switches = int(switch_count)

    # Hybrid fallback accounting
    n_fallback = int(hybrid_fallbacks)
    if n_hybrid > 0:
        fallback_rate = float((n_fallback / n_hybrid) * 100.0)
        edge_completed = max(0, n_hybrid - n_fallback)
        hybrid_status = "OBSERVED"
    else:
        fallback_rate = 0.0
        edge_completed = 0
        hybrid_status = "NOT OBSERVED"

    return {
        "total_decisions": total,
        "edge_count": n_edge,
        "cloud_count": n_cloud,
        "hybrid_count": n_hybrid,
        "edge_percentage": pct_edge,
        "cloud_percentage": pct_cloud,
        "hybrid_percentage": pct_hybrid,
        "offloading_ratio": offload_ratio,
        "switch_count": switches,
        "hybrid_fallback_count": n_fallback,
        "edge_completed_hybrid_count": edge_completed,
        "hybrid_fallback_rate": fallback_rate,
        "hybrid_status": hybrid_status,
    }
