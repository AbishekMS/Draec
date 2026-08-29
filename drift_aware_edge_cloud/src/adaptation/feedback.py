"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/adaptation/feedback.py
Phase    : Phase 9
Status   : IMPLEMENTED

Causal delayed-feedback queue and feedback record management.
Enforces memory boundedness and strict isolation of evaluation data (test1).
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

from src.adaptation.base import FeedbackRecord


class FeedbackQueue:
    """Bounded, causally managed feedback buffer for delayed streaming labels.

    Enforces:
    1. Acausal arrival prevention (arrival_index >= observation_index).
    2. Future feedback leakage prevention (eligible only if arrival_index <= current_index).
    3. Strict evaluation stream quarantine (test1 observations rejected).
    4. Bounded heap consumption (FIFO eviction at max_size).
    """

    def __init__(self, max_size: int = 5000) -> None:
        self.max_size = max(1, int(max_size))
        self._records: dict[int, FeedbackRecord] = {}
        self._order: deque[int] = deque()
        self._total_recorded = 0
        self._total_labeled = 0

    def record_prediction(
        self,
        observation_index: int,
        features: Any,
        prediction: int | None,
        probabilities: dict[int, float] | None,
        model_version: str,
        source: str = "adaptation",
    ) -> FeedbackRecord:
        """Register a new streaming inference prediction awaiting delayed ground truth."""
        src_lower = str(source).strip().lower()
        if "test1" in src_lower:
            raise ValueError(
                f"Data contamination guard: stream source '{source}' cannot enter adaptation feedback queue. "
                f"test1 evaluation data is strictly quarantined."
            )

        rec = FeedbackRecord(
            observation_index=int(observation_index),
            features=features,
            prediction=prediction,
            probabilities=probabilities,
            model_version=str(model_version),
            label=None,
            arrival_index=None,
            is_labeled=False,
            source=source,
        )

        idx = int(observation_index)
        if idx in self._records:
            self._records[idx] = rec
            return rec

        # Bounded eviction
        while len(self._order) >= self.max_size:
            evicted_idx = self._order.popleft()
            self._records.pop(evicted_idx, None)

        self._records[idx] = rec
        self._order.append(idx)
        self._total_recorded += 1
        return rec

    def provide_feedback(
        self,
        observation_index: int,
        label: int,
        arrival_index: int,
    ) -> FeedbackRecord:
        """Attach delayed ground truth label to a previously recorded prediction."""
        idx = int(observation_index)
        if idx not in self._records:
            raise KeyError(
                f"Observation index {idx} not found in feedback queue (may have been evicted or never recorded)."
            )

        rec = self._records[idx]
        if arrival_index < rec.observation_index:
            raise ValueError(
                f"Acausal feedback arrival: arrival_index {arrival_index} < observation_index {rec.observation_index}."
            )

        labeled_rec = rec.with_label(label=int(label), arrival_index=int(arrival_index))
        self._records[idx] = labeled_rec
        self._total_labeled += 1
        return labeled_rec

    def get_eligible_feedback(
        self,
        current_index: int | None = None,
        max_samples: int | None = None,
    ) -> list[FeedbackRecord]:
        """Return causally eligible labeled feedback records.

        A record is eligible if and only if:
        1. It has received a ground truth label (is_labeled == True).
        2. Its feedback arrival occurred on or before current_index (if specified).
        """
        eligible: list[FeedbackRecord] = []
        for idx in self._order:
            rec = self._records.get(idx)
            if rec is None or not rec.is_labeled:
                continue
            if current_index is not None and rec.arrival_index is not None:
                if rec.arrival_index > current_index:
                    # Feedback arrived in the future relative to current_index -> NOT eligible!
                    continue
            eligible.append(rec)

        if max_samples is not None and max_samples > 0:
            return eligible[-max_samples:]
        return eligible

    def count_eligible(self, current_index: int | None = None) -> int:
        """Return the number of causally eligible labeled feedback records."""
        return len(self.get_eligible_feedback(current_index=current_index))

    def get_pending_count(self) -> int:
        """Return the number of pending predictions that have not received labels."""
        return sum(1 for r in self._records.values() if not r.is_labeled)

    def get_stats(self) -> dict[str, Any]:
        """Return cumulative feedback queue statistics."""
        return {
            "total_recorded": self._total_recorded,
            "total_labeled": self._total_labeled,
            "current_buffer_size": len(self._records),
            "pending_count": self.get_pending_count(),
            "max_size": self.max_size,
        }

    def clear(self) -> None:
        """Clear all records from the queue."""
        self._records.clear()
        self._order.clear()

    def reset(self) -> None:
        """Reset records and counters."""
        self.clear()
        self._total_recorded = 0
        self._total_labeled = 0
