"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/drift/persistence.py
Phase    : Phase 3
Status   : IMPLEMENTED

Drift persistence tracking mechanism.
Distinguishes isolated transient alarms from persistent drift based on configurable
consecutive streak or rolling-window recurrence criteria.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Deque, Mapping


class DriftPersistence:
    """Configurable persistence evaluator for drift alarms.

    A single statistical change alarm from ADWIN does not necessarily indicate
    sustained regime change. DriftPersistence tracks alarm occurrences over time to
    confirm persistent drift before downstream components (e.g. retraining or
    adaptation in later phases) are triggered.

    Supported criteria:
    - 'consecutive': requires K consecutive drift detections. If an observation
      reports no drift, the consecutive streak resets to 0.
    - 'windowed_count': requires at least N detections within the last T observations.
    """

    def __init__(
        self,
        config: Mapping[str, Any] | None = None,
        *,
        criterion: str = "consecutive",
        consecutive_threshold: int = 3,
        window_size: int = 10,
        count_threshold: int = 3,
        **kwargs: Any,
    ) -> None:
        cfg_params: dict[str, Any] = {}
        if config is not None:
            dd_sec = config.get("drift_detection")
            if not isinstance(dd_sec, Mapping):
                drift_sec = config.get("drift")
                if isinstance(drift_sec, Mapping):
                    dd_sec = drift_sec.get("detection") or drift_sec.get("drift_detection")
            if isinstance(dd_sec, Mapping):
                persist_sec = dd_sec.get("persistence")
                if isinstance(persist_sec, Mapping):
                    cfg_params.update(persist_sec)

        self._criterion = str(kwargs.get("criterion", cfg_params.get("criterion", criterion)))
        self._consecutive_threshold = int(
            kwargs.get("consecutive_threshold", cfg_params.get("consecutive_threshold", consecutive_threshold))
        )
        self._window_size = int(kwargs.get("window_size", cfg_params.get("window_size", window_size)))
        self._count_threshold = int(
            kwargs.get("count_threshold", cfg_params.get("count_threshold", count_threshold))
        )

        if self._criterion not in ("consecutive", "windowed_count", "n_in_t"):
            raise ValueError(f"Unknown persistence criterion: {self._criterion!r}")
        if self._consecutive_threshold < 1:
            raise ValueError(f"consecutive_threshold must be >= 1, got {self._consecutive_threshold}")
        if self._window_size < 1:
            raise ValueError(f"window_size must be >= 1, got {self._window_size}")
        if self._count_threshold < 1:
            raise ValueError(f"count_threshold must be >= 1, got {self._count_threshold}")

        self._current_streak: int = 0
        self._total_alarms: int = 0
        self._total_updates: int = 0
        self._window_history: Deque[bool] = deque(maxlen=self._window_size)
        self._is_persistent: bool = False

    @property
    def criterion(self) -> str:
        return self._criterion

    @property
    def consecutive_threshold(self) -> int:
        return self._consecutive_threshold

    @property
    def window_size(self) -> int:
        return self._window_size

    @property
    def count_threshold(self) -> int:
        return self._count_threshold

    @property
    def is_persistent(self) -> bool:
        """True if the current evidence satisfies the persistence criterion."""
        return self._is_persistent

    @property
    def current_streak(self) -> int:
        """Current consecutive drift alarm count."""
        return self._current_streak

    @property
    def total_alarms(self) -> int:
        """Total number of drift alarms passed to update()."""
        return self._total_alarms

    @property
    def total_updates(self) -> int:
        """Total number of update calls made."""
        return self._total_updates

    @property
    def alarm_rate(self) -> float:
        """Fraction of update calls that were alarms."""
        if self._total_updates == 0:
            return 0.0
        return self._total_alarms / self._total_updates

    def update(self, drift_detected: bool) -> bool:
        """Update persistence state with the latest ADWIN drift detection indicator.

        Parameters
        ----------
        drift_detected : bool
            Whether drift was detected on the current observation.

        Returns
        -------
        bool
            True if the drift evidence is deemed persistent.
        """
        is_alarm = bool(drift_detected)
        self._total_updates += 1
        if is_alarm:
            self._total_alarms += 1
            self._current_streak += 1
        else:
            self._current_streak = 0

        self._window_history.append(is_alarm)

        if self._criterion == "consecutive":
            self._is_persistent = self._current_streak >= self._consecutive_threshold
        elif self._criterion in ("windowed_count", "n_in_t"):
            alarms_in_window = sum(self._window_history)
            self._is_persistent = alarms_in_window >= self._count_threshold

        return self._is_persistent

    def reset(self) -> None:
        """Reset internal streak, window history, and persistence status."""
        self._current_streak = 0
        self._total_alarms = 0
        self._total_updates = 0
        self._window_history.clear()
        self._is_persistent = False

    def get_info(self) -> dict[str, Any]:
        """Return introspection, diagnostic, and state metadata."""
        return {
            "criterion": self._criterion,
            "is_persistent": self._is_persistent,
            "current_streak": self._current_streak,
            "consecutive_threshold": self._consecutive_threshold,
            "window_size": self._window_size,
            "count_threshold": self._count_threshold,
            "alarms_in_window": sum(self._window_history),
            "total_alarms": self._total_alarms,
            "total_updates": self._total_updates,
            "alarm_rate": self.alarm_rate,
        }
