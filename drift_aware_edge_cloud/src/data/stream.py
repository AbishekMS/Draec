"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/data/stream.py
Phase    : Phase 1 / Step 3
Status   : IMPLEMENTED

Chronological streaming engine: t1 -> t2 -> ... -> tn.
MUST NOT shuffle, reorder, or look ahead.

What this module is
-------------------
The single arrival mechanism for the whole simulation. Everything downstream --
Edge prediction, prediction error, ADWIN, reliability, LRI, WDS, the controller,
adaptation -- sees data only through the iterators defined here. Confining the
time axis to one module is what makes the causality claim auditable: there is
exactly one place where "what has arrived so far" is defined.

Guarantees, enforced at runtime rather than promised in prose
------------------------------------------------------------
* `streaming.shuffle: true` raises. A shuffled stream is not a stream.
* Emitted row indices are strictly increasing; emitted timestamps are
  non-decreasing. Both are asserted on every emission, not sampled.
* No window is emitted before its final row has arrived. A window covering rows
  [s, e) is emitted at cursor position e-1, never earlier.
* Nothing outside [0, cursor] is read. There is no code path that computes a
  statistic over the full frame.

What this module deliberately does NOT do
-----------------------------------------
No imputation, no normalization, no filtering, and -- importantly -- no feature
extraction. `preprocessing.windowing.feature_extraction` (mean/std/slope/mode/
n_changes) is Step 4's responsibility. Windows here carry RAW row slices. Putting
aggregation here would make it impossible to tell whether a downstream statistic
was fitted causally, because two modules would be computing statistics.

Trailing partial windows are dropped, not zero-padded. Padding would inject
fabricated rows into the evaluation region.
"""

from __future__ import annotations

import time
import warnings
from dataclasses import dataclass, field
from typing import Any, Iterator, Mapping, Protocol, Sequence

import numpy as np
import pandas as pd

from src.data.loader import ConfigError, DataError


class StreamOrderError(DataError):
    """Raised when an emission would violate chronological order."""


class StreamSource(Protocol):
    """What the streaming engine needs from its input.

    Satisfied by both `loader.LoadedFile` and `generator.DriftedStream`, so a
    clean stream and a drifted stream are consumed by identical code. If the two
    took different paths, a scenario difference could be an artefact of the
    reader rather than of the drift.
    """

    key: str
    role: str
    frame: pd.DataFrame
    timestamps: pd.Series
    block_id: pd.Series


# -----------------------------------------------------------------------------
# Emitted units
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class Sample:
    """One arriving observation."""

    index: int
    timestamp: pd.Timestamp
    block_id: int
    values: pd.Series

    def __getitem__(self, column: str) -> Any:
        return self.values[column]


@dataclass(frozen=True)
class Window:
    """A contiguous raw slice, [start_index, end_index).

    `frame` is a slice view-copy of raw rows. No aggregation has been applied.
    """

    window_id: int
    start_index: int
    end_index: int
    start_time: pd.Timestamp
    end_time: pd.Timestamp
    block_ids: tuple[int, ...]
    n_rows: int
    contiguous: bool
    valid_fraction: float
    frame: pd.DataFrame

    def __len__(self) -> int:
        return self.n_rows

    @property
    def span_s(self) -> float:
        return float((self.end_time - self.start_time).total_seconds())


@dataclass
class StreamStats:
    """Counters, so anything skipped is reported instead of silently vanishing."""

    n_rows_available: int = 0
    n_samples_emitted: int = 0
    n_windows_candidate: int = 0
    n_windows_emitted: int = 0
    n_windows_skipped_noncontiguous: int = 0
    n_windows_skipped_invalid: int = 0
    n_rows_truncated_by_max_samples: int = 0
    trailing_rows_dropped: int = 0
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"rows available            : {self.n_rows_available:,}",
            f"samples emitted           : {self.n_samples_emitted:,}",
            f"windows emitted           : {self.n_windows_emitted:,}"
            f" of {self.n_windows_candidate:,} candidates",
            f"  skipped, non-contiguous : {self.n_windows_skipped_noncontiguous:,}",
            f"  skipped, low validity   : {self.n_windows_skipped_invalid:,}",
            f"trailing rows dropped     : {self.trailing_rows_dropped:,}",
        ]
        if self.n_rows_truncated_by_max_samples:
            lines.append(
                f"rows withheld by max_samples: "
                f"{self.n_rows_truncated_by_max_samples:,}"
            )
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


# -----------------------------------------------------------------------------
# Configuration resolution
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class StreamPlan:
    """Everything the iterators need, resolved and validated once, up front."""

    emit: str
    max_samples: int | None
    realtime_factor: float | None
    window_size: int
    step_size: int
    require_contiguous: bool
    min_valid_fraction: float
    sampling_interval_s: float

    @property
    def emits_samples(self) -> bool:
        return self.emit in {"samples", "both"}

    @property
    def emits_windows(self) -> bool:
        return self.emit in {"windows", "both"}


def plan(config: Mapping[str, Any], *, n_rows: int | None = None) -> StreamPlan:
    """Validate the streaming and windowing configuration."""
    st = config.get("streaming") or {}
    win = ((config.get("preprocessing") or {}).get("windowing")) or {}

    if st.get("shuffle", False):
        raise ConfigError(
            "streaming.shuffle is true. The research question concerns drift over "
            "time; shuffling destroys the time axis, makes drift undetectable in "
            "principle, and leaks post-drift rows into the pre-drift region."
        )

    emit = str(st.get("emit", "windows"))
    if emit not in {"windows", "samples", "both"}:
        raise ConfigError(
            f"streaming.emit must be 'windows', 'samples' or 'both', got {emit!r}"
        )

    max_samples = st.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        if max_samples <= 0:
            raise ConfigError(f"streaming.max_samples must be >= 1, got {max_samples}")

    rf = st.get("realtime_factor")
    if rf is not None:
        rf = float(rf)
        if rf <= 0:
            raise ConfigError(f"streaming.realtime_factor must be > 0, got {rf}")

    window_size = int(win.get("window_size", 0))
    step_size = int(win.get("step_size", 0))
    if window_size <= 0:
        raise ConfigError("preprocessing.windowing.window_size must be >= 1")
    if step_size <= 0:
        raise ConfigError("preprocessing.windowing.step_size must be >= 1")
    if step_size > window_size:
        warnings.warn(
            f"step_size ({step_size}) exceeds window_size ({window_size}); "
            f"{step_size - window_size} row(s) between consecutive windows will "
            f"never be evaluated",
            RuntimeWarning,
            stacklevel=2,
        )

    mvf = float(win.get("min_valid_fraction", 1.0))
    if not 0.0 < mvf <= 1.0:
        raise ConfigError(
            f"preprocessing.windowing.min_valid_fraction must be in (0, 1], got {mvf}"
        )

    if n_rows is not None and n_rows < window_size:
        raise ConfigError(
            f"the stream has {n_rows} row(s) but window_size is {window_size}; "
            f"no window could ever be emitted"
        )

    interval = float((config.get("dataset") or {}).get("expected_sampling_interval_s", 1))

    return StreamPlan(
        emit=emit,
        max_samples=max_samples,
        realtime_factor=rf,
        window_size=window_size,
        step_size=step_size,
        require_contiguous=bool(win.get("require_contiguous", True)),
        min_valid_fraction=mvf,
        sampling_interval_s=interval,
    )


# -----------------------------------------------------------------------------
# The engine
# -----------------------------------------------------------------------------


class ChronologicalStream:
    """Replays a stream in recorded order, once, forwards.

    The cursor is the only notion of "now". `iter_windows` emits a window only
    when the cursor has reached its last row, so a consumer can never be handed
    a window built from rows that have not arrived yet.
    """

    def __init__(
        self,
        source: StreamSource,
        config: Mapping[str, Any],
        *,
        valid_mask: Sequence[bool] | np.ndarray | None = None,
        columns: Sequence[str] | None = None,
    ):
        self.source = source
        self.config = config
        self.frame = source.frame if columns is None else source.frame.loc[:, list(columns)]
        self.timestamps = pd.Series(source.timestamps).reset_index(drop=True)
        self.block_id = (
            pd.Series(source.block_id).reset_index(drop=True)
            if source.block_id is not None
            else pd.Series(np.zeros(len(self.frame), dtype=int))
        )
        self.plan = plan(config, n_rows=len(self.frame))
        self.stats = StreamStats(n_rows_available=len(self.frame))

        # Validity. In Step 3 nothing has been imputed yet, so validity means
        # "the raw row is complete". Step 4's pipeline will pass its own mask,
        # which then takes precedence -- this module never guesses.
        if valid_mask is not None:
            mask = np.asarray(valid_mask, dtype=bool)
            if mask.shape[0] != len(self.frame):
                raise ConfigError(
                    f"valid_mask has {mask.shape[0]} entries but the stream has "
                    f"{len(self.frame)} rows"
                )
            self._valid = mask
            self.stats.notes.append("validity taken from the caller-supplied mask")
        else:
            self._valid = self.frame.notna().all(axis=1).to_numpy(dtype=bool)
            self.stats.notes.append(
                "validity derived from raw completeness (no imputation has run yet)"
            )

        self._limit = len(self.frame)
        if self.plan.max_samples is not None and self.plan.max_samples < self._limit:
            self.stats.n_rows_truncated_by_max_samples = (
                self._limit - self.plan.max_samples
            )
            self._limit = self.plan.max_samples
            self.stats.notes.append(
                f"streaming.max_samples caps the replay at {self._limit:,} rows"
            )

        self._cursor = -1
        self._last_index = -1
        self._last_time: pd.Timestamp | None = None
        self._next_window_id = 0

    # -- order enforcement ----------------------------------------------------

    def _advance(self, index: int) -> None:
        """Move the cursor forward one row, asserting order on the way."""
        if index <= self._last_index:
            raise StreamOrderError(
                f"row index went backwards: {index} after {self._last_index}"
            )
        ts = self.timestamps.iloc[index]
        if self._last_time is not None and ts < self._last_time:
            raise StreamOrderError(
                f"timestamp went backwards at row {index}: {ts} after {self._last_time}"
            )
        self._last_index = index
        self._last_time = ts
        self._cursor = index

    def _pace(self) -> None:
        if self.plan.realtime_factor is not None:
            time.sleep(self.plan.sampling_interval_s / self.plan.realtime_factor)

    # -- windows --------------------------------------------------------------

    def _window_starts(self) -> range:
        last_start = self._limit - self.plan.window_size
        if last_start < 0:
            return range(0)
        return range(0, last_start + 1, self.plan.step_size)

    def _build(self, start: int) -> Window | None:
        """Assemble the window ending at `start + window_size`, or reject it.

        Reads rows [start, end) only. `end - 1 <= cursor` is asserted, which is
        the mechanical no-lookahead guarantee.
        """
        end = start + self.plan.window_size
        if end - 1 > self._cursor:
            raise StreamOrderError(
                f"window [{start}, {end}) would read row {end - 1} but the cursor "
                f"is at {self._cursor}: that is lookahead"
            )
        self.stats.n_windows_candidate += 1

        blocks = self.block_id.iloc[start:end].to_numpy()
        unique_blocks = tuple(int(b) for b in np.unique(blocks))
        contiguous = len(unique_blocks) == 1
        if self.plan.require_contiguous and not contiguous:
            self.stats.n_windows_skipped_noncontiguous += 1
            return None

        valid_fraction = float(self._valid[start:end].mean())
        if valid_fraction < self.plan.min_valid_fraction:
            self.stats.n_windows_skipped_invalid += 1
            return None

        w = Window(
            window_id=self._next_window_id,
            start_index=start,
            end_index=end,
            start_time=self.timestamps.iloc[start],
            end_time=self.timestamps.iloc[end - 1],
            block_ids=unique_blocks,
            n_rows=self.plan.window_size,
            contiguous=contiguous,
            valid_fraction=valid_fraction,
            frame=self.frame.iloc[start:end].copy(deep=False),
        )
        self._next_window_id += 1
        self.stats.n_windows_emitted += 1
        return w

    # -- public iterators -----------------------------------------------------

    def samples(self) -> Iterator[Sample]:
        """Yield one Sample per row, in recorded order."""
        for i in range(self._limit):
            self._advance(i)
            self.stats.n_samples_emitted += 1
            self._pace()
            yield Sample(
                index=i,
                timestamp=self.timestamps.iloc[i],
                block_id=int(self.block_id.iloc[i]),
                values=self.frame.iloc[i],
            )

    def windows(self) -> Iterator[Window]:
        """Yield raw windows as soon as -- and not before -- they complete."""
        starts = self._window_starts()
        if not starts:
            return
        pending = iter(starts)
        nxt = next(pending, None)
        for i in range(self._limit):
            self._advance(i)
            self._pace()
            while nxt is not None and nxt + self.plan.window_size - 1 == i:
                w = self._build(nxt)
                nxt = next(pending, None)
                if w is not None:
                    yield w
        self._record_trailing(starts)

    def events(self) -> Iterator[tuple[str, Sample | Window]]:
        """Yield ('sample', Sample) and ('window', Window) in arrival order.

        A window is emitted immediately after the sample that completes it, which
        is the correct simulation order: the row arrives, then the window it
        finishes becomes available.
        """
        starts = self._window_starts()
        pending = iter(starts)
        nxt = next(pending, None)
        for i in range(self._limit):
            self._advance(i)
            self.stats.n_samples_emitted += 1
            self._pace()
            yield "sample", Sample(
                index=i,
                timestamp=self.timestamps.iloc[i],
                block_id=int(self.block_id.iloc[i]),
                values=self.frame.iloc[i],
            )
            while nxt is not None and nxt + self.plan.window_size - 1 == i:
                w = self._build(nxt)
                nxt = next(pending, None)
                if w is not None:
                    yield "window", w
        self._record_trailing(starts)

    def run(self) -> Iterator[Any]:
        """Dispatch on `streaming.emit`."""
        if self.plan.emit == "samples":
            return self.samples()
        if self.plan.emit == "windows":
            return self.windows()
        return self.events()

    def _record_trailing(self, starts: range) -> None:
        if not starts:
            self.stats.trailing_rows_dropped = self._limit
            return
        covered = starts[-1] + self.plan.window_size
        self.stats.trailing_rows_dropped = max(0, self._limit - covered)


# -----------------------------------------------------------------------------
# Convenience wrappers
# -----------------------------------------------------------------------------


def iter_samples(
    source: StreamSource, config: Mapping[str, Any], **kwargs: Any
) -> Iterator[Sample]:
    return ChronologicalStream(source, config, **kwargs).samples()


def iter_windows(
    source: StreamSource, config: Mapping[str, Any], **kwargs: Any
) -> Iterator[Window]:
    return ChronologicalStream(source, config, **kwargs).windows()


def window_index(
    config: Mapping[str, Any], n_rows: int
) -> list[tuple[int, int]]:
    """The (start, end) index pairs a full replay of `n_rows` would consider.

    Evaluation-side helper: mapping a ground-truth drift row onto window ids
    needs the window grid, and recomputing it by hand invites an off-by-one.
    Pure arithmetic -- reads no data.
    """
    p = plan(config, n_rows=n_rows)
    return [(s, s + p.window_size) for s in range(0, max(0, n_rows - p.window_size) + 1, p.step_size)]


def aggregate_label(values: Sequence[Any], config: Mapping[str, Any]) -> Any:
    """Collapse a window's per-row labels to one window label.

    Kept here because it is a windowing decision, but only callable once a label
    column exists. `dataset.task` is still `unresolved`, so today every caller
    hits the ConfigError below -- deliberately, rather than a fabricated default.
    """
    win = ((config.get("preprocessing") or {}).get("windowing")) or {}
    mode = str(win.get("label_aggregation", "any_positive"))
    arr = np.asarray(values)
    if arr.size == 0:
        raise ConfigError("aggregate_label received an empty window")
    if mode == "any_positive":
        positive = (config.get("dataset") or {}).get("positive_class")
        if positive is None:
            raise ConfigError(
                "label_aggregation 'any_positive' needs dataset.positive_class, "
                "which is null because dataset.task is still 'unresolved'"
            )
        return positive if np.any(arr == positive) else arr[0]
    if mode == "majority":
        labels, counts = np.unique(arr, return_counts=True)
        return labels[int(np.argmax(counts))]
    if mode == "last":
        return arr[-1]
    raise ConfigError(
        f"preprocessing.windowing.label_aggregation must be 'any_positive', "
        f"'majority' or 'last', got {mode!r}"
    )
