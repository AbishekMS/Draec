"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/utils/logger.py
Phase    : Phase 1 / Step 5
Status   : IMPLEMENTED

Structured event logging (CSV-backed).

Two jobs, kept separate on purpose
----------------------------------
`configure()` sets up human-readable stdlib logging at `output.log_level`.
`EventLog` writes machine-readable CSV rows with a declared, fixed schema.

Why the schema is declared up front and enforced per row: a results table is only
analysable if every row has the same columns in the same order. A logger that
accepts arbitrary keyword arguments produces a ragged CSV that has to be repaired
by hand later, and the repair is where numbers get quietly dropped.

Integrity boundary -- this module is WRITE-ONLY
----------------------------------------------
Event logs are output artefacts. Nothing in the pipeline reads one back during a
run, which is what allows drift ground truth to be recorded in an evaluation log
without becoming an input to the detector, the reliability estimator, or the
controller. `EventLog` therefore exposes no read method at all: the absence is
the enforcement. Ground-truth fields must be logged only from evaluation-side
code, and are marked as such by `GROUND_TRUTH_FIELDS`.
"""

from __future__ import annotations

import csv
import io
import logging
import sys
from dataclasses import dataclass, field
from pathlib import Path
from types import TracebackType
from typing import Any, Iterable, Mapping, Sequence

_CONFIGURED = False

LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")

# Field names that carry drift ground truth. Permitted in an evaluation log,
# never in a log that any component consumes -- and no component can consume any
# log, because EventLog cannot be read. Listed so that a future reviewer can
# grep for them.
GROUND_TRUTH_FIELDS = frozenset({
    "scenario", "drift_start_index", "drift_end_index", "affected_features",
    "drift_magnitude", "realised_magnitude", "random_seed",
})

# Schemas used in Phase 1. A schema is a tuple, so column order is fixed.
SCHEMAS: dict[str, tuple[str, ...]] = {
    "stream": ("window_id", "start_index", "end_index", "start_time", "end_time",
               "n_rows", "contiguous", "valid_fraction"),
    "quality": ("stage", "n_rows", "n_valid", "n_validation_failed",
                "n_range_violation", "n_outlier", "n_filled", "n_unfilled",
                "note"),
    "features": ("window_id", "start_index", "end_index", "n_features",
                 "n_outlier_flags", "n_range_flags", "valid_fraction"),
    "run": ("event", "config_name", "config_fingerprint", "master_seed",
            "detail"),
}


class LoggingError(RuntimeError):
    """Raised when a log cannot be written honestly."""


# -----------------------------------------------------------------------------
# Human-readable logging
# -----------------------------------------------------------------------------


def configure(
    config: Mapping[str, Any] | None = None,
    *,
    level: str | None = None,
    stream: Any = None,
    force: bool = False,
) -> logging.Logger:
    """Configure the root project logger once per process."""
    global _CONFIGURED
    lvl = level or str(((config or {}).get("output") or {}).get("log_level", "INFO"))
    lvl = lvl.upper()
    if lvl not in LEVELS:
        raise LoggingError(f"log level {lvl!r} is not one of {LEVELS}")
    root = logging.getLogger("dace")
    if _CONFIGURED and not force:
        root.setLevel(lvl)
        return root
    for h in list(root.handlers):
        root.removeHandler(h)
    handler = logging.StreamHandler(stream or sys.stderr)
    handler.setFormatter(
        logging.Formatter("%(asctime)s %(levelname)-8s %(name)s: %(message)s",
                          datefmt="%Y-%m-%d %H:%M:%S")
    )
    root.addHandler(handler)
    root.setLevel(lvl)
    root.propagate = False
    _CONFIGURED = True
    return root


def get_logger(name: str) -> logging.Logger:
    """A child logger under the project root. Safe before `configure()`."""
    return logging.getLogger(f"dace.{name}" if not name.startswith("dace") else name)


# -----------------------------------------------------------------------------
# Machine-readable event log
# -----------------------------------------------------------------------------


@dataclass
class EventLog:
    """Append-only CSV writer with a fixed, declared schema.

    Deliberately has no read method: see the module docstring on the write-only
    integrity boundary.
    """

    path: Path
    fields: tuple[str, ...]
    name: str = "event"
    flush_every: int = 200
    n_written: int = 0
    _fh: Any = field(default=None, repr=False)
    _writer: Any = field(default=None, repr=False)
    _since_flush: int = field(default=0, repr=False)

    @classmethod
    def create(
        cls,
        name: str,
        *,
        config: Mapping[str, Any] | None = None,
        root: Path | str = ".",
        fields: Sequence[str] | None = None,
        directory: Path | str | None = None,
        suffix: str = "",
        overwrite: bool = True,
    ) -> "EventLog":
        """Open a log named after a declared schema (or an explicit field list)."""
        if fields is None:
            if name not in SCHEMAS:
                raise LoggingError(
                    f"no declared schema for {name!r}. Known: {sorted(SCHEMAS)}. "
                    f"Pass `fields=` explicitly, or add the schema to SCHEMAS so "
                    f"the column order is recorded in source."
                )
            cols = SCHEMAS[name]
        else:
            cols = tuple(fields)
            if len(set(cols)) != len(cols):
                raise LoggingError(f"duplicate field name in schema: {cols}")
        if not cols:
            raise LoggingError("a schema must declare at least one field")
        if directory is None:
            results = str(((config or {}).get("output") or {})
                          .get("results_dir", "results"))
            directory = Path(root) / results
        d = Path(directory)
        d.mkdir(parents=True, exist_ok=True)
        path = d / f"{name}{suffix}.csv"
        if path.exists() and not overwrite:
            raise LoggingError(f"{path} exists and overwrite is false")
        log = cls(path=path, fields=cols, name=name)
        log._open()
        return log

    def _open(self) -> None:
        self._fh = io.open(self.path, "w", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(self._fh, fieldnames=list(self.fields),
                                      extrasaction="raise")
        self._writer.writeheader()

    def write(self, **row: Any) -> None:
        """Write one row. Unknown or missing fields are errors, not defaults."""
        if self._writer is None:
            raise LoggingError(f"{self.path.name}: log is closed")
        unknown = [k for k in row if k not in self.fields]
        if unknown:
            raise LoggingError(
                f"{self.name}: field(s) {unknown} are not in the declared schema "
                f"{list(self.fields)}. A ragged CSV cannot be analysed."
            )
        missing = [k for k in self.fields if k not in row]
        if missing:
            raise LoggingError(
                f"{self.name}: field(s) {missing} absent. Pass an explicit value "
                f"(None renders as empty); a silent default would be indistinguishable "
                f"from a measured zero."
            )
        self._writer.writerow({k: _render(row[k]) for k in self.fields})
        self.n_written += 1
        self._since_flush += 1
        if self._since_flush >= self.flush_every:
            self._fh.flush()
            self._since_flush = 0

    def write_many(self, rows: Iterable[Mapping[str, Any]]) -> int:
        n = 0
        for r in rows:
            self.write(**dict(r))
            n += 1
        return n

    def close(self) -> None:
        if self._fh is not None:
            self._fh.flush()
            self._fh.close()
        self._fh = None
        self._writer = None

    def __enter__(self) -> "EventLog":
        return self

    def __exit__(self, exc_type: type[BaseException] | None,
                 exc: BaseException | None, tb: TracebackType | None) -> None:
        self.close()


def _render(value: Any) -> Any:
    """Render one cell. Sequences become `|`-joined so a CSV stays one row."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, (list, tuple, set, frozenset)):
        return "|".join(str(v) for v in value)
    return value


def log_quality(log: EventLog, stage: str, quality: Any, *, note: str = "") -> None:
    """Record a preprocessing QualityReport as one row of the `quality` schema."""
    log.write(
        stage=stage,
        n_rows=quality.n_rows,
        n_valid=int(quality.valid.sum()),
        n_validation_failed=int(quality.validation_failed.sum()),
        n_range_violation=int(quality.range_violation.sum()),
        n_outlier=int(quality.outlier.sum()),
        n_filled=int(quality.filled.sum()),
        n_unfilled=int(quality.unfilled.sum()),
        note=note,
    )
