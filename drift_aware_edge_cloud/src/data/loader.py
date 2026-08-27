"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/data/loader.py
Phase    : Phase 1 / Step 3
Status   : IMPLEMENTED

Load the real HAI files. Every format decision -- delimiter, timestamp column
and format, feature selection, column types, file roles -- is driven by
configuration (config/*.yaml). Nothing here is hard-coded to HAI.

MUST NOT fabricate or substitute HAI data. This module only reads.

Responsibilities
----------------
1. Resolve `dataset.files` into file specs, honouring their declared ROLES.
2. Load ONE file at a time. Never concatenate (dataset.concatenate_files).
3. Validate the schema and the time axis, and report -- not silently repair.
4. Enforce CAUSALITY: the baseline file must precede the inference stream.
5. Profile the BASELINE ONLY: per-column mean/std/range/cardinality, and the
   continuous-vs-discrete split. Downstream components (drift injection in
   generator.py, normalization in preprocessing.py) take their statistics from
   this profile so there is exactly one place they can come from.
6. Refuse to invent a prediction target while `dataset.task` is unresolved.

LEAKAGE boundaries enforced here
--------------------------------
* Baseline statistics are computed from the baseline file only. There is no
  code path that computes them over the inference stream or over all files.
* A baseline whose time range does not precede the inference stream is rejected
  unless `dataset.allow_acausal_baseline` is explicitly true.
* Zero-variance detection and column typing use the baseline only. Dropping a
  column because it happens to be constant in the TEST file would be leakage.

Nothing in this module imputes, normalizes, filters, or windows. That is
preprocessing.py (Step 4), so that the raw-versus-processed boundary stays
auditable.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

# -----------------------------------------------------------------------------
# Errors
# -----------------------------------------------------------------------------
# Distinct types, because these failures have genuinely different meanings and
# the tests need to assert on which one occurred. `src/utils/` is Step 5, so
# they live here rather than in a module that does not exist yet.


class DataError(RuntimeError):
    """Base class for every failure raised by the data layer."""


class ConfigError(DataError):
    """The configuration is internally inconsistent or requests the impossible."""


class SchemaError(DataError):
    """A file does not match what the configuration says it should contain."""


class TimeAxisError(DataError):
    """The time axis violates a requirement (order, duplicates, interval)."""


class CausalityError(DataError):
    """An operation would use information unavailable at inference time."""


class UnresolvedTaskError(DataError):
    """A prediction target was requested while `dataset.task` is unresolved."""


# -----------------------------------------------------------------------------
# Value objects
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class FileSpec:
    """One entry of `dataset.files`, with its declared (VERIFIED) properties."""

    key: str
    path: Path
    role: str
    declared_rows: int | None = None
    declared_time_range: tuple[str, str] | None = None


@dataclass(frozen=True)
class TimeAxisReport:
    """Measured properties of a file's time axis. Facts, not assumptions."""

    n_rows: int
    first: pd.Timestamp
    last: pd.Timestamp
    monotonic_increasing: bool
    n_duplicate_timestamps: int
    modal_interval_s: float
    n_gaps: int
    max_gap_s: float
    n_blocks: int

    def summary(self) -> str:
        return (
            f"{self.n_rows:,} rows  "
            f"{self.first} -> {self.last}  "
            f"interval={self.modal_interval_s:g}s  "
            f"gaps={self.n_gaps}  dup_ts={self.n_duplicate_timestamps}  "
            f"blocks={self.n_blocks}"
        )


@dataclass(frozen=True)
class ValidationReport:
    """Outcome of the structural checks. Findings are recorded, not repaired."""

    key: str
    path: Path
    n_rows: int
    n_columns: int
    n_missing_cells: int
    missing_by_column: dict[str, int]
    non_numeric_columns: tuple[str, ...]
    row_count_matches_config: bool | None
    time_range_matches_config: bool | None
    time_axis: TimeAxisReport
    findings: tuple[str, ...] = ()

    @property
    def clean(self) -> bool:
        return not self.findings


@dataclass(frozen=True)
class LoadedFile:
    """A single raw file, loaded and validated. `frame` excludes the timestamp."""

    key: str
    role: str
    path: Path
    frame: pd.DataFrame          # process variables only, original order
    timestamps: pd.Series        # parsed, aligned to frame.index
    block_id: pd.Series          # contiguity block, for gap-aware windowing
    report: ValidationReport

    def __len__(self) -> int:
        return len(self.frame)


@dataclass(frozen=True)
class ColumnProfile:
    """Per-column baseline statistics. The ONLY sanctioned source of sigma."""

    name: str
    n_unique: int
    is_discrete: bool
    mean: float
    std: float
    minimum: float
    maximum: float

    @property
    def value_range(self) -> float:
        return self.maximum - self.minimum

    @property
    def zero_variance(self) -> bool:
        return self.n_unique <= 1


@dataclass(frozen=True)
class BaselineProfile:
    """Statistics measured on the BASELINE file(s) only, then frozen.

    LEAKAGE: constructed exclusively from files whose role is a baseline role.
    `profile_baseline` is the only constructor and it refuses any other input.
    """

    source_keys: tuple[str, ...]
    n_rows: int
    columns: dict[str, ColumnProfile]
    feature_names: tuple[str, ...]
    continuous: tuple[str, ...]
    discrete: tuple[str, ...]
    dropped_zero_variance: tuple[str, ...]
    zero_variance_agreement: dict[str, Any] = field(default_factory=dict)

    def sigma(self, column: str) -> float:
        """Baseline standard deviation of `column`.

        A column dropped as zero-variance keeps its measured profile in
        `columns` -- the measurement is a fact worth retaining -- but its sigma
        is 0.0, which is not a usable scale: any caller dividing by it gets inf
        or nan. So this refuses, rather than handing back a number that silently
        poisons a magnitude calculation.
        """
        if column in self.dropped_zero_variance:
            raise ConfigError(
                f"{column!r} was dropped as zero-variance on the baseline "
                f"({self.source_keys}); its sigma is 0.0 and cannot be used as a "
                f"scale. Its measured profile is retained in `columns` for "
                f"reference only."
            )
        try:
            return self.columns[column].std
        except KeyError:  # pragma: no cover - defensive
            raise ConfigError(
                f"no baseline profile for column {column!r}; "
                f"it is not present in the baseline file"
            ) from None

    def top_variance(self, k: int, *, continuous_only: bool = True) -> tuple[str, ...]:
        """The k highest-variance columns, deterministic on ties (name order).

        Used by generator.py to place drift on informative channels. Ranking by
        baseline sigma -- not by sigma over the drifted stream, which would be
        circular.
        """
        pool = self.continuous if continuous_only else self.feature_names
        ranked = sorted(pool, key=lambda c: (-self.columns[c].std, c))
        return tuple(ranked[:k])


# -----------------------------------------------------------------------------
# Config access helpers
# -----------------------------------------------------------------------------
# `config` is an already-resolved mapping (after `_extends` merging). Resolution
# itself is src/utils/config.py, which is Step 5; this module deliberately does
# not reach for it, so Step 3 can be verified on its own.

BASELINE_ROLES = frozenset({"baseline_train", "baseline_validation"})
INFERENCE_ROLE = "inference_stream"


def _section(config: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    sec = config.get(name)
    if not isinstance(sec, Mapping):
        raise ConfigError(f"config section {name!r} is missing or not a mapping")
    return sec


def file_specs(config: Mapping[str, Any], root: Path | str = ".") -> dict[str, FileSpec]:
    """Resolve `dataset.files` into FileSpecs. Roles come from config only."""
    root = Path(root)
    files = _section(config, "dataset").get("files")
    if not isinstance(files, Mapping) or not files:
        raise ConfigError("dataset.files is missing or empty")

    specs: dict[str, FileSpec] = {}
    for key, entry in files.items():
        if not isinstance(entry, Mapping):
            raise ConfigError(f"dataset.files.{key} is not a mapping")
        for required in ("path", "role"):
            if not entry.get(required):
                raise ConfigError(f"dataset.files.{key}.{required} is missing")
        tr = entry.get("time_range")
        specs[key] = FileSpec(
            key=key,
            path=root / str(entry["path"]),
            role=str(entry["role"]),
            declared_rows=entry.get("rows"),
            declared_time_range=(str(tr[0]), str(tr[1])) if tr else None,
        )

    roles = [s.role for s in specs.values()]
    if roles.count(INFERENCE_ROLE) != 1:
        raise ConfigError(
            f"exactly one file must have role {INFERENCE_ROLE!r}; "
            f"found {roles.count(INFERENCE_ROLE)}"
        )
    unknown = {r for r in roles if r not in BASELINE_ROLES and r != INFERENCE_ROLE}
    if unknown:
        raise ConfigError(f"unrecognised file role(s): {sorted(unknown)}")
    return specs


def inference_key(config: Mapping[str, Any]) -> str:
    """Key of the file whose role is `inference_stream`."""
    for key, spec in file_specs(config).items():
        if spec.role == INFERENCE_ROLE:
            return key
    raise ConfigError(f"no file has role {INFERENCE_ROLE!r}")  # pragma: no cover


def resolve_baseline_keys(config: Mapping[str, Any]) -> tuple[str, ...]:
    """Which file(s) the baseline is drawn from, with the acausality guard.

    CAUSALITY: `train1_and_train2` is acausal for HAI, because train2 is
    recorded after the inference stream. Selecting it requires flipping
    `dataset.allow_acausal_baseline` explicitly, so the violation can never be
    an accident.
    """
    ds = _section(config, "dataset")
    source = ds.get("baseline_source")
    allow_acausal = bool(ds.get("allow_acausal_baseline", False))
    specs = file_specs(config)

    if source == "train1_only":
        keys = ("train1",)
    elif source == "train1_and_train2":
        keys = ("train1", "train2")
        if not allow_acausal:
            raise CausalityError(
                "dataset.baseline_source is 'train1_and_train2' but "
                "dataset.allow_acausal_baseline is false. For HAI, train2 is "
                "recorded AFTER the inference stream (train1 < test1 < train2), "
                "so this baseline would use information unavailable at "
                "inference time. Set allow_acausal_baseline: true only for a "
                "deliberately-labelled acausal ablation."
            )
    else:
        raise ConfigError(
            f"dataset.baseline_source must be 'train1_only' or "
            f"'train1_and_train2', got {source!r}"
        )

    for k in keys:
        if k not in specs:
            raise ConfigError(f"baseline_source names {k!r}, absent from dataset.files")
        if specs[k].role not in BASELINE_ROLES:
            raise ConfigError(
                f"baseline_source names {k!r} but its role is "
                f"{specs[k].role!r}, not a baseline role"
            )
    if len(keys) > 1 and not allow_acausal:  # pragma: no cover - unreachable
        raise CausalityError("multi-file baseline without allow_acausal_baseline")
    return keys


# -----------------------------------------------------------------------------
# Loading
# -----------------------------------------------------------------------------


def _read_frame(spec: FileSpec, ds: Mapping[str, Any], nrows: int | None) -> pd.DataFrame:
    fmt = str(ds.get("format", "csv")).lower()
    if fmt != "csv":
        raise ConfigError(
            f"dataset.format {fmt!r} is not supported in Phase 1; "
            f"HAI files are delimiter-separated text ('csv')"
        )
    if not spec.path.exists():
        raise SchemaError(f"{spec.key}: file not found at {spec.path}")

    ts_col = str(ds["timestamp_column"])
    raw = pd.read_csv(
        spec.path,
        sep=str(ds.get("delimiter", ",")),
        nrows=nrows,
        dtype={ts_col: "string"},   # parse the time axis explicitly, not by guess
        engine="c",
    )
    if ds.get("strip_column_whitespace", True):
        raw.columns = [str(c).strip() for c in raw.columns]
    return raw


def _order_and_select(raw: pd.DataFrame, ds: Mapping[str, Any], key: str) -> pd.DataFrame:
    """Apply the configured causal order and an inclusive timestamp range.

    WUSTL is a flow event log: several flows can share a StartTime.  A stable
    sort by configured, non-target header fields makes their within-timestamp
    order deterministic without consulting a label or a later observation.
    """
    ts_col = str(ds["timestamp_column"])
    order = ds.get("ordering") or {}
    tie = list(order.get("tie_breaker_columns") or [])
    forbidden = {str(ds.get("target_column") or ""), str(ds.get("label_column") or "")}
    if any(c in forbidden for c in tie):
        raise ConfigError("ordering.tie_breaker_columns must not contain a target or label")
    missing = [c for c in [ts_col, *tie] if c not in raw.columns]
    if missing:
        raise SchemaError(f"{key}: ordering column(s) absent from the raw file: {missing}")
    entry = _section(ds, "files").get(key) or {}
    selected = entry.get("selection_time_range")
    if selected:
        ts = pd.to_datetime(raw[ts_col], format=str(ds["timestamp_format"]), errors="coerce")
        lo, hi = pd.Timestamp(selected[0]), pd.Timestamp(selected[1])
        raw = raw.loc[(ts >= lo) & (ts <= hi)].reset_index(drop=True)
    if order.get("sort_by_timestamp", False):
        raw = raw.sort_values([ts_col, *tie], kind="mergesort", ignore_index=True)
    return raw


def _parse_timestamps(raw: pd.DataFrame, ds: Mapping[str, Any], key: str) -> pd.Series:
    ts_col = str(ds["timestamp_column"])
    if ts_col not in raw.columns:
        raise SchemaError(
            f"{key}: timestamp column {ts_col!r} not found. "
            f"First columns present: {list(raw.columns[:6])}"
        )
    values = raw[ts_col]
    if ds.get("strip_value_whitespace", True):
        values = values.str.strip()
    ts = pd.to_datetime(values, format=str(ds["timestamp_format"]), errors="coerce")
    n_bad = int(ts.isna().sum())
    if n_bad:
        sample = values[ts.isna()].head(3).tolist()
        raise SchemaError(
            f"{key}: {n_bad:,} timestamps do not match "
            f"dataset.timestamp_format={ds['timestamp_format']!r}; e.g. {sample}"
        )
    return ts.reset_index(drop=True)


def _assess_time_axis(
    ts: pd.Series, ds: Mapping[str, Any]
) -> tuple[TimeAxisReport, pd.Series]:
    """Measure the time axis and assign contiguity block ids."""
    expected = float(ds.get("expected_sampling_interval_s", 1))
    max_gap = float(ds.get("max_gap_s_before_new_block", 2))

    deltas = ts.diff().dt.total_seconds().to_numpy()
    body = deltas[1:] if len(deltas) > 1 else np.array([], dtype=float)

    if body.size:
        vals, counts = np.unique(body, return_counts=True)
        modal = float(vals[int(np.argmax(counts))])
        max_observed = float(np.nanmax(body))
        n_gaps = int(np.sum(body > max_gap))
    else:  # single-row file
        modal, max_observed, n_gaps = expected, 0.0, 0

    # Block id increments whenever the step exceeds the gap tolerance, so
    # preprocessing and windowing never bridge a discontinuity.
    breaks = np.zeros(len(ts), dtype=bool)
    if body.size:
        breaks[1:] = body > max_gap
    block_id = pd.Series(np.cumsum(breaks), name="block_id")

    report = TimeAxisReport(
        n_rows=len(ts),
        first=ts.iloc[0],
        last=ts.iloc[-1],
        monotonic_increasing=bool(ts.is_monotonic_increasing),
        n_duplicate_timestamps=int(ts.duplicated().sum()),
        modal_interval_s=modal,
        n_gaps=n_gaps,
        max_gap_s=max_observed,
        n_blocks=int(block_id.iloc[-1]) + 1,
    )
    return report, block_id


def load_file(
    config: Mapping[str, Any],
    key: str,
    *,
    root: Path | str = ".",
    max_rows: int | None = None,
) -> LoadedFile:
    """Load and validate exactly one file from `dataset.files`.

    Findings are recorded in the returned ValidationReport. Hard structural
    violations (unparseable timestamps, wrong row count, non-monotonic time when
    monotonicity is required) raise, because silently continuing would
    invalidate every downstream measurement.
    """
    ds = _section(config, "dataset")
    mode = str(ds.get("mode", "")).lower()
    if mode not in {"hai", "csv"}:
        raise ConfigError(f"dataset.mode must be 'hai' or 'csv', got {mode!r}")

    specs = file_specs(config, root)
    if key not in specs:
        raise ConfigError(f"unknown dataset.files key {key!r}; have {sorted(specs)}")
    spec = specs[key]

    # A flow-level time range must be selected after the complete file has been
    # ordered; a file-order head would be an acausal, shuffled pseudo-split.
    entry = _section(ds, "files").get(key) or {}
    raw = _read_frame(spec, ds, None if entry.get("selection_time_range") else max_rows)
    raw = _order_and_select(raw, ds, key)
    if max_rows is not None:
        raw = raw.iloc[:max_rows].reset_index(drop=True)
    ts = _parse_timestamps(raw, ds, key)

    findings: list[str] = []

    # --- row count: the config's `rows` is a VERIFIED fact, so a mismatch means
    # --- the raw file changed. That must never pass quietly.
    rows_ok: bool | None = None
    if spec.declared_rows is not None and max_rows is None:
        rows_ok = len(raw) == int(spec.declared_rows)
        if not rows_ok:
            raise SchemaError(
                f"{key}: config declares rows={spec.declared_rows:,} but the file "
                f"has {len(raw):,}. Either the raw file changed (check "
                f"data/raw/PROVENANCE.json checksums) or the config is stale."
            )

    report_time, block_id = _assess_time_axis(ts, ds)

    if ds.get("require_monotonic_timestamps", True) and not report_time.monotonic_increasing:
        raise TimeAxisError(
            f"{key}: timestamps are not monotonically increasing, and "
            f"dataset.require_monotonic_timestamps is true. Streaming order is "
            f"a hard requirement: ADWIN and the drift schedule are both defined "
            f"over time order."
        )
    if report_time.n_duplicate_timestamps:
        findings.append(
            f"{report_time.n_duplicate_timestamps:,} duplicate timestamps"
        )
    expected_interval = float(ds.get("expected_sampling_interval_s", 1))
    if report_time.modal_interval_s != expected_interval:
        findings.append(
            f"modal sampling interval is {report_time.modal_interval_s:g}s, "
            f"config expects {expected_interval:g}s"
        )
    if report_time.n_gaps:
        findings.append(
            f"{report_time.n_gaps} gap(s) longer than "
            f"{ds.get('max_gap_s_before_new_block')}s; "
            f"{report_time.n_blocks} contiguity blocks"
        )

    # --- declared time range
    range_ok: bool | None = None
    if spec.declared_time_range is not None and max_rows is None:
        want = (pd.Timestamp(spec.declared_time_range[0]),
                pd.Timestamp(spec.declared_time_range[1]))
        range_ok = (report_time.first == want[0]) and (report_time.last == want[1])
        if not range_ok:
            findings.append(
                f"time range {report_time.first} -> {report_time.last} differs "
                f"from config {want[0]} -> {want[1]}"
            )

    # --- process variables: everything that is not the time axis or a target
    ts_col = str(ds["timestamp_column"])
    reserved = {ts_col}
    for opt in ("label_column", "target_column"):
        if ds.get(opt):
            reserved.add(str(ds[opt]))
    reserved.update(str(c) for c in ((_section(ds, "features").get("exclude")) or []))
    frame = raw.drop(columns=[c for c in reserved if c in raw.columns])
    frame = frame.reset_index(drop=True)

    non_numeric = tuple(
        c for c in frame.columns if not pd.api.types.is_numeric_dtype(frame[c])
    )
    if non_numeric:
        findings.append(f"non-numeric process columns: {list(non_numeric)}")

    missing_by_column = {
        c: int(n) for c, n in frame.isna().sum().items() if int(n) > 0
    }
    n_missing = int(sum(missing_by_column.values()))
    if n_missing:
        findings.append(f"{n_missing:,} missing cells across "
                        f"{len(missing_by_column)} column(s)")

    report = ValidationReport(
        key=key,
        path=spec.path,
        n_rows=len(frame),
        n_columns=len(frame.columns),
        n_missing_cells=n_missing,
        missing_by_column=missing_by_column,
        non_numeric_columns=non_numeric,
        row_count_matches_config=rows_ok,
        time_range_matches_config=range_ok,
        time_axis=report_time,
        findings=tuple(findings),
    )
    return LoadedFile(
        key=key,
        role=spec.role,
        path=spec.path,
        frame=frame,
        timestamps=ts,
        block_id=block_id,
        report=report,
    )


def load_baseline(
    config: Mapping[str, Any], *, root: Path | str = ".", max_rows: int | None = None
) -> list[LoadedFile]:
    """Load the baseline file(s) selected by `dataset.baseline_source`.

    Returns a LIST and does not concatenate, even for a multi-file baseline:
    `dataset.concatenate_files` is false, and a seam between two recordings is
    an artificial regime change that would confound the injected drift.
    """
    keys = resolve_baseline_keys(config)
    cap = _section(config, "split").get("max_baseline_rows")
    if max_rows is None and cap is not None:
        max_rows = int(cap)
    return [load_file(config, k, root=root, max_rows=max_rows) for k in keys]


def load_inference_stream(
    config: Mapping[str, Any], *, root: Path | str = ".", max_rows: int | None = None
) -> LoadedFile:
    """Load the file whose role is `inference_stream`."""
    if max_rows is None:
        cap = _section(config, "streaming").get("max_samples")
        if cap is not None:
            max_rows = int(cap)
    return load_file(config, inference_key(config), root=root, max_rows=max_rows)


# -----------------------------------------------------------------------------
# Cross-file checks
# -----------------------------------------------------------------------------


def assert_schema_match(loaded: Sequence[LoadedFile]) -> tuple[str, ...]:
    """Every file must expose an identical column set, in identical order."""
    if not loaded:
        raise ConfigError("assert_schema_match requires at least one file")
    reference = tuple(loaded[0].frame.columns)
    for other in loaded[1:]:
        cols = tuple(other.frame.columns)
        if cols != reference:
            missing = sorted(set(reference) - set(cols))
            extra = sorted(set(cols) - set(reference))
            raise SchemaError(
                f"schema mismatch between {loaded[0].key!r} and {other.key!r}: "
                f"missing={missing} extra={extra} "
                f"(same set, different order: {sorted(cols) == sorted(reference)})"
            )
    return reference


def assert_causal_baseline(
    baseline: Sequence[LoadedFile],
    inference: LoadedFile,
    config: Mapping[str, Any],
) -> None:
    """Every baseline file must END no later than the inference stream STARTS.

    This is the check that catches HAI's train1 < test1 < train2 ordering. It
    operates on measured timestamps, not on the config's declared ranges, so a
    stale config cannot mask a real violation.
    """
    allow = bool(_section(config, "dataset").get("allow_acausal_baseline", False))
    t_infer_start = inference.report.time_axis.first
    for b in baseline:
        b_end = b.report.time_axis.last
        if b_end <= t_infer_start:
            continue
        message = (
            f"acausal baseline: {b.key!r} ends {b_end} but the inference stream "
            f"{inference.key!r} starts {t_infer_start}. Fitting the baseline on "
            f"{b.key!r} would use data recorded after inference began."
        )
        if not allow:
            raise CausalityError(
                message + " Set dataset.allow_acausal_baseline: true only for a "
                "deliberately-labelled acausal ablation."
            )
        # Allowed, but never silent -- an acausal run must be visible in the log.
        import warnings

        warnings.warn(message + " Permitted by allow_acausal_baseline: true.",
                      RuntimeWarning, stacklevel=2)


# -----------------------------------------------------------------------------
# Baseline profiling  (the single source of baseline statistics)
# -----------------------------------------------------------------------------


def _classify_columns(
    stats: Mapping[str, ColumnProfile], features: Iterable[str], ds: Mapping[str, Any]
) -> tuple[tuple[str, ...], tuple[str, ...]]:
    """Split features into (continuous, discrete) per `features.type_detection`.

    Cardinality-based by default. The distinction is not cosmetic: applying a
    continuous offset to a discrete state variable produces a physically
    impossible value, which the specification forbids.
    """
    td = _section(ds, "features").get("type_detection")
    if not isinstance(td, Mapping):
        raise ConfigError("dataset.features.type_detection is missing")
    method = str(td.get("method", "cardinality"))

    if method == "explicit":
        cont = td.get("explicit_continuous")
        disc = td.get("explicit_discrete")
        if not cont or not disc:
            raise ConfigError(
                "type_detection.method is 'explicit' but explicit_continuous / "
                "explicit_discrete are not both populated"
            )
        cont_set, disc_set = set(cont), set(disc)
        overlap = cont_set & disc_set
        if overlap:
            raise ConfigError(f"column(s) listed as both types: {sorted(overlap)}")
        unclassified = set(features) - cont_set - disc_set
        if unclassified:
            raise ConfigError(
                f"type_detection is explicit but {len(unclassified)} feature(s) "
                f"are unclassified: {sorted(unclassified)[:8]}"
            )
        return (tuple(c for c in features if c in cont_set),
                tuple(c for c in features if c in disc_set))

    if method != "cardinality":
        raise ConfigError(
            f"type_detection.method must be 'cardinality' or 'explicit', "
            f"got {method!r}"
        )
    threshold = int(td.get("discrete_max_cardinality", 10))
    # Explicit overrides still apply on top of the cardinality rule, so a
    # high-cardinality integer counter can be declared discrete by hand.
    forced_cont = set(td.get("explicit_continuous") or [])
    forced_disc = set(td.get("explicit_discrete") or [])
    cont, disc = [], []
    for c in features:
        if c in forced_disc:
            disc.append(c)
        elif c in forced_cont:
            cont.append(c)
        elif stats[c].n_unique <= threshold:
            disc.append(c)
        else:
            cont.append(c)
    return tuple(cont), tuple(disc)


def profile_baseline(
    config: Mapping[str, Any], baseline: Sequence[LoadedFile]
) -> BaselineProfile:
    """Measure per-column statistics on the BASELINE file(s) only.

    LEAKAGE: this function refuses any file whose role is not a baseline role.
    That refusal is the structural reason no downstream component can obtain
    statistics computed over the inference stream.
    """
    if not baseline:
        raise ConfigError("profile_baseline requires at least one baseline file")
    for b in baseline:
        if b.role not in BASELINE_ROLES:
            raise CausalityError(
                f"profile_baseline received {b.key!r} with role {b.role!r}. "
                f"Baseline statistics may only be computed from "
                f"{sorted(BASELINE_ROLES)}. Computing them from the inference "
                f"stream would leak post-drift information into normalization "
                f"and into the drift definition itself."
            )
    ds = _section(config, "dataset")
    feats_cfg = _section(ds, "features")
    columns = assert_schema_match(baseline)

    # Multiple baseline files are profiled over their vertical concatenation of
    # VALUES only. This is a statistics pool, not a data stream -- no seam is
    # created because nothing is ever iterated in this order.
    frames = [b.frame for b in baseline]
    pooled = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)

    stats: dict[str, ColumnProfile] = {}
    for c in columns:
        col = pd.to_numeric(pooled[c], errors="coerce")
        stats[c] = ColumnProfile(
            name=c,
            n_unique=int(col.nunique(dropna=True)),
            is_discrete=False,          # filled in after classification
            mean=float(col.mean()),
            std=float(col.std(ddof=0)),
            minimum=float(col.min()),
            maximum=float(col.max()),
        )

    # --- feature selection ---------------------------------------------------
    include = feats_cfg.get("include")
    exclude = set(feats_cfg.get("exclude") or [])
    if include:
        unknown = [c for c in include if c not in columns]
        if unknown:
            raise ConfigError(f"features.include names absent columns: {unknown}")
        candidates = [c for c in include if c not in exclude]
    else:
        candidates = [c for c in columns if c not in exclude]

    # --- zero variance -------------------------------------------------------
    # LEAKAGE: measured on the baseline, never on the inference stream. A column
    # constant in test1 but varying in train1 is NOT dropped -- dropping on a
    # test-set property would be leakage.
    measured_zv = tuple(c for c in candidates if stats[c].zero_variance)
    dropped: tuple[str, ...] = ()
    if feats_cfg.get("drop_zero_variance", True):
        dropped = measured_zv
        candidates = [c for c in candidates if c not in set(dropped)]

    declared_zv = set(feats_cfg.get("known_zero_variance_all_files") or [])
    agreement = {
        "declared_n": len(declared_zv),
        "measured_n": len(measured_zv),
        "declared_but_not_measured": sorted(declared_zv - set(measured_zv)),
        "measured_but_not_declared": sorted(set(measured_zv) - declared_zv),
        "note": "measured wins; the declared list in config is reference only",
    }

    continuous, discrete = _classify_columns(stats, candidates, ds)
    disc_set = set(discrete)
    stats = {
        c: ColumnProfile(**{**p.__dict__, "is_discrete": c in disc_set})
        for c, p in stats.items()
    }

    return BaselineProfile(
        source_keys=tuple(b.key for b in baseline),
        n_rows=int(len(pooled)),
        columns=stats,
        feature_names=tuple(candidates),
        continuous=continuous,
        discrete=discrete,
        dropped_zero_variance=dropped,
        zero_variance_agreement=agreement,
    )


# -----------------------------------------------------------------------------
# Target resolution  (currently blocked by design)
# -----------------------------------------------------------------------------


def resolve_target(config: Mapping[str, Any], profile: BaselineProfile | None = None) -> str:
    """Return the prediction target column, or refuse.

    The HAI process-value files contain no label column. The 23.05 release does
    ship official attack labels, but in a SEPARATE sidecar file per test stream,
    so a target exists only once the configuration names that file. Until then
    `dataset.task` stays 'unresolved' and this function raises. Guessing here
    would silently decide the study's primary metric.
    """
    ds = _section(config, "dataset")
    task = str(ds.get("task", "unresolved"))

    if task == "unresolved":
        raise UnresolvedTaskError(
            "dataset.task is 'unresolved'. The HAI process-value files contain "
            "no label/attack column (see data/raw/PROVENANCE.json -> "
            "finding_no_label_column), and no label has been fabricated. "
            "Resolve dataset.task to one of: 'forecasting_regression' (set "
            "target_column to a continuous channel), 'state_classification' "
            "(set target_column to a discrete channel -- must first survive a "
            "headroom probe, per the rejected SWaT actuator target), or "
            "'labels_from_hai_labels' (set dataset.label_file to the official "
            "label sidecar shipped alongside the inference stream, and "
            "dataset.label_column to its label column)."
        )

    if task == "labels_from_hai_labels":
        label_file = ds.get("label_file")
        if not label_file:
            raise ConfigError(
                "dataset.task is 'labels_from_hai_labels' but "
                "dataset.label_file is null. The HAI release ships attack "
                "labels separately from the process-value files; supply that "
                "file before selecting this task."
            )
        label_column = ds.get("label_column")
        if not label_column:
            raise ConfigError(
                "dataset.task is 'labels_from_hai_labels' but "
                "dataset.label_column is null."
            )
        return str(label_column)

    if task not in {"forecasting_regression", "state_classification", "supervised_classification"}:
        raise ConfigError(f"unrecognised dataset.task {task!r}")

    target = ds.get("target_column")
    if not target:
        raise ConfigError(f"dataset.task is {task!r} but target_column is null")
    target = str(target)

    # A dataset-supplied supervised target is deliberately removed from the
    # feature frame before profiling. Its absence from the profile is therefore
    # a leakage guard, not an error.
    if task == "supervised_classification":
        return target

    if profile is not None:
        if target not in profile.columns:
            raise ConfigError(
                f"target_column {target!r} is not present in the baseline "
                f"profile (dropped as zero-variance, or absent from the file)"
            )
        is_discrete = profile.columns[target].is_discrete
        if task == "state_classification" and not is_discrete:
            raise ConfigError(
                f"task is 'state_classification' but {target!r} is continuous "
                f"({profile.columns[target].n_unique:,} distinct baseline values)"
            )
        if task == "forecasting_regression" and is_discrete:
            raise ConfigError(
                f"task is 'forecasting_regression' but {target!r} is discrete "
                f"({profile.columns[target].n_unique} distinct baseline values)"
            )
    return target


def feature_names_for_target(
    config: Mapping[str, Any], profile: BaselineProfile, target: str
) -> tuple[str, ...]:
    """Features usable to predict `target`, with the sibling leakage trap closed.

    HAI encodes six actuators as <TAG>D (command) / <TAG>Z (feedback) pairs that
    track each other almost exactly. Predicting one while the other is a feature
    leaks the answer -- the same failure that invalidated the SWaT actuator
    target. `features.exclude_target_sibling` closes it.
    """
    feats_cfg = _section(_section(config, "dataset"), "features")
    banned = {target}
    if feats_cfg.get("exclude_target_sibling", True):
        pairs = feats_cfg.get("command_feedback_pairs") or {}
        for members in pairs.values():
            members = list(members)
            if target in members:
                banned.update(members)
    return tuple(c for c in profile.feature_names if c not in banned)
