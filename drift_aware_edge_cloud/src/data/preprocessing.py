"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/data/preprocessing.py
Phase    : Phase 1 / Step 4
Status   : IMPLEMENTED

Causal preprocessing: validation -> missing values -> safe outlier handling
-> ONLINE normalization -> windowing -> feature vectors.
MUST NOT call scaler.fit(all_data). MUST NOT delete post-drift observations
merely because they look like outliers.

The two failure modes this module exists to prevent
---------------------------------------------------
1. LEAKAGE. Every statistic used to transform a sample is either fitted on the
   baseline file or accumulated from samples at or before that sample's own
   timestamp. `fit()` refuses any file whose role is not a baseline role, so
   `scaler.fit(all_data)` is not merely discouraged -- there is no code path
   that reaches it. `normalization.forbid_global_fit` is asserted at fit time.

2. DRIFT CANCELLATION. This is the subtler and more dangerous one. An online
   scaler that keeps adapting subtracts a mean shift back out: the normalized
   stream looks unchanged, the Edge model stops degrading, ADWIN detects
   nothing, and the experiment silently measures nothing. `adaptation:
   frozen_after_baseline` is therefore the default. The other two modes are
   implemented because ASSUMPTION [A6] requires a sensitivity experiment
   comparing them -- and `measure_drift_absorption()` in this module quantifies
   exactly how much of an injected shift each mode destroys, so the hazard is a
   measurement rather than a warning comment.

Ordering of the stages is itself a scientific commitment
--------------------------------------------------------
Outlier statistics are frozen from the baseline and applied to RAW values,
before normalization, so "outlier" keeps a fixed physical meaning. Flags are
carried as METADATA alongside the data, never appended to the feature matrix: a
`validation_failed` column would hand the model a preprocessing artefact to
learn from, and under drift that artefact correlates with the drift itself.

Nothing here deletes a sample by default. `outliers.action: flag` and
`validation.on_failure: flag` annotate and keep, because genuine drift produces
unusual observations and dropping them would erase the signal under study.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Iterable, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.loader import (
    BASELINE_ROLES,
    BaselineProfile,
    CausalityError,
    ConfigError,
    DataError,
    LoadedFile,
)


class PreprocessingError(DataError):
    """Raised when a preprocessing stage cannot proceed honestly."""


class LeakageError(DataError):
    """Raised when an operation would use information from the future."""


ADAPTATION_MODES = ("frozen_after_baseline", "running", "rolling")
CONTINUOUS_STATS = ("mean", "std", "min", "max", "last", "slope")
DISCRETE_STATS = ("mode", "n_changes")


# =============================================================================
# Fitted state
# =============================================================================


@dataclass(frozen=True)
class BaselineStatistics:
    """Everything fitted on the baseline, and nothing else.

    Frozen: there is no method that mutates it. A downstream component holding
    one of these cannot accidentally refit it on post-drift data.
    """

    source_keys: tuple[str, ...]
    n_rows: int
    columns: tuple[str, ...]
    continuous: tuple[str, ...]
    discrete: tuple[str, ...]
    mean: pd.Series
    std: pd.Series
    minimum: pd.Series
    maximum: pd.Series
    q1: pd.Series
    q3: pd.Series
    # Tail of the baseline, kept only when `adaptation: rolling` needs history to
    # seed its window. Bounded by rolling_window rows.
    history_tail: pd.DataFrame | None = None
    notes: tuple[str, ...] = ()

    @property
    def iqr(self) -> pd.Series:
        return self.q3 - self.q1

    def valid_range(self, tolerance_sigma: float) -> tuple[pd.Series, pd.Series]:
        """Plausible range for validation: baseline mean +/- k sigma.

        Union'd with the observed baseline range so a channel that legitimately
        spans more than k sigma is not flagged on every sample.
        """
        lo = np.minimum(self.mean - tolerance_sigma * self.std, self.minimum)
        hi = np.maximum(self.mean + tolerance_sigma * self.std, self.maximum)
        return lo, hi

    def outlier_bounds(self, config: Mapping[str, Any]) -> tuple[pd.Series, pd.Series]:
        """IQR or z-score bounds, from frozen baseline statistics."""
        o = ((config.get("preprocessing") or {}).get("outliers")) or {}
        method = str(o.get("method", "iqr"))
        if method == "iqr":
            k = float(o.get("iqr_multiplier", 3.0))
            return self.q1 - k * self.iqr, self.q3 + k * self.iqr
        if method == "zscore":
            k = float(o.get("zscore_threshold", 4.0))
            return self.mean - k * self.std, self.mean + k * self.std
        raise ConfigError(
            f"preprocessing.outliers.method {method!r} is not implemented in "
            f"Phase 1; 'domain' needs process limits that HAI does not ship"
        )


def fit(
    config: Mapping[str, Any],
    baseline: Sequence[LoadedFile],
    profile: BaselineProfile,
) -> BaselineStatistics:
    """Fit every preprocessing statistic on the BASELINE file(s) only.

    LEAKAGE: refuses any file whose role is not a baseline role. This is the same
    structural barrier loader.profile_baseline uses, repeated here rather than
    assumed, because this is the second place in the codebase where fitting
    happens and a single unguarded entry point would be enough to leak.
    """
    if not baseline:
        raise ConfigError("fit() requires at least one baseline file")
    for b in baseline:
        if b.role not in BASELINE_ROLES:
            raise CausalityError(
                f"fit() received {b.key!r} with role {b.role!r}. Preprocessing "
                f"statistics may only be fitted on {sorted(BASELINE_ROLES)}. "
                f"Fitting on the inference stream is `scaler.fit(all_data)` "
                f"under another name."
            )
    norm = ((config.get("preprocessing") or {}).get("normalization")) or {}
    if not norm.get("forbid_global_fit", True):
        raise ConfigError(
            "preprocessing.normalization.forbid_global_fit is false. Global "
            "fitting is prohibited project-wide; the flag exists so that any "
            "attempt to disable it is a visible config diff."
        )
    adaptation = str(norm.get("adaptation", "frozen_after_baseline"))
    if adaptation not in ADAPTATION_MODES:
        raise ConfigError(
            f"preprocessing.normalization.adaptation must be one of "
            f"{ADAPTATION_MODES}, got {adaptation!r}"
        )

    features = tuple(profile.feature_names)
    frames = [b.frame.loc[:, list(features)] for b in baseline]
    pooled = frames[0] if len(frames) == 1 else pd.concat(frames, ignore_index=True)
    pooled = pooled.astype(float)

    notes: list[str] = []
    tail: pd.DataFrame | None = None
    if adaptation == "rolling":
        w = int(norm.get("rolling_window", 1000))
        tail = pooled.iloc[-w:].reset_index(drop=True)
        notes.append(
            f"adaptation 'rolling': the last {len(tail):,} baseline rows are kept "
            f"as history so the first inference samples are normalized against a "
            f"full window instead of a near-empty one"
        )
    if adaptation != "frozen_after_baseline":
        notes.append(
            f"adaptation is {adaptation!r}, NOT the default "
            f"'frozen_after_baseline'. This mode adapts to the inference stream "
            f"and will absorb part of any mean-shift drift. Valid only as a "
            f"declared ablation -- see measure_drift_absorption()."
        )

    return BaselineStatistics(
        source_keys=tuple(b.key for b in baseline),
        n_rows=int(len(pooled)),
        columns=features,
        continuous=tuple(profile.continuous),
        discrete=tuple(profile.discrete),
        mean=pooled.mean(),
        std=pooled.std(ddof=0),
        minimum=pooled.min(),
        maximum=pooled.max(),
        q1=pooled.quantile(0.25),
        q3=pooled.quantile(0.75),
        history_tail=tail,
        notes=tuple(notes),
    )


# =============================================================================
# Per-sample quality metadata  (NEVER a feature)
# =============================================================================


@dataclass
class QualityReport:
    """Boolean masks describing what preprocessing noticed, per row.

    Deliberately parallel to the data rather than inside it. `valid` is what
    stream.py's `min_valid_fraction` consumes.
    """

    n_rows: int
    valid: np.ndarray
    validation_failed: np.ndarray
    range_violation: np.ndarray
    outlier: np.ndarray
    filled: np.ndarray
    unfilled: np.ndarray
    n_outliers_by_column: dict[str, int] = field(default_factory=dict)
    n_range_violations_by_column: dict[str, int] = field(default_factory=dict)
    degenerate_bound_columns: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def summary(self) -> str:
        top_out = sorted(self.n_outliers_by_column.items(),
                         key=lambda kv: -kv[1])[:5]
        top_rng = sorted(self.n_range_violations_by_column.items(),
                         key=lambda kv: -kv[1])[:5]
        lines = [
            f"rows                 : {self.n_rows:,}",
            f"valid                : {int(self.valid.sum()):,} "
            f"({self.valid.mean():.4%})",
            f"validation failures  : {int(self.validation_failed.sum()):,}",
            f"range violations     : {int(self.range_violation.sum()):,} rows",
            f"outlier-flagged rows : {int(self.outlier.sum()):,} "
            f"({self.outlier.mean():.4%})",
            f"forward-filled rows  : {int(self.filled.sum()):,}",
            f"still-missing rows   : {int(self.unfilled.sum()):,}",
        ]
        if top_out:
            lines.append("top outlier channels : " + ", ".join(
                f"{c}={n:,}" for c, n in top_out))
        if top_rng:
            lines.append("top range channels   : " + ", ".join(
                f"{c}={n:,}" for c, n in top_rng))
        for n in self.notes:
            lines.append(f"note: {n}")
        return "\n".join(lines)


@dataclass(frozen=True)
class PreprocessedStream:
    """Normalized values plus the metadata describing how they got that way."""

    key: str
    role: str
    frame: pd.DataFrame              # normalized feature values
    raw_frame: pd.DataFrame          # post-imputation, pre-normalization
    timestamps: pd.Series
    block_id: pd.Series
    quality: QualityReport
    statistics: BaselineStatistics
    adaptation: str

    def __len__(self) -> int:
        return len(self.frame)


# =============================================================================
# Stage 1 -- validation
# =============================================================================


def _validate(
    frame: pd.DataFrame,
    timestamps: pd.Series,
    stats: BaselineStatistics,
    config: Mapping[str, Any],
    q: QualityReport,
) -> None:
    v = ((config.get("preprocessing") or {}).get("validation")) or {}
    if not v.get("enabled", True):
        q.notes.append("validation disabled by configuration")
        return

    on_failure = str(v.get("on_failure", "flag"))
    if on_failure not in {"flag", "drop", "raise"}:
        raise ConfigError(
            f"preprocessing.validation.on_failure must be 'flag', 'drop' or "
            f"'raise', got {on_failure!r}"
        )
    if on_failure == "drop":
        raise ConfigError(
            "preprocessing.validation.on_failure is 'drop'. Dropping rows that "
            "fail a baseline-derived range check would delete exactly the "
            "post-drift observations this experiment exists to detect. Use "
            "'flag' (README constraint 5)."
        )

    if v.get("check_timestamp_monotonic", True):
        diffs = timestamps.diff().dt.total_seconds().to_numpy()[1:]
        n_back = int(np.sum(diffs < 0))
        if n_back:
            msg = f"{n_back:,} non-monotonic timestamp step(s)"
            if on_failure == "raise":
                raise PreprocessingError(msg)
            q.validation_failed[1:][diffs < 0] = True
            q.notes.append(msg)

    if v.get("check_duplicate_timestamps", True):
        dup = timestamps.duplicated(keep="first").to_numpy()
        if dup.any():
            q.validation_failed |= dup
            q.notes.append(f"{int(dup.sum()):,} duplicate timestamp(s) flagged")

    if v.get("check_dtypes", True):
        bad = [c for c in frame.columns
               if not pd.api.types.is_numeric_dtype(frame[c])]
        if bad:
            msg = f"non-numeric process columns: {bad}"
            if on_failure == "raise":
                raise PreprocessingError(msg)
            q.notes.append(msg)

    if v.get("check_ranges", True):
        source = str(v.get("range_source", "baseline"))
        if source == "none":
            q.notes.append("range checking disabled (range_source: none)")
            return
        if source != "baseline":
            raise ConfigError(
                f"preprocessing.validation.range_source {source!r} is not "
                f"supported in Phase 1; 'explicit' would require inventing "
                f"process limits HAI does not ship"
            )
        tol = float(v.get("range_tolerance_sigma", 6.0))
        lo, hi = stats.valid_range(tol)
        cols = [c for c in frame.columns if c in stats.columns]
        below = frame[cols].lt(lo[cols], axis=1)
        above = frame[cols].gt(hi[cols], axis=1)
        viol = (below | above).fillna(False)
        per_col = viol.sum(axis=0)
        q.n_range_violations_by_column = {
            c: int(n) for c, n in per_col.items() if int(n) > 0
        }
        row_viol = viol.any(axis=1).to_numpy()
        q.range_violation |= row_viol
        q.validation_failed |= row_viol
        q.notes.append(
            f"range check at baseline mean +/- {tol:g} sigma (union'd with the "
            f"observed baseline range): {int(row_viol.sum()):,} row(s) flagged, "
            f"KEPT not dropped"
        )


# =============================================================================
# Stage 2 -- missing values (causal)
# =============================================================================


def _impute(
    frame: pd.DataFrame,
    block_id: pd.Series,
    config: Mapping[str, Any],
    q: QualityReport,
) -> pd.DataFrame:
    m = ((config.get("preprocessing") or {}).get("missing")) or {}
    method = str(m.get("method", "forward_fill"))
    limit = m.get("max_consecutive_fill")
    limit = None if limit is None else int(limit)
    respect_blocks = bool(m.get("respect_block_boundaries", True))

    was_missing = frame.isna()

    # Configuration is validated BEFORE the data is inspected. An acausal
    # `interpolate_direction` is wrong whether or not this particular stream
    # happens to contain a hole; a guard that fires only on streams with missing
    # cells would let the same run pass on test1 and fail on train1.
    if method == "drop_window":
        raise ConfigError(
            "preprocessing.missing.method 'drop_window' is not implemented in "
            "Phase 1: dropping whole windows changes the evaluation grid, so "
            "pre- and post-drift regions would no longer be comparable"
        )
    if method == "interpolate":
        direction = str(m.get("interpolate_direction", "backward_only"))
        if direction != "backward_only":
            raise LeakageError(
                f"preprocessing.missing.interpolate_direction is {direction!r}. "
                f"Interpolating between t-1 and t+1 reads a FUTURE sample. Only "
                f"'backward_only' (equivalent to forward-fill) is causal."
            )
        q.notes.append(
            "method 'interpolate' with backward_only direction is exactly "
            "forward-fill; applied as such"
        )
    elif method != "forward_fill":
        raise ConfigError(
            f"preprocessing.missing.method must be 'forward_fill', "
            f"'interpolate' or 'drop_window', got {method!r}"
        )

    if not was_missing.to_numpy().any():
        q.notes.append("no missing cells; imputation was a no-op")
        return frame

    # Forward-fill only. Never bfill: that would read the future.
    if respect_blocks and block_id is not None:
        filled = frame.groupby(block_id.to_numpy(), sort=False).ffill(limit=limit)
        q.notes.append("forward-fill applied within contiguity blocks only")
    else:
        filled = frame.ffill(limit=limit)

    now_missing = filled.isna()
    q.filled |= (was_missing & ~now_missing).any(axis=1).to_numpy()
    q.unfilled |= now_missing.any(axis=1).to_numpy()
    if now_missing.to_numpy().any():
        q.notes.append(
            f"{int(now_missing.any(axis=1).sum()):,} row(s) still missing after "
            f"a max_consecutive_fill={limit} forward-fill; flagged invalid "
            f"rather than back-filled from the future"
        )
    return filled


# =============================================================================
# Stage 3 -- outliers (flag, do not delete)
# =============================================================================


def _flag_outliers(
    frame: pd.DataFrame,
    stats: BaselineStatistics,
    config: Mapping[str, Any],
    q: QualityReport,
) -> pd.DataFrame:
    o = ((config.get("preprocessing") or {}).get("outliers")) or {}
    method = str(o.get("method", "iqr"))
    if method == "none":
        q.notes.append("outlier detection disabled")
        return frame

    action = str(o.get("action", "flag"))
    if action == "drop":
        raise ConfigError(
            "preprocessing.outliers.action is 'drop'. Genuine drift produces "
            "unusual observations; dropping them erases the signal the "
            "experiment is designed to detect. Use 'flag' (README constraint 5)."
        )
    if action not in {"flag", "clip"}:
        raise ConfigError(
            f"preprocessing.outliers.action must be 'flag' or 'clip', got "
            f"{action!r}"
        )

    source = str(o.get("statistics_source", "baseline_frozen"))
    if source != "baseline_frozen":
        raise LeakageError(
            f"preprocessing.outliers.statistics_source is {source!r}. "
            f"Recomputing outlier bounds over post-drift data lets the "
            f"definition of 'outlier' absorb the drift."
        )

    apply_to = str(o.get("apply_to", "continuous_only"))
    if apply_to == "continuous_only":
        cols = [c for c in stats.continuous if c in frame.columns]
    elif apply_to == "all":
        cols = [c for c in frame.columns if c in stats.columns]
    else:
        raise ConfigError(
            f"preprocessing.outliers.apply_to must be 'continuous_only' or "
            f"'all', got {apply_to!r}"
        )

    lo, hi = stats.outlier_bounds(config)
    sub = frame[cols]

    # OPEN FINDING (behaviour deliberately unchanged): a piecewise-constant ICS
    # channel can have q1 == q3, so the 3*IQR band has ZERO width and every
    # value that is not exactly that constant is flagged. Detected and reported
    # here rather than silently repaired, because choosing a repair (floor the
    # width, switch to z-score, or exclude the channel) is a scientific decision
    # about what "outlier" means for a setpoint. See data/raw/PROVENANCE.json.
    width = (hi[cols] - lo[cols]).astype(float)
    degenerate = [c for c in cols if width[c] <= 0.0]
    q.degenerate_bound_columns = list(degenerate)
    if degenerate:
        q.notes.append(
            f"DEGENERATE OUTLIER BOUNDS on {len(degenerate)} of {len(cols)} "
            f"channel(s) -- zero-width band because q1 == q3: {degenerate}. "
            f"Every value not exactly equal to that constant is flagged, so the "
            f"row-level outlier flag may saturate at 100% and carry no "
            f"information. Behaviour unchanged pending a decision; see "
            f"PROVENANCE.json finding_degenerate_outlier_bounds."
        )

    flagged = (sub.lt(lo[cols], axis=1) | sub.gt(hi[cols], axis=1)).fillna(False)
    per_col = flagged.sum(axis=0)
    q.n_outliers_by_column = {c: int(n) for c, n in per_col.items() if int(n) > 0}
    q.outlier |= flagged.any(axis=1).to_numpy()

    n_rows_flagged = int(q.outlier.sum())
    q.notes.append(
        f"outliers by frozen baseline {method.upper()} on {len(cols)} "
        f"{apply_to} channel(s): {n_rows_flagged:,} row(s) flagged, action="
        f"{action}"
    )
    if degenerate:
        keep = [c for c in cols if c not in set(degenerate)]
        alt = flagged[keep].any(axis=1) if keep else pd.Series(False, index=frame.index)
        q.notes.append(
            f"same measurement EXCLUDING the {len(degenerate)} degenerate "
            f"channel(s): {int(alt.sum()):,} row(s) = {float(alt.mean()):.4%} "
            f"over {len(keep)} channel(s) -- reported for interpretation only; "
            f"the flags above are unmodified"
        )
    if action == "clip":
        out = frame.copy()
        out[cols] = sub.clip(lower=lo[cols], upper=hi[cols], axis=1)
        q.notes.append(
            "action 'clip' MODIFIES values and will attenuate drift; this is an "
            "ablation, not the default"
        )
        return out
    # 'flag' keeps every value exactly as recorded.
    return frame


# =============================================================================
# Stage 4 -- causal filtering (off by default)
# =============================================================================


def _filter(
    frame: pd.DataFrame,
    block_id: pd.Series,
    config: Mapping[str, Any],
    q: QualityReport,
) -> pd.DataFrame:
    f = ((config.get("preprocessing") or {}).get("filtering")) or {}
    if not f.get("enabled", False):
        q.notes.append(
            "filtering disabled (default): smoothing attenuates the abrupt "
            "change that sudden drift creates"
        )
        return frame
    if not f.get("causal", True):
        raise LeakageError(
            "preprocessing.filtering.causal is false. A centred filter averages "
            "over future samples."
        )
    method = str(f.get("method", "moving_average"))
    w = int(f.get("window", 5))
    if w < 1:
        raise ConfigError(f"preprocessing.filtering.window must be >= 1, got {w}")

    grouped = frame.groupby(block_id.to_numpy(), sort=False)
    if method == "moving_average":
        out = grouped.rolling(w, min_periods=1).mean().reset_index(drop=True)
    elif method == "median":
        out = grouped.rolling(w, min_periods=1).median().reset_index(drop=True)
    elif method == "none":
        return frame
    else:
        raise ConfigError(
            f"preprocessing.filtering.method must be 'moving_average', "
            f"'median' or 'none', got {method!r}"
        )
    q.notes.append(
        f"causal (backward-looking) {method} filter, window={w}, applied within "
        f"blocks; this attenuates sudden drift by design"
    )
    return out.loc[:, frame.columns]


# =============================================================================
# Stage 5 -- normalization
# =============================================================================


def _normalize(
    frame: pd.DataFrame,
    stats: BaselineStatistics,
    config: Mapping[str, Any],
    q: QualityReport,
) -> pd.DataFrame:
    """Normalize causally. Three adaptation modes, all backward-looking."""
    n = ((config.get("preprocessing") or {}).get("normalization")) or {}
    if not n.get("enabled", True):
        q.notes.append("normalization disabled")
        return frame

    method = str(n.get("method", "zscore"))
    adaptation = str(n.get("adaptation", "frozen_after_baseline"))
    eps = float(n.get("epsilon", 1e-8))
    cols = list(frame.columns)
    X = frame[cols].to_numpy(dtype=float)

    if method == "minmax":
        if adaptation != "frozen_after_baseline":
            raise ConfigError(
                "minmax normalization is only implemented for "
                "'frozen_after_baseline'; a running min/max is a ratchet that "
                "never recovers and would make drift irreversible"
            )
        lo = stats.minimum[cols].to_numpy(dtype=float)
        hi = stats.maximum[cols].to_numpy(dtype=float)
        Z = (X - lo) / np.maximum(hi - lo, eps)
        q.notes.append("minmax normalization against frozen baseline range")
        return pd.DataFrame(Z, columns=cols, index=frame.index)
    if method == "none":
        return frame
    if method != "zscore":
        raise ConfigError(
            f"preprocessing.normalization.method must be 'zscore', 'minmax' or "
            f"'none', got {method!r}"
        )

    if adaptation == "frozen_after_baseline":
        mu = stats.mean[cols].to_numpy(dtype=float)
        sd = stats.std[cols].to_numpy(dtype=float)
        Z = (X - mu) / np.maximum(sd, eps)
        q.notes.append(
            "z-score against FROZEN baseline statistics: an injected mean shift "
            "survives normalization and stays visible to the drift detector"
        )
        return pd.DataFrame(Z, columns=cols, index=frame.index)

    if adaptation == "running":
        # Seeded cumulative statistics via raw moments. Row t uses rows <= t of
        # the inference stream plus the whole baseline -- no future information.
        n0 = float(stats.n_rows)
        mu0 = stats.mean[cols].to_numpy(dtype=float)
        var0 = np.square(stats.std[cols].to_numpy(dtype=float))
        counts = n0 + np.arange(1, X.shape[0] + 1, dtype=float)[:, None]
        sum_x = n0 * mu0 + np.cumsum(np.nan_to_num(X), axis=0)
        sum_x2 = n0 * (var0 + mu0**2) + np.cumsum(np.nan_to_num(X) ** 2, axis=0)
        mu = sum_x / counts
        var = np.maximum(sum_x2 / counts - mu**2, 0.0)
        Z = (X - mu) / np.maximum(np.sqrt(var), eps)
        q.notes.append(
            f"z-score against RUNNING statistics seeded with {int(n0):,} "
            f"baseline rows: causal, but cumulative adaptation slowly absorbs "
            f"drift -- ablation only"
        )
        return pd.DataFrame(Z, columns=cols, index=frame.index)

    # rolling
    w = int(n.get("rolling_window", 1000))
    if w < 2:
        raise ConfigError(f"normalization.rolling_window must be >= 2, got {w}")
    tail = stats.history_tail
    if tail is None:
        raise PreprocessingError(
            "adaptation 'rolling' needs baseline history; refit with the same "
            "config so BaselineStatistics.history_tail is populated"
        )
    hist = tail.loc[:, cols].to_numpy(dtype=float)
    joined = np.vstack([hist, X])
    dfj = pd.DataFrame(joined, columns=cols)
    roll = dfj.rolling(w, min_periods=1)
    mu = roll.mean().to_numpy()[len(hist):]
    sd = roll.std(ddof=0).to_numpy()[len(hist):]
    Z = (X - mu) / np.maximum(sd, eps)
    q.notes.append(
        f"z-score against a {w}-row ROLLING window seeded with {len(hist):,} "
        f"baseline rows: backward-looking, but absorbs drift fastest of the "
        f"three modes -- ablation only"
    )
    return pd.DataFrame(Z, columns=cols, index=frame.index)


# =============================================================================
# The pipeline
# =============================================================================


def transform(
    config: Mapping[str, Any],
    source: Any,
    stats: BaselineStatistics,
    *,
    columns: Sequence[str] | None = None,
) -> PreprocessedStream:
    """Run the full causal pipeline over one stream.

    `source` is anything with .frame/.timestamps/.block_id/.key/.role -- a
    LoadedFile or a generator.DriftedStream, so the clean and drifted paths are
    byte-for-byte the same code.
    """
    cols = list(columns) if columns is not None else [
        c for c in stats.columns if c in source.frame.columns
    ]
    missing_cols = [c for c in stats.columns if c not in source.frame.columns]
    frame = source.frame.loc[:, cols].astype(float).reset_index(drop=True)
    timestamps = pd.Series(source.timestamps).reset_index(drop=True)
    block_id = (
        pd.Series(source.block_id).reset_index(drop=True)
        if source.block_id is not None
        else pd.Series(np.zeros(len(frame), dtype=int))
    )
    n = len(frame)
    q = QualityReport(
        n_rows=n,
        valid=np.ones(n, dtype=bool),
        validation_failed=np.zeros(n, dtype=bool),
        range_violation=np.zeros(n, dtype=bool),
        outlier=np.zeros(n, dtype=bool),
        filled=np.zeros(n, dtype=bool),
        unfilled=np.zeros(n, dtype=bool),
    )
    if missing_cols:
        q.notes.append(
            f"{len(missing_cols)} baseline feature(s) absent from this stream: "
            f"{missing_cols[:5]}"
        )

    _validate(frame, timestamps, stats, config, q)
    frame = _impute(frame, block_id, config, q)
    frame = _flag_outliers(frame, stats, config, q)
    frame = _filter(frame, block_id, config, q)
    raw_frame = frame.copy(deep=False)
    normalized = _normalize(frame, stats, config, q)

    # Validity for windowing. An outlier flag does NOT invalidate a row: the
    # whole point is that unusual post-drift values reach the detector.
    q.valid = ~(q.unfilled | normalized.isna().any(axis=1).to_numpy())
    q.notes.append(
        "valid = row is complete after causal imputation. Outlier and range "
        "flags do NOT invalidate a row -- excluding them would discard the "
        "post-drift observations under study"
    )

    adaptation = str((((config.get("preprocessing") or {})
                       .get("normalization")) or {})
                     .get("adaptation", "frozen_after_baseline"))
    return PreprocessedStream(
        key=source.key,
        role=source.role,
        frame=normalized,
        raw_frame=raw_frame,
        timestamps=timestamps,
        block_id=block_id,
        quality=q,
        statistics=stats,
        adaptation=adaptation,
    )


# =============================================================================
# Windowed feature extraction  (this is what stream.py deliberately omitted)
# =============================================================================


@dataclass(frozen=True)
class FeatureMatrix:
    """Windowed feature vectors, plus per-window metadata kept OUT of X."""

    names: tuple[str, ...]
    X: np.ndarray
    window_ids: np.ndarray
    start_index: np.ndarray
    end_index: np.ndarray
    start_time: pd.Series
    end_time: pd.Series
    n_outlier_flags: np.ndarray
    n_range_flags: np.ndarray
    valid_fraction: np.ndarray

    @property
    def shape(self) -> tuple[int, int]:
        return self.X.shape

    def as_frame(self) -> pd.DataFrame:
        return pd.DataFrame(self.X, columns=list(self.names))


def feature_names(
    config: Mapping[str, Any], stats: BaselineStatistics
) -> tuple[str, ...]:
    """Deterministic feature ordering: channel-major, then statistic.

    Fixed order matters: a model trained in Phase 2 and an ablation run later
    must index the same column for the same quantity.
    """
    fe = ((((config.get("preprocessing") or {}).get("windowing")) or {})
          .get("feature_extraction")) or {}
    cont = tuple(fe.get("continuous") or CONTINUOUS_STATS)
    disc = tuple(fe.get("discrete") or DISCRETE_STATS)
    bad_c = [s for s in cont if s not in CONTINUOUS_STATS]
    bad_d = [s for s in disc if s not in DISCRETE_STATS]
    if bad_c or bad_d:
        raise ConfigError(
            f"unknown feature_extraction statistic(s): continuous={bad_c} "
            f"discrete={bad_d}. Supported: {CONTINUOUS_STATS} / {DISCRETE_STATS}"
        )
    names = [f"{c}__{s}" for c in stats.continuous for s in cont]
    names += [f"{c}__{s}" for c in stats.discrete for s in disc]
    return tuple(names)


def _window_block(
    values: np.ndarray, wanted: Sequence[str], slope_x: np.ndarray
) -> np.ndarray:
    """Continuous statistics for one window. values: (w, n_cols)."""
    out = []
    for s in wanted:
        if s == "mean":
            out.append(values.mean(axis=0))
        elif s == "std":
            out.append(values.std(axis=0, ddof=0))
        elif s == "min":
            out.append(values.min(axis=0))
        elif s == "max":
            out.append(values.max(axis=0))
        elif s == "last":
            out.append(values[-1])
        elif s == "slope":
            yc = values - values.mean(axis=0)
            out.append((slope_x @ yc) / float(slope_x @ slope_x))
    return np.stack(out, axis=1).ravel()


def _discrete_block(values: np.ndarray, wanted: Sequence[str]) -> np.ndarray:
    """Discrete statistics for one window. Mode and change-count only."""
    out = []
    for s in wanted:
        if s == "mode":
            col_modes = []
            for j in range(values.shape[1]):
                vals, counts = np.unique(values[:, j], return_counts=True)
                col_modes.append(vals[int(np.argmax(counts))])
            out.append(np.asarray(col_modes, dtype=float))
        elif s == "n_changes":
            out.append((np.diff(values, axis=0) != 0).sum(axis=0).astype(float))
    return np.stack(out, axis=1).ravel()


def extract_features(
    config: Mapping[str, Any],
    prepared: PreprocessedStream,
    windows: Iterable[Any],
) -> FeatureMatrix:
    """Turn raw window slices from stream.py into feature vectors.

    Reads only rows inside each window, so it inherits stream.py's no-lookahead
    guarantee rather than re-implementing it.
    """
    fe = ((((config.get("preprocessing") or {}).get("windowing")) or {})
          .get("feature_extraction")) or {}
    cont_stats = tuple(fe.get("continuous") or CONTINUOUS_STATS)
    disc_stats = tuple(fe.get("discrete") or DISCRETE_STATS)
    names = feature_names(config, prepared.statistics)

    cont_cols = [c for c in prepared.statistics.continuous
                 if c in prepared.frame.columns]
    disc_cols = [c for c in prepared.statistics.discrete
                 if c in prepared.frame.columns]
    # Discrete channels are summarised from RAW values: the mode of a z-scored
    # state is not an interpretable state, and n_changes must count real
    # transitions.
    Cn = prepared.frame.loc[:, cont_cols].to_numpy(dtype=float)
    Dn = prepared.raw_frame.loc[:, disc_cols].to_numpy(dtype=float)
    out_flags = prepared.quality.outlier
    rng_flags = prepared.quality.range_violation
    valid = prepared.quality.valid

    rows, ids, si, ei, nof, nrf, vf = [], [], [], [], [], [], []
    slope_x: np.ndarray | None = None
    for w in windows:
        s, e = w.start_index, w.end_index
        if slope_x is None or slope_x.shape[0] != (e - s):
            xs = np.arange(e - s, dtype=float)
            slope_x = xs - xs.mean()
        parts = []
        if cont_cols:
            parts.append(_window_block(Cn[s:e], cont_stats, slope_x))
        if disc_cols:
            parts.append(_discrete_block(Dn[s:e], disc_stats))
        rows.append(np.concatenate(parts) if parts else np.zeros(0))
        ids.append(w.window_id)
        si.append(s)
        ei.append(e)
        nof.append(int(out_flags[s:e].sum()))
        nrf.append(int(rng_flags[s:e].sum()))
        vf.append(float(valid[s:e].mean()))

    X = np.vstack(rows) if rows else np.zeros((0, len(names)))
    if X.shape[1] != len(names):
        raise PreprocessingError(
            f"feature matrix has {X.shape[1]} columns but {len(names)} names "
            f"were declared; the ordering contract is broken"
        )
    si_a = np.asarray(si, dtype=int)
    ei_a = np.asarray(ei, dtype=int)
    return FeatureMatrix(
        names=names,
        X=X,
        window_ids=np.asarray(ids, dtype=int),
        start_index=si_a,
        end_index=ei_a,
        start_time=prepared.timestamps.iloc[si_a].reset_index(drop=True)
        if len(si_a) else pd.Series([], dtype="datetime64[ns]"),
        end_time=prepared.timestamps.iloc[ei_a - 1].reset_index(drop=True)
        if len(ei_a) else pd.Series([], dtype="datetime64[ns]"),
        n_outlier_flags=np.asarray(nof, dtype=int),
        n_range_flags=np.asarray(nrf, dtype=int),
        valid_fraction=np.asarray(vf, dtype=float),
    )


# =============================================================================
# The drift-absorption diagnostic  (makes ASSUMPTION [A6] measurable)
# =============================================================================


def measure_drift_absorption(
    config: Mapping[str, Any],
    source: Any,
    stats_by_mode: Mapping[str, BaselineStatistics],
    affected: Sequence[str],
    drift_start_index: int,
) -> dict[str, dict[str, float]]:
    """Quantify how much of an injected shift each adaptation mode destroys.

    For each mode, measure the post-drift minus pre-drift mean of the affected
    channels IN NORMALIZED SPACE. `frozen_after_baseline` should retain the
    injected shift; the adaptive modes should show it shrinking. That difference
    is the whole reason the default is frozen.

    `drift_start_index` comes from the ground-truth sidecar and is used ONLY to
    slice for this diagnostic -- an evaluation-time measurement, never an input
    to any detector or model.
    """
    out: dict[str, dict[str, float]] = {}
    cols = [c for c in affected if c in source.frame.columns]
    for mode, stats in stats_by_mode.items():
        cfg = {k: (dict(v) if isinstance(v, dict) else v) for k, v in config.items()}
        pp = {k: (dict(v) if isinstance(v, dict) else v)
              for k, v in (cfg.get("preprocessing") or {}).items()}
        norm = dict(pp.get("normalization") or {})
        norm["adaptation"] = mode
        pp["normalization"] = norm
        cfg["preprocessing"] = pp
        prepared = transform(cfg, source, stats)
        Z = prepared.frame.loc[:, cols].to_numpy(dtype=float)
        pre = Z[:drift_start_index].mean(axis=0)
        post = Z[drift_start_index:].mean(axis=0)
        shift = float(np.mean(post - pre))
        out[mode] = {
            "mean_normalized_shift_sigma": shift,
            "per_channel_min": float(np.min(post - pre)),
            "per_channel_max": float(np.max(post - pre)),
        }
    baseline_mode = "frozen_after_baseline"
    if baseline_mode in out:
        ref = out[baseline_mode]["mean_normalized_shift_sigma"]
        for mode, v in out.items():
            v["retained_fraction_vs_frozen"] = (
                v["mean_normalized_shift_sigma"] / ref if ref else float("nan")
            )
    return out
