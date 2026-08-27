"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/data/generator.py
Phase    : Phase 1 / Step 3
Status   : IMPLEMENTED

Apply CONTROLLED SYNTHETIC DRIFT to real HAI observations.

This is NOT an independent IoT sensor generator. It never fabricates data. It
takes a real HAI stream that loader.py read from disk and perturbs selected
channels according to a configured schedule, returning a new frame. The raw
file and the input frame are left untouched.

Scenarios : none | sudden | gradual | recurring | stress
Mechanisms: offset | scale | noise | correlation

Design commitments
------------------
* INFERENCE STREAM ONLY. `drift.injection_target` is `inference_stream_only`,
  and this module refuses any frame whose role is not `inference_stream`. The
  baseline stays as recorded, so post-baseline degradation is attributable to
  the injected drift rather than to a moving reference distribution.
* SIGMA FROM THE BASELINE. Magnitudes in `baseline_sigma` units are scaled by
  BaselineProfile.sigma(), which loader.py can only build from baseline files.
  There is no code path that derives sigma from the stream being drifted.
* DISCRETE CHANNELS ARE NOT OFFSET. A fractional offset on a 1/2 state variable
  is physically impossible. Continuous mechanisms are applied to continuous
  channels only; discrete channels change solely under the explicit
  `state_flip` policy.
* LABELS ARE NEVER MODIFIED (`drift.modify_labels: false`). Drift is injected in
  feature space, producing covariate shift without semantically impossible
  records.
* REALISED MAGNITUDE IS MEASURED, NOT ASSUMED. Physical clipping attenuates the
  injected drift. The realised per-channel shift is measured after clipping and
  reported, so attenuation is visible instead of hidden.

GROUND TRUTH IS QUARANTINED
---------------------------
`inject` returns `(drifted_frame, GroundTruth)` as two separate objects. The
ground truth is NEVER written into the frame, never added as a column, and never
returned inside it. Per `ground_truth.forbidden_consumers`, it may reach only
metrics, plots, and statistical evaluation.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd

from src.data.loader import (
    INFERENCE_ROLE,
    BaselineProfile,
    ConfigError,
    CausalityError,
    LoadedFile,
)

# Component id for seed derivation. Fixed forever: changing it would silently
# change every previously-recorded run's realised noise.
_SEED_COMPONENT_DRIFT = 1001

SCENARIOS = ("none", "sudden", "gradual", "recurring", "stress")
MECHANISMS = ("offset", "scale", "noise", "correlation")


# -----------------------------------------------------------------------------
# Ground truth  (EVALUATION ONLY)
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class GroundTruth:
    """What the generator knows because it created the drift.

    Required to measure detection delay. Forbidden everywhere else. Kept as a
    separate return value precisely so that passing it to a detector has to be
    an explicit, reviewable act rather than an accident of data flow.
    """

    scenario: str
    mechanism: str
    drift_start_index: int | None
    drift_end_index: int | None
    affected_features: tuple[str, ...]
    drift_magnitude: float
    magnitude_units: str
    realised_magnitude: dict[str, float]
    random_seed: int
    n_rows: int
    n_drifted_rows: int
    schedule_summary: dict[str, Any] = field(default_factory=dict)
    notes: tuple[str, ...] = ()

    def to_dict(self, fields: Sequence[str] | None = None) -> dict[str, Any]:
        """Serialise, optionally restricted to `ground_truth.fields`."""
        full = {
            "scenario": self.scenario,
            "mechanism": self.mechanism,
            "drift_start_index": self.drift_start_index,
            "drift_end_index": self.drift_end_index,
            "affected_features": list(self.affected_features),
            "drift_magnitude": self.drift_magnitude,
            "magnitude_units": self.magnitude_units,
            "realised_magnitude": self.realised_magnitude,
            "random_seed": self.random_seed,
            "n_rows": self.n_rows,
            "n_drifted_rows": self.n_drifted_rows,
            "schedule_summary": self.schedule_summary,
            "notes": list(self.notes),
        }
        if fields is None:
            return full
        missing = [f for f in fields if f not in full]
        if missing:
            raise ConfigError(f"ground_truth.fields names unknown field(s): {missing}")
        return {f: full[f] for f in fields}


def write_sidecar(
    gt: GroundTruth, config: Mapping[str, Any], *, root: Path | str = "."
) -> Path | None:
    """Write the ground truth to its sidecar file, outside the feature stream."""
    gt_cfg = config.get("ground_truth") or {}
    if not gt_cfg.get("emit_sidecar", False):
        return None
    path = Path(root) / str(gt_cfg.get("sidecar_path", "data/synthetic/ground_truth.json"))
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = gt.to_dict(gt_cfg.get("fields"))
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    return path


# -----------------------------------------------------------------------------
# Seeding
# -----------------------------------------------------------------------------


def _rng(config: Mapping[str, Any]) -> tuple[np.random.Generator, int]:
    """Deterministic generator derived from the master seed.

    Sub-seeding by component means changing the drift randomness cannot perturb
    any other stochastic component's stream, which is what makes a seed sweep
    interpretable.
    """
    repro = config.get("reproducibility") or {}
    master = repro.get("random_seed")
    if master is None:
        if repro.get("strict", True):
            raise ConfigError(
                "reproducibility.random_seed is missing and "
                "reproducibility.strict is true"
            )
        master = 0
    master = int(master)
    seq = np.random.SeedSequence([master, _SEED_COMPONENT_DRIFT])
    return np.random.default_rng(seq), master


# -----------------------------------------------------------------------------
# Position and schedule
# -----------------------------------------------------------------------------


def _resolve_index(fraction: Any, index: Any, n: int, *, what: str) -> int | None:
    """Resolve a position to a row index. `*_index` takes precedence over `*_fraction`."""
    if index is not None:
        i = int(index)
        if not 0 <= i <= n:
            raise ConfigError(f"drift.{what}_index={i} outside [0, {n}]")
        return i
    if fraction is None:
        return None
    f = float(fraction)
    if not 0.0 <= f <= 1.0:
        raise ConfigError(f"drift.{what}_fraction={f} outside [0, 1]")
    return int(round(f * n))


def magnitude_schedule(
    config: Mapping[str, Any], n_rows: int
) -> tuple[np.ndarray, int | None, int | None, dict[str, Any]]:
    """Per-row requested magnitude m(t), in the configured units.

    Returns (schedule, start_index, end_index, summary). The schedule is a pure
    function of row index, so it is reproducible and contains no randomness.
    """
    d = config.get("drift") or {}
    scenario = str(d.get("scenario", "none"))
    if scenario not in SCENARIOS:
        raise ConfigError(f"drift.scenario must be one of {SCENARIOS}, got {scenario!r}")

    magnitude = float(d.get("magnitude", 0.0) or 0.0)
    schedule = np.zeros(n_rows, dtype=float)
    summary: dict[str, Any] = {"scenario": scenario}

    if scenario == "none" or magnitude == 0.0:
        summary["reason"] = (
            "scenario is 'none'" if scenario == "none" else "magnitude is 0.0"
        )
        return schedule, None, None, summary

    start = _resolve_index(d.get("start_fraction"), d.get("start_index"), n_rows,
                           what="start")
    if start is None:
        raise ConfigError(
            f"drift.scenario is {scenario!r} but neither start_fraction nor "
            f"start_index is set"
        )
    end = _resolve_index(d.get("end_fraction"), None, n_rows, what="end")
    if end is None:
        end = n_rows
    if end <= start:
        raise ConfigError(f"drift end index {end} must exceed start index {start}")

    if scenario == "sudden":
        schedule[start:end] = magnitude
        summary.update(start=start, end=end, plateau=magnitude)

    elif scenario == "gradual":
        dur_frac = d.get("duration_fraction")
        if dur_frac is None:
            raise ConfigError("drift.scenario is 'gradual' but duration_fraction is null")
        ramp = int(round(float(dur_frac) * n_rows))
        if ramp <= 0:
            raise ConfigError("drift.duration_fraction resolves to a zero-length ramp")
        ramp_end = min(start + ramp, end)
        # Linear interpolation 0 -> magnitude across the ramp, then hold.
        span = ramp_end - start
        if span > 0:
            schedule[start:ramp_end] = np.linspace(
                magnitude / span, magnitude, span, endpoint=True
            )
        schedule[ramp_end:end] = magnitude
        summary.update(start=start, ramp_end=ramp_end, end=end,
                       ramp_rows=span, plateau=magnitude)

    elif scenario == "recurring":
        rec = d.get("recurring") or {}
        period = int(round(float(rec.get("period_fraction", 0.15)) * n_rows))
        n_cycles = int(rec.get("n_cycles", 2))
        if period <= 0:
            raise ConfigError("drift.recurring.period_fraction resolves to zero rows")
        segments = []
        cursor = start
        for _ in range(n_cycles):
            b_start, b_end = cursor, min(cursor + period, end)   # regime B (drifted)
            if b_start >= end:
                break
            schedule[b_start:b_end] = magnitude
            segments.append((b_start, b_end))
            cursor = b_end + period                              # regime A (clean)
        summary.update(start=start, end=end, period_rows=period,
                       n_cycles=n_cycles, drifted_segments=segments,
                       plateau=magnitude)

    elif scenario == "stress":
        st = d.get("stress") or {}
        steps = int(st.get("steps", 5))
        if steps <= 0:
            raise ConfigError("drift.stress.steps must be >= 1")
        if not st.get("progressive", True):
            schedule[start:end] = magnitude
            summary.update(start=start, end=end, progressive=False, plateau=magnitude)
        else:
            bounds = np.linspace(start, end, steps + 1).round().astype(int)
            plateaus = []
            for k in range(steps):
                lo, hi = int(bounds[k]), int(bounds[k + 1])
                level = magnitude * (k + 1) / steps
                schedule[lo:hi] = level
                plateaus.append({"from": lo, "to": hi, "magnitude": level})
            summary.update(start=start, end=end, progressive=True,
                           steps=steps, plateaus=plateaus, terminal=magnitude)

    return schedule, start, end, summary


# -----------------------------------------------------------------------------
# Channel selection
# -----------------------------------------------------------------------------


def select_features(
    config: Mapping[str, Any], profile: BaselineProfile, rng: np.random.Generator
) -> tuple[tuple[str, ...], tuple[str, ...], tuple[str, ...]]:
    """Choose which channels drift. Returns (continuous, discrete, notes).

    `actuator_policy: exclude` means discrete channels are never returned in the
    discrete bucket, so no continuous mechanism can reach them.
    """
    d = config.get("drift") or {}
    af = d.get("affected_features") or {}
    policy = str(af.get("actuator_policy", "exclude"))
    if policy not in {"exclude", "state_flip"}:
        raise ConfigError(
            f"drift.affected_features.actuator_policy must be 'exclude' or "
            f"'state_flip', got {policy!r}"
        )
    notes: list[str] = []

    explicit = af.get("explicit")
    if explicit:
        unknown = [c for c in explicit if c not in profile.columns]
        if unknown:
            raise ConfigError(
                f"drift.affected_features.explicit names column(s) absent from "
                f"the baseline profile: {unknown}"
            )
        chosen = tuple(explicit)
    else:
        selection = str(af.get("selection", "top_variance"))
        k = int(af.get("n_features", 0))
        if k <= 0:
            raise ConfigError("drift.affected_features.n_features must be >= 1")
        pool = profile.continuous
        if k > len(pool):
            notes.append(
                f"n_features={k} exceeds the {len(pool)} continuous channels "
                f"available; using all of them"
            )
            k = len(pool)
        if selection == "top_variance":
            chosen = profile.top_variance(k)
        elif selection == "random":
            picked = rng.choice(np.asarray(pool, dtype=object), size=k, replace=False)
            chosen = tuple(sorted(str(c) for c in picked))
        elif selection == "all_continuous":
            chosen = tuple(pool)
        else:
            raise ConfigError(
                f"drift.affected_features.selection must be 'top_variance', "
                f"'random' or 'all_continuous', got {selection!r}"
            )

    continuous = tuple(c for c in chosen if not profile.columns[c].is_discrete)
    discrete = tuple(c for c in chosen if profile.columns[c].is_discrete)

    if discrete and policy == "exclude":
        notes.append(
            f"actuator_policy 'exclude': {len(discrete)} discrete channel(s) "
            f"{list(discrete)} were selected but will NOT be modified -- a "
            f"continuous offset on a discrete state is physically impossible"
        )
        discrete = ()
    return continuous, discrete, tuple(notes)


# -----------------------------------------------------------------------------
# Mechanisms
# -----------------------------------------------------------------------------


def _physical_bounds(
    p: Any, config: Mapping[str, Any]
) -> tuple[float, float] | None:
    """Plausible value range for a channel, from the baseline observed range.

    `physical_range_expansion: 1.5` means the permitted band is 1.5x the width of
    the baseline range, centred on it. A drifted value outside that band is
    treated as physically implausible and clipped.
    """
    d = config.get("drift") or {}
    if not d.get("clip_to_physical_range", True):
        return None
    source = str(d.get("physical_range_source", "baseline"))
    if source != "baseline":
        raise ConfigError(
            f"drift.physical_range_source {source!r} is not supported in Phase 1; "
            f"only 'baseline' bounds are measured rather than invented"
        )
    expansion = float(d.get("physical_range_expansion", 1.0))
    if expansion < 1.0:
        raise ConfigError(
            f"drift.physical_range_expansion={expansion} < 1.0 would clip inside "
            f"the observed baseline range, deleting real data"
        )
    half = (p.value_range * expansion) / 2.0
    centre = (p.maximum + p.minimum) / 2.0
    return centre - half, centre + half


def _apply_continuous(
    values: np.ndarray,
    schedule: np.ndarray,
    sigma: float,
    mechanism: str,
    units: str,
    rng: np.random.Generator,
) -> np.ndarray:
    """Apply one continuous mechanism to one channel. Pure; returns a new array."""
    if units == "baseline_sigma":
        step = schedule * sigma
    elif units == "absolute":
        step = schedule.copy()
    elif units == "relative":
        step = schedule * np.abs(values)
    else:
        raise ConfigError(
            f"drift.magnitude_units must be 'baseline_sigma', 'absolute' or "
            f"'relative', got {units!r}"
        )

    if mechanism == "offset":
        return values + step
    if mechanism == "scale":
        # Gain degradation: relative by definition, so `step` is not sigma-scaled.
        return values * (1.0 + schedule)
    if mechanism == "noise":
        scale = np.where(step > 0, step, 0.0)
        draw = rng.standard_normal(values.shape)
        return values + draw * scale
    raise ConfigError(f"mechanism {mechanism!r} is not a per-channel mechanism")


def _apply_correlation(
    frame: pd.DataFrame,
    columns: Sequence[str],
    schedule: np.ndarray,
    profile: BaselineProfile,
) -> tuple[dict[str, np.ndarray], tuple[str, ...]]:
    """Rotate consecutive channel PAIRS, changing their relationship.

    Standardised by baseline mean/sigma first so the rotation is scale-free,
    then mapped back. Rotation angle is m * pi/4, i.e. magnitude 1.0 rotates the
    pair by 45 degrees. This is the mechanism that produces genuine concept
    drift -- a change in the relationship between features -- without touching
    any label.
    """
    out: dict[str, np.ndarray] = {}
    notes: list[str] = []
    theta = schedule * (np.pi / 4.0)
    cos_t, sin_t = np.cos(theta), np.sin(theta)

    for i in range(0, len(columns) - 1, 2):
        a, b = columns[i], columns[i + 1]
        pa, pb = profile.columns[a], profile.columns[b]
        sa = pa.std if pa.std > 0 else 1.0
        sb = pb.std if pb.std > 0 else 1.0
        za = (frame[a].to_numpy(dtype=float) - pa.mean) / sa
        zb = (frame[b].to_numpy(dtype=float) - pb.mean) / sb
        out[a] = (za * cos_t - zb * sin_t) * sa + pa.mean
        out[b] = (za * sin_t + zb * cos_t) * sb + pb.mean

    if len(columns) % 2 == 1:
        notes.append(
            f"mechanism 'correlation' rotates pairs; {columns[-1]!r} was left "
            f"unmodified because an odd number of channels was selected"
        )
    return out, tuple(notes)


# -----------------------------------------------------------------------------
# Entry point
# -----------------------------------------------------------------------------


@dataclass(frozen=True)
class DriftedStream:
    """The drifted inference stream. Ground truth is deliberately NOT in here."""

    key: str
    role: str
    frame: pd.DataFrame
    timestamps: pd.Series
    block_id: pd.Series
    schedule: np.ndarray

    def __len__(self) -> int:
        return len(self.frame)


def inject(
    config: Mapping[str, Any],
    stream: LoadedFile,
    profile: BaselineProfile,
) -> tuple[DriftedStream, GroundTruth]:
    """Inject controlled synthetic drift into a real HAI inference stream.

    Parameters
    ----------
    config  : resolved configuration mapping.
    stream  : the inference stream as read by loader.py. Not modified.
    profile : baseline statistics from loader.profile_baseline. The only source
              of sigma and of the continuous/discrete split.

    Returns
    -------
    (DriftedStream, GroundTruth) -- two separate objects, on purpose.
    """
    d = config.get("drift") or {}

    # --- guards --------------------------------------------------------------
    target = str(d.get("injection_target", "inference_stream_only"))
    if target != "inference_stream_only":
        raise ConfigError(
            f"drift.injection_target must be 'inference_stream_only', got "
            f"{target!r}. Perturbing the baseline would make drift unmeasurable: "
            f"the reference distribution would move with it."
        )
    if stream.role != INFERENCE_ROLE:
        raise CausalityError(
            f"inject() received {stream.key!r} with role {stream.role!r}. Drift "
            f"is injected into the {INFERENCE_ROLE!r} file only (ASSUMPTION "
            f"[A18]); the baseline must remain as recorded."
        )
    if stream.key in profile.source_keys:
        raise CausalityError(
            f"{stream.key!r} is both the inference stream and a source of the "
            f"baseline profile. Drift sigma would then be derived from the data "
            f"being drifted, which is circular."
        )
    ref = str(d.get("reference_stream", "inference_stream"))
    if ref not in {"inference_stream", "post_baseline"}:
        raise ConfigError(f"drift.reference_stream {ref!r} is not recognised")
    if d.get("modify_labels", False):
        raise ConfigError(
            "drift.modify_labels is true. Phase 1 injects drift in feature space "
            "only (ASSUMPTION [A13]); relabelling risks semantically impossible "
            "records."
        )

    mechanism = str(d.get("mechanism", "offset"))
    if mechanism not in MECHANISMS:
        raise ConfigError(f"drift.mechanism must be one of {MECHANISMS}, got {mechanism!r}")
    units = str(d.get("magnitude_units", "baseline_sigma"))

    rng, master_seed = _rng(config)
    n = len(stream)
    schedule, start, end, summary = magnitude_schedule(config, n)
    notes: list[str] = []

    # `enforce_drift_after_baseline` holds by construction under separate-file
    # splitting -- the baseline is a different file. Recorded, not assumed.
    split = config.get("split") or {}
    if split.get("enforce_drift_after_baseline", True):
        if str(split.get("mode")) == "separate_file":
            notes.append(
                "enforce_drift_after_baseline satisfied by construction: the "
                f"baseline is file(s) {list(profile.source_keys)}, drift is "
                f"injected into {stream.key!r}"
            )
        else:
            frac = float(split.get("baseline_fraction", 0.0))
            boundary = int(round(frac * n))
            if start is not None and start < boundary:
                raise ConfigError(
                    f"drift starts at row {start} but the leading-fraction "
                    f"baseline occupies rows 0..{boundary}. Injecting drift "
                    f"inside the baseline would contaminate the reference "
                    f"distribution."
                )

    drifted = stream.frame.copy(deep=True)

    # --- no drift ------------------------------------------------------------
    if start is None:
        gt = GroundTruth(
            scenario=str(d.get("scenario", "none")),
            mechanism=mechanism,
            drift_start_index=None,
            drift_end_index=None,
            affected_features=(),
            drift_magnitude=float(d.get("magnitude", 0.0) or 0.0),
            magnitude_units=units,
            realised_magnitude={},
            random_seed=master_seed,
            n_rows=n,
            n_drifted_rows=0,
            schedule_summary=summary,
            notes=tuple(notes + ["control condition: stream returned unmodified"]),
        )
        return (
            DriftedStream(stream.key, stream.role, drifted, stream.timestamps,
                          stream.block_id, schedule),
            gt,
        )

    # --- select channels -----------------------------------------------------
    continuous, discrete, sel_notes = select_features(config, profile, rng)
    notes.extend(sel_notes)
    if not continuous and not discrete:
        raise ConfigError(
            "drift is configured but no channel was selected for it. Check "
            "drift.affected_features and the continuous/discrete split."
        )

    active = schedule > 0
    n_drifted = int(active.sum())
    realised: dict[str, float] = {}

    # --- apply ---------------------------------------------------------------
    if mechanism == "correlation":
        rotated, rot_notes = _apply_correlation(stream.frame, continuous, schedule, profile)
        notes.extend(rot_notes)
        new_values = rotated
    else:
        new_values = {}
        for c in continuous:
            new_values[c] = _apply_continuous(
                stream.frame[c].to_numpy(dtype=float),
                schedule,
                profile.sigma(c),
                mechanism,
                units,
                rng,
            )

    # --- discrete state flips (only under the explicit policy) ---------------
    af = d.get("affected_features") or {}
    if discrete:
        p_flip = float(af.get("state_flip_probability", 0.0))
        if p_flip <= 0.0:
            notes.append(
                "actuator_policy is 'state_flip' but state_flip_probability is "
                "0.0, so no discrete channel was modified"
            )
        else:
            for c in discrete:
                original = stream.frame[c].to_numpy()
                states = np.unique(original)
                if states.size < 2:
                    notes.append(f"{c!r} has a single state; not flippable")
                    continue
                # Flip probability scales with the schedule, so discrete drift
                # follows the same onset shape as the continuous mechanism.
                m_max = float(np.max(schedule)) or 1.0
                prob = p_flip * (schedule / m_max)
                draw = rng.random(original.shape[0])
                flip = draw < prob
                if not flip.any():
                    continue
                replacement = original.copy()
                for i in np.flatnonzero(flip):
                    alternatives = states[states != original[i]]
                    replacement[i] = alternatives[rng.integers(alternatives.size)]
                drifted[c] = replacement
                realised[c] = float(np.mean(replacement[active] != original[active]))

    # --- clip to physical plausibility, then MEASURE what actually landed -----
    #
    # Two rules, both non-negotiable:
    #
    # 1. Clipping applies ONLY to rows where the drift is active. Bounding the
    #    whole column would alter genuine undrifted observations, so the
    #    pre-drift region would no longer be the recorded HAI stream.
    # 2. Clipping never moves a value past where the raw observation already
    #    was. HAI's inference stream legitimately visits states outside the
    #    baseline's expanded range; pulling those back would delete the real
    #    behaviour the experiment exists to observe (README constraint 5). The
    #    effective bound is therefore the band UNIONED with the original value.
    original_frame = stream.frame
    promoted: list[str] = []
    for c, values in new_values.items():
        p = profile.columns[c]
        original = original_frame[c].to_numpy(dtype=float)
        was_integral = pd.api.types.is_integer_dtype(original_frame[c].dtype)
        bounds = _physical_bounds(p, config)
        if bounds is not None:
            lo, hi = bounds
            lo_eff = np.minimum(lo, original)
            hi_eff = np.maximum(hi, original)
            clipped = np.where(active, np.clip(values, lo_eff, hi_eff), values)
            n_clipped = int(np.sum(active & (clipped != values)))
            n_already_out = int(np.sum(active & ((original < lo) | (original > hi))))
            if n_clipped:
                notes.append(
                    f"{c}: {n_clipped:,} drifted value(s) clipped to the plausible "
                    f"band [{lo:.4g}, {hi:.4g}]"
                )
            if n_already_out:
                notes.append(
                    f"{c}: {n_already_out:,} raw observation(s) in the drift region "
                    f"already lay outside that band and were NOT pulled back"
                )
            values = clipped
        # Rows outside the drift window keep the recorded value exactly.
        values = np.where(active, values, original)
        drifted[c] = values
        if was_integral:
            promoted.append(c)

        # Realised magnitude: the shift that survived clipping, in sigma units,
        # over the drifted rows only. Requested vs realised is the honest report.
        if d.get("report_realised_magnitude", True) and n_drifted and p.std > 0:
            delta = values[active] - original[active]
            if mechanism == "noise":
                realised[c] = float(np.std(delta) / p.std)
            else:
                realised[c] = float(np.mean(delta) / p.std)

    requested_mean = float(np.mean(schedule[active])) if n_drifted else 0.0
    if promoted:
        # Recorded, not hidden: a fractional-sigma offset is not representable in
        # an integer column, so the dtype must widen. VALUES outside the drift
        # window are unchanged; only the storage type is. Anything comparing
        # frames with pandas `.equals` will see this and should compare values.
        notes.append(
            f"dtype promoted int -> float on {len(promoted)} channel(s) "
            f"{promoted}: a fractional offset is not representable as an "
            f"integer. Undrifted values are bit-identical."
        )
    continuous_realised = [realised[c] for c in new_values if c in realised]
    if continuous_realised and mechanism in {"offset", "noise"} and units == "baseline_sigma":
        # Averaged over CONTINUOUS channels only. Discrete flip rates live in the
        # same dict but are a different quantity and must not be mixed in.
        achieved = float(np.mean(continuous_realised))
        summary["requested_mean_magnitude_sigma"] = requested_mean
        summary["realised_mean_magnitude_sigma"] = achieved
        if requested_mean > 0:
            ratio = achieved / requested_mean
            summary["attenuation_ratio"] = ratio
            if ratio < 0.95:
                notes.append(
                    f"physical clipping attenuated the drift: realised "
                    f"{achieved:.3f} sigma vs requested {requested_mean:.3f} "
                    f"sigma ({ratio:.1%} of target)"
                )

    gt = GroundTruth(
        scenario=str(d.get("scenario", "none")),
        mechanism=mechanism,
        drift_start_index=start,
        drift_end_index=end,
        affected_features=tuple(continuous) + tuple(discrete),
        drift_magnitude=float(d.get("magnitude", 0.0) or 0.0),
        magnitude_units=units,
        realised_magnitude=realised,
        random_seed=master_seed,
        n_rows=n,
        n_drifted_rows=n_drifted,
        schedule_summary=summary,
        notes=tuple(notes),
    )
    return (
        DriftedStream(stream.key, stream.role, drifted, stream.timestamps,
                      stream.block_id, schedule),
        gt,
    )
