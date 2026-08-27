"""Controlled drift injection into the real HAI inference stream.

The generator is the one component that knows the truth, so most of these tests
are about what it must NOT do: touch the baseline, leak ground truth into the
stream object, offset a discrete actuator, or report a requested magnitude it did
not actually deliver.
"""

from __future__ import annotations

import copy
import json

import numpy as np
import pytest

from src.data import generator, loader


# -----------------------------------------------------------------------------
# Schedules are pure functions of row index
# -----------------------------------------------------------------------------


def _sched(cfg, drift, n=1000):
    c = copy.deepcopy(cfg)
    c["drift"] = {**(cfg.get("drift") or {}), **drift}
    return generator.magnitude_schedule(c, n)


def test_control_condition_has_a_flat_zero_schedule(cfg):
    sch, start, end, summary = generator.magnitude_schedule(cfg, 1000)
    assert cfg["drift"]["scenario"] == "none"
    assert start is None and end is None
    assert not sch.any()
    assert "reason" in summary


def test_zero_magnitude_is_a_control_even_if_a_scenario_is_named(cfg):
    sch, start, _, summary = _sched(cfg, {"scenario": "sudden", "magnitude": 0.0,
                                          "start_fraction": 0.5})
    assert start is None and not sch.any()
    assert summary["reason"] == "magnitude is 0.0"


def test_sudden_schedule_is_a_step(cfg):
    sch, start, end, _ = _sched(cfg, {"scenario": "sudden", "magnitude": 2.0,
                                      "start_fraction": 0.5})
    assert start == 500 and end == 1000
    assert not sch[:500].any()
    assert np.allclose(sch[500:], 2.0)


def test_gradual_schedule_ramps_monotonically_then_holds(cfg):
    sch, start, end, summary = _sched(cfg, {"scenario": "gradual", "magnitude": 2.0,
                                            "start_fraction": 0.5,
                                            "duration_fraction": 0.2})
    ramp_end = summary["ramp_end"]
    assert start == 500 and ramp_end == 700
    assert not sch[:500].any()
    ramp = sch[500:ramp_end]
    assert np.all(np.diff(ramp) > 0), "a gradual drift must not jump"
    assert ramp[-1] == pytest.approx(2.0)
    assert np.allclose(sch[ramp_end:end], 2.0)


def test_recurring_schedule_alternates_regimes(cfg):
    sch, start, _, summary = _sched(cfg, {"scenario": "recurring", "magnitude": 2.0,
                                          "start_fraction": 0.2,
                                          "recurring": {"period_fraction": 0.1,
                                                        "n_cycles": 3}})
    segments = summary["drifted_segments"]
    assert len(segments) == 3
    for a, b in segments:
        assert np.allclose(sch[a:b], 2.0)
    for (a1, b1), (a2, _) in zip(segments, segments[1:]):
        assert not sch[b1:a2].any(), "the clean regime between cycles must be clean"


def test_stress_schedule_escalates_in_steps(cfg):
    sch, start, end, summary = _sched(cfg, {"scenario": "stress", "magnitude": 5.0,
                                            "start_fraction": 0.1,
                                            "stress": {"steps": 5,
                                                       "progressive": True}})
    levels = [p["magnitude"] for p in summary["plateaus"]]
    assert levels == sorted(levels) and len(levels) == 5
    assert levels[-1] == pytest.approx(5.0)
    assert sch[end - 1] == pytest.approx(5.0)
    assert sch[start] == pytest.approx(1.0)


def test_schedule_contains_no_randomness(cfg):
    a, *_ = _sched(cfg, {"scenario": "gradual", "magnitude": 2.0,
                         "start_fraction": 0.5, "duration_fraction": 0.2})
    b, *_ = _sched(cfg, {"scenario": "gradual", "magnitude": 2.0,
                         "start_fraction": 0.5, "duration_fraction": 0.2})
    assert np.array_equal(a, b)


@pytest.mark.parametrize("drift,match", [
    ({"scenario": "elephant", "magnitude": 1.0}, "drift.scenario"),
    ({"scenario": "sudden", "magnitude": 1.0}, "neither start_fraction"),
    ({"scenario": "sudden", "magnitude": 1.0, "start_fraction": 1.5}, "outside"),
    ({"scenario": "sudden", "magnitude": 1.0, "start_fraction": 0.5,
      "end_fraction": 0.2}, "must exceed"),
    ({"scenario": "gradual", "magnitude": 1.0, "start_fraction": 0.5,
      "duration_fraction": None}, "duration_fraction"),
])
def test_malformed_schedules_are_refused(cfg, drift, match):
    with pytest.raises(loader.ConfigError, match=match):
        _sched(cfg, drift)


# -----------------------------------------------------------------------------
# Channel selection
# -----------------------------------------------------------------------------


def test_selection_ranks_by_baseline_sigma_not_by_drifted_sigma(cfg, profile):
    c = copy.deepcopy(cfg)
    c["drift"]["affected_features"] = {"selection": "top_variance",
                                       "n_features": 5,
                                       "actuator_policy": "exclude"}
    cont, disc, _ = generator.select_features(c, profile, np.random.default_rng(0))
    assert cont == profile.top_variance(5)
    assert not disc


def test_discrete_actuators_are_never_given_a_continuous_offset(cfg, profile):
    """A fractional-sigma offset on a two-state valve is a physically
    impossible value. `actuator_policy: exclude` is what prevents it."""
    c = copy.deepcopy(cfg)
    c["drift"]["affected_features"] = {"explicit": list(profile.discrete[:2]),
                                       "actuator_policy": "exclude"}
    cont, disc, notes = generator.select_features(c, profile,
                                                  np.random.default_rng(0))
    assert cont == () and disc == ()
    assert any("physically impossible" in n for n in notes)


def test_explicit_selection_of_an_absent_column_is_refused(cfg, profile):
    c = copy.deepcopy(cfg)
    c["drift"]["affected_features"] = {"explicit": ["NOT_A_CHANNEL"],
                                       "actuator_policy": "exclude"}
    with pytest.raises(loader.ConfigError, match="absent from"):
        generator.select_features(c, profile, np.random.default_rng(0))


def test_random_selection_is_seed_reproducible(cfg, profile):
    c = copy.deepcopy(cfg)
    c["drift"]["affected_features"] = {"selection": "random", "n_features": 6,
                                       "actuator_policy": "exclude"}
    a, _, _ = generator.select_features(c, profile, np.random.default_rng(7))
    b, _, _ = generator.select_features(c, profile, np.random.default_rng(7))
    d, _, _ = generator.select_features(c, profile, np.random.default_rng(8))
    assert a == b and len(a) == 6
    assert a != d


def test_bad_actuator_policy_is_refused(cfg, profile):
    c = copy.deepcopy(cfg)
    c["drift"]["affected_features"] = {"selection": "top_variance",
                                       "n_features": 2,
                                       "actuator_policy": "smash"}
    with pytest.raises(loader.ConfigError, match="actuator_policy"):
        generator.select_features(c, profile, np.random.default_rng(0))


# -----------------------------------------------------------------------------
# Injection into the real stream
# -----------------------------------------------------------------------------


def test_the_control_config_returns_the_stream_unmodified(cfg, infer, profile):
    ds, gt = generator.inject(cfg, infer, profile)
    assert gt.drift_start_index is None and gt.n_drifted_rows == 0
    assert gt.affected_features == ()
    assert ds.frame.equals(infer.frame)


def test_injection_leaves_the_source_stream_untouched(cfg_sudden, infer, profile):
    before = infer.frame.copy(deep=True)
    generator.inject(cfg_sudden, infer, profile)
    assert infer.frame.equals(before), "inject() must not mutate its input"


def test_pre_drift_rows_are_the_recorded_hai_values(injected, infer):
    ds, gt = injected
    start = gt.drift_start_index
    for c in gt.affected_features:
        assert np.array_equal(ds.frame[c].to_numpy()[:start],
                              infer.frame[c].to_numpy(dtype=float)[:start]), c


def test_unaffected_channels_are_bit_identical_everywhere(injected, infer):
    ds, gt = injected
    untouched = [c for c in ds.frame.columns if c not in set(gt.affected_features)]
    assert len(untouched) == len(infer.frame.columns) - len(gt.affected_features)
    for c in untouched:
        assert ds.frame[c].equals(infer.frame[c]), c


def test_the_drift_actually_moved_the_affected_channels(injected, infer, profile):
    ds, gt = injected
    start, end = gt.drift_start_index, gt.drift_end_index
    assert gt.n_drifted_rows == end - start > 0
    for c in gt.affected_features:
        delta = (ds.frame[c].to_numpy()[start:end]
                 - infer.frame[c].to_numpy(dtype=float)[start:end])
        assert abs(float(np.mean(delta))) > 0.1 * profile.sigma(c), c


def test_realised_magnitude_is_measured_not_copied_from_the_request(injected,
                                                                   infer, profile):
    ds, gt = injected
    start, end = gt.drift_start_index, gt.drift_end_index
    assert gt.realised_magnitude, "an unmeasured injection cannot be reported"
    for c, reported in gt.realised_magnitude.items():
        delta = (ds.frame[c].to_numpy()[start:end]
                 - infer.frame[c].to_numpy(dtype=float)[start:end])
        recomputed = float(np.mean(delta)) / profile.sigma(c)
        assert reported == pytest.approx(recomputed, rel=1e-9, abs=1e-12), c
    # Clipping means realised <= requested; that gap is a finding, not a bug.
    assert min(gt.realised_magnitude.values()) <= gt.drift_magnitude + 1e-9


def test_clipping_never_pulls_a_raw_observation_back(injected, infer):
    """HAI legitimately visits states outside the baseline's expanded range.
    Outside the drift window nothing may be clipped at all."""
    ds, gt = injected
    end = gt.drift_end_index
    for c in gt.affected_features:
        assert np.array_equal(ds.frame[c].to_numpy()[end:],
                              infer.frame[c].to_numpy(dtype=float)[end:]), c


def test_injection_is_reproducible_and_seed_sensitive(cfg_sudden, infer, profile):
    a, gt_a = generator.inject(cfg_sudden, infer, profile)
    b, gt_b = generator.inject(cfg_sudden, infer, profile)
    assert a.frame.equals(b.frame) and gt_a == gt_b

    c = copy.deepcopy(cfg_sudden)
    c["reproducibility"]["random_seed"] = 43
    d, gt_d = generator.inject(c, infer, profile)
    assert gt_d.random_seed == 43
    assert gt_d.drift_start_index == gt_a.drift_start_index, \
        "the schedule is deterministic; only the noise realisation may move"


def test_noise_mechanism_changes_spread_not_mean(cfg_sudden, infer, profile):
    c = copy.deepcopy(cfg_sudden)
    c["drift"]["mechanism"] = "noise"
    ds, gt = generator.inject(c, infer, profile)
    start, end = gt.drift_start_index, gt.drift_end_index
    for ch in gt.affected_features:
        delta = (ds.frame[ch].to_numpy()[start:end]
                 - infer.frame[ch].to_numpy(dtype=float)[start:end])
        assert float(np.std(delta)) > 0
        assert abs(float(np.mean(delta))) < float(np.std(delta))


def test_scale_mechanism_is_multiplicative_about_the_baseline_mean(cfg_sudden,
                                                                  infer, profile):
    c = copy.deepcopy(cfg_sudden)
    c["drift"]["mechanism"] = "scale"
    ds, gt = generator.inject(c, infer, profile)
    assert gt.mechanism == "scale"
    start, end = gt.drift_start_index, gt.drift_end_index
    moved = [ch for ch in gt.affected_features
             if not np.allclose(ds.frame[ch].to_numpy()[start:end],
                                infer.frame[ch].to_numpy(dtype=float)[start:end])]
    assert moved, "a scale drift that changes nothing is not a drift"


# -----------------------------------------------------------------------------
# Guards: the baseline is off limits
# -----------------------------------------------------------------------------


def test_injecting_into_the_baseline_is_refused(cfg_sudden, baseline, profile):
    with pytest.raises(loader.CausalityError, match="role"):
        generator.inject(cfg_sudden, baseline[0], profile)


def test_injecting_into_a_profile_source_is_refused(cfg_sudden, infer, profile):
    import dataclasses
    circular = dataclasses.replace(profile, source_keys=(infer.key,))
    with pytest.raises(loader.CausalityError, match="circular"):
        generator.inject(cfg_sudden, infer, circular)


@pytest.mark.parametrize("patch,match", [
    ({"injection_target": "everything"}, "injection_target"),
    ({"modify_labels": True}, "modify_labels"),
    ({"mechanism": "teleport"}, "drift.mechanism"),
    ({"physical_range_expansion": 0.5}, "physical_range_expansion"),
    ({"physical_range_source": "invented"}, "physical_range_source"),
])
def test_integrity_weakening_drift_options_are_refused(cfg_sudden, infer, profile,
                                                       patch, match):
    c = copy.deepcopy(cfg_sudden)
    c["drift"].update(patch)
    with pytest.raises(loader.ConfigError, match=match):
        generator.inject(c, infer, profile)


# -----------------------------------------------------------------------------
# Ground truth stays outside the stream
# -----------------------------------------------------------------------------


def test_ground_truth_is_a_separate_object_from_the_stream(injected):
    ds, gt = injected
    for field in ("scenario", "drift_start_index", "drift_end_index",
                  "affected_features", "drift_magnitude", "random_seed"):
        assert not hasattr(ds, field), f"DriftedStream exposes ground truth: {field}"
    assert not set(ds.frame.columns) & set(generator.GroundTruth.__annotations__)


def test_the_schedule_on_the_stream_is_not_the_answer_key(injected):
    """`schedule` is the requested magnitude, needed to reproduce the run. It is
    still evaluation metadata: no Phase 1 consumer reads it, and it does not
    appear in the feature frame."""
    ds, gt = injected
    assert len(ds.schedule) == len(ds.frame)
    assert "schedule" not in ds.frame.columns


def test_sidecar_is_written_only_when_asked_and_only_declared_fields(cfg_sudden,
                                                                    injected,
                                                                    tmp_path):
    _, gt = injected
    c = copy.deepcopy(cfg_sudden)
    c["ground_truth"] = {"emit_sidecar": False}
    assert generator.write_sidecar(gt, c, root=tmp_path) is None

    fields = ["scenario", "drift_start_index", "affected_features"]
    c["ground_truth"] = {"emit_sidecar": True, "fields": fields,
                         "sidecar_path": "gt.json"}
    path = generator.write_sidecar(gt, c, root=tmp_path)
    payload = json.loads(path.read_text(encoding="utf-8"))
    assert list(payload) == fields
    assert payload["drift_start_index"] == gt.drift_start_index


def test_sidecar_refuses_an_unknown_field(injected):
    _, gt = injected
    with pytest.raises(loader.ConfigError, match="unknown field"):
        gt.to_dict(["scenario", "the_answer"])


def test_ground_truth_records_the_seed_that_produced_it(injected, cfg_sudden):
    _, gt = injected
    assert gt.random_seed == cfg_sudden["reproducibility"]["random_seed"]
    assert gt.magnitude_units == "baseline_sigma"
