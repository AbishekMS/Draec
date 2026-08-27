"""Loading, validating and profiling the real HAI recording.

Nothing here fabricates a value. Where a test needs a pathological file it
constructs one explicitly in `tmp_path` and says so -- the raw files under
`data/raw/` are never written to.
"""

from __future__ import annotations

import copy
import io

import numpy as np
import pandas as pd
import pytest

from src.data import loader


# -----------------------------------------------------------------------------
# File roles come from configuration, not from filenames
# -----------------------------------------------------------------------------


def test_roles_are_declared_in_config_not_inferred_from_filenames(cfg, project_root):
    specs = loader.file_specs(cfg, project_root)
    assert {s.role for s in specs.values()} == {"baseline_train",
                                               "baseline_validation",
                                               "inference_stream"}
    assert loader.inference_key(cfg) == "test1"
    assert loader.resolve_baseline_keys(cfg) == ("train1",)


def test_baseline_is_not_concatenated_with_the_inference_stream(cfg):
    assert cfg["dataset"]["concatenate_files"] is False
    assert loader.inference_key(cfg) not in loader.resolve_baseline_keys(cfg)


def test_baseline_roles_exclude_the_inference_role():
    assert loader.INFERENCE_ROLE not in loader.BASELINE_ROLES


def test_unknown_key_is_refused(cfg, project_root):
    with pytest.raises(loader.ConfigError, match="unknown dataset.files key"):
        loader.load_file(cfg, "train3", root=project_root)


# -----------------------------------------------------------------------------
# Measured properties of the recording
# -----------------------------------------------------------------------------


def test_declared_row_counts_are_verified_against_the_files(cfg, baseline, infer):
    for lf in [*baseline, infer]:
        declared = cfg["dataset"]["files"][lf.key].get("rows")
        assert declared is None or len(lf) == int(declared), lf.key
        assert lf.report.row_count_matches_config in (True, None)


def test_time_axis_is_monotonic_1hz_and_gapless(baseline, infer):
    for lf in [*baseline, infer]:
        ta = lf.report.time_axis
        assert ta.monotonic_increasing, lf.key
        assert ta.n_duplicate_timestamps == 0, lf.key
        assert ta.modal_interval_s == 1.0, lf.key
        assert ta.n_blocks >= 1


def test_timestamp_column_is_not_a_feature(baseline, infer, cfg):
    ts_col = cfg["dataset"]["timestamp_column"]
    for lf in [*baseline, infer]:
        assert ts_col not in lf.frame.columns
        assert len(lf.timestamps) == len(lf.frame)
        assert lf.timestamps.index.equals(lf.frame.index)
        assert lf.block_id.index.equals(lf.frame.index)


def test_frame_preserves_original_row_order(infer):
    assert infer.timestamps.is_monotonic_increasing
    assert list(infer.frame.index) == list(range(len(infer)))


def test_all_feature_columns_are_numeric(baseline, infer):
    for lf in [*baseline, infer]:
        assert not lf.report.non_numeric_columns, (lf.key,
                                                   lf.report.non_numeric_columns)
        assert all(pd.api.types.is_numeric_dtype(t) for t in lf.frame.dtypes)


def test_schemas_match_across_files(baseline, infer):
    cols = loader.assert_schema_match([*baseline, infer])
    assert len(cols) == 86


def test_schema_mismatch_is_detected(baseline, infer):
    import dataclasses
    shuffled = dataclasses.replace(
        infer, frame=infer.frame[list(reversed(list(infer.frame.columns)))])
    with pytest.raises(loader.SchemaError, match="schema mismatch"):
        loader.assert_schema_match([*baseline, shuffled])


# -----------------------------------------------------------------------------
# Causality
# -----------------------------------------------------------------------------


def test_the_chosen_baseline_is_causal(cfg, baseline, infer):
    loader.assert_causal_baseline(baseline, infer, cfg)   # must not raise
    for b in baseline:
        assert b.report.time_axis.last <= infer.report.time_axis.first


def test_train2_is_acausal_and_is_refused_as_a_baseline(cfg, project_root, infer):
    """HAI's recording order is train1 < test1 < train2. Using train2 to fit
    would be fitting on data recorded after inference began."""
    train2 = loader.load_file(cfg, "train2", root=project_root, max_rows=5_000)
    assert train2.report.time_axis.first > infer.report.time_axis.first
    with pytest.raises(loader.CausalityError, match="acausal baseline"):
        loader.assert_causal_baseline([train2], infer, cfg)


def test_acausal_ablation_is_possible_but_never_silent(cfg, project_root, infer):
    train2 = loader.load_file(cfg, "train2", root=project_root, max_rows=5_000)
    c = copy.deepcopy(cfg)
    c["dataset"]["allow_acausal_baseline"] = True
    with pytest.warns(RuntimeWarning, match="acausal baseline"):
        loader.assert_causal_baseline([train2], infer, c)


# -----------------------------------------------------------------------------
# Baseline profile: the single source of sigma
# -----------------------------------------------------------------------------


def test_profile_is_built_from_baseline_files_only(cfg, profile, baseline):
    assert profile.source_keys == tuple(lf.key for lf in baseline)
    assert loader.inference_key(cfg) not in profile.source_keys
    assert profile.n_rows == sum(len(lf) for lf in baseline)


def test_profiling_the_inference_stream_is_refused(cfg, infer):
    with pytest.raises(loader.CausalityError):
        loader.profile_baseline(cfg, [infer])


def test_profile_statistics_match_an_independent_recomputation(profile, baseline):
    frame = baseline[0].frame
    for name in list(profile.continuous)[:12]:
        col = frame[name].to_numpy(dtype=float)
        p = profile.columns[name]
        assert p.mean == pytest.approx(float(np.mean(col)), rel=1e-12, abs=1e-12)
        assert p.std == pytest.approx(float(np.std(col, ddof=0)),
                                      rel=1e-9, abs=1e-12)
        assert p.minimum == pytest.approx(float(np.min(col)))
        assert p.maximum == pytest.approx(float(np.max(col)))


def test_continuous_and_discrete_partition_the_features(profile):
    assert set(profile.continuous) | set(profile.discrete) == \
        set(profile.feature_names)
    assert not set(profile.continuous) & set(profile.discrete)
    assert len(profile.feature_names) == 66
    assert len(profile.continuous) == 58 and len(profile.discrete) == 8


def test_zero_variance_channels_are_dropped_and_reported(profile):
    """Dropped from the feature set, but their measured profile is retained --
    the measurement is a fact, and `zero_variance_agreement` reports whether it
    matched the list declared in config."""
    assert profile.dropped_zero_variance
    assert not set(profile.dropped_zero_variance) & set(profile.feature_names)
    assert not set(profile.dropped_zero_variance) & set(profile.continuous)
    assert not set(profile.dropped_zero_variance) & set(profile.discrete)
    for name in profile.dropped_zero_variance:
        assert profile.columns[name].zero_variance
        assert profile.columns[name].std == 0.0
    agreement = profile.zero_variance_agreement
    assert agreement["measured_n"] == len(profile.dropped_zero_variance)
    assert "measured wins" in agreement["note"]


def test_a_dropped_channel_refuses_to_supply_a_scale(profile):
    """0.0 is not a usable sigma: dividing by it would silently produce inf."""
    dropped = profile.dropped_zero_variance[0]
    with pytest.raises(loader.ConfigError, match="zero-variance"):
        profile.sigma(dropped)


def test_sigma_is_the_only_sanctioned_scale(profile):
    name = profile.continuous[0]
    assert profile.sigma(name) == profile.columns[name].std
    assert profile.sigma(name) > 0
    with pytest.raises(loader.ConfigError, match="no baseline profile"):
        profile.sigma("NOT_A_CHANNEL")


def test_top_variance_ranks_by_baseline_sigma_deterministically(profile):
    top = profile.top_variance(5)
    assert len(top) == 5
    sigmas = [profile.sigma(c) for c in top]
    assert sigmas == sorted(sigmas, reverse=True)
    assert profile.top_variance(5) == top
    assert all(c in profile.continuous for c in top)


# -----------------------------------------------------------------------------
# The unresolved task, and the sibling leakage trap
# -----------------------------------------------------------------------------


def test_target_resolution_refuses_to_invent_a_label(cfg, profile):
    assert cfg["dataset"]["task"] == "unresolved"
    with pytest.raises(loader.UnresolvedTaskError, match="no label"):
        loader.resolve_target(cfg, profile)


def test_regression_target_must_be_continuous(cfg, profile):
    c = copy.deepcopy(cfg)
    c["dataset"]["task"] = "forecasting_regression"
    c["dataset"]["target_column"] = profile.continuous[0]
    assert loader.resolve_target(c, profile) == profile.continuous[0]

    c["dataset"]["target_column"] = profile.discrete[0]
    with pytest.raises(loader.ConfigError, match="discrete"):
        loader.resolve_target(c, profile)


def test_state_classification_target_must_be_discrete(cfg, profile):
    c = copy.deepcopy(cfg)
    c["dataset"]["task"] = "state_classification"
    c["dataset"]["target_column"] = profile.discrete[0]
    assert loader.resolve_target(c, profile) == profile.discrete[0]

    c["dataset"]["target_column"] = profile.continuous[0]
    with pytest.raises(loader.ConfigError, match="continuous"):
        loader.resolve_target(c, profile)


def test_hai_label_task_requires_a_label_file_that_was_not_supplied(cfg):
    c = copy.deepcopy(cfg)
    c["dataset"]["task"] = "labels_from_hai_labels"
    with pytest.raises(loader.ConfigError, match="label_file"):
        loader.resolve_target(c)


def test_command_feedback_sibling_is_excluded_from_the_feature_set(cfg, profile):
    pairs = cfg["dataset"]["features"]["command_feedback_pairs"]
    assert pairs, "the sibling map is what closes the leak"
    tag, members = next(iter(pairs.items()))
    target = next(m for m in members if m in profile.feature_names)
    feats = loader.feature_names_for_target(cfg, profile, target)
    for m in members:
        assert m not in feats, f"{m} is a sibling of {target} and leaks it"
    assert len(feats) < len(profile.feature_names)


def test_sibling_exclusion_can_be_switched_off_only_deliberately(cfg, profile):
    c = copy.deepcopy(cfg)
    c["dataset"]["features"]["exclude_target_sibling"] = False
    pairs = cfg["dataset"]["features"]["command_feedback_pairs"]
    members = next(iter(pairs.values()))
    target = next(m for m in members if m in profile.feature_names)
    kept = loader.feature_names_for_target(c, profile, target)
    siblings = [m for m in members if m != target and m in profile.feature_names]
    assert any(m in kept for m in siblings)


# -----------------------------------------------------------------------------
# Structural violations raise instead of passing quietly
# -----------------------------------------------------------------------------


def _tiny_config(tmp_path, body, *, rows=None, monotonic=True):
    """A two-file toy dataset written into tmp_path.

    `file_specs` requires exactly one inference_stream, so a second file is
    declared to satisfy the role contract; only "tiny" is ever loaded.
    """
    io.open(tmp_path / "tiny.txt", "w", encoding="utf-8").write(body)
    io.open(tmp_path / "other.txt", "w", encoding="utf-8").write(GOOD)
    return {
        "dataset": {
            "mode": "hai", "delimiter": ";",
            "timestamp_column": "time", "timestamp_format": "%Y-%m-%d %H:%M:%S",
            "expected_sampling_interval_s": 1,
            "require_monotonic_timestamps": monotonic,
            "max_gap_s_before_new_block": 2,
            "concatenate_files": False, "allow_acausal_baseline": False,
            "task": "unresolved", "baseline_source": ["tiny"],
            "files": {
                "tiny": {"path": "tiny.txt", "role": "baseline_train",
                         "rows": rows},
                "other": {"path": "other.txt", "role": "inference_stream"},
            },
            "features": {"exclude_target_sibling": True,
                         "command_feedback_pairs": {},
                         "type_detection": {"method": "cardinality",
                                            "max_unique_for_discrete": 10}},
        },
        "split": {"max_baseline_rows": None},
        "streaming": {"max_samples": None},
    }


HEAD = "time;A;B\n"
GOOD = HEAD + "".join(f"2019-07-01 00:00:0{i};{i}.0;{i % 2}\n" for i in range(5))


def test_a_stale_row_count_raises_rather_than_passing_quietly(tmp_path):
    c = _tiny_config(tmp_path, GOOD, rows=999)
    with pytest.raises(loader.SchemaError, match="config declares rows"):
        loader.load_file(c, "tiny", root=tmp_path)


def test_out_of_order_timestamps_raise(tmp_path):
    body = HEAD + ("2019-07-01 00:00:02;1.0;0\n"
                   "2019-07-01 00:00:01;2.0;1\n"
                   "2019-07-01 00:00:03;3.0;0\n")
    c = _tiny_config(tmp_path, body)
    with pytest.raises(loader.TimeAxisError, match="monotonic"):
        loader.load_file(c, "tiny", root=tmp_path)


def test_a_time_gap_becomes_a_block_boundary_not_an_interpolation(tmp_path):
    body = HEAD + ("2019-07-01 00:00:01;1.0;0\n"
                   "2019-07-01 00:00:02;2.0;1\n"
                   "2019-07-01 00:10:00;3.0;0\n"
                   "2019-07-01 00:10:01;4.0;1\n")
    lf = loader.load_file(_tiny_config(tmp_path, body), "tiny", root=tmp_path)
    assert len(lf) == 4, "no row was invented to fill the gap"
    assert lf.report.time_axis.n_gaps == 1
    assert lf.report.time_axis.n_blocks == 2
    assert list(lf.block_id) == [0, 0, 1, 1]


def test_a_missing_cell_is_reported_not_silently_filled(tmp_path):
    body = HEAD + ("2019-07-01 00:00:01;1.0;0\n"
                   "2019-07-01 00:00:02;;1\n"
                   "2019-07-01 00:00:03;3.0;0\n")
    lf = loader.load_file(_tiny_config(tmp_path, body), "tiny", root=tmp_path)
    assert lf.report.n_missing_cells == 1
    assert lf.report.missing_by_column["A"] == 1
    assert not lf.report.clean and lf.report.findings
    assert bool(lf.frame["A"].isna().iloc[1]), "the hole is preserved for Step 4"


def test_max_rows_reads_a_prefix_not_a_sample(cfg, project_root):
    head = loader.load_file(cfg, "train1", root=project_root, max_rows=1_000)
    assert len(head) == 1000
    assert head.timestamps.is_monotonic_increasing
