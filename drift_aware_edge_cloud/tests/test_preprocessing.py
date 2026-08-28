"""The causal preprocessing pipeline.

Two failure modes drive this file. LEAKAGE: nothing may be fitted on the
inference stream. DRIFT CANCELLATION: an adaptive normalizer subtracts an
injected mean shift back out, so the stream looks unchanged and the experiment
silently measures nothing. Both are tested by measurement, not by inspection.
"""

from __future__ import annotations

import copy
import dataclasses

import numpy as np
import pandas as pd
import pytest

from src.data import loader, preprocessing as pp, stream


def _variant(cfg, path, value):
    c = copy.deepcopy(cfg)
    node = c
    parts = path.split(".")
    for k in parts[:-1]:
        node = node.setdefault(k, {})
    node[parts[-1]] = value
    return c


# -----------------------------------------------------------------------------
# Fitting happens on the baseline, or not at all
# -----------------------------------------------------------------------------


def test_statistics_are_fitted_on_the_baseline_only(cfg, stats, baseline, profile):
    assert stats.source_keys == tuple(lf.key for lf in baseline)
    assert stats.n_rows == profile.n_rows
    assert len(stats.continuous) == len(profile.continuous) and len(stats.discrete) == len(profile.discrete)


def test_fitting_on_the_inference_stream_is_refused(cfg, infer, profile):
    """This is `scaler.fit(all_data)` under another name."""
    with pytest.raises(loader.CausalityError):
        pp.fit(cfg, [infer], profile)


def test_fitting_on_baseline_plus_inference_is_refused(cfg, baseline, infer, profile):
    with pytest.raises(loader.CausalityError):
        pp.fit(cfg, [*baseline, infer], profile)


def test_disabling_the_global_fit_guard_is_refused(cfg, baseline, profile):
    c = _variant(cfg, "preprocessing.normalization.forbid_global_fit", False)
    with pytest.raises(loader.ConfigError, match="forbid_global_fit"):
        pp.fit(c, baseline, profile)


def test_fitting_with_no_baseline_is_refused(cfg, profile):
    with pytest.raises((loader.ConfigError, pp.PreprocessingError)):
        pp.fit(cfg, [], profile)


def test_baseline_statistics_match_an_independent_recomputation(stats, baseline):
    frame = baseline[0].frame
    for name in list(stats.continuous)[:10]:
        col = frame[name].to_numpy(dtype=float)
        assert stats.mean[name] == pytest.approx(float(np.mean(col)),
                                                 rel=1e-12, abs=1e-12)
        assert stats.std[name] == pytest.approx(float(np.std(col, ddof=0)),
                                                rel=1e-9, abs=1e-12)
        assert stats.q1[name] == pytest.approx(float(np.quantile(col, 0.25)))
        assert stats.q3[name] == pytest.approx(float(np.quantile(col, 0.75)))
        assert stats.iqr[name] == pytest.approx(stats.q3[name] - stats.q1[name])


def test_history_tail_is_kept_only_for_rolling(cfg, baseline, profile):
    assert pp.fit(cfg, baseline, profile).history_tail is None
    c = _variant(cfg, "preprocessing.normalization.adaptation", "rolling")
    tail = pp.fit(c, baseline, profile).history_tail
    w = int(c["preprocessing"]["normalization"]["rolling_window"])
    assert tail is not None and len(tail) == w


def test_an_unknown_adaptation_mode_is_refused(cfg, baseline, profile):
    c = _variant(cfg, "preprocessing.normalization.adaptation", "psychic")
    with pytest.raises(loader.ConfigError, match="adaptation"):
        pp.fit(c, baseline, profile)


# -----------------------------------------------------------------------------
# Frozen normalization is self-consistent on the data it was fitted to
# -----------------------------------------------------------------------------


def test_frozen_zscore_standardizes_the_baseline_it_was_fitted_on(cfg, baseline,
                                                                 stats):
    prepared = pp.transform(cfg, baseline[0], stats)
    cont = list(stats.continuous)
    m = prepared.frame[cont].mean().abs().max()
    s = (prepared.frame[cont].std(ddof=0) - 1.0).abs().max()
    assert m < 1e-9, f"max |mean| on the baseline is {m:g}"
    assert s < 1e-9, f"max |std - 1| on the baseline is {s:g}"


def test_normalization_uses_frozen_statistics_on_the_inference_stream(
        cfg, infer, stats, prepared_clean):
    name = stats.continuous[0]
    expected = ((infer.frame[name].to_numpy(dtype=float) - stats.mean[name])
                / stats.std[name])
    assert np.allclose(prepared_clean.frame[name].to_numpy(), expected,
                       equal_nan=True)


def test_running_minmax_is_refused_as_a_ratchet(cfg, baseline, profile):
    c = _variant(cfg, "preprocessing.normalization.method", "minmax")
    c = _variant(c, "preprocessing.normalization.adaptation", "running")
    with pytest.raises(loader.ConfigError):
        pp.transform(c, baseline[0], pp.fit(c, baseline, profile))


# -----------------------------------------------------------------------------
# The pipeline is non-destructive
# -----------------------------------------------------------------------------


def test_every_row_survives_the_pipeline(prepared_clean, infer):
    q = prepared_clean.quality
    assert q.n_rows == len(infer)
    assert len(prepared_clean.frame) == len(infer)
    assert len(prepared_clean.raw_frame) == len(infer)


def test_raw_values_are_preserved_alongside_the_normalized_ones(prepared_clean,
                                                               infer, stats):
    name = stats.continuous[0]
    assert np.allclose(prepared_clean.raw_frame[name].to_numpy(),
                       infer.frame[name].to_numpy(dtype=float), equal_nan=True)


def test_flags_are_metadata_and_never_become_features(prepared_clean, features,
                                                      stats):
    q = prepared_clean.quality
    for attr in ("validation_failed", "range_violation", "outlier", "filled"):
        assert len(getattr(q, attr)) == q.n_rows
        assert attr not in prepared_clean.frame.columns
    assert not any("validation_failed" in n or "outlier" in n
                   for n in features.names), \
        "a preprocessing artefact correlated with drift must not be a feature"


def test_an_outlier_flag_does_not_invalidate_a_row(prepared_clean):
    q = prepared_clean.quality
    assert q.outlier.any(), "nothing was flagged, so this test proves nothing"
    both = q.outlier & q.valid
    assert both.any(), "flagged rows were silently dropped from the stream"


def test_validity_is_exactly_unfilled_or_nan(prepared_clean):
    q = prepared_clean.quality
    expected = ~(q.unfilled | prepared_clean.frame.isna().any(axis=1))
    assert bool((q.valid == expected).all())


def test_dropping_rows_is_refused_at_every_stage(cfg, baseline, profile, infer,
                                                 stats):
    """A drop rule would delete exactly the post-drift observations the
    experiment exists to detect."""
    for path, match in [
        ("preprocessing.validation.on_failure", "drop"),
        ("preprocessing.outliers.action", "drop"),
    ]:
        c = _variant(cfg, path, "drop")
        with pytest.raises(loader.ConfigError):
            pp.transform(c, infer, stats)
    c = _variant(cfg, "preprocessing.missing.method", "drop_window")
    with pytest.raises(loader.ConfigError):
        pp.transform(c, infer, stats)


def test_backward_interpolation_is_refused_as_lookahead(cfg, infer, stats):
    """`backward_only` names the direction the fill LOOKS: at earlier rows only.
    Anything else reads a value that has not arrived."""
    assert cfg["preprocessing"]["missing"]["interpolate_direction"] == \
        "backward_only"
    for direction in ("both", "forward", "forward_only"):
        c = _variant(cfg, "preprocessing.missing.interpolate_direction",
                     direction)
        c = _variant(c, "preprocessing.missing.method", "interpolate")
        with pytest.raises(pp.LeakageError):
            pp.transform(c, infer, stats)


def test_the_imputation_guard_does_not_depend_on_the_data_having_holes(
        cfg, infer, stats, prepared_clean):
    """test1 contains no missing cells, so an acausal setting must still be
    refused on the configuration alone."""
    assert int(prepared_clean.quality.filled.sum()) == 0
    c = _variant(cfg, "preprocessing.missing.method", "interpolate")
    c = _variant(c, "preprocessing.missing.interpolate_direction", "both")
    with pytest.raises(pp.LeakageError):
        pp.transform(c, infer, stats)


def test_an_unknown_imputation_method_is_refused(cfg, infer, stats):
    c = _variant(cfg, "preprocessing.missing.method", "guess")
    with pytest.raises(loader.ConfigError, match="missing.method"):
        pp.transform(c, infer, stats)


def test_outlier_statistics_may_only_come_from_the_frozen_baseline(cfg, infer,
                                                                  stats):
    c = _variant(cfg, "preprocessing.outliers.statistics_source", "stream")
    with pytest.raises(pp.LeakageError):
        pp.transform(c, infer, stats)


def test_an_acausal_filter_is_refused(cfg, infer, stats):
    c = _variant(cfg, "preprocessing.filtering.enabled", True)
    c = _variant(c, "preprocessing.filtering.causal", False)
    with pytest.raises(pp.LeakageError):
        pp.transform(c, infer, stats)


def test_range_bounds_may_only_come_from_the_baseline(cfg, infer, stats):
    c = _variant(cfg, "preprocessing.validation.range_source", "stream")
    with pytest.raises((pp.LeakageError, loader.ConfigError)):
        pp.transform(c, infer, stats)


# -----------------------------------------------------------------------------
# Open finding 3: the 3xIQR rule degenerates on constant setpoints
# -----------------------------------------------------------------------------


def test_degenerate_outlier_bounds_are_self_reporting(prepared_clean):
    """Behaviour is deliberately unchanged pending a decision; what must not
    happen is a 100% flag rate presented as a clean result."""
    q = prepared_clean.quality
    if float(q.outlier.mean()) > 0.5:
        assert q.degenerate_bound_columns, \
            "a near-total flag rate with no degeneracy note is a silent artefact"
        assert any("DEGENERATE" in n for n in q.notes)
        assert any("EXCLUDING" in n for n in q.notes), \
            "the note must quantify the rate without the degenerate channels"


def test_zero_width_bounds_come_from_q1_equals_q3(stats, cfg, prepared_clean):
    lo, hi = stats.outlier_bounds(cfg)
    for c in prepared_clean.quality.degenerate_bound_columns:
        assert stats.q1[c] == stats.q3[c]
        assert hi[c] - lo[c] <= 0.0


def test_zscore_outlier_rule_is_available_and_not_degenerate(cfg, stats):
    c = _variant(cfg, "preprocessing.outliers.method", "zscore")
    lo, hi = stats.outlier_bounds(c)
    widths = (hi - lo)[list(stats.continuous)]
    assert float(widths.min()) > 0.0


def test_domain_outlier_bounds_are_refused_because_none_were_measured(cfg, stats):
    c = _variant(cfg, "preprocessing.outliers.method", "domain")
    with pytest.raises(loader.ConfigError):
        stats.outlier_bounds(c)


# -----------------------------------------------------------------------------
# Drift must survive normalization  (ASSUMPTION [A6], settled by measurement)
# -----------------------------------------------------------------------------


def test_injected_drift_survives_frozen_normalization(prepared_drift, injected,
                                                      stats):
    _, gt = injected
    start = gt.drift_start_index
    cols = [c for c in gt.affected_features if c in prepared_drift.frame.columns]
    pre = prepared_drift.frame[cols].iloc[:start].mean()
    post = prepared_drift.frame[cols].iloc[start:].mean()
    shift = float((post - pre).mean())
    assert shift > 1.0, (
        f"only {shift:.4f} sigma of the injected shift reached normalized space; "
        f"the Edge model would never degrade and ADWIN would detect nothing")


def test_the_clean_stream_shows_no_such_shift(prepared_clean, injected):
    """Confounder control: the same slice on the undrifted stream."""
    _, gt = injected
    start = gt.drift_start_index
    cols = [c for c in gt.affected_features if c in prepared_clean.frame.columns]
    pre = prepared_clean.frame[cols].iloc[:start].mean()
    post = prepared_clean.frame[cols].iloc[start:].mean()
    assert abs(float((post - pre).mean())) < 0.5


def test_adaptive_normalizers_absorb_the_drift_which_is_why_frozen_is_default(
        cfg_sudden, baseline, profile, injected):
    ds, gt = injected
    by_mode = {}
    for mode in pp.ADAPTATION_MODES:
        c = _variant(cfg_sudden, "preprocessing.normalization.adaptation", mode)
        by_mode[mode] = pp.fit(c, baseline, profile)
    measured = pp.measure_drift_absorption(
        cfg_sudden, ds, by_mode, gt.affected_features, gt.drift_start_index)

    frozen = measured["frozen_after_baseline"]["mean_normalized_shift_sigma"]
    running = measured["running"]["mean_normalized_shift_sigma"]
    rolling = measured["rolling"]["mean_normalized_shift_sigma"]
    assert frozen > running > rolling, \
        f"frozen={frozen:.4f} running={running:.4f} rolling={rolling:.4f}"
    assert measured["frozen_after_baseline"]["retained_fraction_vs_frozen"] == \
        pytest.approx(1.0)
    assert measured["rolling"]["retained_fraction_vs_frozen"] < 0.25, \
        "a rolling scaler cancels the drift; using it would measure nothing"


# -----------------------------------------------------------------------------
# Feature extraction
# -----------------------------------------------------------------------------


def test_feature_names_are_deterministic_and_channel_major(cfg_sudden, stats):
    names = pp.feature_names(cfg_sudden, stats)
    assert names == pp.feature_names(cfg_sudden, stats)
    assert len(names) == len(stats.continuous) * 6 + len(stats.discrete) * 2
    first = stats.continuous[0]
    assert names[:6] == tuple(f"{first}__{s}" for s in pp.CONTINUOUS_STATS)
    assert all(n.count("__") == 1 for n in names)


def test_unknown_statistics_are_refused(cfg_sudden, stats):
    c = _variant(cfg_sudden,
                 "preprocessing.windowing.feature_extraction.continuous",
                 ["mean", "kurtosis_of_vibes"])
    with pytest.raises(loader.ConfigError):
        pp.feature_names(c, stats)


def test_feature_matrix_shape_matches_the_window_grid(features, windows,
                                                      cfg_sudden, stats):
    assert features.shape == (len(windows),
                              len(pp.feature_names(cfg_sudden, stats)))
    assert list(features.window_ids) == [w.window_id for w in windows]
    assert list(features.start_index) == [w.start_index for w in windows]
    assert not np.isnan(features.X).any()


def test_every_statistic_matches_an_independent_recomputation(features, windows,
                                                             prepared_drift,
                                                             stats):
    names = list(features.names)
    frame = prepared_drift.frame
    raw = prepared_drift.raw_frame
    for wi in (0, len(windows) // 2, len(windows) - 1):
        w = windows[wi]
        row = features.X[wi]
        for c in list(stats.continuous)[:6]:
            y = frame[c].to_numpy(dtype=float)[w.start_index:w.end_index]
            x = np.arange(len(y), dtype=float)
            xc = x - x.mean()
            expect = {
                "mean": np.mean(y), "std": np.std(y, ddof=0),
                "min": np.min(y), "max": np.max(y), "last": y[-1],
                "slope": float(xc @ (y - y.mean()) / (xc @ xc)),
            }
            for s, v in expect.items():
                got = row[names.index(f"{c}__{s}")]
                assert got == pytest.approx(v, rel=1e-9, abs=1e-9), \
                    f"window {wi} {c}__{s}"
        for c in list(stats.discrete)[:3]:
            v = raw[c].to_numpy()[w.start_index:w.end_index]
            vals, counts = np.unique(v, return_counts=True)
            assert row[names.index(f"{c}__mode")] == \
                pytest.approx(float(vals[int(np.argmax(counts))]))
            assert row[names.index(f"{c}__n_changes")] == \
                pytest.approx(float((np.diff(v) != 0).sum()))


def test_discrete_statistics_are_read_from_raw_not_normalized_values(
        features, prepared_drift, stats, windows):
    """The mode of a z-scored state is not an interpretable state."""
    c = stats.discrete[0]
    col = features.X[:, list(features.names).index(f"{c}__mode")]
    raw_states = set(np.unique(prepared_drift.raw_frame[c].to_numpy()))
    assert set(np.unique(col)) <= {float(s) for s in raw_states}


def test_no_lookahead_poisoning_rows_after_a_window_cannot_change_it(
        features, windows, prepared_drift, cfg_sudden):
    wk = windows[len(windows) // 3]
    poisoned = prepared_drift.frame.copy()
    poisoned.iloc[wk.end_index:, :] = 999999.0
    poisoned_raw = prepared_drift.raw_frame.copy()
    poisoned_raw.iloc[wk.end_index:, :] = 999999.0
    prep_poisoned = dataclasses.replace(prepared_drift, frame=poisoned,
                                        raw_frame=poisoned_raw)
    again = pp.extract_features(cfg_sudden, prep_poisoned, [wk])
    original = features.X[list(features.window_ids).index(wk.window_id)]
    assert np.array_equal(again.X[0], original), \
        "the window's features moved when future rows changed: that is lookahead"


def test_extraction_is_deterministic_and_idempotent(cfg_sudden, prepared_drift,
                                                    windows, features):
    again = pp.extract_features(cfg_sudden, prepared_drift, windows[:50])
    assert np.array_equal(again.X, features.X[:50])


def test_drift_reaches_the_feature_matrix(features, injected, cfg_sudden):
    ds, gt = injected
    grid = stream.window_index(cfg_sudden, len(ds))
    onset = gt.drift_start_index
    names = list(features.names)
    cols = [names.index(f"{c}__mean") for c in gt.affected_features
            if f"{c}__mean" in names]
    pre = [i for i, (s, e) in enumerate(grid) if e <= onset and i < len(features.X)]
    post = [i for i, (s, _) in enumerate(grid) if s >= onset and i < len(features.X)]
    shift = float(features.X[np.ix_(post, cols)].mean()
                  - features.X[np.ix_(pre, cols)].mean())
    assert shift > 1.0, f"window-level shift is only {shift:.4f} sigma"


# -----------------------------------------------------------------------------
# The clean and drifted paths are the same code
# -----------------------------------------------------------------------------


def test_a_drifted_stream_and_a_loaded_file_take_the_same_code_path(
        cfg, cfg_sudden, infer, stats, injected):
    a = pp.transform(cfg, infer, stats)
    b = pp.transform(cfg_sudden, injected[0], stats)
    assert type(a) is type(b) is pp.PreprocessedStream
    assert a.frame.shape == b.frame.shape
    assert list(a.frame.columns) == list(b.frame.columns)
    assert a.adaptation == b.adaptation == "frozen_after_baseline"


def test_transform_is_deterministic(cfg, infer, stats, prepared_clean):
    again = pp.transform(cfg, infer, stats)
    assert again.frame.equals(prepared_clean.frame)
    assert int(again.quality.outlier.sum()) == \
        int(prepared_clean.quality.outlier.sum())
