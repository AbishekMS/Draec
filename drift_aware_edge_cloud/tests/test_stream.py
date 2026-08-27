"""Chronological replay: one pass, forwards, no lookahead.

The central property is mechanical rather than statistical -- a window is
assembled only after the row that completes it has arrived, and the engine
raises if that is ever violated. These tests attack that property directly by
poisoning rows the window must not have seen.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
import pytest

from src.data import loader, stream


# -----------------------------------------------------------------------------
# Plan validation
# -----------------------------------------------------------------------------


def test_plan_reads_the_config(cfg):
    p = stream.plan(cfg)
    win = cfg["preprocessing"]["windowing"]
    assert p.window_size == win["window_size"]
    assert p.step_size == win["step_size"]
    assert p.emits_windows and p.sampling_interval_s == 1.0


def test_shuffling_the_stream_is_refused(cfg):
    c = copy.deepcopy(cfg)
    c["streaming"]["shuffle"] = True
    with pytest.raises(loader.ConfigError, match="shuffle"):
        stream.plan(c)


@pytest.mark.parametrize("path,value,match", [
    (("streaming", "emit"), "telepathy", "streaming.emit"),
    (("streaming", "max_samples"), 0, "max_samples"),
    (("streaming", "realtime_factor"), 0, "realtime_factor"),
    (("preprocessing", "windowing", "window_size"), 0, "window_size"),
    (("preprocessing", "windowing", "step_size"), 0, "step_size"),
    (("preprocessing", "windowing", "min_valid_fraction"), 0.0,
     "min_valid_fraction"),
    (("preprocessing", "windowing", "min_valid_fraction"), 1.5,
     "min_valid_fraction"),
])
def test_malformed_stream_plans_are_refused(cfg, path, value, match):
    c = copy.deepcopy(cfg)
    node = c
    for k in path[:-1]:
        node = node[k]
    node[path[-1]] = value
    with pytest.raises(loader.ConfigError, match=match):
        stream.plan(c)


def test_a_step_larger_than_the_window_warns_about_unevaluated_rows(cfg):
    c = copy.deepcopy(cfg)
    c["preprocessing"]["windowing"]["step_size"] = \
        c["preprocessing"]["windowing"]["window_size"] + 10
    with pytest.warns(RuntimeWarning, match="never be evaluated"):
        stream.plan(c)


def test_a_stream_shorter_than_one_window_is_refused(cfg):
    with pytest.raises(loader.ConfigError, match="no window could ever"):
        stream.plan(cfg, n_rows=5)


# -----------------------------------------------------------------------------
# A synthetic-index toy source, so ordering can be tested without 54,000 rows
# -----------------------------------------------------------------------------


class ToySource:
    """A stream whose only channel is its own row index.

    Not fabricated experimental data -- it carries no physical meaning and is
    never measured. It exists so that "did this window read row k?" has an
    unambiguous arithmetic answer.
    """

    key = "toy"
    role = loader.INFERENCE_ROLE

    def __init__(self, n=100, blocks=None):
        self.frame = pd.DataFrame({"idx": np.arange(n, dtype=float)})
        self.timestamps = pd.Series(
            pd.date_range("2019-07-01", periods=n, freq="1s"))
        self.block_id = pd.Series(
            np.zeros(n, dtype=int) if blocks is None else blocks)


def _toy_cfg(cfg, window, step, **win):
    c = copy.deepcopy(cfg)
    c["preprocessing"]["windowing"].update(window_size=window, step_size=step,
                                           **win)
    c["streaming"]["max_samples"] = None
    c["streaming"]["realtime_factor"] = None
    return c


def test_windows_tile_the_stream_on_the_configured_grid(cfg):
    c = _toy_cfg(cfg, 10, 5)
    ws = list(stream.iter_windows(ToySource(100), c))
    assert [w.start_index for w in ws] == list(range(0, 91, 5))
    assert all(w.n_rows == 10 for w in ws)
    assert [w.window_id for w in ws] == list(range(len(ws)))


def test_a_window_contains_exactly_its_own_rows(cfg):
    c = _toy_cfg(cfg, 10, 10)
    for w in stream.iter_windows(ToySource(100), c):
        assert list(w.frame["idx"]) == list(range(w.start_index, w.end_index))


def test_no_lookahead_the_engine_raises_if_a_window_outruns_the_cursor(cfg):
    """Directly provoke the guard rather than trusting that it is unreachable."""
    c = _toy_cfg(cfg, 10, 10)
    engine = stream.ChronologicalStream(ToySource(100), c)
    with pytest.raises(stream.StreamOrderError, match="lookahead"):
        engine._build(0)          # cursor is still -1: no row has arrived


def test_a_window_is_emitted_after_the_row_that_completes_it(cfg):
    c = _toy_cfg(cfg, 10, 10)
    c["streaming"]["emit"] = "both"
    engine = stream.ChronologicalStream(ToySource(40), c)
    seen_rows = 0
    for kind, item in engine.events():
        if kind == "sample":
            seen_rows = item.index + 1
        else:
            assert item.end_index <= seen_rows, \
                "a window became available before its last row arrived"


def test_replaying_backwards_is_refused(cfg):
    engine = stream.ChronologicalStream(ToySource(50), _toy_cfg(cfg, 10, 10))
    engine._advance(5)
    with pytest.raises(stream.StreamOrderError, match="went backwards"):
        engine._advance(4)


def test_a_gap_block_boundary_suppresses_the_straddling_window(cfg):
    blocks = np.array([0] * 50 + [1] * 50)
    c = _toy_cfg(cfg, 10, 10, require_contiguous=True)
    ws = list(stream.iter_windows(ToySource(100, blocks), c))
    assert all(len(w.block_ids) == 1 for w in ws)
    assert all(w.contiguous for w in ws)

    engine = stream.ChronologicalStream(ToySource(100, blocks),
                                        _toy_cfg(cfg, 10, 3,
                                                 require_contiguous=True))
    emitted = list(engine.windows())
    assert engine.stats.n_windows_skipped_noncontiguous > 0
    assert engine.stats.n_windows_emitted == len(emitted)
    assert engine.stats.n_windows_candidate > len(emitted)


def test_non_contiguous_windows_can_be_kept_but_are_labelled(cfg):
    blocks = np.array([0] * 50 + [1] * 50)
    c = _toy_cfg(cfg, 10, 3, require_contiguous=False)
    ws = list(stream.iter_windows(ToySource(100, blocks), c))
    straddling = [w for w in ws if not w.contiguous]
    assert straddling, "with require_contiguous false they must be emitted"
    assert all(len(w.block_ids) > 1 for w in straddling)


def test_low_validity_windows_are_skipped_and_counted(cfg):
    mask = np.ones(100, dtype=bool)
    mask[20:25] = False
    c = _toy_cfg(cfg, 10, 10, min_valid_fraction=1.0)
    engine = stream.ChronologicalStream(ToySource(100), c, valid_mask=mask)
    ws = list(engine.windows())
    assert [w.start_index for w in ws] == [0, 10, 30, 40, 50, 60, 70, 80, 90]
    assert engine.stats.n_windows_skipped_invalid == 1


def test_a_partial_validity_threshold_admits_the_window_and_reports_it(cfg):
    mask = np.ones(100, dtype=bool)
    mask[20:22] = False
    c = _toy_cfg(cfg, 10, 10, min_valid_fraction=0.5)
    ws = list(stream.iter_windows(ToySource(100), c, valid_mask=mask))
    w = next(w for w in ws if w.start_index == 20)
    assert w.valid_fraction == pytest.approx(0.8)


def test_a_wrong_length_valid_mask_is_refused(cfg):
    with pytest.raises(loader.ConfigError, match="valid_mask has"):
        list(stream.iter_windows(ToySource(100), _toy_cfg(cfg, 10, 10),
                                 valid_mask=np.ones(99, dtype=bool)))


def test_trailing_rows_are_dropped_but_counted(cfg):
    engine = stream.ChronologicalStream(ToySource(105), _toy_cfg(cfg, 10, 10))
    list(engine.windows())
    assert engine.stats.trailing_rows_dropped == 5
    assert "5" in engine.stats.summary()


def test_max_samples_truncates_the_replay_and_says_so(cfg):
    c = _toy_cfg(cfg, 10, 10)
    c["streaming"]["max_samples"] = 40
    engine = stream.ChronologicalStream(ToySource(100), c)
    ws = list(engine.windows())
    assert len(ws) == 4 and ws[-1].end_index == 40
    assert engine.stats.n_rows_truncated_by_max_samples == 60
    assert any("max_samples" in n for n in engine.stats.notes)


def test_samples_arrive_once_each_in_order(cfg):
    engine = stream.ChronologicalStream(ToySource(50), _toy_cfg(cfg, 10, 10))
    idx = [s.index for s in engine.samples()]
    assert idx == list(range(50))
    assert engine.stats.n_samples_emitted == 50


def test_emit_dispatches_on_config(cfg):
    for emit, kind in [("samples", stream.Sample), ("windows", stream.Window)]:
        c = _toy_cfg(cfg, 10, 10)
        c["streaming"]["emit"] = emit
        first = next(iter(stream.ChronologicalStream(ToySource(50), c).run()))
        assert isinstance(first, kind)
    c = _toy_cfg(cfg, 10, 10)
    c["streaming"]["emit"] = "both"
    kind, _ = next(iter(stream.ChronologicalStream(ToySource(50), c).run()))
    assert kind == "sample"


# -----------------------------------------------------------------------------
# The evaluation-side grid helper
# -----------------------------------------------------------------------------


def test_window_index_matches_an_actual_replay(cfg):
    c = _toy_cfg(cfg, 10, 5)
    grid = stream.window_index(c, 100)
    replayed = [(w.start_index, w.end_index)
                for w in stream.iter_windows(ToySource(100), c)]
    assert grid == replayed


def test_window_index_reads_no_data(cfg):
    """Pure arithmetic: it takes a row count, never a frame."""
    assert stream.window_index(_toy_cfg(cfg, 10, 10), 100)[0] == (0, 10)


def test_window_index_maps_the_real_drift_onset(cfg_sudden, injected, windows):
    _, gt = injected
    grid = stream.window_index(cfg_sudden, len(injected[0]))
    onset = gt.drift_start_index
    first_fully_drifted = next(i for i, (s, _) in enumerate(grid) if s >= onset)
    straddling = [i for i, (s, e) in enumerate(grid) if s < onset < e]
    assert first_fully_drifted > 0
    assert len(straddling) >= 1, \
        "detection latency must be measured from a known onset window"


# -----------------------------------------------------------------------------
# Real HAI replay
# -----------------------------------------------------------------------------


def test_real_stream_replays_in_recorded_order(cfg_sudden, injected,
                                               prepared_drift, windows):
    ds = injected[0]
    assert len(windows) > 5000
    assert [w.window_id for w in windows] == list(range(len(windows)))
    times = [w.start_time for w in windows]
    assert times == sorted(times)
    assert all(w.start_time <= w.end_time for w in windows)
    assert windows[-1].end_index <= len(ds)


def test_real_windows_align_with_the_arithmetic_grid(cfg_sudden, injected, windows):
    grid = stream.window_index(cfg_sudden, len(injected[0]))
    assert [(w.start_index, w.end_index) for w in windows] == grid


def test_window_span_matches_1hz_sampling(windows):
    w = windows[0]
    assert w.span_s == pytest.approx(w.n_rows - 1)


# -----------------------------------------------------------------------------
# Label aggregation refuses to invent a label
# -----------------------------------------------------------------------------


def test_label_aggregation_refuses_without_a_positive_class(cfg):
    """Aggregation must not guess which class is positive.

    Before 2026-08-27 the shipped config had positive_class null, so this test
    could use it directly. The official labels now declare positive_class: 1, so
    the guard is exercised by removing it -- the refusal itself is unchanged.
    """
    assert cfg["dataset"]["positive_class"] == 1
    c = copy.deepcopy(cfg)
    c["dataset"]["positive_class"] = None
    with pytest.raises(loader.ConfigError, match="positive_class"):
        stream.aggregate_label([0, 0, 1], c)


def test_label_aggregation_modes_once_a_label_exists(cfg):
    c = copy.deepcopy(cfg)
    c["dataset"]["positive_class"] = 1
    assert stream.aggregate_label([0, 0, 1], c) == 1
    assert stream.aggregate_label([0, 0, 0], c) == 0

    c["preprocessing"]["windowing"]["label_aggregation"] = "majority"
    assert stream.aggregate_label([0, 0, 1], c) == 0
    c["preprocessing"]["windowing"]["label_aggregation"] = "last"
    assert stream.aggregate_label([0, 0, 1], c) == 1
    c["preprocessing"]["windowing"]["label_aggregation"] = "vibes"
    with pytest.raises(loader.ConfigError, match="label_aggregation"):
        stream.aggregate_label([0, 0, 1], c)


def test_an_empty_window_has_no_label(cfg):
    with pytest.raises(loader.ConfigError, match="empty window"):
        stream.aggregate_label([], cfg)
