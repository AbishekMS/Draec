"""Event logging: fixed schemas, no ragged rows, write-only by construction."""

from __future__ import annotations

import io
import logging

import pytest

from src.utils import logger as logmod

STREAM_ROW = dict(window_id=0, start_index=0, end_index=50, start_time="t0",
                  end_time="t1", n_rows=50, contiguous=True, valid_fraction=1.0)


def test_declared_schemas_have_unique_ordered_fields():
    for name, fields in logmod.SCHEMAS.items():
        assert isinstance(fields, tuple), f"{name}: order must be fixed"
        assert len(set(fields)) == len(fields), f"{name}: duplicate field"
        assert fields, f"{name}: empty schema"


def test_header_matches_declared_order(cfg, tmp_path):
    with logmod.EventLog.create("stream", config=cfg, root=tmp_path) as log:
        log.write(**STREAM_ROW)
    text = io.open(log.path, encoding="utf-8").read().strip().splitlines()
    assert text[0] == ",".join(logmod.SCHEMAS["stream"])
    assert len(text) == 2


def test_unknown_field_is_rejected(cfg, tmp_path):
    with logmod.EventLog.create("stream", config=cfg, root=tmp_path) as log:
        with pytest.raises(logmod.LoggingError, match="not in the declared schema"):
            log.write(**{**STREAM_ROW, "bogus": 1})


def test_missing_field_is_rejected_not_defaulted(cfg, tmp_path):
    with logmod.EventLog.create("stream", config=cfg, root=tmp_path) as log:
        partial = {k: v for k, v in STREAM_ROW.items() if k != "n_rows"}
        with pytest.raises(logmod.LoggingError, match="absent"):
            log.write(**partial)


def test_undeclared_schema_is_rejected(cfg, tmp_path):
    with pytest.raises(logmod.LoggingError, match="no declared schema"):
        logmod.EventLog.create("invented", config=cfg, root=tmp_path)


def test_explicit_fields_allow_an_ad_hoc_schema(cfg, tmp_path):
    with logmod.EventLog.create("adhoc", config=cfg, root=tmp_path,
                                fields=["a", "b"]) as log:
        log.write(a=1, b=2)
    assert io.open(log.path, encoding="utf-8").read().startswith("a,b")


def test_duplicate_field_names_are_rejected(cfg, tmp_path):
    with pytest.raises(logmod.LoggingError, match="duplicate"):
        logmod.EventLog.create("adhoc", config=cfg, root=tmp_path,
                               fields=["a", "a"])


def test_rendering_of_bools_sequences_and_none(cfg, tmp_path):
    with logmod.EventLog.create("adhoc", config=cfg, root=tmp_path,
                                fields=["flag", "items", "empty"]) as log:
        log.write(flag=True, items=["x", "y"], empty=None)
        log.write(flag=False, items=(1, 2), empty=None)
    rows = io.open(log.path, encoding="utf-8").read().strip().splitlines()
    assert rows[1] == "1,x|y,"
    assert rows[2] == "0,1|2,"


def test_log_is_write_only_by_construction(cfg, tmp_path):
    """The absence of a reader is the integrity mechanism -- see module docstring."""
    log = logmod.EventLog.create("stream", config=cfg, root=tmp_path)
    try:
        for forbidden in ("read", "load", "rows", "readlines", "iter_rows"):
            assert not hasattr(log, forbidden)
    finally:
        log.close()


def test_ground_truth_field_names_are_documented():
    assert {"drift_start_index", "affected_features", "random_seed"} <= \
        logmod.GROUND_TRUTH_FIELDS
    # No Phase 1 schema consumed by the pipeline may carry ground truth.
    for name, fields in logmod.SCHEMAS.items():
        leaked = set(fields) & logmod.GROUND_TRUTH_FIELDS
        assert not leaked, f"schema {name!r} carries ground truth {leaked}"


def test_writing_after_close_is_refused(cfg, tmp_path):
    log = logmod.EventLog.create("stream", config=cfg, root=tmp_path)
    log.close()
    with pytest.raises(logmod.LoggingError, match="closed"):
        log.write(**STREAM_ROW)


def test_write_many_counts_rows(cfg, tmp_path):
    with logmod.EventLog.create("stream", config=cfg, root=tmp_path) as log:
        n = log.write_many([STREAM_ROW, {**STREAM_ROW, "window_id": 1}])
    assert n == log.n_written == 2


def test_log_quality_records_measured_counts(cfg, tmp_path, prepared_clean):
    with logmod.EventLog.create("quality", config=cfg, root=tmp_path) as log:
        logmod.log_quality(log, "inference_stream", prepared_clean.quality)
    row = io.open(log.path, encoding="utf-8").read().strip().splitlines()[1]
    cells = dict(zip(logmod.SCHEMAS["quality"], row.split(",")))
    q = prepared_clean.quality
    assert cells["stage"] == "inference_stream"
    assert int(cells["n_rows"]) == q.n_rows
    assert int(cells["n_valid"]) == int(q.valid.sum())
    assert int(cells["n_outlier"]) == int(q.outlier.sum())
    assert int(cells["n_range_violation"]) == int(q.range_violation.sum())
    assert int(cells["n_unfilled"]) == int(q.unfilled.sum())


def test_configure_respects_level(cfg):
    buf = io.StringIO()
    logmod.configure(cfg, stream=buf, force=True)
    logmod.get_logger("t").info("shown")
    logmod.get_logger("t").debug("hidden")
    out = buf.getvalue()
    for h in list(logging.getLogger("dace").handlers):
        logging.getLogger("dace").removeHandler(h)
    assert "shown" in out and "hidden" not in out


def test_configure_rejects_a_bad_level(cfg):
    with pytest.raises(logmod.LoggingError, match="not one of"):
        logmod.configure(cfg, level="LOUD", force=True)


def test_logger_namespace(cfg):
    assert logmod.get_logger("x").name == "dace.x"
    assert logmod.get_logger("dace.y").name == "dace.y"
