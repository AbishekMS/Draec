"""Configuration resolution, validation and provenance."""

from __future__ import annotations

import copy
import io
import json

import pytest
import yaml

from src.utils import config as cfgmod

SHIPPED = ("default", "sudden_drift", "gradual_drift", "stress_test")


def test_all_shipped_configs_load(config_dir):
    for name in SHIPPED:
        assert cfgmod.load(name, config_dir=config_dir)["meta"]["phase"] == 1


def test_available_lists_shipped_configs(config_dir):
    assert set(SHIPPED) <= set(cfgmod.available(config_dir))


def test_deep_merge_replaces_lists_not_appends():
    merged = cfgmod.deep_merge({"a": [1, 2], "b": {"c": 1, "d": 2}},
                               {"a": [9], "b": {"c": 7}})
    assert merged["a"] == [9], "appending would make results depend on depth"
    assert merged["b"] == {"c": 7, "d": 2}


def test_overlays_touch_only_meta_and_drift(config_dir):
    for name in SHIPPED[1:]:
        touched = cfgmod.assert_overlay_discipline(f"{name}.yaml",
                                                   config_dir=config_dir)
        assert set(touched) <= set(cfgmod.OVERLAY_ALLOWED)
        assert "drift" in touched, "an overlay that changes no drift is pointless"


def test_overlay_touching_a_forbidden_section_is_refused(tmp_path):
    io.open(tmp_path / "base.yaml", "w", encoding="utf-8").write("meta: {name: b}\n")
    io.open(tmp_path / "bad.yaml", "w", encoding="utf-8").write(
        "_extends: base.yaml\ndrift: {scenario: sudden}\n"
        "streaming: {window_size: 999}\n")
    with pytest.raises(cfgmod.ConfigError, match="may only touch"):
        cfgmod.assert_overlay_discipline("bad.yaml", config_dir=tmp_path)


def test_base_config_is_not_an_overlay(config_dir):
    with pytest.raises(cfgmod.ConfigError, match="not an overlay"):
        cfgmod.assert_overlay_discipline("default.yaml", config_dir=config_dir)


def test_circular_extends_is_detected(tmp_path):
    io.open(tmp_path / "a.yaml", "w", encoding="utf-8").write("_extends: b.yaml\n")
    io.open(tmp_path / "b.yaml", "w", encoding="utf-8").write("_extends: a.yaml\n")
    with pytest.raises(cfgmod.ConfigError, match="circular"):
        cfgmod.resolve("a", config_dir=tmp_path)


def test_extends_must_be_single_inheritance(tmp_path):
    io.open(tmp_path / "base.yaml", "w", encoding="utf-8").write("meta: {}\n")
    io.open(tmp_path / "m.yaml", "w", encoding="utf-8").write(
        "_extends: [base.yaml]\n")
    with pytest.raises(cfgmod.ConfigError, match="single"):
        cfgmod.resolve("m", config_dir=tmp_path)


def test_empty_and_non_mapping_configs_are_refused(tmp_path):
    io.open(tmp_path / "empty.yaml", "w", encoding="utf-8").write("")
    io.open(tmp_path / "list.yaml", "w", encoding="utf-8").write("- 1\n- 2\n")
    with pytest.raises(cfgmod.ConfigError, match="empty"):
        cfgmod.resolve("empty", config_dir=tmp_path)
    with pytest.raises(cfgmod.ConfigError, match="mapping"):
        cfgmod.resolve("list", config_dir=tmp_path)


def _mutate(base, path, value):
    out = copy.deepcopy(base)
    node = out
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


@pytest.mark.parametrize("path,value,match", [
    ("preprocessing.normalization.forbid_global_fit", False, "forbid_global_fit"),
    ("dataset.allow_acausal_baseline", True, "acausal"),
    ("dataset.concatenate_files", True, "concatenate"),
    ("reproducibility.random_seed", None, "random_seed"),
    ("output.log_level", "CHATTY", "log level"),
])
def test_validate_rejects_integrity_weakening(cfg, path, value, match):
    with pytest.raises(cfgmod.ConfigError, match=match):
        cfgmod.validate(_mutate(cfg, path, value))


def test_validate_rejects_unknown_section(cfg):
    with pytest.raises(cfgmod.ConfigError, match="unknown top-level"):
        cfgmod.validate({**cfg, "phase_6_decision": {"w1": 1}})


@pytest.mark.parametrize("section", ["dataset", "preprocessing", "streaming"])
def test_validate_requires_core_sections(cfg, section):
    with pytest.raises(cfgmod.ConfigError, match=section):
        cfgmod.validate({k: v for k, v in cfg.items() if k != section})


def test_fingerprint_is_order_blind_and_value_sensitive(cfg):
    fp = cfgmod.fingerprint(cfg)
    assert cfgmod.fingerprint(dict(reversed(list(cfg.items())))) == fp
    assert cfgmod.fingerprint(
        _mutate(cfg, "preprocessing.windowing.window_size", 51)) != fp
    assert len(fp) == 12


def test_shipped_configs_have_distinct_fingerprints(config_dir):
    fps = {n: cfgmod.fingerprint(cfgmod.resolve(n, config_dir=config_dir))
           for n in SHIPPED}
    assert len(set(fps.values())) == len(fps)


def test_canonical_is_stable_json(cfg):
    assert json.loads(cfgmod.canonical(cfg))["meta"]["phase"] == 1


def test_dotted_get_and_require(cfg):
    assert cfgmod.get(cfg, "preprocessing.normalization.adaptation") == \
        "frozen_after_baseline"
    assert cfgmod.get(cfg, "nope.nope", default="fallback") == "fallback"
    with pytest.raises(cfgmod.ConfigError):
        cfgmod.get(cfg, "nope.nope")
    cfgmod.require(cfg, ["dataset.files", "streaming.emit"])
    with pytest.raises(cfgmod.ConfigError, match="absent"):
        cfgmod.require(cfg, ["dataset.files", "does.not.exist"])


def test_diff_reports_only_real_differences(cfg, cfg_sudden):
    d = cfgmod.diff(cfg, cfg_sudden)
    assert d, "sudden_drift must differ from default"
    assert all(k.split(".")[0] in cfgmod.OVERLAY_ALLOWED for k in d), \
        f"differences outside meta/drift: {sorted(d)}"
    assert not cfgmod.diff(cfg, copy.deepcopy(cfg))


def test_save_resolved_honours_flag_and_round_trips(cfg, tmp_path):
    path = cfgmod.save_resolved(cfg, root=tmp_path)
    assert path is not None and path.parent.name == "results"
    written = yaml.safe_load(io.open(path, encoding="utf-8").read())
    assert written["_fingerprint"] == cfgmod.fingerprint(cfg)
    body = {k: v for k, v in written.items() if not k.startswith("_")}
    assert json.dumps(body, sort_keys=True, default=str) == \
        json.dumps(cfg, sort_keys=True, default=str)

    off = copy.deepcopy(cfg)
    off["output"]["save_resolved_config"] = False
    assert cfgmod.save_resolved(off, root=tmp_path) is None
