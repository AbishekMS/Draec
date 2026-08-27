"""Seed derivation: independence, reproducibility, and the registry contract."""

from __future__ import annotations

import copy

import numpy as np
import pytest

from src.data import generator
from src.utils import seed as seedmod


def test_master_seed_read_from_config(cfg):
    assert seedmod.master_seed(cfg) == 42


def test_strict_mode_refuses_an_unseeded_run(cfg):
    c = copy.deepcopy(cfg)
    c["reproducibility"]["random_seed"] = None
    with pytest.raises(seedmod.SeedError, match="random_seed"):
        seedmod.master_seed(c)


def test_non_strict_mode_falls_back_to_zero(cfg):
    c = copy.deepcopy(cfg)
    c["reproducibility"]["random_seed"] = None
    c["reproducibility"]["strict"] = False
    assert seedmod.master_seed(c) == 0


def test_registry_ids_are_unique():
    both = {**seedmod.COMPONENTS, **seedmod.RESERVED_COMPONENTS}
    assert len(set(both.values())) == len(both)
    assert set(seedmod.COMPONENTS) & set(seedmod.RESERVED_COMPONENTS) == set()


def test_drift_component_id_is_frozen_at_1001():
    # Changing this silently changes every drift realisation ever produced.
    assert seedmod.COMPONENTS["drift"] == 1001
    assert generator._SEED_COMPONENT_DRIFT == 1001


def test_seed_module_matches_generator_bit_for_bit(cfg):
    a = generator._rng(cfg)[0].standard_normal(1000)
    b = seedmod.rng(cfg, "drift").standard_normal(1000)
    assert np.array_equal(a, b)


def test_spawn_key_is_auditable(cfg):
    assert seedmod.spawn_key(cfg, "drift") == [42, 1001]


def test_component_streams_are_independent(cfg):
    draws = {c: seedmod.rng(cfg, c).standard_normal(300)
             for c in seedmod.COMPONENTS}
    names = sorted(draws)
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            assert not np.array_equal(draws[a], draws[b])


def test_same_component_is_reproducible(cfg):
    assert np.array_equal(seedmod.rng(cfg, "drift").standard_normal(300),
                          seedmod.rng(cfg, "drift").standard_normal(300))


def test_different_master_seeds_diverge():
    assert not np.array_equal(
        seedmod.rng_for_seed(1, "drift").standard_normal(300),
        seedmod.rng_for_seed(2, "drift").standard_normal(300))


def test_reserved_components_are_refused_until_implemented():
    with pytest.raises(seedmod.SeedError, match="RESERVED"):
        seedmod.component_id("edge_model")


def test_unknown_component_is_refused():
    with pytest.raises(seedmod.SeedError, match="unknown seed component"):
        seedmod.component_id("nonexistent")


def test_sweep_seeds_come_verbatim_from_config(cfg):
    assert seedmod.sweep_seeds(cfg) == tuple(range(1, 11))


def test_sweep_seeds_reject_duplicates(cfg):
    c = copy.deepcopy(cfg)
    c["reproducibility"]["seeds"] = [1, 1, 2]
    with pytest.raises(seedmod.SeedError, match="duplicate"):
        seedmod.sweep_seeds(c)


def test_sweep_falls_back_to_master_when_absent(cfg):
    c = copy.deepcopy(cfg)
    c["reproducibility"]["seeds"] = []
    assert seedmod.sweep_seeds(c) == (42,)


def test_seed_record_is_complete_without_touching_global_state(cfg):
    before = np.random.get_state()[1][:5].copy()
    rec = seedmod.seed_everything(cfg, set_global=False)
    assert np.array_equal(np.random.get_state()[1][:5], before)
    assert rec.master == 42 and rec.strict is True
    assert set(rec.components) == set(seedmod.COMPONENTS)
    assert "master seed" in rec.summary()


def test_global_seeding_is_deterministic(cfg):
    seedmod.seed_everything(cfg, set_global=True)
    a = np.random.rand(5)
    seedmod.seed_everything(cfg, set_global=True)
    assert np.array_equal(a, np.random.rand(5))
