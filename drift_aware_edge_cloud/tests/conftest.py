"""Shared fixtures for the Phase 1 test suite.

The real HAI files are loaded ONCE per session and reused. Loading 280,800 rows
per test would make the suite slow enough that people stop running it, and a test
suite nobody runs is worse than none.

Nothing here fabricates data. Every fixture is either the real recording, a real
config, or something derived from them by the code under test.
"""

from __future__ import annotations

import sys
from pathlib import Path

import gc
import pytest

ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(ROOT))

from src.data import generator, loader, preprocessing, stream  # noqa: E402
from src.utils import config as cfgmod  # noqa: E402


@pytest.fixture(autouse=True)
def _cleanup_memory():
    yield
    gc.collect()


@pytest.fixture(scope="session")
def project_root() -> Path:
    return ROOT


@pytest.fixture(scope="session")
def config_dir() -> Path:
    return CONFIG_DIR


@pytest.fixture(scope="session")
def cfg() -> dict:
    """The resolved default configuration: no drift, the control condition."""
    return cfgmod.load("default", config_dir=CONFIG_DIR)


@pytest.fixture(scope="session")
def cfg_sudden() -> dict:
    return cfgmod.load("sudden_drift", config_dir=CONFIG_DIR)


@pytest.fixture(scope="session")
def cfg_gradual() -> dict:
    return cfgmod.load("gradual_drift", config_dir=CONFIG_DIR)


@pytest.fixture(scope="session")
def cfg_stress() -> dict:
    return cfgmod.load("stress_test", config_dir=CONFIG_DIR)


@pytest.fixture(scope="session")
def baseline(cfg):
    return loader.load_baseline(cfg, root=ROOT)


@pytest.fixture(scope="session")
def profile(cfg, baseline):
    return loader.profile_baseline(cfg, baseline)


@pytest.fixture(scope="session")
def infer(cfg):
    return loader.load_inference_stream(cfg, root=ROOT)


@pytest.fixture(scope="session")
def stats(cfg, baseline, profile):
    return preprocessing.fit(cfg, baseline, profile)


@pytest.fixture(scope="session")
def prepared_clean(cfg, infer, stats):
    return preprocessing.transform(cfg, infer, stats)


@pytest.fixture(scope="session")
def injected(cfg_sudden, infer, profile):
    """(DriftedStream, GroundTruth) for the sudden-drift scenario."""
    return generator.inject(cfg_sudden, infer, profile)


@pytest.fixture(scope="session")
def prepared_drift(cfg_sudden, injected, stats):
    return preprocessing.transform(cfg_sudden, injected[0], stats)


@pytest.fixture(scope="session")
def windows(cfg_sudden, injected, prepared_drift):
    return list(stream.iter_windows(injected[0], cfg_sudden,
                                    valid_mask=prepared_drift.quality.valid))


@pytest.fixture(scope="session")
def features(cfg_sudden, prepared_drift, windows):
    return preprocessing.extract_features(cfg_sudden, prepared_drift, windows)
