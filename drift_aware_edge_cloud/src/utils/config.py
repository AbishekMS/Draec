"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/utils/config.py
Phase    : Phase 1 / Step 5
Status   : IMPLEMENTED

YAML configuration loading and validation.

Why this module is small and strict
-----------------------------------
Every scientific parameter in this project lives in `config/*.yaml`, not in
Python. That only buys reproducibility if resolution is deterministic and if the
resolved object is auditable. So:

* `_extends` is SINGLE inheritance with a deep merge and cycle detection. There
  is no multiple inheritance and no include-list, because merge order would then
  be an unwritten scientific parameter.
* Scenario overlays may touch ONLY `meta` and `drift`. That restriction is what
  makes cross-scenario results comparable: if `sudden_drift.yaml` could quietly
  widen a window or switch a normalizer, a difference between scenarios could no
  longer be attributed to the drift. `assert_overlay_discipline()` proves it per
  run instead of trusting review.
* `fingerprint()` hashes the resolved config so a results file can name the
  exact configuration that produced it.

The verify_stepN.py harnesses deliberately re-implement `deep_merge`/`resolve`
independently rather than importing this module. A harness that imports the code
under test cannot detect a bug in it. `verify_step5.py` asserts the two
implementations agree on every shipped config, which is the useful check.
"""

from __future__ import annotations

import copy
import hashlib
import io
import json
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import yaml


class ConfigError(RuntimeError):
    """Raised when a configuration cannot be resolved or violates a contract."""


# Top-level sections a Phase 1 configuration may contain. Unknown keys are an
# error, not a warning: a typo'd section silently does nothing, which is the
# worst possible failure mode for a scientific parameter.
KNOWN_SECTIONS = (
    "meta",
    "reproducibility",
    "dataset",
    "split",
    "preprocessing",
    "drift",
    "ground_truth",
    "streaming",
    "output",
    "reserved_for_later_phases",
)

# Sections a scenario overlay is permitted to override. See module docstring.
OVERLAY_ALLOWED = ("meta", "drift")

BASE_CONFIG = "default.yaml"


# -----------------------------------------------------------------------------
# Resolution
# -----------------------------------------------------------------------------


def deep_merge(base: Mapping[str, Any], over: Mapping[str, Any]) -> dict:
    """Recursive dict merge. Non-dict values are replaced, never combined.

    Lists are replaced wholesale. Appending would make the result depend on
    inheritance depth, and `drift.affected_features.columns` must mean exactly
    what the overlay says.
    """
    out = copy.deepcopy(dict(base))
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def read_raw(path: Path | str) -> dict:
    """Read one YAML file with no `_extends` resolution."""
    p = Path(path)
    if not p.is_file():
        raise ConfigError(f"configuration file not found: {p}")
    text = io.open(p, encoding="utf-8").read()
    data = yaml.safe_load(text)
    if data is None:
        raise ConfigError(f"configuration file is empty: {p}")
    if not isinstance(data, dict):
        raise ConfigError(f"configuration root must be a mapping, got "
                          f"{type(data).__name__}: {p}")
    return data


def resolve(
    name: str,
    *,
    config_dir: Path | str = "config",
    _seen: tuple[str, ...] = (),
) -> dict:
    """Resolve one config file, following `_extends` to its base.

    `name` may be given with or without the `.yaml` suffix.
    """
    filename = name if name.endswith(".yaml") else f"{name}.yaml"
    if filename in _seen:
        raise ConfigError(
            f"circular _extends chain: {' -> '.join(_seen + (filename,))}"
        )
    raw = read_raw(Path(config_dir) / filename)
    parent = raw.pop("_extends", None)
    if parent is None:
        return raw
    if not isinstance(parent, str):
        raise ConfigError(
            f"{filename}: _extends must be a single filename (single "
            f"inheritance), got {type(parent).__name__}"
        )
    base = resolve(parent, config_dir=config_dir, _seen=_seen + (filename,))
    return deep_merge(base, raw)


def load(
    name: str = "default",
    *,
    config_dir: Path | str = "config",
    validate_sections: bool = True,
    enforce_overlay_discipline: bool = True,
) -> dict:
    """Resolve, validate, and return a configuration ready for use."""
    cfg = resolve(name, config_dir=config_dir)
    if validate_sections:
        validate(cfg, name=name)
    if enforce_overlay_discipline:
        filename = name if name.endswith(".yaml") else f"{name}.yaml"
        raw = read_raw(Path(config_dir) / filename)
        if "_extends" in raw:
            assert_overlay_discipline(filename, config_dir=config_dir)
    return cfg


# -----------------------------------------------------------------------------
# Validation
# -----------------------------------------------------------------------------


def validate(config: Mapping[str, Any], *, name: str = "<config>") -> None:
    """Structural validation of a resolved Phase 1 configuration."""
    unknown = [k for k in config if k not in KNOWN_SECTIONS]
    if unknown:
        raise ConfigError(
            f"{name}: unknown top-level section(s) {unknown}. Known sections: "
            f"{list(KNOWN_SECTIONS)}. A typo'd section silently changes nothing, "
            f"so it is rejected rather than ignored."
        )
    for required in ("meta", "reproducibility", "dataset", "preprocessing",
                     "streaming", "output"):
        if required not in config:
            raise ConfigError(f"{name}: required section '{required}' is absent")

    repro = config.get("reproducibility") or {}
    if repro.get("strict", True) and repro.get("random_seed") is None:
        raise ConfigError(
            f"{name}: reproducibility.strict is true but random_seed is null. "
            f"An unseeded run is not reproducible."
        )
    seeds = repro.get("seeds")
    if seeds is not None and not isinstance(seeds, list):
        raise ConfigError(f"{name}: reproducibility.seeds must be a list")

    # Integrity invariants that must hold in EVERY configuration, base or
    # overlay, and are therefore checked here rather than in one call site.
    norm = ((config.get("preprocessing") or {}).get("normalization")) or {}
    if not norm.get("forbid_global_fit", True):
        raise ConfigError(
            f"{name}: preprocessing.normalization.forbid_global_fit is false. "
            f"Fitting on all data is prohibited project-wide."
        )
    ds = config.get("dataset") or {}
    if ds.get("allow_acausal_baseline", False):
        raise ConfigError(
            f"{name}: dataset.allow_acausal_baseline is true. HAI's chronology "
            f"is train1 < test1 < train2, so a baseline including train2 would "
            f"fit on data recorded after the inference stream."
        )
    if ds.get("concatenate_files", False):
        raise ConfigError(
            f"{name}: dataset.concatenate_files is true. Joining two recordings "
            f"introduces an artificial regime change that would be "
            f"indistinguishable from injected drift."
        )
    out = config.get("output") or {}
    lvl = str(out.get("log_level", "INFO")).upper()
    if lvl not in {"DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"}:
        raise ConfigError(f"{name}: output.log_level {lvl!r} is not a log level")


def assert_overlay_discipline(
    filename: str,
    *,
    config_dir: Path | str = "config",
    allowed: Sequence[str] = OVERLAY_ALLOWED,
) -> tuple[str, ...]:
    """Prove that an overlay changes ONLY the sections it is allowed to change.

    Returns the tuple of sections the overlay actually touches. Raises if it
    touches anything else -- because then a difference measured between two
    scenarios could no longer be attributed to the drift.
    """
    raw = read_raw(Path(config_dir) / filename)
    parent = raw.get("_extends")
    if parent is None:
        raise ConfigError(f"{filename} has no _extends; it is not an overlay")
    raw = {k: v for k, v in raw.items() if k != "_extends"}
    touched = tuple(sorted(raw))
    illegal = [k for k in touched if k not in set(allowed)]
    if illegal:
        raise ConfigError(
            f"{filename} overrides section(s) {illegal}, but a scenario overlay "
            f"may only touch {list(allowed)}. Carving out anything else breaks "
            f"cross-scenario comparability: a measured difference could then be "
            f"caused by the carve-out instead of by the drift."
        )
    return touched


def diff(base: Mapping[str, Any], other: Mapping[str, Any],
         *, _prefix: str = "") -> dict[str, tuple[Any, Any]]:
    """Flat dotted-key diff between two resolved configs. Audit helper."""
    out: dict[str, tuple[Any, Any]] = {}
    for k in sorted(set(base) | set(other)):
        key = f"{_prefix}{k}"
        a, b = base.get(k, KeyError), other.get(k, KeyError)
        if isinstance(a, dict) and isinstance(b, dict):
            out.update(diff(a, b, _prefix=f"{key}."))
        elif a != b:
            out[key] = (None if a is KeyError else a, None if b is KeyError else b)
    return out


# -----------------------------------------------------------------------------
# Dotted access
# -----------------------------------------------------------------------------


_MISSING = object()


def get(config: Mapping[str, Any], path: str, default: Any = _MISSING) -> Any:
    """Read a dotted path. Raises unless a default is supplied."""
    node: Any = config
    for part in path.split("."):
        if not isinstance(node, Mapping) or part not in node:
            if default is _MISSING:
                raise ConfigError(f"configuration key not found: {path!r}")
            return default
        node = node[part]
    return node


def require(config: Mapping[str, Any], paths: Iterable[str]) -> None:
    """Assert several dotted paths exist. Fails loudly at start of a run.

    The probe uses its own sentinel: passing `_MISSING` back into `get` would be
    read as "no default supplied", so `get` would raise on the first absent path
    and the caller would never see the full list.
    """
    absent = object()
    missing = [p for p in paths if get(config, p, absent) is absent]
    if missing:
        raise ConfigError(f"required configuration key(s) absent: {missing}")


# -----------------------------------------------------------------------------
# Provenance
# -----------------------------------------------------------------------------


def canonical(config: Mapping[str, Any]) -> str:
    """Deterministic JSON rendering: sorted keys, no incidental whitespace."""
    return json.dumps(config, sort_keys=True, separators=(",", ":"),
                      ensure_ascii=True, default=str)


def fingerprint(config: Mapping[str, Any], *, length: int = 12) -> str:
    """Short SHA-256 of the resolved config, for naming results."""
    return hashlib.sha256(canonical(config).encode("utf-8")).hexdigest()[:length]


def save_resolved(
    config: Mapping[str, Any],
    *,
    root: Path | str = ".",
    filename: str | None = None,
    force: bool = False,
) -> Path | None:
    """Write the fully resolved config next to the results it explains.

    Honours `output.save_resolved_config`; returns None when disabled. This is
    an OUTPUT artefact: nothing in the pipeline ever reads it back, so it cannot
    become a hidden input.
    """
    out = config.get("output") or {}
    if not force and not out.get("save_resolved_config", True):
        return None
    results_dir = Path(root) / str(out.get("results_dir", "results"))
    results_dir.mkdir(parents=True, exist_ok=True)
    name = config.get("meta", {}).get("name", "config")
    fname = filename or f"resolved_{name}_{fingerprint(config)}.yaml"
    path = results_dir / fname
    payload = {
        "_resolved_from": name,
        "_fingerprint": fingerprint(config),
        "_note": "Machine-generated audit artefact. Never read back as input.",
        **copy.deepcopy(dict(config)),
    }
    io.open(path, "w", encoding="utf-8").write(
        yaml.safe_dump(payload, sort_keys=False, allow_unicode=False,
                       default_flow_style=False)
    )
    return path


def available(config_dir: Path | str = "config") -> tuple[str, ...]:
    """Names of the shipped configurations, without the .yaml suffix."""
    d = Path(config_dir)
    if not d.is_dir():
        raise ConfigError(f"config directory not found: {d}")
    return tuple(sorted(p.stem for p in d.glob("*.yaml")))
