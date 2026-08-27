"""Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT.

Module   : src/utils/seed.py
Phase    : Phase 1 / Step 5
Status   : IMPLEMENTED

Centralised reproducible random seed control.

The property this module buys
----------------------------
Every stochastic component draws from its OWN independent stream, derived from
one master seed by `numpy.random.SeedSequence([master, component_id])`. That
matters for a specific reason: it means adding, removing, or reordering a
stochastic component cannot perturb any other component's numbers. Without it, a
seed sweep is uninterpretable -- a change in results could come from the change
under study or from every downstream draw shifting by one.

Consequences deliberately accepted:

* `COMPONENTS` ids are FIXED FOREVER. Changing one silently changes every result
  that component ever produced. New components take a new id; ids are never
  reused or renumbered.
* `src/data/generator.py` already uses id 1001 directly, from before this module
  existed. It is registered here as `drift` and `verify_step5.py` asserts the two
  paths produce bit-identical streams, rather than assuming they agree.
* Global seeding (`seed_everything`) is provided because third-party code
  (river, scikit-learn, xgboost) may consult global state, but every component in
  THIS codebase must take an explicit Generator. Relying on global state makes
  execution order a hidden parameter.
"""

from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


class SeedError(RuntimeError):
    """Raised when a run cannot be made reproducible."""


# Component ids. FIXED FOREVER -- see module docstring.
COMPONENTS: dict[str, int] = {
    "drift": 1001,        # src/data/generator.py, already in use
    "stream": 1002,       # arrival jitter / sampling, if ever enabled
    "preprocessing": 1003,
}

# Identifiers reserved so that later phases cannot collide with the above.
# NOTHING is implemented for these; they are numbers, not behaviour.
RESERVED_COMPONENTS: dict[str, int] = {
    "edge_model": 2001,
    "cloud_model": 2002,
    "drift_detector": 3001,
    "reliability": 4001,
    "network": 5001,
    "resources": 5002,
    "controller": 6001,
    "adaptation": 7001,
    "baselines": 8001,
    "experiment": 9001,
}


def _registry() -> dict[str, int]:
    both = {**COMPONENTS, **RESERVED_COMPONENTS}
    if len(both) != len(COMPONENTS) + len(RESERVED_COMPONENTS):
        raise SeedError("component name collision between active and reserved")
    ids = list(both.values())
    if len(set(ids)) != len(ids):
        raise SeedError(f"duplicate component id in seed registry: {both}")
    return both


def master_seed(config: Mapping[str, Any]) -> int:
    """The run's master seed, or raise when `reproducibility.strict` demands one."""
    repro = config.get("reproducibility") or {}
    master = repro.get("random_seed")
    if master is None:
        if repro.get("strict", True):
            raise SeedError(
                "reproducibility.random_seed is missing and "
                "reproducibility.strict is true. An unseeded run cannot be "
                "reproduced, so it is refused rather than silently randomised."
            )
        return 0
    return int(master)


def component_id(component: str) -> int:
    """Look up a registered component id."""
    reg = _registry()
    if component not in reg:
        raise SeedError(
            f"unknown seed component {component!r}. Registered: "
            f"{sorted(reg)}. Register new components in COMPONENTS with a NEW "
            f"id; never reuse or renumber an existing one."
        )
    if component in RESERVED_COMPONENTS:
        raise SeedError(
            f"seed component {component!r} is RESERVED for a later phase. Its id "
            f"({RESERVED_COMPONENTS[component]}) is registered so streams stay "
            f"disjoint, but no behaviour exists for it yet."
        )
    return reg[component]


def spawn_key(config: Mapping[str, Any], component: str) -> list[int]:
    """The exact SeedSequence entropy for a component. Auditable by eye."""
    return [master_seed(config), component_id(component)]


def rng(config: Mapping[str, Any], component: str) -> np.random.Generator:
    """An independent Generator for one component.

    Bit-identical to what `generator._rng` produces for `component='drift'`;
    asserted in verify_step5.py.
    """
    return np.random.default_rng(np.random.SeedSequence(spawn_key(config, component)))


def rng_for_seed(seed: int, component: str) -> np.random.Generator:
    """Same derivation from an explicit master seed, for a seed sweep."""
    return np.random.default_rng(
        np.random.SeedSequence([int(seed), component_id(component)])
    )


def sweep_seeds(config: Mapping[str, Any]) -> tuple[int, ...]:
    """The repetition seeds for a multi-run experiment.

    Read from `reproducibility.seeds`. Deliberately NOT generated from the master
    seed: the list is written in the config so a published result can name the
    exact seeds it used.
    """
    repro = config.get("reproducibility") or {}
    seeds = repro.get("seeds")
    if not seeds:
        return (master_seed(config),)
    if not isinstance(seeds, (list, tuple)):
        raise SeedError("reproducibility.seeds must be a list")
    out = tuple(int(s) for s in seeds)
    if len(set(out)) != len(out):
        raise SeedError(f"reproducibility.seeds contains duplicates: {out}")
    return out


@dataclass(frozen=True)
class SeedRecord:
    """What was seeded, for the provenance of a results file."""

    master: int
    strict: bool
    components: dict[str, list[int]]
    sweep: tuple[int, ...]
    pythonhashseed: str | None
    global_seeded: bool

    def summary(self) -> str:
        lines = [
            f"master seed      : {self.master}",
            f"strict           : {self.strict}",
            f"sweep seeds      : {list(self.sweep)}",
            f"PYTHONHASHSEED   : {self.pythonhashseed or 'unset'}",
            f"global RNGs set  : {self.global_seeded}",
        ]
        for name, key in sorted(self.components.items()):
            lines.append(f"  {name:<14s} SeedSequence({key})")
        return "\n".join(lines)


def seed_everything(
    config: Mapping[str, Any], *, set_global: bool = True
) -> SeedRecord:
    """Seed global RNG state and return an auditable record.

    Global state is seeded for third-party libraries only. Components in this
    codebase must still take an explicit Generator: depending on global state
    would make execution order a hidden experimental parameter.
    """
    master = master_seed(config)
    repro = config.get("reproducibility") or {}
    if set_global:
        random.seed(master)
        np.random.seed(master % (2**32))
    return SeedRecord(
        master=master,
        strict=bool(repro.get("strict", True)),
        components={name: [master, cid] for name, cid in COMPONENTS.items()},
        sweep=sweep_seeds(config),
        pythonhashseed=os.environ.get("PYTHONHASHSEED"),
        global_seeded=bool(set_global),
    )
