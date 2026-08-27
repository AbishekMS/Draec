"""Step 5 verification harness -- utils/config.py, utils/seed.py, utils/logger.py.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test.

The central check here is check 2: `src/utils/config.py` must agree, byte for
byte, with the INDEPENDENT `_extends` resolver that verify_step2/3/4 implement by
hand. A harness that imports the code it tests cannot detect a bug in it, so the
duplication is deliberate and this check is what makes it worth having.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe drift_aware_edge_cloud/verify_step5.py
"""

from __future__ import annotations

import copy
import io
import json
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
sys.path.insert(0, str(ROOT))

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def note(msg: str) -> None:
    _details.append(msg)


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def expect(exc_types, fn, *a, **kw) -> tuple[bool, str]:
    try:
        fn(*a, **kw)
    except exc_types as e:
        return True, f"{type(e).__name__}: {str(e).splitlines()[0][:100]}"
    except Exception as e:
        return False, f"WRONG EXCEPTION {type(e).__name__}: {str(e)[:100]}"
    return False, "no exception raised"


# -----------------------------------------------------------------------------
# INDEPENDENT resolver -- identical to verify_step2/3/4, written by hand
# -----------------------------------------------------------------------------


def h_deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = h_deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def h_resolve(filename: str, _seen: tuple[str, ...] = ()) -> dict:
    if filename in _seen:
        raise ValueError(f"circular _extends: {' -> '.join(_seen + (filename,))}")
    raw = yaml.safe_load(io.open(CONFIG_DIR / filename, encoding="utf-8").read())
    parent = raw.pop("_extends", None)
    if parent is None:
        return raw
    return h_deep_merge(h_resolve(parent, _seen + (filename,)), raw)


# =============================================================================
# 1. Modules import and are actually implemented
# =============================================================================
try:
    from src.data import generator
    from src.utils import config as cfgmod
    from src.utils import logger as logmod
    from src.utils import seed as seedmod

    impl, defs = [], {}
    for mod, label in ((cfgmod, "config.py"), (seedmod, "seed.py"),
                       (logmod, "logger.py")):
        text = Path(mod.__file__).read_text(encoding="utf-8")
        defs[label] = sum(1 for ln in text.splitlines()
                          if ln.startswith("def ") or ln.startswith("class "))
        impl.append("Status   : IMPLEMENTED" in text)
    check(
        "1. all three utils modules import and are marked IMPLEMENTED",
        all(impl) and all(v >= 5 for v in defs.values()),
        ", ".join(f"{k}: {v} top-level defs" for k, v in defs.items()),
    )
except Exception as e:
    check("1. utils modules import", False, f"{type(e).__name__}: {e}")
    print("FATAL: cannot import; aborting.")
    raise SystemExit(1)

NAMES = ("default", "sudden_drift", "gradual_drift", "stress_test")

# =============================================================================
# 2. config.resolve() agrees with the independent hand-written resolver
# =============================================================================
agree, mismatch = {}, []
for n in NAMES:
    mine = cfgmod.resolve(n, config_dir=CONFIG_DIR)
    theirs = h_resolve(f"{n}.yaml")
    same = json.dumps(mine, sort_keys=True, default=str) == json.dumps(
        theirs, sort_keys=True, default=str)
    agree[n] = same
    if not same:
        mismatch.append((n, cfgmod.diff(theirs, mine)))
for n, d in mismatch:
    note(f"  MISMATCH in {n}: {d}")
check(
    "2. canonical resolver matches the independent harness resolver",
    all(agree.values()),
    f"{sum(agree.values())}/{len(NAMES)} configs resolve bit-identically "
    f"({', '.join(NAMES)}); the harnesses' hand-written _extends logic is "
    f"therefore a real cross-check, not a copy",
)

# =============================================================================
# 3. Overlay discipline is proved, not trusted
# =============================================================================
touched = {}
for n in NAMES[1:]:
    touched[n] = cfgmod.assert_overlay_discipline(f"{n}.yaml", config_dir=CONFIG_DIR)
guards = []
with tempfile.TemporaryDirectory() as td:
    tmp = Path(td)
    io.open(tmp / "base.yaml", "w", encoding="utf-8").write("meta: {name: b}\n")
    # An overlay that touches a forbidden section must be refused.
    io.open(tmp / "bad.yaml", "w", encoding="utf-8").write(
        "_extends: base.yaml\nmeta: {name: x}\n"
        "preprocessing: {windowing: {window_size: 999}}\n")
    ok, msg = expect(cfgmod.ConfigError, cfgmod.assert_overlay_discipline,
                     "bad.yaml", config_dir=tmp)
    guards.append(("overlay touching preprocessing", ok, msg))
    # Circular inheritance.
    io.open(tmp / "a.yaml", "w", encoding="utf-8").write("_extends: b2.yaml\nmeta: {}\n")
    io.open(tmp / "b2.yaml", "w", encoding="utf-8").write("_extends: a.yaml\nmeta: {}\n")
    ok, msg = expect(cfgmod.ConfigError, cfgmod.resolve, "a", config_dir=tmp)
    guards.append(("circular _extends", ok, msg))
    # Multiple inheritance is not single inheritance.
    io.open(tmp / "multi.yaml", "w", encoding="utf-8").write(
        "_extends: [base.yaml, base.yaml]\nmeta: {}\n")
    ok, msg = expect(cfgmod.ConfigError, cfgmod.resolve, "multi", config_dir=tmp)
    guards.append(("_extends as a list", ok, msg))
    ok, msg = expect(cfgmod.ConfigError, cfgmod.resolve, "nope", config_dir=tmp)
    guards.append(("missing file", ok, msg))
for n, t in touched.items():
    note(f"  {n}.yaml overrides exactly {list(t)}")
for label, ok, msg in guards:
    note(f"  guard {'OK  ' if ok else 'FAIL'}  {label}: {msg}")
check(
    "3. overlays touch only meta/drift; malformed inheritance refused",
    (all(set(t) <= set(cfgmod.OVERLAY_ALLOWED) for t in touched.values())
     and all(ok for _, ok, _ in guards)),
    f"{len(touched)} overlays verified as zero-carve-out; "
    f"{sum(ok for _, ok, _ in guards)}/{len(guards)} inheritance guards fired",
)

# =============================================================================
# 4. validate() rejects every integrity-weakening config
# =============================================================================
base = cfgmod.resolve("default", config_dir=CONFIG_DIR)


def mutate(path: str, value):
    out = copy.deepcopy(base)
    node = out
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


vguards = []
for label, mutated in (
    ("unknown top-level section", {**base, "phase_2_models": {"x": 1}}),
    ("forbid_global_fit: false",
     mutate("preprocessing.normalization.forbid_global_fit", False)),
    ("allow_acausal_baseline: true", mutate("dataset.allow_acausal_baseline", True)),
    ("concatenate_files: true", mutate("dataset.concatenate_files", True)),
    ("strict with null seed", mutate("reproducibility.random_seed", None)),
    ("invalid log_level", mutate("output.log_level", "CHATTY")),
    ("required section removed", {k: v for k, v in base.items() if k != "dataset"}),
):
    ok, msg = expect(cfgmod.ConfigError, cfgmod.validate, mutated, name=label)
    vguards.append((label, ok, msg))
    note(f"  guard {'OK  ' if ok else 'FAIL'}  {label}: {msg}")
all_valid = True
for n in NAMES:
    try:
        cfgmod.load(n, config_dir=CONFIG_DIR)
    except Exception as e:
        all_valid = False
        note(f"  SHIPPED CONFIG REJECTED: {n}: {e}")
check(
    "4. validate() rejects weakened configs, accepts all shipped ones",
    all(ok for _, ok, _ in vguards) and all_valid,
    f"{sum(ok for _, ok, _ in vguards)}/{len(vguards)} integrity guards fired; "
    f"all {len(NAMES)} shipped configs load clean through cfgmod.load()",
)

# =============================================================================
# 5. Fingerprint and canonical rendering
# =============================================================================
fps = {n: cfgmod.fingerprint(cfgmod.resolve(n, config_dir=CONFIG_DIR))
       for n in NAMES}
stable = fps["default"] == cfgmod.fingerprint(
    cfgmod.resolve("default", config_dir=CONFIG_DIR))
# Key order must not change the fingerprint; a value change must.
shuffled = dict(reversed(list(base.items())))
order_blind = cfgmod.fingerprint(shuffled) == fps["default"]
value_sensitive = cfgmod.fingerprint(
    mutate("preprocessing.windowing.window_size", 51)) != fps["default"]
for n, f in fps.items():
    note(f"  {n:14s} fingerprint {f}")
check(
    "5. fingerprint is deterministic, order-blind and value-sensitive",
    (stable and order_blind and value_sensitive
     and len(set(fps.values())) == len(fps)),
    f"stable across calls; insensitive to key order; changes when "
    f"window_size 50 -> 51; all {len(fps)} shipped configs distinct",
)

# =============================================================================
# 6. save_resolved() writes an audit artefact and honours its flag
# =============================================================================
with tempfile.TemporaryDirectory() as td:
    p_on = cfgmod.save_resolved(base, root=td)
    off = copy.deepcopy(base)
    off["output"]["save_resolved_config"] = False
    p_off = cfgmod.save_resolved(off, root=td)
    written = yaml.safe_load(io.open(p_on, encoding="utf-8").read())
    round_trip = {k: v for k, v in written.items() if not k.startswith("_")}
    faithful = (json.dumps(round_trip, sort_keys=True, default=str)
                == json.dumps(base, sort_keys=True, default=str))
    check(
        "6. save_resolved() is faithful, flag-honouring, output-only",
        (p_on is not None and p_off is None and faithful
         and written["_fingerprint"] == fps["default"]
         and p_on.parent.name == "results"),
        f"wrote {p_on.name} under results/ with fingerprint "
        f"{written['_fingerprint']}; content round-trips exactly; returns None "
        f"when save_resolved_config is false",
    )

# =============================================================================
# 7. seed.py agrees bit-for-bit with generator.py's pre-existing derivation
# =============================================================================
g_rng, g_master = generator._rng(base)
s_rng = seedmod.rng(base, "drift")
g_draw = g_rng.standard_normal(2000)
s_draw = s_rng.standard_normal(2000)
identical = np.array_equal(g_draw, s_draw)
key = seedmod.spawn_key(base, "drift")
check(
    "7. seed.rng(config,'drift') is bit-identical to generator._rng()",
    identical and g_master == seedmod.master_seed(base) == 42
    and key == [42, 1001],
    f"SeedSequence({key}); 2,000 standard_normal draws identical: {identical}; "
    f"master seed {g_master} agrees. Step 3's generator predates this module, "
    f"so agreement is asserted rather than assumed",
)

# =============================================================================
# 8. Component streams are independent and reproducible
# =============================================================================
draws = {c: seedmod.rng(base, c).standard_normal(500) for c in seedmod.COMPONENTS}
pairwise_distinct = all(
    not np.array_equal(draws[a], draws[b])
    for i, a in enumerate(sorted(draws)) for b in sorted(draws)[i + 1:]
)
reproducible = np.array_equal(draws["drift"],
                              seedmod.rng(base, "drift").standard_normal(500))
seed_varies = not np.array_equal(
    seedmod.rng_for_seed(1, "drift").standard_normal(500),
    seedmod.rng_for_seed(2, "drift").standard_normal(500))
sguards = []
ok, msg = expect(seedmod.SeedError, seedmod.component_id, "edge_model")
sguards.append(("reserved component refused", ok, msg))
ok, msg = expect(seedmod.SeedError, seedmod.component_id, "not_a_component")
sguards.append(("unknown component refused", ok, msg))
ok, msg = expect(seedmod.SeedError, seedmod.master_seed,
                 mutate("reproducibility.random_seed", None))
sguards.append(("strict + null seed refused", ok, msg))
all_ids = {**seedmod.COMPONENTS, **seedmod.RESERVED_COMPONENTS}
for label, ok, msg in sguards:
    note(f"  guard {'OK  ' if ok else 'FAIL'}  {label}: {msg}")
check(
    "8. per-component streams independent, reproducible, seed-sensitive",
    (pairwise_distinct and reproducible and seed_varies
     and len(set(all_ids.values())) == len(all_ids)
     and all(ok for _, ok, _ in sguards)),
    f"{len(seedmod.COMPONENTS)} active + {len(seedmod.RESERVED_COMPONENTS)} "
    f"reserved component ids, all distinct; streams pairwise different, each "
    f"reproducible, and different master seeds diverge",
)

# =============================================================================
# 9. Seed sweep and the auditable record
# =============================================================================
sweep = seedmod.sweep_seeds(base)
record = seedmod.seed_everything(base, set_global=False)
note("seed record:")
for ln in record.summary().splitlines():
    note(f"  {ln}")
check(
    "9. sweep seeds read from config; seed record is complete",
    (sweep == tuple(range(1, 11)) and len(set(sweep)) == len(sweep)
     and record.master == 42 and record.strict is True
     and set(record.components) == set(seedmod.COMPONENTS)),
    f"{len(sweep)} sweep seeds {list(sweep)} taken verbatim from "
    f"reproducibility.seeds -- not derived from the master seed, so a published "
    f"result can name them",
)

# =============================================================================
# 10. EventLog enforces its declared schema
# =============================================================================
with tempfile.TemporaryDirectory() as td:
    log = logmod.EventLog.create("stream", config=base, root=td)
    log.write(window_id=0, start_index=0, end_index=50, start_time="t0",
              end_time="t1", n_rows=50, contiguous=True, valid_fraction=1.0)
    log.write(window_id=1, start_index=10, end_index=60, start_time="t1",
              end_time="t2", n_rows=50, contiguous=False, valid_fraction=0.9)
    lguards = []
    ok, msg = expect(logmod.LoggingError, log.write, window_id=2, bogus=1)
    lguards.append(("unknown field", ok, msg))
    ok, msg = expect(logmod.LoggingError, log.write, window_id=2)
    lguards.append(("missing field", ok, msg))
    ok, msg = expect(logmod.LoggingError, logmod.EventLog.create,
                     "no_such_schema", config=base, root=td)
    lguards.append(("undeclared schema", ok, msg))
    log.close()
    rows = io.open(log.path, encoding="utf-8").read().strip().splitlines()
    header_ok = rows[0] == ",".join(logmod.SCHEMAS["stream"])
    bool_rendered = rows[1].endswith("50,1,1.0") and rows[2].endswith("50,0,0.9")
    no_read = not any(hasattr(log, m) for m in ("read", "load", "rows", "readlines"))
    for label, ok, msg in lguards:
        note(f"  guard {'OK  ' if ok else 'FAIL'}  {label}: {msg}")
    check(
        "10. EventLog: fixed schema, no ragged rows, write-only by construction",
        (header_ok and bool_rendered and log.n_written == 2
         and all(ok for _, ok, _ in lguards) and no_read),
        f"header matches the declared column order; {log.n_written} rows; "
        f"booleans render as 0/1; unknown AND missing fields both rejected; "
        f"exposes no read method, which is what keeps ground-truth logging from "
        f"becoming an input channel",
    )

# =============================================================================
# 11. End-to-end: config -> seed -> preprocess -> log, on the real data
# =============================================================================
from src.data import loader, preprocessing, stream  # noqa: E402

cfg = cfgmod.load("sudden_drift", config_dir=CONFIG_DIR)
_ = seedmod.seed_everything(cfg, set_global=False)
baseline = loader.load_baseline(cfg, root=ROOT)
profile = loader.profile_baseline(cfg, baseline)
infer = loader.load_inference_stream(cfg, root=ROOT)
stats = preprocessing.fit(cfg, baseline, profile)
drifted, gt = generator.inject(cfg, infer, profile)
prepared = preprocessing.transform(cfg, drifted, stats)
windows = list(stream.iter_windows(drifted, cfg,
                                   valid_mask=prepared.quality.valid))
fm = preprocessing.extract_features(cfg, prepared, windows)
with tempfile.TemporaryDirectory() as td:
    with logmod.EventLog.create("quality", config=cfg, root=td) as ql:
        logmod.log_quality(ql, "inference_stream", prepared.quality,
                           note=f"fingerprint={cfgmod.fingerprint(cfg)}")
    with logmod.EventLog.create("features", config=cfg, root=td) as fl:
        fl.write_many([
            {"window_id": int(w), "start_index": int(s), "end_index": int(e),
             "n_features": fm.X.shape[1], "n_outlier_flags": int(o),
             "n_range_flags": int(r), "valid_fraction": float(v)}
            for w, s, e, o, r, v in zip(fm.window_ids, fm.start_index,
                                        fm.end_index, fm.n_outlier_flags,
                                        fm.n_range_flags, fm.valid_fraction)
        ])
        n_feat_rows = fl.n_written
    qtext = io.open(Path(td) / "results" / "quality.csv", encoding="utf-8").read()
    qrows = qtext.strip().splitlines()
    n_valid_logged = int(qrows[1].split(",")[2])
check(
    "11. the whole Phase 1 chain runs from a loaded config and logs cleanly",
    (n_feat_rows == len(windows) == fm.X.shape[0]
     and n_valid_logged == int(prepared.quality.valid.sum())
     and len(qrows) == 2),
    f"config.load -> seed_everything -> load_baseline -> fit -> inject -> "
    f"transform -> {len(windows):,} windows -> {fm.X.shape[0]:,} x "
    f"{fm.X.shape[1]} features -> {n_feat_rows:,} logged rows; quality log "
    f"records {n_valid_logged:,} valid rows",
)

# =============================================================================
# 12. Logging configuration honours output.log_level
# =============================================================================
buf = io.StringIO()
lg = logmod.configure(cfg, stream=buf, force=True)
logmod.get_logger("verify").info("visible at INFO")
logmod.get_logger("verify").debug("hidden at INFO")
out_info = buf.getvalue()
buf2 = io.StringIO()
logmod.configure(cfg, level="DEBUG", stream=buf2, force=True)
logmod.get_logger("verify").debug("visible at DEBUG")
out_debug = buf2.getvalue()
for h in list(logging.getLogger("dace").handlers):
    logging.getLogger("dace").removeHandler(h)
check(
    "12. configure() respects log level and namespaces the project logger",
    ("visible at INFO" in out_info and "hidden at INFO" not in out_info
     and "visible at DEBUG" in out_debug and lg.name == "dace"
     and str(cfg["output"]["log_level"]) == "INFO"),
    f"config log_level=INFO suppresses DEBUG and emits INFO; explicit "
    f"level='DEBUG' emits it; all records under the 'dace' namespace",
)

# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("STEP 5 VERIFICATION -- src/utils/{config,seed,logger}.py")
print("=" * 78)
for ln in _details:
    print(ln)
print("-" * 78)
n_pass = sum(1 for _, ok, _ in results if ok)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        for ln in str(detail).splitlines():
            print(f"       {ln}")
print("-" * 78)
print(f"{n_pass}/{len(results)} checks passed")
print("=" * 78)
raise SystemExit(0 if n_pass == len(results) else 1)
