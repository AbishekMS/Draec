"""
Step 2 configuration verification.

Standalone verification harness for the HAI configuration transition. Executes
the ten checks specified for Step 2. Reads config/*.yaml and data/raw/ only;
imports nothing from src/ (src/ is still docstring-only placeholders).

Run:  python verify_step2.py
Exit: 0 if every check passes, 1 otherwise.

This file is a Phase 1 verification tool, not part of the simulation. It is
deliberately kept out of src/ so it cannot be mistaken for a component.
"""

from __future__ import annotations

import copy
import io
import json
import re
import sys
from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parent
CONFIG_DIR = ROOT / "config"
DATA_RAW = ROOT / "data" / "raw"

BASE = "default.yaml"
OVERLAYS = ["sudden_drift.yaml", "gradual_drift.yaml", "stress_test.yaml"]

results: list[tuple[str, bool, str]] = []
_details: list[str] = []


def check(name: str, ok: bool, detail: str = "") -> bool:
    results.append((name, ok, detail))
    return ok


def note(line: str) -> None:
    _details.append(line)


# -----------------------------------------------------------------------------
# _extends resolution -- same semantics the future config loader must implement
# -----------------------------------------------------------------------------
def deep_merge(base: dict, over: dict) -> dict:
    """Recursive dict merge. Scalars and lists in `over` replace `base`."""
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(filename: str, _seen: tuple[str, ...] = ()) -> dict:
    """Load `filename` and apply single-parent `_extends` inheritance."""
    if filename in _seen:
        raise ValueError(f"circular _extends: {' -> '.join(_seen + (filename,))}")
    raw = yaml.safe_load(io.open(CONFIG_DIR / filename, encoding="utf-8").read())
    parent = raw.pop("_extends", None)
    if parent is None:
        return raw
    return deep_merge(resolve(parent, _seen + (filename,)), raw)


def flatten(d, prefix=""):
    """dict -> {dotted.path: scalar}. Lists are compared as whole values."""
    out = {}
    for k, v in d.items():
        p = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, p + "."))
        else:
            out[p] = v
    return out


# =============================================================================
# 1. Parse all YAML files
# =============================================================================
raw_docs: dict[str, dict] = {}
parse_ok = True
for fn in [BASE] + OVERLAYS:
    path = CONFIG_DIR / fn
    if not path.exists():
        parse_ok = False
        note(f"  MISSING: {fn}")
        continue
    try:
        doc = yaml.safe_load(io.open(path, encoding="utf-8").read())
        if not isinstance(doc, dict):
            parse_ok = False
            note(f"  {fn}: top level is {type(doc).__name__}, expected mapping")
            continue
        raw_docs[fn] = doc
        note(f"  {fn}: parsed, {len(doc)} top-level keys, "
             f"{len(io.open(path, encoding='utf-8').read().splitlines())} lines")
    except yaml.YAMLError as e:
        parse_ok = False
        note(f"  {fn}: YAMLError {e}")
check("1. All YAML files parse", parse_ok and len(raw_docs) == 4,
      f"{len(raw_docs)}/4 files")

# =============================================================================
# 2. Verify _extends resolution
# =============================================================================
resolved: dict[str, dict] = {}
ext_ok = True

if BASE in raw_docs and "_extends" in raw_docs[BASE]:
    ext_ok = False
    note("  default.yaml declares _extends -- base config must not inherit")
else:
    note("  default.yaml: no _extends (correct -- it is the root)")

for fn in OVERLAYS:
    if fn not in raw_docs:
        ext_ok = False
        continue
    parent = raw_docs[fn].get("_extends")
    if parent != BASE:
        ext_ok = False
        note(f"  {fn}: _extends is {parent!r}, expected {BASE!r}")
        continue
    try:
        resolved[fn] = resolve(fn)
    except Exception as e:  # noqa: BLE001 -- verification harness reports, not raises
        ext_ok = False
        note(f"  {fn}: resolution failed: {type(e).__name__}: {e}")
        continue

    base_keys = set(raw_docs[BASE])
    res_keys = set(resolved[fn])
    if not base_keys.issubset(res_keys):
        ext_ok = False
        note(f"  {fn}: lost base sections {sorted(base_keys - res_keys)}")
    else:
        # deep merge must inherit, not replace: a key present only in the base
        # branch of an overridden section must survive.
        inherited = flatten(resolved[fn]).keys() - flatten(raw_docs[fn]).keys()
        note(f"  {fn}: _extends -> {parent}; {len(res_keys)} sections, "
             f"{len(inherited)} inherited leaf keys")

if BASE in raw_docs:
    resolved[BASE] = raw_docs[BASE]

check("2. _extends resolves correctly", ext_ok, f"{len(resolved)}/4 resolved")

# =============================================================================
# 3. Verify all scenarios inherit the same baseline settings
#    (overlays may differ ONLY inside `meta` and `drift`)
# =============================================================================
ALLOWED_DIVERGENCE = ("meta.", "drift.")
inherit_ok = True
if BASE in resolved and len(resolved) == 4:
    base_flat = flatten(resolved[BASE])
    for fn in OVERLAYS:
        ov_flat = flatten(resolved[fn])
        illegal = []
        for k in sorted(set(base_flat) | set(ov_flat)):
            if k.startswith(ALLOWED_DIVERGENCE):
                continue
            b, o = base_flat.get(k, "<absent>"), ov_flat.get(k, "<absent>")
            if b != o:
                illegal.append(f"{k}: base={b!r} -> {fn}={o!r}")
        if illegal:
            inherit_ok = False
            for line in illegal:
                note(f"  ILLEGAL DIVERGENCE {line}")
        else:
            differing = [k for k in ov_flat
                         if k.startswith(ALLOWED_DIVERGENCE)
                         and base_flat.get(k, "<absent>") != ov_flat[k]]
            note(f"  {fn}: 0 illegal divergences; "
                 f"{len(differing)} intentional overrides, all in meta/drift")
else:
    inherit_ok = False
check("3. Scenarios inherit identical non-drift settings", inherit_ok,
      "overlays touch only meta/ and drift/")

# =============================================================================
# 4. Verify HAI paths are correct and configurable
# =============================================================================
path_ok = True
ds = resolved.get(BASE, {}).get("dataset", {})
files = ds.get("files", {})
EXPECTED = {
    "train1": ("data/raw/hai-train1.txt", 280800),
    "train2": ("data/raw/hai-train2.txt", 291600),
    "test1": ("data/raw/hai-test1.txt", 54000),
}
for key, (exp_path, exp_rows) in EXPECTED.items():
    entry = files.get(key)
    if not isinstance(entry, dict):
        path_ok = False
        note(f"  dataset.files.{key}: missing or not a mapping")
        continue
    p = entry.get("path")
    if p != exp_path:
        path_ok = False
        note(f"  dataset.files.{key}.path = {p!r}, expected {exp_path!r}")
        continue
    on_disk = ROOT / p
    if not on_disk.exists():
        path_ok = False
        note(f"  {p}: declared in config but NOT ON DISK")
        continue
    # row count is a config claim; verify it against the file
    with io.open(on_disk, encoding="utf-8") as fh:
        actual_rows = sum(1 for _ in fh) - 1  # minus header
    if actual_rows != exp_rows or entry.get("rows") != exp_rows:
        path_ok = False
        note(f"  {p}: config rows={entry.get('rows')}, measured={actual_rows}, "
             f"expected={exp_rows}")
    else:
        note(f"  dataset.files.{key}.path = {p} "
             f"[exists, {actual_rows:,} data rows, matches config]")

# configurable = no HAI filename hard-coded anywhere in src/ or tests/
hardcoded = []
for py in list((ROOT / "src").rglob("*.py")) + list((ROOT / "tests").rglob("*.py")) \
        + list((ROOT / "adaptation").rglob("*.py")) + [ROOT / "main.py"]:
    if not py.exists():
        continue
    txt = io.open(py, encoding="utf-8").read()
    if re.search(r"hai-(train1|train2|test1)\.txt", txt):
        hardcoded.append(str(py.relative_to(ROOT)))
if hardcoded:
    path_ok = False
    note(f"  HARD-CODED HAI paths found in: {hardcoded}")
else:
    note("  no HAI filename hard-coded in src/, tests/, adaptation/ or main.py")
check("4. HAI paths correct and configurable", path_ok,
      "3/3 files declared, exist, row counts verified")

# =============================================================================
# 5. Verify train1/train2/test1 roles are explicit
# =============================================================================
role_ok = True
EXPECTED_ROLES = {
    "train1": "baseline_train",
    "train2": "baseline_validation",
    "test1": "inference_stream",
}
for key, exp_role in EXPECTED_ROLES.items():
    got = files.get(key, {}).get("role")
    if got != exp_role:
        role_ok = False
        note(f"  dataset.files.{key}.role = {got!r}, expected {exp_role!r}")
    else:
        note(f"  dataset.files.{key}.role = {got}")

# separation and causality must be explicit, not implied
REQUIRED_FLAGS = {
    "dataset.concatenate_files": False,
    "dataset.baseline_source": "train1_only",
    "dataset.allow_acausal_baseline": False,
    "split.mode": "separate_file",
    "drift.reference_stream": "inference_stream",
    "drift.injection_target": "inference_stream_only",
}
base_flat_all = flatten(resolved.get(BASE, {}))
for k, exp in REQUIRED_FLAGS.items():
    got = base_flat_all.get(k, "<absent>")
    if got != exp:
        role_ok = False
        note(f"  {k} = {got!r}, expected {exp!r}")
    else:
        note(f"  {k} = {got!r}")

# roles must survive inheritance into every overlay
for fn in OVERLAYS:
    for key, exp_role in EXPECTED_ROLES.items():
        got = resolved.get(fn, {}).get("dataset", {}).get("files", {}) \
                      .get(key, {}).get("role")
        if got != exp_role:
            role_ok = False
            note(f"  {fn}: dataset.files.{key}.role = {got!r}")
check("5. File roles explicit and inherited", role_ok,
      "roles + separation/causality flags present in all 4 configs")

# =============================================================================
# 6. Verify no SWaT-specific paths remain
# =============================================================================
swat_ok = True
SWAT_PATHY = re.compile(
    r"(attack\.csv|SWaT_Dataset|Normal/Attack|"
    r"\b(LIT|FIT|AIT|DPIT|PIT|MV|P[1-6]0[0-9]|UV401)\d*\b)",
    re.IGNORECASE,
)
for fn in [BASE] + OVERLAYS:
    text = io.open(CONFIG_DIR / fn, encoding="utf-8").read()
    offenders = []
    for i, line in enumerate(text.splitlines(), 1):
        if not SWAT_PATHY.search(line):
            continue
        stripped = line.strip()
        # comments are documentation of the supersession, not configuration
        if stripped.startswith("#"):
            continue
        offenders.append(f"{fn}:{i}: {stripped}")
    if offenders:
        swat_ok = False
        for o in offenders:
            note(f"  ACTIVE SWAT VALUE {o}")
    else:
        n_comment = sum(1 for line in text.splitlines()
                        if SWAT_PATHY.search(line) and line.strip().startswith("#"))
        note(f"  {fn}: 0 active SWaT values "
             f"({n_comment} historical mentions in comments)")

# no config value may point at the SWaT file
for fn in [BASE] + OVERLAYS:
    for k, v in flatten(resolved.get(fn, {})).items():
        if isinstance(v, str) and "attack.csv" in v.lower():
            swat_ok = False
            note(f"  {fn}: {k} = {v!r} still points at the SWaT file")
check("6. No active SWaT-specific paths or values", swat_ok,
      "all remaining mentions are comments explaining the supersession")

# =============================================================================
# 7. Verify no synthetic dataset is treated as a raw input
# =============================================================================
# The question is specifically about INPUTS. `data/synthetic/` is a legitimate
# OUTPUT location -- the drift generator writes its drift-injected derivative
# and the ground-truth sidecar there. So the check must (a) look only at
# path-valued keys, not at prose that happens to contain the word "synthetic",
# and (b) treat exactly the raw-input keys as inputs and everything else as
# output.
syn_ok = True
INPUT_KEY = re.compile(r"^dataset\.files\.[A-Za-z0-9_]+\.path$")
PATH_KEY = re.compile(r"(^|\.)(path|dir|file|sidecar_path|path_template)$")
PATH_VALUE = re.compile(r"^[A-Za-z0-9_./\\-]+(/|\\)[A-Za-z0-9_./\\{}-]+$")

for fn in [BASE] + OVERLAYS:
    for k, v in flatten(resolved.get(fn, {})).items():
        if not isinstance(v, str):
            continue
        # only consider values that are actually paths
        if not (PATH_KEY.search(k) and PATH_VALUE.match(v.strip())):
            continue
        low = v.replace("\\", "/").lower()
        if "data/synthetic" not in low and "data/processed" not in low:
            continue
        if INPUT_KEY.match(k):
            syn_ok = False
            note(f"  {fn}: {k} = {v!r} -- DERIVED path used as a RAW INPUT")
        else:
            note(f"  {fn}: {k} = {v} [output/derivative key, not an input]")

for key, entry in files.items():
    p = str(entry.get("path", "")).replace("\\", "/")
    if not p.startswith("data/raw/"):
        syn_ok = False
        note(f"  dataset.files.{key}.path = {p} -- not under data/raw/")
if syn_ok:
    note("  all 3 dataset.files.*.path resolve under data/raw/ (real HAI files)")

syn_dir = ROOT / "data" / "synthetic"
syn_contents = sorted(p.name for p in syn_dir.iterdir()
                      if syn_dir.exists() and p.name != ".gitkeep")
# Since Step 3, `data/synthetic/` legitimately holds generator OUTPUT (the
# ground-truth sidecar). The invariant is not "this directory is empty" -- it is
# that nothing in it is ever read back as an INPUT, which the path-key scan above
# already establishes. Files here are only a violation if a config points at them.
GENERATED_OK = {"ground_truth.json"}
# `experiments/phase1_demo.py` runs all three scenarios in one pass and gives
# each its own sidecar, rather than letting the third silently overwrite the
# first two. Those files are the same kind of object the exact-name whitelist
# already permitted -- metadata about an injection -- so the whitelist matches
# the shape instead of one name. Anything that is not a ground-truth sidecar
# (a drifted stream, a feature dump) is still a violation.
SIDECAR_RE = re.compile(r"ground_truth(_[a-z0-9_]+)?\.json")
unexpected = [f for f in syn_contents
              if f not in GENERATED_OK and not SIDECAR_RE.fullmatch(f)]
if unexpected:
    syn_ok = False
    note(f"  data/synthetic/ holds unrecognised files: {unexpected}")
elif syn_contents:
    note(f"  data/synthetic/ holds generator OUTPUT only: {syn_contents}; "
         f"no config reads it as an input")
else:
    note("  data/synthetic/ empty (output-only directory, nothing generated yet)")
check("7. No synthetic dataset used as raw input", syn_ok,
      "raw inputs are the 3 real HAI files only")

# =============================================================================
# 8. Verify no fabricated data was created
# =============================================================================
fab_ok = True
SOURCE_SHA = {
    "hai-train1.txt": "53007b0ba604fbf338e7ac2e08cd81d874b5d1388f3aecb213ddcba5bf2bec4a",
    "hai-train2.txt": "0e520e82bf78a661ab19ce4967f3c766bd809820f457a9c90c365102d4534c56",
    "hai-test1.txt": "78c7f1d4de1f2ab9ccc2f8c719f80f831033543adb0c81d0d78f84f40838d4be",
}
import hashlib

for name, exp_sha in SOURCE_SHA.items():
    p = DATA_RAW / name
    if not p.exists():
        fab_ok = False
        note(f"  {name}: absent")
        continue
    h = hashlib.sha256()
    with io.open(p, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    got = h.hexdigest()
    if got != exp_sha:
        fab_ok = False
        note(f"  {name}: sha256 {got[:16]}... != recorded {exp_sha[:16]}... "
             f"-- RAW FILE MODIFIED")
    else:
        note(f"  {name}: sha256 {got[:16]}... matches PROVENANCE (unmodified)")

# No DERIVED DATA STREAM has been written to disk. The three raw checksums above
# are the real guarantee that nothing was fabricated; this adds that no drifted
# feature stream has been silently persisted and could be mistaken for raw data.
# The ground-truth sidecar is metadata about the injection, not a data stream.
for sub in ("processed", "synthetic"):
    d = ROOT / "data" / sub
    extra = [p.name for p in d.iterdir()
             if d.exists() and p.name != ".gitkeep"
             and not SIDECAR_RE.fullmatch(p.name)]   # defined in check 7
    if extra:
        fab_ok = False
        note(f"  data/{sub}/ contains a persisted data stream: {extra}")
    else:
        note(f"  data/{sub}/ holds no persisted feature stream")

# label must not have been invented
lbl = ds.get("label_column", "<absent>")
task = ds.get("task", "<absent>")
if lbl is not None:
    fab_ok = False
    note(f"  dataset.label_column = {lbl!r} -- HAI has no label column; "
         f"must be null")
else:
    note("  dataset.label_column = None (no label fabricated)")
if task != "unresolved":
    fab_ok = False
    note(f"  dataset.task = {task!r} -- expected 'unresolved' until the user "
         f"decides")
else:
    note("  dataset.task = 'unresolved' (not guessed)")
check("8. No fabricated data or labels", fab_ok,
      "3/3 raw files byte-identical; label null; task unresolved")

# =============================================================================
# 9. Verify no Phase 2+ configuration was introduced
# =============================================================================
phase_ok = True
ALLOWED_SECTIONS = {
    "_extends", "meta", "reproducibility", "dataset", "split", "preprocessing",
    "drift", "ground_truth", "streaming", "output", "reserved_for_later_phases",
}
# Names that would mean a later-phase component had been configured for real.
LATER_PHASE_KEYS = re.compile(
    r"^(model|models|edge|cloud|hybrid|adwin|drift_detector|detector|"
    r"reliability|lri|wds|controller|decision|network|edge_resources|"
    r"resources|adaptation|retraining|simpy|baselines|ablations|experiments|"
    r"metrics|statistics|dashboard|mlflow)$"
)
for fn in [BASE] + OVERLAYS:
    doc = raw_docs[fn]
    unexpected = sorted(set(doc) - ALLOWED_SECTIONS)
    if unexpected:
        phase_ok = False
        note(f"  {fn}: unexpected top-level sections {unexpected}")
    later = sorted(k for k in doc if LATER_PHASE_KEYS.match(k))
    if later:
        phase_ok = False
        note(f"  {fn}: Phase 2+ sections configured: {later}")
    if not unexpected and not later:
        note(f"  {fn}: top-level sections all Phase 1 -> {sorted(doc)}")

# reserved_for_later_phases must be documentation (strings), not live values
rlp = resolved.get(BASE, {}).get("reserved_for_later_phases", {})
non_str = {k: type(v).__name__ for k, v in flatten(rlp).items()
           if not isinstance(v, str)}
if non_str:
    phase_ok = False
    note(f"  reserved_for_later_phases holds non-string values: {non_str}")
else:
    note(f"  reserved_for_later_phases: {len(flatten(rlp))} entries, "
         f"all descriptive strings (no live values)")

# pending_phase_5 must NOT exist as a YAML key. The Phase 5 placeholders are a
# comment in stress_test.yaml, so the "overlays differ only in meta/drift"
# invariant checked at step 3 holds with no exceptions.
pp5_keys = [fn for fn in [BASE] + OVERLAYS if "pending_phase_5" in raw_docs[fn]]
if pp5_keys:
    phase_ok = False
    note(f"  pending_phase_5 present as a YAML key in {pp5_keys} -- must be a "
         f"comment, or it breaks the meta/drift-only invariant")
else:
    note("  no pending_phase_5 YAML key: Phase 5 network/resource placeholders "
         "are comments only")
    note("  stress_test.yaml declares its own incompleteness via "
         "meta.incomplete_until_phase = "
         f"{resolved.get('stress_test.yaml', {}).get('meta', {}).get('incomplete_until_phase')}")

# Every src/ module outside the CURRENT phase step must still be a placeholder.
# This is the check that would catch Phase 2+ code appearing early. It is scoped
# to a whitelist rather than relaxed to "any implemented module is fine".
import ast

IMPLEMENTED_BY_STEP = {
    "src\\data\\loader.py": "Phase 1 / Step 3",
    "src\\data\\generator.py": "Phase 1 / Step 3",
    "src\\data\\stream.py": "Phase 1 / Step 3",
    "src\\data\\preprocessing.py": "Phase 1 / Step 4",
    "src\\utils\\config.py": "Phase 1 / Step 5",
    "src\\utils\\seed.py": "Phase 1 / Step 5",
    "src\\utils\\logger.py": "Phase 1 / Step 5",
}
nonempty, expected = [], []
for py in (ROOT / "src").rglob("*.py"):
    tree = ast.parse(io.open(py, encoding="utf-8").read())
    body = [n for n in tree.body
            if not (isinstance(n, ast.Expr) and isinstance(n.value, ast.Constant)
                    and isinstance(n.value.value, str))]
    if not body:
        continue
    rel = str(py.relative_to(ROOT))
    key = rel.replace("/", "\\")
    (expected if key in IMPLEMENTED_BY_STEP else nonempty).append(rel)
if nonempty:
    phase_ok = False
    note(f"  src/ modules implemented ahead of their phase: {sorted(nonempty)}")
else:
    n_py = len(list((ROOT / "src").rglob("*.py")))
    steps = sorted({IMPLEMENTED_BY_STEP[r.replace("/", "\\")] for r in expected})
    note(f"  src/: {n_py} modules; {len(expected)} implemented "
         f"({', '.join(sorted(expected))}), all for {' + '.join(steps)}")
    note(f"  remaining {n_py - len(expected)} modules still docstring-only "
         f"(0 executable statements) -- no Phase 2+ implementation")
check("9. No Phase 2+ configuration introduced", phase_ok,
      "Phase 1 sections only; later phases documented, not valued")

# =============================================================================
# 10. Cross-cutting integrity re-checks (leakage + ground-truth quarantine)
# =============================================================================
integ_ok = True
INTEGRITY = {
    "preprocessing.normalization.forbid_global_fit": True,
    "preprocessing.normalization.adaptation": "frozen_after_baseline",
    "preprocessing.missing.interpolate_direction": "backward_only",
    "preprocessing.filtering.causal": True,
    "preprocessing.outliers.statistics_source": "baseline_frozen",
    "drift.modify_labels": False,
    "drift.magnitude_units": "baseline_sigma",
    "drift.affected_features.actuator_policy": "exclude",
    "drift.clip_to_physical_range": True,
    "drift.report_realised_magnitude": True,
    "streaming.shuffle": False,
    "reproducibility.strict": True,
}
for fn in [BASE] + OVERLAYS:
    flat = flatten(resolved.get(fn, {}))
    bad = {k: flat.get(k, "<absent>") for k, exp in INTEGRITY.items()
           if flat.get(k, "<absent>") != exp}
    if bad:
        integ_ok = False
        note(f"  {fn}: integrity keys wrong -> {bad}")
gt = resolved.get(BASE, {}).get("ground_truth", {})
forbidden = set(gt.get("forbidden_consumers") or [])
allowed = set(gt.get("allowed_consumers") or [])
MUST_FORBID = {"models", "drift_detectors", "reliability", "controller", "wds",
               "lri", "edge", "cloud", "hybrid", "adaptation"}
missing_forbid = MUST_FORBID - forbidden
if missing_forbid:
    integ_ok = False
    note(f"  ground_truth.forbidden_consumers missing {sorted(missing_forbid)}")
if forbidden & allowed:
    integ_ok = False
    note(f"  ground_truth: consumer in BOTH lists: {sorted(forbidden & allowed)}")
if integ_ok:
    note(f"  {len(INTEGRITY)} leakage/integrity keys correct in all 4 configs")
    note(f"  ground_truth: {len(forbidden)} forbidden / {len(allowed)} allowed "
         f"consumers, disjoint")
check("10. Leakage + ground-truth quarantine intact", integ_ok,
      "no config change weakened a Phase 1 integrity guarantee")

# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("STEP 2 CONFIGURATION VERIFICATION  --  HAI TRANSITION")
print("=" * 78)
for line in _details:
    print(line)
print("-" * 78)
passed = sum(1 for _, ok, _ in results if ok)
for name, ok, detail in results:
    print(f"[{'PASS' if ok else 'FAIL'}] {name}")
    if detail:
        print(f"       {detail}")
print("-" * 78)
print(f"{passed}/{len(results)} checks passed")
print("=" * 78)
sys.exit(0 if passed == len(results) else 1)
