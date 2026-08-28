"""Step 3 verification harness -- loader.py / generator.py / stream.py.

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test. Reads the real HAI
files from data/raw/ and the real configs from config/.

This is NOT the pytest suite (that is Step 6). It is an end-to-end executable
check that Step 3's three modules do what Step 3 claims.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe drift_aware_edge_cloud/verify_step3.py
"""

from __future__ import annotations

import copy
import io
import sys
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
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


# -----------------------------------------------------------------------------
# Config resolution -- identical semantics to verify_step2.py
# -----------------------------------------------------------------------------


def deep_merge(base: dict, over: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in over.items():
        if k in out and isinstance(out[k], dict) and isinstance(v, dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve(filename: str, _seen: tuple[str, ...] = ()) -> dict:
    if filename in _seen:
        raise ValueError(f"circular _extends: {' -> '.join(_seen + (filename,))}")
    raw = yaml.safe_load(io.open(CONFIG_DIR / filename, encoding="utf-8").read())
    parent = raw.pop("_extends", None)
    if parent is None:
        return raw
    return deep_merge(resolve(parent, _seen + (filename,)), raw)


CFG = {name: resolve(f"{name}.yaml")
       for name in ("default", "sudden_drift", "gradual_drift", "stress_test")}

# =============================================================================
# 1. Modules import and are actually implemented
# =============================================================================
try:
    from src.data import generator, loader, stream

    impl = []
    for mod, path in ((loader, "loader.py"), (generator, "generator.py"),
                      (stream, "stream.py")):
        src = Path(mod.__file__).read_text(encoding="utf-8")
        n_def = src.count("\ndef ") + src.count("\nclass ")
        impl.append((path, n_def, "Status   : IMPLEMENTED" in src))
        note(f"  {path:<14} {n_def:>3} top-level defs/classes, "
             f"status IMPLEMENTED={impl[-1][2]}")
    import_ok = all(n > 3 and st for _, n, st in impl)
    import_err = ""
except Exception as exc:                                        # pragma: no cover
    import_ok, import_err = False, f"{type(exc).__name__}: {exc}"
    note(f"  import failed: {import_err}")

check("1. Step 3 modules import and are implemented", import_ok, import_err)
if not import_ok:
    print("\n".join(_details))
    sys.exit(1)

# =============================================================================
# 2. loader: real HAI baseline loads and re-measures its VERIFIED properties
# =============================================================================
d = CFG["default"]
baseline_keys = loader.resolve_baseline_keys(d)
note(f"  dataset.baseline_source -> keys {list(baseline_keys)}")

baseline = loader.load_baseline(d, root=ROOT)
b0 = baseline[0]
r = b0.report
t = r.time_axis
note(f"  {b0.key}: {r.n_rows:,} rows x {r.n_columns} process columns, "
     f"role={b0.role}")
note(f"  {b0.key}: {t.summary()}")

base_ok = (
    len(baseline) == 1
    and b0.key == "train1"
    and r.n_rows == 304_166
    and r.n_missing_cells == 0
    and r.row_count_matches_config is True
    and r.time_range_matches_config is True
    and t.monotonic_increasing
    and t.n_duplicate_timestamps == 297_940
    and t.modal_interval_s == 0.0
    and t.n_gaps == 0
    and t.n_blocks == 1
    and not r.non_numeric_columns
)
check("2. Baseline train1 loads; VERIFIED properties re-measured", base_ok,
      f"304,166 rows, 0 missing, modal interval 0.0s (flow_level), {t.n_blocks} contiguity block, "
      f"causal single-file baseline")

# =============================================================================
# 3. loader: inference stream loads and shares the baseline schema
# =============================================================================
infer = loader.load_inference_stream(d, root=ROOT)
ir = infer.report
columns = loader.assert_schema_match([b0, infer])
note(f"  {infer.key}: {ir.n_rows:,} rows, role={infer.role}, "
     f"{ir.time_axis.first} -> {ir.time_axis.last}")
note(f"  schema identical across train1/test1: {len(columns)} columns, same order")

infer_ok = (
    infer.key == "test1"
    and infer.role == loader.INFERENCE_ROLE
    and ir.n_rows == 624_613
    and ir.n_missing_cells == 0
    and ir.time_axis.n_gaps == 0
    and ir.time_axis.n_blocks == 1
    and len(columns) == ir.n_columns == 37
)
check("3. Inference stream test1 loads; schema matches baseline exactly", infer_ok,
      f"624,613 rows, {len(columns)} shared columns, 1 block")

# =============================================================================
# 4. Causality is enforced, not documented
# =============================================================================
causal_ok = True
with warnings.catch_warnings(record=True) as w:
    warnings.simplefilter("always")
    loader.assert_causal_baseline(baseline, infer, d)
    note(f"  assert_causal_baseline(train1, test1) passed, "
         f"{len(w)} warning(s) raised")

acausal = copy.deepcopy(d)
acausal["dataset"]["baseline_source"] = "train1_and_train2"
try:
    loader.resolve_baseline_keys(acausal)
    causal_ok = False
    note("  FAIL: acausal baseline_source was accepted")
except loader.CausalityError as exc:
    note(f"  acausal baseline refused: {type(exc).__name__}")

try:
    loader.profile_baseline(d, [infer])
    causal_ok = False
    note("  FAIL: profile_baseline accepted the inference stream")
except loader.CausalityError:
    note("  profile_baseline refused a non-baseline role (sigma cannot come "
         "from test1)")

# The target. Until 2026-08-27 this asserted only that resolve_target REFUSES,
# because dataset.task was 'unresolved'. HAI 23.05 ships official label sidecars
# and the task is now resolved, so asserting a refusal on the live config would
# be asserting something false. The guard is not dropped -- it is split into the
# four statements it was standing in for, three of which are new:
#   (a) on the live config the target resolves to the sidecar's label column,
#   (b) and that column is NOT a process channel (i.e. not a derived label),
#   (c) the refusal still fires on a config whose task is 'unresolved',
#   (d) and a task claiming official labels with no label_file still refuses,
#       so the resolution cannot be half-declared.
target = loader.resolve_target(d)
declared_label = d["dataset"].get("target_column") or d["dataset"].get("label_column")
if target != declared_label:
    causal_ok = False
    note(f"  FAIL: resolve_target returned {target!r}, expected the declared "
         f"target/label column {declared_label!r}")
elif target in set(columns):
    causal_ok = False
    note(f"  FAIL: target {target!r} is also a feature column -- that "
         f"would be a derived label, not the official one")
else:
    note(f"  resolve_target -> {target!r}, read from dataset; "
         f"absent from the {len(columns)} feature columns (not derived)")

unresolved = copy.deepcopy(d)
unresolved["dataset"]["task"] = "unresolved"
try:
    loader.resolve_target(unresolved)
    causal_ok = False
    note("  FAIL: resolve_target returned a target while task is 'unresolved'")
except loader.UnresolvedTaskError:
    note("  resolve_target still raises while dataset.task is 'unresolved' "
         "(no label fabricated)")

half = copy.deepcopy(d)
half["dataset"]["target_column"] = None
half["dataset"]["label_column"] = None
try:
    loader.resolve_target(half)
    causal_ok = False
    note("  FAIL: resolve_target accepted 'supervised_classification' with no "
         "target_column")
except loader.ConfigError:
    note("  resolve_target refuses 'supervised_classification' without a "
         "target_column (the target must be explicitly declared)")

check("4. Leakage/causality guards fire: acausal baseline, inference-derived "
      "sigma, half-declared or unresolved target", causal_ok,
      "each violation raises rather than degrading quietly")

# =============================================================================
# 5. Baseline profile: measured statistics agree with PROVENANCE.json
# =============================================================================
profile = loader.profile_baseline(d, baseline)
zv = profile.zero_variance_agreement
note(f"  profiled {profile.n_rows:,} baseline rows from {list(profile.source_keys)}")
note(f"  zero-variance: declared={zv['declared_n']} measured={zv['measured_n']} "
     f"declared_but_not_measured={zv['declared_but_not_measured']} "
     f"measured_but_not_declared={zv['measured_but_not_declared']}")
note(f"  after dropping zero-variance: {len(profile.feature_names)} features = "
     f"{len(profile.continuous)} continuous + {len(profile.discrete)} discrete")
top5 = profile.top_variance(5)
note("  top_variance(5): " + ", ".join(
    f"{c} (sigma={profile.sigma(c):.4g})" for c in top5))

profile_ok = (
    profile.n_rows == 304_166
    and zv["measured_n"] == 0
    and not zv["declared_but_not_measured"]
    and not zv["measured_but_not_declared"]
    and len(profile.continuous) == 30
    and len(profile.discrete) == 7
    and len(profile.feature_names) == 37
    and len(profile.continuous) + len(profile.discrete) == len(profile.feature_names)
    and profile.top_variance(5) == top5           # deterministic on repeat
    and all(profile.sigma(c) > 0 for c in profile.continuous)
)
check("5. Baseline profile matches independently measured PROVENANCE figures",
      profile_ok,
      f"0/0 zero-variance columns agree exactly; 30 continuous + "
      f"{len(profile.discrete)} discrete features; top_variance deterministic")

# =============================================================================
# 6. Drift schedule: shapes and row indices are exactly as the configs claim
# =============================================================================
N = ir.n_rows
sched_ok = True


def sched(cfg: dict):
    return generator.magnitude_schedule(cfg, N)


s_none, st_none, en_none, _ = sched(CFG["default"])
if not (st_none is None and en_none is None and s_none.max() == 0.0):
    sched_ok = False
    note("  FAIL: default.yaml (scenario none) produced a non-zero schedule")
note(f"  none    : all {N:,} rows at magnitude 0.0, start=None (control condition)")

s_sud, st_sud, en_sud, sum_sud = sched(CFG["sudden_drift"])
exp_sud = (st_sud == 312_306 and en_sud == 624_613
           and s_sud[312_305] == 0.0 and s_sud[312_306] == 2.0
           and int((s_sud > 0).sum()) == 312_307)
sched_ok &= exp_sud
note(f"  sudden  : step 0 -> 2.0 sigma at row {st_sud:,} "
     f"({int((s_sud > 0).sum()):,} drifted rows)   expected 312,306 -> {exp_sud}")

s_gra, st_gra, en_gra, sum_gra = sched(CFG["gradual_drift"])
exp_gra = (st_gra == 249_845 and sum_gra["ramp_end"] == 374_768
           and sum_gra["ramp_rows"] == 124_923
           and s_gra[249_844] == 0.0 and 0 < s_gra[249_845] < 2.0
           and abs(s_gra[374_767] - 2.0) < 1e-12 and s_gra[624_612] == 2.0
           and bool(np.all(np.diff(s_gra) >= -1e-15)))
sched_ok &= exp_gra
note(f"  gradual : ramp rows {st_gra:,} -> {sum_gra['ramp_end']:,} "
     f"({sum_gra['ramp_rows']:,} rows), then holds 2.0 sigma; monotonic "
     f"non-decreasing -> {exp_gra}")

s_str, st_str, en_str, sum_str = sched(CFG["stress_test"])
levels = sorted({round(float(v), 6) for v in np.unique(s_str) if v > 0})
plateaus = sum_str["plateaus"]
exp_str = (st_str == 218_615 and len(plateaus) == 5
           and levels == [0.6, 1.2, 1.8, 2.4, 3.0]
           and plateaus[0]["from"] == 218_615 and plateaus[-1]["to"] == 624_613
           and all(81_198 <= p["to"] - p["from"] <= 81_201 for p in plateaus))
sched_ok &= exp_str
note(f"  stress  : 5 plateaus of ~{plateaus[0]['to'] - plateaus[0]['from']:,} rows "
     f"from row {st_str:,}, levels {levels} sigma -> {exp_str}")

rec = copy.deepcopy(CFG["default"])
rec["drift"].update(scenario="recurring", start_fraction=0.2, magnitude=2.0)
s_rec, st_rec, en_rec, sum_rec = sched(rec)
segs = sum_rec["drifted_segments"]
exp_rec = (len(segs) == 2 and segs[0] == (124_923, 218_615)
           and segs[1] == (312_307, 405_999)
           and s_rec[250_000] == 0.0 and s_rec[130_000] == 2.0)
sched_ok &= exp_rec
note(f"  recurring: drifted segments {segs}, clean gap between them -> {exp_rec}")

check("6. Drift schedules resolve to the exact row indices the configs document",
      bool(sched_ok),
      "sudden@312306, gradual ramp 249845->374768, stress 5 plateaus from 218615, "
      "recurring 2 cycles with a clean interval")

# =============================================================================
# 7. Generator purity: raw input untouched, only selected channels changed
# =============================================================================
raw_before = infer.frame.copy(deep=True)
base_before = b0.frame.copy(deep=True)

drifted, gt = generator.inject(CFG["sudden_drift"], infer, profile)

changed = [c for c in drifted.frame.columns
           if not np.array_equal(drifted.frame[c].to_numpy(dtype=float),
                                 infer.frame[c].to_numpy(dtype=float))]
discrete_changed = [c for c in changed if profile.columns[c].is_discrete]

purity_ok = (
    infer.frame.equals(raw_before)                     # input frame untouched
    and b0.frame.equals(base_before)                   # baseline untouched
    and drifted.frame is not infer.frame
    and set(changed) == set(gt.affected_features)
    and len(changed) == 5
    and not discrete_changed
    and len(drifted.frame) == N
    and list(drifted.frame.columns) == list(infer.frame.columns)
)
note(f"  input frame unmodified: {infer.frame.equals(raw_before)}; "
     f"baseline unmodified: {b0.frame.equals(base_before)}")
note(f"  columns changed: {len(changed)} = {sorted(changed)}")
note(f"  discrete/actuator columns changed: {len(discrete_changed)} "
     f"(actuator_policy 'exclude')")
check("7. Generator is pure: raw inputs unmodified, exactly the 5 selected "
      "continuous channels changed", purity_ok,
      "no actuator/state channel offset; column set and row count preserved")

# =============================================================================
# 8. Pre-drift region is value-identical; post-drift shift is measured
# =============================================================================
# Value equality, NOT DataFrame.equals: an offset of a fractional sigma cannot be
# stored in an integer column, so the drifted channels are legitimately promoted
# int64 -> float64. The scientific claim is that no undrifted VALUE moved.
def values_equal(lo: int, hi: int) -> bool:
    return all(
        np.array_equal(
            drifted.frame[c].to_numpy(dtype=float)[lo:hi],
            infer.frame[c].to_numpy(dtype=float)[lo:hi],
        )
        for c in drifted.frame.columns
    )


pre_identical = values_equal(0, 312_306)
promoted = [c for c in drifted.frame.columns
            if drifted.frame[c].dtype != infer.frame[c].dtype]
post_differs = all(
    not np.array_equal(
        drifted.frame[c].to_numpy(dtype=float)[312_306:],
        infer.frame[c].to_numpy(dtype=float)[312_306:],
    )
    for c in gt.affected_features
)
untouched_post = [
    c for c in drifted.frame.columns
    if c not in gt.affected_features
    and not np.array_equal(
        drifted.frame[c].to_numpy(dtype=float)[312_306:],
        infer.frame[c].to_numpy(dtype=float)[312_306:],
    )
]
note(f"  rows 0..312,305 value-identical to raw WUSTL in all channels: "
     f"{pre_identical}")
note(f"  dtype promoted int64->float64 (values unchanged): {sorted(promoted)}")
note(f"  rows 312,306..624,612 differ in every affected channel: {post_differs}; "
     f"unaffected channels changed there: {len(untouched_post)}")
note(f"  requested {gt.drift_magnitude} sigma; realised per channel:")
for c, v in gt.realised_magnitude.items():
    note(f"    {c:<16} {v:+.4f} sigma  (sigma={profile.sigma(c):.6g})")
ss = gt.schedule_summary
note(f"  realised mean {ss.get('realised_mean_magnitude_sigma', float('nan')):.4f} "
     f"sigma vs requested {ss.get('requested_mean_magnitude_sigma', float('nan')):.4f} "
     f"sigma  (attenuation ratio "
     f"{ss.get('attenuation_ratio', float('nan')):.4f})")
for n_ in gt.notes:
    note(f"  note: {n_}")

shift_ok = (
    pre_identical and post_differs and not untouched_post
    and len(gt.realised_magnitude) == 5
    and gt.n_drifted_rows == 312_307
    and all(v > 0 for v in gt.realised_magnitude.values())
)
check("8. Pre-drift region value-identical to raw WUSTL; realised magnitude "
      "measured after clipping", shift_ok,
      f"realised mean {ss.get('realised_mean_magnitude_sigma', 0):.4f} sigma of "
      f"{ss.get('requested_mean_magnitude_sigma', 0):.1f} requested "
      f"({ss.get('attenuation_ratio', 0):.1%} of target)")

# =============================================================================
# 9. Determinism, and the seed actually matters
# =============================================================================
import gc
d2, gt2 = generator.inject(CFG["sudden_drift"], infer, profile)
same = d2.frame.equals(drifted.frame) and gt2.realised_magnitude == gt.realised_magnitude
del d2, gt2
gc.collect()

noisy = copy.deepcopy(CFG["sudden_drift"])
noisy["drift"]["mechanism"] = "noise"
n_a, gta = generator.inject(noisy, infer, profile)
n_b, gtb = generator.inject(noisy, infer, profile)
noise_reproducible = n_a.frame.equals(n_b.frame)
del n_b, gtb
gc.collect()

reseeded = copy.deepcopy(noisy)
reseeded["reproducibility"]["random_seed"] = int(
    noisy["reproducibility"]["random_seed"]) + 1
n_c, gtc = generator.inject(reseeded, infer, profile)
seed_matters = not n_c.frame.equals(n_a.frame)
del n_a, n_c, gta, gtc
gc.collect()

note(f"  offset run twice, identical frames: {same}")
note(f"  stochastic 'noise' mechanism reproducible under the same seed: "
     f"{noise_reproducible}")
note(f"  changing reproducibility.random_seed changes the realisation: "
     f"{seed_matters}")
check("9. Generator is deterministic under a fixed seed and responsive to it",
      bool(same and noise_reproducible and seed_matters),
      "same seed -> identical bytes; different seed -> different realisation")

# =============================================================================
# 10. Ground truth stays out of the feature stream
# =============================================================================
gt_fields = set(d["ground_truth"]["fields"])
leaked_cols = gt_fields & set(map(str, drifted.frame.columns))
extra_attrs = [a for a in ("ground_truth", "gt", "drift_start_index", "schedule_summary")
               if hasattr(drifted, a)]
sidecar = generator.write_sidecar(gt, d, root=ROOT)
payload = yaml.safe_load(sidecar.read_text(encoding="utf-8"))
note(f"  drifted frame columns carrying a ground-truth field name: "
     f"{sorted(leaked_cols) or 'none'}")
note(f"  DriftedStream attributes exposing ground truth: {extra_attrs or 'none'}")
note(f"  sidecar written: {sidecar.relative_to(ROOT)}")
note(f"  sidecar keys == ground_truth.fields: "
     f"{sorted(payload) == sorted(gt_fields)}  ({len(payload)} fields)")
note(f"  sidecar drift_start_index={payload['drift_start_index']}, "
     f"scenario={payload['scenario']}, seed={payload['random_seed']}, "
     f"affected={len(payload['affected_features'])} channels")

quarantine_ok = (
    not leaked_cols
    and not extra_attrs
    and sidecar.parent.name == "synthetic"
    and sorted(payload) == sorted(gt_fields)
    and payload["drift_start_index"] == 312_306
)
check("10. Ground truth quarantined: absent from the frame, present only in the "
      "sidecar", quarantine_ok,
      f"{len(payload)} evaluation-only fields in "
      f"{sidecar.relative_to(ROOT).as_posix()}")

# =============================================================================
# 11. Streaming: order, no lookahead, and the window grid
# =============================================================================
shuffled = copy.deepcopy(d)
shuffled["streaming"]["shuffle"] = True
try:
    stream.plan(shuffled)
    shuffle_refused = False
except loader.ConfigError:
    shuffle_refused = True
note(f"  streaming.shuffle=true refused: {shuffle_refused}")

cs = stream.ChronologicalStream(drifted, d)
p = cs.plan
expected_grid = stream.window_index(d, N)
seen_starts: list[int] = []
seen_ids: list[int] = []
last_end_time = None
order_ok = True
for w in cs.windows():
    seen_starts.append(w.start_index)
    seen_ids.append(w.window_id)
    if last_end_time is not None and w.end_time < last_end_time:
        order_ok = False
    last_end_time = w.end_time
    if w.end_index - w.start_index != p.window_size or not w.contiguous:
        order_ok = False

n_expected = (N - p.window_size) // p.step_size + 1
note(f"  window_size={p.window_size} step_size={p.step_size} emit={p.emit!r}")
note(f"  windows emitted {len(seen_starts):,} of {n_expected:,} candidates; "
     f"first start={seen_starts[0]} last start={seen_starts[-1]}")
note(f"  grid matches window_index() helper: "
     f"{seen_starts == [s for s, _ in expected_grid]}")
note("  " + cs.stats.summary().replace("\n", "\n  "))

stream_ok = (
    shuffle_refused and order_ok
    and len(seen_starts) == n_expected == 62_457
    and seen_starts == sorted(seen_starts)
    and seen_starts == [s for s, _ in expected_grid]
    and seen_ids == list(range(len(seen_ids)))
    and cs.stats.trailing_rows_dropped == (N - p.window_size) % p.step_size == 3
    and cs.stats.n_windows_skipped_noncontiguous == 0
    and cs.stats.n_windows_skipped_invalid == 0
    and cs.stats.n_windows_emitted == len(seen_starts)
)
check("11. Streaming is chronological, refuses shuffling, and emits the exact "
      "window grid", stream_ok,
      f"62,457 windows over 624,613 rows, strictly increasing, "
      f"{cs.stats.trailing_rows_dropped} trailing rows dropped")

# =============================================================================
# 12. No lookahead, and the drift onset lands in the predicted window
# =============================================================================
look_ok = True

# A window may only be built once its final row has arrived. Force the violation
# and confirm the guard, rather than trusting the docstring.
probe = stream.ChronologicalStream(drifted, d)
probe._cursor = 10
try:
    probe._build(0)          # needs row 49, cursor is at 10
    look_ok = False
    note("  FAIL: a window was built from rows that had not arrived")
except stream.StreamOrderError:
    note("  _build() refuses to read past the cursor (no lookahead)")

backwards = stream.ChronologicalStream(drifted, d)
backwards._last_index = 100
try:
    backwards._advance(50)
    look_ok = False
    note("  FAIL: the cursor moved backwards")
except stream.StreamOrderError:
    note("  _advance() refuses a backwards index")

# Ground truth -> window id, using only the evaluation-side helper.
onset = payload["drift_start_index"]
first_post = next(i for i, (s, _) in enumerate(expected_grid) if s >= onset)
straddling = [i for i, (s, e) in enumerate(expected_grid) if s < onset < e]
note(f"  drift onset row {onset:,} -> first fully post-drift window id "
     f"{first_post:,} (start row {expected_grid[first_post][0]:,})")
note(f"  {len(straddling)} window(s) straddle the onset "
     f"(ids {straddling[0]}..{straddling[-1]}) -- these are neither pre- nor "
     f"post-drift and must be excluded from the post-drift metric")

# The drifted stream and the clean stream are consumed by identical code.
clean_stream = stream.ChronologicalStream(infer, d)
clean_starts = [w.start_index for w in clean_stream.windows()]
note(f"  clean and drifted streams yield the same grid: "
     f"{clean_starts == seen_starts}")

look_ok &= (
    first_post == 31_231
    and expected_grid[first_post][0] == 312_310
    and len(straddling) == 5
    and clean_starts == seen_starts
)
check("12. No-lookahead guards fire; ground truth maps onto the window grid",
      bool(look_ok),
      "onset row 312,306 -> window 31,231; 5 straddling windows identified for "
      "exclusion; clean and drifted streams share one code path")

# =============================================================================
# 13. The other two scenario configs inject end-to-end on the real stream
# =============================================================================
scen_ok = True
for name, expect_n, expect_start in (("gradual_drift", 5, 249_845),
                                     ("stress_test", 10, 218_615)):
    ds_, gt_ = generator.inject(CFG[name], infer, profile)
    s_ = gt_.schedule_summary
    pre_ok = all(
        np.array_equal(ds_.frame[c].to_numpy(dtype=float)[:expect_start],
                       infer.frame[c].to_numpy(dtype=float)[:expect_start])
        for c in ds_.frame.columns
    )
    disc = [c for c in gt_.affected_features if profile.columns[c].is_discrete]
    ok = (
        gt_.drift_start_index == expect_start
        and len(gt_.affected_features) == expect_n
        and not disc
        and pre_ok
        and len(gt_.realised_magnitude) == expect_n
        and ds_.frame.shape == infer.frame.shape
    )
    scen_ok &= ok
    att = s_.get("attenuation_ratio", float("nan"))
    note(f"  {name:<14} start={gt_.drift_start_index:,} "
         f"channels={len(gt_.affected_features)} (discrete={len(disc)}) "
         f"realised_mean={s_.get('realised_mean_magnitude_sigma', float('nan')):.4f} "
         f"of {s_.get('requested_mean_magnitude_sigma', float('nan')):.4f} sigma "
         f"({att:.1%})  pre-drift untouched={pre_ok} -> {ok}")
    del ds_, gt_
    gc.collect()

check("13. gradual_drift and stress_test inject end-to-end on the real stream",
      bool(scen_ok),
      "correct onset rows, channel counts (5 / 10), no actuator channels, "
      "pre-drift region untouched in both")

# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("STEP 3 VERIFICATION  --  loader.py / generator.py / stream.py")
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
