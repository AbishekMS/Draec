"""Step 4 verification harness -- preprocessing.py (causal pipeline).

Standalone and re-runnable. Lives at the project root, not under src/, so it can
never be mistaken for a component of the system under test. Reads the real HAI
files from data/raw/ and the real configs from config/.

This is NOT the pytest suite (that is Step 6). It is an end-to-end executable
check that Step 4 does what Step 4 claims -- in particular that no statistic is
ever fitted on data the model would not yet have seen, and that the default
normalization preserves the injected drift instead of cancelling it.

Run:
    PYTHONIOENCODING=utf-8 .venv/Scripts/python.exe drift_aware_edge_cloud/verify_step4.py
"""

from __future__ import annotations

import copy
import dataclasses
import gc
import io
import sys
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
# Config resolution -- identical semantics to verify_step2.py / verify_step3.py
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


def variant(base: dict, path: str, value) -> dict:
    """Return a deep copy of `base` with one dotted key overridden."""
    out = copy.deepcopy(base)
    node = out
    parts = path.split(".")
    for p in parts[:-1]:
        node = node.setdefault(p, {})
    node[parts[-1]] = value
    return out


def expect(exc_types, fn, *a, **kw) -> tuple[bool, str]:
    try:
        fn(*a, **kw)
    except exc_types as e:
        return True, f"{type(e).__name__}: {str(e).splitlines()[0][:110]}"
    except Exception as e:  # wrong exception type is a failure, not a pass
        return False, f"WRONG EXCEPTION {type(e).__name__}: {str(e)[:110]}"
    return False, "no exception raised"


# =============================================================================
# 1. Module imports and is actually implemented
# =============================================================================
try:
    from src.data import generator, loader, preprocessing, stream

    src_text = Path(preprocessing.__file__).read_text(encoding="utf-8")
    n_defs = sum(1 for ln in src_text.splitlines()
                 if ln.startswith("def ") or ln.startswith("class "))
    implemented = "Status   : IMPLEMENTED" in src_text
    check(
        "1. preprocessing.py imports, marked IMPLEMENTED, has real API",
        implemented and n_defs >= 15,
        f"{n_defs} top-level defs/classes; Status: "
        f"{'IMPLEMENTED' if implemented else 'NOT IMPLEMENTED'}",
    )
except Exception as e:  # pragma: no cover
    check("1. preprocessing.py imports", False, f"{type(e).__name__}: {e}")
    print("FATAL: cannot import; aborting.")
    raise SystemExit(1)

# =============================================================================
# 2. Load the real data once
# =============================================================================
cfg = CFG["default"]
baseline = loader.load_baseline(cfg, root=ROOT)
profile = loader.profile_baseline(cfg, baseline)
infer = loader.load_inference_stream(cfg, root=ROOT)
note(f"baseline files: {[b.key for b in baseline]}; "
     f"{len(baseline[0].frame):,} rows")
note(f"inference stream: {infer.key}; {len(infer.frame):,} rows")

stats = preprocessing.fit(cfg, baseline, profile)
check(
    "2. fit() on the baseline only",
    (stats.source_keys == tuple(b.key for b in baseline)
     and stats.n_rows == len(baseline[0].frame)
     and len(stats.columns) == len(profile.feature_names)
     and len(stats.continuous) == len(profile.continuous)
     and len(stats.discrete) == len(profile.discrete)
     and stats.history_tail is None),
    f"fitted on {stats.source_keys} / {stats.n_rows:,} rows; "
    f"{len(stats.columns)} features "
    f"({len(stats.continuous)} continuous + {len(stats.discrete)} discrete); "
    f"history_tail=None under the default frozen mode",
)

# =============================================================================
# 3. Leakage barriers actually fire
# =============================================================================
guards: list[tuple[str, bool, str]] = []

ok, msg = expect(loader.CausalityError, preprocessing.fit, cfg, [infer], profile)
guards.append(("fit() on the inference stream", ok, msg))

ok, msg = expect(
    loader.ConfigError, preprocessing.fit,
    variant(cfg, "preprocessing.normalization.forbid_global_fit", False),
    baseline, profile)
guards.append(("forbid_global_fit: false", ok, msg))

ok, msg = expect(
    preprocessing.LeakageError, preprocessing.transform,
    variant(cfg, "preprocessing.outliers.statistics_source", "rolling"),
    infer, stats)
guards.append(("outlier stats recomputed on post-drift data", ok, msg))

ok, msg = expect(
    loader.ConfigError, preprocessing.transform,
    variant(cfg, "preprocessing.outliers.action", "drop"), infer, stats)
guards.append(("outliers.action: drop", ok, msg))

ok, msg = expect(
    loader.ConfigError, preprocessing.transform,
    variant(cfg, "preprocessing.validation.on_failure", "drop"), infer, stats)
guards.append(("validation.on_failure: drop", ok, msg))

_c = variant(cfg, "preprocessing.filtering.enabled", True)
_c = variant(_c, "preprocessing.filtering.causal", False)
ok, msg = expect(preprocessing.LeakageError, preprocessing.transform,
                 _c, infer, stats)
guards.append(("non-causal (centred) filter", ok, msg))

_c = variant(cfg, "preprocessing.missing.method", "interpolate")
_c = variant(_c, "preprocessing.missing.interpolate_direction", "both")
# Only reachable when something is missing; force a hole in a copy.
_hole = dataclasses.replace(infer, frame=infer.frame.copy())
_hole.frame.iloc[10, 0] = np.nan
ok, msg = expect(preprocessing.LeakageError, preprocessing.transform,
                 _c, _hole, stats)
guards.append(("interpolate across a future sample", ok, msg))

ok, msg = expect(
    loader.ConfigError, preprocessing.fit,
    variant(cfg, "preprocessing.normalization.adaptation", "adaptive"),
    baseline, profile)
guards.append(("unknown adaptation mode", ok, msg))

for label, ok, msg in guards:
    note(f"  guard {'OK  ' if ok else 'FAIL'}  {label}: {msg}")
check(
    "3. Leakage / destructive-default barriers all fire",
    all(ok for _, ok, _ in guards),
    f"{sum(ok for _, ok, _ in guards)}/{len(guards)} guards raised the right "
    f"exception",
)
del _hole, guards, _c
gc.collect()

# =============================================================================
# 4. Frozen z-score is correct on the data it was fitted on
# =============================================================================
prep_base = preprocessing.transform(cfg, baseline[0], stats)
Zb = prep_base.frame.to_numpy(dtype=float)
mu_err = float(np.max(np.abs(Zb.mean(axis=0))))
sd = Zb.std(axis=0, ddof=0)
sd_err = float(np.max(np.abs(sd - 1.0)))
check(
    "4. frozen z-score self-consistency on the baseline",
    mu_err < 1e-9 and sd_err < 1e-9,
    f"max |mean| = {mu_err:.3e}, max |std - 1| = {sd_err:.3e} over "
    f"{Zb.shape[0]:,} x {Zb.shape[1]} values",
)
del prep_base, Zb
gc.collect()

# =============================================================================
# 5. Purity -- transform mutates nothing it was given
# =============================================================================
before_infer = infer.frame.copy()
before_baseline = baseline[0].frame.copy()
prep_inf = preprocessing.transform(cfg, infer, stats)
same_inf = infer.frame.equals(before_infer)
same_base = baseline[0].frame.equals(before_baseline)
# action: flag must leave every recorded value exactly as recorded
raw_matches = all(
    np.array_equal(prep_inf.raw_frame[c].to_numpy(dtype=float),
                   before_infer[c].to_numpy(dtype=float))
    for c in prep_inf.raw_frame.columns
)
check(
    "5. transform() is non-destructive; action 'flag' preserves raw values",
    same_inf and same_base and raw_matches,
    f"source frames unchanged: infer={same_inf}, baseline={same_base}; "
    f"all {len(prep_inf.raw_frame.columns)} post-pipeline raw columns "
    f"value-identical to the recorded file: {raw_matches}",
)
del before_infer, before_baseline
gc.collect()

# =============================================================================
# 6. What the pipeline measured on the real, undrifted inference stream
# =============================================================================
q = prep_inf.quality
n_out = int(q.outlier.sum())
n_rng = int(q.range_violation.sum())
n_valid = int(q.valid.sum())
note("quality report on clean hai-test1:")
for ln in q.summary().splitlines():
    note(f"  {ln}")
check(
    "6. quality report is populated and keeps every row",
    (q.n_rows == len(infer.frame) and n_valid == q.n_rows
     and int(q.unfilled.sum()) == 0 and len(prep_inf.frame) == q.n_rows),
    f"{q.n_rows:,} rows in, {len(prep_inf.frame):,} rows out (nothing dropped); "
    f"valid={n_valid:,}; outlier-flagged={n_out:,} ({n_out / q.n_rows:.3%}); "
    f"range-flagged={n_rng:,} ({n_rng / q.n_rows:.3%})",
)

# =============================================================================
# 7. Outlier flags never invalidate a row
# =============================================================================
check(
    "7. outlier / range flags do not remove rows from the valid mask",
    bool(np.all(q.valid[q.outlier])) and bool(np.all(q.valid[q.range_violation])),
    f"of {n_out:,} outlier-flagged rows, "
    f"{int(q.valid[q.outlier].sum()):,} remain valid; of {n_rng:,} "
    f"range-flagged rows, {int(q.valid[q.range_violation].sum()):,} remain "
    f"valid -- post-drift observations are kept by construction",
)

# =============================================================================
# 8. Inject sudden drift and confirm it SURVIVES normalization
# =============================================================================
scfg = CFG["sudden_drift"]
sinfer = infer
sbaseline = baseline
sprofile = profile
sstats = stats
drifted, gt = generator.inject(scfg, sinfer, sprofile)
affected = list(gt.affected_features)
start = int(gt.drift_start_index)
note(f"sudden_drift: start={start:,}, channels={affected}, "
     f"requested={gt.drift_magnitude}{gt.magnitude_units}, "
     f"realised={gt.realised_magnitude}")

prep_drift = preprocessing.transform(scfg, drifted, sstats)
prep_clean = prep_inf
Zd = prep_drift.frame.loc[:, affected].to_numpy(dtype=float)
Zc = prep_clean.frame.loc[:, affected].to_numpy(dtype=float)
shift_drift = float(np.mean(Zd[start:].mean(axis=0) - Zd[:start].mean(axis=0)))
shift_clean = float(np.mean(Zc[start:].mean(axis=0) - Zc[:start].mean(axis=0)))
pre_identical = np.allclose(Zd[:start], Zc[:start], atol=0, rtol=0)
survived = abs(shift_drift - shift_clean)
note(f"normalized post-minus-pre mean shift on affected channels: "
     f"drifted={shift_drift:+.4f} sigma, clean={shift_clean:+.4f} sigma")
check(
    "8. injected drift survives the default frozen normalization",
    pre_identical and survived > 1.0,
    f"pre-drift normalized values bit-identical to the clean run: "
    f"{pre_identical}; drift-attributable shift = {survived:.4f} sigma "
    f"(clean-stream shift {shift_clean:+.4f} is the confound and is subtracted)",
)
del Zd, Zc
import gc
gc.collect()
names = preprocessing.feature_names(scfg, sstats)
mean_cols = [names.index(f"{c}__mean") for c in affected]
clean_windows = list(stream.iter_windows(infer, scfg, valid_mask=prep_inf.quality.valid))
fm_clean = preprocessing.extract_features(scfg, prep_inf, clean_windows)
pre_mask_c = fm_clean.end_index <= start
post_mask_c = fm_clean.start_index >= start
pre_c = fm_clean.X[pre_mask_c][:, mean_cols].mean(axis=0)
post_c = fm_clean.X[post_mask_c][:, mean_cols].mean(axis=0)
delta_c = float(np.mean(post_c - pre_c))
del fm_clean, clean_windows

# Pre-calculate Check 15 metrics before releasing memory
deg = list(q.degenerate_bound_columns)
lo_b, hi_b = stats.outlier_bounds(cfg)
deg_rates = {c: int(((infer.frame[c] < lo_b[c]) | (infer.frame[c] > hi_b[c])).sum()) for c in deg}
keep = [c for c in stats.continuous if c not in set(deg)]
sub_keep = prep_inf.raw_frame[keep]
alt = (sub_keep.lt(lo_b[keep], axis=1) | sub_keep.gt(hi_b[keep], axis=1)).any(axis=1)
q_outlier_mean = float(q.outlier.mean())
alt_mean = float(alt.mean())
infer_len = len(infer.frame)
q_notes = list(q.notes)

zmean = prep_inf.frame.mean()
n_gt1 = int((zmean.abs() > 1.0).sum())
n_gt3 = int((zmean.abs() > 3.0).sum())
worst = zmean.abs().idxmax()
worst_z = float(zmean[worst])
worst_std = float(prep_inf.frame[worst].std(ddof=0))
zmean_abs_mean = float(zmean.abs().mean())
zmean_abs_median = float(zmean.abs().median())
zmean_len = len(zmean)

del sub_keep, alt, prep_clean, prep_inf, infer, sinfer, q
gc.collect()

# =============================================================================
# 9. Drift absorption by adaptation mode -- the [A6] measurement
# =============================================================================
modes = ("frozen_after_baseline", "running", "rolling")
stats_by_mode = {
    m: preprocessing.fit(
        variant(scfg, "preprocessing.normalization.adaptation", m),
        sbaseline, sprofile)
    for m in modes
}
absorb = preprocessing.measure_drift_absorption(
    scfg, drifted, stats_by_mode, affected, start)
note("adaptation-mode sensitivity (ASSUMPTION [A6]), measured not assumed:")
for m in modes:
    v = absorb[m]
    note(f"  {m:24s} normalized shift = "
         f"{v['mean_normalized_shift_sigma']:+8.4f} sigma   "
         f"retained vs frozen = {v['retained_fraction_vs_frozen']:6.2%}")
f_shift = abs(absorb["frozen_after_baseline"]["mean_normalized_shift_sigma"])
r_shift = abs(absorb["running"]["mean_normalized_shift_sigma"])
w_shift = abs(absorb["rolling"]["mean_normalized_shift_sigma"])
check(
    "9. adaptive normalization measurably absorbs drift; frozen does not",
    f_shift > r_shift > w_shift,
    f"|shift|: frozen={f_shift:.4f} > running={r_shift:.4f} > "
    f"rolling={w_shift:.4f} sigma -- the ordering the module's docstring "
    f"predicts, and the reason frozen_after_baseline is the default",
)
del stats_by_mode, absorb, sbaseline, sprofile, baseline
gc.collect()

# =============================================================================
# 10. Windowed feature extraction
# =============================================================================
names = preprocessing.feature_names(scfg, sstats)
expected_n = len(sstats.continuous) * 6 + len(sstats.discrete) * 2
windows = list(stream.iter_windows(drifted, scfg,
                                   valid_mask=prep_drift.quality.valid))
fm = preprocessing.extract_features(scfg, prep_drift, windows)
grid = stream.window_index(scfg, len(drifted.frame))
check(
    "10. feature matrix shape, naming and window grid",
    (len(names) == expected_n and fm.X.shape == (len(windows), expected_n)
     and len(windows) == len(grid)
     and list(zip(fm.start_index.tolist(), fm.end_index.tolist())) == grid
     and not np.isnan(fm.X).any()),
    f"{len(windows):,} windows x {expected_n} features "
    f"({len(sstats.continuous)}x6 continuous + {len(sstats.discrete)}x2 "
    f"discrete); grid matches stream.window_index(); 0 NaN",
)

# =============================================================================
# 11. Feature values are arithmetically what they claim to be
# =============================================================================
w = windows[1234]
col = sstats.continuous[0]
seg = prep_drift.frame[col].to_numpy(dtype=float)[w.start_index:w.end_index]
xs = np.arange(len(seg), dtype=float)
xs -= xs.mean()
want = {
    "mean": seg.mean(), "std": seg.std(ddof=0), "min": seg.min(),
    "max": seg.max(), "last": seg[-1],
    "slope": float((xs @ (seg - seg.mean())) / (xs @ xs)),
}
got = {s: float(fm.X[1234, names.index(f"{col}__{s}")]) for s in want}
cont_ok = all(abs(got[s] - float(want[s])) < 1e-12 for s in want)

dcol = sstats.discrete[0]
dseg = prep_drift.raw_frame[dcol].to_numpy(dtype=float)[w.start_index:w.end_index]
vals, counts = np.unique(dseg, return_counts=True)
d_want = {"mode": float(vals[int(np.argmax(counts))]),
          "n_changes": float((np.diff(dseg) != 0).sum())}
d_got = {s: float(fm.X[1234, names.index(f"{dcol}__{s}")]) for s in d_want}
disc_ok = all(abs(d_got[s] - d_want[s]) < 1e-12 for s in d_want)
note(f"window 1234 [{w.start_index}:{w.end_index}] {col}: "
     + ", ".join(f"{s}={got[s]:.6g}" for s in want))
note(f"window 1234 discrete {dcol}: "
     + ", ".join(f"{s}={d_got[s]:.6g}" for s in d_want))
check(
    "11. every statistic recomputed independently and matches",
    cont_ok and disc_ok,
    f"6/6 continuous statistics and 2/2 discrete statistics agree to <1e-12; "
    f"discrete features read RAW values so 'mode' stays an interpretable state",
)

# =============================================================================
# 12. No lookahead -- a window cannot see rows after its own end
# =============================================================================
k = 2000
wk = windows[k]
cutoff = min(len(prep_drift.frame), wk.end_index + 1000)
poisoned = prep_drift.frame.iloc[:cutoff].copy()
poisoned.iloc[wk.end_index:, :] = 999999.0
poisoned_raw = prep_drift.raw_frame.iloc[:cutoff].copy()
poisoned_raw.iloc[wk.end_index:, :] = 999999.0
prep_poisoned = dataclasses.replace(
    prep_drift, frame=poisoned, raw_frame=poisoned_raw)
fm_p = preprocessing.extract_features(scfg, prep_poisoned, windows[:k + 1])
unchanged = np.array_equal(fm_p.X[k], fm.X[k])
# window k-1 ends before window k, so it must also be untouched
earlier_unchanged = np.array_equal(fm_p.X[k - 1], fm.X[k - 1])
check(
    "12. corrupting every row AFTER a window changes none of its features",
    unchanged and earlier_unchanged,
    f"rows {wk.end_index:,}+ overwritten with 999999; window {k} "
    f"[{wk.start_index}:{wk.end_index}] features bit-identical: {unchanged}; "
    f"window {k - 1} also identical: {earlier_unchanged}",
)
del poisoned, poisoned_raw, prep_poisoned, fm_p
gc.collect()

# =============================================================================
# 13. Determinism / idempotence
# =============================================================================
prep_again = preprocessing.transform(scfg, drifted, sstats)
del drifted
gc.collect()
fm_again = preprocessing.extract_features(scfg, prep_again, windows)
det = (prep_again.frame.equals(prep_drift.frame)
       and np.array_equal(fm_again.X, fm.X)
       and preprocessing.feature_names(scfg, sstats) == names)
check(
    "13. pipeline is deterministic and idempotent",
    det,
    f"re-running transform() + extract_features() on the same inputs gives a "
    f"bit-identical {fm.X.shape[0]:,} x {fm.X.shape[1]} matrix and the same "
    f"feature ordering",
)
del prep_again, fm_again, prep_drift
gc.collect()

# =============================================================================
# 14. Drift is visible in the FEATURES, not just the raw stream
# =============================================================================
onset_window = int(np.searchsorted(fm.start_index, start))
pre_mask = fm.end_index <= start
post_mask = fm.start_index >= start
straddle = int(len(fm.X) - pre_mask.sum() - post_mask.sum())
mean_cols = [names.index(f"{c}__mean") for c in affected]
pre_v = fm.X[pre_mask][:, mean_cols].mean(axis=0)
post_v = fm.X[post_mask][:, mean_cols].mean(axis=0)
delta = float(np.mean(post_v - pre_v))
fm_start_onset = int(fm.start_index[onset_window])
fm_end_onset_prev = int(fm.end_index[onset_window - 1])

del fm, windows
gc.collect()

# delta_c was computed right after check 8 to minimize cumulative memory pressure

note(f"window-level __mean shift on affected channels: drifted={delta:+.4f}, "
     f"clean={delta_c:+.4f}, attributable={delta - delta_c:+.4f} sigma")
note(f"onset row {start:,} -> window {onset_window:,}; {straddle} straddling "
     f"window(s) to exclude from any post-drift metric")
check(
    "14. drift reaches the feature vectors and the onset maps to a window",
    abs(delta - delta_c) > 1.0 and straddle >= 1
    and fm_start_onset >= start
    and fm_end_onset_prev > start,
    f"attributable window-level shift {delta - delta_c:+.4f} sigma; onset row "
    f"{start:,} -> first fully post-drift window {onset_window:,}; "
    f"{straddle} straddling window(s)",
)

# =============================================================================
# 15. OPEN FINDINGS -- measured, reported, behaviour unchanged
# =============================================================================
# 15a. Degenerate IQR bounds on piecewise-constant channels.
note("FINDING: degenerate outlier bounds (q1 == q3 -> zero-width band):")
for c in deg:
    note(f"  {c}: q1=q3={float(stats.q1[c]):.5f}, baseline std="
         f"{float(stats.std[c]):.5f}, flags {deg_rates[c]:,} rows "
         f"({deg_rates[c] / infer_len:.2%})")
note(f"  row-level flag rate WITH degenerate channels    : "
     f"{q_outlier_mean:.4%}  ({len(stats.continuous)} channels)")
note(f"  row-level flag rate WITHOUT degenerate channels : "
     f"{alt_mean:.4%}  ({len(keep)} channels)")
note("  behaviour unchanged; awaiting a decision on what 'outlier' means for a "
     "constant setpoint")

# 15b. Pre-existing regime difference between train1 and test1, before any
# injected drift. Constant across the whole stream, so it does not confound the
# post-minus-pre contrast -- but a detector will see it from row 0.
note("FINDING: pre-existing train1 -> test1 shift, in frozen-z units, with NO "
     "drift injected:")
note(f"  |z| > 1 sigma: {n_gt1} of {zmean_len} features; |z| > 3 sigma: "
     f"{n_gt3}; mean |z| = {zmean_abs_mean:.4f}, median "
     f"{zmean_abs_median:.4f}")
note(f"  worst: {worst} at {worst_z:+.4f} sigma, within-test1 std "
     f"{worst_std:.3e}")
note(f"  clean-stream post-minus-pre shift measured in check 8 was "
     f"{shift_clean:+.4f} sigma, i.e. this offset is flat across the stream and "
     f"is subtracted out of the drift measurement")
check(
    "15. both open findings are self-reporting and quantified",
    (len(deg) > 0 and any("DEGENERATE OUTLIER BOUNDS" in n for n in q_notes)
     and any("EXCLUDING the" in n for n in q_notes)
     and alt_mean < q_outlier_mean
     and abs(shift_clean) < 0.2),
    f"{len(deg)} degenerate channel(s) named in the quality report; flag rate "
    f"{q_outlier_mean:.2%} -> {alt_mean:.2%} once excluded; "
    f"{n_gt1}/{zmean_len} features already shifted >1 sigma at deployment; "
    f"flat-offset confound bounded at {abs(shift_clean):.4f} sigma",
)

# =============================================================================
# Report
# =============================================================================
print("=" * 78)
print("STEP 4 VERIFICATION -- src/data/preprocessing.py")
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
