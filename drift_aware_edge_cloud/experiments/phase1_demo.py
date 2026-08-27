"""Phase 1 end-to-end demonstration.

Runs the whole Phase 1 data path -- configuration, seeding, loading, baseline
profiling, causal statistic fitting, controlled drift injection, preprocessing,
windowing, feature extraction -- over the three drift scenarios, and writes what
it MEASURED to `results/` and `plots/`.

What this script is
-------------------
A demonstration that the pipeline runs and that its outputs are what the design
claims. Every number it prints is computed during the run; nothing is asserted
in advance and nothing is hard-coded.

What this script is NOT
-----------------------
It is not an experiment. There is no model, no drift detector, no reliability
estimator and no orchestration decision in Phase 1, so there is no accuracy,
latency or Edge/Cloud/Hybrid result to report. Any such number here would be
fabricated. The one comparative measurement it makes -- how much of an injected
shift survives each normalization mode -- is a property of the preprocessing
code, and it is reported whichever way it comes out.

Ground truth
------------
`GroundTruth` is read here only for plotting and for the drift-survival
measurement. `config/*.yaml -> ground_truth.allowed_consumers` lists exactly
`[metrics, plots, statistical_evaluation]`; this script is the third of those.
It is never passed to anything that will later have to detect the drift on its
own -- there is nothing of that kind in Phase 1 to pass it to.

Usage
-----
    python experiments/phase1_demo.py
    python experiments/phase1_demo.py --scenarios sudden_drift --max-rows 20000
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import generator, loader, preprocessing as pp, stream   # noqa: E402
from src.utils import config as cfgmod, logger as logmod, seed as seedmod  # noqa: E402

SCENARIOS = ("sudden_drift", "gradual_drift", "stress_test")
BASE = "default"


# -----------------------------------------------------------------------------
# reporting helpers -- narrow, so that no formatting code can invent a value
# -----------------------------------------------------------------------------

def rule(title: str = "") -> None:
    print("=" * 78 if not title else f"\n{'=' * 78}\n{title}\n{'=' * 78}")


def say(label: str, value: object) -> None:
    print(f"  {label:.<48} {value}")


def write_csv(frame: pd.DataFrame, path: Path, *, note: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)
    say(f"wrote {path.relative_to(ROOT)}", f"{len(frame):,} rows [{note}]")
    return path


# -----------------------------------------------------------------------------
# stage 1 -- configuration, seed, logging  (shared by every scenario)
# -----------------------------------------------------------------------------

def stage_configuration() -> dict:
    rule("STAGE 1  configuration, seeding, logging")
    say("configs discovered", ", ".join(cfgmod.available(ROOT / "config")))
    cfg = cfgmod.resolve(BASE, config_dir=ROOT / "config")
    cfgmod.validate(cfg)
    say("base config", f"{BASE}  fingerprint={cfgmod.fingerprint(cfg)}")

    logmod.configure(cfg, force=True)
    record = seedmod.seed_everything(cfg)
    say("master seed", f"{record.master} (strict={record.strict}, "
                       f"global_seeded={record.global_seeded})")
    say("component seeds", f"{len(seedmod.COMPONENTS)} registered, "
                           f"drift={seedmod.component_id('drift')}")
    say("task", f"{cfg['dataset']['task']!r} -- no target resolved, "
                f"no label fabricated")
    return cfg


# -----------------------------------------------------------------------------
# stage 2 -- load the real HAI files and fit everything on the baseline
# -----------------------------------------------------------------------------

def stage_baseline(cfg: dict, max_rows: int | None):
    rule("STAGE 2  loading + baseline profiling  (causal: baseline only)")
    t0 = time.perf_counter()
    baseline = loader.load_baseline(cfg, root=ROOT, max_rows=max_rows)
    for lf in baseline:
        say(f"baseline {lf.key} [{lf.role}]", f"{len(lf.frame):,} rows x "
                                              f"{lf.frame.shape[1]} columns")
    say("baseline keys", f"{loader.resolve_baseline_keys(cfg)}  "
                         f"(train2 excluded: acausal)")

    profile = loader.profile_baseline(cfg, baseline)
    say("profiled columns", f"{len(profile.columns)}")
    say("feature channels", f"{len(profile.feature_names)} "
                            f"({len(profile.continuous)} continuous + "
                            f"{len(profile.discrete)} discrete)")
    say("zero-variance dropped", f"{len(profile.dropped_zero_variance)} "
                                 f"-> {profile.zero_variance_agreement['note']}")

    stats = pp.fit(cfg, baseline, profile)
    say("statistics fitted on", f"{stats.source_keys}, {stats.n_rows:,} rows")

    # Self-consistency, measured rather than assumed: z-scoring the baseline with
    # its own frozen statistics must return mean 0 / std 1.
    check = pp.transform(cfg, baseline[0], stats)
    cont = list(stats.continuous)
    say("frozen z-score self-check",
        f"max|mean|={check.frame[cont].mean().abs().max():.3e}  "
        f"max|std-1|={(check.frame[cont].std(ddof=0) - 1).abs().max():.3e}")

    infer = loader.load_inference_stream(cfg, root=ROOT, max_rows=max_rows)
    say(f"inference {infer.key} [{infer.role}]", f"{len(infer.frame):,} rows")
    say("elapsed", f"{time.perf_counter() - t0:.1f}s")
    return baseline, profile, stats, infer


# -----------------------------------------------------------------------------
# stage 3 -- the clean stream: a confounder control, not a scenario
# -----------------------------------------------------------------------------

def stage_clean(cfg: dict, infer, stats) -> pp.PreprocessedStream:
    rule("STAGE 3  clean inference stream  (control: no drift injected)")
    prepared = pp.transform(cfg, infer, stats)
    q = prepared.quality
    say("rows in / rows out", f"{len(infer.frame):,} -> {len(prepared.frame):,} "
                              f"(nothing dropped)")
    say("valid rows", f"{int(q.valid.sum()):,} / {q.n_rows:,}")
    say("outlier-flagged", f"{int(q.outlier.sum()):,} "
                           f"({q.outlier.mean() * 100:.1f}%)")
    say("degenerate IQR channels", f"{len(q.degenerate_bound_columns)} "
                                   f"[OPEN FINDING 3, unrepaired]")
    for n in q.notes:
        print(f"      note: {n}")
    return prepared


# -----------------------------------------------------------------------------
# stage 4 -- one scenario end to end
# -----------------------------------------------------------------------------

def run_scenario(name: str, baseline, profile, stats, infer) -> dict:
    rule(f"STAGE 4  scenario: {name}")
    cfg = cfgmod.resolve(name, config_dir=ROOT / "config")
    cfgmod.validate(cfg)
    touched = cfgmod.assert_overlay_discipline(f"{name}.yaml",
                                               config_dir=ROOT / "config")
    say("overlay discipline", f"{name}.yaml touches only {touched}")
    say("fingerprint", cfgmod.fingerprint(cfg))
    say("drift config", f"scenario={cfg['drift']['scenario']} "
                        f"mechanism={cfg['drift']['mechanism']} "
                        f"magnitude={cfg['drift']['magnitude']} "
                        f"{cfg['drift']['magnitude_units']}")

    # -- injection ------------------------------------------------------------
    drifted, gt = generator.inject(cfg, infer, profile)
    say("affected channels", f"{len(gt.affected_features)}: "
                             f"{', '.join(gt.affected_features)}")
    say("declared onset", f"index {gt.drift_start_index:,} "
                          f"(end={gt.drift_end_index})")
    say("drifted rows", f"{gt.n_drifted_rows:,} / {gt.n_rows:,}")
    # `realised_magnitude` is per-channel: clipping to the physical range bites
    # each channel differently, so a single number would hide the spread.
    realised = dict(gt.realised_magnitude)
    realised_mean = float(np.mean(list(realised.values()))) if realised \
        else float("nan")
    say("realised magnitude", f"mean {realised_mean:.4f} sigma of "
                              f"{gt.drift_magnitude} requested "
                              f"(attenuation "
                              f"{gt.schedule_summary['attenuation_ratio']:.4f})")
    for c, v in realised.items():
        print(f"      {c:.<24} {v:.4f} sigma")
    for n in gt.notes:
        print(f"      note: {n}")

    # Independent recomputation of the realised shift: trust the measurement,
    # not the generator's own bookkeeping.
    raw_shift = _measured_raw_shift(infer.frame, drifted.frame, gt, profile)
    say("recomputed raw shift", f"{raw_shift:.4f} sigma "
                                f"(delta vs reported "
                                f"{abs(raw_shift - realised_mean):.2e})")

    # The sidecar path is per-config; give each scenario its own file so three
    # runs do not overwrite one another. An output location, not a parameter.
    cfg_out = json.loads(json.dumps(cfg))
    cfg_out["ground_truth"]["sidecar_path"] = \
        f"{cfg['output']['synthetic_dir']}/ground_truth_{name}.json"
    sidecar = generator.write_sidecar(gt, cfg_out, root=ROOT)
    say("ground-truth sidecar", sidecar.relative_to(ROOT) if sidecar else "off")

    # -- preprocessing --------------------------------------------------------
    prepared = pp.transform(cfg, drifted, stats)
    say("preprocessed", f"{len(prepared.frame):,} rows, "
                        f"adaptation={prepared.adaptation}")

    # -- windowing ------------------------------------------------------------
    plan = stream.plan(cfg, n_rows=len(prepared.frame))
    windows = list(stream.iter_windows(prepared, cfg))
    say("window grid", f"size={plan.window_size} step={plan.step_size} "
                       f"-> {len(windows):,} windows emitted")

    # -- features -------------------------------------------------------------
    features = pp.extract_features(cfg, prepared, windows)
    say("feature matrix", f"{features.X.shape[0]:,} x {features.X.shape[1]} "
                          f"({len(stats.continuous)}x6 + "
                          f"{len(stats.discrete)}x2)")

    # -- did the drift survive to the feature matrix? -------------------------
    win_shift = _measured_window_shift(features, gt)
    say("window-level shift", f"{win_shift:.4f} sigma, measured on the "
                              f"per-window means of the affected channels")

    # -- logs -------------------------------------------------------------
    results = ROOT / cfg["output"]["results_dir"]
    with logmod.EventLog.create("run", config=cfg, root=ROOT,
                                suffix=f"_{name}") as log:
        log.write(event="scenario_complete", config_name=name,
                  config_fingerprint=cfgmod.fingerprint(cfg),
                  master_seed=seedmod.master_seed(cfg),
                  detail=f"{len(windows)} windows, {features.X.shape[1]} features")
    with logmod.EventLog.create("quality", config=cfg, root=ROOT,
                                suffix=f"_{name}") as log:
        logmod.log_quality(log, f"drifted_{name}", prepared.quality)
    with logmod.EventLog.create("stream", config=cfg, root=ROOT,
                                suffix=f"_{name}") as log:
        log.write_many([
            dict(window_id=w.window_id, start_index=w.start_index,
                 end_index=w.end_index, start_time=str(w.start_time),
                 end_time=str(w.end_time), n_rows=w.n_rows,
                 contiguous=w.contiguous, valid_fraction=w.valid_fraction)
            for w in windows])
    say("event logs", f"run/quality/stream _{name}.csv in "
                      f"{results.relative_to(ROOT)}")

    # -- feature CSV ----------------------------------------------------------
    fframe = pd.DataFrame(features.X, columns=list(features.names))
    fframe.insert(0, "window_id", features.window_ids)
    fframe.insert(1, "start_index", features.start_index)
    fframe.insert(2, "end_index", features.end_index)
    write_csv(fframe, results / f"features_{name}.csv",
              note="normalized window features, no label column")

    return {
        "scenario": name,
        "fingerprint": cfgmod.fingerprint(cfg),
        "mechanism": gt.mechanism,
        "requested_magnitude_sigma": gt.drift_magnitude,
        "realised_magnitude_sigma": realised_mean,
        "attenuation_ratio": gt.schedule_summary["attenuation_ratio"],
        "realised_per_channel": "|".join(f"{c}={v:.4f}"
                                         for c, v in realised.items()),
        "recomputed_raw_shift_sigma": raw_shift,
        "window_level_shift_sigma": win_shift,
        "n_affected_channels": len(gt.affected_features),
        "affected_channels": "|".join(gt.affected_features),
        "drift_start_index": gt.drift_start_index,
        "drift_end_index": gt.drift_end_index if gt.drift_end_index is not None
                           else -1,
        "n_drifted_rows": gt.n_drifted_rows,
        "n_windows": len(windows),
        "n_features": int(features.X.shape[1]),
        "outlier_rate": float(prepared.quality.outlier.mean()),
        "valid_rate": float(prepared.quality.valid.mean()),
        "_cfg": cfg,
        "_prepared": prepared,
        "_drifted": drifted,
        "_gt": gt,
    }


def _measured_raw_shift(clean: pd.DataFrame, drifted: pd.DataFrame,
                        gt, profile) -> float:
    """Mean post-onset minus pre-onset shift, in baseline sigma, recomputed."""
    vals = []
    for c in gt.affected_features:
        sigma = profile.sigma(c)
        a = clean[c].to_numpy(dtype=float)
        b = drifted[c].to_numpy(dtype=float)
        vals.append(float(np.mean(b[gt.drift_start_index:]
                                  - a[gt.drift_start_index:]) / sigma))
    return float(np.mean(vals)) if vals else float("nan")


def _measured_window_shift(features, gt) -> float:
    names = list(features.names)
    cols = [names.index(f"{c}__mean") for c in gt.affected_features
            if f"{c}__mean" in names]
    if not cols:
        return float("nan")
    onset = gt.drift_start_index
    pre = [i for i, e in enumerate(features.end_index) if e <= onset]
    post = [i for i, s in enumerate(features.start_index) if s >= onset]
    if not pre or not post:
        return float("nan")
    return float(features.X[np.ix_(post, cols)].mean()
                 - features.X[np.ix_(pre, cols)].mean())


# -----------------------------------------------------------------------------
# stage 5 -- ASSUMPTION [A6]: does the normalizer cancel the drift?
# -----------------------------------------------------------------------------

def stage_absorption(scn: dict, baseline, profile) -> pd.DataFrame:
    rule("STAGE 5  does normalization absorb the drift?  [ASSUMPTION A6]")
    cfg = scn["_cfg"]
    by_mode = {m: pp.fit(_with_adaptation(cfg, m), baseline, profile)
               for m in pp.ADAPTATION_MODES}
    measured = pp.measure_drift_absorption(
        cfg, scn["_drifted"], by_mode,
        scn["_gt"].affected_features, scn["_gt"].drift_start_index)
    rows = []
    for mode, m in measured.items():
        say(mode, f"shift={m['mean_normalized_shift_sigma']:+.4f} sigma  "
                  f"retained={m['retained_fraction_vs_frozen'] * 100:6.2f}% "
                  f"of frozen")
        rows.append({"scenario": scn["scenario"], "adaptation": mode, **m})
    print("\n  Reading: an adaptive scaler re-centres on the drifted stream and")
    print("  subtracts the injected shift back out. The retained fraction is how")
    print("  much survives. This is why `frozen_after_baseline` is the default --")
    print("  the measurement above, not a preference.")
    return pd.DataFrame(rows)


def _with_adaptation(cfg: dict, mode: str) -> dict:
    c = json.loads(json.dumps(cfg))
    c["preprocessing"]["normalization"]["adaptation"] = mode
    return c


# -----------------------------------------------------------------------------
# stage 7 -- figure
# -----------------------------------------------------------------------------

PLOT_SMOOTH = 600          # display only -- see `_display_mean`


def _display_mean(y: np.ndarray, w: int = PLOT_SMOOTH) -> np.ndarray:
    """Trailing mean, FOR PLOTTING ONLY.

    The channels sample at 1 Hz and oscillate every few seconds, so 54,000 raw
    points render as a solid band and the shift is only visible as an envelope.
    This smooths the *already-computed* series so the eye can find the onset.

    It is not a preprocessing step and touches nothing downstream:
    `preprocessing.filtering.enabled` is deliberately false, because smoothing
    inside the pipeline would attenuate exactly the abrupt change a sudden-drift
    scenario exists to create. It is trailing (backward-looking) so that even the
    picture does not show the reader something the stream had not yet delivered.
    """
    return pd.Series(y).rolling(w, min_periods=1).mean().to_numpy()


def stage_figure(scns: list[dict], clean: pp.PreprocessedStream, baseline,
                 profile, cfg: dict) -> Path:
    rule("STAGE 7  figure")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(len(scns) + 1, 1, figsize=(11, 3.1 * (len(scns) + 1)),
                             sharex=False)
    fig.suptitle("Phase 1: controlled drift injected into the HAI inference "
                 "stream\n(measured output; no model, detector or "
                 "orchestration decision exists yet)", fontsize=11)

    for ax, scn in zip(axes, scns):
        gt, prepared = scn["_gt"], scn["_prepared"]
        ch = gt.affected_features[0]
        y = prepared.frame[ch].to_numpy(dtype=float)
        ax.plot(y, lw=0.3, color="#b2182b", alpha=0.20)
        ax.plot(_display_mean(y), lw=1.6, color="#b2182b",
                label=f"{ch} drifted")
        if ch in clean.frame.columns:
            c = clean.frame[ch].to_numpy(dtype=float)
            ax.plot(c, lw=0.3, color="#2166ac", alpha=0.20)
            ax.plot(_display_mean(c), lw=1.6, color="#2166ac",
                    label=f"{ch} clean control")
        ax.axvline(gt.drift_start_index, color="k", ls="--", lw=0.9)
        ax.annotate(f"onset {gt.drift_start_index:,}",
                    (gt.drift_start_index, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8)
        if gt.drift_end_index is not None:
            ax.axvline(gt.drift_end_index, color="k", ls=":", lw=0.9)
        ax.set_title(f"{scn['scenario']}  |  requested "
                     f"{scn['requested_magnitude_sigma']} sigma, realised "
                     f"{scn['realised_magnitude_sigma']:.3f} "
                     f"(clipping attenuation "
                     f"{scn['attenuation_ratio']:.3f}), window-level "
                     f"{scn['window_level_shift_sigma']:.3f}", fontsize=9)
        ax.set_ylabel("z-score\n(frozen baseline)", fontsize=8)
        ax.legend(fontsize=7, loc="upper left")
        ax.tick_params(labelsize=7)

    # bottom panel: the A6 measurement
    ax = axes[-1]
    scn = scns[0]
    ch = scn["_gt"].affected_features[0]
    for mode, colour in zip(pp.ADAPTATION_MODES,
                            ("#b2182b", "#ef8a62", "#2166ac")):
        st = pp.fit(_with_adaptation(scn["_cfg"], mode), baseline, profile)
        y = pp.transform(_with_adaptation(scn["_cfg"], mode),
                         scn["_drifted"], st).frame[ch].to_numpy(dtype=float)
        ax.plot(_display_mean(y), lw=1.6, color=colour, label=mode)
    ax.axvline(scn["_gt"].drift_start_index, color="k", ls="--", lw=0.9)
    ax.set_title(f"[A6] the same drifted channel ({ch}, {scn['scenario']}) "
                 f"under each normalization mode -- the rolling scaler cancels "
                 f"the shift", fontsize=9)
    ax.set_xlabel(f"row index in the inference stream   "
                  f"(bold traces: trailing {PLOT_SMOOTH}-sample mean, "
                  f"for display only -- the pipeline applies no filter)",
                  fontsize=8)
    ax.set_ylabel("normalized value", fontsize=8)
    ax.legend(fontsize=7, loc="upper left")
    ax.tick_params(labelsize=7)

    fig.tight_layout(rect=(0, 0, 1, 0.965))
    out = ROOT / cfg["output"]["plots_dir"] / "phase1_demo.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150)
    plt.close(fig)
    say(f"wrote {out.relative_to(ROOT)}", f"{out.stat().st_size / 1024:.0f} KiB")
    return out


# -----------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scenarios", nargs="+", default=list(SCENARIOS),
                    choices=list(SCENARIOS))
    ap.add_argument("--max-rows", type=int, default=None,
                    help="truncate every file (smoke run); default is all rows")
    ap.add_argument("--no-figure", action="store_true")
    args = ap.parse_args(argv)

    t0 = time.perf_counter()
    rule()
    print("Phase 1 demonstration -- drift-aware edge/cloud orchestration")
    print("Every value below is measured during this run.")
    rule()

    cfg = stage_configuration()
    baseline, profile, stats, infer = stage_baseline(cfg, args.max_rows)
    clean = stage_clean(cfg, infer, stats)

    scns = [run_scenario(n, baseline, profile, stats, infer)
            for n in args.scenarios]
    absorption = stage_absorption(scns[0], baseline, profile)

    rule("STAGE 6  summary tables")
    results = ROOT / cfg["output"]["results_dir"]
    summary = pd.DataFrame([{k: v for k, v in s.items()
                             if not k.startswith("_")} for s in scns])
    write_csv(summary, results / "phase1_summary.csv",
              note="one row per scenario, all measured")
    write_csv(absorption, results / "phase1_normalization_absorption.csv",
              note="ASSUMPTION A6")

    gtframe = pd.DataFrame([{
        "scenario": s["scenario"], "mechanism": s["mechanism"],
        "drift_start_index": s["drift_start_index"],
        "drift_end_index": s["drift_end_index"],
        "n_drifted_rows": s["n_drifted_rows"],
        "affected_channels": s["affected_channels"],
        "requested_magnitude_sigma": s["requested_magnitude_sigma"],
        "realised_magnitude_sigma": s["realised_magnitude_sigma"],
    } for s in scns])
    write_csv(gtframe, results / "phase1_ground_truth.csv",
              note="EVALUATION ONLY -- never an input to a detector")

    if not args.no_figure:
        stage_figure(scns, clean, baseline, profile, cfg)

    rule("done")
    print(summary[["scenario", "realised_magnitude_sigma",
                   "window_level_shift_sigma", "n_windows",
                   "n_features"]].to_string(index=False))
    print(f"\nElapsed {time.perf_counter() - t0:.1f}s. "
          f"No model, detector, reliability score or orchestration decision was "
          f"computed -- none exists in Phase 1.")
    rule()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
