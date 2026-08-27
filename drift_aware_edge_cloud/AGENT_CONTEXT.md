# AGENT_CONTEXT — read this before writing any code

You are working on a research-grade simulation whose output is intended to be
**publishable**. That single fact generates every rule below. An unfounded
shortcut here is worse than an incomplete result, because an incomplete result
is honest and a shortcut silently invalidates everything built on top of it.

This file exists because development spans multiple sessions and, deliberately,
multiple models and APIs. Nothing in your own memory carries across that
boundary — this repository is the only context that survives. Treat it as
authoritative and treat your own recollection as absent.

---

## 1. Sixty-second bootstrap

Read in this order, then run the commands. Do not start writing code before the
commands have actually been run and their output seen.

| # | Read | For |
|---|---|---|
| 1 | this file | the rules and the working protocol |
| 2 | `PROJECT_STATE.md` | what is implemented **right now** (generated, not written by hand) |
| 3 | `DECISIONS.md` | *why* the tree looks like this — decisions, errors, fixes |
| 4 | `README.md` | the research design: question, hypothesis, causal chain, dataset diagnostics |
| 5 | `data/raw/PROVENANCE.json` | checksums and every measured finding, machine-readable |
| 6 | `config/default.yaml` | every parameter and every `ASSUMPTION [An]` |

```bash
cd drift_aware_edge_cloud
export PYTHONIOENCODING=utf-8
../.venv/Scripts/python.exe state_report.py --check
../.venv/Scripts/python.exe verify_step2.py
../.venv/Scripts/python.exe verify_step3.py
../.venv/Scripts/python.exe verify_step4.py
../.venv/Scripts/python.exe verify_step5.py
../.venv/Scripts/python.exe -m pytest -q
../.venv/Scripts/python.exe verify_handover.py
```

If `state_report.py --check` reports `STALE`, `PROJECT_STATE.md` describes a tree
that has since moved — regenerate it with `--write` and read it again before
trusting anything it says.

**Expected at the last known-good point:** `verify_step2` 10/10, `verify_step3`
13/13, `verify_step4` 15/15, `verify_step5` 12/12, `pytest` 230 passed.

Those five numbers are the definition of "verified" for this project. They are
recorded here as the value to *reproduce*, not as evidence. If what you get
differs, the tree has moved — find out why before building on it. Never quote
them without having run the commands.

## 2. What the project is

**Research question.** Does incorporating online prediction drift and reliability
into Edge–Cloud orchestration improve post-drift prediction performance while
maintaining acceptable latency and resource consumption compared with existing
approaches?

**The decision is computed, never asserted:**

```
WDS(a) = w1·LRI_a + w2·E_a + w3·B_a + w4·(1 − R_a) + w5·C_a   for a ∈ {Edge, Cloud, Hybrid}
a*     = argmin_a WDS(a)
```

There is deliberately **no** `if drift then Cloud` rule anywhere, and there must
never be one.

**Data principle.** Real HAI is the foundation; "synthetic" means controlled
drift injected *into* real HAI observations. Never a replacement dataset.

## 3. Hard prohibitions

These are correctness requirements, not style preferences. Most are enforced by
`tests/test_integrity.py`, which will fail you — but the tests are a safety net,
not the specification. The specification is this list.

1. **No fabricated data.** HAI is the real source. Never generate a replacement
   dataset or invent "normal" records. The raw files stay byte-identical; do not
   open anything under `data/raw/` for writing.
2. **No fabricated labels.** The HAI process-value files carry no label column.
   HAI 23.05 ships official attack labels in a *separate sidecar file per test
   stream*, and `dataset.task` is now `labels_from_hai_labels` — resolved on
   2026-08-27 **by the user**, only after the sidecar's alignment was proven
   elementwise. `loader.resolve_target()` still raises for any config that has
   not named a label source, and the labels are quarantined
   (`dataset.label_usage: evaluation_only`). **The causal baseline is
   unlabelled** — HAI ships no labels for the training streams — so do not assign
   `label = 0` to training rows, and do not carve a labelled split out of the
   inference stream. See §6.
3. **No fabricated results.** Do not state that an experiment succeeded before
   running it. Do not hard-code a metric value anywhere. Report failures and
   unfavourable results as readily as successes.
4. **No future information, ever.** Fitting on all data, or on train + test, is
   forbidden. This includes *file-level* leakage: HAI's measured chronology is
   **train1 < test1 < train2 < test2** — not the order the filenames imply — so
   the second training file lies in the inference stream's future and is excluded
   from the baseline (`dataset.baseline_source: train1_only`).
5. **Ground truth is evaluation-only.** Scenario, drift start/end index, affected
   features, magnitude and seed must never reach a model, drift detector,
   severity or persistence module, reliability estimator, controller, WDS, LRI,
   or any Edge/Cloud/Hybrid component. `config -> ground_truth.allowed_consumers`
   is the whole permitted set.
6. **No forced orchestration decision.** A literal `return "cloud"` beside a
   drift check would decide the study's finding in advance.
7. **No silent scientific assumptions.** Record each as `ASSUMPTION [An]` in
   `config/default.yaml` and surface open ones for a user decision.
8. **No silent repairs of findings.** If you measure something questionable,
   record it in `PROVENANCE.json` with `severity`, `claim`, `measured`,
   `consequence`, `action_taken` and `awaiting`, and leave behaviour unchanged.
   Several findings are open on purpose; see `PROJECT_STATE.md`.
9. **File roles live in configuration, not in Python.** No HAI filename may
   appear in any `.py` file under `src/`, `tests/`, `adaptation/`, or in
   `main.py` — **including inside docstrings and comments**, because
   `verify_step2.py` check 4 is a plain-text scan. Resolve names from
   `dataset.files.*.path`.
10. **Do not redesign the architecture** or skip components. Later-phase modules
    already exist as documented placeholders carrying their contracts; fill them
    in, do not move them.
11. **Do not tune the proposed method on test results**, and do not select WDS
    weights after inspecting only its own outcome. Baselines, ablations and the
    proposed method share one identical evaluation protocol.
12. **Do not add continuous offsets to discrete actuator/state channels**
    blindly.
13. **Do not recreate or modify the virtual environment** at `D:\tactics\.venv`.

## 4. Working protocol

One step at a time:

1. Implement exactly one step.
2. **Verify by running real code against the real data** — not by reasoning that
   it should work.
3. Report the measured numbers, including anything that failed.
4. **Stop and wait for explicit confirmation** before the next step.

Each Phase 1 step left a re-runnable `verify_stepN.py` at the project root (not
under `src/`, so it cannot be mistaken for a component). It prints measured
values and a PASS/FAIL tally, and exits non-zero on failure. Continue that
pattern. **Earlier harnesses must stay green** — when a later step legitimately
invalidates an assertion, tighten it, never delete it.

When a measurement contradicts something a document claims, the document is
wrong. Fix the document.

## 5. Environment, and the things that will trip you up

| Fact | Consequence |
|---|---|
| Python **3.14.2** at `D:\tactics\.venv` | invoke as `../.venv/Scripts/python.exe` from `drift_aware_edge_cloud/` |
| Windows console is cp1252 | set `PYTHONIOENCODING=utf-8` or unicode output raises |
| Not a git repository | there is no history to consult; `DECISIONS.md` is the history |
| Raw data is ~360 MB | the SHA-256 test is behind `@pytest.mark.slow` |
| `tests/test_integrity.py` scans source with `tokenize` | prose in a docstring naming a forbidden construct is fine there |
| `verify_step2.py` scans with a **plain regex** | prose in a docstring is **not** fine there — see prohibition 9 |
| `data/synthetic/` is output-only | nothing may read it back as input; only ground-truth sidecars belong there |

## 6. What is settled, what is still blocked, and why you must not unblock it

**The target is RESOLVED — and it was the user's decision, not a model's.**
`dataset.task: labels_from_hai_labels`, `label_column: label`,
`positive_class: 1`, reading the official sidecar named by `dataset.label_file`.
Of the three options recorded in `PROVENANCE.json` →
`ACTIVE_DATASET_DECISION.blocking_open_question`, `labels_from_hai_labels` was
taken; `forecasting_regression` and `state_classification` were not. Each still
carries its measured consequences there, because a superseded option is evidence
about why the taken one was taken.

It was adopted only after `audit_alignment.py` proved, 6/6, that row *i* of
`label-test1` describes row *i* of `hai-test1` — equal timestamps on all 54,000
rows, not merely equal counts and endpoints. **Post-drift Macro-F1 remains the
primary metric**; a majority-class predictor already scores 94.48% accuracy at a
5.52% positive rate, which is why accuracy is not the metric. Choosing a target
changes the study's primary metric and so remains **a scientific decision
reserved to the user** — if you ever find yourself picking one to make progress,
stop.

**Two quarantines, never merged.** Official HAI labels say what the plant actually
did; drift ground truth says what we injected. `dataset.label_usage:
evaluation_only` with 13 forbidden consumers, and `ground_truth.forbidden_consumers`
separately. Neither may reach a model, detector, reliability estimator or the
controller.

**Still open, awaiting user decisions — do not repair these:**

| open finding | measured | what it blocks |
|---|---|---|
| `finding_training_labels_absent` | HAI 23.05 ships labels for **test streams only**, so the causal baseline `train1` is unlabelled — ASSUMPTION [A18] treats it as *unlabelled*, not as all-normal | whether the Edge/Cloud models are one-class/reconstruction detectors or supervised under an explicit attack-free assumption — a **Phase 2 model-design** decision |
| `finding_label_test2_minute_resolution` | `label-test2` is minute-resolution: 3,841 distinct stamps for 230,400 rows, 226,560 rows disagreeing elementwise — alignment plausible but **unverifiable** | `test2` as a second inference stream; it is declared under `dataset.reserved_files`, a section no loader path reads |

Three further findings (`finding_near_collinear_pairs`,
`finding_continuous_command_channels`, `finding_degenerate_outlier_bounds`) are
open by design and also await user decisions. All are listed with their
measurements in `PROJECT_STATE.md`.

**Phase 2 is unblocked** by the target resolution. It is still not started; see
prohibition 10 and §4.

## 7. Before you write code, check

- Am I about to invent a value the config should own? → put it in `config/`.
- Am I about to fit a statistic on anything but the causal baseline? → stop.
- Am I about to read `GroundTruth` from a module that is not an evaluation
  consumer? → stop.
- Am I about to write a metric, latency or decision literal? → stop.
- Am I about to "fix" a finding recorded as `awaiting: user decision`? → stop and
  ask.
- Am I about to claim a result I have not run? → run it.
- Does a `verify_step*.py` still pass after my change? → run all of them.

## 8. Handing over — do this at the end of every working session

```bash
cd drift_aware_edge_cloud
export PYTHONIOENCODING=utf-8
../.venv/Scripts/python.exe -m pytest -q          # and the four verify harnesses
../.venv/Scripts/python.exe state_report.py --write
../.venv/Scripts/python.exe verify_handover.py
```

1. Append an entry to `DECISIONS.md` for every decision, error and fix — with the
   reasoning, not just the outcome. It is **append-only**; do not rewrite history.
2. Regenerate `PROJECT_STATE.md`. `state_report.py --check` exits non-zero when it
   no longer matches the tree, so a stale one is detectable rather than merely
   misleading.
3. Run `verify_handover.py`. Twelve checks, same shape as the step harnesses. It
   verifies the generated state is current, that **this file and `README.md` quote
   identical verification counts**, that every entry in the log carries reasoning,
   and that no finding recorded as `awaiting: user decision` is missing from the
   log. It is deliberately not part of `pytest` — see `DECISIONS.md → D-013`.
4. Update `README.md` only where the research design actually changed, and
   `PROVENANCE.json` for anything newly measured.

If you change a verification count, change it in **both** this file and
`README.md`, or check 6 of `verify_handover.py` will fail — which is the point.

Status counts belong in `PROJECT_STATE.md`, which is generated. Reasoning belongs
in `DECISIONS.md`, which is permanent. Neither belongs in a model's memory.
