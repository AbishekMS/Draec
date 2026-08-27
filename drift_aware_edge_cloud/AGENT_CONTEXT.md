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
2. **No fabricated labels.** HAI ships no label column. `dataset.task` is
   `unresolved` and `loader.resolve_target()` raises rather than guessing. Do not
   invent a target to unblock yourself. See §6.
3. **No fabricated results.** Do not state that an experiment succeeded before
   running it. Do not hard-code a metric value anywhere. Report failures and
   unfavourable results as readily as successes.
4. **No future information, ever.** Fitting on all data, or on train + test, is
   forbidden. This includes *file-level* leakage: HAI's chronology is
   train1 < test1 < train2, so the second training file lies in the inference
   stream's future and is excluded from the baseline
   (`dataset.baseline_source: train1_only`).
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

## 6. What is blocked, and why you must not unblock it yourself

**`dataset.task` is `unresolved` because HAI ships no label column.** This blocks
all of Phase 2 (no model can be trained without a target) and window label
aggregation. Three options are recorded in `PROVENANCE.json` →
`ACTIVE_DATASET_DECISION.blocking_open_question`, each with its measured
consequences: `forecasting_regression`, `state_classification`,
`labels_from_hai_labels`.

Choosing between them changes the study's primary metric. **It is a scientific
decision reserved to the user.** Do not pick one to make progress.

Three further findings are open by design and also await user decisions. They are
listed with their measurements in `PROJECT_STATE.md`.

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
