# PROJECT_STATE — generated, do not edit by hand

Generated `2026-08-28 12:37:05Z` by `state_report.py` from 81 code, config
and provenance files. Surface fingerprint **`a686902805ba5a83`**.

Regenerate after any change:

```bash
cd drift_aware_edge_cloud && ../.venv/Scripts/python.exe state_report.py --write
```

Everything below was read off the tree just now. Nothing here is a
remembered value, and nothing here is an experimental result — this
script runs no test and no model. For pass counts, run the commands in
[Verification](#verification) yourself.

## Where the rest of the context lives

| File | Holds |
|---|---|
| `AGENT_CONTEXT.md` | The contract: what you must not do, how to work, how to verify. **Read first.** |
| `DECISIONS.md` | Append-only log of decisions, errors and fixes, with the reasoning. Explains *why* the tree looks like this. |
| `README.md` | The research design: question, hypothesis, causal chain, dataset diagnostics, scope. |
| `data/raw/PROVENANCE.json` | Machine-readable dataset provenance, checksums, every measured diagnostic and finding. |
| `config/*.yaml` | Every parameter and assumption. Nothing is hard-coded in Python. |
| `tests/test_integrity.py` | The prohibitions as executable tests — the only form of context that cannot be forgotten. |

## Implementation status

Read from the `Status :` header of 37 source modules: **18 IMPLEMENTED**, **19 not**.

- **Phase 1 / Step 3** — complete
  - [x] `src/data/generator.py` — IMPLEMENTED
  - [x] `src/data/loader.py` — IMPLEMENTED
  - [x] `src/data/stream.py` — IMPLEMENTED
- **Phase 1 / Step 4** — complete
  - [x] `src/data/preprocessing.py` — IMPLEMENTED
- **Phase 1 / Step 5** — complete
  - [x] `src/utils/config.py` — IMPLEMENTED
  - [x] `src/utils/logger.py` — IMPLEMENTED
  - [x] `src/utils/seed.py` — IMPLEMENTED
- **Phase 2** — complete
  - [x] `src/models/base.py` — IMPLEMENTED
  - [x] `src/models/cloud_model.py` — IMPLEMENTED
  - [x] `src/models/edge_model.py` — IMPLEMENTED
  - [x] `src/models/trainer.py` — IMPLEMENTED
- **Phase 3** — complete
  - [x] `src/drift/adwin_detector.py` — IMPLEMENTED
  - [x] `src/drift/persistence.py` — IMPLEMENTED
  - [x] `src/drift/severity.py` — IMPLEMENTED
- **Phase 4** — complete
  - [x] `src/reliability/base.py` — IMPLEMENTED
  - [x] `src/reliability/estimator.py` — IMPLEMENTED
- **Phase 5** — 2/5
  - [x] `src/decision/base.py` — IMPLEMENTED
  - [x] `src/decision/engine.py` — IMPLEMENTED
  - [ ] `src/network/lri.py` — NOT IMPLEMENTED
  - [ ] `src/network/network_model.py` — NOT IMPLEMENTED
  - [ ] `src/resources/edge_resources.py` — NOT IMPLEMENTED
- **Phase 6** — not started
  - [ ] `src/resources/controller.py` — NOT IMPLEMENTED
  - [ ] `src/resources/wds.py` — NOT IMPLEMENTED
  - [ ] `src/simulation/cloud.py` — NOT IMPLEMENTED
  - [ ] `src/simulation/edge.py` — NOT IMPLEMENTED
  - [ ] `src/simulation/environment.py` — NOT IMPLEMENTED
  - [ ] `src/simulation/hybrid.py` — NOT IMPLEMENTED
- **Phase 7** — not started
  - [ ] `src/adaptation/deployment.py` — NOT IMPLEMENTED
  - [ ] `src/adaptation/retrainer.py` — NOT IMPLEMENTED
  - [ ] `src/adaptation/validator.py` — NOT IMPLEMENTED
- **Phase 8** — not started
  - [ ] `adaptation/baselines/cloud_only.py` — NOT IMPLEMENTED
  - [ ] `adaptation/baselines/edge_only.py` — NOT IMPLEMENTED
  - [ ] `adaptation/baselines/hecif_baseline.py` — NOT IMPLEMENTED
- **Phase 10** — not started
  - [ ] `src/metrics/decision.py` — NOT IMPLEMENTED
  - [ ] `src/metrics/drift.py` — NOT IMPLEMENTED
  - [ ] `src/metrics/prediction.py` — NOT IMPLEMENTED
  - [ ] `src/metrics/system.py` — NOT IMPLEMENTED

## Blocking decisions — reserved to the user

Configuration keys deliberately left unset rather than guessed:

| Key | Value |
|---|---|
| `dataset.label_column` | `null` |
| `dataset.label_file` | `null` |

### TASK_UNRESOLVED — `dataset.task`  ·  **RESOLVED**

- resolved: **2026-08-27** by **user**
- value: `dataset.task` was `unresolved`, now `labels_from_hai_labels`
- how: The HAI 23.05 release ships official attack labels in separate sidecar files. Once label-test1.txt was present and its alignment to hai-test1.txt was proven ELEMENTWISE (audit_alignment.py, 6/6), the `labels_from_hai_labels` option stopped requiring an absent file. See `finding_official_labels_shipped_separately`.

HAI's process-value files ship no label column (see `finding_no_label_column`). The primary metric in the specification is post-drift Macro-F1, which requires a classification target. Until `dataset.task` was resolved, no target existed.

Options on record, with their consequences:

- **`forecasting_regression`** — target: next-step value of one or more continuous process channels. Primary metric changes from post-drift Macro-F1 to a post-drift regression error (e.g. RMSE/MAE). Deviates from the specification's stated primary metric.
  - _status_: NOT_TAKEN
  - _leakage_note_: Must exclude the target's own future values and, for actuator targets, its command/feedback sibling. See `finding_command_feedback_pairs`.
- **`state_classification`** — target: a discrete process/actuator state derived from HAI's own columns. Preserves post-drift Macro-F1 as the primary metric.
  - _risk_: This is the failure mode that disqualified SWaT: PLC-controlled actuator states can be trivially recoverable from sensors, collapsing accuracy headroom and forcing R_edge ~= R_cloud, which kills the reliability axis of the WDS. MUST be probed empirically before adoption, exactly as SWaT was probed.
  - _status_: NOT_TAKEN -- no headroom probe was ever run, so this option was never eligible.
  - _leakage_note_: Requires exclude_target_sibling: true.
- **`labels_from_hai_labels`** — target: official HAI attack labels. Preserves post-drift Macro-F1 and gives a genuine anomaly-detection task.
  - _requires_: An additional label file from the HAI distribution (the HAI release ships attack labels separately from the process-value files). SUPPLIED 2026-08-27 as label-test1.txt and label-test2.txt.
  - _status_: TAKEN -- the only option that preserves the specification's primary metric without inventing a target.

_No label was fabricated at any point. While the label file was absent, dataset.label_column was null and dataset.task was 'unresolved' rather than guessed. The label now in use is READ FROM A FILE THE DISTRIBUTION SHIPPED, is not a column of the process-value files, and is quarantined as evaluation-only._

### Open findings

Each was measured and deliberately **not** repaired, because repairing it would be a scientific choice.

- **`finding_label_test2_minute_resolution`** (OPEN_DECISION) — label-test2.txt cannot be joined to hai-test2.txt on timestamp. Its timestamp column is MINUTE-resolution, textually formatted '2022-08-17 0:00' rather than '2022-08-17 00:00:01', so it is not a key against a 1 Hz stream.
  - action taken: NONE -- not repaired, not assumed. hai-test2.txt and label-test2.txt are declared under dataset.reserved_files, which no loader code path reads, so test2 is unreachable structurally rather than behind a flag that could be flipped without reading this finding. Reported per the standing rule 'do not invent scientific assumptions silently'.
  - awaiting: user decision
- **`finding_training_labels_absent`** (OPEN_DECISION) — HAI 23.05 ships label sidecars for the TEST streams only. There is no label file for train1 or train2, so the causally valid baseline (train1) carries NO labels.
  - action taken: NONE beyond recording it. dataset.training_labels_available: false and ASSUMPTION [A18]: train1 is treated as UNLABELLED, not as all-normal.
  - awaiting: Phase 2 model-design decision by the user
- **`finding_near_collinear_pairs`** (OPEN_DECISION) — Four continuous channel pairs on the baseline exceed |Pearson r| > 0.999, and none of them appear in dataset.features.command_feedback_pairs, which covers only the six P1_ actuators.
  - action taken: NONE -- behaviour deliberately unchanged. Reported rather than silently fixed, per the standing rule 'do not invent scientific assumptions silently'.
  - awaiting: user decision
- **`finding_continuous_command_channels`** (OPEN_DECISION) — drift.affected_features.actuator_policy: exclude filters DISCRETE channels only, so continuous control-command channels are still eligible for drift injection. Two of the five channels selected for the single-drift scenarios are command/demand channels rather than sensors.
  - action taken: NONE -- behaviour deliberately unchanged. actuator_policy currently means 'exclude discrete states', not 'exclude actuators'.
  - awaiting: user decision
- **`finding_degenerate_outlier_bounds`** (OPEN_DECISION) — The configured 3xIQR outlier rule degenerates on piecewise-constant ICS channels. When q1 == q3 the band has ZERO width, so every value not exactly equal to that constant is flagged. One such channel saturates the row-level outlier flag at 100%, making it carry no information.
  - action taken: NONE -- behaviour deliberately unchanged. preprocessing._flag_outliers now DETECTS zero-width bands, names the channels in QualityReport.degenerate_bound_columns, and reports both the saturated and the excluding-degenerate flag rates. Reporting only; not a repair.
  - awaiting: user decision

### Findings already settled or accepted as measured fact

| Finding | Severity | Resolution |
|---|---|---|
| `finding_no_label_column` | RESOLVED_BY_OFFICIAL_LABEL_SIDECARS | dataset.task: labels_from_hai_labels, dataset.label_column: label, dataset.label_file: the official sidecar f… |
| `finding_official_labels_shipped_separately` | RESOLUTION_RECORD | dataset.task resolved to labels_from_hai_labels by the user |
| `finding_temporal_acausality` | BLOCKER_RESOLVED_BY_CONFIG | dataset.baseline_source: train1_only (ASSUMPTION [A17]), dataset.allow_acausal_baseline: false, dataset.conca… |
| `finding_command_feedback_pairs` | LEAKAGE_TRAP | Recorded in config as dataset.features.command_feedback_pairs with dataset.features.exclude_target_sibling: t… |
| `finding_discrete_vs_continuous` | CORRECTNESS | drift.affected_features.actuator_policy: exclude by default in all Phase 1 scenarios; dataset.features.type_d… |
| `finding_realised_magnitude_attenuation` | MEASURED_FACT | generator.py measures realised magnitude per channel and records it in GroundTruth.realised_magnitude and sch… |
| `finding_pre_existing_regime_shift` | MEASURED_FACT | NONE -- this is a property of the real HAI recordings, not a defect |

## Leakage and integrity guards — live configured values

| Key | Value |
|---|---|
| `dataset.concatenate_files` | `False` |
| `dataset.allow_acausal_baseline` | `False` |
| `dataset.baseline_source` | `train1_only` |
| `streaming.shuffle` | `False` |
| `preprocessing.normalization.adaptation` | `frozen_after_baseline` |
| `preprocessing.normalization.forbid_global_fit` | `True` |
| `drift.injection_target` | `inference_stream_only` |
| `drift.modify_labels` | `False` |

Ground-truth consumers, from `ground_truth`:

- **allowed_consumers**: ['metrics', 'plots', 'statistical_evaluation']
- **forbidden_consumers**: ['models', 'drift_detectors', 'severity', 'persistence', 'reliability', 'controller', 'wds', 'lri', 'edge', 'cloud', 'hybrid', 'adaptation']

Configurations present: `default`, `gradual_drift`, `stress_test`, `sudden_drift`

- `default` — resolves and validates, fingerprint `3927fda9e98d`
- `gradual_drift` — resolves and validates, fingerprint `40669e78dd74`
- `stress_test` — resolves and validates, fingerprint `0ead60fdfa9d`
- `sudden_drift` — resolves and validates, fingerprint `95ae953d155a`

## Raw data

Roles come from `config/default.yaml` — `dataset.files` for the
process-value streams, `dataset.labels` for the official label
sidecars, `dataset.reserved_files` for what the loader must not
reach. The file names are never written in Python.

| File | Recorded role | Size on disk | Matches record | Modified flag |
|---|---|---|---|---|
| `wustl_iiot_2021.csv` | active_dataset_wustl | 409,800,698 | yes | False |
| `hai-train1.txt` | baseline_train | 162,418,984 | yes | False |
| `hai-train2.txt` | baseline_validation | 169,121,615 | yes | False |
| `hai-test1.txt` | inference_stream | 31,255,559 | yes | False |
| `label-test1.txt` | inference_labels | 1,242,017 | yes | False |
| `hai-test2.txt` | reserved_second_inference_stream | 132,946,575 | yes | False |
| `label-test2.txt` | reserved_inference_labels | 4,500,018 | yes | False |

_SHA-256 not checked in this run; pass `--hash` (or run the `slow` pytest marker) to verify._

## Verification

This script did **not** run any of these. Run them; quote the counts they print.

```bash
cd drift_aware_edge_cloud
export PYTHONIOENCODING=utf-8
../.venv/Scripts/python.exe verify_step2.py
../.venv/Scripts/python.exe verify_step3.py
../.venv/Scripts/python.exe verify_step4.py
../.venv/Scripts/python.exe verify_step5.py
../.venv/Scripts/python.exe -m pytest -q
```

Handover artifacts have their own harness, run at session end rather than per edit (it checks this file for staleness, so it fails until it is regenerated):

```bash
../.venv/Scripts/python.exe verify_handover.py
```

Test suite on disk, parsed with `ast`:

| Module | Test functions | Marked slow |
|---|---|---|
| `tests/test_config.py` | 18 | 0 |
| `tests/test_decision.py` | 28 | 0 |
| `tests/test_drift.py` | 20 | 0 |
| `tests/test_generator.py` | 31 | 0 |
| `tests/test_integrity.py` | 27 | 1 |
| `tests/test_loader.py` | 33 | 0 |
| `tests/test_logger.py` | 16 | 0 |
| `tests/test_models.py` | 14 | 0 |
| `tests/test_preprocessing.py` | 40 | 0 |
| `tests/test_reliability.py` | 26 | 0 |
| `tests/test_seed.py` | 17 | 0 |
| `tests/test_stream.py` | 28 | 0 |
| **total declared** | **298** | **1** |

Declared functions are not the collected count: `pytest` expands parametrised cases, so the number it reports is higher. The collected/passed count is only knowable by running it.

## Generated output present on disk

Regenerable, not authoritative. `data/synthetic/` must contain only ground-truth sidecars — it is an output directory that nothing reads back as input.

| Directory | Entries |
|---|---|
| `results/` | 15 — `features_gradual_drift.csv`, `features_stress_test.csv`, `features_sudden_drift.csv`, `phase1_ground_truth.csv`, `phase1_normalization_absorption.csv`, `phase1_summary.csv`, … (+9 more) |
| `plots/` | 1 — `phase1_demo.png` |
| `data/synthetic/` | 4 — `ground_truth.json`, `ground_truth_gradual_drift.json`, `ground_truth_stress_test.json`, `ground_truth_sudden_drift.json` |
| `data/processed/` | 0 |

## Environment

Interpreter running this report: **3.14.2** (`D:\tactics\.venv\Scripts\python.exe`)

| Package | Installed |
|---|---|
| numpy | 2.4.4 |
| pandas | 3.0.5 |
| scipy | 1.18.1 |
| scikit-learn | 1.9.0 |
| PyYAML | 6.0.3 |
| xgboost | 3.4.1 |
| river | 0.26.1 |
| simpy | 4.1.2 |
| matplotlib | 3.11.1 |
| pytest | 9.1.1 |

---

_End of generated report. Fingerprint `a686902805ba5a83`; `state_report.py --check` fails if the tree has moved since._
