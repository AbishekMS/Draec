# DECISIONS — append-only log

Every decision, error and fix, with the reasoning that produced it. This is the
project's memory across sessions **and across models**: `PROJECT_STATE.md` records
*what* the tree looks like and is regenerated on demand, `README.md` records the
research design, and this file records *why* — the part that cannot be recovered
by reading code.

**Rules for this file**

- **Append only.** Never edit or delete an entry. If a decision is reversed, add a
  new entry that supersedes it and say so in both directions.
- One entry per decision, error or measured finding. `D-` for decisions, `E-` for
  errors and their fixes. Numbers never reused.
- Record the reasoning and the cost, not just the outcome. "Chose X" is useless in
  six weeks; "chose X because Y measured Z, at the cost of W" is not.
- No measurement in this file that was not actually run.

Entries below `D-001`/`E-001` were backfilled on **2026-08-26** at the end of
Phase 1, from the work as recorded in `PROVENANCE.json`, the config, the verify
harnesses and the session transcripts. Dates are day-accurate.

---

## D-001 · 2026-08-25 · decision · SWaT abandoned, HAI adopted as the real dataset

**Context.** SWaT (Secure Water Treatment) was the original primary dataset. The
supplied `attack.csv` contains **54,621 `Attack` rows and 0 `Normal` rows** —
a single-class target. Macro-F1, the specification's primary metric, is degenerate
with one class, and "learnable-then-changed behaviour" needs a pre-drift normal
regime that does not exist in that file.

**Decision.** HAI 22.04 (HIL-based Augmented ICS Security Dataset, National
Security Research Institute, Republic of Korea) becomes the real industrial
foundation. Strategy: **HAI + controlled synthetic drift injected into real HAI
observations.**

**Why not rescue SWaT.** Both routes were investigated and rejected on measured
evidence rather than preference:

- *Predict an actuator state instead of the attack label* — rejected. SWaT
  actuators are PLC-controlled by deterministic sensor thresholds, so the target is
  trivially recoverable: a chronological-split probe reached **Macro-F1 1.0000**.
  No accuracy headroom means `R_edge ≈ R_cloud`, which removes the reliability axis
  the WDS depends on, and post-drift Macro-F1 could show neither degradation nor
  recovery.
- *Concatenate SWaT's separate Normal file onto this subset* — rejected. The join
  introduces one large artificial regime change, making ADWIN detections
  unattributable to the injected drift.

**Consequence.** `attack.csv` remains in `data/raw/` referenced by no
configuration. No substitute or fabricated data was created at any point. The
actuator-triviality probe is now the **precedent any HAI `state_classification`
target must survive** before adoption.

**Where it lives.** `PROVENANCE.json → ACTIVE_DATASET_DECISION`,
`superseded_swat_record`; README §3.

---

## D-002 · 2026-08-25 · decision · File roles live in configuration, never in Python

**Decision.** Which HAI file is the baseline, which is the inference stream and
which is validation is declared in `config/*.yaml → dataset.files.*`, with a
`role` per entry. No HAI filename appears in any `.py` under `src/`, `tests/`,
`adaptation/`, or in `main.py`.

**Why.** The loader must be a loader, not an HAI adapter — the study should be
re-runnable on another ICS dataset by editing config. Hard-coding a filename also
hides a role assignment inside code where no reviewer looks for it.

**Enforcement.** `verify_step2.py` check 4 scans those paths for HAI filenames and
fails. Note it is a **plain-text regex**, so a filename in prose trips it too; see
`E-002`.

---

## D-003 · 2026-08-25 · decision · Baseline is train1 only — train2 is acausal

**Context.** Measured chronology of the three supplied files is
**train1 < test1 < train2**: test1 ends `2022-08-13 07:00:00`, train2 begins
`2022-08-13 07:00:01`, one second later.

**Decision.** `dataset.baseline_source: train1_only` (ASSUMPTION **[A17]**),
`dataset.allow_acausal_baseline: false`, `split.mode: separate_file`.

**Why.** Fitting any baseline statistic on train1 + train2 would fit on data
recorded *after* the inference stream. That is temporal leakage regardless of
whether labels exist.

**Cost, stated plainly.** The baseline is 280,800 rows (78 h) instead of 572,400
(159 h). A smaller causal baseline is scientifically valid; a larger acausal one
is not.

**Enforcement.** Two independent fit entry points guard the role —
`loader.profile_baseline()` and `preprocessing.fit()` both raise
`CausalityError`. A single unguarded one would have been enough to leak.

---

## D-004 · 2026-08-25 · decision · The task stays unresolved because HAI ships no label

**Context.** Three independent probes agree that none of the three files contains
an attack/anomaly indicator: all 87 column names are identical across
train1/train2/test1 (so the test file has no extra column); a name scan for
label-like identifiers matched nothing; and a structural probe for *"constant in
both training files but varying in the test file"* — the signature an appended
label would have — returned nothing.

**Decision.** No label was invented. `dataset.label_column: null`,
`dataset.task: unresolved`, and `loader.resolve_target()` raises
`UnresolvedTaskError` rather than guessing.

**Why.** The standing rule is that absent labels are not fabricated. Guessing a
target would silently choose the study's primary metric.

**What it actually blocks, measured rather than assumed.** Steps 3 and 4 both
shipped without it. Everything that does not need a target — loading, validation,
causality enforcement, baseline profiling, drift injection, windowed streaming,
causal preprocessing, windowed feature extraction (a 5,396 × 364 matrix) — is
implemented and verified against the real files. Blocked: `stream.aggregate_label`
and all of Phase 2.

**Open.** Three options with consequences in `PROVENANCE.json →
ACTIVE_DATASET_DECISION.blocking_open_question`. **Reserved to the user.**

---

## D-005 · 2026-08-25 · decision · Config inheritance is single-level, overlays touch only `meta` and `drift`

**Decision.** `config/*.yaml` support `_extends` with deep merge, single
inheritance. The three scenario overlays (`sudden_drift`, `gradual_drift`,
`stress_test`) may modify **only** the `meta` and `drift` sections.

**Why.** A scenario is a drift specification, not a licence to change
preprocessing or leakage guards. If an overlay could quietly flip
`normalization.adaptation`, two scenarios would no longer be comparable and the
difference would be invisible in the results.

**Enforcement.** `config.assert_overlay_discipline(filename, allowed=('meta','drift'))`,
called from `config.load()` whenever `_extends` is present.

---

## D-006 · 2026-08-26 · decision · `frozen_after_baseline` normalization — settled by measurement, not argument

**Context.** ASSUMPTION **[A6]** claimed that an online scaler would absorb
mean-shift drift. Rather than argue it, `experiments/phase1_demo.py` measures how
much of an injected 2.0 σ sudden drift each mode retains.

**Measured** on the real inference stream:

| Mode | Retained shift | Fraction of frozen |
|---|---|---|
| `frozen_after_baseline` | **+1.7586 σ** | 100.0 % |
| `running` | +1.5952 σ | 90.7 % |
| `rolling` | **−0.0260 σ** | −1.5 % |

**Decision.** `frozen_after_baseline` is the default; `running` and `rolling` are
declared **ablations**, not alternatives.

**Why.** The adaptive modes demonstrably erase the signal the experiment exists to
detect. A rolling scaler re-centres after the onset, so a drift detector
downstream would see nothing. Assumption [A6] is therefore true in a way that is
fatal, not helpful.

---

## D-007 · 2026-08-26 · decision · Verify harnesses live at the project root, and are never deleted

**Decision.** Each Phase 1 step ends with a re-runnable `verify_stepN.py` at the
project root — deliberately not under `src/`, so it cannot be mistaken for a
component. Each prints measured values and a PASS/FAIL tally, exiting non-zero on
failure.

**Why root.** A verification harness that lives among the components eventually
gets imported by one.

**Standing rule.** Earlier harnesses must stay green. When a later step
legitimately invalidates an assertion, **tighten** it; never delete it. See `E-007`
for the one case where a harness was widened, and the limit placed on that.

---

## D-008 · 2026-08-26 · decision · Four findings recorded and deliberately left unrepaired

**Decision.** Each of the following was measured, recorded in `PROVENANCE.json`
with `severity`/`claim`/`measured`/`consequence`/`action_taken`/`awaiting`, and
**behaviour was left unchanged**:

1. `finding_near_collinear_pairs` — four continuous pairs exceed |r| > 0.999 and
   none are in `command_feedback_pairs`; three involve channels that
   `top_variance` selects for drift. Injecting drift into one member while its
   twin stays as recorded is physically inconsistent and hands a model an
   untouched proxy.
2. `finding_continuous_command_channels` — `actuator_policy: exclude` filters
   *discrete* channels, so two of five selected channels are continuous
   command/demand channels. Offsetting a setpoint is a different physical story
   from sensor bias — arguably legitimate, but a different drift.
3. `finding_degenerate_outlier_bounds` — 3×IQR collapses to a zero-width band when
   `q1 == q3`; 2 of 58 continuous channels are degenerate, saturating the row-level
   flag at **100 %** (22.98 % once excluded). Made self-reporting via
   `QualityReport.degenerate_bound_columns`; not repaired.
4. `finding_pre_existing_regime_shift` — 8 of 66 features already sit >1 σ from
   baseline before any injection, one at −3.96 σ. A property of the real
   recordings, not a defect.

**Why not fix them.** Each fix is a scientific choice about what the experiment
means, and the standing rule forbids inventing scientific assumptions silently.

**Consequence for Phase 3.** Because genuine shift exists from row 0, detection
latency must be measured **relative to the injected onset with the detector allowed
to settle on the pre-drift segment** — not from stream start.

---

## D-009 · 2026-08-26 · decision · Per-component seed derivation

**Decision.** Components derive their own generator from
`np.random.SeedSequence([master_seed, component_id])` rather than sharing one
global RNG.

**Why.** With a shared stream, adding a single draw anywhere shifts every
downstream component's sequence, so a change to the streamer silently changes the
injected drift. Reproducibility has to survive editing.

---

## D-010 · 2026-08-26 · decision · The demonstration plot smooths for display only

**Decision.** `experiments/phase1_demo.py` applies a 600-sample **trailing** mean
to the already-computed series before plotting, and says so on the axis.

**Why.** At 1 Hz over 15 h, 54,000 raw points render as a solid band and the shift
is invisible. But `preprocessing.filtering.enabled` stays **false**, because
smoothing inside the pipeline would attenuate exactly the abrupt change a
sudden-drift scenario exists to create. The mean is trailing so that even the
picture never shows the reader something the stream had not yet delivered.

---

## D-011 · 2026-08-26 · decision · Handover artifacts: generated state, append-only reasoning

**Context.** Development spans sessions and different models/APIs. Assistant-local
memory does not cross that boundary, and a hand-maintained status document is
correct on the day it is written and silently wrong afterwards.

**Decision.** Three project-local, tool-agnostic artifacts:

- `AGENT_CONTEXT.md` — the contract. Prohibitions, working protocol, bootstrap
  commands, environment gotchas. Read first.
- `DECISIONS.md` — this file. Append-only reasoning.
- `state_report.py` → `PROJECT_STATE.md` — **generated** status, measured off the
  tree, carrying a fingerprint over the code/config/provenance surface so
  `--check` exits non-zero when it goes stale.

**Why the split.** Status decays and must be generated; reasoning does not decay
and must be permanent. Putting either in the other's file is how handover
documents start lying.

**Verified.** Staleness detection was tested by perturbing the surface: `--check`
reported `STALE` and exited 1, then returned to `CURRENT` and exit 0 once the
perturbation was removed.

---

## E-001 · 2026-08-26 · error+fix · `verify_step2.py` check 4 dropped to 9/10 when the test suite landed

**Symptom.** After Step 6 added 8 test modules, check 4 failed:
`HARD-CODED HAI paths found in: ['tests\\test_integrity.py']` at line 209.

**Root cause.** Check 4 scans `src/`, `tests/`, `adaptation/` and `main.py`. Until
Step 6 the `tests/` directory held only `.gitkeep`, so that arm of the loop had
nothing to examine and **passed vacuously**. Adding the suite put it in scope and
it immediately caught a genuine violation of `D-002`: the new test named the
train2 file literally.

**Fix.** The **test** was changed, not the harness — the harness was enforcing the
rule correctly. The filename is now resolved via
`Path(cfg["dataset"]["files"]["train2"]["path"]).name`.

**Lesson.** A scan over an empty directory is a green light that means nothing.
Worth knowing for every remaining phase: harness coverage grows with the tree, and
a check that has never had anything to look at has never been tested.

---

## E-002 · 2026-08-26 · error+fix · The same check failed again on prose in the replacement docstring

**Symptom.** After `E-001`, check 4 still failed on the same file.

**Root cause.** The docstring explaining *why* the filename must not be hard-coded
contained the filename. `verify_step2.py` uses a plain-text regex over raw file
bytes; unlike `tests/test_integrity.py`, which blanks string literals and comments
with `tokenize` (including Python 3.12+ `FSTRING_START/MIDDLE/END`), it cannot
distinguish prose from code.

**Fix.** The docstring was reworded to say "the train2 file". The scanner was **not**
weakened.

**Lesson.** Two scanners in this repo have deliberately different strictness. The
integrity suite tolerates prose so the codebase can explain itself; check 4 does
not. Write documentation to satisfy the stricter one.

---

## E-003 · 2026-08-26 · error+fix · Three API mismatches while writing the demonstration

**Why record something this small.** Each cost a run, and each will cost the next
model a run too. Signature trivia is not in the README and is not guessable from a
call site; this is exactly the class of knowledge that dies at a session boundary.

| Symptom | Reality |
|---|---|
| `AttributeError: 'SeedRecord' object has no attribute 'master_seed'` | fields are `master`, `strict`, `components`, `sweep`, `pythonhashseed`, `global_seeded` |
| `TypeError: read_raw() got an unexpected keyword argument 'config_dir'` | `read_raw(path)` takes a path; `assert_overlay_discipline(filename, *, config_dir=...)` takes a filename and does its own reading |
| `ConfigError: configuration file not found: ...\config\sudden_drift` | `assert_overlay_discipline` needs the **`.yaml`** extension; `load()` does not |

---

## E-004 · 2026-08-26 · error+fix · `realised_magnitude` is per-channel, not a scalar

**Symptom.** `TypeError: unsupported format string passed to dict.__format__`.

**Root cause.** `GroundTruth.realised_magnitude` is a **per-channel dict**. The
scalar lives at `gt.schedule_summary['realised_mean_magnitude_sigma']`, with
`attenuation_ratio` beside it.

**Fix.** The demonstration now prints per-channel values *and* a computed mean, and
records `attenuation_ratio` and `realised_per_channel` in the summary row.

**Why the API is right and the caller was wrong.** Physical clipping bites each
channel differently, so a single number hides the spread — which is exactly the
open question in `finding_realised_magnitude_attenuation`.

---

## E-005 · 2026-08-26 · finding · Physical clipping delivers less drift than requested

**Measured** on the real inference stream, every figure recomputed independently of
the generator's own bookkeeping (agreement to 0.00e+00):

| Scenario | Requested | Realised | Attenuation | Window-level | Windows |
|---|---|---|---|---|---|
| `sudden_drift` | 2.0 σ | 1.7951 σ | 0.898 | 1.7584 σ | 5,396 |
| `gradual_drift` | 2.0 σ | 1.5154 σ | 0.909 | 1.4974 σ | 5,396 |
| `stress_test` | 3.0 σ | **1.4822 σ** | **0.823** | 1.4219 σ | 5,396 |

**The uncomfortable part, reported because it is true.** `stress_test` requests
3.0 σ and delivers **less absolute drift than `sudden_drift`**, which requests 2.0 σ.
`P2_SCO` saturates against its physical range on 28,080 of 35,100 drifted rows
(80 %). A scenario named "stress" is currently the *weakest* of the three.

**Action taken.** None. Recorded in `PROVENANCE.json →
finding_realised_magnitude_attenuation`; realised magnitude is measured per run and
never assumed. Whether the stress scenario should be re-specified is a scientific
decision. **Awaiting the user.**

---

## E-006 · 2026-08-26 · error+fix · `verify_step2.py` checks 7 and 8 fell to 8/10 on the demonstration's own output

**Symptom.** Running all three scenarios in one pass writes three ground-truth
sidecars to `data/synthetic/`; both checks whitelisted the single exact name
`ground_truth.json`.

**Root cause.** The checks' own comments state the invariant as *"nothing here is
read back as an INPUT"* and *"the ground-truth sidecar is metadata about the
injection, not a data stream"* — but they implemented it as an exact-name match.
Per-scenario sidecars are the same kind of object; giving each scenario its own
file is an output-location choice, and letting the third silently overwrite the
first two would have been worse.

**Fix.** This is the one case where a **harness** was changed rather than the code.
Both checks now match the sidecar *shape*, `ground_truth(_<scenario>)?\.json`,
which is what their comments already described. Any file in `data/synthetic/` that
is not a ground-truth sidecar — a drifted stream, a feature dump — still fails.

**Why this was legitimate and `E-001` was not.** In `E-001` the harness correctly
caught a real violation, so the code changed. Here the harness's implementation was
narrower than its stated invariant, and the output was legitimate. The test for
which side to change is: *does the harness's own statement of intent permit this?*

---

## E-007 · 2026-08-26 · finding · Three defects the pytest suite found that all four verify harnesses had missed

Reported because they were real, and because a suite that finds nothing has not
been tested.

1. **`src/utils/config.py` — `require()` could never report more than one missing
   key.** It passed the module's own "no default supplied" sentinel into `get()`,
   so `get()` raised on the first absent path and the aggregation branch was
   unreachable dead code. Fixed with a local sentinel; it now lists every missing
   key at once, which is the entire point of a start-of-run assertion.
2. **`src/data/loader.py` — `BaselineProfile.sigma()` handed back an unusable
   scale.** For a channel dropped as zero-variance it returned `0.0`, contradicting
   its own docstring; any caller dividing by it would silently produce `inf`/`nan`
   inside a drift-magnitude calculation. It now raises. (Audited: only members of
   `profile.continuous` are ever passed to it, so no live call site changed.)
3. **`src/data/preprocessing.py` — a leakage guard that fired only on data that
   happened to have holes.** `_impute()` validated `interpolate_direction` *after*
   short-circuiting on "no missing cells". The inference stream contains zero
   missing cells, so an acausal `interpolate_direction: both` passed silently there
   and would have failed only on a stream with gaps — the same configuration
   accepted or rejected depending on the input. The validation is now hoisted above
   the short-circuit, pinned by
   `test_the_imputation_guard_does_not_depend_on_the_data_having_holes`.

**Lesson, and a standing instruction.** The verify harnesses and the pytest suite
are **independent layers** that catch different classes of defect. Running only one
gives a false green. Run all five.

---

## D-012 · 2026-08-26 · milestone · Phase 1 complete

All six steps done. Measured at completion, each printed by the run that produced
it:

| Command | Result |
|---|---|
| `verify_step2.py` | 10/10 |
| `verify_step3.py` | 13/13 |
| `verify_step4.py` | 15/15 |
| `verify_step5.py` | 12/12 |
| `pytest -q` | 230 passed |
| `experiments/phase1_demo.py` | ran end to end, 36 s, 3 scenarios, full 54,000-row stream |

**Not reported, deliberately:** accuracy, Macro-F1, latency, resource consumption,
or any Edge/Cloud/Hybrid decision. No model, drift detector, reliability estimator
or controller exists yet, so any such number would be fabricated.

**Phase 2 is blocked, not pending** — see `D-004`.

---

## D-013 · 2026-08-26 · decision · The handover mechanism has its own harness

**Context.** `AGENT_CONTEXT.md` asserts that executable tests are the only form of
project context that cannot be forgotten. Shipping the handover artifacts with
nothing checking them would contradict that in the one place a newcomer trusts
most.

**Decision.** `verify_handover.py` at the project root, twelve checks, same
PASS/FAIL-tally and non-zero-exit shape as the four step harnesses (`D-007`).
The checks that matter are the cross-document ones: that `PROJECT_STATE.md` still
matches the tree's fingerprint, that `AGENT_CONTEXT.md` and `README.md` quote
**identical** verification counts, and that no finding recorded as
`awaiting: user decision` is missing from this log.

**Why not add it to `pytest`.** Regenerating `PROJECT_STATE.md` is a session-end
ritual, not a per-edit obligation. A staleness check inside the suite would leave
it red during ordinary development and destroy the signal that "the suite is
green" currently carries. It also keeps the suite at 230, so the counts quoted in
two documents do not need editing every time this mechanism changes.

**It is not vacuous — it failed on first run.** Two genuine defects, both fixed:

1. `E-003` was the only entry in this log with no `**Why**`/`**Root cause**`
   heading — its rationale was a bare sentence in a different phrasing. The **log**
   was brought to the house shape rather than the checker's pattern widened, so the
   file stays uniform.
2. `README.md` did not mention `AGENT_CONTEXT.md` or `DECISIONS.md` at all, so a
   newcomer landing on the README — the overwhelmingly likely entry point — would
   never have found the contract. Fixed with a pointer block under the status
   header and a new "Continuity across sessions and models" subsection in §8.

The second is the failure the harness exists for: three well-written artifacts that
nobody is routed to are worth nothing.

**Measured after the fixes.** `verify_handover.py` 12/12, and the pre-existing five
unaffected: `verify_step2` 10/10, `verify_step3` 13/13, `verify_step4` 15/15,
`verify_step5` 12/12, `pytest` 230 passed.

---

## D-014 · 2026-08-27 · dataset · HAI 23.05 identified from content; three files added, none overwritten

**Context.** Six official HAI files arrived in `data/raw/HAI_23.05_downloads/`. Three
files with the same names were already in `data/raw/` and had every Phase 1
measurement resting on them. The standing rules forbid identifying a release from a
filename or a size, and forbid overwriting anything before the new files are
verified.

**Measured.** `audit_dataset.py` (root-level, version-agnostic, directories from
argv, no filename written into it) read all nine files. SHA-256 proved the three
files already present are **byte-identical** to the newly supplied ones:

| file | sha256 (first 16) |
|---|---|
| `hai-train1.txt` | `53007b0ba604fbf3` |
| `hai-train2.txt` | `0e520e82bf78a661` |
| `hai-test1.txt` | `78c7f1d4de1f2ab9` |

So the earlier drop was **already 23.05 and merely incomplete** — it lacked `test2`
and both label sidecars. Nothing needed overwriting, and `cp -n` added exactly three
genuinely new files. The pre-existing files' mtimes are unchanged.

**Release identified from content**, not names: 87 columns with one byte-identical
column set across all four process-value streams (hash `d800fcbb4abe27bd`); the
`x100*` SETPOINT/ASSIGN/SUM channels that earlier releases lack; attack labels in
**separate sidecar files** rather than as a column; exactly train1/train2/test1/test2
as the released split; 2022-08-04 .. 2022-08-19, exact 1 Hz, 0 gaps, 0 missing cells.

**Correction on the record.** Revisions 1–4 of `PROVENANCE.json` called this
"HAI 22.04". That attribution was never measured — it was inferred when only three
files were present. Corrected to 23.05 with the evidence above, and the wrong value
kept in `dataset_version_history` rather than quietly replaced.

**Why train3/train4 are absent from the config.** The user excluded them from the
first experiment. They are not declared in any YAML, so no code path can reach
them — absence is enforced structurally rather than by a comment.

---

## D-015 · 2026-08-27 · decision · Chronology re-measured with test2; `train1_only` independently confirmed

**Context.** `D-003` set `dataset.baseline_source: train1_only` (ASSUMPTION [A17])
because train2 lies in test1's future. Adding a fourth stream could have changed
that picture, so it was re-measured rather than assumed to still hold.

**Measured** (first/last parsed timestamp per file, `audit_dataset.py`):

```
train1 : 2022-08-04 18:00:01 -> 08-08 00:00:00   (78.0 h, 280,800 rows)
test1  : 2022-08-12 16:00:01 -> 08-13 07:00:00   (15.0 h,  54,000 rows)
train2 : 2022-08-13 07:00:01 -> 08-16 16:00:00   (81.0 h, 291,600 rows)
test2  : 2022-08-17 00:00:01 -> 08-19 16:00:00   (64.0 h, 230,400 rows)
```

True order is **train1 < test1 < train2 < test2** — not the order the filenames
imply. `train1` remains the only file recorded entirely before `test1`.

**Decision.** No change. `baseline_source: train1_only` is **confirmed**, not
revised, and the 78 h causal baseline stands against the 159 h acausal one.

**Why this is worth an entry despite changing nothing.** A re-measurement that
confirms a prior decision is evidence; skipping it because the answer was expected
is how a stale assumption survives a dataset change. `test2` being the latest stream
would have admitted a *larger* legitimate baseline — that is precisely the
temptation this check was run against.

---

## D-016 · 2026-08-27 · decision · Target resolved to `labels_from_hai_labels` — by the user

**Context.** `TASK_UNRESOLVED` had blocked Phase 2 since Step 2. Of the three
options on record, `labels_from_hai_labels` was marked `REQUIRES_ADDITIONAL_FILE`.
HAI 23.05 supplies that file, so the option stopped requiring an absent input. The
choice was the user's, as `AGENT_CONTEXT.md` reserved it.

**Why alignment had to be proven before adopting it.** Matching row counts and
matching first/last timestamps do **not** establish alignment. A dropped second in
one file and a duplicated second in the other cancel out in both summaries while
shifting every label by one row. Row *i* of the labels must be shown to describe row
*i* of the data, elementwise, or the evaluation measures noise. `audit_alignment.py`
was written for exactly this and compares the full timestamp vectors position by
position.

**Measured on `hai-test1` ↔ `label-test1`, 6/6:** 54,000 labels for 54,000 rows;
timestamps equal **elementwise** on all 54,000; timestamp sets identical; label
stamps unique (a joinable key); 0 missing; 2 classes; distribution 0: 51,019
(94.4796%) / 1: 2,981 (5.5204%); **14 contiguous positive-class episodes**, i.e.
attack runs rather than scattered positives.

**Consequence for the primary metric.** Post-drift **Macro-F1 is preserved**, as the
specification requires. The 5.52% positive rate is also why Macro-F1 rather than
accuracy: a majority-class predictor already scores 94.48% accuracy here.

**Configuration.** `task: labels_from_hai_labels`, `label_column: label`,
`positive_class: 1`, `label_file` naming the sidecar. `target_column` stays **null** —
no process channel is the target. Second quarantine added alongside the drift
ground-truth one: `label_usage: evaluation_only` with 13 forbidden consumers. The two
are never merged; they are different objects — what the plant actually did, versus
what we injected.

---

## D-017 · 2026-08-27 · finding · `label-test2.txt` is minute-resolution, so `test2` is reserved and unreachable

**Measured** (`audit_alignment.py` on `hai-test2` ↔ `label-test2`, only **3/6**):

* PASS — row count matches the stream exactly (230,400), 0 missing labels, 2 classes.
* FAIL — label stamps are not unique: **3,841 distinct stamps for 230,400 rows**
  (226,559 duplicates), because the timestamp column is minute-resolution, textually
  `'2022-08-17 0:00'` rather than `'2022-08-17 00:00:01'`.
* FAIL — timestamps disagree elementwise on **226,560 of 230,400** rows.
* FAIL — the label file starts `00:00:00`; the stream starts `00:00:01`.

**Why it was not repaired.** Positional alignment is *plausible* — the row count
matches to the row, which would be a striking coincidence otherwise — but
**unverifiable from the timestamps**. Assuming it would silently offset every label
by up to 60 s, and a 60 s offset against 38 attack episodes in a 1 Hz stream would
corrupt exactly the boundaries detection latency is measured at. Recorded as
`finding_label_test2_minute_resolution`, `awaiting: user decision`.

**Decision.** `hai-test2.txt` and `label-test2.txt` are declared under
`dataset.reserved_files`, a section **no loader code path reads**. The first design
used `active: false` inside `dataset.files`; that was replaced because a flag can be
flipped by someone who never reads this entry, whereas an unreachable section cannot.
The stream file itself is sound (exact 1 Hz, 0 gaps, 0 duplicates, 0 missing) — what
blocks it is only its labels.

**The cheap honest fix is named, not taken.** Verifying that each minute-stamped
label block covers exactly 60 consecutive stream rows would upgrade positional
alignment from plausible to proven. It is deliberately out of scope: `test2` is not
the first experiment, and doing it here would widen a verification task into a data
repair.

---

## D-018 · 2026-08-27 · finding · The causal baseline is unlabelled — ASSUMPTION [A18]

**Measured.** HAI 23.05 ships label sidecars for the **test** streams only. There is
no label file for `train1` or `train2`, so the one causally valid baseline carries no
labels.

**Why this is a real consequence and not a detail.** Resolving the target to official
labels gives a two-class target on the *inference* stream while leaving the
*training* stream unlabelled. Assigning `label = 0` to every training row would be
**fabricating labels** — forbidden. HAI's documented intent is that the training
files are attack-free normal operation, but that is the distribution's claim about
its own data, not something measurable from files that ship no labels.

**Recorded, not resolved.** `dataset.training_labels_available: false`,
ASSUMPTION [A18]: *train1 is treated as unlabelled, NOT as all-normal.*
`finding_training_labels_absent`, `awaiting: Phase 2 model-design decision`.

**Why not decided now.** Whether the Edge/Cloud models are therefore one-class /
reconstruction detectors, or supervised under an explicit attack-free assumption, is
a **Phase 2 model-design** decision. Taking it inside a dataset-verification task
would settle the study's model architecture as a side effect of loading files. The
third option — carving a labelled development split out of `test1` — is rejected in
advance as leakage, and is already forbidden by `allow_acausal_baseline: false` and
`loader.profile_baseline`.

---

## D-019 · 2026-08-27 · decision · Label sidecars are declared outside `dataset.files`

**Root cause of the failure that forced this.** The label sidecars and the reserved
`test2` were first added as entries in `dataset.files` with new role strings.
`verify_step3.py` then died at `loader.file_specs`:

```
ConfigError: unrecognised file role(s): ['inference_labels',
             'reserved_inference_labels', 'reserved_second_inference_stream']
```

`dataset.files` is not a file list. `file_specs` enforces a contract on it: exactly
one entry with role `inference_stream`, **every other entry a baseline candidate**.

**Why the loader was not changed.** Widening `BASELINE_ROLES` to accept the new roles
would have made the error disappear while leaving a label file sitting in the section
the loader draws baseline training data from — one future edit away from normalising
an answer key as if it were process values. The loader's contract was right; the
config was wrong.

**Decision.** Three sections with three meanings:

* `dataset.files` — process-value streams only, unchanged, still 3 entries.
* `dataset.labels` — official label sidecars, keyed by the stream each aligns with.
* `dataset.reserved_files` — declared for provenance and chronology, unreachable.

**Enforced, not just documented.** `verify_step2.py` check 4 now (a) re-counts all
six declared files from disk, (b) **fails if any entry of `dataset.files` looks like
a label file**, and (c) requires every active sidecar to name a declared stream and
carry `alignment: elementwise_verified`. Check 8 additionally asserts
`dataset.label_file` and `dataset.labels.<stream>.path` name the same file, because
two declarations of one fact drift apart and the file the loader reads must be the
file whose alignment was proven.

**Lesson.** A role vocabulary is a type system. When a new kind of object arrives,
the cost of widening the vocabulary is paid later and elsewhere, by whoever assumes
the old invariant still holds.

---

## E-008 · 2026-08-27 · error+fix · Eight assertions whose premises the dataset made false

**What happened.** After the migration, `verify_step2` fell to 9/10, `verify_step3`
to 12/13, and `pytest` to 6 failed / 224 passed. Every failure was an assertion whose
*premise* had become false, not a broken implementation:

| assertion | premise that expired |
|---|---|
| `verify_step2` check 8 | "HAI has no label column; `label_column` must be null" |
| `verify_step3` check 4 | `resolve_target` must raise |
| `test_integrity.py::test_provenance_records_every_supplied_file` | three raw files |
| `test_integrity.py::test_no_module_fabricates_a_label` | `task == 'unresolved'` |
| `test_integrity.py::test_resolving_a_target_today_raises...` | must raise |
| `test_loader.py::test_target_resolution_refuses_to_invent_a_label` | must raise |
| `test_loader.py::...requires_a_label_file_that_was_not_supplied` | file not supplied |
| `test_stream.py::...refuses_while_the_task_is_unresolved` | `positive_class is None` |

**Why none of them was deleted or loosened.** "Do not weaken a test to make it pass"
is the rule, and a test whose premise expired is the most tempting thing in a
codebase to delete — it is *genuinely* obsolete, which is exactly what makes removing
it look free. Each was instead replaced by the stronger statement it had been
standing in for:

* *the label must be null* → **the label must be read from a file the distribution
  shipped, with a recorded checksum, and must not be a column of the process-value
  stream** (i.e. provably not derived);
* *resolution must raise* → **resolution returns the sidecar's column and not a
  process channel, AND still raises on an unresolved config, AND still raises on a
  half-declared one** (`label_file` or `label_column` removed) — one assertion became
  four;
* *provenance records three files* → six, each with checksum, alignment status and
  usage restriction.

`test_provenance_records_every_supplied_file` needed no change at all: it was
correctly demanding that `PROVENANCE.json` record the three new files, and the fix
was to write the record.

**One real (not premise) defect fixed.** `src/data/loader.py`'s `resolve_target`
docstring and its `UnresolvedTaskError` message both stated as fact that HAI ships no
labels and that `labels_from_hai_labels` "requires `dataset.label_file`, not
currently supplied". Both were now false and would mislead the next reader. Reworded
without naming any filename, because `verify_step2` check 4 scans `src/` for
hard-coded HAI filenames including inside docstrings.

**Measured after the fixes.** `verify_step2` 10/10, `verify_step3` 13/13,
`verify_step4` 15/15, `verify_step5` 12/12, `pytest` 230 passed — the same counts as
before the migration, against strictly more assertions.

---

## E-009 · 2026-08-27 · error+fix · `audit_dataset.py` was left on disk with a NUL byte in it

**What happened.** The audit tool the previous session created could not run at all:

```
SyntaxError: source code cannot contain null bytes
```

**Root cause.** A literal `\x00` had been written as the separator in the
column-hash expression at line 84 (`"\x00".join(cols)`) — found by byte scan at
offset 3095, not visible in any editor view of the line.

**Fix.** A targeted byte patch (`b'\x00'` → `b'|'`) rather than rewriting the file,
so nothing else in a 250-line tool could change silently under cover of the repair.
Confirmed with `py_compile`, then run.

**Lesson.** The previous session's only artifact was a file that had never been
executed. "A tool exists" and "a tool runs" are different claims, and the repository
records the first while only execution establishes the second — which is why the
state was re-derived by running everything rather than by reading what was left
behind.

## D-020 · 2026-08-28 · decision · Migration from HAI to WUSTL-IIoT-2021 as Active Dataset

**Context.** The user directed migrating the active dataset from HAI 23.05 to WUSTL-IIoT-2021 (`data/raw/wustl_iiot_2021.csv`, 409,800,698 bytes, SHA-256 `f897b24578cc6fdeb3e7a0e9ff63efd5bbdc926a545abbda725e0dbb348c6bca`), while preserving HAI records as historical provenance.

**Why.** WUSTL-IIoT-2021 provides real network traffic in industrial IoT SCADA systems with embedded multi-class / binary attack labels (`Target`: 0=normal, 1=attack), eliminating reliance on external sidecars for supervised classification.

**Audited Dataset Semantics and Rigorous Safeguards:**
* **Flow-level Event Stream:** WUSTL is a flow-level event stream (`stream_semantics: flow_level`), not a continuous time-series sampled at fixed seconds. Equal timestamps share `modal_interval_s: 0.0`.
* **Deterministic Tie-Breaking:** With 25,267 unique `StartTime` timestamps and 1,169,197 duplicates, ordering is strictly defined by chronological `StartTime` followed by a deterministic tie-breaker tuple: `['SrcAddr', 'DstAddr', 'Sport', 'Dport', 'Proto', 'sIpId', 'dIpId']`.
* **Tie-Breaker Column Quarantine:** All 7 tie-breaker columns are strictly excluded from model feature extraction and drift generation.
* **Causal Three-Way Partitions:**
  - `train1` (baseline training): `09:46:03` to `11:29:48`, 304,166 rows (295,926 normal, 8,240 attack).
  - `train2` (validation / post-baseline): `11:29:49` to `13:07:36`, 265,685 rows (187,380 normal, 78,305 attack).
  - `test1` (inference stream): `13:07:37` to `16:48:11`, 624,613 rows (624,142 normal, 471 attack).
* **Feature Schema:** Total 49 columns. 12 excluded (5 metadata: `Target`, `Traffic`, `StartTime`, `LastTime`, `RunTime` + 7 tie-breakers). Exactly 37 model features (30 continuous, 7 discrete). 0 zero-variance features on `train1`.
* **Observation Windowing:** Size 50, step 10 on 624,613 rows yields `(624,613 - 50) // 10 + 1 = 62,457` windows; remainder `(624,613 - 50) % 10 = 3` trailing rows dropped. Feature matrix is `62,457 x 194`.
* **Memory Optimization:** Chunked partition loading (chunksize 200,000) maintains RAM under 200 MB per partition, preventing array allocation exhaustion on Windows VM without changing any scientific semantics.

**Verification.** All verification harnesses pass cleanly: `verify_step2.py` (10/10), `verify_step3.py` (13/13), `verify_step4.py` (15/15), `verify_step5.py` (12/12), `pytest` (228 passed, 2 skipped, 0 failed).

---

## E-010 · 2026-08-28 · error+fix · Superseded file check and dataset premise updates in unit test suite

**What happened.** Initial pytest run after WUSTL migration had 4 failures:
1. `test_integrity.py::test_provenance_records_every_supplied_file`: failed on unrecorded `attack.csv`.
2. `test_loader.py::test_hai_label_task_requires_a_label_file`: failed because `cfg["dataset"]["label_file"]` is None for WUSTL, triggering the first assertion instead of the intended `label_column` check.
3. `test_preprocessing.py::test_statistics_are_fitted_on_the_baseline_only`: asserted hard-coded 58 continuous / 8 discrete features from HAI instead of dynamic profile counts (30 continuous / 7 discrete for WUSTL).
4. `test_stream.py::test_plan_reads_the_config`: asserted sampling interval 1.0s instead of expected 0.0s for flow-level streams.

**Root cause.** Hard-coded HAI assumptions in test assertions and lack of inclusion of the superseded `attack.csv` (recorded under `superseded_swat_record` in `PROVENANCE.json`) in the raw file inventory check.

**Fix.**
1. Included `superseded_swat_record`'s file in `test_provenance_records_every_supplied_file`.
2. Set an explicit dummy label file before checking `label_column` requirement and added `target_column` refusal assertion for `supervised_classification`.
3. Tested `len(stats.continuous) == len(profile.continuous)` and `len(stats.discrete) == len(profile.discrete)`.
4. Compared `sampling_interval_s` to `expected_sampling_interval_s` from config.

## D-021 · 2026-08-28 · decision · Phase 2 Edge and Cloud Prediction Models Implementation

**Context.** Phase 2 required building and verifying two supervised prediction models using the exact verified Phase 1 feature representation (37 features) from WUSTL-IIoT-2021:
- Edge tier: River Hoeffding Tree Classifier (`river.tree.HoeffdingTreeClassifier`)
- Cloud tier: XGBoost Classifier (`xgboost.XGBClassifier`)

**Architectural Decisions & Invariants:**
1. **Common Model Protocol (`src/models/base.py`):**
   - Abstract base class `BaseModel` specifies: `fit(X, y)`, `predict(X)`, `predict_proba(X)`, `predict_one(x)`, `predict_proba_one(x)`, `get_info()`.
   - Tracks `is_trained`, `n_features`, `feature_names`, `last_inference_time_s`, `mean_inference_time_per_sample_s`.
   - Enforces feature dimension and column name alignment; raises `NotTrainedError` on inference before fitting and `InputDimensionError` on misaligned schemas.
2. **Edge Model Implementation (`src/models/edge_model.py`):**
   - Wraps River's `HoeffdingTreeClassifier`.
   - Supports online/incremental learning via `learn_one(x, y)` and `learn_many(X, y)`.
   - Provides low-latency single-observation (`predict_one`, `predict_proba_one`) and batch prediction (`predict`, `predict_proba`).
   - Normalizes probabilities to ensure both classes `{0, 1}` exist and sum to 1.0.
3. **Cloud Model Implementation (`src/models/cloud_model.py`):**
   - Wraps `xgboost.XGBClassifier`.
   - Seeded deterministically using `src.utils.seed.master_seed(config)`.
   - Provides high-throughput vectorised batch prediction (`predict`, `predict_proba`) and single-observation wrappers (`predict_one`, `predict_proba_one`).
4. **Causal Data Pipeline & Quarantine (`src/models/trainer.py`):**
   - Strictly enforces causal partitions: model fitting is permitted ONLY on `baseline_train` (`train1`, 304,166 rows).
   - Preprocessing statistics and profile are fitted on `baseline_train` only; evaluation data (`baseline_validation`, `inference_stream`) reuse frozen statistics.
   - `Target` column is extracted causally as ground truth and strictly excluded from feature matrix $X$.
   - All 11 metadata/tie-breaker columns (`Target`, `Traffic`, `StartTime`, `LastTime`, `RunTime`, `SrcAddr`, `DstAddr`, `Sport`, `Dport`, `Proto`, `sIpId`, `dIpId`) are checked and forbidden from entering $X$.
   - Refuses training on `baseline_validation` or `inference_stream` with `CausalityError`.
5. **Evaluation Metrics:**
   - Evaluates Macro-F1 (primary metric under class imbalance), accuracy, macro-precision, macro-recall, and confusion matrix.

**Verification.**
- All Phase 1 harnesses remain 100% green:
  - `verify_step2.py`: 10/10 PASS
  - `verify_step3.py`: 13/13 PASS
  - `verify_step4.py`: 15/15 PASS
  - `verify_step5.py`: 12/12 PASS
- Phase 2 verification harness `verify_phase2.py`: 12/12 PASS
- Unit test suite `pytest`: all Phase 1 and Phase 2 tests pass (242 passed, 2 skipped, 0 failed).
- Small smoke test executed on causal partitions without RAM bottlenecks.

---

## D-022 · 2026-08-28 · decision · Phase 3 Drift Detection Layer (ADWIN, Persistence, Severity)

**Context.** Phase 3 requires detecting when the incoming stream experiences statistically meaningful change, tracking the persistence of the detected change, and quantifying its continuous severity $D \in [0, 1]$, without consuming future ground truth or labels.

**Architectural Decisions & Invariants:**
1. **Separation of Concerns:**
   - **ADWIN (`src/drift/adwin_detector.py`):** Wraps River's `ADWIN`. Evaluates whether changes in the running mean of the monitored signal are statistically significant under Hoeffding bounds with confidence parameter $\delta=0.002$.
   - **Persistence (`src/drift/persistence.py`):** Filters isolated transient alarms from sustained regime shifts using configurable criteria (`consecutive` streak threshold $K$ or `windowed_count` $N$ alarms in $T$ observations).
   - **Severity (`src/drift/severity.py`):** Quantifies the magnitude of observed change into a continuous normalized metric $D \in [0, 1]$.
   - **Pipeline (`src/drift/__init__.py`):** Integrates the three components into a causal streaming coordinator `DriftPipeline`.

2. **Monitored Inference Signal:**
   - Default monitored signal is `prediction_probability` ($P(\text{Target}=1 | x_t)$ from model inference).
   - Supports `uncertainty` ($2 \cdot (1 - \max_k P(k))$), `prediction` ($\hat{y} \in \{0, 1\}$), and evaluation-only `prediction_error` (which requires explicit scalar inputs and never queries labels).
   - Ground truth binary labels (`Target`) and synthetic drift sidecar metadata (`ground_truth.json`) are strictly evaluation-only and forbidden from entering the detector.

3. **Drift Severity Formula & Distinction:**
   - Raw severity is defined by the exact mathematical formula:
     $$D = \min\left(1.0, \frac{|\text{current\_shift} - \text{baseline\_mean}|}{\text{max\_shift}}\right)$$
     (with alternative `exponential`: $D = 1 - \exp(-\lambda \cdot |\text{current\_shift} - \text{baseline\_mean}|)$).
   - `baseline_mean` is computed causally from `baseline_train` using `compute_baseline_signal_mean(model, X_baseline_train)` or configured. It is never estimated from validation or test data.
   - `raw_severity` and `smoothed_severity` are strictly separated:
     $$\text{smoothed\_severity}_t = \alpha \cdot \text{smoothed\_severity}_{t-1} + (1 - \alpha) \cdot \text{raw\_severity}_t$$
     where $\alpha \in [0, 1)$ is the configurable smoothing factor.

4. **Memory Optimization:**
   - Preallocated 2D feature matrix in `src/data/preprocessing.py` (`extract_features`) replaced dynamic Python list accumulation of 62,457 small numpy arrays, eliminating heap fragmentation and `ArrayMemoryError` on 624k-row WUSTL streams without changing any calculation.

**Verification.**
- `verify_step2.py`: 10/10 PASS
- `verify_step3.py`: 13/13 PASS
- `verify_step4.py`: 15/15 PASS
- `verify_step5.py`: 12/12 PASS
- `verify_phase2.py`: 12/12 PASS
- `verify_phase3.py`: 12/12 PASS
- `pytest tests/test_drift.py`: 20/20 PASS
- Handover contract passes with zero discrepancies.

## D-023 · 2026-08-28 · decision · Phase 4 DRAEC Prediction Reliability Estimation Layer

**Context.** Phase 4 defines and implements the DRAEC prediction reliability estimation layer $R_t \in [0, 1]$, combining prediction confidence $C_t$, recent prediction error $E_t$, smoothed drift severity $D_t$, and causal data/sensor quality $Q_t$.

**Mathematical Formulations & Invariants:**
1. **Prediction Confidence $C_t$:**
   Evaluated immediately at inference time from binary class probabilities $\{P_t(0), P_t(1)\}$ without requiring ground-truth Target:
   $$C_t = 2 \cdot (\max(P_t(0), P_t(1)) - 0.5) \in [0, 1]$$
   $C_t = 0.0$ corresponds to maximum classification ambiguity ($P(0) = P(1) = 0.5$); $C_t = 1.0$ represents deterministic certainty.

2. **Instantaneous Loss & Recent Error $E_t$ (Delayed Feedback Support):**
   When legitimate ground-truth feedback arrives:
   $$e_t = \mathbb{I}(\hat{y}_t \ne y_t) \in \{0, 1\}$$
   Recent error is tracked via exponential moving average:
   $$E_t = \alpha_E \cdot E_{t-1} + (1 - \alpha_E) \cdot e_t, \quad \alpha_E = 0.8$$
   In operational streaming where labels are delayed or absent, the estimator retains its previous $E_t$ state without fabricating errors or looking ahead.

3. **Drift Severity $D_t$:**
   Directly consumes Phase 3 `smoothed_severity` ($D_t \in [0, 1]$), which measures distance from the frozen causal baseline mean. It does not use ADWIN's transient boolean alarms.

4. **Data/Sensor Quality $Q_t$:**
   Dataset-independent formulation for $N_F$ features:
   $$Q_t = \frac{1}{N_F} \sum_{j=1}^{N_F} q_{j,t} \in [0, 1]$$
   Instantiated on WUSTL-IIoT-2021 with $N_F = 37$ features, computed causally from Phase 1 `QualityReport` without introducing arbitrary penalties.

5. **Weighted Harmonic Mean Combination ($R_t$):**
   Reliability-oriented factors: $r_C = C_t$, $r_E = 1 - E_t$, $r_D = 1 - D_t$, $r_Q = Q_t \in [0, 1]$.
   $$R_t = \frac{w_C + w_E + w_D + w_Q}{\frac{w_C}{r_C + \epsilon} + \frac{w_E}{r_E + \epsilon} + \frac{w_D}{r_D + \epsilon} + \frac{w_Q}{r_Q + \epsilon}}$$
   Initial experimental equal weights: $w_C = w_E = w_D = w_Q = 0.25$, with stability parameter $\epsilon = 10^{-8}$.
   Satisfies the weakest-link principle: severe degradation in any single factor collapses $R_t$ toward zero.

6. **Global Scope & Causality:**
   $R_t$ is a global inference condition reliability metric; it does not introduce action scores ($R_{\text{edge}}, R_{\text{cloud}}$) or network/controller logic (deferred to later phases). Ground truth remains strictly quarantined in `ground_truth.forbidden_consumers`.

**Verification.**
- `verify_phase4.py`: 12/12 PASS
- `pytest tests/test_reliability.py`: 26/26 PASS
- Complete regression suite green.

## D-024 · 2026-08-28 · decision · Phase 5 DRAEC Decision Engine and Minimal Execution Layer

**Context.** Phase 5 defines and implements the DRAEC Decision Engine and minimal execution layer, routing streaming inference across the discrete action space $a_t \in \{\text{EDGE}, \text{CLOUD}, \text{HYBRID}\}$ driven by the frozen Phase 4 prediction reliability signal $R_t \in [0, 1]$ and causal runtime state.

**Architectural Formulations & Invariants:**
1. **Discrete Action Space & Semantics:**
   $$a_t \in \{\text{EDGE}, \text{CLOUD}, \text{HYBRID}\}$$
   - **EDGE:** Execute Edge River Hoeffding Tree Classifier and return its prediction.
   - **CLOUD:** Execute Cloud XGBoost Classifier and return its prediction.
   - **HYBRID:** Edge-first inference. Edge executes first; if the causal confidence condition indicates Edge is insufficient ($C_{\text{edge}} < \tau_{\text{fallback}}$, default 0.60), Cloud fallback is invoked and Cloud provides the final result (`cloud_fallback = True`). No probability averaging or ensemble fusion.

2. **Adaptive Controller with State-Machine Hysteresis:**
   - Primary reliability signal: $R_t \in [0, 1]$ from Phase 4.
   - Configurable experimental default thresholds:
     $$\tau_{\text{critical}} = 0.30 < \tau_{\text{cloud}} = 0.50 < \tau_{\text{return}} = 0.70$$
   - Deterministic transitions respecting current state:
     - When `EDGE`: $R_t \ge 0.50 \implies \text{EDGE}$; $0.30 \le R_t < 0.50 \implies \text{HYBRID}$; $R_t < 0.30 \implies \text{CLOUD}$.
     - When `HYBRID`: $R_t \ge 0.70 \implies \text{EDGE}$; $R_t < 0.30 \implies \text{CLOUD}$; $0.30 \le R_t < 0.70 \implies \text{HYBRID}$.
     - When `CLOUD`: $R_t \ge 0.70 \implies \text{EDGE}$; $R_t < 0.70 \implies \text{CLOUD}$.
   - The deadband $[0.50, 0.70)$ prevents rapid high-frequency chatter between Edge and Cloud under noisy reliability fluctuations. Thresholds are documented as configurable experimental defaults, not scientifically optimal constants.

3. **Static Baseline Controller:**
   - Dedicated `StaticBaselineController` that operates completely independently of $R_t$, $D_t$, and adaptive reliability feedback.
   - Supports fixed deterministic policies (`edge_only`, `cloud_only`, `fixed_ratio`, `round_robin`, `static_hybrid`) under the uniform `BaseController` interface to enable unbiased scientific baseline comparisons in Phase 10.

4. **Two-Level Hybrid Architecture:**
   - **Level 1 (Action Selection):** Controller selects $a_t \in \{\text{EDGE}, \text{CLOUD}, \text{HYBRID}\}$ driven by $R_t$.
   - **Level 2 (Execution):** When $a_t = \text{HYBRID}$, Edge runs first; if $C_{\text{edge}} = 2 \cdot (\max(P_0, P_1) - 0.5) < 0.60$, Cloud is invoked and provides the final output.

5. **Causality, Anti-Leakage & Scope Boundaries:**
   - Operates strictly on causal observation $t$ feature inputs without querying `Target` or inspecting `ground_truth.json`.
   - Preserves frozen WUSTL dataset semantics: training on `train1` only; `train2` and `test1` labels strictly quarantined.
   - Scope is restricted to Phase 5 minimal execution: no physical deployment, MQTT, containerization, formal monitoring (Phase 7), or model retraining (Phase 9).

6. **Lightweight Instrumentation:**
   - Memory-bounded telemetry tracking decision counts, action counts (Edge/Cloud/Hybrid), fallback counts, switch counts, and wall-clock execution latencies.

**Verification.**
- `verify_phase5.py`: 21/21 PASS
- `pytest tests/test_decision.py`: 28/28 PASS
- Complete regression suite green.

## D-025 · 2026-08-29 · decision · Phase 6 Hardened Execution Layer, Latency Measurement, and Failure Handling

**Context.** Phase 6 hardens the execution layer between the DRAEC Decision Engine and the Phase 2 Edge and Cloud models. It establishes strict input/output validation, fine-grained wall-clock timing, explicit failure handling without prediction fabrication, and bounded telemetry, preserving the frozen Phase 4 reliability signal $R_t$ and Phase 5 decision engine hysteresis thresholds.

**Architectural Formulations & Invariants:**
1. **Execution Interface & Status:**
   - Unified `ExecutionResult` and `ExecutionStatus` (`SUCCESS`, `FALLBACK`, `FAILED`).
   - Backward-compatible with Phase 5 `ExecutionResult` fields (`decision`, `action`, `prediction`, `probabilities`, `model_used`, `inference_latency_s`, `cloud_fallback`), adding `success`, `status`, `edge_latency_s`, `cloud_latency_s`, `hybrid_latency_s`, `error`, `observation_index`, `timestamp`.

2. **Causal Input & Output Validation:**
   - `validate_input()`: Validates numeric finite values, shape, and expected dimensions. Rejects None, empty arrays, non-finite values, and strictly blocks forbidden leakage columns (`Target`, `ground_truth`, `Traffic`).
   - `validate_output()`: Enforces binary label contract $\hat{y}_t \in \{0, 1\}$ and valid probability simplex $P_t(y) \in [0, 1]$ summing to $1.0 \pm 10^{-4}$ with finite values.

3. **Fine-Grained Latency Measurement:**
   - $T_{\text{edge}}$: Measured duration of Edge River Hoeffding Tree execution via `time.perf_counter()`.
   - $T_{\text{cloud}}$: Measured duration of Cloud XGBoost execution via `time.perf_counter()`. Documented explicitly as local software execution latency, not network delay, packet transmission, or internet service latency.
   - $T_{\text{hybrid}}$: Measured complete wall-clock duration of the entire Hybrid execution path via a dedicated timer covering Edge execution, confidence check, and conditional Cloud fallback. Component durations $T_{\text{edge}}$ and $T_{\text{cloud}}$ are tracked independently and never fabricated as $T_{\text{edge}} + T_{\text{cloud}}$.

4. **Explicit Failure Semantics (No Fabrication):**
   - Inference failures catch explicit hardware/software exceptions and return `success=False`, `status=ExecutionStatus.FAILED`, `prediction=None`, `probabilities=None`, and the error string. Never fabricates predictions on failure.
   - **EDGE:** Edge succeeds $\implies$ `SUCCESS`; Edge fails $\implies$ `FAILED`.
   - **CLOUD:** Cloud succeeds $\implies$ `SUCCESS`; Cloud fails $\implies$ `FAILED`.
   - **HYBRID:**
     - Edge fails $\implies$ `FAILED`.
     - Edge succeeds and $C_{\text{edge}} \ge 0.60 \implies$ `SUCCESS` (`model_used="hybrid_edge"`, `cloud_fallback=False`).
     - Edge succeeds, $C_{\text{edge}} < 0.60$, and Cloud succeeds $\implies$ `FALLBACK` (`model_used="hybrid_cloud"`, `cloud_fallback=True`).
     - Edge succeeds, $C_{\text{edge}} < 0.60$, and Cloud fails $\implies$ `FAILED` (`cloud_fallback=True`, `prediction=None`).

5. **Memory-Bounded Telemetry:**
   - Streaming-safe telemetry buffer respecting `max_records` (default 10,000) using ring buffers / deques.
   - Computes streaming min/max/mean latency statistics without storing unbounded history.

6. **Scope Boundary:**
   - Phase 6 only. No Phase 7 monitoring, Phase 8 physical deployment, Phase 9 adaptation/retraining, or Phase 10 statistical evaluation.

**Verification.**
- `verify_phase6.py`: 21/21 PASS
- `pytest tests/test_execution.py`: 26/26 PASS
- Full regression suite across Phases 1–5 passing.

---

### D-026: Phase 7 — DRAEC Model Management & Monitoring

- **Date**: 2026-08-29
- **Status**: DECIDED / IMPLEMENTED
- **Phase**: Phase 7
- **Scope**: Observability, Model-State Registry, and Telemetry Engine.

**Context.**
Following hardened execution in Phase 6, the DRAEC architecture requires a dedicated, causal observability layer answering: *"What is happening to the DRAEC system and its models over time?"* The layer must track model health, monitor incoming reliability and drift signals, trace routing decisions, record execution latencies, maintain bounded historical telemetry, and expose data-readiness for downstream Phase 10 evaluations without altering frozen Phase 1–6 components.

**Decisions.**

1. **Purely Observational Layer:**
   - Phase 7 observes, records, and aggregates system state. It does NOT modify the frozen Phase 4 reliability formulation $R_t = f(C_t, E_t, D_t, Q_t)$, Phase 5 decision thresholds ($\tau_{\text{critical}}=0.30, \tau_{\text{cloud}}=0.50, \tau_{\text{return}}=0.70$), or Phase 6 two-level execution paths.
   - It does NOT trigger automatic retraining, model switching, model parameter synchronization, or compression (strictly quarantined to Phase 9).
   - It does NOT run comparative evaluation benchmarks or claim performance superiority (strictly quarantined to Phase 10).
   - It does NOT perform physical deployment or MQTT networking (strictly quarantined to Phase 8).

2. **Model Registry (`ModelRegistry`):**
   - Implemented in `src/monitoring/registry.py`. Tracks Edge (`EdgeHoeffdingTree`) and Cloud (`CloudXGBoost`) models.
   - Maintains `ModelMetadata` including `model_id`, `model_type`, `execution_location`, `model_version`, `status` (`ModelHealthStatus`), `feature_names`, `n_features`, execution counts, and last status/error.
   - Strictly non-mutating: provides observational lifecycle tracking without weight updates.

3. **Central Observability Engine (`DRAECMonitor` / `SystemMonitor`):**
   - Implemented in `src/monitoring/monitor.py`.
   - Ingests causal stream tuples $(t, \text{ReliabilityScore}, \text{drift\_status}, \text{DecisionResult}, \text{ExecutionResult})$.
   - Distinguishes bounded recent history (ring buffer deque capped at `max_records`, default 10,000) from global cumulative stream statistics (total observations, routing counts, distribution %, hybrid fallback rate, execution success rate, streaming min/max/mean for $R_t, D_t, T_{\text{edge}}, T_{\text{cloud}}, T_{\text{hybrid}}$).
   - Missing execution paths preserve `None` rather than fabricating zero values.

4. **Non-Actionable Informational Alerts:**
   - Tracks condition flags (`reliability_degraded`, `drift_active`, `execution_failure_detected`, `cloud_fallback_active`, `model_unavailable`) as observational telemetry only. No automated control intervention is permitted.

5. **Phase 10 Data Readiness:**
   - Implements `get_records_dataframe()` producing a stable 23-column schema supporting all future Phase 10 evaluation requirements (R_t over time, D_t over time, routing distribution, hybrid fallback rate, execution latency, and policy comparison).

**Verification.**
- `verify_phase7.py`: 24/24 PASS
- `pytest tests/test_monitoring.py`: 27/27 PASS
- Full regression suite across Phases 1–6 passing.

---

<!-- Append new entries ABOVE this line, in ascending id order.
     Never edit or delete an existing entry; supersede it with a new one. -->





