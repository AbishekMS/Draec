# Phase 10 Step 10: Publication-Ready Experimental Results and System Analysis

**Document Type:** IEEE Publication Deliverable  
**Authoritative Evidence Base:** Step 8 Multi-Seed Benchmark (`results/step8_combined_runs.csv`, `results/step8_raw_per_seed_results.json`) & Step 9 Independent Scientific Audit (`experiments/verify_step9.py`, Decisions D-040, D-041).  
**Status:** IMPLEMENTED & VERIFIED

---

## 1. Experimental Configurations and Multi-Seed Evaluation Protocol

The evaluation was conducted under the pre-declared Phase 10 benchmark protocol across five evaluation seeds $[42, 123, 456, 789, 2024]$ on the active industrial IoT network stream **WUSTL-IIoT-2021** ($N = 25,000$ chronological steps; baseline: $[0:12,500]$; post-drift: $[12,500:25,000]$):

1. **Configuration A (Moderate Feature-Space Drift):**
   - Magnitude: $2.0\sigma$
   - Affected Features: $n=5$ top-variance flow features
   - Drift Scenario: Sudden step-offset
   - Evaluation Steps: $W = 25,000$
2. **Configuration B (Severe Feature-Space Drift):**
   - Magnitude: $5.0\sigma$
   - Affected Features: $n=8$ top-variance flow features
   - Drift Scenario: Sudden step-offset
   - Evaluation Steps: $W = 25,000$

---

## 2. Table I: Predictive Classification Performance & Candidate Model Validation

### Per-Seed Results

| Config | Seed | Post Accuracy | Post Macro-F1 | Post MCC | Post Prec | Post Recall | Candidate Macro-F1 | Candidate Status | Active Version |
|---|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| **Config A (2.0$\sigma$, $n$=5)** | 42 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416515 | REJECTED | v1 |
| | 123 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416515 | REJECTED | v1 |
| | 456 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.414012 | REJECTED | v1 |
| | 789 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416515 | REJECTED | v1 |
| | 2024 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416379 | REJECTED | v1 |
| **Config B (5.0$\sigma$, $n$=8)** | 42 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416855 | REJECTED | v1 |
| | 123 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416869 | REJECTED | v1 |
| | 456 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416855 | REJECTED | v1 |
| | 789 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416447 | REJECTED | v1 |
| | 2024 | 0.9960 | 0.4990 | 0.0000 | 0.4980 | 0.5000 | 0.416828 | REJECTED | v1 |

### Aggregated Multi-Seed Summary

| Metric | Config A ($2.0\sigma, n=5$) | Config B ($5.0\sigma, n=8$) | df | Variance Source | Statistical Nature / Interpretation |
|---|:---:|:---:|:---:|:---:|---|
| **Post-Drift Accuracy** | $0.9960 \pm 0.0000$ | $0.9960 \pm 0.0000$ | — | Deterministic by Design | Invariant across seeds; reflects $99.68\%$ benign class prior |
| **Post-Drift Precision** | $0.4980 \pm 0.0000$ | $0.4980 \pm 0.0000$ | — | Deterministic by Design | Reflects zero attack precision ($0.0\%$ TP) |
| **Post-Drift Recall** | $0.5000 \pm 0.0000$ | $0.5000 \pm 0.0000$ | — | Deterministic by Design | Reflects $100\%$ benign recall and $0.0\%$ attack recall |
| **Post-Drift Macro-F1** | $0.4990 \pm 0.0000$ | $0.4990 \pm 0.0000$ | — | Deterministic by Design | Floor performance of trivial majority-class predictor |
| **Post-Drift MCC** | $0.0000 \pm 0.0000$ | $0.0000 \pm 0.0000$ | — | Deterministic by Design | Zero correlation with true binary attack labels |
| **Candidate Macro-F1 (Val)** | **$0.4160 \pm 0.0014$** | **$0.4168 \pm 0.0002$** | 4 | **Stochastic / Seed-Dependent** | Genuine variance from seeded baseline sampling \& XGBoost fitting |
| \quad *95% Student-$t$ CI* | $[0.4146, 0.4174]$ | $[0.4165, 0.4170]$ | 4 | Stochastic ($s=0.0011 / 0.0002$) | $df = 4, t_{\text{crit}} = 2.776445$ |
| **Candidate Status** | REJECTED ($10/10$) | REJECTED ($10/10$) | — | Deterministic Criterion | Failed authoritative threshold ($\tau_{\text{val}} = 0.70$) |
| **Successful Deployments** | $0/10$ | $0/10$ | — | Deterministic Safety Gate | Atomic deployer blocked all degraded candidate models |
| **Final Active Version** | `v1` ($10/10$) | `v1` ($10/10$) | — | Deterministic Safety Gate | Active system state preserved at validated baseline |

---

## 3. Table II: Reliability Dynamics & System Orchestration

| Metric | Config A ($2.0\sigma, n=5$) | Config B ($5.0\sigma, n=8$) | Variance Source | Operational / Routing Impact |
|---|:---:|:---:|:---:|---|
| **Drift Onset Step ($t_{\text{drift}}$)** | 12,500 | 12,500 | Deterministic | Midpoint synthetic feature drift injection |
| **ADWIN Detection Step ($t_{\text{detect}}$)** | 12,575 | 12,511 | Deterministic | Rapid change-point detection on feature scalar $S(x_t)$ |
| **ADWIN Detection Delay** | $75 \pm 0$ steps | $11 \pm 0$ steps | Deterministic | Severe drift detected $6.8\times$ faster than moderate drift |
| **Persistent Drift Events** | $100 \pm 0$ | $100 \pm 0$ | Deterministic | 100-step consecutive window confirmation |
| **Maximum Drift Severity ($\max D_t$)** | $0.7299 \pm 0.0000$ | $0.9652 \pm 0.0000$ | Deterministic | Both well-conditioned, strictly $< 1.0$ (pre-saturation) |
| **Mean Post-Drift Severity ($\bar{D}_t$)** | $0.1849 \pm 0.0000$ | $0.8926 \pm 0.0000$ | Deterministic | Severe drift maintains persistently elevated feature deviation |
| **Minimum System Reliability ($\min R_t$)** | **$0.5968 \pm 0.0000$** | **$0.1261 \pm 0.0000$** | Deterministic | Config A: stays $> \tau_{\text{cloud}}$ (0.50); Config B: drops $< \tau_{\text{crit}}$ (0.30) |
| **Mean Post-Drift Reliability ($\bar{R}_t$)** | $0.9444 \pm 0.0000$ | $0.3238 \pm 0.0000$ | Deterministic | Severe drift induces sustained reliability collapse |
| **Edge Execution Ratio** | **$100.000\% \pm 0.000\%$** | **$50.032\% \pm 0.000\%$** | Deterministic | Edge-preserving routing under moderate drift |
| **Hybrid Execution Ratio** | $0.000\% \pm 0.000\%$ | $0.080\% \pm 0.000\%$ | Deterministic | Transient hysteresis corridor transition (20 steps) |
| **Cloud Offloading Ratio** | **$0.000\% \pm 0.000\%$** | **$49.888\% \pm 0.000\%$** | Deterministic | Reliable Cloud offloading under severe drift |
| **Controller Switches** | $0 \pm 0$ | $2 \pm 0$ | Deterministic | Clean two-step transition without hysteretic flapping |

---

## 4. Table III: Adaptation Lifecycle & Safety Firewall Verification

| Adaptation Parameter / Metric | Config A ($2.0\sigma, n=5$) | Config B ($5.0\sigma, n=8$) | Authoritative Source | Verification Outcome |
|---|:---:|:---:|:---:|---|
| **Adaptation Trigger Count** | 2 | 2 | Evaluator Trace | First at detection confirmation; second after 50-step cooldown |
| **Adaptation Cooldown Window** | 50 steps | 50 steps | `evaluation.py:501` | Suppressed redundant retraining requests |
| **Feedback Samples at Trigger** | 1,000 | 1,000 | `FeedbackQueue` | Effective queue capacity reached |
| **Stratified Baseline Budget** | 200 (194 C0, 6 C1) | 200 (194 C0, 6 C1) | Decision D-038 | Proportional class prior preserved ($97.26\% / 2.74\%$) |
| **Total Retraining Sample Count** | 1,200 | 1,200 | `CloudRetrainer` | 1,000 feedback + 200 baseline samples |
| **Baseline Sampling Seed** | `seed` (42..2024) | `seed` (42..2024) | Decision D-039 | 5 distinct sample hashes verified across runs |
| **Candidate Model Architecture** | Cloud XGBoost | Cloud XGBoost | `CloudXGBoost` | Seeded via `random_state=seed` |
| **Candidate Validation Split** | `train2` ($N=25,000$) | `train2` ($N=25,000$) | Causal Protocol | Independent chronological validation split |
| **Authoritative Threshold ($\tau_{\text{val}}$)** | **0.70 (Macro-F1)** | **0.70 (Macro-F1)** | `default.yaml:726` | Formal safety floor |
| **Candidate Validation Macro-F1** | $0.4160 \pm 0.0014$ | $0.4168 \pm 0.0002$ | Evaluator Trace | Failed threshold ($0.4168 < 0.7000$) |
| **Candidate Validation Status** | **REJECTED ($10/10$)** | **REJECTED ($10/10$)** | `CandidateValidator` | Degraded candidate rejected in 100% of runs |
| **Deployment Execution Decision** | **BLOCKED ($10/10$)** | **BLOCKED ($10/10$)** | `AtomicDeployer` | Safety firewall prevented version promotion |
| **Successful Deployments** | 0 | 0 | Step 8 Benchmark | Zero unauthorized deployments |
| **Final Active Model Version** | `v1` ($10/10$) | `v1` ($10/10$) | Step 8 Benchmark | Active production model preserved at `v1` |

---

## 5. Phase 10B: Reliability and Orchestration Dynamics Analysis

### Two-Regime Behavioral Separation
The multi-seed evaluation demonstrates the clear empirical separation between moderate and severe distribution shifts within the DRAEC orchestration framework:

1. **Moderate Drift ($2.0\sigma, n=5$ features): Edge-Preserving Operation:**
   - Synthetic drift injection at step 12,500 alters 5 network flow features by $2.0\sigma$.
   - ADWIN detects the shift at step 12,575 (75-step delay).
   - Drift severity peaks at $\max(D_t) = 0.7299$ and averages $\bar{D}_t = 0.1849$ post-drift.
   - Multi-factor harmonic reliability drops to $\min(R_t) = 0.5968$, remaining strictly above the Cloud offloading threshold ($\tau_{\text{cloud}} = 0.50$).
   - Consequently, the controller preserves **100% Edge execution**, incurring zero network offloading latency and zero offloading switches.
   - *Critical Scientific Caveat:* The controller successfully preserved Edge execution under moderate drift based on the reliability score; however, **this must not be interpreted as preservation of attack-detection performance**, because the underlying Edge classifier exhibited poor minority-class attack recall independently of the synthetic drift intervention.

2. **Severe Drift ($5.0\sigma, n=8$ features): Dynamic Cloud Escalation:**
   - Synthetic drift alters 8 network flow features by $5.0\sigma$.
   - ADWIN detects the distribution shift within 11 steps ($t = 12,511$).
   - Drift severity rapidly escalates to $\max(D_t) = 0.9652$ (remaining well-conditioned and strictly below the $1.0$ saturation cliff).
   - Reliability collapses to $\min(R_t) = 0.1261$, crossing both the Cloud offloading boundary ($\tau_{\text{cloud}} = 0.50$) and the critical threshold ($\tau_{\text{crit}} = 0.30$).
   - The hysteretic controller shifts from Edge to Cloud execution: 50.032% Edge (pre-drift baseline), 0.080% Hybrid (20 steps during transition), and 49.888% Cloud offloading (sustained post-drift execution).
   - Offloading switches are exactly 2 (Edge $\to$ Hybrid $\to$ Cloud), verifying that the hysteresis margins ($0.05$) prevent controller flapping.

### The $n=8 / n=9$ Saturation Boundary Finding
The breadth of affected features governs the saturation boundary of the feature-space scalar detector:
- At $n=8$ features ($5.0\sigma$): $\max(D_t) = 0.9652 < 1.0$ and $\min(R_t) = 0.1261 > \epsilon$. The system operates in a well-conditioned, non-saturated linear regime.
- At $n=9$ features ($5.0\sigma$): $D_t$ reaches $1.0000$ and $R_t$ hits the numerical epsilon floor ($4.0 \times 10^{-8}$).
- At $n \ge 10$ features: Saturated regime.
- *Methodological Scope:* The boundary at $n=8$ is an empirical property of the generic feature-space scalar formulation ($S_t = \frac{1}{37} \sum \min(|z_{t,j}|, 5.0)$) and the top-variance feature structure of WUSTL-IIoT-2021; it is not claimed as a universal boundary for arbitrary real-world distributions.

---

## 6. Phase 10C: Adaptation Lifecycle & Safety Firewall Operation

The full closed-loop adaptation pipeline executed end-to-end:
$$\text{Distribution Shift} \longrightarrow S(x_t) \longrightarrow \text{ADWIN} \longrightarrow \text{Persistence (100 steps)} \longrightarrow \text{Retraining} \longrightarrow \text{Validation} \longrightarrow \text{Rejection} \longrightarrow \text{Active } v_1$$

1. **Triggering & Cooldown:**
   Under persistent drift ($D_t > 0.05$ for 100 consecutive steps), the `AdaptationManager` triggered retraining at $t = 12,511$. A secondary trigger occurred at $t = 12,562$ following the 50-step cooldown.
2. **Retraining Composition:**
   Retraining assembled $1,200$ samples: $1,000$ delayed feedback samples accumulated in the FIFO feedback queue plus $200$ stratified baseline samples (194 Class 0, 6 Class 1) seeded by the run-level RNG.
3. **Candidate Validation & Safety Rejection:**
   The retrained `CloudXGBoost` model was evaluated on the independent chronological validation partition `train2` ($N = 25,000$).
   - Across all 10 runs, candidate models achieved Macro-F1 of $0.4168 \pm 0.0002$, failing the authoritative safety threshold ($\tau_{\text{val}} = 0.70$).
   - `CandidateValidator` issued rejection notices: `"Candidate macro_f1 (0.4168) below minimum threshold (0.7000)."`.
   - `AtomicModelDeployer` suppressed deployment, locking the active production version to `v1`.
4. **Safety Architecture Significance:**
   Candidate rejection in 10 / 10 runs represents the **successful operation of the safety firewall**. Rather than deploying a model that failed out-of-distribution validation, the architecture shielded the edge-cloud inference stream from degraded model promotion.

---

## 7. Phase 10D: Predictive Classifier Analysis (Strictly Decoupled)

### Separation of Concerns: Classification vs. Orchestration
A central finding of the Step 9 audit is that **DRAEC orchestration efficacy is strictly orthogonal to predictive model classification accuracy**. The two must not be conflated.

### Empirical Evidence on Base Classifiers:
1. **Clean Test Stream Performance:**
   When evaluated on clean, un-drifted test data (`test1 clean`), both `EdgeHoeffdingTree` and `CloudXGBoost` (fitted on `train1`) predicted 100% majority class (benign), detecting zero actual attacks ($\text{Recall} = 0.0000, \text{Precision} = 0.0000, \text{MCC} = 0.0000$).
2. **Apparent High Accuracy is an Imbalance Artifact:**
   The reported post-drift streaming accuracy ($99.60\%$) and Macro-F1 ($0.4990$) reflect the baseline prevalence of benign traffic ($99.68\%$) in the selected test window, not effective intrusion detection.
3. **Root Cause: Cross-Partition Feature Inversion:**
   Statistical audit of the WUSTL-IIoT-2021 dataset revealed that key flow features invert their class-conditional correlation between training (`train1`) and downstream evaluation (`train2`, `test1`). For example, in `train1`, attack flows have *smaller* `SAppBytes` than benign flows (mean $-0.08$ vs $+0.002$), whereas in `train2` and `test1`, attack flows have *substantially larger* `SAppBytes` (mean $+2.15$ and $+31.59$). Supervised tree splits learned on `train1` naturally misclassify attacks in downstream partitions.
4. **Authoritative Claim Boundary:**
   **This evaluation does NOT claim that DRAEC improves intrusion detection accuracy or minority-class attack recall.** DRAEC provides drift-aware reliability tracking, dynamic offloading, and safe model lifecycle management; the underlying predictive models remain subject to base classifier generalization limits.

---

## 8. Phase 10E: Methodological Limitations & Scientific Disclosures

1. **Label-Informed Window Selection:**
   The evaluation window ($[87,160 : 112,160]$) was selected using ground-truth class information to ensure sufficient representation of minority attack events (80 attacks) in both window halves. Ground-truth labels were not accessible to the runtime controller; however, using label information for offline window selection limits the interpretation of the benchmark as a fully blind streaming evaluation.
2. **Pre-Drift Real-Attack Confound:**
   The pre-drift stream region already contains naturally occurring attack flows and natural concept shifts. The evaluated post-drift performance represents the incremental effect of synthetic drift superimposed upon an imperfect baseline.
3. **Inert Data Quality Axis ($Q_t$):**
   In the WUSTL-IIoT-2021 flow stream, observations arrive with complete network flow records (zero missing cells). `quality=[True]*37` reflects this complete-feature baseline. The sensor-quality axis $Q_t$ was not dynamically exercised in this evaluation ($Q_t \equiv 1.0$); reported reliability reflects confidence, error, and drift dynamics.
4. **Variance Sources & Reproducibility:**
   Due to deterministic construction, exactly 17 of the 18 evaluated metrics in `results/step8_aggregated_summary.csv` have sample standard deviation $s = 0.000000$ across seeds (including $D_t, R_t$, routing percentages, switches, delays). Zero variance arises from deterministic mathematics and fixed input streams, not empirical robustness. Candidate Macro-F1 represents the only stochastically varying quantity ($s = 0.001106$ in Config A, $s = 0.000182$ in Config B).
5. **Configuration & Runtime Parameter Disclosures:**
   - Stratified baseline retraining budget: $200$ samples (194 Class 0, 6 Class 1 per Decision D-038).
   - Effective validator threshold: $0.70$ Macro-F1 per Decision D-041.
   - Retraining random seed: dynamically threaded from run-level seed per Decision D-039.
   - Minimum feedback samples at runtime: $25$ per Decision D-041 Addendum.
6. **Operating Point Scope:**
   The severe drift configuration ($5.0\sigma, n=8$) represents an empirically validated pre-saturation operating point specific to the evaluated setup, not a universal optimum.

---

## 9. Phase 10F: Claim-Evidence Matrix

| Paper Claim | Supporting Artifact | Quantitative Metric / Evidence | Nature of Evidence | Claim Scope | Required Caveat | Supported? |
|---|---|---|:---:|---|---|:---:|
| **Drift-Aware Reliability Tracking** | `step8_aggregated_summary.csv` | $R_t$ drops from $0.9444$ to $0.3238$ under $5\sigma$ drift | Deterministic | Evaluated streaming pipeline | Reflects confidence, error, and drift ($Q_t$ inert) | **SUPPORTED** |
| **Edge-Preserving Routing under Moderate Drift** | `step8_config_a_moderate.csv` | 100.0% Edge execution, $\min R_t = 0.5968 > 0.50$ | Deterministic | Moderate drift ($2\sigma, n=5$) | Does NOT imply attack detection preservation | **SUPPORTED** |
| **Dynamic Cloud Offloading under Severe Drift** | `step8_config_b_severe.csv` | 49.888% Cloud execution, $\min R_t = 0.1261 < 0.30$ | Deterministic | Severe drift ($5\sigma, n=8$) | Offloading driven by reliability collapse | **SUPPORTED** |
| **Fast Change-Point Detection** | `step8_aggregated_summary.csv` | ADWIN delay = 11 steps under $5\sigma$ vs 75 steps under $2\sigma$ | Deterministic | Tested feature-scalar $S(x)$ | Specific to generic scalar formulation | **SUPPORTED** |
| **Closed-Loop Adaptation Triggering** | `step8_combined_runs.csv` | Exactly 2 triggers per run under persistent drift | Deterministic | Tested cooldown & persistence | Triggered by persistence counter (100 steps) | **SUPPORTED** |
| **Safety Firewall Blocks Degraded Models** | `step8_combined_runs.csv` | 10/10 candidates rejected ($0.4168 < 0.70$); 0 deployments | Stochastic / Deterministic Gate | Evaluated validation split (`train2`) | Rejection demonstrates safety gate efficacy | **SUPPORTED** |
| **Active Version Locked at Baseline** | `step8_raw_per_seed_results.json` | Final active version = `v1` across all 10 runs | Deterministic | Evaluated deployer trace | Prevents production model degradation | **SUPPORTED** |
| **Bit-Exact Multi-Seed Reproducibility** | `experiments/verify_step9.py` | 12/12 checks passed; 0 differences on Seed 42 re-run | Deterministic / Stochastic Re-run | Repository artifacts | Fully reproducible via verification script | **SUPPORTED** |
| **Improved Intrusion Detection Accuracy** | `step8_combined_runs.csv` | Post MCC = 0.0000; Attack Recall = 0.0000 | Deterministic | None | Base classifiers detect 0 attacks on test split | **NOT SUPPORTED** |
| **Universal Optimality of $n=8$** | `feature_breadth_sweep_stage3.csv` | $n=8$ pre-saturation; $n=9$ saturation cliff | Empirical sweep | Specific to tested setup | Property of tested scalar and dataset | **NOT SUPPORTED** |
| **Empirical Robustness from Zero Variance** | `step8_aggregated_summary.csv` | Standard deviation = 0.0000 across 17/18 metrics | Deterministic | Mathematical artifact | Arises from deterministic construction | **NOT SUPPORTED** |
