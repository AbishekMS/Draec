# Phase 10 — Read-Only Root-Cause Diagnostic Report

**Project:** Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT (DRAEC)  
**Date:** 2026-08-30  
**Status:** READ-ONLY DIAGNOSIS COMPLETE (Strict Stop Condition Honored; No Source Code, Config, or Phase 1–9 Logic Modified)

---

## Executive Summary

This diagnostic investigates why the current Phase 10 empirical evaluation on the WUSTL-IIoT-2021 dataset yielded degenerate metrics ($\text{Accuracy}=1.0$, $\text{Macro-F1}=1.0$, $\text{MCC}=0.0$, $\text{ADWIN detections}=0$, $\text{Drift Severity } D_t=1.0$, $\text{Reliability } R_t \approx 4.0 \times 10^{-8}$, $\text{Cloud Offloading}=100\%$, $\text{Adaptation Events}=0$, $\text{Packet Loss}=0$, and $p=1.0$).

Through systematic read-only code and data tracing, the degeneration was traced to **three independent root causes**:
1. **Temporal Label Stratification & Slicing:** The WUSTL-IIoT-2021 dataset is a temporal flow log where attack events are heavily clustered late in time. In `train1`, the first 223,771 rows are 100% normal (Class 0). In `test1`, the first 90,595 rows are 100% normal (Class 0). Phase 10 evaluated a truncated window (`stream_steps = 1000` or `500`), and trained models on `X_train[:5000]`. Consequently, models trained on 100% Class 0 and evaluated on 100% Class 0, yielding trivial 100% F1 and undefined/zero MCC.
2. **ADWIN & Severity Signal Inversion:** In [`src/metrics/evaluation.py`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L292), the drift detector was passed `drift_val = max(p0, p1)` (model prediction confidence) rather than posterior class probability or prediction error. Because the model predicted Class 0 with confidence $\approx 1.0$ everywhere, ADWIN observed zero change in mean ($1.0 \to 1.0$), resulting in 0 alarms. Simultaneously, severity was updated with $|1.0 - 0.5| \times 2.0 = 1.0$, permanently pinning $D_t = 1.0$ from observation 0, collapsing harmonic reliability $R_t \to 4.0 \times 10^{-8}$ and forcing 100% Cloud routing.
3. **Network Telemetry Provenance Stripping:** In [`src/deployment/environment.py`](file:///d:/tactics/drift_aware_edge_cloud/src/deployment/environment.py#L331), `to_execution_result()` converts `DeploymentExecutionResult` to `ExecutionResult`, dropping the `packet_lost` attribute. In [`evaluation.py`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L571), delivered transmissions were counted as `l is not None` on `network_latency_s`, which was `0.0` on failure, incorrectly counting failed packets as delivered.

---

## A. Dataset Identity

- **Actual Dataset File Loaded:** [`data/raw/wustl_iiot_2021.csv`](file:///d:/tactics/drift_aware_edge_cloud/data/raw/wustl_iiot_2021.csv)
- **Actual File Path:** `d:/tactics/drift_aware_edge_cloud/data/raw/wustl_iiot_2021.csv`
- **File Size:** 409,800,698 bytes (390.82 MB)
- **Dataframe Shape (Complete Raw File):** 1,194,464 rows $\times$ 49 columns
- **Dataset Confirmation:** Confirmed as **genuine WUSTL-IIoT-2021**. The file contains standard network flow telemetry headers (`StartTime`, `LastTime`, `SrcAddr`, `DstAddr`, `Mean`, `Sport`, `Dport`, `SrcPkts`, `DstPkts`, `TotPkts`, `DstBytes`, `SrcBytes`, `TotBytes`, `SrcLoad`, `DstLoad`, `Load`, `SrcRate`, `DstRate`, `Rate`, `SrcLoss`, `DstLoss`, `Loss`, `pLoss`, `SrcJitter`, `DstJitter`, `SIntPkt`, `DIntPkt`, `Proto`, `Dur`, `TcpRtt`, `IdleTime`, `Sum`, `Min`, `Max`, `sDSb`, `sTtl`, `dTtl`, `sIpId`, `dIpId`, `SAppBytes`, `DAppBytes`, `TotAppByte`, `SynAck`, `RunTime`, `sTos`, `SrcJitAct`, `DstJitAct`, `Traffic`, `Target`).
- **HAI / SWaT Cross-Check:** No HAI sensors (`P1_FCV01D`, `P3_LCV01D`, etc.) or SWaT tags are present in `wustl_iiot_2021.csv`.
- **Configured Partitions in [`config/default.yaml`](file:///d:/tactics/drift_aware_edge_cloud/config/default.yaml):**
  - `train1` (`baseline_train`): 304,166 rows (`selection_time_range: ["2019-08-19 09:46:03", "2019-08-19 11:29:48"]`)
  - `train2` (`baseline_validation`): 265,685 rows (`selection_time_range: ["2019-08-19 11:29:49", "2019-08-19 13:07:36"]`)
  - `test1` (`inference_stream`): 624,613 rows (`selection_time_range: ["2019-08-19 13:07:37", "2019-08-19 16:48:11"]`)
  - Total partition sum: $304,166 + 265,685 + 624,613 = 1,194,464$ rows (100% of raw dataset, zero rows omitted).

---

## B. Actual Schema

- **Total Columns in CSV:** 49
- **Label Column:** `Target` (binary integer: 0 = Normal, 1 = Attack)
- **12 Quarantined Leakage Columns:**  
  `Target`, `Traffic`, `StartTime`, `LastTime`, `RunTime`, `SrcAddr`, `DstAddr`, `Sport`, `Dport`, `Proto`, `sIpId`, `dIpId`.
- **37 Modeled Continuous/Discrete Features:**  
  `Mean`, `SrcPkts`, `DstPkts`, `TotPkts`, `DstBytes`, `SrcBytes`, `TotBytes`, `SrcLoad`, `DstLoad`, `Load`, `SrcRate`, `DstRate`, `Rate`, `SrcLoss`, `DstLoss`, `Loss`, `pLoss`, `SrcJitter`, `DstJitter`, `SIntPkt`, `DIntPkt`, `Dur`, `TcpRtt`, `IdleTime`, `Sum`, `Min`, `Max`, `sDSb`, `sTtl`, `dTtl`, `SAppBytes`, `DAppBytes`, `TotAppByte`, `SynAck`, `sTos`, `SrcJitAct`, `DstJitAct`.
- **Schema Alignment:** Perfectly matches the repository's expected 37-feature WUSTL-IIoT representation.

---

## C. Actual Label Distribution

| Partition | Total Rows | Class 0 (Normal) | Class 1 (Attack) | Positive Rate | First Class 1 Index in Partition |
|---|---|---|---|---|---|
| **Entire Dataset** | 1,194,464 | 1,107,448 | 87,016 | 7.285% | Row 223,772 |
| **`train1`** | 304,166 | 295,926 | 8,240 | 2.709% | **Row 223,772** |
| **`train2`** | 265,685 | 187,380 | 78,305 | 29.473% | **Row 158,047** |
| **`test1`** | 624,613 | 624,142 | 471 | 0.075% | **Row 90,596** |
| **Phase 10 Training Slice (`train1[:5000]`)** | 5,000 | 5,000 | **0** | **0.000%** | N/A (Single-Class) |
| **Phase 10 Validation Slice (`train2[:3000]`)** | 3,000 | 3,000 | **0** | **0.000%** | N/A (Single-Class) |
| **Phase 10 Test Slice (`test1[:1000]`)** | 1,000 | 1,000 | **0** | **0.000%** | N/A (Single-Class) |
| **Phase 10 Test Slice (`test1[:500]`)** | 500 | 500 | **0** | **0.000%** | N/A (Single-Class) |

### Determination:
**Finding C: Both `y_true` and `y_pred` are single-valued.**
The evaluation subset `test1[:1000]` contains **zero attack instances**. Because attacks only begin after row 90,596 in `test1`, evaluating only the first 500 or 1,000 rows tests a trivial 100% negative stream. Furthermore, the models were trained on `train1[:5000]`, which also contains **zero attack instances**.

---

## D. Prediction Distribution

- **`y_true.value_counts()`:**
  - Class 0: 1,000 (100.0%)
  - Class 1: 0 (0.0%)
- **`y_pred.value_counts()`:**
  - Class 0: 1,000 (100.0%)
  - Class 1: 0 (0.0%)
- **Total Evaluation Samples:** 1,000
- **Unique `y_true` classes:** 1 (`{0}`)
- **Unique `y_pred` classes:** 1 (`{0}`)

---

## E. Confusion Matrix

$$\begin{pmatrix} \text{TN} & \text{FP} \\ \text{FN} & \text{TP} \end{pmatrix} = \begin{pmatrix} 1000 & 0 \\ 0 & 0 \end{pmatrix}$$

- **True Negatives (TN):** 1,000
- **False Positives (FP):** 0
- **False Negatives (FN):** 0
- **True Positives (TP):** 0

### Mathematical Derivation of Metrics:
- $\text{Accuracy} = \frac{1000 + 0}{1000} = 1.0000$ (100%)
- $\text{Precision} = \frac{\text{TN}}{\text{TN} + \text{FN}} = \frac{1000}{1000} = 1.0000$
- $\text{Recall} = \frac{\text{TN}}{\text{TN} + \text{FP}} = \frac{1000}{1000} = 1.0000$
- $\text{Macro-F1} = 1.0000$
- $\text{MCC} = \frac{\text{TP} \cdot \text{TN} - \text{FP} \cdot \text{FN}}{\sqrt{(\text{TP}+\text{FP})(\text{TP}+\text{FN})(\text{TN}+\text{FP})(\text{TN}+\text{FN})}} = \frac{0}{\sqrt{0}} \xrightarrow{\text{scikit-learn}} 0.0$

The Matthews Correlation Coefficient is mathematically zero because the denominator is zero when predictions or targets are constant.

---

## F. Drift Injection Verification

Drift was injected via [`src/data/generator.py:inject()`](file:///d:/tactics/drift_aware_edge_cloud/src/data/generator.py#L477) under the `sudden` scenario with requested magnitude $2.0 \sigma$:
- **Affected Features Selected by `top_variance`:**  
  `('Load', 'SrcLoad', 'TotAppByte', 'TotBytes', 'IdleTime')`
- **Onset Index:** Row 250 (in 500-step test) / Row 500 (in 1000-step test)
- **Target Stream:** `test1` (inference stream)
- **Measured Empirical Statistics:**

| Feature | Changed Rows | Pre-Drift Mean | Post-Drift Mean | Requested $\sigma$ | Realised $\sigma$ | Status |
|---|---|---|---|---|---|---|
| `Load` | 250 / 250 | 3,438,827.43 | 110,496,944.97 | 2.0 | 2.0000 | **Shifted** |
| `SrcLoad` | 250 / 250 | 3,334,800.36 | 108,880,138.03 | 2.0 | 2.0000 | **Shifted** |
| `TotAppByte` | 250 / 250 | 56.30 | 73,588,468.20 | 2.0 | 2.0000 | **Shifted** |
| `TotBytes` | 250 / 250 | 1,058.98 | 61,817,313.33 | 2.0 | 2.0000 | **Shifted** |
| `IdleTime` | 0 / 250 | 1,548,788,864.00 | 1,548,788,864.00 | 2.0 | 0.0000 | Clipped to Range |

- **Verification Outcome:** The drift injection **actually modifies 4 of the 5 features** with full $2.0\sigma$ shift. It is **not** a silent no-op. `IdleTime` was unshifted solely because physical range clipping bounded the epoch timestamp channel.

---

## G. ADWIN Input Diagnostic

In [`src/metrics/evaluation.py` lines 291–294](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L291-L294):
```python
# 1. Drift Detection
drift_val = max(p0, p1)
adwin_flag = detector.update(drift_val)
```

- **Exact Signal Consumed:** `drift_val = max(p0, p1)`
- **True Label $y_t$:** $0$ for all $t \in [0, 1000]$
- **Predicted Probability:** $p_0 \approx 1.0, p_1 \approx 0.0$ for all $t \in [0, 1000]$
- **Signal Value:** `drift_val` $= 1.0$ constantly across the entire stream.
- **Trace Around Drift Point ($t=250$):**
  - $t = 248$: `p0=1.0000, p1=0.0000, drift_val=1.0000, error=0.0, adwin_alarm=False`
  - $t = 249$: `p0=1.0000, p1=0.0000, drift_val=1.0000, error=0.0, adwin_alarm=False`
  - $t = 250$ (Onset): `p0=1.0000, p1=0.0000, drift_val=1.0000, error=0.0, adwin_alarm=False`
  - $t = 251$: `p0=1.0000, p1=0.0000, drift_val=1.0000, error=0.0, adwin_alarm=False`
  - $t = 252$: `p0=1.0000, p1=0.0000, drift_val=1.0000, error=0.0, adwin_alarm=False`
- **Why ADWIN Failed to Alarm:** ADWIN detects shifts in the *running mean* of its input signal. Because the Hoeffding tree was trained on pure Class 0, it maintained 100% confidence in Class 0 even after features shifted. Thus, the mean of `drift_val` was flat at $1.0$. ADWIN saw variance = 0 and delta = 0, mathematically preventing any Hoeffding bound cut.

---

## H. Reliability Component Diagnostic ($C_t, E_t, D_t, Q_t, R_t$)

| Component | Initial | Min | Max | Pre-Drift Mean | Post-Drift Mean | Delta | Pinned State | Cause |
|---|---|---|---|---|---|---|---|---|
| **$C_t$ (Confidence)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Pinned at 1.0 | $2 \times (\max(1.0, 0.0) - 0.5) = 1.0$ |
| **$E_t$ (Error)** | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | 0.0000 | Pinned at 0.0 | No errors occurred ($y=\hat{y}=0$), AND feedback was never routed to `reliability_est` |
| **$D_t$ (Drift Severity)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Pinned at 1.0 | `abs(drift_val - 0.5)*2.0` evaluated with `drift_val=1.0` yields $1.0$ from $t=0$ |
| **$Q_t$ (Quality)** | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 0.0000 | Pinned at 1.0 | Hard-coded `quality=[True]*37` |
| **$R_t$ (Reliability)** | $4.00 \times 10^{-8}$ | $4.00 \times 10^{-8}$ | $4.00 \times 10^{-8}$ | $4.00 \times 10^{-8}$ | $4.00 \times 10^{-8}$ | 0.0000 | Pinned at $4.0 \times 10^{-8}$ | Weakest-link harmonic mean denominator dominated by $\frac{0.25}{1 - D_t + 10^{-8}} = 2.5 \times 10^7$ |

### Mathematical Collapse of $R_t$:
$$R_t = \frac{1}{\frac{0.25}{C_t + \epsilon} + \frac{0.25}{1 - E_t + \epsilon} + \frac{0.25}{1 - D_t + \epsilon} + \frac{0.25}{Q_t + \epsilon}}$$
Because $D_t = 1.0$, the third denominator term evaluates to:
$$\frac{0.25}{1 - 1 + 10^{-8}} = \frac{0.25}{10^{-8}} = 25,000,000$$
$$R_t = \frac{1}{0.25 + 0.25 + 25,000,000 + 0.25} \approx \frac{1}{2.500000075 \times 10^7} \approx 3.99999988 \times 10^{-8}$$

---

## I. Routing Diagnostic

- **Thresholds in `AdaptiveController`:**
  - $\tau_{\text{critical}} = 0.30$
  - $\tau_{\text{cloud}} = 0.50$
  - $\tau_{\text{return}} = 0.70$
- **$R_t$ Distribution:**
  - $R_t \ge 0.70$: **0**
  - $0.50 \le R_t < 0.70$: **0**
  - $0.30 \le R_t < 0.50$: **0**
  - $R_t < 0.30$: **1,000 (100.0%)**
- **Actions Executed:**
  - **EDGE:** 0 (0.0%)
  - **HYBRID:** 0 (0.0%)
  - **CLOUD:** 1,000 (100.0%)
- **Conclusion:** Zero Hybrid invocations and 100% Cloud offloading are **100% direct downstream consequences** of $R_t = 4.0 \times 10^{-8} < \tau_{\text{critical}}$.

---

## J. Phase 9 Adaptation Diagnostic

- **Gate 1: Persistent Drift Confirmed?**  
  **FAIL (False).** Because ADWIN detections were 0, persistence streak counter never incremented.
- **Gate 2: Severity $\ge \tau_{\text{severity}} (0.30)$?**  
  **PASS (True).** $D_t = 1.0 \ge 0.30$.
- **Gate 3: Eligible Feedback $\ge N_{\text{min}} (25)$?**  
  **PASS (True).** 485 delayed feedback records arrived and queued in `FeedbackQueue`.
- **Gate 4: Cooldown Elapsed?**  
  **PASS (True).** Initial state is ready.
- **Conclusion:** Adaptation was inhibited **exclusively by Gate 1**.

---

## K. Network Failure Diagnostic

In [`results/network_metrics.csv`](file:///d:/tactics/drift_aware_edge_cloud/results/network_metrics.csv), `packet_loss` and `disconnected` both reported 300 transmissions, 300 delivered, loss_rate = 0.0.

### Complete Cause Trace:
1. **Simulator Execution:** Under `NetworkSimulator(available=False)`, `transmit()` returned `TransmissionResult(success=False, status=DISCONNECTED, latency_s=0.0, packet_lost=False)`. Under `packet_loss_probability=0.05`, `transmit()` dropped 10 of 300 packets with `success=False, packet_lost=True, latency_s=0.0`.
2. **Runtime Packaging:** `DeploymentEnvironment.execute_cloud()` created a `DeploymentExecutionResult` with `success=False` and `packet_lost=True`.
3. **Data Loss Point 1:** In [`src/deployment/environment.py:execute()`](file:///d:/tactics/drift_aware_edge_cloud/src/deployment/environment.py#L331):
   ```python
   return dep_res.to_execution_result()
   ```
   `DeploymentExecutionResult` was cast to standard `ExecutionResult` ([`src/decision/base.py`](file:///d:/tactics/drift_aware_edge_cloud/src/decision/base.py#L173)). **`ExecutionResult` has no `packet_lost` field!** The packet loss flag was silently dropped.
4. **Data Loss Point 2:** In [`src/metrics/evaluation.py` lines 569–574](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L569-L574):
   ```python
   delivered_transmissions=len([l for l in sim_run["latencies_network"] if l is not None]),
   packet_loss_count=sum(1 for er in sim_run["execution_results"] if getattr(er, "packet_lost", False)),
   ```
   - `getattr(er, "packet_lost", False)` always returned `False` because `er` is an `ExecutionResult`.
   - On transmission failure, `network_latency_s` was set to `0.0`. Because in Python `0.0 is not None` evaluates to `True`, all 300 transmissions (including failed and disconnected) were tallied as `delivered = 300`.

---

## L. Statistical Interpretation

- **Reported Value:** $p = 1.0$ across all paired tests.
- **Cause:** In [`src/metrics/evaluation.py` lines 707–710](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L707-L710):
  ```python
  diff = sub1[:n] - sub2[:n]
  if np.allclose(diff, diff[0], atol=1e-12):
      test_name = "Exact zero-variance paired difference"
      p_val = 1.0
  ```
  Every evaluated method (`FULL_DRAEC`, `STATIC_BASELINE`, `DRAEC_WITHOUT_ADAPTATION`) scored **identically 1.0000 post-drift Macro-F1 across all 5 random seeds**. The difference vector was exactly `[0.0, 0.0, 0.0, 0.0, 0.0]`. With zero variance, Student's t-distribution is undefined and the safety fallback set $p = 1.0$.
- **Interpretation:** $p = 1.0$ is an artifact of the single-class test slice, **not** evidence of scientific equivalence between methods.

---

## M. Root-Cause Classification

| Metric / Phenomenon | Classification | Explanation |
|---|---|---|
| **1. 100% Accuracy / F1** | **DOWNSTREAM SYMPTOM** | Caused by single-class evaluation slice (`test1[:1000]` has 0 attacks) and single-class training slice (`train1[:5000]` has 0 attacks). |
| **2. MCC = 0.0** | **DOWNSTREAM SYMPTOM** | Mathematical property of MCC when predictions or targets are constant. |
| **3. ADWIN Detections = 0** | **ROOT CAUSE** | ADWIN monitored `drift_val = max(p0, p1) = 1.0` (constant confidence), exhibiting 0 mean shift. |
| **4. $D_t = 1.0$** | **ROOT CAUSE** | Severity was calculated as $|1.0 - 0.5| \times 2.0 = 1.0$, saturating drift severity at 1.0 from $t=0$. |
| **5. $R_t \approx 4.0 \times 10^{-8}$** | **DOWNSTREAM SYMPTOM** | Direct mathematical consequence of $D_t = 1.0$ entering harmonic mean denominator ($2.5 \times 10^7$). |
| **6. Hybrid Invocations = 0** | **DOWNSTREAM SYMPTOM** | $R_t = 4\times 10^{-8} < \tau_{\text{critical}} (0.30)$ forces direct Cloud routing without testing Edge confidence. |
| **7. Cloud Offloading = 100%** | **DOWNSTREAM SYMPTOM** | Direct consequence of collapsed $R_t$. |
| **8. Adaptation Events = 0** | **DOWNSTREAM SYMPTOM** | Gate 1 requires `is_persistent_drift == True`, which requires ADWIN alarms. |
| **9. Network Packet Loss = 0** | **INDEPENDENT ISSUE** | `packet_lost` attribute dropped in `to_execution_result()`; `evaluation.py` checked `getattr(er, "packet_lost", False)`. |
| **10. Disconnection Failures = 0** | **INDEPENDENT ISSUE** | Disconnected latency `0.0` was counted as delivered because `0.0 is not None`. |
| **11. $p = 1.0$** | **DOWNSTREAM SYMPTOM** | Paired difference vector is zero-variance `[0.0, 0.0, 0.0, 0.0, 0.0]`. |

---

## N. Recommended Fixes (Read-Only Proposal)

### Fix 1: Non-Stationary Temporal Evaluation Windowing
- **File / Module:** [`src/models/trainer.py`](file:///d:/tactics/drift_aware_edge_cloud/src/models/trainer.py), [`src/metrics/evaluation.py`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py)
- **Suspected Issue:** `train1[:5000]` and `test1[:1000]` slice the very beginning of the chronological log, where zero attacks exist. Attacks only start at row 223,772 in `train1` and row 90,596 in `test1`.
- **Evidence:** `Counter(y_train[:5000]) = {0: 5000}`, `Counter(y_test1[:1000]) = {0: 1000}`.
- **Proposed Correction:**
  1. Train models on a representative partition that spans both normal and attack regimes (e.g. from row 220,000 onward in `train1`, or using a stratified baseline slice).
  2. Evaluate streaming simulation on a stream window that spans both classes (e.g. starting around row 90,000 in `test1`, or windowing 5,000 steps across the attack transition).
- **Affected Phase:** Phase 10 / Model Training & Evaluation.

### Fix 2: ADWIN Monitored Signal Correction
- **File / Module:** [`src/metrics/evaluation.py:run_streaming_simulation()`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L292)
- **Suspected Issue:** ADWIN receives `drift_val = max(p0, p1)` instead of its designed Phase 3 interface.
- **Evidence:** `detector.update(max(p0, p1))` feeds a constant $1.0$.
- **Proposed Correction:** Call `detector.update_from_prediction(prob_dict)` or pass Class 1 posterior probability `probs[1]` (or prediction error $e_t$ when available), matching the contract in [`src/drift/adwin_detector.py`](file:///d:/tactics/drift_aware_edge_cloud/src/drift/adwin_detector.py#L187).
- **Affected Phase:** Phase 10 evaluation harness.

### Fix 3: Drift Severity Calculation Alignment
- **File / Module:** [`src/metrics/evaluation.py`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py#L302)
- **Suspected Issue:** Severity updated with `abs(drift_val - 0.5) * 2.0`. When `drift_val = 1.0`, severity is permanently $1.0$.
- **Evidence:** $D_t = 1.0$ from observation 0 pre-drift.
- **Proposed Correction:** Consume Phase 3 `DriftPipeline` or compute drift severity from distribution distance between windowed predictions/features and the frozen baseline profile mean, as defined in [`src/drift/severity.py`](file:///d:/tactics/drift_aware_edge_cloud/src/drift/severity.py).
- **Affected Phase:** Phase 10 evaluation harness.

### Fix 4: Network Telemetry & Execution Result Preservation
- **File / Module:** [`src/deployment/base.py`](file:///d:/tactics/drift_aware_edge_cloud/src/deployment/base.py), [`src/decision/base.py`](file:///d:/tactics/drift_aware_edge_cloud/src/decision/base.py), [`src/metrics/evaluation.py`](file:///d:/tactics/drift_aware_edge_cloud/src/metrics/evaluation.py)
- **Suspected Issue:** `packet_lost` is omitted from `ExecutionResult`; `evaluation.py` checks `l is not None` on `0.0`.
- **Evidence:** `packet_lost` flags are dropped; failed transmissions are counted as delivered.
- **Proposed Correction:**
  1. Add `packet_lost: bool = False` to `ExecutionResult` and propagate it in `to_execution_result()`.
  2. In `evaluation.py`, count delivered transmissions as `sum(1 for er in sim_run["execution_results"] if er.success)` and only record non-zero or valid network latencies when delivery succeeds.
- **Affected Phase:** Phase 8 / Phase 10.

---

## Conclusion & Strict Stop Verification

All 11 diagnostic points have been empirically traced to their origin. In accordance with the prompt's strict stop conditions:
- **Zero source files were modified.**
- **Zero configurations or thresholds were altered.**
- **No full experimental suites were re-run.**
- **Awaiting user review of this diagnostic before proceeding to any corrective action.**
