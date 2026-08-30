# Phase 10 — Deterministic WUSTL-IIoT-2021 Temporal Windowing Proposal

**Project:** Drift-Aware Adaptive Edge-Cloud-Hybrid Orchestration for Non-Stationary IoT (DRAEC)  
**Date:** 2026-08-30  
**Phase:** Phase 10 / Step 2 Evaluation Methodology  
**Status:** PROPOSAL & ANALYSIS ONLY (Strict Stop Condition Honored; No Implementation Applied Yet)

---

## 1. Executive Summary

In Phase 10 evaluation, the initial chronological slices:
- `train1[:5000]`
- `train2[:3000]`
- `test1[:1000]`

were shown to be degenerate because attacks in WUSTL-IIoT-2021 do not appear until much later in time (index 223,772 in `train1`, index 158,047 in `train2`, and index 90,596 in `test1`). Consequently, models were trained and evaluated on 100% Class 0, collapsing metrics to trivial artifacts ($\text{Accuracy}=1.0$, $\text{Macro-F1}=1.0$, $\text{MCC}=0.0$).

This document proposes a **deterministic, causal, dataset-driven windowing rule** that:
1. Selects the earliest contiguous window of fixed length $W$ where both the **first half (Pre-Drift)** and the **second half (Post-Drift)** independently contain at least $M$ minority-class (Class 1 / attack) samples.
2. Is strictly independent of downstream model performance, routing behavior, or DRAEC metrics.
3. Preserves chronological causality and partition boundaries (`train1` $\to$ `train2` $\to$ `test1`).
4. Avoids artificial oversampling, undersampling, shuffling, or window cherry-picking.

---

## 2. Dataset Identity & Verified Schema

- **Dataset Name:** WUSTL-IIoT-2021
- **File Path:** [`data/raw/wustl_iiot_2021.csv`](file:///d:/tactics/drift_aware_edge_cloud/data/raw/wustl_iiot_2021.csv)
- **Total Dataset Size:** 1,194,464 rows $\times$ 49 columns (409.80 MB)
- **Target Column:** `Target` (binary integer: 0 = Normal, 1 = Attack)
- **Timestamp Column:** `StartTime` (format: `%Y-%m-%d %H:%M:%S`)
- **Ordering Tie-Breakers:** `['SrcAddr', 'DstAddr', 'Sport', 'Dport', 'Proto', 'sIpId', 'dIpId']`
- **Quarantined Metadata / Leakage Columns (12):** `Target`, `Traffic`, `StartTime`, `LastTime`, `RunTime`, `SrcAddr`, `DstAddr`, `Sport`, `Dport`, `Proto`, `sIpId`, `dIpId`
- **Modeled Features (37):** `Mean`, `SrcPkts`, `DstPkts`, `TotPkts`, `DstBytes`, `SrcBytes`, `TotBytes`, `SrcLoad`, `DstLoad`, `Load`, `SrcRate`, `DstRate`, `Rate`, `SrcLoss`, `DstLoss`, `Loss`, `pLoss`, `SrcJitter`, `DstJitter`, `SIntPkt`, `DIntPkt`, `Dur`, `TcpRtt`, `IdleTime`, `Sum`, `Min`, `Max`, `sDSb`, `sTtl`, `dTtl`, `SAppBytes`, `DAppBytes`, `TotAppByte`, `SynAck`, `sTos`, `SrcJitAct`, `DstJitAct`

---

## 3. Partition Definitions & Overall Class Distribution

Partitions are defined chronologically in [`config/default.yaml`](file:///d:/tactics/drift_aware_edge_cloud/config/default.yaml) by strict inclusive `StartTime` ranges:

| Partition | Role | Selection Time Range | Total Rows | Class 0 (Normal) | Class 1 (Attack) | Class 1 % | First Class 1 Index | Last Class 1 Index |
|---|---|---|---|---|---|---|---|---|
| **`train1`** | `baseline_train` | `2019-08-19 09:46:03` to `11:29:48` | 304,166 | 295,926 | 8,240 | 2.7090% | **Row 223,772** | Row 304,120 |
| **`train2`** | `baseline_validation` | `2019-08-19 11:29:49` to `13:07:36` | 265,685 | 187,380 | 78,305 | 29.4729% | **Row 158,047** | Row 265,684 |
| **`test1`** | `inference_stream` | `2019-08-19 13:07:37` to `16:48:11` | 624,613 | 624,142 | 471 | 0.0754% | **Row 90,596** | Row 269,301 |
| **Total** | Full Dataset | `2019-08-19 09:46:03` to `16:48:11` | **1,194,464** | **1,107,448** | **87,016** | **7.2849%** | Row 223,772 | Row 1,194,463 |

---

## 4. Temporal Attack Density Analysis

Attacks in WUSTL-IIoT-2021 are not uniformly distributed; they occur in episodic bursts late in each recording session:

### `train1` Decile Breakdown ($N = 304,166$)
- Deciles 1–7 (rows 0 to 212,916): **0 attacks (0.000%)**
- Decile 8 (rows 212,916 to 243,332): **900 attacks (2.959%)** (First attack at row 223,772)
- Decile 9 (rows 243,332 to 273,749): **3,119 attacks (10.254%)**
- Decile 10 (rows 273,749 to 304,166): **4,221 attacks (13.877%)**

### `train2` Decile Breakdown ($N = 265,685$)
- Deciles 1–5 (rows 0 to 132,842): **0 attacks (0.000%)**
- Decile 6 (rows 132,842 to 159,411): **872 attacks (3.282%)** (First attack at row 158,047)
- Deciles 7–10 (rows 159,411 to 265,685): **77,433 attacks (72.84%)**

### `test1` Decile Breakdown ($N = 624,613$)
- Decile 1 (rows 0 to 62,461): **0 attacks (0.000%)**
- Decile 2 (rows 62,461 to 124,922): **131 attacks (0.210%)** (First attack at row 90,596)
- Decile 3 (rows 124,922 to 187,383): **128 attacks (0.205%)**
- Decile 4 (rows 187,383 to 249,845): **164 attacks (0.263%)**
- Decile 5 (rows 249,845 to 312,306): **48 attacks (0.077%)** (Last attack at row 269,301)
- Deciles 6–10 (rows 312,306 to 624,613): **0 attacks (0.000%)**

---

## 5. Local Attack Density Around Known Onset (`test1[90596]`)

Inspecting contiguous spans starting from the first attack row in `test1` demonstrates how attacks accumulate:

| Span Length | Row Range | Total Samples | Total Class 1 | Class 1 % | First Half Class 1 | Second Half Class 1 |
|---|---|---|---|---|---|---|
| **1,000** | `[90,596 : 91,596]` | 1,000 | 4 | 0.400% | 3 (0.60%) | 1 (0.20%) |
| **2,000** | `[90,596 : 92,596]` | 2,000 | 7 | 0.350% | 4 (0.40%) | 3 (0.30%) |
| **5,000** | `[90,596 : 95,596]` | 5,000 | 13 | 0.260% | 8 (0.32%) | 5 (0.20%) |
| **10,000** | `[90,596 : 100,596]` | 10,000 | 35 | 0.350% | 13 (0.26%) | 22 (0.44%) |
| **20,000** | `[90,596 : 110,596]` | 20,000 | 72 | 0.360% | 35 (0.35%) | 37 (0.37%) |
| **50,000** | `[90,596 : 140,596]` | 50,000 | 202 | 0.404% | 90 (0.36%) | 112 (0.45%) |

---

## 6. Deterministic Window Selection Rule

### Mathematical Formulation
To eliminate arbitrary window selection, the evaluation window is determined by a deterministic pure function:

$$\text{find\_representative\_window}(y, W, M)$$

where:
- $y \in \{0, 1\}^N$ is the chronologically ordered target label sequence of partition $P$.
- $W \in \mathbb{N}$ is the fixed window size.
- $M \in \mathbb{N}$ is the predetermined minimum minority count required in each half.

A window starting at index $s \in [0, N - W]$ is valid if and only if:

$$\sum_{i=s}^{s + \lfloor W/2 \rfloor - 1} I(y_i = 1) \ge M \quad \text{AND} \quad \sum_{i=s + \lfloor W/2 \rfloor}^{s + W - 1} I(y_i = 1) \ge M$$

The function returns the **earliest** valid starting index:

$$s^* = \min \{ s \in [0, N - W] \mid \text{valid}(s, W, M) \}$$

If no index satisfies the condition, the function strictly returns `None`.

### Why Minority Representation Must Be Satisfied in Both Halves
In the Phase 10 experimental protocol:
- Injected synthetic drift begins at fraction $\alpha = 0.50$ (the exact midpoint of the evaluation stream).
- $[s, s + W/2)$ represents the **Pre-Drift Regime**.
- $[s + W/2, s + W)$ represents the **Post-Drift Regime**.
- If minority samples existed only in the second half, the pre-drift metrics would be single-class ($FN=0, TP=0$).
- If minority samples existed only in the first half, the post-drift metrics would be single-class.
- **Checking both halves guarantees that both Pre-Drift and Post-Drift evaluation periods possess legitimate ground-truth attack samples, ensuring mathematically defined and non-degenerate Macro-F1 and MCC across both regimes.**

---

## 7. Candidate Window Search Results Across Partitions

The pure function was executed across all candidate window sizes $W \in \{5k, 10k, 25k, 50k, 100k\}$ and minority thresholds $M \in \{10, 20, 30, 50, 100\}$:

### Comprehensive Evaluation Matrix

| Partition | Window Size $W$ | Min Req $M$ | Valid? | Start Index | End Index | Total Class 1 | Total Class 1 % | First Half Class 1 | Second Half Class 1 |
|---|---|---|---|---|---|---|---|---|---|
| **`test1`** | 5,000 | 10 | **True** | 95,838 | 100,838 | 23 | 0.460% | 10 | 13 |
| **`test1`** | 5,000 | 20 | **False** | — | — | 0 | 0.000% | 0 | 0 |
| **`test1`** | 5,000 | 30 | **False** | — | — | 0 | 0.000% | 0 | 0 |
| **`test1`** | 10,000 | 10 | **True** | 89,300 | 99,300 | 28 | 0.280% | 10 | 18 |
| **`test1`** | 10,000 | 20 | **True** | 94,886 | 104,886 | 40 | 0.400% | 20 | 20 |
| **`test1`** | 10,000 | 30 | **False** | — | — | 0 | 0.000% | 0 | 0 |
| **`test1`** | 25,000 | 10 | **True** | 81,800 | 106,800 | 53 | 0.212% | 10 | 43 |
| **`test1`** | 25,000 | 20 | **True** | 85,133 | 110,133 | 70 | 0.280% | 20 | 50 |
| **`test1`** | 25,000 | 30 | **True** | **87,160** | **112,160** | **80** | **0.320%** | **30** | **50** |
| **`test1`** | 25,000 | 50 | **True** | 96,141 | 121,141 | 100 | 0.400% | 50 | 50 |
| **`test1`** | 50,000 | 30 | **True** | 74,660 | 124,660 | 129 | 0.258% | 30 | 99 |
| **`test1`** | 100,000 | 30 | **True** | 49,660 | 149,660 | 247 | 0.247% | 30 | 217 |
| **`train1`** | 5,000 | 30 | **True** | 222,016 | 227,016 | 160 | 3.200% | 30 | 130 |
| **`train1`** | 10,000 | 30 | **True** | 219,516 | 229,516 | 298 | 2.980% | 30 | 268 |
| **`train1`** | 25,000 | 30 | **True** | **212,016** | **237,016** | **686** | **2.744%** | **30** | **656** |
| **`train2`** | 5,000 | 30 | **True** | 155,577 | 160,577 | 1,592 | 31.840% | 30 | 1,562 |
| **`train2`** | 10,000 | 30 | **True** | 153,077 | 163,077 | 3,080 | 30.800% | 30 | 3,050 |
| **`train2`** | 25,000 | 30 | **True** | **145,577** | **170,577** | **7,101** | **28.404%** | **30** | **7,071** |

---

## 8. Minority-Class Criterion Justification

### Why $M = 30$?
1. **Central Limit Theorem & Normal Approximation:** In binary classification evaluation, contingency table cells with $N < 30$ exhibit high variance and skewness. A single misclassification with $N = 10$ alters recall by $10\%$ ($0.10$), creating metric instability. $M \ge 30$ ensures statistical stability for contingency table estimators (Precision, Recall, F1, MCC).
2. **Empirical Feasibility in WUSTL-IIoT-2021:**
   - In `test1`, the entire partition has only 471 attacks.
   - For $W = 5,000$, achieving $M \ge 20$ in both halves is mathematically impossible (the maximum achievable is 10).
   - For $W = 10,000$, achieving $M \ge 30$ in both halves is mathematically impossible (the maximum achievable is 20).
   - For $W = 25,000$, achieving $M \ge 30$ in both halves is **fully satisfied** at index $87,160$ ($30$ in first half, $50$ in second half, $80$ total).

---

## 9. Windowing Options for User Alignment

### Recommended Option: Statistical Rigor ($W = 25,000, M = 30$)
- **`train1` (Baseline Train):** `[212,016 : 237,016]` ($N = 25,000$, 686 Class 1, 2.74%)
- **`train2` (Baseline Validation):** `[145,577 : 170,577]` ($N = 25,000$, 7,101 Class 1, 28.40%)
- **`test1` (Evaluation Stream):** `[87,160 : 112,160]` ($N = 25,000$, 80 Class 1, 0.32%)
  - Pre-Drift ($t = 0 \dots 12,500$): **30 Class 1 samples**
  - Drift Onset ($t = 12,500$): Injected perturbation starts at midpoint
  - Post-Drift ($t = 12,500 \dots 25,000$): **50 Class 1 samples**
- **Pros:** Full mathematical stability ($M \ge 30$ in both halves); ample post-drift stream (12,500 steps) for ADWIN detection, persistence, adaptation cooldown, and recovery observation.
- **Runtime:** ~2.5 minutes per streaming simulation run ($\times 5$ seeds = ~12.5 minutes).

### Alternative Option: Compact Stream Window ($W = 10,000, M = 20$)
- **`train1` (Baseline Train):** `[219,516 : 229,516]` ($N = 10,000$, 298 Class 1, 2.98%)
- **`train2` (Baseline Validation):** `[153,077 : 163,077]` ($N = 10,000$, 3,080 Class 1, 30.80%)
- **`test1` (Evaluation Stream):** `[94,886 : 104,886]` ($N = 10,000$, 40 Class 1, 0.40%)
  - Pre-Drift ($t = 0 \dots 5,000$): **20 Class 1 samples**
  - Drift Onset ($t = 5,000$): Injected perturbation starts at midpoint
  - Post-Drift ($t = 5,000 \dots 10,000$): **20 Class 1 samples**
- **Pros:** Faster execution (~60 seconds per seed); perfectly balanced attacks across both halves ($20$ and $20$).
- **Cons:** $M = 20$ in each half is slightly below the classical $N \ge 30$ asymptotic rule of thumb, though $40$ total in the window.

---

## 10. Scientific Integrity Verification

1. **Zero Peeking / No Performance Optimization:**
   The window search algorithm evaluated **only** label arrays $y$. Neither model predictions ($\hat{y}$), model probabilities ($p_0, p_1$), accuracy, F1, MCC, reliability scores, nor ADWIN alarms were consulted at any point in this analysis.
2. **Causal Ordering Preserved:**
   All partitions and windows maintain strict chronological flow:
   `train1` ($09:46 \dots 11:29$) $\to$ `train2` ($11:29 \dots 13:07$) $\to$ `test1` ($13:07 \dots 16:48$). No observations are shuffled or reordered.
3. **No Artificial Data Generation:**
   Zero synthetic attack samples were manufactured; all labels are authentic ground truth from WUSTL-IIoT-2021.

---

## 11. Proposed Architectural Decision (to be appended to `DECISIONS.md`)

```markdown
## D-030 · 2026-08-30 · decision · Deterministic Temporal Windowing for WUSTL-IIoT-2021 Phase 10

**Context.** In Phase 10, initial chronological window slices (train1[:5000], train2[:3000], test1[:1000]) contained 100% Class 0 because attacks in WUSTL-IIoT-2021 appear late (index 223,772 in train1, 158,047 in train2, 90,596 in test1). This produced degenerate single-class metrics (F1=1.0, MCC=0.0).

**Decision.**
1. Adopt a deterministic, dataset-driven pure function `find_representative_window(y, W, M)` that identifies the earliest contiguous window where both the first half (pre-drift) and the second half (post-drift) independently satisfy `Class1 >= M`.
2. Model performance, predictions, routing, and adaptation were strictly prohibited from influencing window selection.
3. Preserves chronological causality (train1 -> train2 -> test1).
```

---

**STRICT STOP CONDITION HONORED:** No source code or Phase 10 evaluation windows have been modified. Awaiting review and approval before proceeding to implementation.
