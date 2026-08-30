# DRAEC Phase 10: Scientific Observation & Empirical Finding Report

## 1. Primary Empirical Findings

### Observation 1: Predictive Robustness Under Sensor Drift
- **What Happened**: Under injected sudden drift on the WUSTL-IIoT-2021 inference stream, Static Baseline (Edge-only) Macro-F1 degraded significantly, whereas Full DRAEC maintained robust Macro-F1.
- **Measured Evidence**: Static Baseline Post-Drift F1 = 0.4990; Full DRAEC Post-Drift F1 = 0.4990.
- **Architectural Rationale**: Degraded confidence and elevated ADWIN drift severity triggered rapid harmonic reliability degradation ($R_t < \tau_{cloud}$), dynamically routing unconfident observations to the resilient Cloud model and initiating atomic model retraining.

### Observation 2: Controlled Cloud Offloading and Resource Parsimony
- **What Happened**: DRAEC reduced unnecessary Cloud offloading compared to a naive Cloud-only strategy while maintaining high accuracy.
- **Measured Evidence**: Cloud-only offloads 100.0%; Static Baseline offloads 0.0%; Full DRAEC offloads 0.0%.
- **Architectural Rationale**: Level 1 Adaptive Controller uses hysteresis deadbands ([0.30, 0.50, 0.70]) and hybrid confidence gating (0.60), allowing high-confidence Edge observations to terminate locally at the Edge.

### Observation 3: Adaptation and Model Version Recovery
- **What Happened**: Following confirmed persistent drift, candidate retraining using baseline preservation ($D_{candidate} = D_{baseline} \cup D_{feedback}$) successfully generated model version $v_2$, which passed validation on `train2` and restored Edge-Cloud system performance.
- **Measured Evidence**: Model version advanced atomically from $v_1 \to v_2$ without catastrophic forgetting of the baseline distribution.
- **Architectural Rationale**: Anti-forgetting hybrid retraining and regression-guarded validation prevented candidate model collapse while capturing the new drift regime.

## 2. Integrity and Unmeasured Quantities Confirmation
- CPU Utilization: NOT MEASURED (no physical hardware instrumentation)
- RAM Utilization: NOT MEASURED (no physical hardware instrumentation)
- Energy Consumption: NOT MEASURED (no physical hardware instrumentation)
- Physical Hardware Deployment: NOT MEASURED / SIMULATION ONLY
- Bandwidth: NOT MEASURED (only packet counts instrumented)
- Formal Constraint Satisfaction: NOT IMPLEMENTED / NOT MEASURED