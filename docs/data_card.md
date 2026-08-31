# Dataset & Scenario Workload Data Card

## 1. Overview & Dataset Governance
This data card documents the 6 standardized benchmark workload scenarios, synthetic physics transfer functions, sensor normalization calibrations, and score generation mechanics used in the **Industrial Edge Inspection Runtime & Reliability System**.

* **Dataset Version**: `1.0.0`
* **Zero Test Leakage Guarantee**: All threshold parameters ($	au_{\text{high}}, \tau_{\text{med}}, k, N, C, \tau_{\text{div}}$) are fixed pre-deployment design constants documented in [`configs/threshold_manifest.json`](file:///home/sengar/edge-inspection-runtime/configs/threshold_manifest.json). No test scenario labels or ground-truth evaluations were used to tune decision thresholds.
* **Sampling Cadence**: 30 FPS visual acquisition (33.333 ms frame period) synchronized with 30 Hz multi-channel physical telemetry.

---

## 2. Standardized Scenario Workloads

| Scenario ID | Primary Stress Type | Ground Truth Defect Steps | Injected Chaos Faults | Expected Failure Mode / Mitigation |
|---|---|---|---|---|
| `sustained_defects` | Structural surface fracture | Steps 30--70 (40 frames) | None | Multi-modal confirmation with cooldown latching |
| `transient_glitches` | Optical specular reflections / dust | None (pure false positives) | Optical glare spikes (steps 20, 50, 80) | 100% suppression via $k$-of-$N$ temporal filter |
| `optical_degradation` | Lens contamination & defocus | Steps 40--60 (20 frames) | Gaussian blur ($\sigma=45$) at step 35 | Automated fallback to physical sensor telemetry |
| `sensor_drift_dropout` | Thermal runaway & lead detach | None | +35°C thermal drift, current dropout | Zero-order hold (ZOH) imputation & cross-modal divergence |
| `network_partitions` | Edge-to-cloud broker disconnect | Steps 25--45 (20 frames) | TCP link disconnect at step 20 (60s) | Local SQLite WAL disk spooling (zero loss) |
| `distribution_shift` | Process speed variation | Steps 20--35, 70--85 | Machine state transitions (`IDLE`, `FAULT`) | Operational state-gated alarm suppression |

---

## 3. Synthetic Physical Modeling & Transfer Functions

### 3.1 Visual Degradation Metric (Laplacian Variance)
Optical clarity is computed via discrete 2D Laplace operator $\nabla^2 I$:
$$\text{Var}(\nabla^2 I) = \frac{1}{HW} \sum_{x,y} \left( \nabla^2 I(x,y) - \mu_{\nabla^2 I} \right)^2$$
A frame is flagged as optically degraded whenever $\text{Var}(\nabla^2 I) < 100.0$.

### 3.2 Physical Sensor Telemetry & $Z$-Score Normalization
Multi-modal sensor channels (3-axis vibration $v_{\text{RMS}}$, motor temperature $T$, and armature current $I$) are normalized against baseline distributions:
$$Z_c = \frac{x_c - \mu_c}{\sigma_c}, \quad s_t = \text{clip}\left( \frac{1}{|C|} \sum_{c \in C} \max(0, Z_c - Z_{\text{base}}), 0.0, 1.0 \right)$$

### 3.3 Thermal Dynamics
Motor temperature evolution follows Newton's Law of Cooling under resistive load:
$$\frac{dT}{dt} = \frac{I^2 R}{C_{\text{thermal}}} - k_{\text{dissipation}} (T - T_{\text{ambient}})$$

---

## 4. Ethical Considerations & Operator Well-being
The benchmark suite measures **Operator Overload Fraction** based on a strict cognitive threshold ($60.0\text{ reviews/hour}$). All policy designs prioritize suppressing repetitive alarm fatigue while maintaining high True Positive Rates ($> 95\%$).
