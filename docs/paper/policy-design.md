# Temporal Decision Policies & Cross-Modal Risk Engine Design

## 1. Theoretical Foundations & Problem Statement

In industrial visual inspection, single-frame deep learning inference suffers from two catastrophic failure modes:
1. **High False Positive Rates (Alert Fatigue)**: Transient lighting flickers, minor dust particles, or camera sensor jitter cause instantaneous confidence spikes that trip factory emergency stops if unmitigated.
2. **Silent Hardware Failures**: Optical defocus, camera lens smudge, or loose sensor wiring cause the vision or sensor model to emit deceptively low anomaly scores while a physical machine breakdown is underway.

The **Temporal Policy Engine** resolves both issues via multi-stage temporal filtering, sliding-window confirmation, anti-fatigue state latching, and cross-modal correlation.

---

## 2. Mathematical Formulations

### 2.1 Exponential Moving Average (EMA) Smoothing
Instantaneous noisy scores $v_t$ (vision) and $s_t$ (physical sensors) are smoothed to suppress high-frequency noise while preserving true drift:

$$v_{\text{ema}, t} = \alpha_v v_t + (1 - \alpha_v) v_{\text{ema}, t-1}, \quad \text{with } \alpha_v = 0.35$$

$$s_{\text{ema}, t} = \alpha_s s_t + (1 - \alpha_s) s_{\text{ema}, t-1}, \quad \text{with } \alpha_s = 0.25$$

At $t=0$, $v_{\text{ema}, 0} = v_0$ and $s_{\text{ema}, 0} = s_0$.

### 2.2 Sliding-Window $k$-of-$N$ Confirmation Filter
To distinguish persistent surface defects from transient 1-2 frame glitches, the engine tracks threshold exceedances over an $N$-frame FIFO queue:

$$E_t = \mathbb{I}(v_{\text{ema}, t} \ge \tau_{\text{high}}), \quad \mathcal{H}_N = [E_{t-N+1}, \dots, E_t]$$

$$\text{Confirmed}_{\text{vision}} = \left( \sum_{i=1}^N \mathcal{H}_N[i] \ge k \right), \quad \text{where } N=10, k=4$$

A visual defect is only escalated to `HIGH_SEVERITY` if confirmed in at least $k=4$ frames within window $N=10$.

### 2.3 Cross-Modal Divergence Metric
The cross-modal divergence $\Delta_{\text{modal}}$ isolates sensor glitches from multi-modal faults:

$$\Delta_{\text{modal}, t} = |v_{\text{ema}, t} - s_{\text{ema}, t}|$$

If $\Delta_{\text{modal}, t} \ge \tau_{\text{div}} = 0.45$ and an anomaly is present, the engine safely flags `CROSS_MODAL_DISCREPANCY` with `REVIEW_REQUIRED`, preventing an unverified catastrophic lockout.

---

## 3. Anti-Fatigue Cooldown State Machine

Once a `HIGH_SEVERITY` escalation is emitted, the engine initializes a cooldown counter $C = 15$ steps. During cooldown:
- Repeat `HIGH_SEVERITY` alerts are suppressed to prevent operator alarm storms.
- Ongoing anomalous conditions route to `REVIEW_REQUIRED` (`COOLDOWN_ACTIVE`).
- The counter decrements on each tick until $C = 0$.

```
       +------------------+
       |      NORMAL      |<-------------------------+
       +------------------+                          |
         |              |                            |
         | (k-of-N met) | (Degradation / Divergence) | (C = 0 & Nominal)
         v              v                            |
+---------------+     +-----------------+            |
| HIGH_SEVERITY |     | REVIEW_REQUIRED |            |
+---------------+     +-----------------+            |
         |                      ^                    |
         | Set C = 15           |                    |
         v                      |                    |
+---------------------------------------+            |
|            COOLDOWN_ACTIVE            |------------+
|        (Alert Storm Suppressed)       |
+---------------------------------------+
```

---

## 4. Operational State Gating Matrix

| Machine Operational State | Visual Condition | Sensor Condition | Assigned RiskState | TriggerReason Code |
| :--- | :---: | :---: | :---: | :---: |
| **RUNNING** | $v_{\text{ema}} < 0.50$ | $s_{\text{ema}} < 0.70$ | `NORMAL` | `NOMINAL_OPERATION` |
| **RUNNING** | $v_{\text{ema}} \ge 0.80$ ($k \ge 4$) | $s_{\text{ema}} \ge 0.70$ | `HIGH_SEVERITY` | `MULTI_MODAL_CONFIRMED_FAULT` |
| **RUNNING** | $v_{\text{ema}} \ge 0.80$ ($k \ge 4$) | $s_{\text{ema}} < 0.70$ | `HIGH_SEVERITY` | `SUSTAINED_VISION_ANOMALY` |
| **RUNNING** | $v_{\text{ema}} < 0.50$ | $s_{\text{ema}} \ge 0.70$ | `REVIEW_REQUIRED` | `SUSTAINED_SENSOR_ANOMALY` |
| **RUNNING** | $\Delta_{\text{modal}} \ge 0.45$ | $v_{\text{ema}} \ge 0.50$ | `REVIEW_REQUIRED` | `CROSS_MODAL_DISCREPANCY` |
| **IDLE** | Any Anomaly | Any Telemetry | `REVIEW_REQUIRED` / `NORMAL` | `STATE_GATED_SUPPRESSION` |
| **MAINTENANCE** | Any Anomaly | Any Telemetry | `REVIEW_REQUIRED` / `NORMAL` | `STATE_GATED_SUPPRESSION` |
| **FAULT** | Any | Any | `HIGH_SEVERITY` | `CRITICAL_MACHINE_FAULT` |