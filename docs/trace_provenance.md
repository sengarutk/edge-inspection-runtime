# Physical Telemetry Trace Provenance & Sensor Simulation Governance

## 1. Executive Summary & Nomenclature Policy
This document formalizes the data provenance, preprocessing pipelines, and experimental classification for all physical telemetry streams evaluated in **Edge Quality Intelligence** (Flagship 4).

To maintain scientific integrity, strict methodological honesty, and publication transparency:
- **No Unsubstantiated Claims:** Evaluated physical telemetry streams must **never** be described as "real-world factory telemetry" or "live production floor sensor recordings."
- **Standardized Classification:** 
  1. **NASA IMS Bearing Stream:** Classified as a **`NASA IMS-informed synthetic bearing-degradation trace`** (calibrated against run-to-failure accelerometer characteristics from the NASA Intelligent Maintenance Systems bearing dataset).
  2. **NASA C-MAPSS Turbofan Stream:** Classified as **`C-MAPSS-informed synthetic turbofan-degradation trace`** (calibrated against simulated run-to-failure degradation trajectories from the NASA Commercial Modular Aero-Propulsion System Simulation benchmark).
  3. **Operational Edge Sensor Simulation:** Classified as a **`physics-inspired first-order sensor simulator`** (based on discrete Newtonian thermal dissipation, dynamic mechanical load scaling, and Gaussian observation noise, rather than empirically fitted plant machinery).

---

## 2. NASA IMS Bearing Degradation Trace

### 2.1 Scientific Classification
* **Dataset Attribution:** Inspired by and calibrated against the NASA Prognostics Center of Excellence IMS Bearing dataset (Rexnord ZA-2115 double-row bearings run to catastrophic failure).
* **Provenance Category:** `NASA IMS-informed synthetic bearing-degradation trace`.
* **Execution Paradigm:** Deterministic offline trace replay through `PhysicalTraceReplayer`.

### 2.2 Telemetry Specifications & Channels
* **Sampling Rate:** $30\,\mathrm{Hz}$ ($\Delta t = 33.333\,\mathrm{ms}$ frame cycle), synchronized 1:1 with camera inspection cadence.
* **Physical Signals Monitored:**
  1. `vibration_rms`: Standardized root-mean-square acceleration ($g$), tracking radial bearing housing vibration.
  2. `temperature_c`: Bearing housing surface temperature ($^{\circ}\mathrm{C}$), modeling secondary frictional heating.
  3. `current_amps`: Motor armature current ($A$), tracking electrical load under mechanical friction.

### 2.3 Degradation Horizon & Failure Onset Protocol
* **Total Trace Horizon:** 600 discrete cycles ($20.0\,\mathrm{s}$ at $30\,\mathrm{FPS}$).
* **Warmup & Baseline Calibration:**
  - $T_{\mathrm{warm}} = 50$ samples ($1.67\,\mathrm{s}$): Baseline reference window used to compute empirical running mean $\mu_c$ and variance $\sigma_c^2$.
* **Degradation Phases:**
  - *Phase 1 (Healthy Operation, steps 0–240):* Vibration centered at $\mu = 0.35\,g$ ($\sigma = 0.02\,g$); temperature at $48.0^{\circ}\mathrm{C}$; current at $10.5\,\mathrm{A}$.
  - *Phase 2 (Incipient Defect Onset, steps 240–420):* Sub-surface fatigue micro-spalling. Progressive vibration rise ($0.35\,g \to 0.80\,g$), temperature rises $+6.0^{\circ}\mathrm{C}$.
  - *Phase 3 (Severe Fault Runaway, steps 420–600):* Inner race defect spalling. Exponential runaway from $0.80\,g \to 2.85\,g$, temperature reaches $76.0^{\circ}\mathrm{C}$, current escalates to $16.5\,\mathrm{A}$.
* **Ground Truth Fault Onset:** Quantitative failure threshold occurs at **Step 420** ($vibration \ge 0.80\,g$). Alerts emitted prior to step 420 are classified as False Alarms ($FA$).

---

## 3. NASA C-MAPSS Turbofan Degradation Trace

### 3.1 Scientific Classification
* **Dataset Attribution:** Calibrated against degradation trajectories of the NASA Commercial Modular Aero-Propulsion System Simulation (C-MAPSS) benchmark (FD001 sub-regime: Sea level, single operating condition).
* **Provenance Category:** `C-MAPSS-informed synthetic turbofan-degradation trace`.
* **Execution Paradigm:** Deterministic offline trace replay through `PhysicalTraceReplayer`.

### 3.2 Telemetry Specifications & Channels
* **Sampling Rate:** $30\,\mathrm{Hz}$ ($\Delta t = 33.333\,\mathrm{ms}$).
* **Physical Signals Monitored:**
  1. `vibration_rms`: Shaft and casing vibration RMS ($g$).
  2. `temperature_c`: High-Pressure Compressor (HPC) outlet temperature ($^{\circ}\mathrm{C}$).
  3. `current_amps`: Actuator and pump motor current ($A$).

### 3.3 Degradation Horizon & Failure Onset Protocol
* **Total Trace Horizon:** 600 discrete cycles ($20.0\,\mathrm{s}$ at $30\,\mathrm{FPS}$).
* **Warmup & Baseline Calibration:**
  - $T_{\mathrm{warm}} = 50$ samples ($1.67\,\mathrm{s}$): Baseline variance calibration window.
* **Degradation Phases:**
  - *Phase 1 (Nominal Operation, steps 0–210):* Thermal baseline $52.0^{\circ}\mathrm{C}$, vibration $0.42\,g$, current $11.5\,\mathrm{A}$.
  - *Phase 2 (Incipient Wear, steps 210–390):* HPC blade erosion with gradual thermal creep ($52.0^{\circ}\mathrm{C} \to 66.0^{\circ}\mathrm{C}$).
  - *Phase 3 (Severe Degradation, steps 390–600):* Thermal creep runaway ($66.0^{\circ}\mathrm{C} \to 88.5^{\circ}\mathrm{C}$), current escalation to $24.8\,\mathrm{A}$, casing vibration increases to $1.32\,g$.
* **Ground Truth Fault Onset:** Severe degradation threshold occurs at **Step 390** ($T \ge 66.0^{\circ}\mathrm{C}, I \ge 15.5\,\mathrm{A}$).

---

## 4. Physics-Inspired First-Order Sensor Simulator

The runtime telemetry generator (`src/runtime/sensor_simulator.py`) is governed by:
1. **Newtonian Thermal Inertia:**
   $$\bar{T}_t = \bar{T}_{t-1} + k_{\mathrm{thermal}} \cdot \left( T_{\mathrm{target}}(M_t) - \bar{T}_{t-1} \right) + \mathrm{drift\_rate} \cdot \Delta t$$
   where $k_{\mathrm{thermal}} = 0.08$, $\mathrm{drift\_rate} = 2.5^{\circ}\mathrm{C}/\mathrm{hr}$, and $T_{\mathrm{target}}$ depends on operational machine state $M_t$.
2. **Dynamic Machine State Scaling:**
   Mechanical vibration and electrical current scale with load factor: $\text{IDLE} \to 0.15\times$, $\text{RUNNING} \to 1.0\times$, $\text{FAULT} \to 3.5\times$.
3. **Additive Stochastic Noise:**
   Independent zero-mean Gaussian observation noise $\mathcal{N}(0, \sigma_c^2)$ added per channel.
4. **Z-Score Mapping & Saturation:**
   Composite score mapped into $[0.0, 1.0]$ via weighted excess over threshold $Z_{\mathrm{thresh}} = 0.70$ and exponential saturation $s_t = 1 - e^{-s_{\mathrm{raw}}}$.
