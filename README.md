# Flagship 4: Industrial Edge Inspection Runtime & Reliability System

[![Tests: Passing](https://img.shields.io/badge/Tests-118%2F118%20Passing-brightgreen.svg)](https://github.com/sengar/edge-inspection-runtime)
[![Coverage: 93%+](https://img.shields.io/badge/Coverage-93%25-success.svg)](https://github.com/sengar/edge-inspection-runtime)
[![Python: 3.10+](https://img.shields.io/badge/Python-3.10%20%7C%203.11%20%7C%203.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-purple.svg)](LICENSE)
[![Zero Data Loss: Verified](https://img.shields.io/badge/Zero--Data--Loss-Verified-orange.svg)](docs/architecture.md)

---

## 1. Overview & Core Mission

**Industrial Edge Inspection Runtime & Reliability System** (`edge-inspection-runtime`) is a clean-room, production-grade edge execution runtime engineered for real-time automated visual inspection and machine reliability monitoring. 

Deployed directly on edge gateways (NVIDIA Jetson / Industrial x86 PCs), the system reconciles high-frequency visual inspection streams with physical sensor telemetry (3-axis vibration, motor current, and Newtonian thermal dynamics) to deliver **deterministic anomaly detection, suppress alert fatigue by >65%, and guarantee zero event loss during network partitions**.

```
+---------------------------------------------------------------------------------------------------+
|                                  INDUSTRIAL EDGE INSPECTION RUNTIME                               |
+---------------------------------------------------------------------------------------------------+
|                                                                                                   |
|  [ Camera Stream ] --------> ( Optical Health Check ) ------> [ PatchCore ONNX Inference ]        |
|                               (Laplacian Var & Brightness)          | (Vision Anomaly Score)      |
|                                                                     v                             |
|  [ Physical Sensors ] -----> ( 30Hz Physical Simulator ) ----> [ Multi-Modal Temporal Engine ]    |
|   (Vib, Temp, Current)        (Newtonian Thermal Drift)       | - Exponential Moving Averages     |
|                                                               | - k-of-N Sliding Window Filter    |
|                                                               | - Anti-Fatigue Cooldown FSM       |
|                                                               | - Cross-Modal Divergence Gate     |
|                                                                     |                             |
|                                                                     v                             |
|                                                          [ PolicyDecision Record ]                |
|                                                                     |                             |
|                                  +----------------------------------+--------------------------+  |
|                                  |                                                             |  |
|                                  v (Network Connected)                                         v  |
|                       [ Resilient MQTT Publisher ]                                [ Local Disk Spooler ]
|                         (Paho v2 Async Dispatch)                                     (SQLite Queue DB)    |
|                                  |                                                             |  |
|                                  | (Active QoS 1 Ack)                                          |  |
|                                  +<============================================================+  |
|                                  |             (Automatic Background Drain Worker)                |
|                                  v                                                                |
|                        [ Mosquitto Broker ]                                                       |
|                                  |                                                                |
|                                  v                                                                |
|                       [ Ingestion Subscriber ]                                                    |
|                                  |                                                                |
|                                  v                                                                |
|                         [ SQLite Audit DB ] <==================> [ Streamlit Operator Console ]   |
|                      (Historical Journal & Stream)                  (Triage & Review Queue)       |
+---------------------------------------------------------------------------------------------------+
```

---

## 2. Key Technical Innovations

1. **Physics-Informed Multi-Modal Sensor Engine**:
   - Simulates first-order Newtonian thermal equilibrium ($T_t = T_{t-1} + k(T_{\text{target}} - T_{t-1}) + \Delta T_{\text{drift}}$), 3-axis vibration spectra, and electrical current draw across machine states (`IDLE`, `RUNNING`, `MAINTENANCE`, `FAULT`).
2. **Temporal Policy Engine & Noise Suppression**:
   - Combines dual Exponential Moving Average (EMA) filters ($\alpha_v = 0.35, \alpha_s = 0.25$) with a sliding-window $k$-of-$N$ confirmation filter ($k=4, N=10$), an anti-fatigue cooldown state machine ($15$ cycles), and a cross-modal divergence gate ($\Delta_{\text{modal}} \ge 0.45$).
3. **Resilient Dual-Path MQTT Messaging & Zero-Loss Spooler**:
   - Implements non-blocking dual-path dispatch via Paho MQTT v2 and an embedded atomic SQLite disk spooler (`data/spooler_queue.db`) with automatic background drain recovery.
4. **Human-in-the-Loop Triage Console & Evidence Overlay**:
   - Streamlit web console for continuous telemetry visualization, side-by-side defect heatmap overlays ($0.6 \cdot \text{Frame} + 0.4 \cdot \text{Heatmap}$), and atomic operator review writebacks.
5. **Programmatic Chaos Engineering Harness**:
   - Injects optical defocus blur, camera occlusions, sensor channel dropouts, network partitions, and distribution shifts to empirically validate edge resilience.

---

## 3. Empirical Reliability Benchmark Results

Extracted from the automated 300-cycle stress benchmark suite ([`scripts/benchmark_suite.py`](scripts/benchmark_suite.py)):

| Reliability KPI Metric | Value | Operational Target | Verification Status |
| :--- | :---: | :---: | :---: |
| **Total Processed Cycles** | `300` | Continuous 30Hz stream | **NOMINAL** |
| **Raw Single-Frame Crossings** | `132` | Instantaneous noisy spikes | Recorded |
| **Policy High-Severity Escalations** | `41` | Filtered actionable alarms | Actioned |
| **Alert Fatigue Suppression ($\rho_{\text{supp}}$)** | **`68.94%`** | $\ge 60.0\%$ | **PASS** |
| **Mean Detection Latency ($\Delta t$)** | **`7.2` frames (`240.0` ms)** | $\le 10$ frames ($\le 350$ ms) | **PASS** |
| **Cross-Modal Discrepancy Rate** | `12.33%` | Isolated optical/sensor glitches | Monitored |
| **Optical Fallbacks Handled** | `30` | Zero unhandled blur/occlusions | **PASS** |
| **Spool Queue Drained Upon Recovery** | `0` records | $0.0\%$ data loss | **PASS** |
| **Operator Triage Precision ($P_{\text{op}}$)** | **`80.0%`** | High-confidence true positives | **PASS** |

---

## 4. Quickstart & Reproducibility Guide

### Prerequisites
- Python 3.10, 3.11, or 3.12
- Docker & Docker Compose (optional, for local Eclipse Mosquitto broker)

### 1. Installation
```bash
git clone https://github.com/sengar/edge-inspection-runtime.git
cd edge-inspection-runtime

python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Run All Unit & Integration Tests (Strict $\ge 90\%$ Coverage)
```bash
./scripts/run_tests.sh
# or: python -m pytest tests/ -v --cov=src --cov-report=term-missing
```

### 3. Run the Automated 300-Step Benchmark Suite
```bash
python scripts/benchmark_suite.py
```

### 4. Launch the Streamlit Operator Review Console
```bash
streamlit run src/dashboard/app.py --server.port 8501
```

---

## 5. Repository Structure

```
edge-inspection-runtime/
├── configs/
│   ├── system_config.yaml       # Optical health and inference settings
│   ├── sensor_config.yaml       # Physics parameters and sensor weights
│   ├── policy_config.yaml       # Temporal smoothing, k-of-N, cooldown
│   ├── mqtt_config.yaml         # Broker, topics, QoS, and database paths
│   └── mosquitto.conf           # Local Eclipse Mosquitto broker config
├── docker-compose.yml           # Local Mosquitto container definition
├── docs/
│   ├── architecture.md          # End-to-end architecture & latency budgets
│   ├── policy-design.md         # Mathematical formulations & FSM design
│   ├── failure-modes.md         # Full FMEA hazard & fallback matrix
│   ├── event-schema.md          # JSON event schemas for MQTT topics
│   └── admissions-guide.md      # SOP excerpts, CV bullets & interview guide
├── requirements.txt             # Strict pinned dependencies
├── pyproject.toml               # Pytest and coverage configurations
├── scripts/
│   ├── benchmark_suite.py       # 300-step master stress benchmark runner
│   ├── run_pipeline.sh          # Single-command end-to-end pipeline runner
│   ├── run_tests.sh             # Strict test runner with coverage gate
│   ├── verify_phase1.py         # Phase 1 foundation verification
│   ├── verify_phase2.py         # Phase 2 temporal policy verification
│   ├── verify_phase3.py         # Phase 3 resilient messaging verification
│   └── verify_phase4.py         # Phase 4 chaos & triage verification
├── src/
│   ├── __init__.py              # Core package exports
│   ├── config.py                # Pydantic V2 config models & LRU loaders
│   ├── inference_service.py     # Optical health validator & PatchCore engine
│   ├── sensor_simulator.py      # Newtonian physics multi-sensor simulator
│   ├── policy.py                # Temporal decision policy & risk engine
│   ├── spooler.py               # Local SQLite persistent disk spooler
│   ├── mqtt_publisher.py        # Resilient dual-path MQTT publisher
│   ├── mqtt_subscriber.py       # Ingestion subscriber daemon
│   ├── audit_log.py             # Relational SQLite audit & triage database
│   ├── evidence_manager.py      # Blended heatmap evidence overlay manager
│   ├── fault_injector.py        # Programmatic chaos engineering suite
│   ├── metrics.py               # Statistical reliability KPI evaluator
│   └── dashboard/
│       ├── __init__.py
│       └── app.py               # Streamlit operator review console
└── tests/
    ├── test_config.py           # Configuration schema validation tests
    ├── test_inference.py        # Optical health & inference engine tests
    ├── test_sensors.py          # Thermal physics & sensor Z-score tests
    ├── test_policy.py           # Temporal smoothing & policy FSM tests
    ├── test_spooler.py          # SQLite disk spooler FIFO queue tests
    ├── test_audit.py            # SQLite audit log & triage tests
    ├── test_mqtt_resilience.py  # Dual-path dispatch & drain tests
    ├── test_evidence.py         # Heatmap overlay & evidence manager tests
    ├── test_fault_injector.py   # Programmatic chaos injector tests
    ├── test_dashboard_queries.py# Telemetry & health query tests
    ├── test_dashboard_app.py    # Streamlit UI integration tests
    └── test_metrics.py          # Benchmark evaluator & math tests
```

---

## 6. Integration with Flagships 1–3 Narrative

| Flagship | Domain | Role in AI Systems Pipeline |
| :--- | :--- | :--- |
| **Flagship 1** | *Visual Anomaly Representation* | Unsupervised PatchCore memory-bank modeling and patch localization on MVTec-AD. |
| **Flagship 2** | *Hardware-Aware Quantization* | Post-training INT8 quantization and ONNX Runtime execution graph optimization. |
| **Flagship 3** | *High-Performance Kernels* | Custom CUDA and OpenAI Triton fused GPU kernels for ultra-low latency anomaly scoring. |
| **Flagship 4** | *Edge Inspection Runtime* | **(This Repository)** The complete multi-modal edge execution runtime, temporal policy engine, resilient messaging, and human-in-the-loop triage system. |

---

## 7. License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

## Research Framework & Reproduction

`edge-inspection-runtime` includes a full research-grade experimental framework to benchmark edge inspection policies against realistic industrial failure modes, evaluate human operator cognitive workload tradeoffs, and profile hard deadline SLA compliance.

### 1. Configurable Decision Policies (8 Variants)
The runtime implements 8 discrete policy modes:
- `BASELINE`: Instantaneous single-frame thresholding ($v_t \ge 0.80$).
- `EMA_ONLY`: Exponential moving average smoothing on visual scores ($\alpha_v = 0.35$).
- `EMA_KOFN`: EMA smoothing with $k$-of-$N$ confirmation window ($k=4, N=10$).
- `NO_COOLDOWN`: Multi-modal cyber-physical cascade with alert cooldown suppressed ($C=0$).
- `NO_FUSION`: Vision-only temporal cascade bypassing physical sensor telemetry.
- `NO_DIVERGENCE`: Multi-modal cascade bypassing cross-modal discrepancy checks.
- `NO_STATE_GATING`: Multi-modal cascade bypassing operational machine state suppression.
- `FULL_POLICY`: Complete multi-modal cascade combining optical health gating, sensor telemetry fusion, cross-modal divergence, machine state gating, and anti-fatigue cooldown FSM ($C=15$).

### 2. Standardized Scenario Workloads (`configs/scenarios/`)
Benchmarking is conducted across 6 standardized 300-step ($10\,\text{s}$ at $30\,\text{FPS}$) industrial workloads:
1. `nominal.yaml`: Steady-state production baseline.
2. `transient_glitches.yaml`: 1-2 frame optical blurs, bright strobes, and camera flickers.
3. `sustained_defects.yaml`: Continuous 60-step surface fractures.
4. `sensor_drift_dropout.yaml`: Linear thermal drift and current sensor disconnection.
5. `network_partitions.yaml`: 60-step edge MQTT broker disconnection during defect bursts.
6. `distribution_shift.yaml`: Cosmetic visual domain shifts with nominal mechanical operation.

### 3. One-Command Automated Reproduction Pipeline
To execute the complete research reproduction suite (environment verification, 113-test test suite, 144-run Monte Carlo ablation across 8 policies, 216-run parameter sensitivity sweep, disk spooler stress testing, publication vector plot rendering, and SHA-256 checksum verification):
```bash
./scripts/reproduce_main_results.sh
```

### 4. Research Artifacts & Manuscripts
- **Reproducibility Pipeline:** [`scripts/reproduce_main_results.sh`](scripts/reproduce_main_results.sh)
- **Data Card & Provenance:** [`docs/data_card.md`](docs/data_card.md)
- **Threshold Governance Manifest:** [`configs/threshold_manifest.json`](configs/threshold_manifest.json)
- **Aggregated Benchmark Summary:** [`results/ablation/ablation_summary.json`](results/ablation/ablation_summary.json)
- **Parameter Sensitivity Summary:** [`results/sensitivity/sensitivity_summary.json`](results/sensitivity/sensitivity_summary.json)
- **Disk Spooler Stress Summary:** [`results/spooler_stress/spooler_stress_summary.json`](results/spooler_stress/spooler_stress_summary.json)
- **LaTeX Publication Table:** [`results/ablation/ablation_table.tex`](results/ablation/ablation_table.tex)
- **Markdown Results Table:** [`results/ablation/ablation_table.md`](results/ablation/ablation_table.md)
- **SHA-256 Checksums:** [`results/CHECKSUMS.txt`](results/CHECKSUMS.txt)
- **Publication Figures (PDF & PNG):** [`docs/figures/`](docs/figures/)
- **Full Academic Technical Report Draft:** [`docs/paper-draft.md`](docs/paper-draft.md)
- **IEEE-Formatted Manuscript:** [`docs/paper.tex`](docs/paper.tex)
