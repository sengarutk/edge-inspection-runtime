# Edge Inspection Runtime: Multi-Modal Temporal Policy Gating and Durable Event Spooling

[![Tests](https://img.shields.io/badge/pytest-190%20passed-brightgreen.svg)](tests/)
[![Paper](https://img.shields.io/badge/IEEE%20Format-4%20Pages%20Camera--Ready-blue.svg)](docs/paper/main.tex)
[![Artifact](https://img.shields.io/badge/Release-v0.4.0-orange.svg)](https://github.com/sengarutk/edge-inspection-runtime/releases/tag/v0.4.0)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)

An industrial cyber-physical edge inspection runtime designed to mitigate **operator alert fatigue**, handle **cross-modal sensory divergence**, and eliminate **telemetry evidence loss** during edge network partitions.

This repository hosts the complete, reproducible artifact for the research paper:
> **"Reducing Alert Fatigue in Industrial Edge Inspection Through Multi-Modal Temporal Policy Gating and Durable Event Spooling"**  
> *Utkarsh Sengar, Department of Computer Science and Engineering, Indian Institute of Technology Dharwad*

---

## 🏗️ System Architecture

The runtime executes a high-throughput priority cascade processing visual anomaly streams ($224 \times 224$) and physical sensor frames (tri-axial accelerometry, temperature):

```text
┌────────────────┐     ┌─────────────────────────────┐
│  Video Stream  │ --->│ Optical Health Verification │ --- (Blur Flag) ---> [Optical Degradation]
└────────────────┘     └──────────────┬──────────────┘
                                      │ (Valid Frames)
                                      ▼
                               ┌───────────────┐
                               │   PatchCore   │ ===> Visual Score (v_t)
                               └───────────────┘        │
┌────────────────┐     ┌─────────────────────────────┐  │
│ Physical Sensor│ --->│   Telemetry Preprocessing   │ ===> Phys Score (s_t)
└────────────────┘     └─────────────────────────────┘  │
                                                        ▼
┌────────────────────────────────────────────────────────────────────────┐
│                   Multi-Modal Temporal Policy Gating                   │
│                                                                        │
│ 1. Dual Exponential Moving Averages (EMA): v̄_t (α=0.35), s̄_t (α=0.25)  │
│ 2. Operational Machine-State Gating (IDLE / MAINTENANCE Suppression)  │
│ 3. Incident Refractory Cooldown FSM (T_cool = 15 frames)               │
│ 4. k-of-N Sliding Window Confirmation (k=4, N=10 exceedances)          │
│ 5. Cross-Modal Divergence Triage (|v̄_t - s̄_t| ≥ 0.45)                  │
└───────────────────────────────────┬────────────────────────────────────┘
                                    │
                    ┌───────────────┴───────────────┐
                    ▼                               ▼
      [Critical Defect Escalation]     [Operator Review Queue]
                    │                               │
                    └───────────────┬───────────────┘
                                    ▼
                     ┌───────────────────────────────┐
                     │ SQLite Write-Ahead-Log Spooler│ (FIFO Bounded Queue)
                     └───────────────┬───────────────┘
                                     ▼
                        [MQTT Broker / Cloud Sync]
```

---

## 📊 Key Experimental Findings

### 1. Multi-Modal Policy Ablation (Table I)
Evaluated across 8 policy variants and 6 standardized industrial workloads ($N_{\mathrm{seeds}} = 3$, $B = 2{,}000$ bootstrap resamples):
- **Transient Glitches:** Complete false-alarm elimination ($120.0 \to 0.0$ FA/hr, 100.0% suppression).
- **Sustained Defects:** Nuisance re-alerts reduced from $180.0 \to 12.0$ alerts/hr (93.3% reduction) with bounded 3.0-frame delay while maintaining 100.0% actionable incident routing recall ($\Delta_{\max} = 15$ frames).
- **Queue Backlog Stability:** Under an $M/M/1$ review model ($\mu = 60$ reviews/hr), `FULL_POLICY` bounds operator triage utilization to $\rho = 0.20$ ($W_q = 0.25$ min). Baselines enter mathematically unstable queue regimes ($\lambda \ge 60, \rho \ge 1.0$) with unbounded backlog growth.

### 2. Spooler Partition Resilience & Durability
Under simulated 30-second broker disconnections, the SQLite WAL spooler (`synchronous=NORMAL`, MQTT QoS 1) recorded:
- **0 missing persisted records**
- **0 duplicate persisted records**
- **0 sequence-order violations**
- **0 queue overflows** across 120 generated records (peak depth 120 vs 50,000 limit).
- Post-reconnection drain recovery completed in $< 0.45$ seconds.

### 3. Execution Latency
- **Core Gating & Inference Path:** $p95 = 8.8\,\mathrm{ms}$ (parameterized on $224 \times 224$ inputs with forward-pass delay $8.20 \pm 0.25\,\mathrm{ms}$).
- **End-to-End Pipeline:** Mean $10.13\,\mathrm{ms}$ ($0.0\%$ deadline misses against the $33.333\,\mathrm{ms}$ / 30-FPS line cycle target).

---

## 🔬 Scientific Boundaries & Transparency

To ensure full peer-review rigor:
- **Synthetic Telemetry Proxies:** Offline run-to-failure trace evaluations use mathematical degradation proxies informed by NASA IMS bearing and C-MAPSS turbofan data (600 steps, 20.0 s at 30 FPS). They validate interface compatibility and policy logic rather than factory generalization.
- **Storage Durability Model:** Durability refers to file-backed retention during broker network partitions and controlled client process restarts; it does not model host power-loss, physical disk destruction, or filesystem corruption.
- **Queue Eviction:** The spooler enforces a strict FIFO bounded-queue retention policy when queue capacity is reached.
- **Latency Measurement:** Latency benchmarks isolate policy, telemetry fusion, and local disk spooling overheads; physical camera driver I/O and external broker transit latency are excluded.

---

## ⚡ Quickstart & 1-Command Reproducibility

### Prerequisites
- Linux / WSL2 (Ubuntu 22.04+)
- Python 3.10+
- TeX Live 2023+ (`pdflatex`, `bibtex`)

### Setup Environment
```bash
git clone https://github.com/sengarutk/edge-inspection-runtime.git
cd edge-inspection-runtime
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Full Test Suite (190 Tests)
```bash
pytest tests/ -v
```

### End-to-End Paper Reproduction Pipeline
Regenerate all calibration summaries, stress benchmarks, LaTeX macros, and build the camera-ready 4-page PDF with a single script:
```bash
bash scripts/reproduce_all_paper_results.sh
```
The compiled PDF will be located at `docs/paper/main.pdf`.

---

## 📂 Repository Structure

```text
├── configs/            # Runtime and scenario YAML configurations
├── data/               # Standardized workload traces and NASA proxy datasets
├── docs/
│   └── paper/          # LaTeX manuscript, figures, macros, and references
│       ├── figures/    # Vector PDF figures (Pareto, Queue, Attribution)
│       ├── generated_metrics.tex # Auto-generated macro single-source-of-truth
│       ├── main.tex    # Camera-ready 4-page IEEEtran manuscript
│       └── references.bib # BibTeX bibliographic entries
├── results/            # Empirical JSON results and latency profiles
├── scripts/
│   ├── benchmark_latency.py           # Phase 4: 5000-cycle latency profiler
│   ├── benchmark_spooler_resilience.py# Phase 2: Broker partition stress test
│   ├── evaluate_multimodal_truth_table.py # Phase 3: Truth table diagnostic
│   ├── run_real_trace_benchmark.py    # Phase 1: NASA proxy trace replay
│   ├── validate_paper_claims.py       # Phase 5: Single-source-of-truth validator
│   └── reproduce_all_paper_results.sh # Phase 7: Master reproduction pipeline
├── src/                # Production runtime source code
│   └── runtime/        # Policy gating, spooler, and MQTT bridge
└── tests/              # 190 pytest unit and integration tests
```

---

## 📜 Citation

```bibtex
@inproceedings{sengar2026reducing,
  author    = {Sengar, Utkarsh},
  title     = {Reducing Alert Fatigue in Industrial Edge Inspection Through Multi-Modal Temporal Policy Gating and Durable Event Spooling},
  booktitle = {Proceedings of the IEEE/ACM Workshop on Edge Computing and Industrial Systems},
  year      = {2026},
  url       = {https://github.com/sengarutk/edge-inspection-runtime}
}
```

---

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.
