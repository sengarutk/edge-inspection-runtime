# Multi-Modal Temporal Policy Gating and Spooled Resilience for Real-Time Industrial Edge Inspection Runtime

## Technical Report & Research Manuscript Draft

**Authors**: Edge Systems Research Group  
**Affiliation**: Autonomous Edge Computing & Reliability Lab, Industrial IoT Systems Division  
**Contact**: `research@edge-inspection-runtime.org`  

---

### Abstract
Visual anomaly detection models deployed on industrial edge devices suffer from transient false alarms, high operator cognitive burden, and data loss during intermittent network partitions. We present an industrial edge inspection runtime architecture combining patch-level visual embeddings, synchronized 3-axis physical telemetry, and temporal finite-state policy gating. Our multi-modal policy framework executes optical clarity checks, exponential moving-average smoothing, $k$-of-$N$ sliding window confirmation, operational state suppression, and an incident refractory cooldown finite state machine. To guarantee fault tolerance, a local Write-Ahead Log disk spooler acts as a non-volatile buffer during upstream broker disconnections. We evaluate our system across 8 ablation policy modes and 6 standardized industrial workload scenarios with multi-seed Monte Carlo simulations, multi-level percentile bootstrap confidence intervals, and step-down Holm-Bonferroni hypothesis tests. Results show 100.0% false alarm suppression in transient glitch workloads, 93.3% suppression with 3.0-frame median delay in sustained defects, zero observed event loss under tested partition scenarios with capped 50,000-record spool capacity, and a 10.7 ms mean end-to-end pipeline latency with 0.0% deadline miss rate against a 30 FPS (33.333 ms) SLA.

---

### 1. Introduction & Research Questions
Deploying deep visual anomaly detection models (e.g. PatchCore) directly on industrial production lines presents unique cyber-physical challenges:
1. **Transient Optical Noise**: Specular highlights and dust particles generate brief, intense anomaly score spikes.
2. **Cognitive Alarm Fatigue**: Constant false alarms overwhelm human triage operators beyond safe cognitive limits ($>60\text{ reviews/hour}$).
3. **Intermittent Network Outages**: Shop-floor Wi-Fi/Ethernet partitions drop mission-critical inspection records without local spooling.

We address these challenges by formulating three core Research Questions:
* **RQ1 (Temporal Confirmation)**: How effectively does sliding-window temporal confirmation filter isolated optical false positives without inducing excessive detection delays?
* **RQ2 (Operator Workload Mitigation)**: What is the reduction in cognitive operator triage burden under correlated multi-modal sensor fusion and state-gated suppression?
* **RQ3 (Resilience & SLA Compliance)**: Can edge persistence and Write-Ahead Log spooling guarantee zero event loss under intermittent network partitions while preserving real-time 30 FPS deadline compliance?

---

### 2. Empirical Findings Across 8 Policy Variants

The 8 evaluated policy variants include:
1. `BASELINE`: Instantaneous raw thresholding ($v_t \ge \tau_{\text{high}}$).
2. `EMA_ONLY`: Moving-average smoothing ($\alpha_v = 0.35$).
3. `EMA_KOFN`: Moving-average + $k$-of-$N$ confirmation ($k=4, N=10$).
4. `NO_COOLDOWN`: Multi-modal cascade with cooldown disabled ($C=0$).
5. `NO_FUSION`: Vision-only temporal confirmation (ignoring sensor telemetry).
6. `NO_DIVERGENCE`: Multi-modal cascade without cross-modal divergence checking.
7. `NO_STATE_GATING`: Multi-modal cascade without operational machine state gating.
8. `FULL_POLICY`: Complete architecture integrating all filters, fusion gates, and cooldown FSM.

#### Summary of Key Metrics:
* **False Alarm Suppression**: 100.0% suppression on transient optical noise; 93.3% suppression in sustained defect regimes.
* **Detection Delay**: Median 3.0 frames (100.0 ms) detection latency at 30 FPS.
* **SLA Compliance**: 10.7 ms mean latency; 0.0% deadline miss rate across 5,000 simulated cycles against the 33.333 ms SLA.
* **Partition Resilience**: Zero observed event loss under tested partition scenarios with capped 50,000-record spool capacity.

---

### 3. Threats to Validity
1. **Simulated Physical Sensor Transfer Functions**: Physical telemetry is generated via calibrated stochastic differential equations; real vibration signals exhibit mechanical harmonics and nonlinearities.
2. **Discrete Scenario Boundaries**: Benchmarks evaluate discrete fault windows rather than continuous non-stationary distributions.
3. **Static Operator Modeling**: Cognitive fatigue is modeled using a static 60 reviews/hr threshold; human fatigue in practice is state-dependent.
4. **Synthetic Frame Rendering**: Synthetic frames approximate brushed-steel grain but may not reflect full optical variability of all industrial materials.

---

### 4. Reproduction & Artifact Verification
To replicate all empirical benchmarks, sensitivity sweeps, spooler stress tests, and publication vector plots:
```bash
./scripts/reproduce_main_results.sh
```
All outputs are hashed with SHA-256 in [`results/CHECKSUMS.txt`](file:///home/sengar/edge-inspection-runtime/results/CHECKSUMS.txt).
