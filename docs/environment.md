# Research Benchmark Environment & Software Stack

This document records the exact software runtime, compiler baselines, operating system kernel, and target hardware environments used for benchmarking the **Industrial Edge Inspection Runtime**.

---

## 1. Locked Software Dependencies

| Component / Library | Version | Role in Runtime Architecture |
|---|---|---|
| **Python** | `3.12.3` | Core Execution Runtime |
| **NumPy** | `^1.26.0` | Tensor manipulation, signal filtering, bootstrap statistics |
| **SciPy** | `^1.12.0` | Wilcoxon signed-rank significance testing, distribution modeling |
| **Matplotlib** | `^3.8.0` | Vector (PDF) and camera-ready raster figure rendering |
| **Pydantic** | `^2.6.0` | Strict zero-overhead schema & configuration validation |
| **Streamlit** | `^1.31.0` | Real-time industrial operator console & dynamic FMEA |
| **OpenCV (`opencv-python-headless`)** | `^4.9.0` | Optical clarity gating (Laplacian variance) & corruption streams |
| **Loguru** | `^0.7.2` | Structured asynchronous audit and telemetry logging |
| **Paho-MQTT** | `^1.6.1` | Edge MQTT pub/sub transport layer with QoS-1 delivery |
| **PyYAML** | `^6.0.1` | Declarative scenario workload ingestion |
| **Pytest** | `^7.4.4` | Automated test suite and coverage profiling |

---

## 2. Operating System & Kernel Specification

* **Host Platform**: Linux (Ubuntu 24.04 LTS via WSL2 on Windows 11 Enterprise)
* **Kernel Version**: `5.15.167.4-microsoft-standard-WSL2` x86_64
* **Compiler / Toolchain**: `GCC 13.2.0` / `Clang 18.1.3`
* **Threading Runtime**: `POSIX pthreads`, OpenMP 4.5

---

## 3. Hardware Deployment Targets

### 3.1 Primary Edge Evaluation Baseline
* **Target Edge SOC**: NVIDIA Jetson Orin Nano (6-core Arm Cortex-A78AE, 1024-core Ampere GPU)
* **Memory**: 8 GB LPDDR5 (68 GB/s bandwidth)
* **Storage**: NVMe PCIe Gen3 x4 (local SQLite WAL buffer)
* **Sampling Rate**: 30 FPS visual stream (33.333 ms budget) + 30 Hz multi-channel physical telemetry.

### 3.2 Industrial Gateway x86 Target
* **Target Platform**: Advantech UNO Industrial Gateway (Intel Core i7-1185GRE, 16 GB DDR4)
* **Industrial Protocols**: Modbus TCP, OPC-UA, MQTT over TLS
* **Max Measured Pipeline Latency**: 10.7 ms (0.0% deadline miss rate against 33.333 ms SLA).
