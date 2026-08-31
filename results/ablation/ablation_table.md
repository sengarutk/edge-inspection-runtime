# Comprehensive Ablation Study Master Results Table

| Workload Scenario | Policy Mode | Suppression (rho_supp) | Latency (Delta t) | Review Load (/hr) | Overload Frac | TPR | FPR | Tradeoff Index |
| :--- | :--- | :---: | :---: | :---: | :---: | :---: | :---: | :---: |
| **distribution_shift** | `BASELINE` | **`0.0%`** | `0.0` frames | `36003.6` | `100.0%` | `100.0%` | `33.3%` | **`0.000`** |
| **distribution_shift** | `EMA_ONLY` | **`0.0%`** | `0.1` frames | `36003.6` | `100.0%` | `100.0%` | `33.3%` | **`0.000`** |
| **distribution_shift** | `EMA_KOFN` | **`0.0%`** | `0.1` frames | `37803.8` | `100.0%` | `100.0%` | `35.0%` | **`0.000`** |
| **distribution_shift** | `NO_COOLDOWN` | **`0.0%`** | `0.1` frames | `37803.8` | `100.0%` | `100.0%` | `35.0%` | **`0.000`** |
| **distribution_shift** | `NO_FUSION` | **`91.6%`** | `0.1` frames | `37803.8` | `100.0%` | `100.0%` | `35.0%` | **`0.858`** |
| **distribution_shift** | `FULL_POLICY` | **`91.6%`** | `0.1` frames | `37803.8` | `100.0%` | `100.0%` | `35.0%` | **`0.858`** |
| **network_partitions** | `BASELINE` | **`0.0%`** | `0.0` frames | `10801.1` | `100.0%` | `100.0%` | `0.0%` | **`0.000`** |
| **network_partitions** | `EMA_ONLY` | **`3.9%`** | `0.2` frames | `10801.1` | `100.0%` | `96.7%` | `0.4%` | **`0.032`** |
| **network_partitions** | `EMA_KOFN` | **`0.0%`** | `0.2` frames | `12601.3` | `100.0%` | `96.7%` | `2.2%` | **`0.000`** |
| **network_partitions** | `NO_COOLDOWN` | **`0.0%`** | `0.2` frames | `12601.3` | `100.0%` | `96.7%` | `2.2%` | **`0.000`** |
| **network_partitions** | `NO_FUSION` | **`91.9%`** | `0.2` frames | `12601.3` | `100.0%` | `96.7%` | `2.2%` | **`0.778`** |
| **network_partitions** | `FULL_POLICY` | **`91.9%`** | `0.2` frames | `12601.3` | `100.0%` | `96.7%` | `2.2%` | **`0.778`** |
| **nominal** | `BASELINE` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **nominal** | `EMA_ONLY` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **nominal** | `EMA_KOFN` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **nominal** | `NO_COOLDOWN` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **nominal** | `NO_FUSION` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **nominal** | `FULL_POLICY` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **sensor_drift_dropout** | `BASELINE` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **sensor_drift_dropout** | `EMA_ONLY` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **sensor_drift_dropout** | `EMA_KOFN` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **sensor_drift_dropout** | `NO_COOLDOWN` | **`100.0%`** | `4.0` frames | `50885.1` | `100.0%` | `100.0%` | `47.1%` | **`0.200`** |
| **sensor_drift_dropout** | `NO_FUSION` | **`100.0%`** | `0.0` frames | `0.0` | `0.0%` | `100.0%` | `0.0%` | **`1.000`** |
| **sensor_drift_dropout** | `FULL_POLICY` | **`100.0%`** | `4.0` frames | `50885.1` | `100.0%` | `100.0%` | `47.1%` | **`0.200`** |
| **sustained_defects** | `BASELINE` | **`0.0%`** | `0.0` frames | `43204.3` | `100.0%` | `100.0%` | `0.0%` | **`0.000`** |
| **sustained_defects** | `EMA_ONLY` | **`0.3%`** | `0.1` frames | `42964.3` | `100.0%` | `97.8%` | `1.1%` | **`0.003`** |
| **sustained_defects** | `EMA_KOFN` | **`0.0%`** | `0.1` frames | `46564.7` | `100.0%` | `97.8%` | `6.7%` | **`0.000`** |
| **sustained_defects** | `NO_COOLDOWN` | **`0.0%`** | `0.1` frames | `46804.7` | `100.0%` | `98.3%` | `6.7%` | **`0.000`** |
| **sustained_defects** | `NO_FUSION` | **`92.1%`** | `0.1` frames | `46564.7` | `100.0%` | `97.8%` | `6.7%` | **`0.825`** |
| **sustained_defects** | `FULL_POLICY` | **`92.1%`** | `0.1` frames | `46804.7` | `100.0%` | `98.3%` | `6.7%` | **`0.836`** |
| **transient_glitches** | `BASELINE` | **`0.0%`** | `0.0` frames | `1800.2` | `100.0%` | `100.0%` | `0.0%` | **`0.000`** |
| **transient_glitches** | `EMA_ONLY` | **`100.0%`** | `0.3` frames | `360.0` | `66.7%` | `20.0%` | `0.0%` | **`0.833`** |
| **transient_glitches** | `EMA_KOFN` | **`100.0%`** | `0.3` frames | `360.0` | `66.7%` | `20.0%` | `0.0%` | **`0.833`** |
| **transient_glitches** | `NO_COOLDOWN` | **`100.0%`** | `0.3` frames | `2520.3` | `100.0%` | `40.0%` | `1.7%` | **`0.833`** |
| **transient_glitches** | `NO_FUSION` | **`100.0%`** | `0.3` frames | `2160.2` | `100.0%` | `20.0%` | `1.7%` | **`0.833`** |
| **transient_glitches** | `FULL_POLICY` | **`100.0%`** | `0.3` frames | `2520.3` | `100.0%` | `40.0%` | `1.7%` | **`0.833`** |