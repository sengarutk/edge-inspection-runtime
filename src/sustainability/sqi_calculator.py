"""
Sustainability Quality Index (SQI) Calculator for Edge Quality Intelligence.
Combines Material Savings Factor (MSF), Energy Savings Factor (ESF),
Carbon Savings Factor (CSF), and Circularity Factor (CF).
"""

from typing import Dict, Any, Optional
from src.sustainability.qcf_engine import SustainabilityParameters, QualityCarbonFootprintEngine


class SustainabilityQualityIndexCalculator:
    """
    Calculates Sustainability Quality Index (SQI in [0.0, 1.0]):
    * MSF = 1.0 - (Scrapped_Mass_Policy / Scrapped_Mass_Baseline)
    * ESF = 1.0 - (Energy_Policy / Energy_Baseline)
    * CSF = 1.0 - (QCF_Policy / QCF_Baseline)
    * CF = (N_TN + (1 - gamma_fatal)*N_TP + eta_recycle * gamma_fatal * N_TP) / N_total
    * SQI = 0.30 * MSF + 0.25 * ESF + 0.30 * CSF + 0.15 * CF (clamped to [0.0, 1.0]).
    """

    def __init__(self, params: Optional[SustainabilityParameters] = None):
        self.params = params or SustainabilityParameters()
        self.qcf_engine = QualityCarbonFootprintEngine(self.params)

    def calculate_circularity_factor(self, tp: int, tn: int, fp: int, fn: int) -> float:
        n_total = tp + tn + fp + fn
        if n_total == 0:
            return 0.0
        p = self.params
        circ_numerator = (
            float(tn)
            + (1.0 - p.gamma_fatal) * float(tp)
            + p.eta_recycle * p.gamma_fatal * float(tp)
        )
        return max(0.0, min(1.0, circ_numerator / float(n_total)))

    def calculate_sqi(
        self,
        policy_counts: Dict[str, int],
        baseline_counts: Dict[str, int],
        policy_latency: float = 0.05,
        baseline_latency: float = 0.05,
    ) -> Dict[str, float]:
        policy_fp = self.qcf_engine.compute_footprint(
            tp=policy_counts.get("tp", 0),
            tn=policy_counts.get("tn", 0),
            fp=policy_counts.get("fp", 0),
            fn=policy_counts.get("fn", 0),
            latency_sec=policy_latency,
        )
        baseline_fp = self.qcf_engine.compute_footprint(
            tp=baseline_counts.get("tp", 0),
            tn=baseline_counts.get("tn", 0),
            fp=baseline_counts.get("fp", 0),
            fn=baseline_counts.get("fn", 0),
            latency_sec=baseline_latency,
        )

        # MSF: Material Savings Factor
        m_base = baseline_fp["scrapped_mass_annual_kg"]
        m_pol = policy_fp["scrapped_mass_annual_kg"]
        if m_base > 1e-9:
            msf = max(0.0, min(1.0, 1.0 - (m_pol / m_base)))
        else:
            msf = 1.0 if m_pol <= 1e-9 else 0.0

        # ESF: Energy Savings Factor
        e_base = baseline_fp["energy_annual_kwh"]
        e_pol = policy_fp["energy_annual_kwh"]
        if e_base > 1e-9:
            esf = max(0.0, min(1.0, 1.0 - (e_pol / e_base)))
        else:
            esf = 1.0 if e_pol <= 1e-9 else 0.0

        # CSF: Carbon Savings Factor
        c_base = baseline_fp["qcf_annual_kgco2e"]
        c_pol = policy_fp["qcf_annual_kgco2e"]
        if c_base > 1e-9:
            csf = max(0.0, min(1.0, 1.0 - (c_pol / c_base)))
        else:
            csf = 1.0 if c_pol <= 1e-9 else 0.0

        # CF: Circularity Factor
        cf = self.calculate_circularity_factor(
            tp=policy_counts.get("tp", 0),
            tn=policy_counts.get("tn", 0),
            fp=policy_counts.get("fp", 0),
            fn=policy_counts.get("fn", 0),
        )

        # SQI = 0.30 * MSF + 0.25 * ESF + 0.30 * CSF + 0.15 * CF (clamped to [0.0, 1.0])
        raw_sqi = 0.30 * msf + 0.25 * esf + 0.30 * csf + 0.15 * cf
        sqi = max(0.0, min(1.0, raw_sqi))

        return {
            "msf": msf,
            "esf": esf,
            "csf": csf,
            "cf": cf,
            "sqi": sqi,
            "policy_qcf_annual": c_pol,
            "baseline_qcf_annual": c_base,
            "policy_energy_annual": e_pol,
            "baseline_energy_annual": e_base,
            "policy_mass_annual": m_pol,
            "baseline_mass_annual": m_base,
        }
