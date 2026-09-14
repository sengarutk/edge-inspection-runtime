"""
Quality Carbon Footprint (QCF) Engine for Edge Quality Intelligence.
Calculates GHG emissions (kg CO2e) associated with scrap, rework, escape penalties, and edge compute.
"""

from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import yaml


@dataclass
class SustainabilityParameters:
    part_name: str = "metal_nut"
    m_part: float = 0.45            # kg / part
    kappa_mat: float = 2.89         # kg CO2e / kg material
    E_rework: float = 1.85          # kWh / rework event
    gamma_fatal: float = 0.35       # fraction of true defects that cannot be reworked (must scrap)
    gamma_false_scrap: float = 0.05 # fraction of false alarms erroneously scrapped
    theta_tier: float = 8.0         # downstream customer tier escape penalty multiplier
    eta_recycle: float = 0.85       # material circularity / recyclability factor
    xi_grid: float = 0.230          # kg CO2e / kWh (EU-27 grid average)
    P_edge: float = 15.0            # Watts (edge compute draw)
    N_annual: int = 1_000_000       # parts / year normalization baseline

    @classmethod
    def from_yaml(cls, yaml_path: str, part_name: str = "metal_nut") -> "SustainabilityParameters":
        path = Path(yaml_path)
        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {yaml_path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        xi_grid = float(cfg.get("grid_intensity_xi_grid", 0.230))
        P_edge = float(cfg.get("edge_power_watts", 15.0))
        N_annual = int(cfg.get("annual_production_volume", 1_000_000))

        parts_cfg = cfg.get("part_types", {})
        if part_name not in parts_cfg:
            raise KeyError(f"Part type '{part_name}' not defined in {yaml_path}. Available: {list(parts_cfg.keys())}")

        p = parts_cfg[part_name]
        return cls(
            part_name=part_name,
            m_part=float(p["m_part"]),
            kappa_mat=float(p["kappa_mat"]),
            E_rework=float(p["E_rework"]),
            gamma_fatal=float(p["gamma_fatal"]),
            gamma_false_scrap=float(p.get("gamma_false_scrap", 0.05)),
            theta_tier=float(p["theta_tier"]),
            eta_recycle=float(p.get("eta_recycle", 0.85)),
            xi_grid=xi_grid,
            P_edge=P_edge,
            N_annual=N_annual,
        )


class QualityCarbonFootprintEngine:
    """
    Calculates Quality Carbon Footprint (QCF in kg CO2e/yr).
    
    Formula:
    * Scrapped items: (N_TP * gamma_fatal + N_FP * gamma_false_scrap) * m_part * kappa_mat
    * Reworked items: ((1 - gamma_fatal) * N_TP + N_FP) * (E_rework * xi_grid)
    * Escaped defects: N_FN * (theta_tier * m_part * kappa_mat + 2.0 * E_rework * xi_grid)
    * Edge compute footprint: (N_total * latency_sec / 3600) * (P_edge / 1000) * xi_grid
    * Annualize to N_annual = 1,000,000 parts/year.
    """

    def __init__(self, params: Optional[SustainabilityParameters] = None):
        self.params = params or SustainabilityParameters()

    def compute_footprint(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        latency_sec: float = 0.05,
    ) -> Dict[str, float]:
        p = self.params
        n_total = tp + tn + fp + fn
        if n_total == 0:
            return {
                "n_total": 0,
                "scrapped_mass_batch_kg": 0.0,
                "scrapped_mass_annual_kg": 0.0,
                "energy_batch_kwh": 0.0,
                "energy_annual_kwh": 0.0,
                "qcf_scrap_kgco2e": 0.0,
                "qcf_rework_kgco2e": 0.0,
                "qcf_escape_kgco2e": 0.0,
                "qcf_compute_kgco2e": 0.0,
                "qcf_batch_kgco2e": 0.0,
                "qcf_annual_kgco2e": 0.0,
            }

        # 1. Scrapped items
        n_scrapped = tp * p.gamma_fatal + fp * p.gamma_false_scrap
        scrapped_mass_batch = n_scrapped * p.m_part
        qcf_scrap = scrapped_mass_batch * p.kappa_mat

        # 2. Reworked items
        n_reworked = (1.0 - p.gamma_fatal) * tp + fp
        energy_rework = n_reworked * p.E_rework
        qcf_rework = energy_rework * p.xi_grid

        # 3. Escaped defects
        qcf_escape = fn * (p.theta_tier * p.m_part * p.kappa_mat + 2.0 * p.E_rework * p.xi_grid)
        energy_escape = fn * (2.0 * p.E_rework)

        # 4. Edge compute footprint
        hours_compute = (n_total * latency_sec) / 3600.0
        energy_compute = hours_compute * (p.P_edge / 1000.0)
        qcf_compute = energy_compute * p.xi_grid

        # Totals
        qcf_batch = qcf_scrap + qcf_rework + qcf_escape + qcf_compute
        energy_batch = energy_rework + energy_escape + energy_compute

        # Annualization
        scale = float(p.N_annual) / float(n_total)
        qcf_annual = qcf_batch * scale
        scrapped_mass_annual = scrapped_mass_batch * scale
        energy_annual = energy_batch * scale

        return {
            "n_total": n_total,
            "scrapped_mass_batch_kg": scrapped_mass_batch,
            "scrapped_mass_annual_kg": scrapped_mass_annual,
            "energy_batch_kwh": energy_batch,
            "energy_annual_kwh": energy_annual,
            "qcf_scrap_kgco2e": qcf_scrap,
            "qcf_rework_kgco2e": qcf_rework,
            "qcf_escape_kgco2e": qcf_escape,
            "qcf_compute_kgco2e": qcf_compute,
            "qcf_batch_kgco2e": qcf_batch,
            "qcf_annual_kgco2e": qcf_annual,
        }
