"""
Quality Carbon Footprint (QCF) Engine for Edge Quality Intelligence.
Calculates GHG emissions (kg CO2e and metric tons) associated with scrap, rework, escape penalties, and edge compute.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, Optional
from pathlib import Path
import yaml

ROOT_DIR = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG_PATH = ROOT_DIR / "configs" / "sustainability_parameters.yaml"


@dataclass
class SustainabilityParameters:
    part_name: str = "metal_nut"
    m_part: float = 0.45            # component raw mass (kg)
    kappa_mat: float = 2.89         # material embodied carbon (kg CO2e / kg material)
    E_rework: float = 1.85          # energy required for rework (kWh / part)
    xi_grid: float = 0.230          # grid carbon intensity (kg CO2e / kWh)
    gamma_fatal: float = 0.35       # fraction of defects that are fatal scrap
    gamma_false_scrap: float = 0.05 # fraction of false alarms erroneously scrapped
    theta_tier: float = 8.0         # downstream compounding escape penalty factor
    eta_recycle: float = 0.85       # material circularity / recyclability factor
    P_edge: float = 15.0            # edge runtime power consumption (Watts)
    latency_ms: float = 8.5         # per-frame inference latency (ms)
    N_annual: int = 1_000_000       # parts / year normalization baseline

    @classmethod
    def from_yaml(cls, yaml_path: Optional[str | Path] = None, part_name: str = "metal_nut") -> "SustainabilityParameters":
        path = Path(yaml_path) if yaml_path else DEFAULT_CONFIG_PATH
        if not path.is_absolute() and not path.exists():
            path = ROOT_DIR / path

        if not path.exists():
            raise FileNotFoundError(f"Config file not found at: {path}")
        with open(path, "r", encoding="utf-8") as f:
            cfg = yaml.safe_load(f)

        xi_grid = float(cfg.get("grid_intensity", cfg.get("grid_intensity_xi_grid", 0.230)))
        P_edge = float(cfg.get("edge_power_watts", 15.0))
        N_annual = int(cfg.get("annual_production_volume", 1_000_000))

        parts_cfg = cfg.get("part_types", {})
        if part_name not in parts_cfg:
            raise KeyError(f"Part type '{part_name}' not defined in {path}. Available: {list(parts_cfg.keys())}")

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
            latency_ms=float(p.get("latency_ms", 8.5)),
            N_annual=N_annual,
        )


class QualityCarbonFootprintEngine:
    """
    Calculates Quality Carbon Footprint (QCF in kg CO2e/yr and metric tons CO2e/yr).
    
    Formula:
    * Scrapped items: (gamma_fatal * N_TP + gamma_false_scrap * N_FP) * m_part * kappa_mat
    * Reworked items: ((1 - gamma_fatal) * N_TP + N_FP) * (E_rework * xi_grid)
    * Escaped defects: N_FN * (theta_tier * m_part * kappa_mat + 2.0 * E_rework * xi_grid)
    * Edge compute footprint: (N_annual * (latency_ms / (1000 * 3600))) * (P_edge / 1000) * xi_grid
    * Annualize to N_annual = 1,000,000 parts/year.
    """

    def __init__(
        self,
        params: Optional[SustainabilityParameters] = None,
        m_part: Optional[float] = None,
        kappa_mat: Optional[float] = None,
        E_rework: Optional[float] = None,
        xi_grid: Optional[float] = None,
        gamma_fatal: Optional[float] = None,
        gamma_false_scrap: Optional[float] = None,
        theta_tier: Optional[float] = None,
        P_edge: Optional[float] = None,
        latency_ms: Optional[float] = None,
    ):
        self.params = params or SustainabilityParameters()
        if m_part is not None:
            self.params.m_part = float(m_part)
        if kappa_mat is not None:
            self.params.kappa_mat = float(kappa_mat)
        if E_rework is not None:
            self.params.E_rework = float(E_rework)
        if xi_grid is not None:
            self.params.xi_grid = float(xi_grid)
        if gamma_fatal is not None:
            self.params.gamma_fatal = float(gamma_fatal)
        if gamma_false_scrap is not None:
            self.params.gamma_false_scrap = float(gamma_false_scrap)
        if theta_tier is not None:
            self.params.theta_tier = float(theta_tier)
        if P_edge is not None:
            self.params.P_edge = float(P_edge)
        if latency_ms is not None:
            self.params.latency_ms = float(latency_ms)

    def compute_annual_qcf(
        self,
        tp_count: int,
        fp_count: int,
        fn_count: int,
        tn_count: int,
        total_parts: int,
        annual_production: int = 1_000_000,
    ) -> Dict[str, float]:
        p = self.params
        if total_parts <= 0:
            scale = 0.0
        else:
            scale = float(annual_production) / float(total_parts)

        n_tp = float(tp_count) * scale
        n_fp = float(fp_count) * scale
        n_fn = float(fn_count) * scale
        n_tn = float(tn_count) * scale

        # 1. Scrapped items
        ghg_scrap = (p.gamma_fatal * n_tp + p.gamma_false_scrap * n_fp) * p.m_part * p.kappa_mat
        scrapped_mass_kg = (p.gamma_fatal * n_tp + p.gamma_false_scrap * n_fp) * p.m_part

        # 2. Reworked items
        ghg_rework = ((1.0 - p.gamma_fatal) * n_tp + n_fp) * (p.E_rework * p.xi_grid)
        energy_rework = ((1.0 - p.gamma_fatal) * n_tp + n_fp) * p.E_rework

        # 3. Escaped defects
        ghg_escape = n_fn * (p.theta_tier * p.m_part * p.kappa_mat + 2.0 * p.E_rework * p.xi_grid)
        energy_escape = n_fn * (2.0 * p.E_rework)

        # 4. Edge compute footprint
        hours_compute = (float(annual_production) * p.latency_ms) / (1000.0 * 3600.0)
        energy_compute = hours_compute * (p.P_edge / 1000.0)
        ghg_compute = energy_compute * p.xi_grid

        # Totals
        total_qcf_kg = ghg_scrap + ghg_rework + ghg_escape + ghg_compute
        total_qcf_metric_tons = total_qcf_kg / 1000.0
        total_energy_kwh = energy_rework + energy_escape + energy_compute

        return {
            "ghg_scrap": ghg_scrap,
            "ghg_rework": ghg_rework,
            "ghg_escape": ghg_escape,
            "ghg_compute": ghg_compute,
            "total_qcf_kg": total_qcf_kg,
            "total_qcf_metric_tons": total_qcf_metric_tons,
            "qcf_annual_kgco2e": total_qcf_kg,
            "scrapped_mass_annual_kg": scrapped_mass_kg,
            "energy_annual_kwh": total_energy_kwh,
            "scale_factor": scale,
        }

    def compute_footprint(
        self,
        tp: int,
        tn: int,
        fp: int,
        fn: int,
        latency_sec: float = 0.05,
    ) -> Dict[str, float]:
        n_total = tp + tn + fp + fn
        self.params.latency_ms = latency_sec * 1000.0
        res = self.compute_annual_qcf(
            tp_count=tp,
            fp_count=fp,
            fn_count=fn,
            tn_count=tn,
            total_parts=n_total,
            annual_production=self.params.N_annual,
        )
        return {
            "n_total": n_total,
            "scrapped_mass_batch_kg": res["scrapped_mass_annual_kg"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "scrapped_mass_annual_kg": res["scrapped_mass_annual_kg"],
            "energy_batch_kwh": res["energy_annual_kwh"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "energy_annual_kwh": res["energy_annual_kwh"],
            "qcf_scrap_kgco2e": res["ghg_scrap"],
            "qcf_rework_kgco2e": res["ghg_rework"],
            "qcf_escape_kgco2e": res["ghg_escape"],
            "qcf_compute_kgco2e": res["ghg_compute"],
            "qcf_batch_kgco2e": res["total_qcf_kg"] * (float(n_total) / float(self.params.N_annual)) if self.params.N_annual > 0 else 0.0,
            "qcf_annual_kgco2e": res["total_qcf_kg"],
            "total_qcf_metric_tons": res["total_qcf_metric_tons"],
        }
