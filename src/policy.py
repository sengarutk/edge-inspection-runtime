"""Temporal Decision Policies & Cross-Modal Risk Engine.

Implements EMA smoothing, sliding window confirmation filters, anti-fatigue cooldown
state machines, cross-modal discrepancy checks, and degraded hardware fallbacks.
"""

from __future__ import annotations

import collections
import time
import uuid
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Deque, Dict, Optional
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import PolicyConfig, load_policy_config
from src.inference_service import InferenceResult
from src.sensor_simulator import MachineState, SensorReading


class RiskState(str, Enum):
    """Operational risk classification output by the policy engine."""
    NORMAL = "NORMAL"
    REVIEW_REQUIRED = "REVIEW_REQUIRED"
    HIGH_SEVERITY = "HIGH_SEVERITY"


class TriggerReason(str, Enum):
    """Specific operational or diagnostic reason driving the policy decision."""
    NOMINAL_OPERATION = "NOMINAL_OPERATION"
    SUSTAINED_VISION_ANOMALY = "SUSTAINED_VISION_ANOMALY"
    SUSTAINED_SENSOR_ANOMALY = "SUSTAINED_SENSOR_ANOMALY"
    MULTI_MODAL_CONFIRMED_FAULT = "MULTI_MODAL_CONFIRMED_FAULT"
    CROSS_MODAL_DISCREPANCY = "CROSS_MODAL_DISCREPANCY"
    OPTICAL_DEGRADATION_FALLBACK = "OPTICAL_DEGRADATION_FALLBACK"
    SENSOR_DEGRADATION_FALLBACK = "SENSOR_DEGRADATION_FALLBACK"
    STATE_GATED_SUPPRESSION = "STATE_GATED_SUPPRESSION"
    COOLDOWN_ACTIVE = "COOLDOWN_ACTIVE"
    CRITICAL_MACHINE_FAULT = "CRITICAL_MACHINE_FAULT"


class PolicyDecision(BaseModel):
    """Structured temporal decision record emitted on every evaluation cycle."""
    model_config = ConfigDict(extra="forbid")

    decision_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Globally unique decision identifier (UUIDv4)."
    )
    timestamp_utc: str = Field(..., description="ISO-8601 UTC formatted timestamp (YYYY-MM-DDTHH:MM:SS.fffZ).")
    camera_id: str = Field(..., description="Camera identifier evaluated.")
    machine_id: str = Field(..., description="Monitored machine unit identifier.")
    machine_state: MachineState = Field(..., description="Machine operational state during evaluation.")
    risk_state: RiskState = Field(..., description="Assigned operational risk category.")
    trigger_reason: TriggerReason = Field(..., description="Root cause or policy triggering rationale.")
    raw_scores: Dict[str, float] = Field(..., description="Raw instantaneous scores from vision and sensors.")
    smoothed_scores: Dict[str, float] = Field(..., description="Exponentially smoothed vision and sensor scores.")
    window_stats: Dict[str, Any] = Field(
        ..., description="Sliding window confirmation statistics (window size, required k, active exceedances)."
    )
    cooldown_remaining: int = Field(..., ge=0, description="Remaining steps in alert fatigue suppression cooldown.")
    is_degraded: bool = Field(..., description="True if optical stream or sensor telemetry is degraded.")
    frame_id: Optional[str] = Field(default=None, description="Correlated camera frame UUID.")
    reading_id: Optional[str] = Field(default=None, description="Correlated sensor reading UUID.")
    evidence_uri: Optional[str] = Field(default=None, description="Local or remote path to image artifact.")
    latency_ms: float = Field(default=0.0, ge=0.0, description="Total pipeline latency in ms.")
    diagnostics: Dict[str, Any] = Field(
        default_factory=dict, description="Detailed diagnostic telemetry and divergence values."
    )

    def to_mqtt_payload(self) -> Dict[str, Any]:
        """Serialize policy decision strictly conforming to edge messaging event contract."""
        return {
            "event_id": self.decision_id,
            "timestamp_utc": self.timestamp_utc,
            "camera_id": self.camera_id,
            "machine_id": self.machine_id,
            "machine_state": self.machine_state.value,
            "risk_state": self.risk_state.value,
            "trigger_reason": self.trigger_reason.value,
            "raw_scores": self.raw_scores,
            "smoothed_scores": self.smoothed_scores,
            "window_stats": self.window_stats,
            "cooldown_remaining": self.cooldown_remaining,
            "is_degraded": self.is_degraded,
            "frame_id": self.frame_id,
            "reading_id": self.reading_id,
            "evidence_uri": self.evidence_uri,
            "latency_ms": self.latency_ms,
            "diagnostics": self.diagnostics,
        }


class TemporalPolicyEngine:
    """Industrial edge temporal decision policy and cross-modal risk engine."""

    def __init__(
        self,
        config: Optional[PolicyConfig] = None,
        camera_id: str = "line1_overhead_cam01",
        machine_id: str = "press_unit_04",
    ) -> None:
        """Initialize the temporal decision engine with operational policy configurations.

        Args:
            config: Optional PolicyConfig instance. If None, default is loaded from disk.
            camera_id: Monitored camera device identifier.
            machine_id: Monitored machine unit identifier.
        """
        self.config = config or load_policy_config()
        self.camera_id = camera_id
        self.machine_id = machine_id

        # Internal state tracking
        self.vision_ema: Optional[float] = None
        self.sensor_ema: Optional[float] = None
        self.vision_history: Deque[bool] = collections.deque(
            maxlen=self.config.confirmation_window.window_size_n
        )
        self.sensor_history: Deque[bool] = collections.deque(
            maxlen=self.config.confirmation_window.window_size_n
        )
        self.cooldown_counter: int = 0
        self._prev_machine_state: Optional[MachineState] = None

        # Operational telemetry counters
        self.total_evaluations: int = 0
        self.total_escalations: int = 0
        self.total_cooldown_suppressions: int = 0
        self.total_state_suppressions: int = 0

        logger.info(
            f"Initialized TemporalPolicyEngine (camera={self.camera_id}, machine={self.machine_id}, "
            f"window_N={self.config.confirmation_window.window_size_n}, "
            f"consecutive_k={self.config.confirmation_window.consecutive_k}, "
            f"cooldown_steps={self.config.cooldown.cooldown_steps})"
        )

    def reset(self) -> None:
        """Reset internal smoothing histories, sliding windows, counters, and cooldown state."""
        self.vision_ema = None
        self.sensor_ema = None
        self.vision_history.clear()
        self.sensor_history.clear()
        self.cooldown_counter = 0
        self._prev_machine_state = None
        self.total_evaluations = 0
        self.total_escalations = 0
        self.total_cooldown_suppressions = 0
        self.total_state_suppressions = 0
        logger.info("TemporalPolicyEngine state successfully reset.")

    def get_telemetry_stats(self) -> Dict[str, Any]:
        """Return running operational telemetry and performance metrics."""
        suppression_rate = (
            (self.total_cooldown_suppressions + self.total_state_suppressions) / max(1, self.total_evaluations)
        )
        return {
            "total_evaluations": self.total_evaluations,
            "total_escalations": self.total_escalations,
            "total_cooldown_suppressions": self.total_cooldown_suppressions,
            "total_state_suppressions": self.total_state_suppressions,
            "suppression_rate": suppression_rate,
            "active_cooldown": self.cooldown_counter > 0,
            "cooldown_remaining": self.cooldown_counter,
        }

    def _update_ema(self, raw_vision: float, raw_sensor: float) -> None:
        """Update exponential moving averages for vision and physical sensor scores."""
        alpha_v = self.config.temporal_smoothing.alpha_vision
        alpha_s = self.config.temporal_smoothing.alpha_sensor

        if self.vision_ema is None:
            self.vision_ema = float(raw_vision)
        else:
            self.vision_ema = float(alpha_v * raw_vision + (1.0 - alpha_v) * self.vision_ema)

        if self.sensor_ema is None:
            self.sensor_ema = float(raw_sensor)
        else:
            self.sensor_ema = float(alpha_s * raw_sensor + (1.0 - alpha_s) * self.sensor_ema)

    def evaluate(
        self,
        inference_result: InferenceResult,
        sensor_reading: SensorReading,
        evidence_uri: Optional[str] = None,
    ) -> PolicyDecision:
        """Evaluate multi-modal inputs through temporal filtering and decision cascade.

        Args:
            inference_result: Vision inference result from InferenceEngine.
            sensor_reading: Physical telemetry reading from SensorSimulator.
            evidence_uri: Optional filesystem path or URI of captured defective frame.

        Returns:
            PolicyDecision containing risk classification, trigger rationale, and diagnostics.
        """
        eval_start = time.perf_counter()
        self.total_evaluations += 1
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        raw_v = float(inference_result.vision_score)
        raw_s = float(sensor_reading.sensor_score)
        machine_state = sensor_reading.machine_state
        tau_high = self.config.thresholds.vision_high
        tau_med = self.config.thresholds.vision_medium
        tau_sensor = self.config.thresholds.sensor_anomaly
        tau_div = self.config.thresholds.cross_modal_divergence
        k_req = self.config.confirmation_window.consecutive_k
        n_win = self.config.confirmation_window.window_size_n

        # --- Step A: Update Smoothing & Sliding Windows ---
        self._update_ema(raw_v, raw_s)
        assert self.vision_ema is not None and self.sensor_ema is not None

        vision_high_exceeded = self.vision_ema >= tau_high
        sensor_high_exceeded = self.sensor_ema >= tau_sensor

        self.vision_history.append(vision_high_exceeded)
        self.sensor_history.append(sensor_high_exceeded)

        vision_exceedances = sum(self.vision_history)
        sensor_exceedances = sum(self.sensor_history)

        vision_confirmed_high = vision_exceedances >= k_req
        sensor_confirmed = sensor_high_exceeded or (sensor_exceedances >= k_req)
        vision_medium_exceeded = self.vision_ema >= tau_med
        delta_modal = abs(self.vision_ema - self.sensor_ema)

        is_optical_degraded = not inference_result.optical_health.is_valid
        is_sensor_degraded = sensor_reading.is_degraded
        is_system_degraded = is_optical_degraded or is_sensor_degraded

        # Decrement active cooldown at start of step
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1

        window_stats = {
            "window_size_n": n_win,
            "consecutive_k": k_req,
            "active_exceedances_count": {
                "vision_high": vision_exceedances,
                "sensor_high": sensor_exceedances,
            },
            "vision_confirmed_high": vision_confirmed_high,
            "sensor_confirmed": sensor_confirmed,
        }

        raw_scores = {"vision_raw": raw_v, "sensor_raw": raw_s}
        smoothed_scores = {"vision_ema": self.vision_ema, "sensor_ema": self.sensor_ema}
        diagnostics = {
            "cross_modal_divergence": delta_modal,
            "optical_degradation_reason": inference_result.optical_health.degradation_reason,
            "missing_sensor_channels": sensor_reading.missing_channels,
            "cooldown_counter": self.cooldown_counter,
            "incident_latched": False,
        }

        def build_decision(
            risk_state: RiskState, trigger_reason: TriggerReason, is_deg: bool, diag: Dict[str, Any]
        ) -> PolicyDecision:
            eval_elapsed_ms = (time.perf_counter() - eval_start) * 1000.0
            total_latency_ms = float(inference_result.latency_ms + eval_elapsed_ms)
            self._prev_machine_state = machine_state
            return PolicyDecision(
                timestamp_utc=now_utc,
                camera_id=self.camera_id,
                machine_id=self.machine_id,
                machine_state=machine_state,
                risk_state=risk_state,
                trigger_reason=trigger_reason,
                raw_scores=raw_scores,
                smoothed_scores=smoothed_scores,
                window_stats=window_stats,
                cooldown_remaining=self.cooldown_counter,
                is_degraded=is_deg,
                frame_id=inference_result.frame_id,
                reading_id=sensor_reading.reading_id,
                evidence_uri=evidence_uri,
                latency_ms=total_latency_ms,
                diagnostics=diag,
            )

        # --- Step B: Degraded Hardware Fallbacks ---
        if is_optical_degraded:
            if machine_state == MachineState.FAULT or sensor_reading.sensor_score >= tau_sensor:
                self.total_escalations += 1
                return build_decision(
                    RiskState.HIGH_SEVERITY, TriggerReason.CRITICAL_MACHINE_FAULT, True, diagnostics
                )
            return build_decision(
                RiskState.REVIEW_REQUIRED, TriggerReason.OPTICAL_DEGRADATION_FALLBACK, True, diagnostics
            )

        if is_sensor_degraded and raw_v >= tau_high:
            return build_decision(
                RiskState.REVIEW_REQUIRED, TriggerReason.SENSOR_DEGRADATION_FALLBACK, True, diagnostics
            )

        # --- Step C: Machine State Lockout & Safety Gating ---
        if machine_state == MachineState.FAULT:
            self.total_escalations += 1
            if self._prev_machine_state == MachineState.FAULT:
                diagnostics["incident_latched"] = True
            self.cooldown_counter = self.config.cooldown.cooldown_steps
            return build_decision(
                RiskState.HIGH_SEVERITY, TriggerReason.CRITICAL_MACHINE_FAULT, is_system_degraded, diagnostics
            )

        if (
            (machine_state == MachineState.IDLE and self.config.machine_state_gating.suppress_high_severity_on_idle)
            or (
                machine_state == MachineState.MAINTENANCE
                and self.config.machine_state_gating.suppress_high_severity_on_maintenance
            )
        ):
            has_anomaly = (
                vision_confirmed_high
                or sensor_confirmed
                or vision_medium_exceeded
                or raw_v >= tau_med
                or raw_s >= tau_sensor
            )
            if has_anomaly:
                self.total_state_suppressions += 1
                return build_decision(
                    RiskState.REVIEW_REQUIRED, TriggerReason.STATE_GATED_SUPPRESSION, is_system_degraded, diagnostics
                )
            return build_decision(
                RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION, is_system_degraded, diagnostics
            )

        # --- Step D: Cooldown Suppression ---
        if self.cooldown_counter > 0:
            self.total_cooldown_suppressions += 1
            has_anomaly = vision_confirmed_high or sensor_confirmed or vision_medium_exceeded
            risk = RiskState.REVIEW_REQUIRED if has_anomaly else RiskState.NORMAL
            return build_decision(
                risk, TriggerReason.COOLDOWN_ACTIVE, is_system_degraded, diagnostics
            )

        # --- Step E: Multi-Modal Confirmation & Divergence Cascade ---
        if vision_confirmed_high and sensor_confirmed:
            self.total_escalations += 1
            self.cooldown_counter = self.config.cooldown.cooldown_steps
            return build_decision(
                RiskState.HIGH_SEVERITY, TriggerReason.MULTI_MODAL_CONFIRMED_FAULT, is_system_degraded, diagnostics
            )

        if vision_confirmed_high and not sensor_confirmed:
            self.total_escalations += 1
            self.cooldown_counter = self.config.cooldown.cooldown_steps
            return build_decision(
                RiskState.HIGH_SEVERITY, TriggerReason.SUSTAINED_VISION_ANOMALY, is_system_degraded, diagnostics
            )

        if sensor_confirmed and not vision_confirmed_high:
            return build_decision(
                RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_SENSOR_ANOMALY, is_system_degraded, diagnostics
            )

        if delta_modal >= tau_div and (self.vision_ema >= tau_med or self.sensor_ema >= tau_sensor or raw_v >= tau_med):
            return build_decision(
                RiskState.REVIEW_REQUIRED, TriggerReason.CROSS_MODAL_DISCREPANCY, is_system_degraded, diagnostics
            )

        if vision_medium_exceeded:
            return build_decision(
                RiskState.REVIEW_REQUIRED, TriggerReason.SUSTAINED_VISION_ANOMALY, is_system_degraded, diagnostics
            )

        return build_decision(
            RiskState.NORMAL, TriggerReason.NOMINAL_OPERATION, is_system_degraded, diagnostics
        )