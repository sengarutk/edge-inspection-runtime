"""Optical Health Verification & Vision Inference Service.

Provides optical stream health monitoring and visual anomaly inference engine.
"""

from __future__ import annotations

import time
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import cv2
import numpy as np
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field

from src.config import SystemConfig, load_system_config


class InvalidFrameError(ValueError):
    """Raised when an invalid, corrupted, or unsupported video frame is passed to the engine."""


class InferenceEngineError(RuntimeError):
    """Raised when inference forward execution encounters an unrecoverable failure."""


class OpticalHealthStatus(BaseModel):
    """Result of optical health inspection on a single frame."""
    model_config = ConfigDict(extra="forbid")

    is_valid: bool = Field(..., description="True if frame passes all optical health checks.")
    laplacian_var: float = Field(..., ge=0.0, description="Variance of Laplacian focus metric.")
    mean_brightness: float = Field(..., ge=0.0, le=255.0, description="Mean intensity brightness across frame.")
    degradation_reason: Optional[str] = Field(
        default=None, description="Specific optical degradation reason if is_valid is False."
    )


class InferenceResult(BaseModel):
    """Structured inference output containing anomaly score, telemetry, and spatial heatmaps."""
    model_config = ConfigDict(arbitrary_types_allowed=True, extra="forbid")

    frame_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()), description="Globally unique frame identifier (UUIDv4)."
    )
    timestamp_utc: str = Field(..., description="ISO-8601 UTC formatted timestamp string (YYYY-MM-DDTHH:MM:SS.fffZ).")
    camera_id: str = Field(..., description="Unique camera identifier.")
    model_metadata: Dict[str, str] = Field(
        default_factory=dict, description="Metadata defining model name, engine, and version."
    )
    vision_score: float = Field(..., ge=0.0, le=1.0, description="Visual anomaly score bounded in [0.0, 1.0].")
    is_blurred: bool = Field(..., description="Flag indicating image defocus or motion blur.")
    is_occluded: bool = Field(..., description="Flag indicating dark or overexposed optical occlusion.")
    optical_health: OpticalHealthStatus = Field(..., description="Optical health inspection telemetry.")
    heatmap: Optional[np.ndarray] = Field(
        default=None, description="Spatial 2D visual anomaly heatmap normalized to [0.0, 1.0]."
    )
    latency_ms: float = Field(..., ge=0.0, description="Hardware execution latency in milliseconds.")
    metadata: Dict[str, Any] = Field(default_factory=dict, description="Diagnostic and processing telemetry.")


class InferenceEngine:
    """Industrial edge vision inspection and optical quality verification engine."""

    def __init__(self, config: Optional[SystemConfig] = None, seed: Optional[int] = None) -> None:
        """Initialize the inference engine with runtime configuration and deterministic seed.

        Args:
            config: Optional SystemConfig instance. If None, default config is loaded from disk.
            seed: Optional explicit random seed for synthetic inference generation.
        """
        self.config = config or load_system_config()
        self._seed = seed if seed is not None else 42
        self._rng = np.random.RandomState(self._seed)
        self._onnx_session = None
        self._input_name: Optional[str] = None
        self._output_name: Optional[str] = None

        if not self.config.inference.mock_mode and self.config.inference.model_path:
            self._init_onnx_session(self.config.inference.model_path)

        logger.info(
            f"Initialized InferenceEngine (camera_id={self.config.camera_id}, "
            f"mock_mode={self.config.inference.mock_mode}, "
            f"input_resolution={self.config.inference.input_resolution}, seed={self._seed})"
        )

    def _init_onnx_session(self, model_path: str) -> None:
        """Initialize ONNX Runtime inference session if model artifact exists.

        Args:
            model_path: Path to ONNX model file.
        """
        path = Path(model_path)
        if not path.is_file():
            logger.warning(f"Configured ONNX model path not found: {path}. Falling back to mock execution.")
            return

        try:
            import onnxruntime as ort
            self._onnx_session = ort.InferenceSession(str(path))
            self._input_name = self._onnx_session.get_inputs()[0].name
            self._output_name = self._onnx_session.get_outputs()[0].name
            logger.info(f"Successfully loaded ONNX model from {path}")
        except Exception as exc:
            logger.error(f"Failed to initialize ONNX session from {path}: {exc}")
            raise InferenceEngineError(f"ONNX initialization failed: {exc}") from exc

    def infer(self, frame: np.ndarray, inject_anomaly: bool = False) -> InferenceResult:
        """Alias for run_inference."""
        return self.run_inference(frame=frame, inject_anomaly=inject_anomaly)

    def check_optical_health(self, frame: np.ndarray) -> OpticalHealthStatus:
        """Validate optical health of an incoming camera frame.

        Computes the Variance of Laplacian focus measure and frame brightness statistics.
        Identifies optical degradation such as blur, solid black (lens cap/loss of illumination),
        or saturated white (glare/overexposure). Handles float32/float64 normalization.

        Args:
            frame: Numpy array representing an image frame (grayscale or BGR/RGB).

        Returns:
            OpticalHealthStatus containing health metrics and degradation diagnostics.

        Raises:
            InvalidFrameError: If frame is None, non-numpy, empty, or has unsupported dimensions.
        """
        if frame is None or not isinstance(frame, np.ndarray):
            raise InvalidFrameError("Frame must be a non-null numpy.ndarray.")

        if frame.size == 0 or frame.ndim not in (2, 3):
            raise InvalidFrameError(f"Invalid frame dimensions or empty array: shape={getattr(frame, 'shape', None)}")

        # Handle floating point representations and scaling
        if np.issubdtype(frame.dtype, np.floating):
            max_val = float(np.max(frame)) if frame.size > 0 else 0.0
            if max_val <= 1.01:
                frame_scaled = np.clip(frame * 255.0, 0.0, 255.0).astype(np.uint8)
            else:
                frame_scaled = np.clip(frame, 0.0, 255.0).astype(np.uint8)
        elif frame.dtype != np.uint8:
            frame_scaled = np.clip(frame, 0, 255).astype(np.uint8)
        else:
            frame_scaled = frame

        if frame_scaled.ndim == 3:
            if frame_scaled.shape[2] == 3:
                gray = cv2.cvtColor(frame_scaled, cv2.COLOR_BGR2GRAY)
            elif frame_scaled.shape[2] == 1:
                gray = frame_scaled.squeeze(axis=2)
            else:
                raise InvalidFrameError(f"Unsupported number of image channels: {frame_scaled.shape[2]}")
        else:
            gray = frame_scaled

        # Compute focus metric: Variance of Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        laplacian_var = float(laplacian.var())

        # Compute mean brightness
        mean_brightness = float(np.mean(gray))

        # Check optical degradation conditions
        degradation_reason: Optional[str] = None
        dark_thresh = self.config.optical_health.dark_frame_mean_threshold
        bright_thresh = self.config.optical_health.bright_frame_mean_threshold
        blur_thresh = self.config.optical_health.blur_laplacian_threshold

        if mean_brightness < dark_thresh:
            degradation_reason = "OPTICAL_OCCLUDED_DARK"
        elif mean_brightness > bright_thresh:
            degradation_reason = "OPTICAL_OCCLUDED_BRIGHT"
        elif laplacian_var < blur_thresh:
            degradation_reason = "OPTICAL_BLURRED"

        is_valid = degradation_reason is None

        return OpticalHealthStatus(
            is_valid=is_valid,
            laplacian_var=laplacian_var,
            mean_brightness=mean_brightness,
            degradation_reason=degradation_reason,
        )

    def _generate_synthetic_heatmap(self, resolution: Tuple[int, int], is_anomaly: bool) -> np.ndarray:
        """Generate a synthetic 2D Gaussian anomaly heatmap using deterministic engine RNG.

        Args:
            resolution: (height, width) tuple.
            is_anomaly: If True, centers a salient Gaussian activation hotspot.

        Returns:
            Normalized 2D float32 heatmap in range [0.0, 1.0].
        """
        h, w = resolution
        y, x = np.mgrid[0:h, 0:w]

        if is_anomaly:
            center_y = h * (0.4 + 0.2 * self._rng.uniform(-0.5, 0.5))
            center_x = w * (0.5 + 0.2 * self._rng.uniform(-0.5, 0.5))
            sigma_y = h * 0.15
            sigma_x = w * 0.15
            dist_sq = ((y - center_y) ** 2) / (2.0 * sigma_y ** 2) + ((x - center_x) ** 2) / (2.0 * sigma_x ** 2)
            heatmap = np.exp(-dist_sq).astype(np.float32)
            noise = self._rng.uniform(0.0, 0.05, size=(h, w)).astype(np.float32)
            heatmap = np.clip(heatmap + noise, 0.0, 1.0)
        else:
            heatmap = self._rng.uniform(0.0, 0.08, size=(h, w)).astype(np.float32)

        return heatmap

    def run_inference(self, frame: np.ndarray, inject_anomaly: bool = False) -> InferenceResult:
        """Execute visual anomaly inference pipeline on a single frame.

        Validates optical health first. If degradation is detected, marks failure flags and
        returns a degraded result. In mock mode, generates reproducible synthetic anomaly scores
        and activation heatmaps with simulated latency.

        Args:
            frame: Input video image frame as a numpy array.
            inject_anomaly: If True, simulates the presence of a visual defect.

        Returns:
            InferenceResult conforming to updated data contract.

        Raises:
            InvalidFrameError: If frame fails fundamental shape/type checks.
        """
        start_time = time.perf_counter()
        now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f")[:-3] + "Z"

        # Step 1: Optical Health Check
        health = self.check_optical_health(frame)

        if not health.is_valid:
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0
            is_blurred = health.degradation_reason == "OPTICAL_BLURRED"
            is_occluded = health.degradation_reason in ("OPTICAL_OCCLUDED_DARK", "OPTICAL_OCCLUDED_BRIGHT")

            return InferenceResult(
                frame_id=str(uuid.uuid4()),
                timestamp_utc=now_utc,
                camera_id=self.config.camera_id,
                model_metadata={
                    "model_name": "patchcore_mock",
                    "engine": "mock",
                    "version": "v1.0.0",
                },
                vision_score=0.0,
                is_blurred=is_blurred,
                is_occluded=is_occluded,
                optical_health=health,
                heatmap=None,
                latency_ms=elapsed_ms,
                metadata={
                    "optical_health_valid": False,
                    "degradation_reason": health.degradation_reason,
                    "inference_mode": "degraded_optical_bypass",
                },
            )

        # Step 2: Inference Execution (Mock Mode)
        if self.config.inference.mock_mode or self._onnx_session is None:
            mock_latency_s = max(0.0, self.config.inference.mock_latency_ms / 1000.0)
            if mock_latency_s > 0:
                time.sleep(mock_latency_s)

            if inject_anomaly:
                raw_score = float(self._rng.normal(0.85, 0.05))
            else:
                raw_score = float(self._rng.normal(0.05, 0.02))

            vision_score = float(np.clip(raw_score, 0.0, 1.0))
            h_res, w_res = self.config.inference.input_resolution[0], self.config.inference.input_resolution[1]
            heatmap = self._generate_synthetic_heatmap((h_res, w_res), is_anomaly=inject_anomaly)
            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return InferenceResult(
                frame_id=str(uuid.uuid4()),
                timestamp_utc=now_utc,
                camera_id=self.config.camera_id,
                model_metadata={
                    "model_name": "patchcore_mock",
                    "engine": "mock",
                    "version": "v1.0.0",
                },
                vision_score=vision_score,
                is_blurred=False,
                is_occluded=False,
                optical_health=health,
                heatmap=heatmap,
                latency_ms=elapsed_ms,
                metadata={
                    "optical_health_valid": True,
                    "degradation_reason": None,
                    "inference_mode": "mock",
                    "inject_anomaly": inject_anomaly,
                },
            )

        # ONNX Hardware Forward Pass Mode
        try:
            h_in, w_in = self.config.inference.input_resolution[0], self.config.inference.input_resolution[1]
            # Convert float/uint8 frames for ONNX preprocessing
            if np.issubdtype(frame.dtype, np.floating):
                frame_u8 = np.clip(frame * 255.0 if np.max(frame) <= 1.01 else frame, 0, 255).astype(np.uint8)
            else:
                frame_u8 = np.clip(frame, 0, 255).astype(np.uint8)

            resized = cv2.resize(frame_u8, (w_in, h_in))
            if resized.ndim == 2:
                resized = cv2.cvtColor(resized, cv2.COLOR_GRAY2RGB)
            elif resized.shape[2] == 3:
                resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)

            tensor_in = resized.astype(np.float32) / 255.0
            mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
            std = np.array([0.229, 0.224, 0.225], dtype=np.float32)
            tensor_in = (tensor_in - mean) / std
            tensor_in = np.transpose(tensor_in, (2, 0, 1))  # (C, H, W)
            tensor_in = np.expand_dims(tensor_in, axis=0)   # (1, C, H, W)

            onnx_inputs = {self._input_name: tensor_in}
            onnx_outputs = self._onnx_session.run([self._output_name], onnx_inputs)
            output_tensor = onnx_outputs[0]

            # Flexible output parsing for scalars, scores, or spatial heatmaps
            heatmap_out: Optional[np.ndarray] = None
            if output_tensor.ndim >= 3:
                # Patch anomaly map (1, 1, H, W) or (1, H, W)
                heatmap_2d = np.squeeze(output_tensor)
                if heatmap_2d.ndim == 2:
                    heatmap_out = np.clip(heatmap_2d.astype(np.float32), 0.0, 1.0)
                vision_score = float(np.clip(np.max(output_tensor), 0.0, 1.0))
            else:
                vision_score = float(np.clip(np.mean(output_tensor), 0.0, 1.0))

            elapsed_ms = (time.perf_counter() - start_time) * 1000.0

            return InferenceResult(
                frame_id=str(uuid.uuid4()),
                timestamp_utc=now_utc,
                camera_id=self.config.camera_id,
                model_metadata={
                    "model_name": "patchcore_onnx",
                    "engine": "onnxruntime",
                    "version": "v1.0.0",
                },
                vision_score=vision_score,
                is_blurred=False,
                is_occluded=False,
                optical_health=health,
                heatmap=heatmap_out,
                latency_ms=elapsed_ms,
                metadata={
                    "optical_health_valid": True,
                    "degradation_reason": None,
                    "inference_mode": "onnx",
                },
            )
        except Exception as exc:
            logger.error(f"ONNX forward pass error: {exc}")
            raise InferenceEngineError(f"ONNX inference failure: {exc}") from exc