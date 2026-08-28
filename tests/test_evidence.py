"""Unit tests for optical evidence management and heatmap blending."""

from pathlib import Path
import cv2
import numpy as np
import pytest

from src.evidence_manager import EvidenceManager


@pytest.fixture
def evidence_mgr(tmp_path: Path) -> EvidenceManager:
    """Fixture providing an isolated EvidenceManager."""
    return EvidenceManager(storage_dir=str(tmp_path / "evidence"))


def test_save_and_load_evidence_with_heatmap(evidence_mgr: EvidenceManager) -> None:
    """Test saving composite frame with blended heatmap and loading it back."""
    h, w = 100, 100
    frame = np.full((h, w, 3), 50, dtype=np.uint8)
    cv2.circle(frame, (50, 50), 20, (200, 200, 200), -1)

    heatmap = np.zeros((h, w), dtype=np.float32)
    heatmap[40:60, 40:60] = 0.95

    frame_id = "test_frame_01"
    uri = evidence_mgr.save_evidence(frame, heatmap, frame_id)

    assert Path(uri).is_file()
    assert uri.endswith(f"{frame_id}.png")

    loaded = evidence_mgr.load_evidence(uri)
    assert loaded is not None
    assert loaded.shape == (h, 2 * w, 3)


def test_save_and_load_evidence_raw_without_heatmap(evidence_mgr: EvidenceManager) -> None:
    """Test saving raw frame without heatmap overlay."""
    h, w = 80, 80
    frame = np.full((h, w, 3), 120, dtype=np.uint8)

    frame_id = "test_raw_frame"
    uri = evidence_mgr.save_evidence(frame, None, frame_id)

    assert Path(uri).is_file()
    loaded = evidence_mgr.load_evidence(uri)
    assert loaded is not None
    assert loaded.shape == (h, w, 3)


def test_save_evidence_grayscale_and_float_conversion(evidence_mgr: EvidenceManager) -> None:
    """Test handling grayscale and float32 normalized frames."""
    gray_frame = np.ones((60, 60), dtype=np.float32) * 0.5
    heatmap = np.ones((30, 30), dtype=np.float32) * 0.8

    uri = evidence_mgr.save_evidence(gray_frame, heatmap, "gray_test")
    loaded = evidence_mgr.load_evidence(uri)
    assert loaded is not None
    assert loaded.shape == (60, 120, 3)

    # Test shape with (H, W, 1)
    single_ch_frame = np.ones((60, 60, 1), dtype=np.uint8) * 100
    uri2 = evidence_mgr.save_evidence(single_ch_frame, heatmap, "single_ch_test")
    loaded2 = evidence_mgr.load_evidence(uri2)
    assert loaded2 is not None
    assert loaded2.shape == (60, 120, 3)


def test_load_evidence_nonexistent_returns_none(evidence_mgr: EvidenceManager) -> None:
    """Test that loading nonexistent image returns None."""
    assert evidence_mgr.load_evidence("data/nonexistent_path/fake.png") is None