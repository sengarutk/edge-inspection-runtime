import abc
from typing import Tuple, Dict, Any, Optional
import numpy as np
import torch
from torch.utils.data import DataLoader


class BaseAnomalyDetector(abc.ABC):
    """
    Abstract Base Class for all anomaly detection and localization methods.
    Enforces a unified fit/predict interface across memory-bank, statistical,
    and reconstruction-based anomaly detectors.
    """
    def __init__(self, device: Optional[str] = None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    @abc.abstractmethod
    def fit(self, dataloader: DataLoader) -> None:
        """
        Fits the anomaly detection model using nominal (normal-only) training images.
        dataloader yields image batches of shape [B, 3, 256, 256] or dataset items.
        """
        pass

    @abc.abstractmethod
    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        """
        Performs anomaly detection and localization on query image batch tensor [B, 3, 256, 256].
        Returns:
            image_scores: np.ndarray of shape [B], float values representing sample anomaly scores.
            anomaly_maps: np.ndarray of shape [B, 256, 256], float values representing spatial pixel defect heatmaps.
        """
        pass

    def save(self, path: str) -> None:
        """Serializes model state, weights, and/or memory banks to disk."""
        pass

    def load(self, path: str) -> None:
        """Deserializes model state, weights, and/or memory banks from disk."""
        pass
