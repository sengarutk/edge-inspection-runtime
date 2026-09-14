import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import pytest

from src.methods.base import BaseAnomalyDetector
from src.methods.patchcore import PatchCore
from src.methods.padim import PaDiM
from src.methods.autoencoder import ConvAutoencoder


def test_base_interface():
    """
    Verify that PatchCore, PaDiM, and ConvAutoencoder inherit from BaseAnomalyDetector.
    """
    for cls in [PatchCore, PaDiM, ConvAutoencoder]:
        assert issubclass(cls, BaseAnomalyDetector)


def test_patchcore_fit_and_predict():
    """
    Test PatchCore feature extraction, greedy coreset subsampling, and prediction shapes.
    """
    torch.manual_seed(42)
    # Synthetic training dataset: 6 images of size (3, 256, 256)
    synthetic_train = torch.randn(6, 3, 256, 256)
    dl = DataLoader(TensorDataset(synthetic_train), batch_size=2, shuffle=False)

    model = PatchCore(coreset_sampling_ratio=0.10, projection_dim=64, seed=42)
    model.fit(dl)

    assert model.memory_bank is not None
    # 6 images * 1024 patches = 6144 patches -> 10% = 614 patches
    expected_m = max(1, int(6 * 1024 * 0.10))
    assert model.memory_bank.shape[0] == expected_m
    assert model.memory_bank.shape[1] == 384  # 128 (l2) + 256 (l3)

    # Test prediction
    query = torch.randn(2, 3, 256, 256)
    scores, amaps = model.predict(query)

    assert scores.shape == (2,)
    assert amaps.shape == (2, 256, 256)
    assert not np.isnan(scores).any()
    assert not np.isnan(amaps).any()
    assert not np.isinf(scores).any()
    assert not np.isinf(amaps).any()
    assert (scores > 0).all()


def test_padim_fit_and_predict():
    """
    Test PaDiM multi-scale channel extraction, spatial Gaussian modeling, and Mahalanobis scoring.
    """
    torch.manual_seed(42)
    synthetic_train = torch.randn(6, 3, 256, 256)
    dl = DataLoader(TensorDataset(synthetic_train), batch_size=2, shuffle=False)

    model = PaDiM(d_dim=50, seed=42)
    model.fit(dl)

    assert model.mean is not None
    assert model.inv_cov is not None
    assert model.mean.shape == (50, 64, 64)
    assert model.inv_cov.shape == (64, 64, 50, 50)

    query = torch.randn(2, 3, 256, 256)
    scores, amaps = model.predict(query)

    assert scores.shape == (2,)
    assert amaps.shape == (2, 256, 256)
    assert not np.isnan(scores).any()
    assert not np.isnan(amaps).any()
    assert (scores >= 0).all()


def test_autoencoder_fit_and_predict():
    """
    Test Convolutional Autoencoder training loop and reconstruction error maps.
    """
    torch.manual_seed(42)
    synthetic_train = torch.randn(4, 3, 256, 256)
    dl = DataLoader(TensorDataset(synthetic_train), batch_size=2, shuffle=False)

    model = ConvAutoencoder(epochs=2, lr=1e-3)
    model.fit(dl)

    query = torch.randn(2, 3, 256, 256)
    scores, amaps = model.predict(query)

    assert scores.shape == (2,)
    assert amaps.shape == (2, 256, 256)
    assert not np.isnan(scores).any()
    assert not np.isnan(amaps).any()
    assert (scores >= 0).all()
