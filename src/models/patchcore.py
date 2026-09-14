from typing import Tuple, List, Optional
import os
import numpy as np
import scipy.ndimage
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
import torchvision.models as tvm

from .base import BaseAnomalyDetector


class PatchCore(BaseAnomalyDetector):
    """
    PatchCore-inspired Anomaly Detector.
    Extracts multi-scale locally aware patch features from ResNet-18 (layer2 + layer3),
    applies Minimax Greedy Coreset Selection to construct an efficient memory bank,
    and calculates nearest-neighbor patch anomaly distances with 2D Gaussian smoothing.
    """
    def __init__(
        self,
        backbone: str = "resnet18",
        coreset_sampling_ratio: float = 0.10,
        projection_dim: int = 128,
        device: Optional[str] = None,
        seed: int = 42
    ):
        super().__init__(device=device)
        self.backbone_name = backbone
        self.coreset_sampling_ratio = coreset_sampling_ratio
        self.projection_dim = projection_dim
        self.seed = seed

        if backbone == "resnet18":
            weights = tvm.ResNet18_Weights.IMAGENET1K_V1
            net = tvm.resnet18(weights=weights)
        elif backbone == "resnet50":
            weights = tvm.ResNet50_Weights.IMAGENET1K_V2
            net = tvm.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

        self.net = net.to(self.device).eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.avg_pool = nn.AvgPool2d(kernel_size=3, stride=1, padding=1)
        self.memory_bank: Optional[torch.Tensor] = None

    def _extract_multiscale_features(self, x: torch.Tensor) -> torch.Tensor:
        x = x.to(self.device)
        with torch.no_grad():
            x0 = self.net.conv1(x)
            x0 = self.net.bn1(x0)
            x0 = self.net.relu(x0)
            x0 = self.net.maxpool(x0)

            l1 = self.net.layer1(x0)
            l2 = self.net.layer2(l1)
            l3 = self.net.layer3(l2)

            p2 = self.avg_pool(l2)
            p3 = self.avg_pool(l3)

            p3_up = F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)

            features = torch.cat([p2, p3_up], dim=1)
            B, C, H, W = features.shape

            patches = features.permute(0, 2, 3, 1).reshape(B * H * W, C)
            return patches

    def _greedy_coreset_subsampling(self, patches: torch.Tensor) -> torch.Tensor:
        N, D = patches.shape
        M = max(1, int(N * self.coreset_sampling_ratio))

        if M >= N:
            return patches

        if D > self.projection_dim:
            g = torch.Generator(device=self.device)
            g.manual_seed(self.seed)
            proj_mat = torch.randn(D, self.projection_dim, device=self.device, generator=g)
            proj_mat, _ = torch.linalg.qr(proj_mat)
            patches_proj = torch.matmul(patches, proj_mat)
        else:
            patches_proj = patches

        center = torch.mean(patches_proj, dim=0, keepdim=True)
        init_dists = torch.norm(patches_proj - center, dim=1)
        start_idx = int(torch.argmax(init_dists).item())

        selected_indices = [start_idx]

        patches_sq_norm = torch.sum(patches_proj ** 2, dim=1)
        start_pt = patches_proj[start_idx]
        start_sq_norm = torch.sum(start_pt ** 2)
        dot_start = torch.mv(patches_proj, start_pt)
        min_distances_sq = torch.clamp(patches_sq_norm + start_sq_norm - 2.0 * dot_start, min=0.0)

        batch_k = 50
        while len(selected_indices) < M:
            k_cur = min(batch_k, M - len(selected_indices))
            if k_cur == 1:
                next_idx = int(torch.argmax(min_distances_sq).item())
                selected_indices.append(next_idx)
                query_pts = patches_proj[next_idx:next_idx+1]
            else:
                _, top_idx = torch.topk(min_distances_sq, k=k_cur)
                selected_indices.extend(top_idx.cpu().tolist())
                query_pts = patches_proj[top_idx]

            dots = torch.matmul(patches_proj, query_pts.T)
            q_sq = torch.sum(query_pts**2, dim=1).unsqueeze(0)
            dist_sqs = torch.clamp(patches_sq_norm.unsqueeze(1) + q_sq - 2.0 * dots, min=0.0)
            min_distances_sq = torch.minimum(min_distances_sq, torch.min(dist_sqs, dim=1).values)

        return patches[selected_indices[:M]]

    def fit(self, dataloader: DataLoader) -> None:
        all_patches_list = []
        for batch in dataloader:
            if isinstance(batch, (list, tuple)):
                x = batch[0]
            else:
                x = batch
            patches = self._extract_multiscale_features(x)
            all_patches_list.append(patches)

        all_patches = torch.cat(all_patches_list, dim=0)
        self.memory_bank = self._greedy_coreset_subsampling(all_patches)

    def predict(self, x: torch.Tensor) -> Tuple[np.ndarray, np.ndarray]:
        if self.memory_bank is None:
            raise RuntimeError("PatchCore model is not fitted. Call .fit() first.")

        B, _, H_in, W_in = x.shape
        x = x.to(self.device)

        with torch.no_grad():
            x0 = self.net.conv1(x)
            x0 = self.net.bn1(x0)
            x0 = self.net.relu(x0)
            x0 = self.net.maxpool(x0)

            l1 = self.net.layer1(x0)
            l2 = self.net.layer2(l1)
            l3 = self.net.layer3(l2)

            p2 = self.avg_pool(l2)
            p3 = self.avg_pool(l3)
            p3_up = F.interpolate(p3, size=p2.shape[2:], mode="bilinear", align_corners=False)

            features = torch.cat([p2, p3_up], dim=1)
            B_feat, C_feat, H_feat, W_feat = features.shape

            query_patches = features.permute(0, 2, 3, 1).reshape(B_feat * H_feat * W_feat, C_feat)

            chunk_size = 2048
            min_dists_list = []
            for i in range(0, query_patches.shape[0], chunk_size):
                q_chunk = query_patches[i:i + chunk_size]
                dists = torch.cdist(q_chunk, self.memory_bank, p=2.0)
                min_dists, _ = torch.min(dists, dim=1)
                min_dists_list.append(min_dists)

            min_dists_all = torch.cat(min_dists_list, dim=0)
            patch_scores = min_dists_all.reshape(B, 1, H_feat, W_feat)
            amaps_tensor = F.interpolate(patch_scores, size=(H_in, W_in), mode="bilinear", align_corners=False)
            amaps_np = amaps_tensor.squeeze(1).cpu().numpy()

        smoothed_amaps = np.zeros_like(amaps_np)
        image_scores = np.zeros(B, dtype=float)

        for b in range(B):
            smoothed_amaps[b] = scipy.ndimage.gaussian_filter(amaps_np[b], sigma=4)
            image_scores[b] = float(np.max(smoothed_amaps[b]))

        return image_scores, smoothed_amaps

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        torch.save({
            "memory_bank": self.memory_bank.cpu() if self.memory_bank is not None else None,
            "backbone": self.backbone_name,
            "coreset_sampling_ratio": self.coreset_sampling_ratio,
            "projection_dim": self.projection_dim,
            "seed": self.seed
        }, path)

    def load(self, path: str) -> None:
        state = torch.load(path, map_location=self.device)
        mb = state.get("memory_bank")
        self.memory_bank = mb.to(self.device) if mb is not None else None
        self.backbone_name = state.get("backbone", "resnet18")
        self.coreset_sampling_ratio = state.get("coreset_sampling_ratio", 0.10)
        self.projection_dim = state.get("projection_dim", 128)
        self.seed = state.get("seed", 42)
