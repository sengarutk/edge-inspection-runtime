import os
from typing import Dict, List, Any, Optional, Tuple
import cv2
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import torch
from torch.utils.data import DataLoader

from src.methods.base import BaseAnomalyDetector
from src.metrics.image_metrics import compute_optimal_f1
from src.mvtec import denormalize_image


class FailureMiner:
    """
    Automated failure discovery and diagnostic visualization engine.
    Mines 4 failure modes: False Positives, False Negatives, Localization Mismatches,
    and Corruption/Edge Misclassifications, and renders high-resolution 4-panel diagnostic grids.
    """
    def __init__(
        self,
        model: BaseAnomalyDetector,
        dataloader: DataLoader,
        output_dir: str = "results/figures/failure_cases",
        device: Optional[str] = None
    ):
        self.model = model
        self.dataloader = dataloader
        self.output_dir = output_dir
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")

    def mine_failures(self, top_k: int = 4) -> Dict[str, List[Dict[str, Any]]]:
        """
        Evaluates all samples in dataloader, finds optimal decision boundary,
        and catalogs the top k instances of each failure category.
        """
        all_samples: List[Dict[str, Any]] = []

        sample_idx = 0
        for batch in self.dataloader:
            x, y, mask, meta = batch
            scores, amaps = self.model.predict(x)

            B = x.shape[0]
            for b in range(B):
                entry = {
                    "index": sample_idx,
                    "x_tensor": x[b],
                    "y_true": int(y[b]),
                    "mask": mask[b, 0].cpu().numpy() if mask[b].ndim == 3 else mask[b].cpu().numpy(),
                    "score": float(scores[b]),
                    "amap": amaps[b],
                    "meta": {k: (v[b] if isinstance(v, (list, tuple)) else v) for k, v in meta.items()} if isinstance(meta, dict) else {}
                }
                all_samples.append(entry)
                sample_idx += 1

        if len(all_samples) == 0:
            return {
                "false_positives": [],
                "false_negatives": [],
                "localization_mismatches": [],
                "corruption_failures": []
            }

        labels = np.array([s["y_true"] for s in all_samples])
        scores = np.array([s["score"] for s in all_samples])

        f1_info = compute_optimal_f1(labels, scores)
        tau = f1_info["optimal_threshold"]

        # 1. False Positives: y == 0 with highest anomaly scores
        nominals = [s for s in all_samples if s["y_true"] == 0]
        fps = sorted(nominals, key=lambda s: s["score"], reverse=True)[:top_k]

        # 2. False Negatives: y == 1 with lowest anomaly scores
        defectives = [s for s in all_samples if s["y_true"] == 1]
        fns = sorted(defectives, key=lambda s: s["score"])[:top_k]

        # 3. Localization Mismatches: y == 1 with high score, but poor spatial overlap with ground truth
        def calc_overlap(s):
            m = s["mask"]
            if np.sum(m) == 0:
                return 1.0
            # Normalized amap top 10% overlap with mask
            amap_norm = (s["amap"] - np.min(s["amap"])) / (np.ptp(s["amap"]) + 1e-8)
            pred_mask = (amap_norm >= 0.5).astype(float)
            intersection = np.sum(pred_mask * m)
            union = np.sum(np.maximum(pred_mask, m))
            return intersection / (union + 1e-8)

        loc_candidates = [s for s in defectives if s["score"] >= tau]
        loc_mismatches = sorted(loc_candidates, key=calc_overlap)[:top_k]

        # 4. Borderline / Ambiguous Misclassifications
        # Samples closest to the threshold or with largest margin error
        corruption_failures = sorted(
            all_samples,
            key=lambda s: abs(s["score"] - tau) if (s["score"] >= tau) == (s["y_true"] == 1) else -abs(s["score"] - tau)
        )[:top_k]

        # Attach threshold to metadata
        for group in [fps, fns, loc_mismatches, corruption_failures]:
            for item in group:
                item["optimal_threshold"] = tau

        return {
            "false_positives": fps,
            "false_negatives": fns,
            "localization_mismatches": loc_mismatches,
            "corruption_failures": corruption_failures,
        }

    def save_diagnostic_grids(
        self,
        failures: Dict[str, List[Dict[str, Any]]],
        category: str,
        method_name: str
    ) -> List[str]:
        """
        Renders a 4-panel diagnostic figure for each failure instance:
        [Original RGB | Ground Truth Mask | Predicted Anomaly Map | Blended Overlay]
        """
        save_dir = os.path.join(self.output_dir, category)
        os.makedirs(save_dir, exist_ok=True)
        saved_paths: List[str] = []

        for failure_type, items in failures.items():
            for i, entry in enumerate(items):
                rgb_img = denormalize_image(entry["x_tensor"])
                mask = (entry["mask"] * 255).astype(np.uint8)

                # Min-max normalized heatmap for visualization
                amap = entry["amap"]
                amap_norm = (amap - np.min(amap)) / (np.ptp(amap) + 1e-8)
                heat_u8 = (amap_norm * 255).astype(np.uint8)
                heat_color = cv2.applyColorMap(heat_u8, cv2.COLORMAP_JET)
                heat_rgb = cv2.cvtColor(heat_color, cv2.COLOR_BGR2RGB)

                # Overlay
                overlay = (0.65 * rgb_img + 0.35 * heat_rgb).astype(np.uint8)

                fig, axes = plt.subplots(1, 4, figsize=(16, 4))
                axes[0].imshow(rgb_img)
                axes[0].set_title("Original RGB")
                axes[0].axis("off")

                axes[1].imshow(mask, cmap="gray")
                axes[1].set_title("Ground Truth Mask")
                axes[1].axis("off")

                im3 = axes[2].imshow(amap_norm, cmap="jet", vmin=0.0, vmax=1.0)
                axes[2].set_title(f"Anomaly Map (Max: {np.max(amap):.2f})")
                axes[2].axis("off")
                plt.colorbar(im3, ax=axes[2], fraction=0.046, pad=0.04)

                axes[3].imshow(overlay)
                axes[3].set_title("Blended Overlay")
                axes[3].axis("off")

                tau = entry.get("optimal_threshold", 0.0)
                defect_info = entry.get("meta", {}).get("defect_type", "unknown")
                fig.suptitle(
                    f"[{failure_type.upper()}] Cat: {category} | Method: {method_name} | "
                    f"Label: {entry['y_true']} ({defect_info}) | Score: {entry['score']:.4f} (Tau: {tau:.4f})",
                    fontsize=12,
                    fontweight="bold"
                )

                plt.tight_layout()
                out_path = os.path.join(save_dir, f"{method_name}_{failure_type}_{i}.png")
                plt.savefig(out_path, dpi=150, bbox_inches="tight")
                plt.close(fig)
                saved_paths.append(out_path)

        return saved_paths
