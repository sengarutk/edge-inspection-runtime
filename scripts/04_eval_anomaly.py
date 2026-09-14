import argparse
import os
import pickle

import pandas as pd
import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.utils import seed_everything, ensure_dir
from src.mvtec import MVTecTest
from src.models import Encoder
from src.eval_harness import evaluate_category


def _csv_has_header(path: str) -> bool:
    """
    Returns True if CSV exists and looks like it has the expected header row.
    """
    if not os.path.exists(path):
        return False
    try:
        df = pd.read_csv(path)
    except Exception:
        return False
    needed = {"timestamp", "run_name", "method", "mode", "category", "auroc_image", "avg_latency_s"}
    return needed.issubset(set(df.columns))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_simclr", action="store_true")
    ap.add_argument("--mode", type=str, default="patch", choices=["patch", "global"])
    ap.add_argument("--run_name", type=str, default="main")
    args = ap.parse_args()

    cfg = TrainConfig(seed=args.seed)
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    encoder = Encoder(cfg.backbone, out_dim=512).to(device)

    if args.use_simclr:
        w = f"runs/artifacts/simclr_encoder_{args.category}_seed{cfg.seed}.pt"
        encoder.load_state_dict(torch.load(w, map_location=device))
    else:
        w = None

    mb_path = f"runs/artifacts/memorybank_{args.category}_{'simclr' if args.use_simclr else 'imagenet'}_{args.mode}_seed{cfg.seed}.pkl"
    if not os.path.exists(mb_path):
        raise FileNotFoundError(
            f"Memorybank not found: {mb_path}\n"
            f"Run scripts/03_build_memorybank.py first."
        )

    with open(mb_path, "rb") as f:
        obj = pickle.load(f)

    mb = obj["memorybank"]
    knn = obj["knn"]

    test_ds = MVTecTest(cfg.mvtec_root, args.category)
    dl = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=cfg.num_workers)

    # overlays only meaningful in patch mode
    save_dir = None
    if cfg.save_heatmaps and args.mode == "patch":
        save_dir = f"runs/plots/overlays_{args.run_name}_{args.category}_{'simclr' if args.use_simclr else 'imagenet'}_{args.mode}_seed{cfg.seed}"

    res = evaluate_category(
        encoder, knn, mb, dl,
        device=device,
        save_dir=save_dir,
        save_heatmaps=cfg.save_heatmaps
    )

    row = {
        "timestamp": pd.Timestamp.utcnow().isoformat(),
        "run_name": args.run_name,
        "method": "simclr_knn" if args.use_simclr else "imagenet_knn",
        "mode": args.mode,
        "category": args.category,
        "backbone": cfg.backbone,
        "seed": cfg.seed,
        "knn_k": cfg.knn_k,
        "pca_dim": cfg.pca_dim,
        **res
    }

    ensure_dir("runs")
    csv_path = "runs/runs.csv"

    df_row = pd.DataFrame([row])

    # If csv exists but has no valid header, delete it (prevent corruption)
    if os.path.exists(csv_path) and not _csv_has_header(csv_path):
        print("⚠️ runs.csv exists but header is invalid/corrupt -> deleting and recreating:", csv_path)
        os.remove(csv_path)

    # write header if file doesn't exist
    write_header = not os.path.exists(csv_path)

    df_row.to_csv(csv_path, mode="a", header=write_header, index=False)

    print("Eval:", row)


if __name__ == "__main__":
    main()
