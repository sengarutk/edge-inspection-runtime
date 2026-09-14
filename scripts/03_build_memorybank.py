import argparse
import pickle

import torch
from torch.utils.data import DataLoader

from src.config import TrainConfig
from src.utils import seed_everything, ensure_dir
from src.mvtec import MVTecTrainNormal
from src.models import Encoder
from src.anomaly import (
    build_memorybank_patch,
    build_memorybank_global,
    fit_knn,
    score_image_patchwise,
    score_image_global,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--use_simclr", action="store_true")
    ap.add_argument("--mode", type=str, default="patch", choices=["patch", "global"])
    args = ap.parse_args()

    cfg = TrainConfig(seed=args.seed)
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    encoder = Encoder(cfg.backbone, out_dim=512).to(device)

    if args.use_simclr:
        w = f"runs/artifacts/simclr_encoder_{args.category}_seed{cfg.seed}.pt"
        encoder.load_state_dict(torch.load(w, map_location=device))
        print("Loaded SimCLR weights:", w)
    else:
        print("Using ImageNet pretrained weights only.")

    train_ds = MVTecTrainNormal(cfg.mvtec_root, args.category)
    dl = DataLoader(train_ds, batch_size=32, shuffle=False, num_workers=cfg.num_workers)

    if args.mode == "patch":
        mb = build_memorybank_patch(encoder, dl, device=device, pca_dim=cfg.pca_dim)
        mb["score_fn"] = score_image_patchwise
    else:
        mb = build_memorybank_global(encoder, dl, device=device, pca_dim=cfg.pca_dim)
        mb["score_fn"] = score_image_global

    knn = fit_knn(mb, knn_k=cfg.knn_k)

    ensure_dir("runs/artifacts")
    out_path = f"runs/artifacts/memorybank_{args.category}_{'simclr' if args.use_simclr else 'imagenet'}_{args.mode}_seed{cfg.seed}.pkl"
    with open(out_path, "wb") as f:
        pickle.dump({"memorybank": mb, "knn": knn}, f)

    print("Saved memorybank:", out_path)


if __name__ == "__main__":
    main()
