import os
import time
import argparse

import torch
from torch.utils.data import DataLoader
import torchvision.transforms as T
from tqdm import tqdm

from src.config import TrainConfig
from src.utils import seed_everything, ensure_dir
from src.mvtec import MVTecTrainNormal
from src.models import Encoder, ProjectionHead
from src.simclr import nt_xent_loss


class TwoViewDataset(torch.utils.data.Dataset):
    def __init__(self, base_ds, aug):
        self.base = base_ds
        self.aug = aug

    def __len__(self):
        return len(self.base)

    def __getitem__(self, idx):
        x = self.base[idx]  # tensor in [0,1]
        # convert back to PIL-like by tensor transforms: just apply tensor-aug
        return self.aug(x), self.aug(x)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--category", type=str, default="bottle")
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = TrainConfig(seed=args.seed, epochs=args.epochs)
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    aug = T.Compose([
        T.RandomResizedCrop((256, 256), scale=(0.6, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(0.2, 0.2, 0.2, 0.1),
    ])

    base = MVTecTrainNormal(cfg.mvtec_root, args.category)
    ds = TwoViewDataset(base, aug)
    dl = DataLoader(ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, drop_last=True)

    encoder = Encoder(cfg.backbone, out_dim=512).to(device)
    proj = ProjectionHead(512, cfg.proj_dim).to(device)

    opt = torch.optim.AdamW(
        list(encoder.parameters()) + list(proj.parameters()),
        lr=cfg.lr, weight_decay=cfg.weight_decay
    )

    encoder.train(); proj.train()
    for ep in range(cfg.epochs):
        losses = []
        t0 = time.time()
        for x1, x2 in tqdm(dl, desc=f"SimCLR {args.category} ep {ep+1}/{cfg.epochs}", leave=False):
            x1 = x1.to(device); x2 = x2.to(device)
            h1, _ = encoder(x1)
            h2, _ = encoder(x2)
            z1 = proj(h1)
            z2 = proj(h2)
            loss = nt_xent_loss(z1, z2, temperature=cfg.temperature)

            opt.zero_grad()
            loss.backward()
            opt.step()

            losses.append(loss.item())

        print(f"[SimCLR] cat={args.category} epoch={ep+1} loss={sum(losses)/len(losses):.4f} time={time.time()-t0:.1f}s")

    ensure_dir("runs/artifacts")
    out_path = f"runs/artifacts/simclr_encoder_{args.category}_seed{cfg.seed}.pt"
    torch.save(encoder.state_dict(), out_path)
    print("Saved:", out_path)


if __name__ == "__main__":
    main()
