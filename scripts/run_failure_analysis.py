import argparse
import os
import sys

# Ensure repository root is in sys.path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

import torch
from torch.utils.data import DataLoader

from src.utils import seed_everything
from src.mvtec import MVTecTrainNormal, MVTecTest
from src.methods.patchcore import PatchCore
from src.methods.padim import PaDiM
from src.methods.autoencoder import ConvAutoencoder
from src.analysis.failure_catalog import FailureMiner


def main():
    parser = argparse.ArgumentParser(description="Run Automated Failure Mining")
    parser.add_argument("--category", type=str, default="bottle")
    parser.add_argument("--method", type=str, default="patchcore")
    parser.add_argument("--data-root", type=str, default="data/mvtec_ad")
    parser.add_argument("--output-dir", type=str, default="results/figures/failure_cases")
    parser.add_argument("--top-k", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"=== Mining Failure Cases for [{args.category}] with [{args.method}] on {device} ===")

    # 1. Fit Model
    train_ds = MVTecTrainNormal(args.data_root, args.category)
    train_loader = DataLoader(train_ds, batch_size=4, shuffle=True)

    if args.method.lower() == "patchcore":
        model = PatchCore(device=device)
    elif args.method.lower() == "padim":
        model = PaDiM(device=device)
    elif args.method.lower() == "autoencoder":
        model = ConvAutoencoder(device=device, epochs=20)
    else:
        raise ValueError(f"Unknown method: {args.method}")

    model.fit(train_loader)

    # 2. Mine Failures on Test Set
    test_ds = MVTecTest(args.data_root, args.category)
    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    miner = FailureMiner(model, test_loader, output_dir=args.output_dir, device=device)
    failures = miner.mine_failures(top_k=args.top_k)

    saved_images = miner.save_diagnostic_grids(failures, category=args.category, method_name=args.method)

    print(f"Mined {len(saved_images)} diagnostic images:")
    for path in saved_images:
        print(f"  -> {path}")


if __name__ == "__main__":
    main()
