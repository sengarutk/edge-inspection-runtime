# scripts/10_run_grid_multiseed.py
import subprocess
import sys
from pathlib import Path

PY = sys.executable


def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)


def main():
    categories = ["bottle", "cable", "hazelnut", "metal_nut"]
    seeds = [42, 43, 44]

    # grid configs: (use_simclr, mode)
    configs = [
        (False, "global"),  # imagenet global
        (False, "patch"),   # imagenet patch
        (True, "global"),   # simclr global
        (True, "patch"),    # simclr patch
    ]

    # ---------------------------
    # 1) SimCLR pretrain (all seeds/categories)
    # ---------------------------
    for seed in seeds:
        for cat in categories:
            run([PY, "-m", "scripts.02_pretrain_simclr",
                 "--category", cat,
                 "--epochs", "10",
                 "--seed", str(seed)])

    # ---------------------------
    # 2) Build memorybanks
    # ---------------------------
    for seed in seeds:
        for cat in categories:
            for use_simclr, mode in configs:
                cmd = [PY, "-m", "scripts.03_build_memorybank",
                       "--category", cat,
                       "--seed", str(seed),
                       "--mode", mode]
                if use_simclr:
                    cmd.append("--use_simclr")
                run(cmd)

    # ---------------------------
    # 3) Eval anomaly (writes runs/runs.csv)
    # ---------------------------
    run_name = "final_multiseed"
    for seed in seeds:
        for cat in categories:
            for use_simclr, mode in configs:
                cmd = [PY, "-m", "scripts.04_eval_anomaly",
                       "--category", cat,
                       "--seed", str(seed),
                       "--mode", mode,
                       "--run_name", run_name]
                if use_simclr:
                    cmd.append("--use_simclr")
                run(cmd)

    print("\nDONE: multi-seed benchmark completed.")


if __name__ == "__main__":
    main()
