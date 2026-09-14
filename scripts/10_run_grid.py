import subprocess
import sys

PY = sys.executable

CATS = ["bottle", "cable", "hazelnut", "metal_nut"]
SEEDS = [42, 43, 44]

def run(cmd):
    print("\n>>", " ".join(cmd))
    subprocess.check_call(cmd)

def main():
    # Pretrain SimCLR per category (short epochs to finish)
    for cat in CATS:
        run([PY, "-m", "scripts.02_pretrain_simclr", "--category", cat, "--epochs", "10", "--seed", "42"])

    # Multi-seed anomaly evaluation
    for seed in SEEDS:
        for cat in CATS:
            # ImageNet features baseline
            run([PY, "-m", "scripts.03_build_memorybank", "--category", cat, "--seed", str(seed)])
            run([PY, "-m", "scripts.04_eval_anomaly", "--category", cat, "--seed", str(seed), "--run_name", "grid"])

            # SimCLR features
            run([PY, "-m", "scripts.03_build_memorybank", "--category", cat, "--seed", str(seed), "--use_simclr"])
            run([PY, "-m", "scripts.04_eval_anomaly", "--category", cat, "--seed", str(seed), "--use_simclr", "--run_name", "grid"])

    run([PY, "-m", "scripts.05_make_plots"])

if __name__ == "__main__":
    main()
