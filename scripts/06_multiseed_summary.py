import os
import argparse
import pandas as pd

def fmt_mean_std(series):
    mu = series.mean()
    sd = series.std(ddof=1) if len(series) > 1 else 0.0
    return f"{mu:.4f} ± {sd:.4f}"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs_csv", type=str, default="runs/runs.csv")
    ap.add_argument("--out_csv", type=str, default="runs/summary_multiseed.csv")
    ap.add_argument("--out_md", type=str, default="runs/summary_multiseed.md")
    args = ap.parse_args()

    if not os.path.exists(args.runs_csv):
        raise FileNotFoundError(f"runs.csv not found at: {args.runs_csv}")

    df = pd.read_csv(args.runs_csv)

    required = ["category", "method", "mode", "seed", "auroc_image", "avg_latency_s"]
    missing = [c for c in required if c not in df.columns]
    if missing:
        raise ValueError(
            f"❌ Missing required columns in runs.csv: {missing}\n"
            f"Available columns: {list(df.columns)}"
        )

    # Aggregate mean/std across seeds per (method, mode, category)
    group_cols = ["method", "mode", "category"]
    agg = (
        df.groupby(group_cols)
        .agg(
            auroc_mean=("auroc_image", "mean"),
            auroc_std=("auroc_image", "std"),
            latency_mean=("avg_latency_s", "mean"),
            latency_std=("avg_latency_s", "std"),
            n_seeds=("seed", "nunique"),
        )
        .reset_index()
    )

    # Pretty columns for README
    agg["AUROC (mean±std)"] = agg.groupby(group_cols)["auroc_mean"].transform(lambda _: "")
    agg["AUROC (mean±std)"] = agg.apply(
        lambda r: f"{r['auroc_mean']:.4f} ± {0.0 if pd.isna(r['auroc_std']) else r['auroc_std']:.4f}",
        axis=1,
    )
    agg["Latency s/img (mean±std)"] = agg.apply(
        lambda r: f"{r['latency_mean']:.4f} ± {0.0 if pd.isna(r['latency_std']) else r['latency_std']:.4f}",
        axis=1,
    )

    table = agg[["method", "mode", "category", "n_seeds", "AUROC (mean±std)", "Latency s/img (mean±std)"]]
    table = table.sort_values(["category", "method", "mode"]).reset_index(drop=True)

    os.makedirs(os.path.dirname(args.out_csv), exist_ok=True)
    table.to_csv(args.out_csv, index=False)

    # Markdown table for README
    md = table.to_markdown(index=False)
    with open(args.out_md, "w", encoding="utf-8") as f:
        f.write("# Multi-seed Summary (mean ± std)\n\n")
        f.write(md)
        f.write("\n")

    print(f"Saved CSV summary: {args.out_csv}")
    print(f"Saved Markdown summary: {args.out_md}")
    print("\nPreview:\n")
    print(md)

if __name__ == "__main__":
    main()
