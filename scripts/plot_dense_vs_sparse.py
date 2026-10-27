#!/usr/bin/env python3
"""
Generate side-by-side comparison plots for Dense vs Sparse cograph experiments.
Reads val_metrics_history.csv from external/DeFoG/outputs/2026-10-26/
and outputs high-resolution figures to plots/comparison/.
"""

import os
import glob
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker

# Output directory
OUT_DIR = "plots/comparison"
os.makedirs(OUT_DIR, exist_ok=True)

# Datasets and their respective folder patterns
DATASETS = {
    "QM9": {
        "dense": "external/DeFoG/outputs/2026-10-26/*-cograph_dense-cograph_dense/val_metrics_history.csv",
        "sparse": "external/DeFoG/outputs/2026-10-26/*-cograph_sparse-cograph_sparse/val_metrics_history.csv",
    },
    "ZINC": {
        "dense": "external/DeFoG/outputs/2026-10-26/*-cograph_zinc_dense-*/val_metrics_history.csv",
        "sparse": "external/DeFoG/outputs/2026-10-26/*-cograph_zinc_sparse-*/val_metrics_history.csv",
    },
    "PROTEINS": {
        "dense": "external/DeFoG/outputs/2026-10-26/*-cograph_proteins_dense-*/val_metrics_history.csv",
        "sparse": "external/DeFoG/outputs/2026-10-26/*-cograph_proteins_sparse-*/val_metrics_history.csv",
    },
    "COLLAB": {
        "dense": "external/DeFoG/outputs/2026-10-26/*-cograph_collab_dense-*/val_metrics_history.csv",
        "sparse": "external/DeFoG/outputs/2026-10-26/*-cograph_collab_sparse-*/val_metrics_history.csv",
    },
}

# Style aesthetics
plt.rcParams.update({
    "font.size": 12,
    "axes.labelsize": 13,
    "axes.titlesize": 14,
    "xtick.labelsize": 11,
    "ytick.labelsize": 11,
    "legend.fontsize": 11,
    "figure.titlesize": 16,
    "axes.grid": True,
    "grid.alpha": 0.4,
    "grid.linestyle": "--",
})

DENSE_COLOR = "#D95F02"   # Red-Orange
SPARSE_COLOR = "#1B9E77"  # Teal-Green


def load_csv(pattern: str) -> pd.DataFrame:
    matches = sorted(glob.glob(pattern))
    if not matches:
        return None
    return pd.read_csv(matches[-1])


def plot_dataset_curves(name: str, df_dense: pd.DataFrame, df_sparse: pd.DataFrame):
    """Plot 6-panel comparison for a single dataset."""
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    fig.suptitle(f"{name} — Dense (Join) vs. Sparse (Union) Training Curves", fontweight="bold", y=0.98)

    # 1. Topological Validity
    ax = axes[0, 0]
    if "cograph_join_acc" in df_dense.columns:
        ax.plot(df_dense["epoch"], df_dense["cograph_join_acc"] * 100, label="Dense (Join Validity)", color=DENSE_COLOR, marker="o", linewidth=2.2)
    if "cograph_union_acc" in df_sparse.columns:
        ax.plot(df_sparse["epoch"], df_sparse["cograph_union_acc"] * 100, label="Sparse (Union Validity)", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2.2)
    ax.set_ylabel("Validity (%)")
    ax.set_title("Cograph Topological Validity")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=True)

    # 2. Constituent Subgraph Connectivity
    ax = axes[0, 1]
    dense_conn_col = [c for c in df_dense.columns if "subgraph_conn" in c]
    sparse_conn_col = [c for c in df_sparse.columns if "subgraph_conn" in c]
    if dense_conn_col:
        ax.plot(df_dense["epoch"], df_dense[dense_conn_col[0]] * 100, label="Dense (Subgraphs Conn)", color=DENSE_COLOR, marker="o", linewidth=2.2)
    if sparse_conn_col:
        ax.plot(df_sparse["epoch"], df_sparse[sparse_conn_col[0]] * 100, label="Sparse (Subgraphs Conn)", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2.2)
    ax.set_ylabel("Constituent Conn (%)")
    ax.set_title("Constituent Subgraphs Connected (%)")
    ax.set_ylim(-5, 105)
    ax.legend(frameon=True)

    # 3. Degree MMD (log scale)
    ax = axes[0, 2]
    ax.plot(df_dense["epoch"], df_dense["degree"], label="Dense", color=DENSE_COLOR, marker="o", linewidth=2)
    ax.plot(df_sparse["epoch"], df_sparse["degree"], label="Sparse", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2)
    ax.set_yscale("log")
    ax.set_ylabel("MMD (log scale)")
    ax.set_title("Degree Distribution MMD (lower is better)")
    ax.legend(frameon=True)

    # 4. Clustering MMD (log scale)
    ax = axes[1, 0]
    ax.plot(df_dense["epoch"], df_dense["clustering"], label="Dense", color=DENSE_COLOR, marker="o", linewidth=2)
    ax.plot(df_sparse["epoch"], df_sparse["clustering"], label="Sparse", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD (log scale)")
    ax.set_title("Clustering Coefficient MMD (lower is better)")
    ax.legend(frameon=True)

    # 5. Orbit MMD (log scale)
    ax = axes[1, 1]
    ax.plot(df_dense["epoch"], df_dense["orbit"], label="Dense", color=DENSE_COLOR, marker="o", linewidth=2)
    ax.plot(df_sparse["epoch"], df_sparse["orbit"], label="Sparse", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD (log scale)")
    ax.set_title("Orbit (Graphlet) MMD (lower is better)")
    ax.legend(frameon=True)

    # 6. Spectre MMD (log scale)
    ax = axes[1, 2]
    ax.plot(df_dense["epoch"], df_dense["spectre"], label="Dense", color=DENSE_COLOR, marker="o", linewidth=2)
    ax.plot(df_sparse["epoch"], df_sparse["spectre"], label="Sparse", color=SPARSE_COLOR, marker="s", linestyle="--", linewidth=2)
    ax.set_yscale("log")
    ax.set_xlabel("Epoch")
    ax.set_ylabel("MMD (log scale)")
    ax.set_title("Spectral Eigenvalue MMD (lower is better)")
    ax.legend(frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(OUT_DIR, f"{name.lower()}_dense_vs_sparse.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def plot_cross_dataset_summary(summary_records):
    """Plot cross-dataset comparison bar charts at latest epoch."""
    if not summary_records:
        return
    df = pd.DataFrame(summary_records)
    
    metrics_to_plot = [
        ("subgraph_conn", "Constituent Subgraph Conn (%)", False),
        ("orbit", "Orbit MMD (log scale)", True),
        ("degree", "Degree MMD (log scale)", True),
        ("clustering", "Clustering MMD (log scale)", True),
    ]

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle("Cross-Dataset Performance Gap: Dense vs. Sparse", fontweight="bold", y=0.98)

    datasets = df["dataset"].unique()
    x = range(len(datasets))
    width = 0.35

    for idx, (m_col, m_title, is_log) in enumerate(metrics_to_plot):
        ax = axes[idx // 2, idx % 2]
        
        dense_vals = [df[(df["dataset"] == d) & (df["mode"] == "Dense")][m_col].values[0] for d in datasets]
        sparse_vals = [df[(df["dataset"] == d) & (df["mode"] == "Sparse")][m_col].values[0] for d in datasets]
        
        bars1 = ax.bar([i - width/2 for i in x], dense_vals, width, label="Dense (Join)", color=DENSE_COLOR)
        bars2 = ax.bar([i + width/2 for i in x], sparse_vals, width, label="Sparse (Union)", color=SPARSE_COLOR)
        
        if is_log:
            ax.set_yscale("log")
        else:
            ax.set_ylim(0, 105)

        ax.set_xticks(x)
        ax.set_xticklabels(datasets)
        ax.set_title(m_title)
        ax.legend(frameon=True)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    out_file = os.path.join(OUT_DIR, "all_datasets_dense_vs_sparse_summary.png")
    plt.savefig(out_file, dpi=300)
    plt.close()
    print(f"Saved: {out_file}")


def main():
    summary_records = []

    for name, paths in DATASETS.items():
        df_dense = load_csv(paths["dense"])
        df_sparse = load_csv(paths["sparse"])

        if df_dense is None or df_sparse is None:
            print(f"Skipping {name}: missing files.")
            continue

        print(f"Plotting curves for {name} (Dense: {len(df_dense)} pts, Sparse: {len(df_sparse)} pts)...")
        plot_dataset_curves(name, df_dense, df_sparse)

        # Extract latest evaluation row for summary
        latest_dense = df_dense.iloc[-1]
        latest_sparse = df_sparse.iloc[-1]

        dense_conn_col = [c for c in df_dense.columns if "subgraph_conn" in c][0]
        sparse_conn_col = [c for c in df_sparse.columns if "subgraph_conn" in c][0]

        summary_records.append({
            "dataset": name,
            "mode": "Dense",
            "epoch": latest_dense["epoch"],
            "subgraph_conn": latest_dense[dense_conn_col] * 100,
            "degree": latest_dense["degree"],
            "clustering": latest_dense["clustering"],
            "orbit": latest_dense["orbit"],
            "spectre": latest_dense["spectre"],
        })
        summary_records.append({
            "dataset": name,
            "mode": "Sparse",
            "epoch": latest_sparse["epoch"],
            "subgraph_conn": latest_sparse[sparse_conn_col] * 100,
            "degree": latest_sparse["degree"],
            "clustering": latest_sparse["clustering"],
            "orbit": latest_sparse["orbit"],
            "spectre": latest_sparse["spectre"],
        })

    if summary_records:
        plot_cross_dataset_summary(summary_records)


if __name__ == "__main__":
    main()
