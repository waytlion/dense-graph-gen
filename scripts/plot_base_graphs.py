#!/usr/bin/env python3
"""
Plot sample base graphs for QM9, ZINC, PROTEINS, and COLLAB.
Displays a 4x4 visual grid showing structural differences across datasets.
"""

import sys
import os
import random
import matplotlib.pyplot as plt
import networkx as nx
import torch

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from data_gen.data_gen import load_and_filter_base_graphs

DATASETS = ["qm9", "zinc", "proteins", "collab"]
N_SAMPLES = 4
OUT_PATH = "plots/base_graphs_samples.png"
os.makedirs("plots", exist_ok=True)

# Color theme per dataset
DATASET_COLORS = {
    "qm9": "#386cb0",      # Deep blue
    "zinc": "#7fc97f",     # Soft green
    "proteins": "#fdc086", # Warm orange
    "collab": "#beaed4",   # Purple
}


def main():
    random.seed(42)
    fig, axes = plt.subplots(len(DATASETS), N_SAMPLES, figsize=(16, 14))
    fig.suptitle("Sample Base Graphs Across Datasets", fontsize=18, fontweight="bold", y=0.98)

    for row_idx, ds_name in enumerate(DATASETS):
        print(f"Loading base graphs for {ds_name}...")
        try:
            graphs = load_and_filter_base_graphs(dataset_name=ds_name)
        except Exception as e:
            print(f"Error loading {ds_name}: {e}")
            continue

        sample_indices = random.sample(range(len(graphs)), N_SAMPLES)
        color = DATASET_COLORS.get(ds_name, "#4daf4a")

        for col_idx, s_idx in enumerate(sample_indices):
            ax = axes[row_idx, col_idx]
            G = graphs[s_idx]
            n_nodes = G.number_of_nodes()
            n_edges = G.number_of_edges()
            density = nx.density(G)

            # Node sizing based on graph size
            node_size = max(20, min(180, int(1500 / n_nodes)))

            pos = nx.spring_layout(G, seed=42, k=1.0 / (n_nodes ** 0.4))
            nx.draw_networkx_nodes(
                G,
                pos,
                ax=ax,
                node_size=node_size,
                node_color=color,
                edgecolors="black",
                linewidths=0.8,
            )
            nx.draw_networkx_edges(
                G,
                pos,
                ax=ax,
                alpha=0.6,
                edge_color="#444444",
                width=1.0,
            )

            ax.set_title(f"N={n_nodes}, E={n_edges}, D={density:.2f}", fontsize=11)
            ax.axis("off")

            # Row label on the leftmost column
            if col_idx == 0:
                ax.text(
                    -0.2,
                    0.5,
                    ds_name.upper(),
                    transform=ax.transAxes,
                    fontsize=14,
                    fontweight="bold",
                    va="center",
                    ha="right",
                    rotation=0,
                )

    plt.tight_layout(rect=[0.05, 0.02, 1, 0.95])
    plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight")
    plt.close()
    print(f"Saved: {OUT_PATH}")


if __name__ == "__main__":
    main()
