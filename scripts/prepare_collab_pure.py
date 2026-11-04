#!/usr/bin/env python3
"""
Prepare the full COLLAB dataset (all 5,000 graphs) for both DeFoG and DiPhon.

Output:
- DeFoG format: data/collab_pure/{train,val,test}.pt (lists of torch.Tensor adjacency matrices)
- DiPhon format: data/collab_pure.pkl (dict with 'train', 'val', 'test' lists of NetworkX graphs)
"""

import os
from pathlib import Path
import pickle
import random
import numpy as np
import networkx as nx
import torch
from torch_geometric.datasets import TUDataset
from torch_geometric.utils import to_networkx
from tqdm import tqdm

DATA_DIR = Path("data")
OUT_DEFOG_DIR = DATA_DIR / "collab_pure"
OUT_DIPHON_FILE = DATA_DIR / "collab_pure.pkl"

TRAIN_RATIO = 0.8
VAL_RATIO = 0.1
TEST_RATIO = 0.1
SEED = 42


def prepare_full_collab():
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print("Loading raw COLLAB from TUDataset...")
    raw_dataset = TUDataset(root=str(DATA_DIR / "collab"), name="COLLAB")
    print(f"Total raw graphs: {len(raw_dataset)}")

    graphs = []
    node_counts = []
    densities = []

    for data in tqdm(raw_dataset, desc="Converting graphs"):
        G = to_networkx(data, to_undirected=True)
        # Relabel nodes to 0..N-1 and remove node/edge attributes
        G_clean = nx.Graph()
        G_clean.add_nodes_from(range(G.number_of_nodes()))
        G_clean.add_edges_from(G.edges())

        num_nodes = G_clean.number_of_nodes()
        if num_nodes == 0:
            continue

        graphs.append(G_clean)
        node_counts.append(num_nodes)
        densities.append(nx.density(G_clean))

    total = len(graphs)
    print(f"Cleaned graphs: {total}")
    print(f"Node count range: [{min(node_counts)}, {max(node_counts)}], mean: {np.mean(node_counts):.1f}")
    print(f"Average density: {np.mean(densities):.4f}")

    # Shuffle deterministically
    indices = list(range(total))
    random.shuffle(indices)

    n_train = int(total * TRAIN_RATIO)
    n_val = int(total * VAL_RATIO)
    train_idx = indices[:n_train]
    val_idx = indices[n_train:n_train + n_val]
    test_idx = indices[n_train + n_val:]

    splits = {
        "train": [graphs[i] for i in train_idx],
        "val": [graphs[i] for i in val_idx],
        "test": [graphs[i] for i in test_idx],
    }

    print(f"Splits: train={len(splits['train'])}, val={len(splits['val'])}, test={len(splits['test'])}")

    # 1. Save DeFoG format (list of dense adjacency matrices as torch.FloatTensor)
    OUT_DEFOG_DIR.mkdir(parents=True, exist_ok=True)
    for split_name, split_graphs in splits.items():
        adj_list = [
            torch.from_numpy(nx.to_numpy_array(g, dtype=np.float32))
            for g in split_graphs
        ]
        out_pt = OUT_DEFOG_DIR / f"{split_name}.pt"
        torch.save(adj_list, out_pt)
        print(f"Saved {out_pt} ({len(adj_list)} graphs)")

    # 2. Save DiPhon format (.pkl dict of NetworkX graphs)
    with open(OUT_DIPHON_FILE, "wb") as f:
        pickle.dump(splits, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {OUT_DIPHON_FILE}")


if __name__ == "__main__":
    prepare_full_collab()

