from typing import Dict, List
import networkx as nx
import numpy as np
import torch


def calculate_edge_density(adj: torch.Tensor, threshold: float) -> float:
    """
    Calculate undirected edge density: density = 2 * E / (N * (N - 1)).

    Args:
    - adj: 2D adjacency tensor of shape (N, N)
    - threshold: edge existence cutoff for continuous adjacency matrices

    Returns:
    - float edge density in [0.0, 1.0]
    """
    n = adj.shape[0]
    if n <= 1:
        return 0.0
    adj_bin = (adj > threshold).float()
    num_edges = torch.triu(adj_bin, diagonal=1).sum().item()
    max_edges = n * (n - 1) / 2.0
    return float(num_edges / max_edges)


def is_valid_join(adj: torch.Tensor, threshold: float) -> bool:
    """
    Topological validity check for join graph:
    A graph is a join of two non-empty graphs iff its complement is disconnected.

    Args:
    - adj: 2D adjacency tensor of shape (N, N)
    - threshold: edge existence cutoff for continuous adjacency matrices

    Returns:
    - True if complement graph is disconnected, False otherwise.
    """
    n = adj.shape[0]

    # Complement adjacency matrix: 1 where edge does not exist (adj <= threshold)
    comp_adj = 1 - (adj > threshold).long()
    comp_adj.fill_diagonal_(0)

    # Check connectivity of complement graph
    G_comp = nx.from_numpy_array(comp_adj.cpu().numpy())
    return not nx.is_connected(G_comp)


def is_valid_union(adj: torch.Tensor, threshold: float) -> bool:
    """
    Topological validity check for union graph:
    A graph is a disjoint union iff the graph itself is disconnected.

    Args:
    - adj: 2D adjacency tensor of shape (N, N)
    - threshold: edge existence cutoff for continuous adjacency matrices

    Returns:
    - True if graph is disconnected, False otherwise.
    """
    n = adj.shape[0]
    if n < 2:
        return False

    adj_bin = (adj > threshold).long()
    adj_bin.fill_diagonal_(0)

    G = nx.from_numpy_array(adj_bin.cpu().numpy())
    return not nx.is_connected(G)


def validate_dataset(
    adjs: List[torch.Tensor],
    mode: str,
    threshold: float,
) -> Dict[str, float]:
    """
    Compute validation metrics across a dataset of adjacency matrices.

    Args:
    - adjs: list of adjacency tensors
    - mode: "join" (dense) or "union" (sparse)
    - threshold: edge existence cutoff for continuous adjacency matrices

    Returns:
    - dictionary containing validity_rate, mean_density, std_density, mean_nodes
    """
    valid_count = 0
    densities = []
    node_counts = []

    for adj in adjs:
        node_counts.append(adj.shape[0])
        densities.append(calculate_edge_density(adj, threshold=threshold))

        if mode == "join":
            if is_valid_join(adj, threshold=threshold):
                valid_count += 1
        elif mode == "union":
            if is_valid_union(adj, threshold=threshold):
                valid_count += 1
        else:
            raise ValueError(f"Unknown mode: {mode}")

    total = len(adjs)
    return {
        "num_graphs": float(total),
        "validity_rate": float(valid_count / total) if total > 0 else 0.0,
        "mean_density": float(np.mean(densities)) if densities else 0.0,
        "std_density": float(np.std(densities)) if densities else 0.0,
        "mean_nodes": float(np.mean(node_counts)) if node_counts else 0.0,
    }
