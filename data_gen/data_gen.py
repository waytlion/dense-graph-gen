import os
from typing import List, Tuple
import torch
from torch_geometric.datasets import QM9
from torch_geometric.utils import to_networkx, from_networkx
import networkx as nx
from tqdm import tqdm


def load_and_filter_qm9(
    root: str = "data/qm9",
    cache_path: str = "data/qm9_base_unique.pt",
    min_nodes: int = 6,
    max_nodes: int = 9,
    remove_hydrogens: bool = True,
) -> List[nx.Graph]:
    """
    Step 1: Load QM9, filter by node count N in [min_nodes, max_nodes],
    strip node/edge features, and remove isomorphic duplicates.

    Parameters:
    - root: dir with raw and processed QM9 data
    - cache_path: path to load/save cached list of unique NetworkX graphs
    - min_nodes: min number of nodes N per G
    - max_nodes: max number of nodes N per G
    - remove_hydrogens: if True, removes hydrogen atoms (atomic number Z == 1),
      keeping only heavy atoms (C, N, O, F).

    Returns:
    - List of unique, unlabeled NetworkX graphs with N in [min_nodes, max_nodes].
    """
    if cache_path and os.path.exists(cache_path):
        print(f"Loading cached base graphs from {cache_path}...")
        return torch.load(cache_path, weights_only=False)

    dataset = QM9(root=root) # on first exec load from PyG

    unique_graphs: List[nx.Graph] = []
    seen_hashes = set()

    print("Filtering graphs and removing attributes/duplicates...")
    for data in tqdm(dataset):
        G = to_networkx(data, to_undirected=True) # Convert PyG Data -> NetworkX graph
        if remove_hydrogens:
            heavy_nodes = [i for i, z_val in enumerate(data.z) if z_val.item() > 1] # data.z is the atomic number tensor: 1 corresponds to Hydrogen (H)
            G = G.subgraph(heavy_nodes).copy()
            G = nx.convert_node_labels_to_integers(G) # without relabling -> Adj matrix creation causes out of bounds error (Relabel node indices to be 0-indexed [0, ..., N-1]

        # Filter by N
        num_nodes = G.number_of_nodes()
        if not (min_nodes <= num_nodes <= max_nodes):
            continue

        # Strip all node and edge attributes (unlabeled graph structure only)
        G_unlabeled = nx.Graph()
        G_unlabeled.add_nodes_from(range(num_nodes))
        G_unlabeled.add_edges_from(G.edges())

        # Deduplication via Weisfeiler-Lehman graph hash
        wl_hash = nx.weisfeiler_lehman_graph_hash(G_unlabeled)
        if wl_hash not in seen_hashes:
            seen_hashes.add(wl_hash)
            unique_graphs.append(G_unlabeled)

    print(f"Extraction complete: {len(unique_graphs)} unique graphs found.")
    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        torch.save(unique_graphs, cache_path)
        print(f"Cached {len(unique_graphs)} graphs to {cache_path}")

    return unique_graphs


def split_base_graphs(
    graphs: List[nx.Graph],
    train_ratio: float = 0.8,
    val_ratio: float = 0.1,
    test_ratio: float = 0.1,
    seed: int = 42,
) -> Tuple[List[nx.Graph], List[nx.Graph], List[nx.Graph]]:
    """
    Step 2: Split base graphs into train, val, and test subsets (80/10/10).

    Args:
    - graphs: list of unique base graphs
    - train_ratio: fraction for train split (0.8)
    - val_ratio: fraction for val split (0.1)
    - test_ratio: fraction for test split (0.1)
    - seed: random seed for deterministic shuffling

    Returns:
    - (train_graphs, val_graphs, test_graphs)
    """
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1.0"

    #random shuffle graph indices 
    num_total = len(graphs)
    g_cpu = torch.Generator().manual_seed(seed)
    perm = torch.randperm(num_total, generator=g_cpu).tolist()

    # non-overlapping intervals of the single permutation of indices
    num_train = int(round(num_total * train_ratio))
    num_val = int(round(num_total * val_ratio))
    train_indices = perm[:num_train]
    val_indices = perm[num_train : num_train + num_val]
    test_indices = perm[num_train + num_val :]
    train_graphs = [graphs[i] for i in train_indices]
    val_graphs = [graphs[i] for i in val_indices]
    test_graphs = [graphs[i] for i in test_indices]
    print(
        f"Split base graphs: {len(train_graphs)} train, "
        f"{len(val_graphs)} val, {len(test_graphs)} test (seed={seed})"
    )
    return train_graphs, val_graphs, test_graphs


def create_pair_graph(G_A: nx.Graph, G_B: nx.Graph, mode: str = "join") -> torch.Tensor:
    """
    Combine two base graphs G_A and G_B into a single adjacency matrix.

    Args:
    - G_A: first NetworkX base graph with N_A nodes
    - G_B: second NetworkX base graph with N_B nodes
    - mode: "join" (dense: all cross-edges between A and B) or
            "union" (sparse: zero cross-edges between A and B)

    Returns:
    - Adjacency matrix as torch.FloatTensor of shape (N_A + N_B, N_A + N_B)
    """
    n_a = G_A.number_of_nodes(); n_b = G_B.number_of_nodes()
    n_total = n_a + n_b
    adj = torch.zeros((n_total, n_total), dtype=torch.float32)

    # Fill internal edges for G_A (top-left block)
    for u, v in G_A.edges():
        adj[u, v] = 1.0
        adj[v, u] = 1.0

    # Fill internal edges for G_B (bottom-right block, offset by n_a)
    for u, v in G_B.edges():
        adj[n_a + u, n_a + v] = 1.0
        adj[n_a + v, n_a + u] = 1.0

    # If join, connect every node of A to every node of B (bipartite cross-edges)
    if mode == "join":
        adj[:n_a, n_a:] = 1.0
        adj[n_a:, :n_a] = 1.0
    elif mode != "union":
        raise ValueError(f"Unknown mode: {mode}. Choose 'join' or 'union'.")

    return adj


def generate_paired_datasets(
    train_graphs: List[nx.Graph],
    val_graphs: List[nx.Graph],
    test_graphs: List[nx.Graph],
    n_train: int = 10000,
    n_val: int = 1250,
    n_test: int = 1250,
    dense_out_dir: str = "data/cograph_dense",
    sparse_out_dir: str = "data/cograph_sparse",
    seed: int = 42,
) -> None:
    """
    Step 3: Sample pairs of graphs with replacement (G_A, G_B), where G_A, != G_B
    -> From pairs generate datasets: 
        - generate paired Join (dense)
        - generate Union (sparse)

    Args:
    - train_graphs, val_graphs, test_graphs: split base graph lists
    - n_train, n_val, n_test: target number of generated graphs per split
    - dense_out_dir: folder to save dense (join) datasets (train.pt, val.pt, test.pt)
    - sparse_out_dir: folder to save sparse (union) datasets (train.pt, val.pt, test.pt)
    - seed: random seed for deterministic pair sampling
    """
    os.makedirs(dense_out_dir, exist_ok=True)
    os.makedirs(sparse_out_dir, exist_ok=True)

    g_cpu = torch.Generator().manual_seed(seed)
    splits = [
        ("train", train_graphs, n_train),
        ("val", val_graphs, n_val),
        ("test", test_graphs, n_test),
    ]

    for split_name, base_pool, n_target in splits:
        pool_size = len(base_pool)
        print(f"Generating {n_target} pairs for split '{split_name}' (pool={pool_size})...")

        dense_list: List[torch.Tensor] = []
        sparse_list: List[torch.Tensor] = []

        # Sample n_target pairs (idx_a, idx_b) with idx_a != idx_b
        for _ in range(n_target):
            pair = torch.randperm(pool_size, generator=g_cpu)[:2].tolist()
            g_a = base_pool[pair[0]], g_b = base_pool[pair[1]]

            # Construct dense (join) and sparse (union) adjacency matrices
            dense_list.append(create_pair_graph(g_a, g_b, mode="join"))
            sparse_list.append(create_pair_graph(g_a, g_b, mode="union"))

        # Save as standard DeFoG .pt files (list of adjacency matrices)
        torch.save(dense_list, os.path.join(dense_out_dir, f"{split_name}.pt"))
        torch.save(sparse_list, os.path.join(sparse_out_dir, f"{split_name}.pt"))
        print(f"Saved {split_name}.pt to {dense_out_dir} and {sparse_out_dir}")

    print("Step 3 complete: Dense and sparse paired datasets generated.")


if __name__ == "__main__":
    graphs = load_and_filter_qm9()
    train_graphs, val_graphs, test_graphs = split_base_graphs(graphs)
    generate_paired_datasets(train_graphs, val_graphs, test_graphs)
