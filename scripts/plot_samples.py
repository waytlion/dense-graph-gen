import os
import matplotlib.pyplot as plt
import networkx as nx
import torch

from src.data.base_graphs import load_and_filter_qm9
from src.data.cographs import create_pair_graph
from src.eval.topology import calculate_edge_density, is_valid_join, is_valid_union


def plot_sample_cographs(
    out_path: str = "plots/sample_cographs.png",
    seed: int = 42,
) -> None:
    """
    Generate and plot visual comparison between Join (dense) and Union (sparse)
    for sample graph pairs.
    """
    os.makedirs(os.path.dirname(out_path), exist_ok=True)

    graphs = load_and_filter_qm9()
    g_cpu = torch.Generator().manual_seed(seed)
    pair = torch.randperm(len(graphs), generator=g_cpu)[:2].tolist()
    g_a, g_b = graphs[pair[0]], graphs[pair[1]]

    n_a = g_a.number_of_nodes()
    n_b = g_b.number_of_nodes()

    adj_join = create_pair_graph(g_a, g_b, mode="join")
    adj_union = create_pair_graph(g_a, g_b, mode="union")

    G_join = nx.from_numpy_array(adj_join.numpy())
    G_union = nx.from_numpy_array(adj_union.numpy())

    # Color nodes: Group A in cyan, Group B in orange
    node_colors = ["#4da6ff" if i < n_a else "#ff9933" for i in range(n_a + n_b)]

    # 2D spring layout for each subgraph, placed side-by-side with horizontal offset
    pos_a = nx.spring_layout(g_a, seed=seed)
    pos_b = nx.spring_layout(g_b, seed=seed)
    pos = {}
    for i in range(n_a):
        pos[i] = (pos_a[i][0] - 1.8, pos_a[i][1])
    for j in range(n_b):
        pos[n_a + j] = (pos_b[j][0] + 1.8, pos_b[j][1])

    # Partition edges: intra-subgraph (A, B) vs inter-subgraph (join cross-edges)
    edges_intra_a = [(u, v) for u, v in G_join.edges() if u < n_a and v < n_a]
    edges_intra_b = [(u, v) for u, v in G_join.edges() if u >= n_a and v >= n_a]
    edges_cross = [
        (u, v) for u, v in G_join.edges() if (u < n_a and v >= n_a) or (v < n_a and u >= n_a)
    ]

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # Dense Join plot
    nx.draw_networkx_edges(G_join, pos, edgelist=edges_cross, ax=axes[0], alpha=0.15, edge_color="#888888", width=1.0)
    nx.draw_networkx_edges(G_join, pos, edgelist=edges_intra_a, ax=axes[0], alpha=0.9, edge_color="#0066cc", width=2.5)
    nx.draw_networkx_edges(G_join, pos, edgelist=edges_intra_b, ax=axes[0], alpha=0.9, edge_color="#cc5500", width=2.5)
    nx.draw_networkx_nodes(G_join, pos, ax=axes[0], node_color=node_colors, node_size=380, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G_join, pos, ax=axes[0], font_size=9, font_weight="bold", font_color="black")
    d_join = calculate_edge_density(adj_join, threshold=0.5)
    axes[0].set_title(
        f"Join (Dense) | N={n_a+n_b} | Density={d_join:.2f}\n"
        f"Complement Disconnected: {is_valid_join(adj_join, threshold=0.5)}\n"
        f"(Left: Graph A [N={n_a}], Right: Graph B [N={n_b}])"
    )
    axes[0].axis("off")
    axes[0].margins(0.1)

    # Sparse Union plot
    union_intra_a = [(u, v) for u, v in G_union.edges() if u < n_a and v < n_a]
    union_intra_b = [(u, v) for u, v in G_union.edges() if u >= n_a and v >= n_a]
    nx.draw_networkx_edges(G_union, pos, edgelist=union_intra_a, ax=axes[1], alpha=0.9, edge_color="#0066cc", width=2.5)
    nx.draw_networkx_edges(G_union, pos, edgelist=union_intra_b, ax=axes[1], alpha=0.9, edge_color="#cc5500", width=2.5)
    nx.draw_networkx_nodes(G_union, pos, ax=axes[1], node_color=node_colors, node_size=380, edgecolors="white", linewidths=1.2)
    nx.draw_networkx_labels(G_union, pos, ax=axes[1], font_size=9, font_weight="bold", font_color="black")
    d_union = calculate_edge_density(adj_union, threshold=0.5)
    axes[1].set_title(
        f"Union (Sparse) | N={n_a+n_b} | Density={d_union:.2f}\n"
        f"Graph Disconnected: {is_valid_union(adj_union, threshold=0.5)}\n"
        f"(Left: Graph A [N={n_a}], Right: Graph B [N={n_b}])"
    )
    axes[1].axis("off")
    axes[1].margins(0.1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()
    print(f"Saved visualization to {out_path}")


if __name__ == "__main__":
    plot_sample_cographs()
