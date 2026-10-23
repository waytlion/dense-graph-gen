import os
import argparse
import numpy as np
import networkx as nx
import matplotlib.pyplot as plt


def partition_graph(g_comp: nx.Graph):
    """
    Partition graph nodes into two groups (V_1, V_2).
    If g_comp (complement) is disconnected, use connected components.
    Otherwise, use spectral bisection on the normalized Laplacian of g_comp.
    """
    nodes = list(g_comp.nodes())
    n = len(nodes)
    if n <= 1:
        return set(nodes), set()

    # Case 1: Complement is disconnected (exact join structure)
    if not nx.is_connected(g_comp):
        comps = sorted(list(nx.connected_components(g_comp)), key=len, reverse=True)
        # Put largest component in V_1, remaining components in V_2
        v_1 = set(comps[0])
        v_2 = set().union(*comps[1:])
        return v_1, v_2

    # Case 2: Complement is connected (imperfect join) -> Spectral bisection
    L = nx.normalized_laplacian_matrix(g_comp).toarray()
    evals, evecs = np.linalg.eigh(L)

    # Use Fiedler vector (eigenvector corresponding to 2nd smallest eigenvalue)
    fiedler = evecs[:, 1]
    med = np.median(fiedler)

    # Try median cut and zero cut, choose whichever yields fewer cross-edges
    candidates = [
        set(nodes[i] for i in range(n) if fiedler[i] >= med),
        set(nodes[i] for i in range(n) if fiedler[i] >= 0.0),
    ]

    best_v1 = None
    min_cut = float("inf")
    for cand in candidates:
        if 0 < len(cand) < n:
            cand_v2 = set(nodes) - cand
            cut_size = sum(1 for u, v in g_comp.edges() if (u in cand and v in cand_v2))
            if cut_size < min_cut:
                min_cut = cut_size
                best_v1 = cand

    if best_v1 is None or len(best_v1) == 0 or len(best_v1) == n:
        # Fallback: simple half split sorted by Fiedler vector
        sorted_indices = np.argsort(fiedler)
        best_v1 = set(nodes[i] for i in sorted_indices[: n // 2])

    v_2 = set(nodes) - best_v1
    return best_v1, v_2


def plot_single_graph(adj: np.ndarray, out_path: str, idx: int = 0):
    """
    Plot dual-panel visualization for a single generated graph:
    Left: Generated graph G (intra-edges bold, join cross-edges faint).
    Right: Complement graph G_comp (intra-edges bold, violations in red dashed).
    """
    n = adj.shape[0]
    # Ensure binary and zero diagonal
    A = (adj > 0.5).astype(int)
    np.fill_diagonal(A, 0)

    G = nx.from_numpy_array(A)
    G_comp = nx.complement(G)

    v_1, v_2 = partition_graph(G_comp)
    is_disconnected = not nx.is_connected(G_comp)

    # Compute 2D spring layouts for each cluster separately
    sub_1 = G.subgraph(v_1)
    sub_2 = G.subgraph(v_2)
    conn_1 = nx.is_connected(sub_1) if len(v_1) > 0 else False
    conn_2 = nx.is_connected(sub_2) if len(v_2) > 0 else False
    subgraphs_connected = conn_1 and conn_2
    is_join_valid = is_disconnected and len(v_1) > 0 and len(v_2) > 0

    pos_1 = nx.spring_layout(sub_1, seed=42) if len(v_1) > 0 else {}
    pos_2 = nx.spring_layout(sub_2, seed=42) if len(v_2) > 0 else {}

    pos = {}
    for u in v_1:
        pos[u] = (pos_1[u][0] - 1.8, pos_1[u][1])
    for u in v_2:
        pos[u] = (pos_2[u][0] + 1.8, pos_2[u][1])

    # Node colors: Group 1 cyan (#4da6ff), Group 2 orange (#ff9933)
    node_colors = ["#4da6ff" if u in v_1 else "#ff9933" for u in G.nodes()]
    # Node borders: Red if its subgraph is disconnected, white otherwise
    node_edge_colors = [
        "#e60000" if (u in v_1 and not conn_1) or (u in v_2 and not conn_2) else "white"
        for u in G.nodes()
    ]
    node_linewidths = [2.5 if c == "#e60000" else 1.2 for c in node_edge_colors]

    # Edges in G
    intra_1_G = [(u, v) for u, v in G.edges() if u in v_1 and v in v_1]
    intra_2_G = [(u, v) for u, v in G.edges() if u in v_2 and v in v_2]
    cross_G = [(u, v) for u, v in G.edges() if (u in v_1 and v in v_2) or (v in v_1 and u in v_2)]

    # Edges in G_comp
    intra_1_comp = [(u, v) for u, v in G_comp.edges() if u in v_1 and v in v_1]
    intra_2_comp = [(u, v) for u, v in G_comp.edges() if u in v_2 and v in v_2]
    cross_comp = [(u, v) for u, v in G_comp.edges() if (u in v_1 and v in v_2) or (v in v_1 and u in v_2)]

    num_possible_cross = len(v_1) * len(v_2)
    violations = len(cross_comp)
    density_G = G.number_of_edges() / (n * (n - 1) / 2) if n > 1 else 0.0

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # --- Left: Generated Graph G ---
    nx.draw_networkx_edges(G, pos, edgelist=cross_G, ax=axes[0], alpha=0.15, edge_color="#888888", width=1.0)
    nx.draw_networkx_edges(G, pos, edgelist=intra_1_G, ax=axes[0], alpha=0.9, edge_color="#0066cc", width=2.5)
    nx.draw_networkx_edges(G, pos, edgelist=intra_2_G, ax=axes[0], alpha=0.9, edge_color="#cc5500", width=2.5)
    nx.draw_networkx_nodes(
        G,
        pos,
        ax=axes[0],
        node_color=node_colors,
        node_size=380,
        edgecolors=node_edge_colors,
        linewidths=node_linewidths,
    )
    nx.draw_networkx_labels(G, pos, ax=axes[0], font_size=8, font_weight="bold", font_color="black")
    axes[0].set_title(
        f"Generated Graph G #{idx} | N={n} | Density={density_G:.2f}\n"
        f"G1 (Blue): {len(v_1)} nodes (Conn: {conn_1}) | G2 (Orange): {len(v_2)} nodes (Conn: {conn_2})\n"
        f"Join Cross-Edges: {len(cross_G)} / {num_possible_cross} | Subgraphs Both Connected: {subgraphs_connected}"
    )
    axes[0].axis("off")
    axes[0].margins(0.1)

    # --- Right: Complement Graph G_comp ---
    if len(cross_comp) > 0:
        nx.draw_networkx_edges(
            G_comp,
            pos,
            edgelist=cross_comp,
            ax=axes[1],
            alpha=0.85,
            edge_color="#e60000",
            style="dashed",
            width=2.0,
            label="Join Violations",
        )
    nx.draw_networkx_edges(G_comp, pos, edgelist=intra_1_comp, ax=axes[1], alpha=0.8, edge_color="#0066cc", width=2.0)
    nx.draw_networkx_edges(G_comp, pos, edgelist=intra_2_comp, ax=axes[1], alpha=0.8, edge_color="#cc5500", width=2.0)
    nx.draw_networkx_nodes(
        G_comp,
        pos,
        ax=axes[1],
        node_color=node_colors,
        node_size=380,
        edgecolors=node_edge_colors,
        linewidths=node_linewidths,
    )
    nx.draw_networkx_labels(G_comp, pos, ax=axes[1], font_size=8, font_weight="bold", font_color="black")
    axes[1].set_title(
        f"Complement Graph G_comp | Disconnected: {is_disconnected}\n"
        f"Join Valid: {is_join_valid}\n"
        f"Violations (Cross-Edges in Comp): {violations} (Red dashed)"
    )
    axes[1].axis("off")
    axes[1].margins(0.1)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

    return {
        "is_join_valid": is_join_valid,
        "conn_1": conn_1,
        "conn_2": conn_2,
        "subgraphs_connected": subgraphs_connected,
    }


def main():
    parser = argparse.ArgumentParser(description="Plot generated cographs and complements.")
    parser.add_argument(
        "--adjs_path",
        type=str,
        default="external/DeFoG/outputs/2026-10-21/11-49-27-cograph_dense-cograph_dense/generated_adjs.npz",
        help="Path to generated_adjs.npz",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="plots/generated_dense",
        help="Directory to save output plots",
    )
    parser.add_argument(
        "--num_graphs",
        type=int,
        default=9,
        help="Number of graphs to visualize",
    )
    args = parser.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    print(f"Loading {args.adjs_path}...")
    data = np.load(args.adjs_path)
    keys = list(data.files)
    print(f"Total graphs available: {len(keys)}")

    num_to_plot = min(args.num_graphs, len(keys))
    valid_joins = 0
    connected_subgraphs_on_valid = 0

    for i in range(num_to_plot):
        adj = data[keys[i]]
        out_file = os.path.join(args.out_dir, f"graph_{i}.png")
        res = plot_single_graph(adj, out_file, idx=i)
        if res["is_join_valid"]:
            valid_joins += 1
            if res["subgraphs_connected"]:
                connected_subgraphs_on_valid += 1
        print(f"Saved {out_file} (Join Valid: {res['is_join_valid']}, Subgraphs Connected: {res['subgraphs_connected']})")

    print("\n--- Summary ---")
    print(f"Total evaluated: {num_to_plot}")
    print(f"Join validity: {valid_joins}/{num_to_plot} ({100 * valid_joins / num_to_plot:.1f}%)")
    if valid_joins > 0:
        print(f"Subgraph connectivity (on valid joins): {connected_subgraphs_on_valid}/{valid_joins} ({100 * connected_subgraphs_on_valid / valid_joins:.1f}%)")
    print("Visualization complete.")


if __name__ == "__main__":
    main()
