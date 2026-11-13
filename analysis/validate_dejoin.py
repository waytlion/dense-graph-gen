"""Validate dejoin on reference data: exact recovery + robustness to missing join edges.

Usage: python scripts/validate_dejoin.py --name cograph_zinc

Explanation: iteratively drop more join edges, and check wether our method identifies the 2 subgraphs correctly.
    - metrics:
        - drop=fraction of join edges droped
        - exact=share of graphs, where the two subgraphs are correctly identified
        - node_acc: the share of nodes assigned to the correct side.
        - deg/clust: the degree and clustering MMD between the de-joined graphs and the true unions; 0 means identical.
"""
import sys, argparse
import numpy as np, networkx as nx, torch
sys.path.insert(0, "external/DeFoG/src")
from analysis.spectre_utils import dejoin_partition, degree_stats, clustering_stats

p = argparse.ArgumentParser()
p.add_argument("--name", default="cograph_zinc")            # data/<name>_{dense,sparse}/
p.add_argument("--drops", default="0,0.01,0.05,0.1,0.2")    # fraction of join edges removed
p.add_argument("--n", type=int, default=200)                # number of test graphs used
a = p.parse_args()
rng = np.random.default_rng(0)

to_nx = lambda adj: nx.from_numpy_array(adj.numpy())
dense = torch.load(f"data/{a.name}_dense/test.pt")[:a.n]
ref = [to_nx(x) for x in torch.load(f"data/{a.name}_sparse/test.pt")[:a.n]]
val = [to_nx(x) for x in torch.load(f"data/{a.name}_sparse/val.pt")[:a.n]]
print(f"floor (sparse test vs val): deg={degree_stats(ref, val):.4f} clust={clustering_stats(ref, val):.4f}")

for f in map(float, a.drops.split(",")):
    exact, node_acc, dej = 0, [], []
    for Ad, Gs in zip(dense, ref):
        G = to_nx(Ad)
        A = nx.node_connected_component(Gs, 0); B = set(Gs) - A   # true sides
        join = [e for e in G.edges() if (e[0] in A) != (e[1] in A)]
        drop = rng.choice(len(join), int(f * len(join)), replace=False)
        G.remove_edges_from([join[i] for i in drop])
        S, T, _ = dejoin_partition(G)
        exact += S in (A, B)
        node_acc.append(max(len(S & A) + len(T & B), len(S & B) + len(T & A)) / len(G))
        G.remove_edges_from([(u, v) for u, v in list(G.edges()) if (u in S) != (v in S)])
        dej.append(G)
    print(f"drop={f:.2f} exact={exact/len(dense):.3f} node_acc={np.mean(node_acc):.3f} "
          f"deg={degree_stats(ref, dej):.4f} clust={clustering_stats(ref, dej):.4f}")
