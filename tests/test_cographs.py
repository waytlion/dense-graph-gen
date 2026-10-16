import networkx as nx
import torch
import pytest

from src.data.cographs import create_pair_graph
from src.eval.topology import (
    calculate_edge_density,
    is_valid_join,
    is_valid_union,
)


@pytest.fixture # data generator to test if math implemented correctly on knwon graphs 
def sample_pair():
    # Two small test graphs: Triangle (3 nodes) and Path (4 nodes)
    g_a = nx.cycle_graph(3)
    g_b = nx.path_graph(4)
    return g_a, g_b


def test_create_pair_graph_shapes(sample_pair):
    g_a, g_b = sample_pair
    adj_join = create_pair_graph(g_a, g_b, mode="join")
    adj_union = create_pair_graph(g_a, g_b, mode="union")

    expected_n = 3 + 4
    assert adj_join.shape == (expected_n, expected_n)
    assert adj_union.shape == (expected_n, expected_n)
    # No self-loops
    assert torch.equal(torch.diag(adj_join), torch.zeros(expected_n))
    assert torch.equal(torch.diag(adj_union), torch.zeros(expected_n))
    # Symmetric
    assert torch.equal(adj_join, adj_join.T)
    assert torch.equal(adj_union, adj_union.T)


def test_join_topological_validity(sample_pair):
    g_a, g_b = sample_pair
    adj_join = create_pair_graph(g_a, g_b, mode="join")

    # Invariant 1: Join complement must be disconnected
    assert is_valid_join(adj_join, threshold=0.5) is True
    # Invariant 2: Join graph itself should be connected
    assert is_valid_union(adj_join, threshold=0.5) is False
    # Invariant 3: High edge density
    density = calculate_edge_density(adj_join, threshold=0.5)
    assert density >= 0.5


def test_union_topological_validity(sample_pair):
    g_a, g_b = sample_pair
    adj_union = create_pair_graph(g_a, g_b, mode="union")

    # Invariant 1: Union graph must be disconnected
    assert is_valid_union(adj_union, threshold=0.5) is True
    # Invariant 2: Low edge density
    density = calculate_edge_density(adj_union, threshold=0.5)
    assert density < 0.5


def test_saved_dataset_validity():
    # Test on a small slice of actual generated files if they exist
    import os
    if os.path.exists("data/cograph_dense/val.pt"):
        dense_val = torch.load("data/cograph_dense/val.pt", weights_only=False)
        sparse_val = torch.load("data/cograph_sparse/val.pt", weights_only=False)

        # Check first 50 graphs
        for adj in dense_val[:50]:
            assert is_valid_join(adj, threshold=0.5) is True
            assert calculate_edge_density(adj, threshold=0.5) >= 0.5

        for adj in sparse_val[:50]:
            assert is_valid_union(adj, threshold=0.5) is True
            assert calculate_edge_density(adj, threshold=0.5) < 0.5
