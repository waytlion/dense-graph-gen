# Data Generation Plan
1. Load dataset: 
        - start with QM9 dataset (molecules 133,885 small molecules N=~[1-9])
        - filter for range N=[6-9]
        - Remove node attributes, cause we only want graph structure -> remove duplicates
2. Split into train, val, test (80/10/10)
3. Generate 10.000 graphs in train, and proportional amount in val/test, once by doing join, once by doing union. So we basically have 2 groups/2 experiments.
        - Group 1: DENSE: Combine 2x graphs by doing bipartite pairing (let every node from graph A connect to every node in graph B) so we get dense graph
        - Group 2: SPARSE: combine by doing uniun operation
        -Implementation Note: We start by saving .pt files, which are native input for DeFog (Python list of 2D adjacency tensors [adj_1, adj_2, ...], where adj_i has shape (N, N))
            - IF mem becomes a problem when we save the graphs at some point: Dont create and save the graphs, but rather just save the pairing indices (we create all graphs by same alg over a pair and can do this on the fly to save mem)


# Metrics
1. Validity rate on combined Graph
        - join is valid iff complement graph is disconnected-
        - union is valid iff graph is disconnected
2. For valid generated graph, analyse the base graphs:
        - Histogram for train, test, and generated?
                - e.g. degree dist: x-axis=deg from 0 ...N y-axis=number of nodes across all graphs 
        - MMD (uses gaussian or RBG kernel to calc "similarity" of two SETS of histograms, where each graph gets its own histogram)
        - Across metrics:
                - Degree distribution (local connectivity)
                - Clustering coefficient distribution — (triangle density / neighborhood tightness)
                - Orbit count distribution 
                - Spectral distribution — eigenvalues of the graph Laplacian (global shape/connectivity of the graph)


# DeFog adaption -> extend DeFoG with cograph dataset 
- From DeFog readme:
        - create dataset in external/DeFoG/src/datasets/, mirroring spectre_dataset.py
        - register  dataset: external/DeFoG/src/main.py
        - Add config: external/DeFoG/configs/dataset/
        - Custom metrics: external/DeFoG/src/metrics/

# General TODO
- job läuft auf capella: 4425496
- von capella auf alpha geheen, 
        - env nochmal machen, also löschen denke und onchmal neu aufsetzen: cd ~/dense-graph-gen/external/DeFoG
conda env create -p $WS_DIR/defog -f environment.yaml
        - dafür alpha env nehmen in sbatch: conda activate $(ws_find -F horse defog_env)/defog

- tune DeFog so it trains correctly -> eval the two datasets
- while defog runs, do similar for DiPhon paper (which is supposed to work well on dense graphs)
- adding fast O(1) WL-hash uniqueness/novelty checks.


# Cheat sheet
- Cluster Env: conda activate /data/cat/ws/tibo990i-defog_env/defog
- 