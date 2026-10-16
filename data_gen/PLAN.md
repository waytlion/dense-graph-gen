# Data Generation Plan

1. Load dataset: start with molecules N=~[8-12]
        - Remove node attributes, cause we only want graph structure -> remove duplicates
2. Split into train, val, test (80/10/10)
3. Generate 10.000 graphs in train, and proportional amount in val/test, once by doing join, once by doing union. So we basically have 2 groups/2 experiments.
        - Group 1: DENSE: Combine 2x graphs by doing bipartite pairing (let every node from graph A connect to every node in graph B) so we get dense graph
        - Group 2: SPARSE: combine by doing uniun operation
        -Implementation Note: Dont create and save the graphs, but rather just save the pairing indices (we create all graphs by same alg over a pair and can do this on the fly to save mem)
4. We have to think how to val graph gen (prob plot some?!)