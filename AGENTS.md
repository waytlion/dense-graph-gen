<output_verbosity_spec>
- Default: ≤5 concise bullet points
- Simple factual questions: ≤2 bullets <= 5 words!
- Lead with the answer. Give context only if it changes the answer!
- Answer the question asked; stop there!
- If the request is ambiguous, ask one question.
- define/describe all variables briefly when using
- NEVER add, edit, or modify any code files without explicitly proposing the change and getting user confirmation first!
- NEVER run any terminal commands or processes without explicit user permission first!
</output_verbosity_spec>

# Project context: Generative models for dense graphs with low rank-width

## Research goal
We study **graph generation** (learning a distribution p(G) over graphs from samples and generating new graphs) for **dense graphs with simple structure**, i.e. low **rank-width** (to be confirmed with the PI; rank-width is a graph-complexity parameter that, unlike treewidth, can be small for dense graphs).

- **Hypothesis 1:** Current state-of-the-art generators (e.g. DeFoG, DiGress, autoregressive models) perform poorly on dense, low-rank-width graphs.
- **Hypothesis 2:** A new model that exploits the decomposition structure (generating a decomposition tree, then the graph; inspired by TD-GEN) does better.

**Current phase: validating Hypothesis 1.** We train DeFoG on synthetic datasets and compare its performance on dense vs. sparse graphs that share the same building blocks.

## Background (minimal)
- **DeFoG** (Qin, Madeira, Thanou, Frossard; ICML 2025): discrete flow matching for graphs, built on the DiGress codebase. Currently among the strongest models on standard benchmarks (Planar, SBM, Tree).
- **Cograph operations:**
  - **Join:** put two graphs side by side and connect every node of A to every node of B.
  - **Union:** put two graphs side by side with no new edges.
  - Join keeps rank-width low, since rw(join(A, B)) ≤ max(rw(A), rw(B), 1).
- **Key property for validation:** a graph is a join of two (non-empty) graphs **iff its complement is disconnected**.


## Environment
- Windows host; all development runs inside **WSL2 (Ubuntu)**. The repo lives in the WSL file system (`~/projects/dense-graph-gen`), not under `/mnt/c`.
- Conda (Miniforge), environment **`defog`**, created from `external/DeFoG/environment.yaml` (Python 3.9, PyTorch, PyTorch Geometric, graph-tool, RDKit). DeFoG is installed with `pip install -e .`.
- ORCA (orbit counting for the Orbit MMD) is compiled in `external/DeFoG/src/analysis/orca/`.
- DeFoG uses Hydra; entry point `external/DeFoG/src/main.py`. Run from `external/DeFoG/src`:
  - quick test: `python main.py +experiment=debug`
  - example dataset: `python main.py +experiment=tree dataset=tree`
  - evaluate a checkpoint: add `general.test_only=<path/to/ckpt>`
- Larger training runs will happen on a Linux GPU cluster 

## Extending DeFoG with a new dataset (from the DeFoG README)
1. New file in `external/DeFoG/src/datasets/`, modelled on `spectre_dataset.py` (unattributed graphs), defining a PyG `Dataset` class and a `DatasetInfos` class.
2. Register the dataset in `external/DeFoG/src/main.py`.
3. Add a config in `external/DeFoG/configs/dataset/` (and an experiment config if needed).
4. Custom metrics go under `external/DeFoG/src/metrics/`.

## Implemented DeFoG Integration
- **Dataset Loader:** `external/DeFoG/src/datasets/cograph_dataset.py` defines `CographDataset` and `CographDataModule` (loads pre-split `.pt` adjacency matrices from `data/cograph_{dense,sparse}/`).
- **Configs:** Dataset configs in `configs/dataset/cograph_{dense,sparse}.yaml`; experiment configs in `configs/experiment/cograph_{dense,sparse}.yaml` (500 epochs, batch size 256, 500 ODE steps).
- **Metrics & Registration:** Registered in `external/DeFoG/src/main.py`. Topological validity (`is_valid_cograph_join`, `is_valid_cograph_union`) wired into `external/DeFoG/src/analysis/spectre_utils.py` (quadratic isomorphism bottleneck disabled).

## Conventions
- Our own code lives outside the submodule (`data_gen/`, `analysis/`). Changes inside `external/DeFoG/` are limited to dataset/config/registration/metrics files and are committed on the fork's `dense-graphs` branch; then the submodule pointer is committed in the main repo.
- Do not change DeFoG's model or training code silently; baselines must remain comparable.
- Never commit `data/`, `outputs/`, `wandb/`, checkpoints (`*.ckpt`) or build artefacts (e.g. the compiled `orca` binary).
- Fix random seeds; datasets must be reproducible from a seed and a config.

## Open questions (not yet decided)
- The exact definition of "simple structure": rank-width vs. another notion (to confirm with the PI).