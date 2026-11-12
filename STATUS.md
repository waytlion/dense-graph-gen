# Project Status — Handoff Notes (2026-11-12)

## Current Phase
Validating **Hypothesis 1**: DeFoG and DiPhon baselines perform poorly on dense, low-rank-width cographs vs. sparse counterparts.

## Training Runs Status

### DeFoG Cograph Training (all 8 COMPLETE ✅)
All 500-epoch runs finished on cluster alpha. Checkpoints at:
`/data/horse/ws/tibo990i-defog_env/defog_outputs/2026-10-26/`

### DeFoG Offline Evaluation (5 of 8 done)
Script: `scripts/eval_defog_checkpoints.sh` (generates 1000 samples per checkpoint, computes MMD + validity + subgraph metrics).

**Completed results:**

| Dataset | Type | Validity | subgraph_degree | subgraph_clustering |
|---------|------|----------|----------------|-------------------|
| cograph_sparse (QM9) | union | 99.5% | 0.0009 | 0.0115 |
| cograph_zinc_sparse | union | 99.6% | 0.0004 | 0.0003 |
| cograph_zinc_dense | join | 99.8% | 0.0026 | 0.0379 |
| cograph_proteins_dense | join | 99.1% | **0.468** | 0.089 |
| cograph_proteins_sparse | union | 97.4% | 0.012 | 0.078 |

**Missing (timed out at 12h):** `cograph_dense` (QM9), `cograph_collab_dense`, `cograph_collab_sparse`

### DiPhon Cograph Training (4 of 8 done)

**Completed (500 epochs):**

| Job | Dataset | Time | join/union_acc | subgraph_degree | subgraph_clustering |
|-----|---------|------|---------------|----------------|-------------------|
| 2391088 | cograph_dense (QM9) | 16h | join 93.8% | 0.004 | 0.021 |
| 2391089 | cograph_sparse (QM9) | 15.8h | union 21.9% | 0.003 | 0.054 |
| 2391090 | cograph_zinc_dense | 34h | join 90.6% | 0.239 | 0.033 |
| 2391091 | cograph_zinc_sparse | 33h | union 4.7% | 0.093 | 0.010 |

**Failed — need relaunch:**

| Job | Dataset | Failure | Root Cause | Fix |
|-----|---------|---------|-----------|-----|
| 2391092 | cograph_proteins_dense | OOM (RAM) | batch_size=16, max_node_num=100 | batch_size → 8 |
| 2391093 | cograph_proteins_sparse | W&B timeout | ServiceStartTimeoutError | Already have WANDB_MODE=offline in script; just rerun |
| 2391094 | cograph_collab_dense | OOM (RAM, 4min) | batch_size=8, max_node_num=180 | batch_size → 4 |
| 2391095 | cograph_collab_sparse | OOM (RAM, 6min) | batch_size=8, max_node_num=180 | batch_size → 2 |

### Pure COLLAB Baseline (currently running 🏃)
- **Data prep** (job 2391086): ✅ COMPLETE — 4,715 graphs with N≤200 (filtered from 5,000)
- **DeFoG collab_pure** (job 2392883): 🏃 RUNNING since Nov 10 (~500 epochs, batch_size=4)
- **DiPhon collab_pure** (job 2392882): 🏃 RUNNING since Nov 10 (~500 epochs, max_node_num=200, batch_size=4)

## Immediate Next Steps

### 1. Fix DiPhon batch sizes and relaunch 4 failed runs
Easiest via SSH+sed directly on cluster (no git push needed):
```bash
ssh alpha "
  # Proteins: batch_size 16→8
  sed -i 's/batch_size: int = 16/batch_size: int = 8/' ~/dense-graph-gen/external/diphon/configs/config_cograph_proteins_dense.py
  sed -i 's/batch_size: int = 16/batch_size: int = 8/' ~/dense-graph-gen/external/diphon/configs/config_cograph_proteins_sparse.py
  # COLLAB cographs: batch_size 8→4 (dense) and 8→2 (sparse)
  sed -i 's/batch_size: int = 8/batch_size: int = 4/' ~/dense-graph-gen/external/diphon/configs/config_cograph_collab_dense.py
  sed -i 's/batch_size: int = 8/batch_size: int = 2/' ~/dense-graph-gen/external/diphon/configs/config_cograph_collab_sparse.py
"
ssh alpha "cd ~/dense-graph-gen && \
  sbatch scripts/run_diphon_cluster.sh cograph_proteins_dense && \
  sbatch scripts/run_diphon_cluster.sh cograph_proteins_sparse && \
  sbatch scripts/run_diphon_cluster.sh cograph_collab_dense && \
  sbatch scripts/run_diphon_cluster.sh cograph_collab_sparse"
```
**After these complete**, also update local configs to match and commit.

### 2. Relaunch DeFoG eval for 3 missing checkpoints
Either increase walltime to 24h and rerun all, or create per-checkpoint sbatch wraps for the 3 missing:
- `cograph_dense` (QM9 dense)
- `cograph_collab_dense`
- `cograph_collab_sparse`

Example per-checkpoint:
```bash
sbatch -A p_scads_graphgen3d -p alpha --nodes=1 --gres=gpu:1 --cpus-per-task=4 --mem=32G \
  --time=04:00:00 --job-name=eval_cograph_dense \
  --output=/data/horse/ws/tibo990i-defog_env/logs/eval_cograph_dense_%j.log \
  --wrap="export LD_LIBRARY_PATH=/data/horse/ws/tibo990i-defog_env/defog/lib:\$LD_LIBRARY_PATH && \
  export WANDB_MODE=offline && \
  eval \"\$(conda shell.bash hook)\" && conda activate /data/horse/ws/tibo990i-defog_env/defog && \
  cd ~/dense-graph-gen/external/DeFoG/src && \
  python main.py +experiment=cograph_dense \"general.test_only='/data/horse/ws/tibo990i-defog_env/defog_outputs/2026-10-26/14-58-02-cograph_dense-cograph_dense/checkpoints/cograph_dense/epoch=499.ckpt'\" general.final_model_samples_to_generate=1000"
```
Repeat for `cograph_collab_dense` and `cograph_collab_sparse` with their respective checkpoint paths (see `scripts/eval_defog_checkpoints.sh` for paths).

### 3. After all runs complete
- Collect all DeFoG + DiPhon results into a comparison table
- Key comparison: dense (join) vs sparse (union) for each base graph set
- Analyze whether DeFoG/DiPhon struggle more with dense cographs (supporting Hypothesis 1)

## Key Findings So Far
- **DeFoG handles QM9/ZINC cographs well** (both dense and sparse): validity >99%, low subgraph MMD
- **DeFoG struggles with Proteins-dense** cographs: validity still high (99.1%) but subgraph_degree MMD is **0.468** (vs 0.012 for sparse) — strong signal for Hypothesis 1
- **DiPhon struggles significantly with sparse (union) cographs**: union validity is very low (4.7%–21.9% for QM9/ZINC sparse vs 87.5%–93.8% for dense joins)
- **DiPhon zinc_sparse is nearly non-functional**: only 4.7% union validity
- DiPhon requires **much smaller batch sizes** than DeFoG due to dense N×N adjacency padding

## Workflow Tips for Cluster
- **SSH multiplexing**: `ControlMaster auto` in `~/.ssh/config` lets agent reuse your active SSH session
- **Quick config tweaks**: Use `ssh alpha "sed -i ..."` instead of commit-push-pull cycle
- **CLI overrides**: DiPhon supports `--batch-size N` flag; DeFoG supports Hydra overrides
- **All SLURM jobs MUST**: `--account=p_scads_graphgen3d`, `--partition=alpha`, `--nodes=1`, `--gres=gpu:1`
- **All batch scripts MUST**: `export WANDB_MODE=offline` (no interactive W&B login in SLURM)

## File Locations on Cluster
- Conda env: `/data/horse/ws/tibo990i-defog_env/defog`
- DeFoG checkpoints: `/data/horse/ws/tibo990i-defog_env/defog_outputs/2026-10-26/`
- Logs: `/data/horse/ws/tibo990i-defog_env/logs/`
- Data: `~/dense-graph-gen/data/` (symlinked to workspace)
- DiPhon results: `~/dense-graph-gen/external/diphon/results/`

