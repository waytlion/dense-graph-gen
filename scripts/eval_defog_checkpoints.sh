#!/bin/bash
#SBATCH --account=p_scads_graphgen3d
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=04:00:00
#SBATCH --job-name=eval_defog
#SBATCH --output=/data/horse/ws/tibo990i-defog_env/logs/eval_defog_%j.log

set -e

WS_DIR="/data/horse/ws/tibo990i-defog_env"
REPO_DIR="${SLURM_SUBMIT_DIR:-$HOME/dense-graph-gen}"

export LD_LIBRARY_PATH="$WS_DIR/defog/lib:$LD_LIBRARY_PATH"
eval "$(conda shell.bash hook)"
conda activate "$WS_DIR/defog"

cd "$REPO_DIR/external/DeFoG/src"

declare -A CKPTS=(
    ["cograph_dense"]="$WS_DIR/defog_outputs/2026-10-26/14-58-02-cograph_dense-cograph_dense/checkpoints/cograph_dense/epoch=499.ckpt"
    ["cograph_sparse"]="$WS_DIR/defog_outputs/2026-10-26/14-59-16-cograph_sparse-cograph_sparse/checkpoints/cograph_sparse/epoch=499.ckpt"
    ["cograph_zinc_dense"]="$WS_DIR/defog_outputs/2026-10-26/15-08-47-cograph_zinc_dense-cograph_zinc_dense/checkpoints/cograph_zinc_dense/epoch=499.ckpt"
    ["cograph_zinc_sparse"]="$WS_DIR/defog_outputs/2026-10-26/15-09-28-cograph_zinc_sparse-cograph_zinc_sparse/checkpoints/cograph_zinc_sparse/epoch=499.ckpt"
    ["cograph_proteins_dense"]="$WS_DIR/defog_outputs/2026-10-26/18-58-39-cograph_proteins_dense-cograph_proteins_dense/checkpoints/cograph_proteins_dense/epoch=499.ckpt"
    ["cograph_proteins_sparse"]="$WS_DIR/defog_outputs/2026-10-26/18-58-39-cograph_proteins_sparse-cograph_proteins_sparse/checkpoints/cograph_proteins_sparse/epoch=499.ckpt"
    ["cograph_collab_dense"]="$WS_DIR/defog_outputs/2026-10-26/18-58-39-cograph_collab_dense-cograph_collab_dense/checkpoints/cograph_collab_dense/epoch=499.ckpt"
    ["cograph_collab_sparse"]="$WS_DIR/defog_outputs/2026-10-26/19-00-20-cograph_collab_sparse-cograph_collab_sparse/checkpoints/cograph_collab_sparse/epoch=499.ckpt"
)

for exp in "${!CKPTS[@]}"; do
    ckpt="${CKPTS[$exp]}"
    echo "=================================================="
    echo "Evaluating $exp using $ckpt"
    echo "=================================================="
    if [ ! -f "$ckpt" ]; then
        echo "Error: Checkpoint $ckpt not found, skipping."
        continue
    fi
    python main.py +experiment="$exp" "general.test_only='$ckpt'" general.final_model_samples_to_generate=1000
done

echo "All evaluations completed at $(date)"

