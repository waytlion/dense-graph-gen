#!/bin/bash
#SBATCH --account=p_scads_graphgen3d
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=2-00:00:00
#SBATCH --job-name=diphon
#SBATCH --output=/data/horse/ws/tibo990i-defog_env/logs/diphon_%x_%j.log

set -e

MODEL_NAME=$1
if [ -z "$MODEL_NAME" ]; then
    echo "Usage: sbatch scripts/run_diphon_cluster.sh <model_name>"
    echo "Example: sbatch scripts/run_diphon_cluster.sh cograph_dense"
    exit 1
fi

echo "Job $SLURM_JOB_ID running $MODEL_NAME on $(hostname) at $(date)"

# Workspace and conda environment setup
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

export LD_LIBRARY_PATH="$WS_DIR/defog/lib:$LD_LIBRARY_PATH"
source "$WS_DIR/defog/bin/activate" defog

# Ensure required evaluation dependencies are present
pip install --quiet powerlaw pygsp

# Ensure datasets are converted to .pkl
cd "$REPO_DIR"
python scripts/export_cographs_for_diphon.py

# Run DiPhon training
cd "$REPO_DIR/external/diphon"
python main.py train --model "$MODEL_NAME" --device cuda --wandb

echo "Job finished at $(date)"
