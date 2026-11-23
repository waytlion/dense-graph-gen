#!/bin/bash
#SBATCH --account=p_scads_graphgen3d
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=evalcurve_diphon
#SBATCH --array=0-1
#SBATCH --output=/data/horse/ws/tibo990i-defog_env/logs/%x_%A_%a.log
# Exp 1: re-eval every checkpoint of DiPhon QM9/ZINC dense with dejoin metrics (64 samples vs test set).
# (proteins/collab dense train with current code -> dejoin logged during training.)
# Usage (repo root): sbatch scripts/exp1_cograph/eval_diphon_curve.sh
set -e
WS=/data/horse/ws/tibo990i-defog_env
RUNS=(cograph_dense cograph_zinc_dense)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}
OUT=$WS/diphon_outputs/curve_eval/$RUN; mkdir -p $OUT

export WANDB_MODE=offline LD_LIBRARY_PATH=$WS/defog/lib:$LD_LIBRARY_PATH
export LD_PRELOAD=/usr/lib64/libgomp.so.1 SKIP_REF=1      # ratios ignored -> skip slow ref metrics
eval "$(conda shell.bash hook)"; conda activate $WS/defog
cd ${SLURM_SUBMIT_DIR:-$HOME/dense-graph-gen}/external/diphon

for ckpt in checkpoints/$RUN/epoch=*.ckpt; do
  ep=$(basename $ckpt .ckpt); ep=${ep#epoch=}
  [ -f $OUT/epoch$ep.json ] && continue                   # skip done -> resumable
  echo "=== $RUN epoch $ep ($(date))"
  python -u main.py gen --model $RUN --device cuda --checkpoint $ckpt --num-samples 64 \
    --json-out $OUT/epoch$ep.json --save-graphs-path $OUT/epoch$ep.pkl --skip-size-ref
done
echo "done $(date)"
