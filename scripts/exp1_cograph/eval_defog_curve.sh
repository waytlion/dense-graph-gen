#!/bin/bash
#SBATCH --account=p_scads_graphgen3d
#SBATCH --partition=alpha
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1-00:00:00
#SBATCH --job-name=evalcurve_defog
#SBATCH --array=0-3
#SBATCH --output=/data/horse/ws/tibo990i-defog_env/logs/%x_%A_%a.log
# Exp 1: re-eval every checkpoint of the 4 dense DeFoG runs with dejoin metrics (40 samples vs test set).
# Usage (repo root): sbatch scripts/exp1_cograph/eval_defog_curve.sh
set -e
WS=/data/horse/ws/tibo990i-defog_env
RUNS=(14-58-02-cograph_dense 15-08-47-cograph_zinc_dense 18-58-39-cograph_proteins_dense 18-58-39-cograph_collab_dense)
RUN=${RUNS[$SLURM_ARRAY_TASK_ID]}; EXP=${RUN#*-*-*-}      # strip "HH-MM-SS-"
CKPT_DIR=$WS/defog_outputs/2026-10-26/$RUN-$EXP/checkpoints/$EXP
OUT=$WS/defog_outputs/curve_eval/$EXP                     # one folder per checkpoint

export WANDB_MODE=offline LD_LIBRARY_PATH=$WS/defog/lib:$LD_LIBRARY_PATH
eval "$(conda shell.bash hook)"; conda activate $WS/defog
cd ${SLURM_SUBMIT_DIR:-$HOME/dense-graph-gen}/external/DeFoG/src

for ckpt in $CKPT_DIR/epoch=*.ckpt; do
  ep=$(basename $ckpt .ckpt); ep=${ep#epoch=}
  [ -f $OUT/epoch$ep/test_epoch0_res_0_general.txt ] && continue   # skip done -> resumable
  echo "=== $EXP epoch $ep ($(date))"
  python main.py +experiment=$EXP "general.test_only='$ckpt'" \
    general.final_model_samples_to_generate=40 general.final_model_samples_to_save=0 \
    hydra.run.dir=$OUT/epoch$ep
done
echo "done $(date)"
