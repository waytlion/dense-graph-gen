# Project Status: Handoff Notes (updated 2026-11-13)

Every session reads this first (see AGENTS.md → Multi-session coordination) and updates it before ending.

---

## 1. Goal (highest level)
- **Hypothesis 1 (current phase):** state-of-the-art graph generators (DeFoG, DiPhon) perform worse on **dense, low-rank-width** graphs (cograph **joins**) than on **sparse** graphs built from the same building blocks (cograph **unions**).
- **Hypothesis 2 (later):** a model that generates a decomposition tree first, then the graph (TD-GEN-style), does better.
- Open: is rank-width the right notion of "simple structure"? (confirm with PI)

## 2. Agenda: next experiments (medium level, in order)
1. **Make evaluation trustworthy and comparable (DeFoG vs DiPhon)**: we're in exploration mode, so keep it minimal. No full offline pipeline for now.
   - Training behaviour: validation every 50 epochs, generated graphs vs **val** set. Since 2026-11-13, both models save the generated graphs (`val_samples/epoch<N>.pkl`; DeFoG: run output dir, DiPhon: `checkpoints/<name>/val_samples/`), so new metrics can be recomputed on CPU without sampling again. Counts: DeFoG 40, DiPhon 64 (not equal yet; noisy).
   - Final performance: final checkpoint, ~1000 graphs vs **test** set.
   - Metrics: the old subgraph MMD was replaced by the **dejoin MMD** (DeFoG `20f769c`, DiPhon `ad5150f`), so **all earlier cograph metric numbers are outdated**.
   - Still to decide: x-axis epochs vs optimizer steps (batch sizes differ between models); evaluate both with EMA weights? (DiPhon uses EMA; check DeFoG).
2. **Rerun what's needed** (details in §3):
   - collab_pure: resubmit DeFoG; let DiPhon finish (it hits its walltime).
   - Re-evaluate **all 16 cograph runs** (8 DeFoG, 8 DiPhon) with the dejoin metrics. No retraining needed, only evaluation.
   - Make sure DiPhon has finished training on **all 8** cograph datasets.
   - Before re-evaluating DiPhon, delete stale reference metrics: `rm -v /data/horse/ws/tibo990i-defog_env/data/ref_metrics/ref_metrics_cograph_*.pt*`
3. **Analyse:** dense vs sparse per base dataset, DeFoG vs DiPhon, training curves and final test numbers → does it support Hypothesis 1?

## 3. Where each experiment stands (low level, 2026-11-13 17:00)

| Run | Training | Evaluation | Next action |
|---|---|---|---|
| DeFoG, 8 cograph datasets | ✅ done (500 ep, `defog_outputs/2026-10-26/`) | old metrics: 5/8 evaluated; 3 timed out at 12h (`cograph_dense`, `cograph_collab_dense`, `cograph_collab_sparse`) | re-eval all 8 with dejoin metrics; collab ones need >12h or fewer samples |
| DiPhon cograph QM9 dense/sparse, ZINC dense/sparse | ✅ done (jobs 2391088–91) | old metrics only | re-eval with dejoin metrics |
| DiPhon cograph PROTEINS dense/sparse, COLLAB dense/sparse | failed Nov 10 (RAM OOM / W&B timeout), relaunched; job **2392887 (cograph_proteins_\*) running** on 11-13 | n/a | **verify** which of the 4 relaunches run/finished (logs: `diphon_diphon_<jobid>.log`) |
| DeFoG collab_pure | ❌ job 2392883 OOM on 11-12; cause: it trained on **unfiltered** COLLAB (stale DeFoG data cache, up to 492 nodes). Fixed 11-13. | n/a | **resubmit**; log must show 4,715 graphs and `max num nodes: 200` |
| DiPhon collab_pure | 🏃 job 2392882, epoch 128/500 at 11-13 17:00, ~12 min/epoch; will hit the **2-day walltime ~11-14 09:00 at epoch ~195** | n/a | check whether DiPhon can resume from `last.ckpt`; otherwise rerun with a longer walltime |

## 4. Housekeeping to-dos
- **FIRST (next session):** move DiPhon `checkpoints/` (8.3 GB) and `wandb/` from cluster home to horse once no DiPhon job runs. Commands in the agent memory note `todo-move-diphon-outputs`; target `/data/horse/ws/tibo990i-defog_env/diphon_outputs`.
- Optional: set git push URL to SSH in the main repo and `external/diphon` (`git remote set-url --push origin git@github.com:waytlion/<repo>.git`). Their `origin` is HTTPS without stored credentials, so a plain `git push` fails on the laptop.

## 5. Key findings so far (⚠️ based on the OLD subgraph metrics, to be re-validated)
- DeFoG handles QM9/ZINC cographs well (dense and sparse): validity >99%.
- DeFoG on PROTEINS-dense: validity 99.1% but subgraph_degree MMD 0.468 (vs 0.012 sparse), a signal for Hypothesis 1.
- DiPhon struggles with **sparse** (union) cographs: union validity 4.7–21.9% (QM9/ZINC) vs 90–94% join validity.
- DiPhon needs much smaller batch sizes (always pads to max N).

## 6. Reference: how things work (learned 2026-11-13)
- **Data pipeline:** `data/<name>/{train,val,test}.pt` (on horse) is the source.
  - DeFoG copies it into `external/DeFoG/data/<name>/raw/` and builds a cache in `processed/`. This folder is a symlink to `/data/horse/ws/tibo990i-defog_env/defog_data`.
  - DiPhon reads `data/<name>.pkl`, made by `scripts/export_cographs_for_diphon.py` (collab_pure: `scripts/prepare_collab_pure.py` writes both formats).
  - The caches now rebuild automatically when the source changes:
    - DeFoG `source_fingerprint.txt`
    - DiPhon `ref_metrics_*.pt.src`
    - export re-runs if `.pt` is newer than `.pkl`
  - They do **not** notice metric-code changes → delete DiPhon `ref_metrics` by hand after a metric change.
  - Verified 11-13: `.pt` and `.pkl` agree for all 11 datasets (counts, max N, edges).
- **GPU memory (both models: same graph transformer, fp32):** ≈ L·B·n²·c·4 bytes + 2–4 GB.
  - L = layers (10), B = batch size, n = largest graph in batch, c ≈ 2,300 (values stored per node pair per layer for backward; hand count, ±30%).
  - n=200, B=4 → ~17–19 GB. Max n on a 40 GB A100 with B=4 ≈ 310.
  - Memory grows with n², independent of edge count.
- **Cluster:**
  - When alpha's login is in maintenance, use `ssh tibo990i@login2.romeo.hpc.tu-dresden.de` (no GPU; same home and horse filesystems; `squeue` for alpha hangs there).
  - Running Python on romeo needs `export LD_LIBRARY_PATH=/data/horse/ws/tibo990i-defog_env/defog/lib:$LD_LIBRARY_PATH`.
  - The login node kills big jobs; use `srun --account=p_scads_graphgen3d --partition=romeo --mem=64G ...`.
- **Walltimes:** DiPhon script `--time=2-00:00:00` (too short for collab_pure); DeFoG eval 12h (too short for collab cographs with 1000 samples).
- **DeFoG eval example:** see `scripts/eval_defog_checkpoints.sh`; single checkpoint: `python main.py +experiment=<exp> "general.test_only='<ckpt>'" general.final_model_samples_to_generate=1000`. To recompute metrics from saved graphs without sampling: `general.generated_path=<generated_samples_rank0.pkl>`.

## File locations on cluster
- Conda env: `/data/horse/ws/tibo990i-defog_env/defog`
- DeFoG outputs/checkpoints: `/data/horse/ws/tibo990i-defog_env/defog_outputs/` (symlinked from `external/DeFoG/outputs`)
- DeFoG data cache: `/data/horse/ws/tibo990i-defog_env/defog_data/` (symlinked from `external/DeFoG/data`)
- Source data: `~/dense-graph-gen/data/` → `/data/horse/ws/tibo990i-defog_env/data/`
- Logs: `/data/horse/ws/tibo990i-defog_env/logs/`
- DiPhon checkpoints/wandb: still in `~/dense-graph-gen/external/diphon/` (home!), see §4
