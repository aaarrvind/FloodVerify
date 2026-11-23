# Post-hoc study: M3 (MSTA U-Net), clean reproduction and post-image corruption (20-21 Nov)

**Status: post-hoc.** Run after the 18 Nov Nepal test. The final model (M1 weighted CE, seed 0) and
every headline number are unchanged. M3 and the ablation were designed after the Nepal results were known,
so their Nepal numbers are not a blind test. "M3" here is a model name (Multi-Scale Temporal Attention
U-Net), not Member 3.

The notebooks in this folder were renamed for the repo:

| In this folder | Original name used below |
| --- | --- |
| `01_inventory_and_reproduction_check.ipynb` | `inventory.ipynb` |
| `02_m3_build_and_verify.ipynb` | `m3_build_verify.ipynb` |
| `03_m3_training_and_florence_eval.ipynb` | `train_m3.ipynb` |
| `04_corruption_study_nepal.ipynb` | `corruption.ipynb` |
| `05_corruption_analysis.ipynb` | `analysis.ipynb` |

All outputs live on the team Drive in `MAAI 902 X 901 PROJECT/m3_experiments/` (not in git). The code the
notebooks wrote (`m3_model.py`, `corruption.py`, ...) is reproduced in full inside each notebook's
`%%writefile` cells, so the notebooks are self-contained.

---

## Original file map

Every path below is relative to `MAAI 902 X 901 PROJECT/m3_experiments/`. The notebooks run in this order, and each
writes only inside `m3_experiments/`, reading the existing project folders without changing them.

| Notebook | What it does |
| --- | --- |
| `inventory.ipynb` | Creates `m3_experiments/`, verifies the 12 existing checkpoints and the saved Nepal results, spot-checks clean reproduction |
| `m3_build_verify.ipynb` | Writes the M3 code, configs and experiment plan; unit tests, parameter counts, L4 benchmark, smoke runs |
| `train_m3.ipynb` | Trains M3 (3 seeds) and the exploratory ablation (1 seed); Florence evaluation; attention analysis |
| `corruption.ipynb` | Controlled post-image corruption study on Nepal for all 16 checkpoints |
| `analysis.ipynb` | Between-family tests, investigation of unexpected gains, artefact checks, prediction maps |

Some file names carry an older label from when the notebooks were numbered: `stage2_*` files come from
`inventory.ipynb`, `stage3_*` from `m3_build_verify.ipynb`, `stage4_*` from `train_m3.ipynb`, `stage5_*` from
`corruption.ipynb`, and everything named `phase6` from `analysis.ipynb`.

---

## `inventory.ipynb`

| Output | Content |
| --- | --- |
| `README.md` | Short description of the folder (written once) |
| `inventory/write_test.json` | Write-access test of the shared folder |
| `inventory/code_equivalence.json` | Pinned repo commit `8f10e61` vs the branch used for the Nepal run; library versions |
| `inventory/checkpoint_inventory.csv` | The 12 existing checkpoints: files, SHA-256, best epoch, parameters, status |
| `inventory/checkpoint_configs.json` | Stored training config of each checkpoint |
| `inventory/other_run_folders.csv` | Run folders on Drive that are not part of this study |
| `inventory/m3_class_weights_check.json` | M1 weighted-CE class weights vs those used for M3 |
| `inventory/nepal_saved_results_check.csv` | Saved Nepal results: crops, order, metrics recomputed |
| `inventory/clean_spotcheck/*_first64.npz`, `spotcheck_summary.csv` | 64-crop clean re-prediction vs the saved Nepal results |
| `inventory/preservation_check.json` | Nothing outside `m3_experiments/` changed |
| `inventory/stage2_summary.json` | Summary of all checks |

## `m3_build_verify.ipynb`

| Output | Content |
| --- | --- |
| `code/m3_model.py`, `code/m3_register.py`, `code/train_m3.py`, `code/m3_tests.py` | M3 model and ablation, registry patch, training launcher, unit tests |
| `code/MANIFEST.json` | SHA-256 of every code file (updated again by the later notebooks) |
| `code/_previous/` | Older versions of any file that was replaced |
| `configs/m3_msta_unet_ce.yaml`, `configs/exploratory_abl_m1fusion_dualhead_ce.yaml` | Training configs |
| `configs/experiment_plan.json` | Experiment plan, frozen before training |
| `NOTES_for_project_owners.md` | Note about the missing `DONE` file in `eval_extras/nepal_final/` |
| `m3/benchmark/unit_tests.json` | Unit-test results |
| `m3/benchmark/param_counts.csv` | Parameter counts of M1, M3 and the ablation |
| `m3/benchmark/real_data_checks.json` | Forward pass and loss on real Florence and training batches |
| `m3/benchmark/benchmark_l4.json`, `benchmark_l4.csv` | L4 speed and memory benchmark |
| `m3/smoke/`, `logs/smoke_*.log` | Short smoke-test runs (mechanics only, never compared with anything) |
| `m3/benchmark/preservation_check_stage3.json`, `stage3_summary.json` | Preservation check and summary |

## `train_m3.ipynb`

| Output | Content |
| --- | --- |
| `code/metrics_ext.py`, `code/m3_analysis.py` | Extended metrics; run verification and attention analysis |
| `m3/benchmark/nodata_loss_check.json` | Loss check on real crops with no-data pixels |
| `m3/checkpoints/m3_msta_unet_ce_s{0,1,2}/` | M3 runs: `best.pt`, `last.pt`, `config.yaml`, `metrics.jsonl` |
| `m3/exploratory_ablation/checkpoints/exploratory_abl_m1fusion_dualhead_ce_s0/` | Exploratory ablation run |
| `m3/checkpoints/_verification/`, `m3/exploratory_ablation/_verification/` | Verification of each run's files |
| `logs/training_sessions.jsonl`, `logs/*__train_*.log` | Every training launch and its console output |
| `m3/eval_val/` (incl. `reference_m1_ce/`, `EXPLORATORY_ablation/`) | Florence predictions of every `best.pt` |
| `results/stage4_checkpoint_verification.csv` | Verification table |
| `results/stage4_florence_primary_M3_vs_M1ce.csv`, `stage4_florence_primary_mean_std.csv` | Florence results, M3 vs M1 CE |
| `results/stage4_florence_per_class_prf_primary.csv` | Florence per-class precision, recall, F1 |
| `results/stage4_florence_EXPLORATORY_ablation.csv`, `stage4_florence_per_class_prf_EXPLORATORY_ablation.csv` | Ablation, kept separate |
| `results/stage4_attention_analysis.json`, `stage4_attention_scales.csv`, `stage4_attention_knockout.csv` | Attention analysis |
| `results/stage4_training_curves.csv`, `figures/stage4_curves_M3_vs_M1ce.png`, `figures/stage4_curves_EXPLORATORY_ablation.png` | Training curves |
| `results/stage4_compute.csv`, `results/stage4_summary_*.json` | Compute and summary |

## `corruption.ipynb`

| Output | Content |
| --- | --- |
| `code/corruption.py`, `code/corruption_eval.py`, `code/corruption_analysis.py` | Corruptions, evaluation engine, statistics |
| `corruption/manifest/METHODOLOGY.md` | Methodology of the corruption study |
| `corruption/manifest/checkpoints.csv` | The 16 checkpoints with SHA-256 and evaluation status |
| `corruption/manifest/corruption_conditions.csv`, `crop_manifest_nepal.csv` | Conditions; Nepal crops with noise seeds and pixel counts |
| `corruption/manifest/clean_input_identity.csv`, `nepal_source_files.csv`, `preprocessing_equivalence.json` | Proof that clean inputs match the original Nepal evaluation |
| `corruption/input_hashes/<condition>.csv`, `summary.json` | SHA-256 of every corrupted post crop |
| `corruption/predictions/<run>/<condition>.npz` | Predictions of the 15 primary checkpoints |
| `corruption/predictions_EXPLORATORY_ablation/` | Predictions of the ablation |
| `corruption/timing/timing_<condition>_*.json` | Timing of each condition pass |
| `corruption/samples/*.png`, `figures/corruption_examples.png` | Example corrupted images |
| `results/clean_reproduction_check.csv`, `clean_reproduction_gate.json` | Clean reproduction gate |
| `results/corruption_metrics_per_checkpoint_primary.csv` | All metrics per checkpoint and condition |
| `results/corruption_per_class_prf_primary.csv` | Per-class precision, recall, F1 |
| `results/corruption_paired_bootstrap_per_checkpoint_primary.csv`, `..._per_family_primary.csv` | Bootstrap intervals of degradation |
| `results/corruption_family_summary_primary.csv` | Family means and std over seeds |
| `results/consolidated_results_primary.csv` | One long table of all primary results |
| `results/*_EXPLORATORY_ablation.csv` | The same tables for the ablation, kept separate |
| `results/corruption_timing_benchmark_64crops.csv`, `corruption_timing.csv` | Timing |
| `figures/corruption_*_primary.png` (each with `_data.csv`) | Score, macro-F1 and building IoU vs severity; degradation heatmap; per-class F1 |
| `figures/corruption_EXPLORATORY_ablation.png` | Ablation figure, kept separate |
| `results/stage5_summary_*.json` | Summary |

## `analysis.ipynb`

| Output | Content |
| --- | --- |
| `code/phase6_analysis.py` | Analysis functions |
| `results/phase6/precheck_repeated_hashes.csv` | Repeated input hashes = crops with no valid pixels |
| `results/phase6/timing_per_condition.csv`, `timing_gpu_per_checkpoint.csv`, `timing_gpu_per_family.csv` | Timing reconciliation |
| `results/phase6/between_family_tests.csv` | M2 vs M1 CE and M3 vs M1 CE paired tests |
| `results/phase6/class_shift_all.csv`, `investigation_cases.csv`, `investigation_per_seed.csv`, `investigation_confusion_matrices.csv` | Investigation of unexpected gains and destroyed-F1 increases |
| `results/phase6/artifact_diagnostics_per_crop.csv`, `artifact_diagnostics_summary.csv`, `figures/phase6/corruption_artifact_check.png` | Corruption artefact checks |
| `results/phase6/qualitative_predictions_seed0.npz`, `qualitative_sample_agreement.csv` | Predictions for the six sample crops |
| `figures/phase6/qualitative/`, `figures/phase6/qualitative_EXPLORATORY_ablation/` | Prediction maps (seed 0; ablation separate) |
| `results/phase6/phase6_summary_*.json` | Summary |

---

## Evaluation status (applies to every result above)

- **M0, M1 CE, M1 Dice+focal, M2:** frozen 18 Nov Nepal comparison set; reproduced exactly in `corruption.ipynb`.
- **M3:** designed and trained after the 18 Nov Nepal results were known; its Nepal results are post-hoc, not a blind test.
- **Exploratory ablation:** one seed, post-hoc; always reported separately and never used for a significance claim.
- **Florence (validation)** was used to pick every checkpoint, so Florence scores are optimistic.
