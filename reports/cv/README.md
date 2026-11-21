CV report, 3000 to 4000 words. Owners: members 2 and 3. Start from the CV brief.

# How to load and call any CV checkpoint (member 2 -> member 3, pipeline)

```python
import torch
from floodsitrep.cv.models import load_checkpoint
model, cfg = load_checkpoint("<Drive>/checkpoints/<run_name>/best.pt", device="cuda")
with torch.no_grad():
    logits = model(pre, post)          # pre, post: [B,3,512,512] from XbdTiles (ImageNet-normalised)
probs = (logits.float() / T).softmax(1)  # T from <run>/temperature.json (1.0 if not fitted)
# channels: 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed (building damage, not flood extent)
```
The config is stored inside the checkpoint, so nothing else is needed. M0, M1 and M2 share
this call. The run folder also holds `config.yaml`, `metrics.jsonl` (per epoch, validation on
hurricane-florence) and, after evaluation, `eval_<role>.json` and `temperature.json`.

# Commands (Colab notebooks: `notebooks/m2/train_colab.ipynb`, `notebooks/m3/eval_export_colab.ipynb`)
| Step | Command |
|------|---------|
| Train (resumes from `last.pt`) | `python -m floodsitrep.cv.train --config configs/cv/<run>.yaml --xbd-root X --manifest M --out-dir <Drive>/checkpoints` |
| Val metrics + temperature | `python -m floodsitrep.cv.eval.evaluate --checkpoint <run>/best.pt --xbd-root X --role val --fit-temperature` |
| Predicted evidence | `python -m floodsitrep.cv.export_evidence --checkpoint <run>/best.pt --temperature <run>/temperature.json --xbd-root X --role val --out <dir> --schema evidence/schema/evidence.schema.json` |
| Mask-counting floor | `python -m floodsitrep.cv.export_evidence --from-gt --xbd-root X --role val --out <dir> --label-dir X/processed/evidence_labels` |
| Test (once, at the end) | `python -m floodsitrep.cv.eval.evaluate --checkpoint <run>/best.pt --xbd-root X --role test` |

# Definitions to use in the report (state the level next to every number)
- All model metrics are **pixel level** over valid (non-black) pixels.
- Localisation F1: building (classes 1-4) vs background.
- Damage F1 per class: over ground-truth building pixels only (xView2 convention; a GT building
  pixel predicted as background is a miss). Five-class F1/IoU over all pixels are also logged.
- xView2-style score = 0.3 x localisation F1 + 0.7 x harmonic mean of the four damage F1s.
  Ours is a pixel-level approximation on 512 px crops and our event split, not the official score.
- 95% intervals: percentile bootstrap over tiles (1000 resamples), not pixels.
- Calibration: one temperature fitted by NLL on validation pixels (2000 random valid pixels per
  tile); ECE with 15 equal-width bins on max-probability confidence, pixel level.
- Predicted evidence counts buildings as connected components of the predicted mask on the
  stitched 1024 px image (each counted once, by centroid). Touching buildings merge, so counts are
  approximate; the `--from-gt` run measures this floor against polygon-based label evidence.

# Executed notebooks (in the order they were run)
Each notebook is the Colab run that produced the numbers in the report, with its outputs kept.
Data and checkpoints stay on the team Drive (`MAAI 902 X 901 PROJECT/`); nothing here contains them,
but figure outputs show xBD imagery (CC BY-NC-SA 4.0, non-commercial, attribution, share-alike).
Several notebooks clone the feature branch of the day (`m2/models-train`, `m3/eval-export`,
`m3/eval-extras`); all of that code is now on `main`.

| # | Notebook | Date | GPU | What it produced |
|---|----------|------|-----|------------------|
| 1 | `notebooks/m2/01_data_checks_and_seed0_runs.ipynb` | 15 Nov | L4 | Loader and model checks, loss shares, smoke run (Gate 1); M1 Dice + focal, M1 weighted CE and M0 seed 0 (20 epochs each) |
| 2 | `notebooks/m3/01_metrics_calibration_export_m1_seeds.ipynb` | 15-17 Nov | T4 / L4 | Metric toy check, learning curves, Florence metrics with 95% CIs, temperature (T = 1.133), 4,660 predicted evidence files (Gate 2), mask-counting floor; M1 seeds 1-2 for both losses |
| 3 | `notebooks/m3/02_cloud_check.ipynb` | 17 Nov | CPU | Cloud screen of all 4,660 val/test crops and the 24 most-flagged crops |
| 4 | `notebooks/m3/03_paired_bootstrap_calibration_agreement.ipynb` | 17 Nov | T4 | Paired bootstrap CE vs Dice + focal, building-pixel reliability tables, agreement on building cells, cloud sensitivity, building-only temperature |
| 5 | `notebooks/m3/04_building_level_resolution_ladder.ipynb` | 17 Nov | T4 | Building-level metrics (Florence) and the resolution ladder (0.5 m, about 1 m, about 3 m) |
| 6 | `notebooks/m2/02_m0_seeds_and_first_segformer.ipynb` | 17 Nov | L4 | M0 seeds 1-2, M2 smoke test and M2 seed 0 |
| 7 | `notebooks/m2/03_segformer_seeds_and_variants.ipynb` | 18 Nov | A100 | M2 seeds 1-2, the gentle-schedule and ADE20K variants, the all-runs table |
| 8 | `notebooks/m3/05_report_figures.ipynb` | 18 Nov | CPU | Figures 1-9 |
| 9 | `notebooks/m3/06_nepal_final_test.ipynb` | 18 Nov | L4 | The single Nepal test: `decision.txt`, final-model metrics with CIs, building table, calibration transfer, all 12 comparison runs. The Nepal evidence-agreement cell was stopped (too slow) and not rerun |
| 10 | `notebooks/m3/07_destroyed_diagnosis_threshold_test.ipynb` | 19 Nov | L4 | Destroyed-class diagnosis and the decision-threshold test (tuned on half of Florence, checked on the other half) |
| 11 | `notebooks/m3/08_fig10_nepal_intervals.ipynb` | 20 Nov | CPU | Figure 10 (Nepal estimates with 95% intervals) |
| 12 | `notebooks/m3/09_dataset_counts.ipynb` | 21 Nov | CPU | Exact building and pixel counts per class, event and role, with cross-checks |

`notebooks/m2/train_colab.ipynb` and `notebooks/m3/eval_export_colab.ipynb` are the clean templates.
`confidence_table_building_cells.csv`: what each confidence level is worth on building pixels
(Florence and Nepal, T = 1.133), given to Member 4 for the checker.

# Run record (Florence validation, pixel level, best epoch by validation score)
All runs: 5,488 training crops (Harvey, Matthew, Midwest), batch 8, AdamW, warm-up then cosine, AMP.
Checkpoints: `<Drive>/checkpoints/<run name>/best.pt` (not in git).

| Run name | Model | Loss | Seed | Epochs | Best epoch | Val xView2 (pixel) | Checkpoint |
|----------|-------|------|------|--------|------------|--------------------|------------|
| smoke_m1_s0 | M1 siamese U-Net | Dice + focal | 0 | 2 (320 crops) | 1 | 0.157 | smoke_m1_s0/best.pt |
| **m1_siamese_unet_ce_s0 (final model)** | M1 siamese U-Net | weighted CE | 0 | 20 | 12 | **0.363** | m1_siamese_unet_ce_s0/best.pt |
| m1_siamese_unet_ce_s1 | M1 siamese U-Net | weighted CE | 1 | 20 | 12 | 0.363 | m1_siamese_unet_ce_s1/best.pt |
| m1_siamese_unet_ce_s2 | M1 siamese U-Net | weighted CE | 2 | 20 | 18 | 0.356 | m1_siamese_unet_ce_s2/best.pt |
| m1_siamese_unet_s0 | M1 siamese U-Net | Dice + focal | 0 | 20 | 9 | 0.316 | m1_siamese_unet_s0/best.pt |
| m1_siamese_unet_s1 | M1 siamese U-Net | Dice + focal | 1 | 20 | 14 | 0.326 | m1_siamese_unet_s1/best.pt |
| m1_siamese_unet_s2 | M1 siamese U-Net | Dice + focal | 2 | 20 | 11 | 0.313 | m1_siamese_unet_s2/best.pt |
| m0_early_fusion_unet_s0 | M0 early-fusion U-Net | Dice + focal | 0 | 20 | 9 | 0.286 | m0_early_fusion_unet_s0/best.pt |
| m0_early_fusion_unet_s1 | M0 early-fusion U-Net | Dice + focal | 1 | 20 | 7 | 0.274 | m0_early_fusion_unet_s1/best.pt |
| m0_early_fusion_unet_s2 | M0 early-fusion U-Net | Dice + focal | 2 | 20 | 8 | 0.268 | m0_early_fusion_unet_s2/best.pt |
| m2_siamese_segformer_ce_s0 | M2 siamese SegFormer (MiT-B1) | weighted CE | 0 | 20 | 2 | 0.362 | m2_siamese_segformer_ce_s0/best.pt |
| m2_siamese_segformer_ce_s1 | M2 siamese SegFormer (MiT-B1) | weighted CE | 1 | 20 | 6 | 0.334 | m2_siamese_segformer_ce_s1/best.pt |
| m2_siamese_segformer_ce_s2 | M2 siamese SegFormer (MiT-B1) | weighted CE | 2 | 20 | 5 | 0.338 | m2_siamese_segformer_ce_s2/best.pt |
| m2_siamese_segformer_gentle_ce_s0 | M2, learning rates / 5 (variant) | weighted CE | 0 | 10 | 6 | 0.348 | m2_siamese_segformer_gentle_ce_s0/best.pt |
| m2_siamese_segformer_ade_ce_s0 | M2, ADE20K encoder (variant) | weighted CE | 0 | 20 | 3 | 0.356 | m2_siamese_segformer_ade_ce_s0/best.pt |

M2 runs used `configs/cv/m2_siamese_segformer.yaml` with `--set loss=ce lr=6e-4 encoder_lr_mult=0.1`
(variants: gentle `lr=1.2e-4 epochs=10`; ADE20K `backbone=nvidia/segformer-b1-finetuned-ade-512-512`).
Seeds 1-2 used `--set seed=<n>`. Number of loader workers varied between runs (speed only).

# Nepal (test, scored once on 18 Nov; final model fixed in `decision.txt` beforehand)
| Model | Seed | Pixel score | Building score | Loc F1 (pixel) | Destroyed F1 (pixel) |
|-------|------|-------------|----------------|----------------|----------------------|
| **M1 weighted CE (final)** | 0 | **0.415** (95% CI 0.393-0.435) | **0.443** (0.422-0.460) | 0.725 | 0.140 |
| M1 weighted CE | 1 | 0.394 | 0.432 | 0.731 | 0.115 |
| M1 weighted CE | 2 | 0.440 | 0.465 | 0.742 | 0.157 |
| M1 Dice + focal | 0 / 1 / 2 | 0.299 / 0.324 / 0.374 | 0.299 / 0.309 / 0.363 | 0.668 / 0.686 / 0.700 | 0.058 / 0.064 / 0.117 |
| M0 Dice + focal | 0 / 1 / 2 | 0.355 / 0.358 / 0.293 | 0.349 / 0.357 / 0.285 | 0.672 / 0.697 / 0.616 | 0.131 / 0.106 / 0.075 |
| M2 SegFormer, weighted CE | 0 / 1 / 2 | 0.389 / 0.366 / 0.386 | 0.399 / 0.387 / 0.399 | 0.748 / 0.741 / 0.765 | 0.136 / 0.110 / 0.140 |
