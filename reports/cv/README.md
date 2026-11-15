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

# Run record (fill from real outputs only)
| Run name | Model | Loss | Seed | Epochs | Best epoch | Val xView2 (pixel) | Checkpoint |
|----------|-------|------|------|--------|------------|--------------------|------------|
