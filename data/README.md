# Data (never committed to git)

Raw data, tiles, evidence folders and checkpoints live on the shared Google Drive
(`MAAI 902 X 901 PROJECT`). Do not put them in git, and never upload xBD to a public place.

## Sources and licences

| Dataset | Used for | Where | Licence | Checked |
|---------|----------|-------|---------|---------|
| xBD (official, xview2.org) | training, validation, test, evidence | Drive `xbd/` (train, test, hold, tier3) | CC BY-NC-SA 4.0, non-commercial, share alike, credit | Terms page read 2026-11-13 |
| RSCC captions (`BiliSakura/RSCC`, file `RSCC_qvq.jsonl`) | test sentences for the claim checker | Drive `xbd/rscc_flood_manifest.csv` (flood events only) | CC BY 4.0 | Hugging Face card, 2026-11-13 |
| HumAID (`QCRI/HumAID-all`) | training (text classifier) | Drive only (humaid/), not in git | Research use only (humanitarian computing); non-commercial. Terms: https://crisisnlp.qcri.org/terms-of-use.html. Dataset page: https://crisisnlp.qcri.org/humaid_dataset
| Qwen3-VL-4B-Instruct (`Qwen/Qwen3-VL-4B-Instruct`) | main captioner (notebooks/m4/02) | download from Hugging Face | Apache-2.0 | Hugging Face model card, 2026-11-17 |
| Qwen2.5-VL-3B-Instruct (`Qwen/Qwen2.5-VL-3B-Instruct`) | comparison captioner only, not in the demo | download from Hugging Face | Qwen Research License (`qwen-research`): research use, non-commercial. The 7B model is Apache, the 3B is not | Hugging Face model card, 2026-11-17 |
| NCEMA "Rain Floods" guidance (EN and AR) | advisory snippets | Drive `SEM 1 FINAL PROJECT/guidance_ncema_rain_floods_en_ar.md` | Reuse terms not checked [VERIFY]; cite NCEMA | 2026-11-13 |
| FEMA flood sheet P-2406 (AR) | advisory snippets | Drive `guidance_passages_fema_ar_clean.csv` | FEMA: most material free of copyright; cite FEMA | fema.gov website-information page, 2026-11-14 |
| BRIGHT (optional) | evaluation (Libya flood) | download only if used | Mostly CC BY-NC, varies per event | [VERIFY] before use |
| WorldFloods v2 (optional) | likely cut: flood extent only, not before/after | - | CC BY-NC 4.0 | 2026-11-13 |
| Dubai Apr 2024 (Landsat 30 m, Sentinel-2 10 m) | qualitative demo only, no labels | - | NASA/ESA open | no building-level data exists |

xBD citation: Gupta et al., "Creating xBD: A Dataset for Assessing Building Damage from
Satellite Imagery", IEEE CVPR Workshops 2019, pp. 10-17.

## Known limits (state them in the reports)
- xBD has no flood-extent labels; it labels building damage after flood and storm events. The task is building damage, not a flood map.
- tier3 has no `targets/` folder. The Nepal masks (`tier3/targets/`) were drawn by us from the label polygons (un-classified buildings = background; 99.6% pixel agreement with the official masks on 60 Harvey tiles).
- RSCC captions are machine-written (Qwen QvQ-Max) and noisy. Never ground truth. Caption "Level" k means no/minor/major/destroyed for k = 0/1/2/3 (mask value = Level + 1). In a test of 200 statements, damaged-building counts (Levels 1-3) were close to the polygon counts in 80 of 86 cases; "no damage" counts (Level 0) were off in 69 of 114.
- Label evidence counts buildings from the label polygons (exact). Predicted evidence will come from the model's mask, so its counts are approximate. Do not compare them without saying so.
- Black no-data areas exist in some tiles; `valid_fraction` records them and cells below 0.5 are `not_verifiable`.
- `gsd_m` in evidence is a nominal 0.5 m; true resolution varies.
- Evidence comes from annotator labels, so cloud or haze in an image is not captured: `valid_fraction` only detects black no-data pixels. A caption can therefore be checked against labels for a region it could not see. A brightness and contrast test could not tell haze from dark vegetation, so there is no haze flag yet.

## Folder layout on the Drive
```
xbd/
  train/ test/ hold/ tier3/        images/ labels/ targets/   (tier3: Nepal only)
  rscc_flood_manifest.csv          10,148 rows, one per 512 px quarter of a 1024 px tile
  flood_splits.csv                 older full-tile split file (superseded by processed/tiles.csv)
  flood_tile_stats.csv             building counts per class for every 1024 tile
  processed/
    tiles.csv                      tile_id, event, xbd_split, part, role
    evidence_labels/               one label-derived evidence JSON per tile
    class_weights.json             pixel class counts and starting loss weights (train role)
    captions/                      per-cell captions (qwen3vl4b_cells_50.json, qwen25vl3b_cells_20_v2.json), claims_verdicts.csv, captioning.yaml
```
Manifest part numbering: part 1 top-left, part 2 top-right, part 3 bottom-left, part 4
bottom-right (x0, y0 in the manifest give the crop origin).

## Split (see `src/floodsitrep/data/splits.py`)
Whole events, never tiles: train = hurricane-harvey, hurricane-matthew, midwest-flooding;
val = hurricane-florence; test = nepal-flooding (unseen in training).
Share of major + destroyed buildings (official train split): harvey 37.5%, matthew 29.4%,
florence 20.2%, midwest 2.2%, nepal about 12% (few destroyed: 502 buildings).
Midwest has almost no severe damage, so it is not used for validation.

## Rebuild everything (any computer with the xBD folders, no GPU needed)
```
pip install -e . --no-deps && pip install numpy pillow jsonschema
python -m floodsitrep.data.build_evidence --xbd-root <xbd> --manifest <xbd>/rscc_flood_manifest.csv \
  --out-evidence <xbd>/processed/evidence_labels --out-tiles <xbd>/processed/tiles.csv \
  --schema evidence/schema/evidence.schema.json
python -m floodsitrep.data.class_stats --xbd-root <xbd> --manifest <xbd>/rscc_flood_manifest.csv \
  --role train --out <xbd>/processed/class_weights.json
```

## Loading tiles for training (Member 2)
```python
from floodsitrep.data.dataset import XbdTiles
ds = XbdTiles("/content/xbd", "/content/xbd/rscc_flood_manifest.csv", role="train", augment=True)
item = ds[0]  # pre, post (3x512x512 float), mask (512x512 long, 0..4), valid (bool), tile_id
```
Get the data onto Colab's local disk first (reading thousands of files from Drive during
training is slow). `train` and `test` come as single `.tar` files in Drive `xbd_tars/`:
```python
D = "/content/drive/MyDrive/MAAI 902 X 901 PROJECT"
!mkdir -p /content/xbd
!tar -xf "{D}/xbd_tars/train_images_labels_targets.tar" -C /content/xbd
!tar -xf "{D}/xbd_tars/test_images_labels_targets.tar" -C /content/xbd
!cp -r "{D}/xbd/hold" /content/xbd/            # validation tiles are in train/hold/test folders
!cp -r "{D}/xbd/tier3" /content/xbd/           # Nepal (test role)
!cp "{D}/xbd/rscc_flood_manifest.csv" /content/xbd/
```
Colab setup (tested on a T4, 15 Nov): clone the repo, then `pip install -q -e . --no-deps jsonschema`.
The running notebook does not see a package installed after it started, so either restart the
runtime or run `import sys; sys.path.insert(0, "/content/FloodVerify/src")` before importing
`floodsitrep`. Role sizes to expect: train 5,488, val 2,184, test 2,476 tiles.

Ignore pixels where `valid` is False when computing the loss.
Starting loss weights (inverse square root of class frequency, 500 random train tiles,
background / no / minor / major / destroyed): 0.10 / 0.52 / 0.88 / 0.75 / 2.74. About 93% of
pixels are background and only 0.13% are destroyed buildings. Tune them; they are a start.
