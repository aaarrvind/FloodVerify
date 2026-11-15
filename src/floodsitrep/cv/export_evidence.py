"""Predictions -> evidence JSON with source=predicted. Owner: member 3.

Contract signature (common/interfaces.py): ``export(checkpoint, tile_dir, out_dir) -> list[Path]``
where ``tile_dir`` is the xBD root (the folder with train/ hold/ tier3/ ...).

    python -m floodsitrep.cv.export_evidence --checkpoint <run>/best.pt \
        --temperature <run>/temperature.json --xbd-root /content/xbd \
        --manifest /content/xbd/rscc_flood_manifest.csv --role val \
        --out "<Drive>/evidence_predicted/<run>" --schema evidence/schema/evidence.schema.json

How a predicted evidence file is built (same schema and grid code as label evidence):
1. pred = argmax of the model logits per pixel; pixels with valid == False are set to
   background, so no building is invented inside black no-data areas.
2. The four 512 px crops of each 1024 px xBD image are stitched back together, and
   buildings = 4-connected components of pred > 0 (at least ``min_px`` pixels) on the whole
   image. Each component gets the majority damage class of its pixels (ties go to the more
   severe class) and is counted ONCE, in the crop and grid cell that contain its pixel
   centroid, the same rule label evidence uses for polygons. (Without stitching, a building
   cut by a crop edge would be counted in both crops.)
3. Cells are made by member 1's ``grid_cells`` (labels_to_evidence.py), so counts and
   class fractions are computed exactly as for label evidence.
4. ``valid_fraction`` uses the same cell slicing and rounding as member 1's
   ``valid_fractions``. Cells below 0.5 keep the field (the checker needs it).
5. ``confidence`` per cell, from temperature-scaled softmax probabilities:
   mean max-probability over predicted-building pixels in the valid part of the cell; if
   the cell has no predicted building pixels, the mean probability of background over its
   valid pixels; 0.0 if the cell has no valid pixels.

Counts are APPROXIMATE: touching buildings merge into one component, so predicted counts
are biased low against label evidence (which counts polygons). ``--from-gt`` measures this
floor: it runs steps 2-4 on the ground-truth mask and compares with label evidence.
Needs numpy, scipy (connected components), torch for model export.
"""
import argparse
import json
from pathlib import Path

import numpy as np

from floodsitrep.data.labels_to_evidence import CLASSES, grid_cells

SIZE = 512
GRID = 3
MIN_BUILDING_PX = 10  # drop speckle; about 2.5 m2 at a nominal 0.5 m GSD


def valid_fractions_from_array(valid, size=SIZE, grid=GRID):
    """Same slicing and rounding as labels_to_evidence.valid_fractions, from a bool array."""
    out = []
    for r in range(grid):
        for c in range(grid):
            cell = valid[r * size // grid:(r + 1) * size // grid,
                         c * size // grid:(c + 1) * size // grid]
            out.append(round(float(cell.mean()), 3))
    return out


def buildings_from_mask(mask, min_px=0):
    """Damage mask (0 bg, 1-4) -> list of (x, y, class) per 4-connected building component."""
    from scipy import ndimage

    mask = np.asarray(mask)
    lab, n = ndimage.label(mask > 0)
    if n == 0:
        return []
    idx = np.arange(1, n + 1)
    sizes = ndimage.sum_labels(np.ones_like(lab), lab, idx)
    cy, cx = (np.array(v) for v in zip(*ndimage.center_of_mass(np.ones_like(lab), lab, idx)))
    # per-component class histogram (classes 1..4)
    hist = np.stack([ndimage.sum_labels(mask == k, lab, idx) for k in (1, 2, 3, 4)], 1)
    out = []
    for i in range(n):
        if sizes[i] < max(min_px, 1):
            continue
        h = hist[i]
        k = int(np.flatnonzero(h == h.max())[-1])  # ties -> more severe
        out.append((float(cx[i]) + 0.5, float(cy[i]) + 0.5, CLASSES[k]))
    return out


def assign_to_crops(buildings, origins, size=SIZE):
    """Buildings in full-image coords -> {part: [(x, y, cls) relative to that crop]}.
    origins: {part: (x0, y0)}. A building goes to the crop containing its centroid."""
    out = {p: [] for p in origins}
    for x, y, cls in buildings:
        for p, (x0, y0) in origins.items():
            if x0 <= x < x0 + size and y0 <= y < y0 + size:
                out[p].append((x - x0, y - y0, cls))
                break
    return out


def stitched_buildings(crops, min_px=0, size=SIZE):
    """crops: {part: class map [size,size]} of ONE 1024 px image (missing parts allowed).
    Returns {part: buildings relative to the crop}, each building counted once."""
    from floodsitrep.data.tiling import PART_ORIGIN

    full = np.zeros((2 * size, 2 * size), np.uint8)
    for p, m in crops.items():
        x0, y0 = PART_ORIGIN[p]
        full[y0:y0 + size, x0:x0 + size] = m
    origins = {p: PART_ORIGIN[p] for p in crops}
    return assign_to_crops(buildings_from_mask(full, min_px), origins, size)


def cell_confidence(probs, pred, valid, size=SIZE, grid=GRID):
    """Per-cell confidence (row-major) from probabilities [5,H,W]; see module docstring."""
    pmax = probs.max(0)
    out = []
    for r in range(grid):
        for c in range(grid):
            sl = (slice(r * size // grid, (r + 1) * size // grid),
                  slice(c * size // grid, (c + 1) * size // grid))
            v = valid[sl]
            b = v & (pred[sl] > 0)
            if b.any():
                out.append(round(float(pmax[sl][b].mean()), 4))
            elif v.any():
                out.append(round(float(probs[0][sl][v].mean()), 4))
            else:
                out.append(0.0)
    return out


def evidence_from_prediction(tile_id, pred, valid, probs=None, min_px=MIN_BUILDING_PX,
                             mask_invalid=True, source="predicted", gsd_m=0.5, buildings=None):
    """One evidence dict from a class map [H,W] (and optional probabilities [5,H,W]).
    ``buildings``: precomputed (x, y, class) list for this crop (from stitched_buildings);
    if None, components are found in this crop alone."""
    pred = np.asarray(pred).astype(np.uint8)
    valid = np.asarray(valid).astype(bool)
    size = pred.shape[0]
    if mask_invalid:
        pred = np.where(valid, pred, 0)
    if buildings is None:
        buildings = buildings_from_mask(pred, min_px)
    cells = grid_cells(buildings, size)
    conf = cell_confidence(probs, pred, valid, size) if probs is not None else [1.0] * 9
    for cell, vf, cf in zip(cells, valid_fractions_from_array(valid, size), conf):
        cell["confidence"] = cf
        cell["valid_fraction"] = vf
    return {"tile_id": tile_id, "source": source, "gsd_m": gsd_m, "grid": "3x3", "cells": cells}


def evidence_for_image(parts, min_px=MIN_BUILDING_PX, mask_invalid=True):
    """parts: {part: (tile_id, pred, valid, probs or None)} for ONE 1024 px image.
    Returns a list of evidence dicts, buildings counted once across the crops."""
    crops = {}
    for p, (_, pred, valid, _) in parts.items():
        pred = np.asarray(pred).astype(np.uint8)
        crops[p] = np.where(valid, pred, 0) if mask_invalid else pred
    per_crop = stitched_buildings(crops, min_px)
    return [evidence_from_prediction(tid, crops[p], valid, probs, min_px, mask_invalid=False,
                                     buildings=per_crop[p])
            for p, (tid, _, valid, probs) in parts.items()]


def _split_id(tile_id):
    base, part = tile_id.rsplit("_p", 1)
    return base, part


def _softmax(logits, t):
    z = logits / t
    z = z - z.max(0, keepdims=True)
    e = np.exp(z)
    return e / e.sum(0, keepdims=True)


def _write(ev, out_dir, validator=None):
    if validator is not None:
        errs = [e.message for e in validator.iter_errors(ev)]
        if errs:
            raise ValueError(f"{ev['tile_id']}: schema error: {errs[0]}")
    p = Path(out_dir) / f"{ev['tile_id']}.json"
    p.write_text(json.dumps(ev, indent=2))
    return p


def _validator(schema):
    if not schema:
        return None
    import jsonschema
    return jsonschema.Draft202012Validator(json.loads(Path(schema).read_text()))


def export(checkpoint, tile_dir, out_dir, manifest=None, role="val", temperature=None,
           batch_size=8, num_workers=2, device=None, min_px=MIN_BUILDING_PX, limit=0,
           hflip_tta=False, schema=None):
    """Run the checkpoint over a role and write one predicted evidence JSON per tile."""
    import torch

    from floodsitrep.cv.eval.calibration import load_temperature
    from floodsitrep.cv.eval.infer import iter_predictions
    from floodsitrep.cv.models import load_checkpoint

    manifest = manifest or str(Path(tile_dir) / "rscc_flood_manifest.csv")
    t = load_temperature(temperature) if isinstance(temperature, (str, Path)) else (
        float(temperature) if temperature else 1.0)
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val = _validator(schema)
    paths, cur, parts = [], None, {}

    def flush():
        before = len(paths)
        for ev in evidence_for_image(parts, min_px):
            paths.append(_write(ev, out, val))
        if len(paths) // 500 > before // 500:
            print(f"{len(paths)} tiles written", flush=True)

    # rows are sorted so the 4 crops of one image arrive together
    for r in iter_predictions(model, tile_dir, manifest, role, batch_size, num_workers,
                              device, limit=limit, hflip_tta=hflip_tta, sort=True):
        base, part = _split_id(r["tile_id"])
        if base != cur and parts:
            flush()
            parts = {}
        cur = base
        probs = _softmax(r["logits"].astype(np.float64), t).astype(np.float32)
        parts[part] = (r["tile_id"], probs.argmax(0), r["valid"], probs)
    if parts:
        flush()
    return paths


def export_from_gt(tile_dir, manifest, out_dir, role=None, limit=0, schema=None):
    """Steps 2-4 on the ground-truth masks (for the agreement check with label evidence)."""
    from floodsitrep.cv.eval.infer import make_dataset
    from floodsitrep.data.tiling import load_crop, tile_name

    ds = make_dataset(tile_dir, manifest, role, limit, sort=True)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    val = _validator(schema)
    groups = {}
    for row in ds.rows:
        groups.setdefault((row["split"], row["event"], row["tile_id"]), []).append(row)
    paths = []
    for rows in groups.values():
        parts = {}
        for row in rows:
            _, _, mask, valid = load_crop(tile_dir, row)
            parts[row["part"]] = (tile_name(row), np.where(mask <= 4, mask, 0), valid, None)
        for ev in evidence_for_image(parts, min_px=0, mask_invalid=False):
            paths.append(_write(ev, out, val))
    return paths


def compare_with_labels(pred_dir, label_dir):
    """Agreement between evidence folders, over tiles present in both.

    Reports, over cells with valid_fraction >= 0.5: exact building-count match rate, total
    predicted / label building ratio, mean absolute class-fraction error (cells where both
    have buildings), and quantifier-bin agreement per damage class (the bins the checker
    uses), with "no buildings" as its own bin.
    """
    from floodsitrep.common.interfaces import quantifier

    def qbin(cell, k):
        return "none_present" if cell["buildings"] == 0 else quantifier(cell["class_fraction"][k])

    n = exact = both = 0
    tot_p = tot_l = 0
    mae = []
    agree = {k: 0 for k in CLASSES}
    tiles = 0
    for lp in sorted(Path(label_dir).glob("*.json")):
        pp = Path(pred_dir) / lp.name
        if not pp.exists():
            continue
        tiles += 1
        L, P = json.loads(lp.read_text()), json.loads(pp.read_text())
        for lc, pc in zip(L["cells"], P["cells"]):
            if lc.get("valid_fraction", 1.0) < 0.5:
                continue
            n += 1
            exact += lc["buildings"] == pc["buildings"]
            tot_l += lc["buildings"]
            tot_p += pc["buildings"]
            if lc["buildings"] and pc["buildings"]:
                both += 1
                mae.append(np.mean([abs(lc["class_fraction"][k] - pc["class_fraction"][k])
                                    for k in CLASSES]))
            for k in CLASSES:
                agree[k] += qbin(lc, k) == qbin(pc, k)
    return {
        "tiles": tiles, "cells": n,
        "count_exact_match": exact / n if n else float("nan"),
        "building_ratio_pred_over_label": tot_p / tot_l if tot_l else float("nan"),
        "class_fraction_mae": float(np.mean(mae)) if mae else float("nan"),
        "cells_both_with_buildings": both,
        "quantifier_agreement": {k: v / n if n else float("nan") for k, v in agree.items()},
    }


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint")
    ap.add_argument("--temperature", default=None, help="temperature.json from evaluate.py")
    ap.add_argument("--xbd-root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--role", default="val", help="train, val, test or all")
    ap.add_argument("--out", required=True)
    ap.add_argument("--schema", default=None)
    ap.add_argument("--min-px", type=int, default=MIN_BUILDING_PX)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hflip-tta", action="store_true")
    ap.add_argument("--from-gt", action="store_true",
                    help="build from ground-truth masks instead of a model (agreement check)")
    ap.add_argument("--label-dir", default=None,
                    help="label evidence folder: print agreement with it after export")
    a = ap.parse_args(argv)
    manifest = a.manifest or str(Path(a.xbd_root) / "rscc_flood_manifest.csv")
    role = None if a.role == "all" else a.role
    if a.from_gt:
        paths = export_from_gt(a.xbd_root, manifest, a.out, role, a.limit, a.schema)
    else:
        if not a.checkpoint:
            ap.error("--checkpoint is required unless --from-gt")
        paths = export(a.checkpoint, a.xbd_root, a.out, manifest, role, a.temperature,
                       a.batch_size, min_px=a.min_px, limit=a.limit, hflip_tta=a.hflip_tta,
                       schema=a.schema)
    print(f"wrote {len(paths)} evidence files to {a.out}")
    if a.label_dir:
        print(json.dumps(compare_with_labels(a.out, a.label_dir), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
