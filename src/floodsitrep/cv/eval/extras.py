"""Evaluation additions: paired bootstrap, building-pixel calibration, agreement on building cells.

Owner: member 3. All metrics are PIXEL level unless stated.

1. ``collect``: one pass of a checkpoint over a role. Saves, per tile, the 5x5 pixel confusion
   matrix, and a random sample of PREDICTED-building pixels (logits + true label). Predicted-
   building pixels are the ones whose probabilities become a cell's ``confidence`` in the
   evidence, so that is where calibration matters for the checker.
2. ``paired_bootstrap``: resample the SAME tiles for two runs and recompute the difference in a
   metric each time. Tile difficulty is shared by both runs and cancels out, so this tests "is A
   better than B on this kind of data" more sharply than comparing two separate intervals.
3. ``reliability``: ECE plus the per-bin table (confidence vs accuracy) on the building sample.
4. ``building_cell_agreement``: predicted vs label evidence over cells whose LABEL has at least one
   building (valid_fraction >= 0.5), plus how many label-empty cells the model fills with
   buildings. Empty cells agree trivially, so leaving them out gives the honest number.
"""
import json
from pathlib import Path

import numpy as np

from floodsitrep.cv.eval import metrics as M
from floodsitrep.cv.eval.calibration import ece, softmax


def collect(checkpoint, xbd_root, manifest, out_path, role="val", px_per_tile=1000, seed=0,
            batch_size=8, limit=0, device=None):
    """Run ``checkpoint`` over ``role``; save tile ids, per-tile confusions and a building sample."""
    import torch

    from floodsitrep.cv.eval.infer import iter_predictions
    from floodsitrep.cv.models import load_checkpoint

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    rng = np.random.default_rng(seed)
    ids, cms, lg, lb = [], [], [], []
    for r in iter_predictions(model, xbd_root, manifest, role, batch_size, device=device,
                              limit=limit):
        logits, mask, valid = r["logits"], r["mask"], r["valid"]
        pred = logits.argmax(0)
        ids.append(r["tile_id"])
        cms.append(M.confusion(pred, mask, valid))
        sel = np.flatnonzero((valid & (pred > 0) & (mask <= 4)).ravel())
        if len(sel) > px_per_tile:
            sel = rng.choice(sel, px_per_tile, replace=False)
        lg.append(logits.reshape(logits.shape[0], -1)[:, sel].T.astype(np.float16))
        lb.append(mask.ravel()[sel].astype(np.uint8))
    np.savez_compressed(out_path, tile_ids=np.array(ids), cms=np.stack(cms),
                        bld_logits=np.concatenate(lg), bld_labels=np.concatenate(lb))
    return out_path


def load(path):
    z = np.load(path)
    return {k: z[k] for k in z.files}


def paired_bootstrap(a, b, metric="xview2_pixel_score", n_boot=2000, seed=0):
    """Difference metric(a) - metric(b) over the same resampled tiles.

    a, b: dicts from ``load`` (must cover the same tiles). Returns the point difference, the 95%
    percentile interval, and the share of resamples where the difference is <= 0.
    """
    ia = {t: i for i, t in enumerate(a["tile_ids"])}
    order = [ia[t] for t in b["tile_ids"]]  # align a to b's tile order
    ca, cb = a["cms"][order], b["cms"]
    if len(ca) != len(cb):
        raise ValueError("runs cover different tiles")

    def f(cm):
        return M.flat(M.summarise(cm))[metric]

    point = f(ca.sum(0)) - f(cb.sum(0))
    rng = np.random.default_rng(seed)
    diffs = []
    for _ in range(n_boot):
        idx = rng.integers(0, len(cb), len(cb))
        d = f(ca[idx].sum(0)) - f(cb[idx].sum(0))
        if not np.isnan(d):
            diffs.append(d)
    diffs = np.array(diffs)
    lo, hi = np.quantile(diffs, [0.025, 0.975])
    return {"metric": metric, "diff": float(point), "lo": float(lo), "hi": float(hi),
            "share_le_0": float((diffs <= 0).mean()), "tiles": len(cb), "n_boot": len(diffs)}


def reliability(logits, labels, t=1.0, n_bins=10):
    """ECE and per-bin table on a pixel sample. Returns (ece, rows)."""
    p = softmax(np.asarray(logits, dtype=np.float64), t)
    conf, pred = p.max(1), p.argmax(1)
    correct = pred == np.asarray(labels)
    edges = np.linspace(0, 1, n_bins + 1)
    idx = np.clip(np.digitize(conf, edges[1:-1], right=True), 0, n_bins - 1)
    rows = []
    for k in range(n_bins):
        m = idx == k
        if m.any():
            rows.append({"bin": f"{edges[k]:.1f}-{edges[k + 1]:.1f}", "pixels": int(m.sum()),
                         "mean_confidence": float(conf[m].mean()),
                         "accuracy": float(correct[m].mean())})
    return ece(p, labels, n_bins), rows


def building_cell_agreement(pred_dir, label_dir, prefix=""):
    """Agreement over cells whose label has >= 1 building (valid_fraction >= 0.5)."""
    from floodsitrep.common.interfaces import quantifier

    classes = ("none", "minor", "major", "destroyed")

    def qbin(cell, k):
        return "none_present" if cell["buildings"] == 0 else quantifier(cell["class_fraction"][k])

    n = exact = pred_empty = 0
    agree = {k: 0 for k in classes}
    tot_l = tot_p = 0
    empty_label_cells = false_filled = 0
    tiles = 0
    for lp in sorted(Path(label_dir).glob(f"{prefix}*.json")):
        pp = Path(pred_dir) / lp.name
        if not pp.exists():
            continue
        tiles += 1
        lab, prd = json.loads(lp.read_text()), json.loads(pp.read_text())
        for lc, pc in zip(lab["cells"], prd["cells"]):
            if lc.get("valid_fraction", 1.0) < 0.5:
                continue
            if lc["buildings"] == 0:
                empty_label_cells += 1
                false_filled += pc["buildings"] > 0
                continue
            n += 1
            exact += lc["buildings"] == pc["buildings"]
            pred_empty += pc["buildings"] == 0
            tot_l += lc["buildings"]
            tot_p += pc["buildings"]
            for k in classes:
                agree[k] += qbin(lc, k) == qbin(pc, k)
    nan = float("nan")
    return {
        "tiles": tiles,
        "label_cells_with_buildings": n,
        "count_exact_match": exact / n if n else nan,
        "building_ratio_pred_over_label": tot_p / tot_l if tot_l else nan,
        "cells_model_finds_no_building": pred_empty / n if n else nan,
        "quantifier_agreement": {k: v / n if n else nan for k, v in agree.items()},
        "label_empty_cells": empty_label_cells,
        "label_empty_cells_model_fills": (false_filled / empty_label_cells
                                          if empty_label_cells else nan),
    }
