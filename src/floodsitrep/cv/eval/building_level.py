"""Building-level evaluation and the resolution ladder. Owner: member 3.

Pixel metrics weigh a large building more than a small one and say nothing about whether a
building was found as a whole. Here every labelled building counts once.

Per 1024 px xBD image (its 512 px crops are predicted, then stitched back together):

1. Ground-truth buildings = the label polygons (Member 1's ``parse_wkt``; ``un-classified``
   skipped), each rasterised to its own pixels. Buildings with no valid pixels are skipped.
2. A building is DETECTED if at least ``detect_share`` (default 0.5) of its valid pixels are
   predicted as building (classes 1-4). Its predicted damage is the majority predicted class
   among those building pixels (ties go to the more severe class). Otherwise it is MISSED.
3. Predicted buildings = 4-connected components of the predicted building mask with at least
   ``min_px`` pixels (the same rule as the evidence export). A component is FALSE if less than
   half of its pixels fall on any labelled building.

Building-level numbers (our definitions, analogous to the xView2 score; state them):
- localisation recall = detected / labelled buildings
- localisation precision = (components - false components) / components
- damage F1 per class over ALL labelled buildings: a missed building is a miss for its class
- building-level xView2-style score = 0.3 x localisation F1 + 0.7 x harmonic mean of the four
  damage F1s.
Confidence intervals: bootstrap over 1024 px images.

Resolution ladder: ``degrade`` = f shrinks both images by f (area averaging) and enlarges them
back to 512 px (bilinear) before the model sees them. f = 2 and 6 approximate 1 m and 3 m from
a nominal 0.5 m GSD. The model is NOT retrained, so this measures robustness of a 0.5 m model
to coarser input, not what a model trained at 3 m could reach.

Needs numpy, scipy, pillow, torch.
"""
import json

import numpy as np

from floodsitrep.cv.eval import metrics as M

CLS_INDEX = {"no-damage": 1, "minor-damage": 2, "major-damage": 3, "destroyed": 4}


def label_buildings(label_path):
    """Label JSON -> list of (class 1-4, polygon points) for classified buildings."""
    from floodsitrep.data.labels_to_evidence import parse_wkt

    with open(label_path) as fh:
        lab = json.load(fh)
    out = []
    for feat in lab["features"]["xy"]:
        k = CLS_INDEX.get(feat["properties"].get("subtype"))
        pts = parse_wkt(feat["wkt"])
        if k is not None and len(pts) >= 3:
            out.append((k, pts))
    return out


def rasterise(buildings, size=1024):
    """Instance map: 0 = no building, i+1 = building i (later polygons draw on top)."""
    from PIL import Image, ImageDraw

    im = Image.new("I", (size, size), 0)
    d = ImageDraw.Draw(im)
    for i, (_, pts) in enumerate(buildings):
        d.polygon(pts, fill=i + 1)
    return np.array(im, dtype=np.int32)


def score_image(pred, valid, buildings, inst=None, detect_share=0.5, min_px=10):
    """One stitched image. pred uint8 [H,W] 0-4, valid bool [H,W].

    Returns (bcm, n_comp, n_false, n_skipped): bcm is a 4x5 matrix, rows = true class 1-4,
    columns = 0 missed, 1-4 predicted class.
    """
    from scipy import ndimage

    if inst is None:
        inst = rasterise(buildings, pred.shape[0])
    bcm = np.zeros((4, 5), np.int64)
    skipped = 0
    pb = (pred > 0) & valid
    nb = len(buildings) + 1
    n_valid = np.bincount(inst[valid].ravel(), minlength=nb)          # valid px per building
    sel = pb & (inst > 0)
    hist = np.bincount((inst[sel] * 5 + pred[sel]).ravel(), minlength=nb * 5).reshape(nb, 5)
    for i, (k, _) in enumerate(buildings):
        n = int(n_valid[i + 1])
        if n == 0:
            skipped += 1
            continue
        counts = hist[i + 1, 1:]                                         # predicted 1-4
        if counts.sum() < detect_share * n:
            bcm[k - 1, 0] += 1
            continue
        bcm[k - 1, 1 + int(np.flatnonzero(counts == counts.max())[-1])] += 1
    lab, nc = ndimage.label(pb)
    n_comp = n_false = 0
    if nc:
        idx = np.arange(1, nc + 1)
        size = ndimage.sum_labels(np.ones_like(lab), lab, idx)
        on_bld = ndimage.sum_labels(inst > 0, lab, idx)
        keep = size >= max(min_px, 1)
        n_comp = int(keep.sum())
        n_false = int((keep & (on_bld < 0.5 * size)).sum())
    return bcm, n_comp, n_false, skipped


def summarise_buildings(bcm, n_comp, n_false):
    """Building-level numbers from a summed 4x5 matrix and component counts."""
    bcm = np.asarray(bcm, dtype=np.float64)
    total = bcm.sum()
    detected = bcm[:, 1:].sum()
    rec = detected / total if total else float("nan")
    prec = (n_comp - n_false) / n_comp if n_comp else float("nan")
    loc = 2 * prec * rec / (prec + rec) if prec + rec > 0 else 0.0
    f1 = {}
    for c, name in enumerate(("no_damage", "minor", "major", "destroyed")):
        tp = bcm[c, c + 1]
        fn = bcm[c].sum() - tp
        fp = bcm[:, c + 1].sum() - tp
        d = 2 * tp + fp + fn
        f1[name] = float(2 * tp / d) if d else float("nan")
    hm = M.harmonic_mean(list(f1.values()))
    return {"level": "building", "buildings": int(total), "loc_recall": float(rec),
            "loc_precision": float(prec), "loc_f1": float(loc),
            **{f"dmg_f1_{k}": v for k, v in f1.items()}, "damage_f1_hm": hm,
            "xview2_building_score": 0.3 * loc + 0.7 * hm}


def bootstrap_buildings(bcms, comps, falses, n_boot=1000, seed=0):
    """Percentile 95% intervals over images. bcms [N,4,5], comps/falses [N]."""
    bcms, comps, falses = np.asarray(bcms), np.asarray(comps), np.asarray(falses)
    point = summarise_buildings(bcms.sum(0), comps.sum(), falses.sum())
    keys = [k for k, v in point.items() if isinstance(v, float)]
    rng = np.random.default_rng(seed)
    draws = {k: [] for k in keys}
    for _ in range(n_boot):
        i = rng.integers(0, len(bcms), len(bcms))
        s = summarise_buildings(bcms[i].sum(0), comps[i].sum(), falses[i].sum())
        for k in keys:
            draws[k].append(s[k])
    out = {}
    for k in keys:
        a = np.array([x for x in draws[k] if not np.isnan(x)])
        lo, hi = np.quantile(a, [0.025, 0.975]) if len(a) else (float("nan"),) * 2
        out[k] = {"value": point[k], "lo": float(lo), "hi": float(hi)}
    return point, out


def degrade_batch(x, f):
    """Shrink a normalised image batch by factor f (area) and enlarge back (bilinear)."""
    import torch.nn.functional as F

    if f == 1:
        return x
    h, w = x.shape[-2:]
    small = F.interpolate(x, size=(max(1, round(h / f)), max(1, round(w / f))), mode="area")
    return F.interpolate(small, size=(h, w), mode="bilinear", align_corners=False)


def run(checkpoint, xbd_root, manifest, role="val", degrade=1, batch_size=8, limit=0,
        device=None, out_path=None, detect_share=0.5, min_px=10):
    """Predict every crop of ``role``, stitch per image, score pixels and buildings.

    Saves (if out_path) per-crop pixel confusions and per-image building results to .npz.
    """
    import torch
    from torch.utils.data import DataLoader

    from floodsitrep.cv.eval.infer import make_dataset
    from floodsitrep.cv.models import load_checkpoint
    from floodsitrep.data.tiling import PART_ORIGIN, source_paths

    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = load_checkpoint(checkpoint, device)
    ds = make_dataset(xbd_root, manifest, role, limit, sort=True)
    rows = {f"{r['event']}_{r['tile_id']}_p{r['part']}": r for r in ds.rows}
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=2,
                    pin_memory=device == "cuda")
    amp = device == "cuda"
    crop_ids, crop_cms = [], []
    img_ids, bcms, comps, falses, skipped = [], [], [], [], []
    state = {"base": None, "pred": None, "valid": None}

    def flush():
        if state["base"] is None:
            return
        row = state["row"]
        blds = label_buildings(source_paths(xbd_root, row)["label"])
        bcm, nc, nf, sk = score_image(state["pred"], state["valid"], blds,
                                      detect_share=detect_share, min_px=min_px)
        img_ids.append(state["base"])
        bcms.append(bcm)
        comps.append(nc)
        falses.append(nf)
        skipped.append(sk)

    model.eval()
    with torch.no_grad():
        for b in dl:
            pre = degrade_batch(b["pre"].to(device), degrade)
            post = degrade_batch(b["post"].to(device), degrade)
            with torch.autocast("cuda", enabled=amp):
                pred = model(pre, post).argmax(1).cpu().numpy().astype(np.uint8)
            masks, valids = b["mask"].numpy(), b["valid"].numpy()
            for i, tid in enumerate(b["tile_id"]):
                crop_ids.append(tid)
                crop_cms.append(M.confusion(pred[i], masks[i], valids[i]))
                row = rows[tid]
                base = tid.rsplit("_p", 1)[0]
                if base != state["base"]:
                    flush()
                    state.update(base=base, row=row,
                                 pred=np.zeros((1024, 1024), np.uint8),
                                 valid=np.zeros((1024, 1024), bool))
                x0, y0 = PART_ORIGIN[row["part"]]
                state["pred"][y0:y0 + 512, x0:x0 + 512] = pred[i]
                state["valid"][y0:y0 + 512, x0:x0 + 512] = valids[i]
    flush()
    res = {"crop_ids": np.array(crop_ids), "crop_cms": np.stack(crop_cms),
           "img_ids": np.array(img_ids), "bcms": np.stack(bcms), "comps": np.array(comps),
           "falses": np.array(falses), "skipped": np.array(skipped)}
    if out_path:
        np.savez_compressed(out_path, **res)
    return res
