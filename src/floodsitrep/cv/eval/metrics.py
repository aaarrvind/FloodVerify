"""Pixel-level segmentation metrics and the xView2-style score. Owner: member 3.

Everything here is PIXEL-level. Say so next to every number in the report.

Classes (mask values): 0 background, 1 no damage, 2 minor, 3 major, 4 destroyed.
Pixels where ``valid`` is False (black no-data) are never counted.

xView2 score (https://xview2.org/challenge; formula as reported by e.g. RescueNet,
arXiv 2004.07312, eq. 3):

    score = 0.3 * F1_loc + 0.7 * HM(F1_no, F1_minor, F1_major, F1_destroyed)

- F1_loc: binary building (mask > 0) vs background, over all valid pixels.
- Damage F1 per class: computed only over pixels that are building in the GROUND TRUTH,
  as in the official scoring code. A ground-truth building pixel predicted as background
  counts as a miss for its true class. [VERIFY against the official xView2 scoring script
  before final numbers; if it differs, change ``damage_f1`` only.]
- HM: harmonic mean; it is 0 if any damage-class F1 is 0.

The official challenge scores whole 1024 px images over the official holdout set. Ours is
computed on our 512 px crops and our event split, so call it "a pixel-level approximation
of the xView2 score", never "the xView2 score".

Accumulate one 5x5 confusion matrix per tile (``tile_confusions``) so that bootstrap
intervals resample tiles, not pixels.

Needs numpy only.
"""
import numpy as np

N_CLASSES = 5
CLASS_NAMES = ("background", "no_damage", "minor", "major", "destroyed")
DAMAGE = (1, 2, 3, 4)
EPS = 1e-12


def confusion(pred, target, valid=None, n=N_CLASSES):
    """n x n confusion matrix, rows = ground truth, columns = prediction.

    pred, target: integer arrays of the same shape. valid: optional bool array; pixels
    where it is False, or where target is outside 0..n-1 (e.g. 255), are skipped.
    """
    pred = np.asarray(pred).ravel().astype(np.int64)
    target = np.asarray(target).ravel().astype(np.int64)
    keep = (target >= 0) & (target < n) & (pred >= 0) & (pred < n)
    if valid is not None:
        keep &= np.asarray(valid).ravel().astype(bool)
    idx = target[keep] * n + pred[keep]
    return np.bincount(idx, minlength=n * n).reshape(n, n)


def _f1(tp, fp, fn):
    d = 2 * tp + fp + fn
    return float(2 * tp / d) if d > 0 else float("nan")


def _iou(tp, fp, fn):
    d = tp + fp + fn
    return float(tp / d) if d > 0 else float("nan")


def harmonic_mean(values):
    """Harmonic mean; 0 if any value is 0; NaN values (class absent everywhere) are dropped."""
    v = [x for x in values if not np.isnan(x)]
    if not v:
        return float("nan")
    if min(v) <= 0:
        return 0.0
    return float(len(v) / sum(1.0 / x for x in v))


def per_class(cm):
    """Per-class IoU and F1 over all valid pixels (5-class semantic segmentation)."""
    cm = np.asarray(cm, dtype=np.float64)
    out = {}
    for k, name in enumerate(CLASS_NAMES):
        tp = cm[k, k]
        fp = cm[:, k].sum() - tp
        fn = cm[k, :].sum() - tp
        out[name] = {"iou": _iou(tp, fp, fn), "f1": _f1(tp, fp, fn),
                     "support_px": int(cm[k, :].sum())}
    return out


def localisation_f1(cm):
    """Binary F1: building (classes 1-4) vs background (class 0)."""
    cm = np.asarray(cm, dtype=np.float64)
    tp = cm[1:, 1:].sum()
    fp = cm[0, 1:].sum()
    fn = cm[1:, 0].sum()
    return _f1(tp, fp, fn)


def damage_f1(cm):
    """Per-damage-class F1 over ground-truth building pixels only (xView2 style).

    Uses rows 1-4 of the confusion matrix. For class c: TP = cm[c, c];
    FN = rest of row c (including pixels predicted as background);
    FP = column c within rows 1-4 (background rows are excluded).
    """
    cm = np.asarray(cm, dtype=np.float64)
    sub = cm[1:, :]  # GT building rows, all predicted columns
    out = {}
    for c in DAMAGE:
        tp = cm[c, c]
        fn = cm[c, :].sum() - tp
        fp = sub[:, c].sum() - tp
        out[CLASS_NAMES[c]] = _f1(tp, fp, fn)
    return out


def summarise(cm):
    """All headline numbers from one confusion matrix (summed over tiles)."""
    cm = np.asarray(cm)
    pc = per_class(cm)
    dmg = damage_f1(cm)
    loc = localisation_f1(cm)
    hm = harmonic_mean(list(dmg.values()))
    ious = [v["iou"] for v in pc.values() if not np.isnan(v["iou"])]
    score = float("nan") if np.isnan(loc) or np.isnan(hm) else 0.3 * loc + 0.7 * hm
    return {
        "level": "pixel",
        "per_class": pc,
        "miou": float(np.mean(ious)) if ious else float("nan"),
        "loc_f1": loc,
        "damage_f1": dmg,
        "damage_f1_hm": hm,
        "xview2_pixel_score": score,
        "pixels": int(cm.sum()),
    }


def flat(summary):
    """One-level dict of the headline numbers (handy for logs and tables)."""
    s = summary
    out = {"loc_f1": s["loc_f1"], "damage_f1_hm": s["damage_f1_hm"],
           "xview2_pixel_score": s["xview2_pixel_score"], "miou": s["miou"]}
    for k, v in s["damage_f1"].items():
        out[f"dmg_f1_{k}"] = v
    for k, v in s["per_class"].items():
        out[f"f1_{k}"] = v["f1"]
        out[f"iou_{k}"] = v["iou"]
    return out


def bootstrap(tile_cms, n_boot=1000, seed=0, alpha=0.05):
    """Percentile bootstrap over TILES. tile_cms: array [n_tiles, 5, 5].

    Returns {metric: {"value", "lo", "hi"}} for the keys of ``flat``. The point value is
    computed on all tiles; the interval is the (alpha/2, 1-alpha/2) percentile of the
    resampled values (NaN resamples dropped).
    """
    cms = np.asarray(tile_cms, dtype=np.int64)
    if cms.ndim != 3 or len(cms) == 0:
        raise ValueError("tile_cms must be [n_tiles, n, n] with n_tiles > 0")
    point = flat(summarise(cms.sum(0)))
    rng = np.random.default_rng(seed)
    samples = {k: [] for k in point}
    n = len(cms)
    for _ in range(n_boot):
        idx = rng.integers(0, n, n)
        s = flat(summarise(cms[idx].sum(0)))
        for k, v in s.items():
            samples[k].append(v)
    out = {}
    for k, v in point.items():
        arr = np.array([x for x in samples[k] if not np.isnan(x)])
        lo, hi = (np.quantile(arr, [alpha / 2, 1 - alpha / 2]) if len(arr)
                  else (float("nan"), float("nan")))
        out[k] = {"value": v, "lo": float(lo), "hi": float(hi)}
    return out
