"""Hand-checkable toy cases for cv/eval/metrics.py (pixel level)."""
import math

import pytest

np = pytest.importorskip("numpy")
M = pytest.importorskip("floodsitrep.cv.eval.metrics")

# 8 pixels. Rows of the confusion matrix (truth -> prediction):
#   bg  -> bg, no      no -> no, bg      minor -> minor
#   major -> major     destroyed -> destroyed, major
TARGET = np.array([0, 0, 1, 1, 2, 3, 4, 4])
PRED = np.array([0, 1, 1, 0, 2, 3, 4, 3])


def test_confusion_counts_and_ignore():
    cm = M.confusion(PRED, TARGET)
    assert cm.sum() == 8
    assert cm[0, 1] == 1 and cm[1, 0] == 1 and cm[4, 3] == 1
    t = TARGET.copy()
    t[0] = 255  # ignore label
    assert M.confusion(PRED, t).sum() == 7
    valid = np.ones(8, bool)
    valid[1] = False
    assert M.confusion(PRED, TARGET, valid)[0, 1] == 0


def test_localisation_f1():
    # building TP = 5, FP = 1 (bg->no), FN = 1 (no->bg): F1 = 10/12
    assert M.localisation_f1(M.confusion(PRED, TARGET)) == pytest.approx(10 / 12)


def test_damage_f1_on_gt_buildings_only():
    d = M.damage_f1(M.confusion(PRED, TARGET))
    # no_damage: TP 1, FN 1 (predicted bg), FP 0 (the bg->no pixel is NOT counted) -> 2/3
    assert d["no_damage"] == pytest.approx(2 / 3)
    assert d["minor"] == pytest.approx(1.0)
    assert d["major"] == pytest.approx(2 / 3)       # FP from destroyed->major
    assert d["destroyed"] == pytest.approx(2 / 3)   # FN to major


def test_xview2_pixel_score():
    s = M.summarise(M.confusion(PRED, TARGET))
    hm = 4 / (1.5 + 1 + 1.5 + 1.5)
    assert s["damage_f1_hm"] == pytest.approx(hm)
    assert s["xview2_pixel_score"] == pytest.approx(0.3 * 10 / 12 + 0.7 * hm)
    # 5-class F1 for no_damage DOES count the bg->no pixel as FP: 2*1/(2+1+1) = 0.5
    assert s["per_class"]["no_damage"]["f1"] == pytest.approx(0.5)
    assert s["level"] == "pixel"


def test_harmonic_mean_zero_and_nan():
    assert M.harmonic_mean([0.5, 0.0]) == 0.0
    assert M.harmonic_mean([0.5, float("nan")]) == pytest.approx(0.5)
    assert math.isnan(M.harmonic_mean([float("nan")]))


def test_perfect_prediction():
    s = M.summarise(M.confusion(TARGET, TARGET))
    assert s["xview2_pixel_score"] == pytest.approx(1.0)
    assert s["miou"] == pytest.approx(1.0)


def test_bootstrap_over_tiles():
    cm = M.confusion(PRED, TARGET)
    tiles = np.stack([cm, M.confusion(TARGET, TARGET), cm])
    b = M.bootstrap(tiles, n_boot=200, seed=0)
    v = b["xview2_pixel_score"]
    assert v["lo"] <= v["value"] <= v["hi"]
    assert v["value"] == pytest.approx(M.summarise(tiles.sum(0))["xview2_pixel_score"])
