import pytest

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
pytest.importorskip("PIL")
B = pytest.importorskip("floodsitrep.cv.eval.building_level")


def _square(x0, y0, x1, y1):
    return [(x0, y0), (x1, y0), (x1, y1), (x0, y1)]


def _scene():
    # three separated buildings: no damage, major, destroyed
    blds = [(1, _square(10, 10, 30, 30)), (3, _square(50, 10, 70, 30)),
            (4, _square(10, 50, 30, 70))]
    truth = np.zeros((100, 100), np.uint8)
    for k, pts in blds:
        (x0, y0), (x1, y1) = pts[0], pts[2]
        truth[y0:y1, x0:x1] = k
    return blds, truth


def test_perfect_prediction_scores_one():
    blds, truth = _scene()
    bcm, nc, nf, sk = B.score_image(truth, np.ones_like(truth, bool), blds, min_px=1)
    s = B.summarise_buildings(bcm, nc, nf)
    assert s["buildings"] == 3 and nc == 3 and nf == 0 and sk == 0
    assert s["loc_f1"] == pytest.approx(1.0)
    assert s["xview2_building_score"] == pytest.approx(1.0)


def test_missed_wrong_class_and_false_building():
    blds, truth = _scene()
    pred = truth.copy()
    pred[50:70, 10:30] = 0          # destroyed building missed entirely
    pred[10:30, 50:70] = 2          # major predicted as minor
    pred[80:95, 80:95] = 1          # a building that does not exist
    bcm, nc, nf, _ = B.score_image(pred, np.ones_like(truth, bool), blds, min_px=1)
    assert bcm[3, 0] == 1           # destroyed -> missed
    assert bcm[2, 2] == 1           # major -> minor
    assert bcm[0, 1] == 1           # no damage correct
    assert (nc, nf) == (3, 1)
    s = B.summarise_buildings(bcm, nc, nf)
    assert s["loc_recall"] == pytest.approx(2 / 3)
    assert s["loc_precision"] == pytest.approx(2 / 3)
    assert s["dmg_f1_destroyed"] == 0.0


def test_majority_vote_and_detection_threshold():
    blds = [(3, _square(0, 0, 10, 10))]
    pred = np.zeros((20, 20), np.uint8)
    pred[0:10, 0:4] = 4             # 40 px destroyed
    pred[0:10, 4:7] = 3             # 30 px major  -> 70% building, majority destroyed
    bcm, *_ = B.score_image(pred, np.ones((20, 20), bool), blds, min_px=1)
    assert bcm[2, 4] == 1
    pred[0:10, 2:7] = 0             # only 20% building -> missed
    bcm, *_ = B.score_image(pred, np.ones((20, 20), bool), blds, min_px=1)
    assert bcm[2, 0] == 1


def test_building_outside_valid_area_is_skipped():
    blds, truth = _scene()
    valid = np.ones_like(truth, bool)
    valid[0:40, 0:40] = False       # the no-damage building has no imagery
    _, _, _, sk = B.score_image(truth, valid, blds, min_px=1)
    assert sk == 1


def test_degrade_keeps_shape_and_blurs():
    torch = pytest.importorskip("torch")
    x = torch.randn(2, 3, 64, 64)
    y = B.degrade_batch(x, 6)
    assert y.shape == x.shape and y.std() < x.std()
    assert B.degrade_batch(x, 1) is x
