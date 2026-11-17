"""Predicted evidence uses the same code path as label evidence and matches the schema."""
import json
from pathlib import Path

import pytest

from floodsitrep.data import labels_to_evidence as L

np = pytest.importorskip("numpy")
pytest.importorskip("scipy")
PIL = pytest.importorskip("PIL")
X = pytest.importorskip("floodsitrep.cv.export_evidence")
jsonschema = pytest.importorskip("jsonschema")

SCHEMA = json.loads((Path(__file__).parents[2] / "evidence/schema/evidence.schema.json")
                    .read_text())
SUBTYPE = {1: "no-damage", 2: "minor-damage", 3: "major-damage", 4: "destroyed"}

# separated rectangles (x0, y0, x1, y1, mask class), some on cell borders
RECTS = [(10, 10, 40, 30, 1), (100, 20, 130, 60, 2), (160, 160, 200, 190, 4),
         (300, 40, 340, 80, 3), (400, 400, 450, 430, 1), (165, 300, 180, 320, 2),
         (470, 100, 500, 140, 4), (250, 250, 262, 262, 3)]


def _scene():
    mask = np.zeros((512, 512), np.uint8)
    feats = []
    for x0, y0, x1, y1, k in RECTS:
        mask[y0:y1, x0:x1] = k
        feats.append({"properties": {"subtype": SUBTYPE[k]},
                      "wkt": f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))"})
    return mask, {"features": {"xy": feats}}


def test_gt_mask_reproduces_label_evidence_when_buildings_do_not_touch():
    mask, label = _scene()
    valid = np.ones_like(mask, bool)
    ref = L.evidence_from_label(label, "t", 0, 0, valid=X.valid_fractions_from_array(valid))
    got = X.evidence_from_prediction("t", mask, valid, None, min_px=0, mask_invalid=False)
    for a, b in zip(ref["cells"], got["cells"]):
        assert a["cell"] == b["cell"]
        assert a["buildings"] == b["buildings"]
        assert a["class_fraction"] == b["class_fraction"]
        assert a["valid_fraction"] == b["valid_fraction"]
    assert got["source"] == "predicted"


def test_touching_buildings_merge_and_undercount():
    mask = np.zeros((512, 512), np.uint8)
    mask[10:30, 10:30] = 1
    mask[10:30, 30:50] = 4  # shares an edge with the first
    assert len(X.buildings_from_mask(mask)) == 1


def test_majority_class_and_tie_goes_severe():
    mask = np.zeros((64, 64), np.uint8)
    mask[0:10, 0:10] = 2
    mask[0:10, 10:20] = 3  # equal pixel counts -> major (more severe)
    (_, _, cls), = X.buildings_from_mask(mask)
    assert cls == "major"


def test_valid_fraction_matches_member1_function(tmp_path):
    from PIL import Image

    rng = np.random.default_rng(0)
    img = rng.integers(1, 255, (1024, 1024, 3), dtype=np.uint8)
    img[:300, :200] = 0
    Image.fromarray(img).save(tmp_path / "pre.png")
    Image.fromarray(img).save(tmp_path / "post.png")
    for x0, y0 in ((0, 0), (512, 512)):
        ref = L.valid_fractions(tmp_path / "pre.png", tmp_path / "post.png", x0, y0)
        crop = img[y0:y0 + 512, x0:x0 + 512].sum(2) > 0
        assert X.valid_fractions_from_array(crop) == ref


def test_predicted_evidence_schema_and_no_buildings_in_nodata():
    mask, _ = _scene()
    valid = np.ones_like(mask, bool)
    valid[:171, :171] = False  # cell r0c0 has no imagery
    probs = np.full((5, 512, 512), 0.05)
    for k in range(5):
        probs[k][mask == k] = 0.8
    ev = X.evidence_from_prediction("t", mask, valid, probs)
    jsonschema.validate(ev, SCHEMA)
    c0 = ev["cells"][0]
    assert c0["valid_fraction"] < 0.5 and c0["buildings"] == 0 and c0["confidence"] == 0.0
    assert all(0 <= c["confidence"] <= 1 for c in ev["cells"])
    assert ev["cells"][4]["confidence"] == pytest.approx(0.8)


def test_min_px_drops_speckle():
    mask = np.zeros((64, 64), np.uint8)
    mask[5, 5] = 4
    mask[20:30, 20:30] = 1
    assert len(X.buildings_from_mask(mask, min_px=10)) == 1


def test_building_cut_by_crop_edge_counted_once():
    # one building spanning parts 1 (top-left) and 2 (top-right); centroid in part 2
    full = np.zeros((1024, 1024), np.uint8)
    full[100:140, 500:560] = 3
    crops = {"1": full[:512, :512], "2": full[:512, 512:]}
    per = X.stitched_buildings(crops)
    assert len(per["1"]) == 0 and len(per["2"]) == 1
    x, y, cls = per["2"][0]
    assert cls == "major" and x == pytest.approx(530 - 512) and y == pytest.approx(120)
    # without stitching both crops would report it
    assert len(X.buildings_from_mask(crops["1"])) == 1
