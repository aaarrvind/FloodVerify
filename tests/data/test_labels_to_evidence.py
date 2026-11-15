import json
from pathlib import Path

import pytest

from floodsitrep.data.labels_to_evidence import (
    buildings_in_crop,
    centroid,
    evidence_from_label,
    grid_cells,
    parse_wkt,
)
from floodsitrep.data.splits import ROLE_BY_EVENT, role_of
from floodsitrep.data.tiling import read_manifest, source_paths, tile_name


def square(x, y, s=10):
    return f"POLYGON (({x} {y}, {x + s} {y}, {x + s} {y + s}, {x} {y + s}, {x} {y}))"


def label(*items):
    return {"features": {"xy": [{"properties": {"subtype": s}, "wkt": square(x, y)}
                                for (x, y, s) in items]}}


def test_parse_and_centroid():
    pts = parse_wkt(square(100, 200))
    assert pts[0] == (100.0, 200.0)
    cx, cy = centroid(pts[:-1])
    assert abs(cx - 105) < 1e-6 and abs(cy - 205) < 1e-6


def test_crop_offset_and_unclassified_skipped():
    lab = label((600, 100, "destroyed"), (20, 20, "no-damage"), (700, 700, "un-classified"))
    b = buildings_in_crop(lab, 512, 0)  # part 2: top-right
    assert len(b) == 1 and b[0][2] == "destroyed"
    assert abs(b[0][0] - (605 - 512)) < 1e-6


def test_each_building_counted_once_across_parts():
    lab = label((100, 100, "no-damage"), (600, 100, "minor-damage"), (100, 600, "major-damage"),
                (600, 600, "destroyed"), (508, 508, "no-damage"))  # last one straddles the middle
    total = sum(len(buildings_in_crop(lab, x0, y0))
                for x0, y0 in ((0, 0), (512, 0), (0, 512), (512, 512)))
    assert total == 5


def test_grid_cells_fractions():
    cells = grid_cells([(10, 10, "none"), (20, 20, "major"), (500, 500, "destroyed")])
    assert len(cells) == 9
    r0c0 = cells[0]
    assert r0c0["buildings"] == 2 and r0c0["class_fraction"]["none"] == 0.5
    assert cells[8]["class_fraction"]["destroyed"] == 1.0
    empty = cells[4]
    assert empty["buildings"] == 0 and sum(empty["class_fraction"].values()) == 0


def test_evidence_matches_schema():
    jsonschema = pytest.importorskip("jsonschema")
    schema = json.loads(Path("evidence/schema/evidence.schema.json").read_text())
    ev = evidence_from_label(label((100, 100, "no-damage")), "t_p1", 0, 0, valid=[1.0] * 9)
    jsonschema.validate(ev, schema)
    assert ev["source"] == "labels" and ev["grid"] == "3x3"


def test_split_roles():
    assert role_of("nepal-flooding") == "test"
    assert role_of("not-an-event") is None
    assert set(ROLE_BY_EVENT.values()) == {"train", "val", "test"}


def test_manifest_and_paths(tmp_path):
    m = tmp_path / "m.csv"
    m.write_text("event,tile_id,part,split,x0,y0,caption\n"
                 "hurricane-harvey,00000419,2,train,512,0,x\n")
    rows = read_manifest(m)
    assert rows[0]["x0"] == 512 and rows[0]["y0"] == 0
    assert tile_name(rows[0]) == "hurricane-harvey_00000419_p2"
    p = source_paths("/xbd", rows[0])
    assert p["label"].name == "hurricane-harvey_00000419_post_disaster.json"
    assert p["target"].parent.name == "targets"
