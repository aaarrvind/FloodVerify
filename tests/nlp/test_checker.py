import json
from pathlib import Path

from floodsitrep.nlp.checker import bin_of, check_claim, class_fraction
from floodsitrep.nlp.schema import Claim

SAMPLES = Path(__file__).parents[2] / "evidence" / "samples"


def _cell(name, n, none=0.0, minor=0.0, major=0.0, destroyed=0.0, valid=1.0):
    return {
        "cell": name,
        "buildings": n,
        "confidence": 1.0,
        "valid_fraction": valid,
        "class_fraction": {"none": none, "minor": minor, "major": major, "destroyed": destroyed},
    }


def _evidence(*cells):
    return {"tile_id": "t", "source": "labels", "gsd_m": 0.5, "grid": "3x3", "cells": list(cells)}


def _claim(cell, quantity, damage_class):
    return Claim(text="x", cell=cell, quantity=quantity, damage_class=damage_class)


def test_bins():
    assert [bin_of(f) for f in (0, 0.05, 0.2, 0.45, 0.8)] == ["none", "few", "some", "many", "most"]


def test_supported_and_contradicted():
    ev = _evidence(_cell("r0c0", 10, none=0.2, destroyed=0.8))
    assert check_claim(_claim("r0c0", "most", "destroyed"), ev) == "supported"
    assert check_claim(_claim("r0c0", "few", "destroyed"), ev) == "contradicted"


def test_no_usable_imagery_is_not_verifiable():
    ev = _evidence(_cell("r0c0", 10, none=0.2, destroyed=0.8, valid=0.3))
    assert check_claim(_claim("r0c0", "most", "destroyed"), ev) == "not_verifiable"


def test_empty_cell_and_unknown_cell_are_not_verifiable():
    ev = _evidence(_cell("r0c0", 0))
    assert check_claim(_claim("r0c0", "none", "destroyed"), ev) == "not_verifiable"
    assert check_claim(_claim("r2c2", "few", "minor"), ev) == "not_verifiable"


def test_tile_level_claim_skips_invalid_cells():
    ev = _evidence(_cell("r0c0", 10, destroyed=1.0, valid=0.1), _cell("r0c1", 10, none=1.0))
    assert check_claim(_claim(None, "none", "destroyed"), ev) == "supported"


def test_samples_agree_with_themselves():
    for path in SAMPLES.glob("*.json"):
        ev = json.loads(path.read_text())
        for c in ev["cells"]:
            for cls in ("none", "minor", "major", "destroyed", "damaged"):
                frac = class_fraction([c], cls)
                if frac is None or c.get("valid_fraction", 1.0) < 0.5:
                    continue
                claim = _claim(c["cell"], bin_of(frac), cls)
                assert check_claim(claim, ev) == "supported"
