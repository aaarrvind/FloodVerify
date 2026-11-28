import pytest

from floodsitrep.nlp.checker import check_claim
from floodsitrep.nlp.rscc import rscc_claim, rscc_tile_quantity


@pytest.mark.parametrize(
    "caption, expected",
    [
        ("All buildings in the image are undamaged.", "none"),
        ("Both buildings remain structurally intact.", "none"),
        ("Every house shows no visible damage.", "none"),
        ("All buildings are destroyed (level 3).", "most"),
        ("The single building has major damage.", "most"),
        ("All structures show severe damage.", "most"),
        ("No buildings are present in the scene.", None),
        ("All buildings are unclassified.", None),
        ("All buildings are undamaged. Two of the buildings show minor damage.", None),  # partly damaged
        ("The area is flooded.", None),
        ("", None),
    ],
)
def test_rscc_tile_quantity(caption, expected):
    assert rscc_tile_quantity(caption) == expected


def test_rscc_claim_is_a_whole_image_claim():
    c = rscc_claim("All buildings are destroyed (level 3).")
    assert (c.cell, c.quantity, c.damage_class) == (None, "most", "damaged")
    assert rscc_claim("The area is flooded.") is None


def _evidence(*cells):
    return {"tile_id": "t", "source": "labels", "gsd_m": 0.5, "grid": "3x3", "cells": list(cells)}


def _cell(name, n, damaged, valid=1.0):
    d = damaged / n if n else 0.0
    return {"cell": name, "buildings": n, "confidence": 1.0, "valid_fraction": valid,
            "class_fraction": {"none": 1 - d, "minor": 0.0, "major": d, "destroyed": 0.0}}


def test_rscc_claim_checked_against_evidence():
    none, most = rscc_claim("All buildings are undamaged."), rscc_claim("All buildings are destroyed.")
    clean = _evidence(_cell("r0c0", 10, 0), _cell("r0c1", 5, 0))
    wrecked = _evidence(_cell("r0c0", 10, 9), _cell("r0c1", 5, 4))
    assert check_claim(none, clean) == "supported" and check_claim(most, clean) == "contradicted"
    assert check_claim(none, wrecked) == "contradicted" and check_claim(most, wrecked) == "supported"
    assert check_claim(none, _evidence(_cell("r0c0", 10, 0, valid=0.2))) == "not_verifiable"
    assert check_claim(none, _evidence(_cell("r0c0", 0, 0))) == "not_verifiable"
