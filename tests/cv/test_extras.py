import json

import pytest

np = pytest.importorskip("numpy")
EX = pytest.importorskip("floodsitrep.cv.eval.extras")
M = pytest.importorskip("floodsitrep.cv.eval.metrics")


def _run(ids, cms):
    return {"tile_ids": np.array(ids), "cms": np.stack(cms)}


def test_paired_bootstrap_identical_runs_is_zero():
    t = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    p = np.array([0, 1, 1, 0, 2, 3, 4, 3])
    cms = [M.confusion(p, t), M.confusion(t, t), M.confusion(p, t)]
    r = EX.paired_bootstrap(_run(["a", "b", "c"], cms), _run(["a", "b", "c"], cms), n_boot=200)
    assert r["diff"] == 0 and r["lo"] == 0 and r["hi"] == 0


def test_paired_bootstrap_aligns_tile_order_and_detects_better_run():
    t = np.array([0, 0, 1, 1, 2, 3, 4, 4])
    good = M.confusion(t, t)
    bad = M.confusion(np.array([0, 1, 1, 0, 2, 3, 4, 3]), t)
    a = _run(["x", "y", "z"], [good, good, good])
    b = _run(["z", "y", "x"], [bad, bad, bad])  # different order on purpose
    r = EX.paired_bootstrap(a, b, n_boot=200)
    assert r["diff"] > 0 and r["lo"] > 0 and r["share_le_0"] == 0


def test_reliability_hand_example():
    # 4 pixels, confidence 0.8 each (logits ln4 vs 0), 2 correct -> ECE 0.3
    logits = np.array([[np.log(4), 0.0]] * 4)
    e, rows = EX.reliability(logits, np.array([0, 0, 1, 1]))
    assert e == pytest.approx(0.3)
    assert rows[0]["accuracy"] == pytest.approx(0.5)
    assert rows[0]["mean_confidence"] == pytest.approx(0.8)


def _ev(tid, cells):
    full = []
    for i, (b, d) in enumerate(cells):
        cf = {"none": 1 - d, "minor": 0.0, "major": 0.0, "destroyed": d} if b else \
             {"none": 0.0, "minor": 0.0, "major": 0.0, "destroyed": 0.0}
        full.append({"cell": f"r{i // 3}c{i % 3}", "buildings": b, "class_fraction": cf,
                     "confidence": 1.0, "valid_fraction": 1.0})
    return {"tile_id": tid, "cells": full}


def test_building_cell_agreement(tmp_path):
    lab, prd = tmp_path / "lab", tmp_path / "prd"
    lab.mkdir()
    prd.mkdir()
    L = [(2, 0.0), (1, 1.0), (0, 0.0)] + [(0, 0.0)] * 6   # 2 cells with buildings, 7 empty
    P = [(2, 0.0), (0, 0.0), (1, 0.0)] + [(0, 0.0)] * 6   # misses cell 2, fills cell 3
    (lab / "t.json").write_text(json.dumps(_ev("t", L)))
    (prd / "t.json").write_text(json.dumps(_ev("t", P)))
    r = EX.building_cell_agreement(prd, lab)
    assert r["label_cells_with_buildings"] == 2
    assert r["count_exact_match"] == 0.5
    assert r["cells_model_finds_no_building"] == 0.5
    assert r["label_empty_cells"] == 7
    assert r["label_empty_cells_model_fills"] == pytest.approx(1 / 7)
