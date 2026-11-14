from pathlib import Path

import pytest

from floodsitrep.pipeline.run import run_tile


def test_run_tile_on_sample():
    samples = list(Path("evidence/samples").glob("*.json"))
    if not samples:
        pytest.skip("no sample evidence merged yet")
    out = run_tile(samples[0].stem)
    assert out["tile_id"] == samples[0].stem
    assert out["evidence"]["grid"] == "3x3"
