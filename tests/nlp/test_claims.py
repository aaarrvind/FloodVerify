import json
from pathlib import Path

import pytest

from floodsitrep.nlp.checker import bin_of, class_fraction, verify
from floodsitrep.nlp.claims import extract_claims, parse_line, report_from_cells
from floodsitrep.pipeline.run import run_tile


def test_parse_line_kinds():
    assert parse_line("no buildings.") == ("no_buildings", [])
    assert parse_line("Not visible") == ("not_visible", [])
    assert parse_line("few of the buildings show minor damage.") == ("claims", [("few", "minor")])
    assert parse_line("many of the buildings show major damage.") == ("claims", [("many", "major")])


def test_parse_line_no_damage_forms():
    assert parse_line("no damage.") == ("claims", [("none", "damaged")])
    assert parse_line("none of the buildings show any damage.") == ("claims", [("none", "damaged")])
    assert parse_line("none of the buildings show major damage.") == ("claims", [("none", "major")])


def test_parse_line_two_clauses():
    line = "some of the buildings show major damage and few show destroyed damage."
    assert parse_line(line) == ("claims", [("some", "major"), ("few", "destroyed")])


def test_parse_line_unparseable_gives_no_claim():
    assert parse_line("the area looks flooded") == ("claims", [])


def test_extract_claims_reads_cell_lines_and_skips_the_rest():
    text = (
        "r0c0: no buildings.\n"
        "r0c1: few of the buildings show minor damage.\n"
        "R1C2: no damage.\n"
        "an unrelated sentence\n"
        "r3c3: most of the buildings show destroyed damage.\n"  # not a grid cell
        "r2c2: not visible."
    )
    claims = extract_claims(text)
    assert [(c.cell, c.quantity, c.damage_class) for c in claims] == [
        ("r0c1", "few", "minor"),
        ("r1c2", "none", "damaged"),
    ]
    assert all(c.verdict == "not_verifiable" for c in claims)  # extraction alone never verifies


def test_report_from_cells_roundtrip():
    cells = {"r0c1": "no damage.", "r0c0": "no buildings."}
    assert report_from_cells(cells) == "r0c0: no buildings.\nr0c1: no damage."
    assert [c.cell for c in extract_claims(report_from_cells(cells))] == ["r0c1"]


def _perfect_report(evidence):
    """A report that copies the evidence: right bins, right classes."""
    lines = {}
    for c in evidence["cells"]:
        if c["buildings"] == 0:
            lines[c["cell"]] = "no buildings."
            continue
        parts = []
        for k in ("minor", "major", "destroyed"):
            frac = class_fraction([c], k)
            if frac:
                parts.append(
                    f"{bin_of(frac)} of the buildings show {k} damage"
                    if not parts
                    else f"{bin_of(frac)} show {k} damage"
                )
        lines[c["cell"]] = (" and ".join(parts) + ".") if parts else "no damage."
    return report_from_cells(lines)


def _samples():
    files = sorted(Path("evidence/samples").glob("*.json"))
    if not files:
        pytest.skip("no sample evidence merged yet")
    return files


def test_verify_supports_a_report_that_copies_the_evidence():
    for f in _samples():
        evidence = json.loads(f.read_text())
        out = verify(_perfect_report(evidence), evidence)
        assert out, f.name
        assert {c["verdict"] for c in out} <= {"supported", "not_verifiable"}, f.name
        usable = {c["cell"] for c in evidence["cells"] if c.get("valid_fraction", 1.0) >= 0.5}
        assert all(c["verdict"] == "supported" for c in out if c["cell"] in usable), f.name


def test_verify_contradicts_a_wrong_claim():
    for f in _samples():
        evidence = json.loads(f.read_text())
        cell = next(
            (
                c
                for c in evidence["cells"]
                if c["buildings"] > 0
                and class_fraction([c], "damaged") == 0
                and c.get("valid_fraction", 1.0) >= 0.5
            ),
            None,
        )
        if cell is None:
            continue
        out = verify(f"{cell['cell']}: most of the buildings show destroyed damage.", evidence)
        assert [c["verdict"] for c in out] == ["contradicted"]
        return
    pytest.skip("no sample cell with undamaged buildings")


def test_verify_returns_dicts_with_contract_fields():
    evidence = json.loads(_samples()[0].read_text())
    out = verify("r0c0: no damage.", evidence)
    assert set(out[0]) == {"text", "cell", "quantity", "damage_class", "verdict"}


def test_pipeline_now_attaches_claims():
    f = _samples()[0]
    evidence = json.loads(f.read_text())
    result = run_tile(f.stem, report_text=_perfect_report(evidence))
    assert result["claims"], "run_tile should have called nlp.checker.verify"
