import json
from pathlib import Path

import jsonschema

from floodsitrep.common.interfaces import quantifier

SCHEMA = Path(__file__).parents[1] / "evidence/schema/evidence.schema.json"


def test_quantifier_bins():
    assert quantifier(0.05) == "few"
    assert quantifier(0.2) == "some"
    assert quantifier(0.45) == "many"
    assert quantifier(0.8) == "most"


def test_samples_match_schema():
    schema = json.loads(SCHEMA.read_text())
    for f in (SCHEMA.parents[1] / "samples").glob("*.json"):
        jsonschema.validate(json.loads(f.read_text()), schema)
