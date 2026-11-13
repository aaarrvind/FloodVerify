"""Interface stubs. Code against these so tracks can work in parallel.

Owner: member 1. Changes need all five members tagged.
"""
import json
from pathlib import Path

CLASSES = ("none", "minor", "major", "destroyed")
BINS = ((0.10, "few"), (0.30, "some"), (0.60, "many"), (1.01, "most"))


def load_evidence(path):
    """Read one evidence JSON file (dict matching evidence.schema.json)."""
    return json.loads(Path(path).read_text())


def quantifier(fraction):
    """Fraction in [0, 1] -> few / some / many / most."""
    for upper, word in BINS:
        if fraction < upper:
            return word


# Contract signatures (implemented in the owner's package):
#   cv.export_evidence.export(checkpoint, tile_dir, out_dir) -> list[Path]      (member 3)
#   nlp.checker.verify(report_text, evidence) -> list[dict]                      (member 4)
#   advisory.generate.advise(verified_claims, lang) -> str                       (member 5)
#   pipeline.run.run_tile(tile_id) -> dict                                       (member 1)
