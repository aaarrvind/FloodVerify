"""Programmatic claim checker. Owner: member 4."""

from __future__ import annotations

from dataclasses import replace

from floodsitrep.common.interfaces import CLASSES
from floodsitrep.nlp.schema import Claim

MIN_VALID = 0.5  # cells with less real imagery than this cannot be verified
QUANTITIES = ("none", "few", "some", "many", "most")


def bin_of(fraction: float) -> str:
    """Map a fraction of buildings to a quantity word."""
    if fraction == 0:
        return "none"
    if fraction < 0.10:
        return "few"
    if fraction < 0.30:
        return "some"
    if fraction < 0.60:
        return "many"
    return "most"


def class_fraction(cells: list[dict], damage_class: str) -> float | None:
    """Share of buildings in `cells` with the class. 'damaged' = minor+major+destroyed.

    Building counts are whole numbers, so each cell's count per class is rounded to an
    integer first. That removes the error from the 4-decimal fractions stored in the evidence.
    Returns None when there are no buildings or the class is unknown.
    """
    if damage_class == "damaged":
        keys = CLASSES[1:]
    elif damage_class in CLASSES:
        keys = (damage_class,)
    else:
        return None
    total = sum(c["buildings"] for c in cells)
    if total == 0:
        return None
    hits = sum(round(c["buildings"] * c["class_fraction"][k]) for c in cells for k in keys)
    return hits / total


def check_claim(claim: Claim, evidence: dict) -> str:
    """Return 'supported', 'contradicted' or 'not_verifiable' for one claim."""
    cells = evidence["cells"]
    if claim.cell:
        cells = [c for c in cells if c["cell"] == claim.cell]
    usable = [c for c in cells if c.get("valid_fraction", 1.0) >= MIN_VALID]
    if not usable or claim.quantity not in QUANTITIES:
        return "not_verifiable"
    frac = class_fraction(usable, claim.damage_class)
    if frac is None:
        return "not_verifiable"
    return "supported" if bin_of(frac) == claim.quantity else "contradicted"


def verify_claims(claims: list[Claim], evidence: dict) -> list[Claim]:
    """Attach a verdict to every claim."""
    return [replace(c, verdict=check_claim(c, evidence)) for c in claims]
