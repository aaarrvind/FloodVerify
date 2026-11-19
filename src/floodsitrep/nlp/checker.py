"""Programmatic claim checker. Owner: member 4."""

from __future__ import annotations

from dataclasses import asdict, replace

from floodsitrep.common.interfaces import CLASSES
from floodsitrep.nlp.claims import extract_claims
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


def reachable(quantity: str, total: int) -> bool:
    """True when some building count out of `total` falls in the bin of `quantity`.

    Shares of whole buildings are k / total, so with few buildings some bins cannot occur:
    "few" (above 0 and below 10%) needs at least 11 buildings, "some" (10% to 30%) at least 4,
    "many" (30% to 60%) at least 2. A claim in an unreachable bin cannot be supported by any
    evidence, so the checker abstains instead of calling it contradicted.
    """
    return total > 0 and any(bin_of(k / total) == quantity for k in range(total + 1))


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
    """Return 'supported', 'contradicted' or 'not_verifiable' for one claim.

    A claim whose quantity word cannot occur with this many buildings is not_verifiable (see `reachable`).
    """
    cells = evidence["cells"]
    if claim.cell:
        cells = [c for c in cells if c["cell"] == claim.cell]
    usable = [c for c in cells if c.get("valid_fraction", 1.0) >= MIN_VALID]
    if not usable or claim.quantity not in QUANTITIES:
        return "not_verifiable"
    frac = class_fraction(usable, claim.damage_class)
    if frac is None:
        return "not_verifiable"
    if not reachable(claim.quantity, sum(c["buildings"] for c in usable)):
        return "not_verifiable"
    return "supported" if bin_of(frac) == claim.quantity else "contradicted"


def verify_claims(claims: list[Claim], evidence: dict) -> list[Claim]:
    """Attach a verdict to every claim."""
    return [replace(c, verdict=check_claim(c, evidence)) for c in claims]


def verify(report_text: str, evidence: dict) -> list[dict]:
    """Contract entry point: report text + evidence -> list of claim dicts with a verdict."""
    return [asdict(c) for c in verify_claims(extract_claims(report_text), evidence)]
