"""Programmatic claim checker. Owner: member 4."""

from __future__ import annotations

from dataclasses import asdict, replace

from floodsitrep.common.interfaces import CLASSES
from floodsitrep.nlp.claims import extract_claims, extract_tile_claims
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


UNREACHABLE_MODES = ("abstain", "nearest")


def nearest_reachable(quantity: str, total: int) -> tuple[str, ...]:
    """Quantity words that a claim with `quantity` may be compared against for `total` buildings.

    A reachable word is compared as written. An unreachable word is mapped to the closest reachable
    word on the none < few < some < many < most scale (several words when they tie). "none" is never
    a target for another word, because few/some/many all mean at least one building.
    """
    words = [w for w in QUANTITIES if reachable(w, total)]
    if quantity in words:
        return (quantity,)
    targets = [w for w in words if w != "none"] or words
    pos = QUANTITIES.index
    gap = min(abs(pos(w) - pos(quantity)) for w in targets)
    return tuple(w for w in targets if abs(pos(w) - pos(quantity)) == gap)


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


def check_claim(claim: Claim, evidence: dict, unreachable: str = "abstain") -> str:
    """Return 'supported', 'contradicted' or 'not_verifiable' for one claim.

    A claim whose quantity word cannot occur with this many buildings (see `reachable`) is
    not_verifiable when unreachable="abstain" (the default). With unreachable="nearest" it is
    compared against the closest reachable word instead (see `nearest_reachable`), which answers more
    claims in small cells but can accept a claim that the evidence only approximately supports.
    """
    if unreachable not in UNREACHABLE_MODES:
        raise ValueError(f"unreachable must be one of {UNREACHABLE_MODES}, got {unreachable!r}")
    cells = evidence["cells"]
    if claim.cell:
        cells = [c for c in cells if c["cell"] == claim.cell]
    usable = [c for c in cells if c.get("valid_fraction", 1.0) >= MIN_VALID]
    if not usable or claim.quantity not in QUANTITIES:
        return "not_verifiable"
    frac = class_fraction(usable, claim.damage_class)
    if frac is None:
        return "not_verifiable"
    total = sum(c["buildings"] for c in usable)
    if reachable(claim.quantity, total):
        accepted = (claim.quantity,)
    elif unreachable == "nearest":
        accepted = nearest_reachable(claim.quantity, total)
    else:
        return "not_verifiable"
    return "supported" if bin_of(frac) in accepted else "contradicted"


def verify_claims(claims: list[Claim], evidence: dict, unreachable: str = "abstain") -> list[Claim]:
    """Attach a verdict to every claim."""
    return [replace(c, verdict=check_claim(c, evidence, unreachable)) for c in claims]


def verify(
    report_text: str, evidence: dict, tile_level: bool = False, unreachable: str = "abstain"
) -> list[dict]:
    """Contract entry point: report text + evidence -> list of claim dicts with a verdict.

    With tile_level=True, whole-image claims (cell=None) are checked too. They are off by default because
    callers such as the demo expect a cell on every claim. `unreachable` is passed to check_claim."""
    claims = extract_claims(report_text)
    if tile_level:
        claims += extract_tile_claims(report_text)
    return [asdict(c) for c in verify_claims(claims, evidence, unreachable)]
