"""Whole-image claims from RSCC reference captions. Owner: member 4.

RSCC captions are machine-written free text about one image. A caption that clearly says all (or both, or the single)
buildings are undamaged gives the claim "none of the buildings are damaged"; one that says all buildings have major
damage, severe damage or are destroyed gives "most of the buildings are damaged". Captions with unknown or obscured
buildings, partly damaged buildings or no buildings give no claim. These captions are noisy and are never ground truth.
"""

from __future__ import annotations

import re

from floodsitrep.nlp.schema import Claim

BUILD = r"(?:buildings?|structures?|houses?|homes?)"
ALL = rf"(?:\b(?:all|both|every)\b[^.;]{{0,40}}\b{BUILD}\b|\b(?:single|solitary|sole|lone)\s+{BUILD}\b)"
UNDAM = (
    r"(?:\bundamaged\b|\bno damage\b|\bno visible damage\b|\bno structural damage\b|\blevel 0\b|\bunaffected\b|"
    r"\bundisturbed\b|\bremain(?:s|ed)?\s+(?:structurally\s+)?intact\b|\bretain(?:s|ed|ing)?\b[^.;]{0,30}\bintegrity\b)"
)
DAM = r"(?:\bmajor damage\b|\blevel 2\b|\bsevere(?:ly)? damage[d]?\b|\bdestroyed\b|\blevel 3\b)"
UNKNOWN = r"\b(?:unclassified|unknown|unassess\w*|obscured|unclear|undetermined|cannot be assessed|impossible|inconclusive)\b"
NO_BUILDINGS = rf"\bno {BUILD}\b[^.;]{{0,40}}\b(?:present|exist|visible)\b|\bno {BUILD} (?:are|were) present\b"
PARTIAL = (
    rf"\b(?:one|two|three|four|five|six|seven|eight|nine|ten|\d+)\s+(?:of\s+\w+\s+)?{BUILD}\b[^.;]{{0,60}}"
    r"\b(?:major|minor|destroyed|level [123])\b"
)


def rscc_tile_quantity(caption: str) -> str | None:
    """'none' or 'most' when the caption clearly says that all or none of the buildings are damaged, else None."""
    c = caption.lower()
    if re.search(NO_BUILDINGS, c) or re.search(UNKNOWN, c):
        return None
    sents = re.split(r"[.;]", c)
    none = any(re.search(ALL, s) and re.search(UNDAM, s) and not re.search(DAM, s) for s in sents)
    most = any(re.search(ALL, s) and re.search(DAM, s) and not re.search(UNDAM, s) for s in sents)
    if none == most:
        return None
    if none and re.search(PARTIAL, c):
        return None
    return "none" if none else "most"


def rscc_claim(caption: str) -> Claim | None:
    """The whole-image claim (cell=None, class 'damaged') for a caption, or None when the caption gives no clear claim.
    Check it with floodsitrep.nlp.checker.check_claim against the tile's evidence."""
    q = rscc_tile_quantity(caption)
    if q is None:
        return None
    text = "none of the buildings in the whole image are damaged" if q == "none" else "most of the buildings in the whole image are damaged"
    return Claim(text=text, cell=None, quantity=q, damage_class="damaged")
