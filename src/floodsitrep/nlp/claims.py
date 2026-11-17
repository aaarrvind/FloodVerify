"""Caption lines -> atomic typed claims. Owner: member 4.

A report is text with one line per grid cell, for example:
    r1c1: few of the buildings show minor damage.
    r0c2: no buildings.
Lines that make no damage statement ("no buildings", "not visible") produce no claim.
"""

import re

from floodsitrep.nlp.schema import Claim

CELL_LINE = re.compile(r"^\s*(r[0-2]c[0-2])\s*:\s*(.+?)\s*$", re.IGNORECASE)
QUANTITY_CLASS = re.compile(r"\b(none|few|some|many|most)\b[^.,;]*?\b(minor|major|destroyed)\b")


def parse_line(line: str) -> tuple[str, list[tuple[str, str]]]:
    """One caption line (without the cell id) -> (kind, [(quantity, damage_class), ...]).

    kind is 'no_buildings', 'not_visible' or 'claims'. 'no damage' becomes (none, damaged).
    A line with two clauses ("few ... minor damage and some show major damage") gives two pairs.
    """
    low = line.lower().strip()
    if low.startswith("no buildings"):
        return "no_buildings", []
    if low.startswith("not visible"):
        return "not_visible", []
    if low.startswith("no damage"):
        return "claims", [("none", "damaged")]
    if low.startswith("none") and "damage" in low and not QUANTITY_CLASS.search(low):
        return "claims", [("none", "damaged")]
    pairs = []
    for part in low.split(" and "):
        m = QUANTITY_CLASS.search(part)
        if m:
            pairs.append(m.groups())
    return "claims", pairs


def extract_claims(report_text: str) -> list[Claim]:
    """Report text (one 'r0c0: ...' line per cell) -> list of Claim. Other lines are ignored."""
    claims = []
    for raw in report_text.splitlines():
        m = CELL_LINE.match(raw)
        if not m:
            continue
        cell, line = m.group(1).lower(), m.group(2)
        _, pairs = parse_line(line)
        claims += [Claim(text=line, cell=cell, quantity=q, damage_class=c) for q, c in pairs]
    return claims


def report_from_cells(cells: dict[str, str]) -> str:
    """{cell_id: caption line} -> report text, one line per cell in grid order."""
    return "\n".join(f"{cid}: {cells[cid]}" for cid in sorted(cells))
