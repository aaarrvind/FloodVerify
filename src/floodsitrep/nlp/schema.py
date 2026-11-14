"""Claim types. Owner: member 4."""

from dataclasses import dataclass


@dataclass
class Claim:
    text: str
    cell: str | None
    quantity: str | None  # none / few / some / many / most
    damage_class: str | None  # none / minor / major / destroyed / damaged
    verdict: str = "not_verifiable"  # supported / contradicted / not_verifiable
