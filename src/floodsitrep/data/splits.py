"""Event-level split. Owner: member 1.

Every tile of an event goes to ONE role, so neighbouring tiles of one storm never leak
between train, validation and test. Change the mapping here only (and tell the team).

Why this split (official xBD train split, share of buildings that are major or destroyed):
harvey 37.5%, matthew 29.4%, florence 20.2%, midwest 2.2%, nepal about 12%.
Midwest has almost no severe damage, so it is a poor validation event; it stays in training.
Nepal is never seen in training and has real major damage, so it is the unseen test event.
"""

ROLE_BY_EVENT = {
    "hurricane-harvey": "train",
    "hurricane-matthew": "train",
    "midwest-flooding": "train",
    "hurricane-florence": "val",
    "nepal-flooding": "test",
}

ROLES = ("train", "val", "test")


def role_of(event):
    """Role of an event, or None if the event is not part of the flood study."""
    return ROLE_BY_EVENT.get(event)
