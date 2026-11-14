"""xBD manifest -> 512 px tile specs and crops. Owner: member 1.

The manifest (rscc_flood_manifest.csv) has one row per 512 px quarter of a 1024 px xBD tile.
Columns: event, tile_id, part, split, x0, y0, caption. ``split`` is the OFFICIAL xBD folder
(train, test, hold or tier3), not the project role (see splits.py).

Part numbering (checked pixel by pixel against the RSCC sample):
part 1 top-left, part 2 top-right, part 3 bottom-left, part 4 bottom-right.
"""
import csv
from pathlib import Path

SIZE = 512
PART_ORIGIN = {"1": (0, 0), "2": (512, 0), "3": (0, 512), "4": (512, 512)}


def tile_name(row):
    """Project tile id, e.g. hurricane-florence_00000021_p1."""
    return f"{row['event']}_{row['tile_id']}_p{row['part']}"


def read_manifest(path):
    """Read the manifest. x0 and y0 become int; tile_id and part stay strings."""
    with open(path, newline="", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    for r in rows:
        r["x0"], r["y0"] = int(r["x0"]), int(r["y0"])
    return rows


def source_paths(root, row):
    """Paths of pre image, post image, damage target (mask) and label JSON for a row."""
    base = f"{row['event']}_{row['tile_id']}"
    d = Path(root) / row["split"]
    return {
        "pre": d / "images" / f"{base}_pre_disaster.png",
        "post": d / "images" / f"{base}_post_disaster.png",
        "target": d / "targets" / f"{base}_post_disaster_target.png",
        "label": d / "labels" / f"{base}_post_disaster.json",
    }


def crop_box(row, size=SIZE):
    return (row["x0"], row["y0"], row["x0"] + size, row["y0"] + size)


def load_crop(root, row, size=SIZE):
    """Return (pre, post, mask, valid) as numpy arrays for one 512 px tile.

    pre, post: uint8 HxWx3. mask: uint8 HxW with 0 background, 1 no damage, 2 minor,
    3 major, 4 destroyed. valid: bool HxW, False where either image is black (no-data).
    Needs numpy and pillow.
    """
    import numpy as np
    from PIL import Image

    p = source_paths(root, row)
    box = crop_box(row, size)
    pre = np.array(Image.open(p["pre"]).convert("RGB").crop(box))
    post = np.array(Image.open(p["post"]).convert("RGB").crop(box))
    mask = np.array(Image.open(p["target"]).crop(box))
    valid = (pre.sum(axis=2) > 0) & (post.sum(axis=2) > 0)
    return pre, post, mask, valid
