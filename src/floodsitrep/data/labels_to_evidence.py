"""xBD labels -> evidence JSON with source=labels. Owner: member 1.

Buildings are taken from the label polygons (exact), not from the damage mask. Counting
connected blobs in the mask merges touching buildings and undercounts them (tested: 56%
of the true count for nepal-flooding, 79-89% for the hurricane events).

Each building is counted once, in the grid cell that contains its centroid, in the 512 px
tile that contains that centroid. Buildings with subtype ``un-classified`` are skipped.
Pure Python except valid_fractions(), which needs numpy and pillow.
"""
import json
import re
from pathlib import Path

CLASS_OF = {
    "no-damage": "none",
    "minor-damage": "minor",
    "major-damage": "major",
    "destroyed": "destroyed",
}
CLASSES = ("none", "minor", "major", "destroyed")
_NUM = r"-?\d+(?:\.\d+)?(?:[eE][-+]?\d+)?"
_PAIR = re.compile(rf"({_NUM})\s+({_NUM})")


def parse_wkt(wkt):
    """First ring of a WKT POLYGON as a list of (x, y) floats. Empty list if none."""
    return [(float(a), float(b)) for a, b in _PAIR.findall(wkt.split("),")[0])]


def centroid(points):
    """Area centroid of a polygon; mean of the vertices if the area is zero."""
    if not points:
        return None
    area = cx = cy = 0.0
    for (x1, y1), (x2, y2) in zip(points, points[1:] + points[:1]):
        cross = x1 * y2 - x2 * y1
        area += cross
        cx += (x1 + x2) * cross
        cy += (y1 + y2) * cross
    if abs(area) < 1e-9:
        return (sum(p[0] for p in points) / len(points), sum(p[1] for p in points) / len(points))
    area *= 0.5
    return (cx / (6 * area), cy / (6 * area))


def buildings_in_crop(label, x0, y0, size=512):
    """Buildings whose centroid lies in the crop: list of (x, y, class) relative to the crop."""
    out = []
    for feat in label["features"]["xy"]:
        cls = CLASS_OF.get(feat["properties"].get("subtype"))
        if cls is None:  # un-classified or unknown
            continue
        c = centroid(parse_wkt(feat["wkt"]))
        if c is None:
            continue
        x, y = c[0] - x0, c[1] - y0
        if 0 <= x < size and 0 <= y < size:
            out.append((x, y, cls))
    return out


def grid_cells(buildings, size=512, grid=3):
    """Buildings -> list of grid*grid cell dicts in the evidence schema (confidence 1.0)."""
    counts = [[{c: 0 for c in CLASSES} for _ in range(grid)] for _ in range(grid)]
    for x, y, cls in buildings:
        r, c = min(int(y * grid / size), grid - 1), min(int(x * grid / size), grid - 1)
        counts[r][c][cls] += 1
    cells = []
    for r in range(grid):
        for c in range(grid):
            n = sum(counts[r][c].values())
            frac = {k: round(counts[r][c][k] / n, 4) if n else 0.0 for k in CLASSES}
            cells.append({"cell": f"r{r}c{c}", "buildings": n, "class_fraction": frac,
                          "confidence": 1.0})
    return cells


def valid_fractions(pre_path, post_path, x0, y0, size=512, grid=3):
    """Share of each grid cell with real imagery (not black in either image), row-major."""
    import numpy as np
    from PIL import Image

    box = (x0, y0, x0 + size, y0 + size)
    a = np.array(Image.open(pre_path).convert("RGB").crop(box)).sum(axis=2) > 0
    b = np.array(Image.open(post_path).convert("RGB").crop(box)).sum(axis=2) > 0
    valid = a & b
    out = []
    for r in range(grid):
        for c in range(grid):
            cell = valid[r * size // grid:(r + 1) * size // grid,
                         c * size // grid:(c + 1) * size // grid]
            out.append(round(float(cell.mean()), 3))
    return out


def evidence_from_label(label, tile_id, x0, y0, valid=None, size=512, gsd_m=0.5):
    """Build one evidence dict (source=labels). gsd_m is a nominal value; the true xBD
    resolution varies by image, so do not use it for precise measurement."""
    cells = grid_cells(buildings_in_crop(label, x0, y0, size), size)
    if valid is not None:
        for cell, v in zip(cells, valid):
            cell["valid_fraction"] = v
    return {"tile_id": tile_id, "source": "labels", "gsd_m": gsd_m, "grid": "3x3", "cells": cells}


def write_evidence(evidence, out_dir):
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{evidence['tile_id']}.json"
    path.write_text(json.dumps(evidence, indent=2))
    return path
