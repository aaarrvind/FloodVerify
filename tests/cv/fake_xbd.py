"""Tiny synthetic xBD layout for tests and dry runs (no real data).

make_fake_xbd(root) writes 1024 px pre/post images, targets and label JSONs in the official
folder layout (train/, hold/, tier3/) plus a manifest with 4 crops per image. Buildings are
non-touching rectangles; the post image darkens a building more the worse its damage, so a
model can learn something. One crop per image gets a black no-data corner.
"""
import csv
import json
from pathlib import Path

SUBTYPES = ("no-damage", "minor-damage", "major-damage", "destroyed")
EVENTS = (("hurricane-harvey", "train"), ("hurricane-florence", "hold"),
          ("nepal-flooding", "tier3"))


def _image(rng, buildings, post):
    import numpy as np

    img = rng.integers(60, 110, (1024, 1024, 3), dtype=np.uint8)
    for x0, y0, x1, y1, k in buildings:
        colour = np.array([200, 190, 170]) * (1.0 - 0.22 * k if post else 1.0)
        img[y0:y1, x0:x1] = colour.astype(np.uint8)
    return img


def make_fake_xbd(root, per_event=2, seed=0):
    import numpy as np
    from PIL import Image

    rng = np.random.default_rng(seed)
    root = Path(root)
    rows = []
    for event, split in EVENTS:
        for d in ("images", "labels", "targets"):
            (root / split / d).mkdir(parents=True, exist_ok=True)
        for t in range(per_event):
            tile = f"{t:08d}"
            buildings = []
            for gy in range(40, 1000, 90):          # grid of separated buildings
                for gx in range(40, 1000, 90):
                    if rng.random() < 0.45:
                        w, h = rng.integers(20, 50, 2)
                        buildings.append((gx, gy, gx + int(w), gy + int(h),
                                          int(rng.integers(0, 4))))
            pre, post = _image(rng, buildings, False), _image(rng, buildings, True)
            pre[:120, :150] = 0                      # no-data corner in part 1
            post[:120, :150] = 0
            target = np.zeros((1024, 1024), np.uint8)
            feats = []
            for x0, y0, x1, y1, k in buildings:
                target[y0:y1, x0:x1] = k + 1
                wkt = (f"POLYGON (({x0} {y0}, {x1} {y0}, {x1} {y1}, {x0} {y1}, {x0} {y0}))")
                feats.append({"properties": {"subtype": SUBTYPES[k]}, "wkt": wkt})
            base = f"{event}_{tile}"
            Image.fromarray(pre).save(root / split / "images" / f"{base}_pre_disaster.png")
            Image.fromarray(post).save(root / split / "images" / f"{base}_post_disaster.png")
            Image.fromarray(target).save(root / split / "targets" /
                                         f"{base}_post_disaster_target.png")
            (root / split / "labels" / f"{base}_post_disaster.json").write_text(
                json.dumps({"features": {"xy": feats}}))
            for part, (x0, y0) in {"1": (0, 0), "2": (512, 0), "3": (0, 512),
                                   "4": (512, 512)}.items():
                rows.append({"event": event, "tile_id": tile, "part": part, "split": split,
                             "x0": x0, "y0": y0, "caption": ""})
    manifest = root / "rscc_flood_manifest.csv"
    with open(manifest, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    return manifest


if __name__ == "__main__":
    import sys

    print(make_fake_xbd(sys.argv[1], per_event=int(sys.argv[2]) if len(sys.argv) > 2 else 2))
