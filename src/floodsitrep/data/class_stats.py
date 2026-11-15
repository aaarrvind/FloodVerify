"""Class pixel counts and loss weights for the training role. Owner: member 1.

    python -m floodsitrep.data.class_stats --xbd-root /path/to/xbd \
        --manifest /path/to/xbd/rscc_flood_manifest.csv --role train --max-tiles 600

Counts pixels per mask class (0 background, 1 no damage, 2 minor, 3 major, 4 destroyed) over a
random sample of tiles, ignoring black no-data pixels. Weights are inverse square root of the
class frequency, scaled so the mean weight is 1 (a common starting point, not a tuned value).
Needs numpy and pillow.
"""
import argparse
import json
import random

from floodsitrep.data.splits import role_of
from floodsitrep.data.tiling import load_crop, read_manifest


def class_counts(xbd_root, rows):
    import numpy as np

    counts = np.zeros(5, dtype=np.int64)
    for r in rows:
        _, _, mask, valid = load_crop(xbd_root, r)
        counts += np.bincount(mask[valid].ravel(), minlength=5)[:5]
    return counts


def inverse_sqrt_weights(counts):
    import numpy as np

    freq = counts / max(counts.sum(), 1)
    w = 1.0 / np.sqrt(np.maximum(freq, 1e-8))
    return w / w.mean()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--xbd-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--role", default="train")
    ap.add_argument("--max-tiles", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    a = ap.parse_args(argv)
    rows = [r for r in read_manifest(a.manifest) if role_of(r["event"]) == a.role]
    random.Random(a.seed).shuffle(rows)
    rows = rows[:a.max_tiles]
    counts = class_counts(a.xbd_root, rows)
    res = {"role": a.role, "tiles_sampled": len(rows), "pixel_counts": counts.tolist(),
           "fractions": [round(float(x), 5) for x in counts / counts.sum()],
           "inverse_sqrt_weights": [round(float(x), 4) for x in inverse_sqrt_weights(counts)]}
    print(json.dumps(res, indent=2))
    if a.out:
        with open(a.out, "w") as f:
            json.dump(res, f, indent=2)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
