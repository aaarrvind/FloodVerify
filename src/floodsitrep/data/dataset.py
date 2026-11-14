"""PyTorch dataset over 512 px xBD tiles, cut on the fly. Owner: member 1.

No pre-built tile folders are needed: crops are read from the xBD folders using the manifest.
For speed in Colab, copy the xBD folders to local disk first (see data/README.md).

    from floodsitrep.data.dataset import XbdTiles
    ds = XbdTiles("/content/xbd", "/content/xbd/rscc_flood_manifest.csv", role="train", augment=True)
    item = ds[0]   # keys: pre, post (float 3x512x512), mask (long 512x512), valid (bool), tile_id

``valid`` is False where either image is black (no-data). Ignore those pixels in the loss, e.g.
``mask[~valid] = 255`` with ``ignore_index=255``, or multiply the loss by ``valid``.
"""
import random

import numpy as np
import torch
from torch.utils.data import Dataset

from floodsitrep.data.splits import role_of
from floodsitrep.data.tiling import load_crop, read_manifest, tile_name

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _to_tensor(img):
    x = (np.ascontiguousarray(img).astype(np.float32) / 255.0 - MEAN) / STD
    return torch.from_numpy(x.transpose(2, 0, 1))


class XbdTiles(Dataset):
    def __init__(self, xbd_root, manifest, role=None, events=None, augment=False,
                 min_buildings_tiles=None):
        """role: train, val or test (see splits.py). events: optional list to restrict.
        min_buildings_tiles: optional set of tile ids to keep (for example the usable tiles)."""
        rows = read_manifest(manifest)
        if role is not None:
            rows = [r for r in rows if role_of(r["event"]) == role]
        if events is not None:
            rows = [r for r in rows if r["event"] in set(events)]
        if min_buildings_tiles is not None:
            rows = [r for r in rows if tile_name(r) in min_buildings_tiles]
        self.rows, self.root, self.augment = rows, xbd_root, augment

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, i):
        r = self.rows[i]
        pre, post, mask, valid = load_crop(self.root, r)
        if self.augment:
            if random.random() < 0.5:
                pre, post, mask, valid = pre[:, ::-1], post[:, ::-1], mask[:, ::-1], valid[:, ::-1]
            if random.random() < 0.5:
                pre, post, mask, valid = pre[::-1], post[::-1], mask[::-1], valid[::-1]
            k = random.randint(0, 3)
            pre, post, mask, valid = (np.rot90(x, k) for x in (pre, post, mask, valid))
        return {
            "pre": _to_tensor(pre),
            "post": _to_tensor(post),
            "mask": torch.from_numpy(np.ascontiguousarray(mask).astype(np.int64)),
            "valid": torch.from_numpy(np.ascontiguousarray(valid)),
            "tile_id": tile_name(r),
        }
