"""Run a trained checkpoint over a role and stream per-tile results. Owner: member 3.

    for r in iter_predictions(model, xbd_root, manifest, role="val"):
        r["tile_id"], r["logits"] (float32 [5,H,W]), r["mask"] (uint8 [H,W]), r["valid"] (bool)

``hflip_tta`` averages the logits of the image and its horizontal flip (off by default).
"""
import numpy as np
import torch
from torch.utils.data import DataLoader


def make_dataset(xbd_root, manifest, role, limit=0, events=None, sort=False):
    """sort=True orders rows by image then part, so the 4 crops of an image are adjacent."""
    from floodsitrep.data.dataset import XbdTiles

    ds = XbdTiles(xbd_root, manifest, role=role, events=events, augment=False)
    if sort:
        ds.rows = sorted(ds.rows, key=lambda r: (r["event"], r["tile_id"], r["part"]))
    if limit:
        ds.rows = ds.rows[:limit]
    return ds


@torch.no_grad()
def iter_predictions(model, xbd_root, manifest, role="val", batch_size=8, num_workers=2,
                     device=None, amp=True, limit=0, hflip_tta=False, events=None,
                     sort=False):
    device = torch.device(device or next(model.parameters()).device)
    amp = amp and device.type == "cuda"
    ds = make_dataset(xbd_root, manifest, role, limit, events, sort)
    dl = DataLoader(ds, batch_size=batch_size, shuffle=False, num_workers=num_workers,
                    pin_memory=device.type == "cuda")
    model.eval()
    for b in dl:
        pre, post = b["pre"].to(device), b["post"].to(device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(pre, post).float()
            if hflip_tta:
                logits = 0.5 * (logits + model(pre.flip(-1), post.flip(-1)).float().flip(-1))
        logits = logits.cpu().numpy()
        masks, valids = b["mask"].numpy().astype(np.uint8), b["valid"].numpy()
        for i, tid in enumerate(b["tile_id"]):
            yield {"tile_id": tid, "logits": logits[i], "mask": masks[i], "valid": valids[i]}
