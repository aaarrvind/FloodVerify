"""Evaluate a checkpoint on one role; optionally fit the temperature. Owner: member 3.

    # validation (hurricane-florence): metrics + fit temperature + ECE before/after
    python -m floodsitrep.cv.eval.evaluate --checkpoint <run>/best.pt --role val \
        --xbd-root /content/xbd --fit-temperature

    # held-out test (nepal-flooding): ONCE, at the end, with the frozen model and temperature
    python -m floodsitrep.cv.eval.evaluate --checkpoint <run>/best.pt --role test \
        --xbd-root /content/xbd

Writes next to the checkpoint (or to --out-dir):
    eval_<role>.json      pixel-level metrics with 95% bootstrap intervals over tiles,
                          plus the summed confusion matrix and the checkpoint/config used
    temperature.json      (with --fit-temperature, val only) T, NLL and ECE before/after
All numbers are PIXEL-level; the xView2 figure is a pixel-level approximation on our crops.
"""
import argparse
import json
from pathlib import Path

import numpy as np


def evaluate(checkpoint, xbd_root, manifest=None, role="val", out_dir=None, fit_temp=False,
             pixels_per_tile=2000, n_boot=1000, batch_size=8, limit=0, device=None,
             hflip_tta=False, seed=0):
    import torch

    from floodsitrep.cv.eval import calibration as C
    from floodsitrep.cv.eval import metrics as M
    from floodsitrep.cv.eval.infer import iter_predictions
    from floodsitrep.cv.models import load_checkpoint

    if fit_temp and role != "val":
        raise ValueError("fit the temperature on the val role only")
    manifest = manifest or str(Path(xbd_root) / "rscc_flood_manifest.csv")
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    model, cfg = load_checkpoint(checkpoint, device)
    out = Path(out_dir or Path(checkpoint).parent)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    cms, ids, lg, lb = [], [], [], []
    for r in iter_predictions(model, xbd_root, manifest, role, batch_size, device=device,
                              limit=limit, hflip_tta=hflip_tta):
        pred = r["logits"].argmax(0)
        cms.append(M.confusion(pred, r["mask"], r["valid"]))
        ids.append(r["tile_id"])
        if fit_temp:
            x, y = C.sample_pixels(r["logits"], r["mask"], r["valid"], pixels_per_tile, rng)
            lg.append(x)
            lb.append(y)
    cms = np.stack(cms)
    res = {
        "checkpoint": str(checkpoint), "run_name": cfg.get("run_name"), "role": role,
        "tiles": len(ids), "level": "pixel", "hflip_tta": hflip_tta,
        "note": "pixel-level approximation of the xView2 score on 512 px crops",
        "summary": M.summarise(cms.sum(0)),
        "bootstrap_95": M.bootstrap(cms, n_boot=n_boot, seed=seed),
        "confusion": cms.sum(0).tolist(),
    }
    (out / f"eval_{role}.json").write_text(json.dumps(res, indent=2))
    if fit_temp:
        rep = C.calibration_report(np.concatenate(lg), np.concatenate(lb))
        rep.update({"checkpoint": str(checkpoint), "pixels_per_tile": pixels_per_tile})
        C.save_temperature(rep, out / "temperature.json")
        res["calibration"] = rep
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--xbd-root", required=True)
    ap.add_argument("--manifest", default=None)
    ap.add_argument("--role", default="val")
    ap.add_argument("--out-dir", default=None)
    ap.add_argument("--fit-temperature", action="store_true")
    ap.add_argument("--n-boot", type=int, default=1000)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--hflip-tta", action="store_true")
    a = ap.parse_args(argv)
    res = evaluate(a.checkpoint, a.xbd_root, a.manifest, a.role, a.out_dir, a.fit_temperature,
                   n_boot=a.n_boot, batch_size=a.batch_size, limit=a.limit,
                   hflip_tta=a.hflip_tta)
    b = res["bootstrap_95"]
    print(f"{res['run_name']} on {a.role} ({res['tiles']} tiles), PIXEL level, 95% CI over tiles:")
    for k in ("xview2_pixel_score", "loc_f1", "damage_f1_hm", "dmg_f1_no_damage",
              "dmg_f1_minor", "dmg_f1_major", "dmg_f1_destroyed", "miou"):
        print(f"  {k:20s} {b[k]['value']:.4f}  [{b[k]['lo']:.4f}, {b[k]['hi']:.4f}]")
    if "calibration" in res:
        c = res["calibration"]
        print(f"  temperature {c['temperature']:.3f}: ECE {c['ece_before']:.4f} -> "
              f"{c['ece_after']:.4f}, NLL {c['nll_before']:.4f} -> {c['nll_after']:.4f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
