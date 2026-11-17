"""Training entry point for M0 / M1 / M2. Owner: member 2.

    python -m floodsitrep.cv.train --config configs/cv/m1_siamese_unet.yaml \
        --xbd-root /content/xbd --manifest /content/xbd/rscc_flood_manifest.csv \
        --out-dir "/content/drive/MyDrive/MAAI 902 X 901 PROJECT/checkpoints"

Writes to <out-dir>/<run_name>/ (put it on Drive; never in git):
    config.yaml      the resolved config of this run
    last.pt          after EVERY epoch (Colab sessions die); re-running the same command
                     resumes from it automatically
    best.pt          best validation xview2_pixel_score so far
    metrics.jsonl    one line per epoch: train loss, val loss, pixel-level per-class F1/IoU,
                     localisation F1, damage F1 (on ground-truth building pixels) and the
                     pixel-level approximation of the xView2 score

Validation uses the ``val`` role (hurricane-florence) only. The ``test`` role (nepal-flooding)
is never touched here. Pixels with valid == False (black no-data) are ignored in the loss and
in the metrics. Override any config key on the command line: ``--set epochs=1 lr=3e-4``.
"""
import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from floodsitrep.cv.eval import metrics as M
from floodsitrep.cv.models import build_model
from floodsitrep.cv.models.losses import IGNORE, SegLoss

DEFAULTS = {
    "run_name": None,
    "model": "siamese_unet",
    "pretrained": True,
    "n_classes": 5,
    "loss": "dice_focal",
    "class_weights": [0.10, 0.52, 0.88, 0.75, 2.74],
    "focal_gamma": 2.0,
    "dice_weight": 1.0,
    "dice_classes": [0, 1, 2, 3, 4],
    "epochs": 20,
    "batch_size": 8,
    "lr": 3e-4,
    "encoder_lr_mult": 1.0,
    "weight_decay": 1e-4,
    "warmup_steps": 200,
    "num_workers": 2,
    "amp": True,
    "seed": 0,
    "max_train_tiles": 0,   # 0 = all
    "max_val_tiles": 0,     # 0 = all; a fixed seeded subset otherwise
    "log_every": 50,
    "grad_clip": 1.0,
}


def load_config(path, overrides=()):
    import yaml

    cfg = dict(DEFAULTS)
    if path:
        cfg.update(yaml.safe_load(Path(path).read_text()) or {})
    for kv in overrides:
        k, v = kv.split("=", 1)
        cfg[k] = yaml.safe_load(v)
    for k, v in cfg.items():  # PyYAML reads "1e-3" (no dot) as a string
        if isinstance(v, str):
            try:
                cfg[k] = float(v)
            except ValueError:
                pass
    if not cfg.get("run_name"):
        stem = Path(path).stem if path else cfg["model"]
        cfg["run_name"] = f"{stem}_s{cfg['seed']}"
    return cfg


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _worker_init(worker_id):
    s = torch.initial_seed() % 2**32
    random.seed(s)
    np.random.seed(s)


def subset(ds, n, seed):
    """Keep a seeded random subset of n rows (in place). n = 0 keeps all."""
    if n and n < len(ds.rows):
        rows = list(ds.rows)
        random.Random(seed).shuffle(rows)
        ds.rows = rows[:n]
    return ds


def make_loaders(cfg, xbd_root, manifest):
    from floodsitrep.data.dataset import XbdTiles

    tr = subset(XbdTiles(xbd_root, manifest, role="train", augment=True),
                cfg["max_train_tiles"], cfg["seed"])
    va = subset(XbdTiles(xbd_root, manifest, role="val", augment=False),
                cfg["max_val_tiles"], 12345)
    g = torch.Generator().manual_seed(cfg["seed"])
    kw = {"num_workers": cfg["num_workers"], "pin_memory": torch.cuda.is_available(),
          "worker_init_fn": _worker_init, "persistent_workers": cfg["num_workers"] > 0}
    tl = DataLoader(tr, batch_size=cfg["batch_size"], shuffle=True, drop_last=True,
                    generator=g, **kw)
    vl = DataLoader(va, batch_size=cfg["batch_size"], shuffle=False, **kw)
    return tl, vl


def prepare_target(batch, device):
    """mask with 255 on no-data pixels and on any value outside 0..4."""
    mask = batch["mask"].to(device, non_blocking=True)
    valid = batch["valid"].to(device, non_blocking=True)
    mask = mask.clone()
    mask[~valid | (mask > 4)] = IGNORE
    return mask


def param_groups(model, cfg):
    enc = [p for n, p in model.named_parameters() if n.startswith("encoder.")]
    rest = [p for n, p in model.named_parameters() if not n.startswith("encoder.")]
    return [{"params": enc, "lr": cfg["lr"] * cfg["encoder_lr_mult"]},
            {"params": rest, "lr": cfg["lr"]}]


def cosine_with_warmup(total, warmup):
    def f(step):
        if step < warmup:
            return (step + 1) / max(warmup, 1)
        p = (step - warmup) / max(total - warmup, 1)
        return 0.5 * (1 + math.cos(math.pi * min(p, 1.0)))
    return f


@torch.no_grad()
def validate(model, loader, loss_fn, device, amp):
    """Mean val loss and summed pixel confusion matrix over the loader."""
    model.eval()
    cm = torch.zeros(5, 5, dtype=torch.int64, device=device)
    tot, n = 0.0, 0
    for batch in loader:
        pre = batch["pre"].to(device, non_blocking=True)
        post = batch["post"].to(device, non_blocking=True)
        tgt = prepare_target(batch, device)
        with torch.autocast(device.type, enabled=amp):
            logits = model(pre, post)
        tot += float(loss_fn(logits, tgt)) * pre.shape[0]
        n += pre.shape[0]
        pred = logits.argmax(1)
        keep = tgt != IGNORE
        cm += torch.bincount(tgt[keep] * 5 + pred[keep], minlength=25).view(5, 5)
    return tot / max(n, 1), cm.cpu().numpy()


def save(path, obj):
    """Write atomically: a crash mid-save never leaves a broken last.pt."""
    tmp = Path(str(path) + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def train(cfg, xbd_root, manifest, out_dir, device=None):
    import yaml

    device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    amp = bool(cfg["amp"]) and device.type == "cuda"
    run_dir = Path(out_dir) / cfg["run_name"]
    run_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(cfg["seed"])
    torch.backends.cudnn.benchmark = True  # faster; not bit-exact across runs

    tl, vl = make_loaders(cfg, xbd_root, manifest)
    print(f"run {cfg['run_name']}: train {len(tl.dataset)} tiles, val {len(vl.dataset)} tiles, "
          f"device {device}, amp {amp}", flush=True)

    model = build_model(cfg).to(device)
    if cfg["model"] == "siamese_segformer":
        cfg["encoder_config"] = model.encoder.config.to_dict()
    (run_dir / "config.yaml").write_text(yaml.safe_dump(
        {k: v for k, v in cfg.items() if k != "encoder_config"}, sort_keys=False))
    loss_fn = SegLoss(cfg["loss"], cfg["class_weights"], cfg["focal_gamma"],
                      cfg["dice_weight"], cfg["dice_classes"])
    opt = torch.optim.AdamW(param_groups(model, cfg), weight_decay=cfg["weight_decay"])
    total = cfg["epochs"] * len(tl)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, cosine_with_warmup(total, cfg["warmup_steps"]))
    scaler = torch.amp.GradScaler(device.type, enabled=amp)

    start, best = 0, -1.0
    last = run_dir / "last.pt"
    if last.exists():
        ck = torch.load(last, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["optimizer"])
        sched.load_state_dict(ck["scheduler"])
        scaler.load_state_dict(ck["scaler"])
        start, best = ck["epoch"] + 1, ck["best"]
        print(f"resumed from {last} at epoch {start} (best so far {best:.4f})", flush=True)

    for epoch in range(start, cfg["epochs"]):
        seed_everything(cfg["seed"] * 1000 + epoch)
        tl.generator.manual_seed(cfg["seed"] * 1000 + epoch)  # same order after a resume
        model.train()
        t0, run_loss, seen = time.time(), 0.0, 0
        for step, batch in enumerate(tl):
            pre = batch["pre"].to(device, non_blocking=True)
            post = batch["post"].to(device, non_blocking=True)
            tgt = prepare_target(batch, device)
            with torch.autocast(device.type, enabled=amp):
                logits = model(pre, post)
            loss = loss_fn(logits, tgt)
            opt.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            if cfg["grad_clip"]:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg["grad_clip"])
            scaler.step(opt)
            scaler.update()
            sched.step()
            run_loss += loss.item() * pre.shape[0]
            seen += pre.shape[0]
            if (step + 1) % cfg["log_every"] == 0:
                print(f"  epoch {epoch} step {step + 1}/{len(tl)} loss {loss.item():.4f} "
                      f"(running {run_loss / seen:.4f})", flush=True)
        train_loss = run_loss / max(seen, 1)
        val_loss, cm = validate(model, vl, loss_fn, device, amp)
        s = M.flat(M.summarise(cm))
        rec = {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss,
               "lr": sched.get_last_lr()[-1], "seconds": round(time.time() - t0, 1),
               "metric_level": "pixel", **s, "val_confusion": cm.tolist()}
        score = s["xview2_pixel_score"] if not math.isnan(s["xview2_pixel_score"]) else 0.0
        state = {"model": model.state_dict(), "optimizer": opt.state_dict(),
                 "scheduler": sched.state_dict(), "scaler": scaler.state_dict(),
                 "epoch": epoch, "best": max(best, score), "config": cfg, "val": rec}
        save(last, state)
        if score > best:
            best = score
            save(run_dir / "best.pt", {"model": model.state_dict(), "config": cfg,
                                       "epoch": epoch, "val": rec})
        with open(run_dir / "metrics.jsonl", "a") as f:
            f.write(json.dumps(rec) + "\n")
        print(f"epoch {epoch}: train_loss {train_loss:.4f} val_loss {val_loss:.4f} "
              f"xview2_pixel {s['xview2_pixel_score']:.4f} loc_f1 {s['loc_f1']:.4f} "
              f"dmg_f1 minor {s['dmg_f1_minor']:.3f} major {s['dmg_f1_major']:.3f} "
              f"destroyed {s['dmg_f1_destroyed']:.3f} ({rec['seconds']}s)", flush=True)
    return run_dir


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("--config", required=True)
    ap.add_argument("--xbd-root", required=True)
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--device", default=None)
    ap.add_argument("--set", nargs="*", default=[], metavar="KEY=VALUE")
    a = ap.parse_args(argv)
    cfg = load_config(a.config, a.set)
    train(cfg, a.xbd_root, a.manifest, a.out_dir, a.device)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
