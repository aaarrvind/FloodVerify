"""Model registry and the checkpoint loader used by evaluation and evidence export.

Owner: member 2. The exact model call (for member 3 and the pipeline):

    from floodsitrep.cv.models import load_checkpoint
    model, cfg = load_checkpoint("/content/drive/.../checkpoints/<run>/best.pt", device="cuda")
    with torch.no_grad():
        logits = model(pre, post)      # pre, post: float [B,3,H,W] from XbdTiles (ImageNet-
                                       # normalised), H, W divisible by 32
    probs = logits.float().softmax(1)  # [B,5,H,W]: background, no, minor, major, destroyed

``load_checkpoint`` needs only the checkpoint file: the run config is stored inside it, and
no pretrained weights are downloaded (the trained weights replace them).
"""
import torch

MODELS = ("early_fusion_unet", "siamese_unet", "siamese_segformer")


def build_model(cfg, pretrained=None):
    """Build an untrained model from a config dict. ``pretrained`` overrides cfg."""
    name = cfg["model"]
    pre = cfg.get("pretrained", True) if pretrained is None else pretrained
    n = cfg.get("n_classes", 5)
    if name == "early_fusion_unet":
        from floodsitrep.cv.models.siamese_unet import EarlyFusionUNet
        return EarlyFusionUNet(n_classes=n, pretrained=pre)
    if name == "siamese_unet":
        from floodsitrep.cv.models.siamese_unet import SiameseUNet
        return SiameseUNet(n_classes=n, pretrained=pre)
    if name == "siamese_segformer":
        from floodsitrep.cv.models.siamese_segformer import SiameseSegFormer
        return SiameseSegFormer(n_classes=n, backbone=cfg.get("backbone", "nvidia/mit-b1"),
                                pretrained=pre, encoder_config=cfg.get("encoder_config"))
    raise ValueError(f"unknown model {name}; choose from {MODELS}")


def load_checkpoint(path, device="cpu"):
    """Checkpoint written by cv/train.py -> (model in eval mode on ``device``, config dict)."""
    ckpt = torch.load(path, map_location="cpu", weights_only=False)
    cfg = ckpt["config"]
    model = build_model(cfg, pretrained=False)
    model.load_state_dict(ckpt["model"])
    return model.to(device).eval(), cfg
