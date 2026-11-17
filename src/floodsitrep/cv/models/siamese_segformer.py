"""M2 siamese SegFormer (MiT-B1 or MiT-B2 encoder). Owner: member 2. Optional run.

Same interface as M0/M1: forward(pre, post) -> logits [B,5,H,W].

A Mix Transformer encoder with SHARED weights runs on pre and post. Its 4 feature maps
(strides 4, 8, 16, 32) are fused per scale as concat(pre, post, post - pre) -> 1x1 conv,
then an all-MLP SegFormer-style head (project each scale to ``embed`` channels, upsample to
stride 4, concat, 1x1 fuse, classify) and a final bilinear x4 upsample.

Pretrained weights: Hugging Face ``nvidia/mit-b1`` / ``nvidia/mit-b2`` (ImageNet-1k).
Licence: NVIDIA SegFormer licence, research / non-commercial; check it before release.
Needs ``transformers``.
"""
import torch
import torch.nn.functional as F
from torch import nn

from floodsitrep.cv.models.siamese_unet import conv_bn_relu


def _mit(backbone, pretrained, encoder_config=None):
    from transformers import SegformerConfig, SegformerModel

    if encoder_config is not None:  # rebuilding from a checkpoint: no download needed
        return SegformerModel(SegformerConfig(**encoder_config))
    if pretrained:
        return SegformerModel.from_pretrained(backbone)
    try:
        cfg = SegformerConfig.from_pretrained(backbone)
    except OSError:  # offline: MiT-B1 shape
        cfg = SegformerConfig(hidden_sizes=[64, 128, 320, 512], depths=[2, 2, 2, 2])
    return SegformerModel(cfg)


class SiameseSegFormer(nn.Module):
    def __init__(self, n_classes=5, backbone="nvidia/mit-b1", pretrained=True, embed=256,
                 encoder_config=None):
        super().__init__()
        self.encoder = _mit(backbone, pretrained, encoder_config)
        chans = list(self.encoder.config.hidden_sizes)
        self.fuse = nn.ModuleList(conv_bn_relu(3 * c, c, k=1) for c in chans)
        self.proj = nn.ModuleList(nn.Conv2d(c, embed, 1) for c in chans)
        self.mix = conv_bn_relu(embed * len(chans), embed, k=1)
        self.drop = nn.Dropout2d(0.1)
        self.cls = nn.Conv2d(embed, n_classes, 1)

    def forward(self, pre, post):
        b, _, h, w = pre.shape
        out = self.encoder(torch.cat([pre, post], 0), output_hidden_states=True)
        feats = out.hidden_states  # 4 maps [2B, C_i, H/s, W/s]
        size = feats[0].shape[-2:]
        ups = []
        for f, fuse, proj in zip(feats, self.fuse, self.proj):
            a, p = f[:b], f[b:]
            x = proj(fuse(torch.cat([a, p, p - a], 1)))
            ups.append(F.interpolate(x, size=size, mode="bilinear", align_corners=False))
        x = self.cls(self.drop(self.mix(torch.cat(ups, 1))))
        return F.interpolate(x, size=(h, w), mode="bilinear", align_corners=False)
