"""M0 early-fusion U-Net and M1 siamese U-Net (ResNet-34 encoder). Owner: member 2.

Both take a pre and a post image (float [B,3,H,W], ImageNet-normalised, H and W divisible
by 32) and return logits [B,5,H,W] for 0 background, 1 no damage, 2 minor, 3 major,
4 destroyed. This is a building-damage map after a flood or storm, not a flood map.

- M0 ``EarlyFusionUNet``: pre and post stacked as 6 channels into one ResNet-34 U-Net.
  The pretrained first convolution is copied to both halves and halved, so the start
  point matches the 3-channel network on identical inputs.
- M1 ``SiameseUNet``: one ResNet-34 encoder with SHARED weights runs on pre and on post.
  At each of the 5 scales the two feature maps are fused as concat(pre, post, post - pre)
  followed by a 1x1 conv, BN and ReLU back to the encoder width; a U-Net decoder then
  upsamples to full resolution.

Encoder: torchvision ResNet-34 (ImageNet weights ``IMAGENET1K_V1``, BSD-3 code licence;
check the weight terms before any release). Built from torchvision rather than
segmentation_models_pytorch so there is no extra dependency and no API drift on Colab.
"""
import torch
import torch.nn.functional as F
from torch import nn

ENC_CHANNELS = (64, 64, 128, 256, 512)  # strides 2, 4, 8, 16, 32
DEC_CHANNELS = (256, 128, 64, 32, 16)


def _resnet34(pretrained):
    from torchvision.models import ResNet34_Weights, resnet34

    return resnet34(weights=ResNet34_Weights.IMAGENET1K_V1 if pretrained else None)


class ResNetEncoder(nn.Module):
    """ResNet-34 returning 5 feature maps at strides 2, 4, 8, 16, 32."""

    def __init__(self, in_channels=3, pretrained=True):
        super().__init__()
        r = _resnet34(pretrained)
        if in_channels != 3:
            w = r.conv1.weight.data  # [64, 3, 7, 7]
            conv = nn.Conv2d(in_channels, 64, 7, 2, 3, bias=False)
            reps = -(-in_channels // 3)
            conv.weight.data = torch.cat([w] * reps, 1)[:, :in_channels] * (3 / in_channels)
            r.conv1 = conv
        self.stem = nn.Sequential(r.conv1, r.bn1, r.relu)
        self.pool = r.maxpool
        self.layer1, self.layer2, self.layer3, self.layer4 = r.layer1, r.layer2, r.layer3, r.layer4

    def forward(self, x):
        f1 = self.stem(x)
        f2 = self.layer1(self.pool(f1))
        f3 = self.layer2(f2)
        f4 = self.layer3(f3)
        f5 = self.layer4(f4)
        return [f1, f2, f3, f4, f5]


def conv_bn_relu(cin, cout, k=3):
    return nn.Sequential(nn.Conv2d(cin, cout, k, padding=k // 2, bias=False),
                         nn.BatchNorm2d(cout), nn.ReLU(inplace=True))


class DecoderBlock(nn.Module):
    def __init__(self, cin, cskip, cout):
        super().__init__()
        self.conv = nn.Sequential(conv_bn_relu(cin + cskip, cout), conv_bn_relu(cout, cout))

    def forward(self, x, skip=None):
        x = F.interpolate(x, scale_factor=2, mode="nearest")
        if skip is not None:
            x = torch.cat([x, skip], 1)
        return self.conv(x)


class UNetDecoder(nn.Module):
    """Standard U-Net decoder over 5 encoder scales; output at full input resolution."""

    def __init__(self, enc=ENC_CHANNELS, dec=DEC_CHANNELS, n_classes=5):
        super().__init__()
        skips = list(enc[:-1][::-1]) + [0]  # 256, 128, 64, 64, none
        cins = [enc[-1]] + list(dec[:-1])
        self.blocks = nn.ModuleList(DecoderBlock(i, s, o) for i, s, o in zip(cins, skips, dec))
        self.head = nn.Conv2d(dec[-1], n_classes, 3, padding=1)

    def forward(self, feats):
        x = feats[-1]
        skips = feats[:-1][::-1] + [None]
        for block, skip in zip(self.blocks, skips):
            x = block(x, skip)
        return self.head(x)


class EarlyFusionUNet(nn.Module):
    """M0: concat(pre, post) as 6 channels -> ResNet-34 U-Net."""

    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=6, pretrained=pretrained)
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pre, post):
        return self.decoder(self.encoder(torch.cat([pre, post], 1)))


class SiameseUNet(nn.Module):
    """M1: shared ResNet-34 on pre and post, fused per scale, U-Net decoder."""

    def __init__(self, n_classes=5, pretrained=True):
        super().__init__()
        self.encoder = ResNetEncoder(in_channels=3, pretrained=pretrained)
        self.fuse = nn.ModuleList(conv_bn_relu(3 * c, c, k=1) for c in ENC_CHANNELS)
        self.decoder = UNetDecoder(n_classes=n_classes)

    def forward(self, pre, post):
        b = pre.shape[0]
        feats = self.encoder(torch.cat([pre, post], 0))  # one pass, shared weights
        fused = []
        for f, fuse in zip(feats, self.fuse):
            a, p = f[:b], f[b:]
            fused.append(fuse(torch.cat([a, p, p - a], 1)))
        return self.decoder(fused)
