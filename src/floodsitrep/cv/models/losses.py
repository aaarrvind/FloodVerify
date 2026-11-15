"""Losses for 5-class damage segmentation with ignored pixels. Owner: member 2.

Targets use 255 for pixels to ignore (black no-data, set by the training loop from
``valid``). All losses are computed in float32 even under mixed precision.

- ``ce``: weighted cross-entropy.
- ``focal``: weighted focal loss, mean over non-ignored pixels (Lin et al. 2017).
- ``dice``: soft multi-class Dice over the batch, averaged over ``dice_classes``.
- ``dice_focal``: focal + dice_weight * dice.
"""
import torch
import torch.nn.functional as F
from torch import nn

IGNORE = 255


def _weights(w, device):
    return None if w is None else torch.as_tensor(w, dtype=torch.float32, device=device)


def focal_loss(logits, target, weight=None, gamma=2.0, ignore_index=IGNORE):
    logits = logits.float()
    ce = F.cross_entropy(logits, target, weight=weight, ignore_index=ignore_index,
                         reduction="none")
    keep = target != ignore_index
    if not keep.any():
        return logits.sum() * 0.0
    logp = F.log_softmax(logits, 1)
    t = target.clamp(max=logits.shape[1] - 1)
    pt = logp.gather(1, t.unsqueeze(1)).squeeze(1).exp()
    loss = ((1 - pt) ** gamma) * ce
    return loss[keep].mean()


def dice_loss(logits, target, classes=(0, 1, 2, 3, 4), smooth=1.0, ignore_index=IGNORE):
    logits = logits.float()
    n = logits.shape[1]
    keep = (target != ignore_index).unsqueeze(1).float()
    prob = F.softmax(logits, 1) * keep
    onehot = F.one_hot(target.clamp(max=n - 1), n).permute(0, 3, 1, 2).float() * keep
    dims = (0, 2, 3)
    inter = (prob * onehot).sum(dims)
    denom = prob.sum(dims) + onehot.sum(dims)
    dice = (2 * inter + smooth) / (denom + smooth)
    idx = torch.as_tensor(classes, device=logits.device)
    return 1 - dice[idx].mean()


class SegLoss(nn.Module):
    def __init__(self, kind="dice_focal", class_weights=None, focal_gamma=2.0, dice_weight=1.0,
                 dice_classes=(0, 1, 2, 3, 4)):
        super().__init__()
        if kind not in ("ce", "focal", "dice", "dice_focal"):
            raise ValueError(f"unknown loss {kind}")
        self.kind, self.gamma, self.dice_weight = kind, focal_gamma, dice_weight
        self.dice_classes = tuple(dice_classes)
        self.class_weights = None if class_weights is None else list(class_weights)

    def forward(self, logits, target):
        w = _weights(self.class_weights, logits.device)
        if self.kind == "ce":
            return F.cross_entropy(logits.float(), target, weight=w, ignore_index=IGNORE)
        if self.kind == "focal":
            return focal_loss(logits, target, w, self.gamma)
        if self.kind == "dice":
            return dice_loss(logits, target, self.dice_classes)
        return (focal_loss(logits, target, w, self.gamma)
                + self.dice_weight * dice_loss(logits, target, self.dice_classes))
