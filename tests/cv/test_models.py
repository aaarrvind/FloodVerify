"""Forward-pass shapes and losses (no pretrained download, CPU, small inputs)."""
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("torchvision")
models = pytest.importorskip("floodsitrep.cv.models")
losses = pytest.importorskip("floodsitrep.cv.models.losses")


@pytest.mark.parametrize("name", ["early_fusion_unet", "siamese_unet"])
def test_unet_forward_shape(name):
    m = models.build_model({"model": name}, pretrained=False).eval()
    pre, post = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    with torch.no_grad():
        out = m(pre, post)
    assert out.shape == (2, 5, 64, 64)


def test_segformer_forward_shape():
    pytest.importorskip("transformers")
    cfg = {"model": "siamese_segformer",
           "encoder_config": {"hidden_sizes": [16, 32, 48, 64], "depths": [1, 1, 1, 1],
                              "num_attention_heads": [1, 1, 1, 1],
                              "decoder_hidden_size": 32}}
    m = models.build_model(cfg, pretrained=False).eval()
    with torch.no_grad():
        out = m(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert out.shape == (1, 5, 64, 64)


@pytest.mark.parametrize("kind", ["ce", "focal", "dice", "dice_focal"])
def test_losses_ignore_and_finite(kind):
    torch.manual_seed(0)
    logits = torch.randn(2, 5, 8, 8, requires_grad=True)
    tgt = torch.randint(0, 5, (2, 8, 8))
    tgt[0, :4] = 255
    loss = losses.SegLoss(kind, class_weights=[0.1, 0.5, 0.9, 0.8, 2.7])(logits, tgt)
    loss.backward()
    assert torch.isfinite(loss) and loss > 0
    # gradients at ignored pixels are zero for pixel-wise losses
    if kind in ("ce", "focal"):
        assert logits.grad[0, :, :4].abs().sum() == 0


def test_dice_perfect_is_zero():
    tgt = torch.randint(0, 5, (1, 8, 8))
    logits = torch.nn.functional.one_hot(tgt, 5).permute(0, 3, 1, 2).float() * 50
    assert losses.dice_loss(logits, tgt) < 1e-3


def test_six_channel_init_duplicates_and_halves():
    """M0's first conv: both halves equal, so identical pre/post give the 3-channel response."""
    from floodsitrep.cv.models.siamese_unet import ResNetEncoder

    w = ResNetEncoder(6, pretrained=False).stem[0].weight
    assert w.shape[1] == 6
    assert torch.allclose(w[:, :3], w[:, 3:])
    x = torch.randn(1, 3, 32, 32)
    a = torch.nn.functional.conv2d(x, w[:, :3] * 2, stride=2, padding=3)
    b = torch.nn.functional.conv2d(torch.cat([x, x], 1), w, stride=2, padding=3)
    assert torch.allclose(a, b, atol=1e-5)
