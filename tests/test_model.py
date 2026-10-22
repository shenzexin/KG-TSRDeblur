import pytest
import torch
from kg_tsrdeblur.model import KGTSRDeblur, ModelConfig, MotionAdapter, WindowBlock
from kg_tsrdeblur.losses import restoration_loss, ssim_per_image


def small_config(**kwargs):
    return ModelConfig(width=8, encoder_depths=(1, 1), decoder_depths=(1, 1),
                       transformer_blocks=2, refinement_channels=4, **kwargs)


def test_zero_reliability_exact_image_fallback():
    model = KGTSRDeblur(small_config()).eval()
    x = torch.rand(2, 3, 35, 47)
    with torch.no_grad():
        a, gate = model(x, torch.randn(2, 16), torch.zeros(2), True)
        b = model(x, torch.zeros(2, 16), torch.zeros(2))
    assert a.shape == x.shape
    assert torch.equal(a, b)
    assert torch.count_nonzero(gate) == 0


def test_gate_bound_and_motion_gradient():
    model = KGTSRDeblur(small_config())
    r = torch.tensor([0.2, 0.8])
    pred, gate = model(torch.rand(2, 3, 32, 40), torch.randn(2, 16), r, True)
    assert (gate >= 0).all() and (gate <= r[:, None, None, None]).all()
    pred.square().mean().backward()
    assert model.motion.encoder[0].weight.grad.abs().sum() > 0


@pytest.mark.parametrize('shape', [(1, 3, 128, 128), (2, 3, 17, 21), (1, 3, 4, 4)])
def test_shapes_and_finite_gradient(shape):
    model = KGTSRDeblur(small_config())
    x = torch.rand(*shape, requires_grad=True)
    result = model(x)
    assert result.shape == x.shape and torch.isfinite(result).all()
    result.square().mean().backward()
    assert torch.isfinite(x.grad).all()


def test_shift_mask_blocks_cyclic_wrap():
    block = WindowBlock(8, 2, window=4, shift=2).eval()
    # Zero local FFN: inspect attention alone. Top-left must not see wrapped bottom-right.
    for p in block.ffn.parameters():
        p.data.zero_()
    x = torch.randn(1, 8, 8, 8, requires_grad=True)
    block(x)[0, 0, 0, 0].backward()
    assert torch.count_nonzero(x.grad[:, :, -2:, -2:]) == 0


def test_frozen_classifier_transmits_gradients():
    classifier = torch.nn.Sequential(torch.nn.AdaptiveAvgPool2d(1), torch.nn.Flatten(), torch.nn.Linear(3, 2))
    classifier.requires_grad_(False).eval()
    x = torch.rand(2, 3, 16, 16, requires_grad=True)
    _, terms = restoration_loss(x, x.detach(), classifier, torch.tensor([0, 1]), lambda_c=0.1)
    terms['recognition'].backward()
    assert x.grad.abs().sum() > 0
    assert all(p.grad is None for p in classifier.parameters())


def test_identity_loss():
    x = torch.rand(2, 3, 16, 16)
    total, _ = restoration_loss(x, x)
    assert torch.allclose(total, torch.tensor(0.), atol=1e-6)
    assert torch.allclose(ssim_per_image(x, x), torch.ones(2), atol=1e-6)


@pytest.mark.parametrize('variant', ['image_only', 'spatial_only', 'conv_bottleneck'])
def test_ablation_shapes(variant):
    model = KGTSRDeblur(small_config(**{variant: True}))
    x = torch.rand(1, 3, 32, 32)
    assert model(x).shape == x.shape


def test_invalid_reliability():
    model = KGTSRDeblur(small_config())
    with pytest.raises(ValueError):
        model(torch.rand(1, 3, 32, 32), reliability=torch.tensor([1.1]))
