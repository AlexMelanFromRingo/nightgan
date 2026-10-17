import pytest
import torch

from nightgan.config import ModelConfig
from nightgan.discriminator import Discriminator, pyramid, real_pyramid
from nightgan.generator import Generator


def tiny(**kw):
    base = dict(latent_channels=4, latent_res=32, patch=2, dim=32, depth=4, heads=2, z_dim=16, repa_dim=24,
                cmap_dim=64)
    return ModelConfig(**{**base, **kw})


def test_pyramid_resolutions_follow_the_stages():
    stages = [torch.randn(2, 4, 32, 32) for _ in range(4)]
    assert [s.shape[-1] for s in pyramid(stages)] == [4, 8, 16, 32]
    x = torch.randn(2, 4, 32, 32)
    real = real_pyramid(x, 4)
    assert [s.shape[-1] for s in real] == [4, 8, 16, 32]
    torch.testing.assert_close(real[0][0, 0, 0, 0], x[0, 0, :8, :8].mean())
    torch.testing.assert_close(real[-1], x)


@pytest.mark.parametrize('latent_res, patch', [(16, 1), (32, 2), (64, 4)])
def test_logits_and_finest_hidden_shapes(latent_res, patch):
    cfg = tiny(latent_res=latent_res, patch=patch)
    D = Discriminator(cfg)
    scales = real_pyramid(torch.randn(3, 4, latent_res, latent_res), 4)
    logits, hidden = D(scales)
    assert logits.shape == (3, 4)
    assert hidden.shape == (3, 1 + 16 * 16, 32)
    cls, tokens = D.repa_project(hidden)
    assert cls.shape == (3, 24) and tokens.shape == (3, 256, 24)


def test_scales_do_not_see_each_other():
    torch.manual_seed(0)
    D = Discriminator(tiny())
    scales = real_pyramid(torch.randn(2, 4, 32, 32), 4)
    a, _ = D(scales)
    changed = [scales[0] + 1.0] + scales[1:]
    b, _ = D(changed)
    torch.testing.assert_close(a[:, 1:], b[:, 1:], atol=1e-6, rtol=1e-6)
    assert not torch.allclose(a[:, 0], b[:, 0])


def test_class_conditional_logits_depend_on_the_class():
    torch.manual_seed(0)
    D = Discriminator(tiny(num_classes=3))
    with torch.no_grad():
        D.class_embed.weight.normal_()
    scales = real_pyramid(torch.randn(2, 4, 32, 32), 4)
    a, _ = D(scales, torch.tensor([0, 0]))
    b, _ = D(scales, torch.tensor([1, 1]))
    assert not torch.allclose(a, b)


def test_gradients_reach_every_scale_input_and_generator_stages():
    torch.manual_seed(0)
    cfg = tiny()
    G, D = Generator(cfg), Discriminator(cfg)
    stages = G(torch.randn(2, 16), return_stages=True)
    scales = [s.detach().requires_grad_(True) for s in pyramid(stages)]
    logits, _ = D(scales)
    logits.sum().backward()
    assert all(s.grad is not None and s.grad.abs().sum() > 0 for s in scales)


def test_no_projector_without_repa():
    D = Discriminator(tiny(repa_dim=0))
    assert D.projector is None
    with pytest.raises(RuntimeError, match='REPA'):
        D.repa_project(torch.randn(1, 257, 32))
