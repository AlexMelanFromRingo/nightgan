import pytest
import torch

from nightgan.config import ModelConfig
from nightgan.generator import Generator


def tiny(**kw):
    base = dict(latent_channels=4, latent_res=32, patch=2, dim=32, depth=4, heads=2, z_dim=16)
    return ModelConfig(**{**base, **kw})


@pytest.mark.parametrize('latent_res, patch, channels', [(16, 1, 4), (32, 2, 4), (64, 4, 4), (32, 2, 3)])
def test_output_shape(latent_res, patch, channels):
    G = Generator(tiny(latent_res=latent_res, patch=patch, latent_channels=channels))
    out = G(torch.randn(3, 16))
    assert out.shape == (3, channels, latent_res, latent_res)
    assert out.dtype == torch.float32


def test_four_accumulated_stages_and_the_last_is_the_output():
    torch.manual_seed(0)
    G = Generator(tiny())
    z = torch.randn(2, 16)
    stages = G(z, return_stages=True)
    assert len(stages) == 4
    assert all(s.shape == (2, 4, 32, 32) for s in stages)
    torch.testing.assert_close(stages[-1], G(z))
    assert G.stage_blocks == [0, 1, 2, 3]
    assert Generator(tiny(depth=12)).stage_blocks == [2, 5, 8, 11]


def test_different_latents_give_different_images_at_init():
    torch.manual_seed(0)
    G = Generator(tiny())
    out = G(torch.randn(16, 16))
    assert out.std(dim=0).mean() > 1e-3


def test_class_conditioning_changes_the_output():
    torch.manual_seed(0)
    G = Generator(tiny(num_classes=3))
    z = torch.randn(2, 16)
    a = G(z, torch.tensor([0, 0]))
    b = G(z, torch.tensor([2, 2]))
    assert (a - b).abs().mean() > 1e-4


def test_truncation_zero_gives_the_class_average_for_every_latent():
    torch.manual_seed(0)
    G = Generator(tiny(num_classes=2))
    with torch.no_grad():
        G(torch.randn(64, 16), torch.randint(0, 2, (64,)), update_avg=True)
    out = G(torch.randn(4, 16), torch.tensor([1, 1, 1, 1]), truncation=0.0)
    torch.testing.assert_close(out[0].expand_as(out), out)
    out_none = G(torch.randn(4, 16), torch.tensor([1, 1, 1, 1]), truncation=1.0)
    assert out_none.std(dim=0).mean() > 1e-4


def test_update_avg_moves_the_running_mean_of_the_class_only():
    torch.manual_seed(0)
    G = Generator(tiny(num_classes=3))
    assert G.c_avg.abs().sum() == 0
    with torch.no_grad():
        G(torch.randn(8, 16), torch.tensor([1] * 8), update_avg=True)
    assert G.c_avg[1].abs().sum() > 0
    assert G.c_avg[0].abs().sum() == 0 and G.c_avg[2].abs().sum() == 0


def test_gradient_checkpointing_gives_identical_gradients():
    torch.manual_seed(0)
    G = Generator(tiny())
    z = torch.randn(2, 16)
    G(z).square().mean().backward()
    ref = [p.grad.clone() for p in G.parameters()]
    G.zero_grad()
    G.cfg.checkpointing = True
    G(z).square().mean().backward()
    for p, r in zip(G.parameters(), ref):
        torch.testing.assert_close(p.grad, r)
