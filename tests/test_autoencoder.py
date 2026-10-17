import math

import pytest
import torch

from nightgan.autoencoder import LatentNorm, PixelAE, load_autoencoder, sample_latent, to_uint8
from nightgan.encoders import RandomConvEncoder, load_encoder

CUDA = torch.cuda.is_available()


def test_pixel_autoencoder_round_trip_is_exact():
    ae = PixelAE()
    img = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
    moments = ae.encode_moments(img)
    assert moments.shape == (2, 6, 16, 16) and moments.dtype == torch.float32
    assert torch.equal(to_uint8(ae.decode(sample_latent(moments))), img)
    assert (ae.channels, ae.factor, ae.name) == (3, 1, 'pixel')


def test_sample_latent_with_tiny_variance_is_the_mean():
    mean = torch.randn(2, 4, 8, 8)
    moments = torch.cat([mean, torch.full_like(mean, -30.0)], dim=1)
    torch.testing.assert_close(sample_latent(moments), mean, atol=1e-5, rtol=0)


def test_sample_latent_uses_the_variance():
    torch.manual_seed(0)
    moments = torch.cat([torch.zeros(1, 4, 64, 64), torch.full((1, 4, 64, 64), math.log(4.0))], dim=1)
    assert sample_latent(moments).std().item() == pytest.approx(2.0, rel=0.05)


def test_latent_norm_is_invertible():
    norm = LatentNorm(torch.tensor([1.0, -2.0, 0.5]), torch.tensor([2.0, 0.5, 1.0]))
    z = torch.randn(2, 3, 4, 4)
    torch.testing.assert_close(norm.denormalize(norm.normalize(z)), z)
    torch.testing.assert_close(norm.normalize(torch.tensor([1.0, -2.0, 0.5]).view(1, 3, 1, 1)), torch.zeros(1, 3, 1, 1))


def test_to_uint8_clamps():
    assert to_uint8(torch.tensor([-2.0, -1.0, 0.0, 1.0, 3.0])).tolist() == [0, 0, 128, 255, 255]


def test_random_conv_encoder_is_deterministic_and_dino_shaped():
    a, b = RandomConvEncoder(dim=32), RandomConvEncoder(dim=32)
    img = torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8)
    cls, tokens = a(img)
    assert cls.shape == (2, 32) and tokens.shape == (2, 256, 32)
    torch.testing.assert_close(b(img)[1], tokens)
    assert load_encoder('none', 'cpu') is None
    assert isinstance(load_encoder('random', 'cpu'), RandomConvEncoder)


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason='needs a GPU')
def test_sd_vae_reconstructs_a_smooth_image():
    ae = load_autoencoder('stabilityai/sd-vae-ft-mse', 'cuda')
    assert (ae.channels, ae.factor) == (4, 8)
    yy, xx = torch.meshgrid(torch.linspace(0, 1, 256), torch.linspace(0, 1, 256), indexing='ij')
    img = torch.stack([xx, yy, (xx + yy) / 2]) * 255
    img = img.round().to(torch.uint8)[None].cuda()
    moments = ae.encode_moments(img)
    assert moments.shape == (1, 8, 32, 32)
    rec = to_uint8(ae.decode(moments[:, :4])).float()
    mse = (rec - img.float()).square().mean().item()
    assert 10 * math.log10(255 ** 2 / mse) > 25


@pytest.mark.slow
@pytest.mark.skipif(not CUDA, reason='needs a GPU')
def test_dinov2_features():
    enc = load_encoder('dinov2_vitb14', 'cuda')
    cls, tokens = enc(torch.randint(0, 256, (2, 3, 224, 224), dtype=torch.uint8, device='cuda'))
    assert cls.shape == (2, 768) and tokens.shape == (2, 256, 768) and enc.dim == 768
