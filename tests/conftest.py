import numpy as np
import pytest
import torch
from PIL import Image


def write_images(folder, n, size=32, seed=0, ext='png'):
    """Random smooth color images: a gradient plus a colored square, so that they are not all alike."""
    folder.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[0:size, 0:size] / size
    paths = []
    for i in range(n):
        img = np.stack([xx * rng.uniform(), yy * rng.uniform(), (xx + yy) / 2 * rng.uniform()], axis=-1)
        x0, y0 = rng.integers(0, size // 2, 2)
        img[y0:y0 + size // 4, x0:x0 + size // 4] = rng.uniform(size=3)
        path = folder / f'{i:04d}.{ext}'
        Image.fromarray((img * 255).astype(np.uint8)).save(path)
        paths.append(path)
    return paths


def mean_color(images: torch.Tensor) -> torch.Tensor:
    """A tiny stand-in for a feature extractor: the mean color of uint8 images `[B, 3, H, W]`."""
    return images.float().mean(dim=(2, 3)) / 255


@pytest.fixture
def image_folder(tmp_path):
    folder = tmp_path / 'images'
    write_images(folder, 24)
    return folder


def left_right_color(images: torch.Tensor) -> torch.Tensor:
    """Mean colors of the left and the right half: a stand-in feature extractor that sees mirroring."""
    w = images.shape[-1] // 2
    return torch.cat([mean_color(images[..., :w]), mean_color(images[..., w:])], dim=1)


class GrayAE:
    """A lossy stand-in autoencoder: one latent channel, the brightness, decoded back to gray RGB."""
    name = 'gray'
    channels = 1
    factor = 1

    def encode_moments(self, images: torch.Tensor) -> torch.Tensor:
        gray = images.float().mean(dim=1, keepdim=True) / 127.5 - 1
        return torch.cat([gray, torch.full_like(gray, -30.0)], dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.float().expand(-1, 3, -1, -1)
