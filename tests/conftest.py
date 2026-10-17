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
