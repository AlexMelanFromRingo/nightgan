"""The training step must not synchronize the CPU with the GPU outside the one intended check per phase: every sync
stalls the CPU until the GPU is idle, and the GPU then waits for the CPU to queue new work."""

import pytest
import torch

from nightgan.augment import apply_aug, sample_aug, shift_images
from nightgan.config import ModelConfig
from nightgan.discriminator import Discriminator, pyramid, real_pyramid
from nightgan.generator import Generator

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')


def test_forward_and_backward_passes_do_not_synchronize():
    cfg = ModelConfig(latent_channels=4, latent_res=32, patch=2, dim=32, depth=4, heads=2, z_dim=16, repa_dim=24,
                      cmap_dim=32, num_classes=3)
    G, D = Generator(cfg).cuda(), Discriminator(cfg).cuda()
    z = torch.randn(4, 16, device='cuda')
    y = torch.randint(0, 3, (4,), device='cuda')
    real = torch.randn(4, 4, 32, 32, device='cuda')
    imgs = torch.randint(0, 256, (4, 3, 224, 224), dtype=torch.uint8, device='cuda')
    params = sample_aug(4, 32, 'color,translation,cutout', 1.0, 'cuda')
    torch.cuda.synchronize()
    torch.cuda.set_sync_debug_mode('error')
    try:
        with torch.autocast('cuda', torch.bfloat16):
            stages = G(z, y, return_stages=True, update_avg=True)
            logits, hidden = D(pyramid([apply_aug(s, params) for s in stages]), y)
            logits_real, _ = D(real_pyramid(apply_aug(real, params), 4), y)
            cls, tok = D.repa_project(hidden)
        (logits.sum() - logits_real.sum() + cls.sum() + tok.sum()).backward()
        shift_images(imgs, params, 32)
    finally:
        torch.cuda.set_sync_debug_mode('default')
