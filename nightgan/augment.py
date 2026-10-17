"""Differentiable augmentation of normalized latents (DiffAugment as used by GAT).

The parameters are sampled once per sample and can be applied to several tensors (all generator stages of a sample get
the same augmentation). The translation is also applied, scaled to pixels, to the images the REPA targets are computed
from, so the targets stay aligned with what the discriminator sees."""

from dataclasses import dataclass

import torch
import torch.nn.functional as F

OPS = ('color', 'translation', 'cutout')
TRANSLATION_RATIO = 0.125
CUTOUT_RATIO = 0.5


@dataclass
class AugParams:
    brightness: torch.Tensor   # [B, 1, 1, 1], added; 0 = identity
    saturation: torch.Tensor   # [B, 1, 1, 1], factor around the channel mean; 1 = identity
    contrast: torch.Tensor     # [B, 1, 1, 1], factor around the sample mean; 1 = identity
    shift: torch.Tensor        # [B, 2] integer (rows, cols) in latent pixels; 0 = identity
    cutout: torch.Tensor       # [B, 2] integer centre of the zeroed square
    cutout_on: torch.Tensor    # [B] bool
    cutout_size: int
    ops: frozenset


def parse_policy(policy: str) -> frozenset:
    ops = frozenset(op.strip() for op in policy.split(',') if op.strip())
    unknown = sorted(ops - set(OPS))
    if unknown:
        raise ValueError(f'unknown augmentation {", ".join(unknown)}; available: {", ".join(OPS)}')
    return ops


def sample_aug(batch: int, side: int, policy: str, p: float, device, generator=None) -> AugParams:
    """Each operation of the policy is applied to each sample with probability `p`."""
    ops = parse_policy(policy)

    def rand(*shape):
        return torch.rand(*shape, device=device, generator=generator)

    def on(op):
        return ((rand(batch) < p) & (op in ops)).float()

    color = on('color').view(batch, 1, 1, 1)
    brightness = (rand(batch, 1, 1, 1) - 0.5) * color
    saturation = 1 + (rand(batch, 1, 1, 1) * 2 - 1) * color
    contrast = 1 + (rand(batch, 1, 1, 1) - 0.5) * color
    s = int(side * TRANSLATION_RATIO + 0.5)
    shift = torch.randint(-s, s + 1, (batch, 2), device=device, generator=generator) * on('translation')[:, None].long()
    size = int(side * CUTOUT_RATIO + 0.5)
    centre = torch.randint(0, side, (batch, 2), device=device, generator=generator)
    return AugParams(brightness, saturation, contrast, shift, centre, on('cutout').bool(), size, ops)


def translate(x: torch.Tensor, shift: torch.Tensor) -> torch.Tensor:
    """Moves the content of each sample by `shift` (rows, cols) pixels, filling with zeros."""
    b, c, h, w = x.shape
    rows = torch.arange(h, device=x.device).view(1, h, 1) - shift[:, 0].view(b, 1, 1)
    cols = torch.arange(w, device=x.device).view(1, 1, w) - shift[:, 1].view(b, 1, 1)
    padded = F.pad(x, [1, 1, 1, 1])  # out of range indices land on the zero border
    rows = (rows + 1).clamp(0, h + 1)
    cols = (cols + 1).clamp(0, w + 1)
    batch = torch.arange(b, device=x.device).view(b, 1, 1)
    return padded.permute(0, 2, 3, 1)[batch, rows, cols].permute(0, 3, 1, 2)


def apply_aug(x: torch.Tensor, params: AugParams) -> torch.Tensor:
    ops = params.ops
    if 'color' in ops:
        x = x + params.brightness.to(x.dtype)
        mean = x.mean(dim=1, keepdim=True)
        x = (x - mean) * params.saturation.to(x.dtype) + mean
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) * params.contrast.to(x.dtype) + mean
    if 'translation' in ops and params.shift.any():
        x = translate(x, params.shift)
    if 'cutout' in ops and params.cutout_on.any():
        b, _, h, w = x.shape
        size = params.cutout_size
        top = (params.cutout[:, 0] - size // 2).view(b, 1, 1)
        left = (params.cutout[:, 1] - size // 2).view(b, 1, 1)
        rows = torch.arange(h, device=x.device).view(1, h, 1)
        cols = torch.arange(w, device=x.device).view(1, 1, w)
        inside = (rows >= top) & (rows < top + size) & (cols >= left) & (cols < left + size)
        inside &= params.cutout_on.view(b, 1, 1)
        x = x * (~inside).unsqueeze(1).to(x.dtype)
    return x


def shift_images(images: torch.Tensor, params: AugParams, latent_side: int) -> torch.Tensor:
    """Applies the translation of `params` to images whose side is a multiple of the latent side (rounded otherwise)."""
    if 'translation' not in params.ops or not params.shift.any():
        return images
    factor = images.shape[-1] / latent_side
    shift = torch.round(params.shift.float() * factor).long()
    return translate(images, shift)
