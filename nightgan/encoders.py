"""Frozen image encoders that provide the REPA targets: DINOv2, or a fixed random encoder for offline tests."""

import warnings

import torch
import torch.nn as nn
import torch.nn.functional as F

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)
INPUT_SIZE = 224  # 16 x 16 patches of 14 pixels


def _prepare(images: torch.Tensor) -> torch.Tensor:
    x = images.float() / 255
    if x.shape[-1] != INPUT_SIZE:
        x = F.interpolate(x, size=(INPUT_SIZE, INPUT_SIZE), mode='bicubic', align_corners=False)
    mean = torch.tensor(IMAGENET_MEAN, device=x.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, device=x.device).view(1, 3, 1, 1)
    return (x - mean) / std


class DinoV2:
    """DINOv2 ViT/14 from torch.hub. `__call__(uint8 images [B, 3, H, W])` → ([cls] `[B, D]`, patches `[B, 256, D]`)."""

    def __init__(self, name: str = 'dinov2_vitb14', device='cuda', bf16: bool = True):
        self.device = torch.device(device)
        self.bf16 = bf16 and self.device.type == 'cuda'
        with warnings.catch_warnings():  # the hub code reports that the optional xFormers is missing
            warnings.filterwarnings('ignore', message='xFormers is not available')
            self.model = torch.hub.load('facebookresearch/dinov2', name, skip_validation=True, trust_repo=True)
        self.model = self.model.to(self.device).eval().requires_grad_(False)
        self.dim = self.model.embed_dim

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor):
        x = _prepare(images.to(self.device))
        with torch.autocast(self.device.type, torch.bfloat16, enabled=self.bf16):
            out = self.model.forward_features(x)
        return out['x_norm_clstoken'].float(), out['x_norm_patchtokens'].float()


class RandomConvEncoder(nn.Module):
    """A fixed random 14x14 patch embedding with the output format of `DinoV2`, for tests without downloads."""

    def __init__(self, dim: int = 64, seed: int = 0):
        super().__init__()
        self.dim = dim
        gen = torch.Generator().manual_seed(seed)
        self.register_buffer('weight', torch.randn(dim, 3, 14, 14, generator=gen) / 42)

    @torch.inference_mode()
    def __call__(self, images: torch.Tensor):
        x = _prepare(images.to(self.weight.device))
        tokens = F.conv2d(x, self.weight, stride=14).flatten(2).transpose(1, 2)
        return tokens.mean(dim=1), tokens


def load_encoder(name: str, device='cuda'):
    """`'none'` → None (no REPA), `'random'` → `RandomConvEncoder`, else a DINOv2 model name from torch.hub."""
    if name == 'none':
        return None
    if name == 'random':
        return RandomConvEncoder().to(device)
    return DinoV2(name, device)
