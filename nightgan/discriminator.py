"""Scale-wise transformer discriminator (CAT) with a projection head and a REPA projector (GAT)."""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .config import ModelConfig
from .layers import (Attention, EqualLinear, RMSNorm, SwiGLU, grid_coords, init_linear, patchify, rope_freqs,
                     swiglu_hidden)

LAYER_SCALE = 0.1
REPA_HIDDEN = 2048


def pyramid(stages: list[torch.Tensor]) -> list[torch.Tensor]:
    """Generator stage k (of S) is judged at 1 / 2^(S-1-k) of the full resolution (area downsampling)."""
    n = len(stages)
    return [F.avg_pool2d(s, 2 ** (n - 1 - k)) if k < n - 1 else s for k, s in enumerate(stages)]


def real_pyramid(x: torch.Tensor, num_scales: int = 4) -> list[torch.Tensor]:
    """The same resolutions as `pyramid`, all from one full resolution tensor."""
    return pyramid([x] * num_scales)


class DiscriminatorBlock(nn.Module):
    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim)
        self.mlp = SwiGLU(dim, hidden)
        self.ls1 = nn.Parameter(torch.full((dim,), LAYER_SCALE))
        self.ls2 = nn.Parameter(torch.full((dim,), LAYER_SCALE))

    def forward(self, x, cos, sin, splits):
        x = x + self.ls1 * self.attn(self.norm1(x), rope=(cos, sin), splits=splits)
        return x + self.ls2 * self.mlp(self.norm2(x))


class Discriminator(nn.Module):
    """Judges a latent at `num_scales` resolutions, one logit per scale.

    All scales share the patch embedding and the transformer blocks; each scale has its own learned embedding and
    [cls] token, and attends only within itself. Rotary embeddings use the coordinates of the finest grid, so a
    coarse token sits at the centre of the fine tokens it covers."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        dim, heads, n = cfg.D_dim, cfg.D_heads, cfg.num_scales
        side = cfg.tokens_side
        self.embed = nn.Linear(cfg.patch ** 2 * cfg.latent_channels, dim)
        self.scale_embed = nn.Parameter(torch.randn(n, dim) * 0.02)
        self.cls = nn.Parameter(torch.randn(n, dim) * 0.02)

        coords, splits = [], []
        for k in range(n):
            factor = 2 ** (n - 1 - k)
            coords += [torch.zeros(1, 2), grid_coords(side // factor, factor)]  # [cls] at the origin: no rotation
            splits.append(1 + (side // factor) ** 2)
        cos, sin = rope_freqs(torch.cat(coords), dim // heads)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.splits = splits
        self.starts = [sum(splits[:k]) for k in range(n)]  # position of each scale's [cls] token
        # indexing with a device tensor: a Python list would be copied to the GPU synchronously at every call
        self.register_buffer('cls_index', torch.tensor(self.starts), persistent=False)

        hidden = swiglu_hidden(dim, cfg.mlp_ratio)
        self.blocks = nn.ModuleList([DiscriminatorBlock(dim, heads, hidden) for _ in range(cfg.D_depth)])
        self.final_norm = RMSNorm(dim)
        self.final = nn.Linear(dim, cfg.cmap_dim)
        self.class_embed = nn.Embedding(max(cfg.num_classes, 1), cfg.cmap_dim)
        self.class_mapping = nn.Sequential(EqualLinear(cfg.cmap_dim, cfg.cmap_dim, lr_mult=0.01), nn.SiLU(),
                                           EqualLinear(cfg.cmap_dim, cfg.cmap_dim, lr_mult=0.01))
        self.projector = None
        if cfg.repa_dim:
            self.projector = nn.Sequential(nn.Linear(dim, REPA_HIDDEN), nn.SiLU(), nn.Linear(REPA_HIDDEN, REPA_HIDDEN),
                                           nn.SiLU(), nn.Linear(REPA_HIDDEN, cfg.repa_dim))
        init_linear(self)
        nn.init.normal_(self.class_embed.weight, std=0.02)

    def forward(self, scales: list[torch.Tensor], y=None):
        """`scales`: coarsest first, as given by `pyramid`. Returns logits `[B, num_scales]` and the last block's
        output for the finest scale, `[B, 1 + T*T, dim]` with the [cls] token first (input of `repa_project`)."""
        b = scales[0].shape[0]
        tokens = []
        for k, x in enumerate(scales):
            t = self.embed(patchify(x, self.cfg.patch)) + self.scale_embed[k]
            tokens += [self.cls[k].to(t.dtype).expand(b, 1, -1), t]
        x = torch.cat(tokens, dim=1)
        for block in self.blocks:
            x = block(x, self.rope_cos, self.rope_sin, self.splits)

        cls = x.index_select(1, self.cls_index)
        h = self.final(self.final_norm(cls))
        if y is None:
            y = torch.zeros(b, dtype=torch.long, device=x.device)
        e = self.class_mapping(self.class_embed(y))
        logits = (h.float() * e.float().unsqueeze(1)).sum(-1)
        return logits, x[:, self.starts[-1]:]

    def repa_project(self, hidden: torch.Tensor):
        """Projects the finest scale's [cls] and patch tokens to the feature size of the REPA encoder."""
        if self.projector is None:
            raise RuntimeError('this discriminator has no REPA projector (repa_dim = 0)')
        out = self.projector(hidden)
        return out[:, 0], out[:, 1:]
