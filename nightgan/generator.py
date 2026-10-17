"""Transformer generator with mapping-network modulation and four accumulated output stages (GAT, CAT)."""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .config import ModelConfig
from .layers import (Attention, EqualLinear, RMSNorm, SwiGLU, grid_coords, init_linear, normalize_2nd_moment,
                     rope_freqs, sincos_2d, swiglu_hidden, unpatchify)

MODULATION_GAIN = 0.1   # initial scale of the modulation weights (GAT: variance 0.1)
C_AVG_BETA = 0.995      # running average of the mapping output, used by the truncation trick


class GeneratorBlock(nn.Module):
    """Pre-norm transformer block; the mapping output scales the normalized input and gates the residual branch."""

    def __init__(self, dim: int, heads: int, hidden: int):
        super().__init__()
        self.norm1 = RMSNorm(dim, affine=False)
        self.attn = Attention(dim, heads)
        self.norm2 = RMSNorm(dim, affine=False)
        self.mlp = SwiGLU(dim, hidden)
        self.modulation = nn.Linear(dim, 4 * dim)

    def forward(self, x, c, cos, sin):
        scale1, gate1, scale2, gate2 = self.modulation(F.silu(c)).unsqueeze(1).chunk(4, dim=-1)
        x = x + gate1 * self.attn(self.norm1(x) * (1 + scale1), rope=(cos, sin))
        return x + gate2 * self.mlp(self.norm2(x) * (1 + scale2))


class OutputHead(nn.Module):
    def __init__(self, dim: int, out_dim: int):
        super().__init__()
        self.norm = RMSNorm(dim, affine=False)
        self.modulation = nn.Linear(dim, dim)
        self.linear = nn.Linear(dim, out_dim)

    def forward(self, x, c):
        return self.linear(self.norm(x) * (1 + self.modulation(F.silu(c)).unsqueeze(1)))


class Generator(nn.Module):
    """Maps a latent `z` (and a class) to a `C x R x R` latent image.

    The token sequence starts from fixed position embeddings; `z` acts only through the modulation of every block.
    Four output heads after each quarter of the blocks add up to the image (output skip connections), the partial sums
    are the stage outputs the discriminator judges at decreasing resolutions."""

    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.cfg = cfg
        self.checkpointing = False  # recompute the blocks in the backward pass (a training option)
        dim, side = cfg.dim, cfg.tokens_side
        self.register_buffer('pos', sincos_2d(dim, side)[None], persistent=False)
        cos, sin = rope_freqs(grid_coords(side), dim // cfg.heads)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)

        n_classes = max(cfg.num_classes, 1)
        self.class_embed = nn.Embedding(n_classes, cfg.z_dim)
        self.mapping = nn.Sequential(EqualLinear(2 * cfg.z_dim, dim, lr_mult=0.01), nn.SiLU(),
                                     EqualLinear(dim, dim, lr_mult=0.01))
        hidden = swiglu_hidden(dim, cfg.mlp_ratio)
        self.blocks = nn.ModuleList([GeneratorBlock(dim, cfg.heads, hidden) for _ in range(cfg.depth)])
        step = cfg.depth // 4
        self.stage_blocks = [(k + 1) * step - 1 for k in range(4)]
        self.heads = nn.ModuleList([OutputHead(dim, cfg.patch ** 2 * cfg.latent_channels) for _ in range(4)])
        self.register_buffer('c_avg', torch.zeros(n_classes, dim))

        init_linear(self)
        nn.init.normal_(self.class_embed.weight, std=0.02)
        for m in [b.modulation for b in self.blocks] + [h.modulation for h in self.heads]:
            nn.init.xavier_uniform_(m.weight, gain=MODULATION_GAIN)
            nn.init.zeros_(m.bias)

    def map(self, z, y=None):
        """Mapping network: `z [B, z_dim]` and class indices `[B]` → modulation vector `c [B, dim]`."""
        if y is None:
            y = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        e = self.class_embed(y)
        return self.mapping(torch.cat([normalize_2nd_moment(z), normalize_2nd_moment(e)], dim=1))

    @torch.no_grad()
    def _update_c_avg(self, c, y):
        n = self.c_avg.shape[0]
        sums = torch.zeros_like(self.c_avg).index_add_(0, y, c.float())
        counts = torch.bincount(y, minlength=n).to(sums.dtype)
        present = counts > 0
        means = sums[present] / counts[present, None]
        self.c_avg[present] = means.lerp(self.c_avg[present], C_AVG_BETA)

    def forward(self, z, y=None, truncation: float = 1.0, return_stages: bool = False, update_avg: bool = False):
        """`truncation` follows StyleGAN: 1 is no truncation, 0 gives the average image of the class."""
        if y is None:
            y = torch.zeros(z.shape[0], dtype=torch.long, device=z.device)
        c = self.map(z, y)
        if update_avg:
            self._update_c_avg(c.detach(), y)
        if truncation != 1.0:
            avg = self.c_avg[y].to(c.dtype)
            c = avg + truncation * (c - avg)

        cfg = self.cfg
        x = self.pos.expand(z.shape[0], -1, -1)
        acc, stages, head = None, [], 0
        for i, block in enumerate(self.blocks):
            if self.checkpointing and torch.is_grad_enabled():
                x = checkpoint(block, x, c, self.rope_cos, self.rope_sin, use_reentrant=False)
            else:
                x = block(x, c, self.rope_cos, self.rope_sin)
            if i in self.stage_blocks:
                out = self.heads[head](x, c).float()
                acc = out if acc is None else acc + out
                stages.append(unpatchify(acc, cfg.patch, cfg.latent_channels, cfg.tokens_side))
                head += 1
        return stages if return_stages else stages[-1]
