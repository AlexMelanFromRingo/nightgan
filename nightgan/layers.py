"""Transformer building blocks shared by the generator and the discriminator."""


import torch
import torch.nn as nn
import torch.nn.functional as F


class RMSNorm(nn.Module):
    """RMS normalization that returns the dtype of its input.

    Under autocast, `F.rms_norm` and `nn.RMSNorm` return float32, which doubles the memory of the activations saved for
    the backward pass of a bf16 network. The statistics are still accumulated in float32 by the kernel."""

    def __init__(self, dim: int, affine: bool = True, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim)) if affine else None

    def forward(self, x):
        weight = None if self.weight is None else self.weight.to(x.dtype)
        with torch.autocast(x.device.type, enabled=False):
            return F.rms_norm(x, (x.shape[-1],), weight, self.eps)


def grid_coords(side: int, scale: float = 1.0) -> torch.Tensor:
    """Centres of the tokens of a `side x side` grid in units of the finest grid, `[side * side, 2]` as (row, col).

    A token of a grid that is `scale` times coarser covers `scale x scale` fine tokens, its centre is the mean of
    theirs, so the rotary embeddings of all discriminator scales live in the same coordinate frame."""
    idx = (torch.arange(side, dtype=torch.float32) + 0.5) * scale - 0.5
    rows, cols = torch.meshgrid(idx, idx, indexing='ij')
    return torch.stack([rows.flatten(), cols.flatten()], dim=1)


def rope_freqs(coords: torch.Tensor, head_dim: int, theta: float = 100.0):
    """2D rotary embedding angles: half of the rotated pairs encode the row, half the column.

    Returns `(cos, sin)`, each `[N, head_dim // 2]`. Coordinates (0, 0) give the identity rotation."""
    if head_dim % 4:
        raise ValueError(f'head size {head_dim} has to be a multiple of 4 for 2D rotary embeddings')
    n = head_dim // 4
    freqs = theta ** (-torch.arange(n, dtype=torch.float32) / n)
    angles = torch.cat([coords[:, :1] * freqs, coords[:, 1:] * freqs], dim=1)
    return angles.cos(), angles.sin()


def apply_rope(x: torch.Tensor, cos: torch.Tensor, sin: torch.Tensor) -> torch.Tensor:
    """Rotates pairs `(x[..., i], x[..., i + head_dim / 2])` of `x[..., N, head_dim]`."""
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x.chunk(2, dim=-1)
    return torch.cat([x1 * cos - x2 * sin, x1 * sin + x2 * cos], dim=-1)


class Attention(nn.Module):
    """Multi-head self-attention with qk-normalization and optional 2D rotary embeddings.

    `splits` cuts the sequence into segments that attend only within themselves (block diagonal attention), each
    segment is a separate fused attention call instead of one masked call."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        if dim % heads:
            raise ValueError(f'width {dim} is not divisible by {heads} heads')
        self.heads = heads
        self.qkv = nn.Linear(dim, 3 * dim)
        self.q_norm = RMSNorm(dim // heads)
        self.k_norm = RMSNorm(dim // heads)
        self.proj = nn.Linear(dim, dim)

    def forward(self, x, rope=None, splits=None):
        b, n, d = x.shape
        q, k, v = self.qkv(x).view(b, n, 3, self.heads, d // self.heads).permute(2, 0, 3, 1, 4)
        q, k = self.q_norm(q), self.k_norm(k)
        if rope is not None:
            q, k = apply_rope(q, *rope), apply_rope(k, *rope)
        if splits is None:
            out = F.scaled_dot_product_attention(q, k, v)
        else:
            out = torch.cat([F.scaled_dot_product_attention(qs, ks, vs) for qs, ks, vs in
                             zip(q.split(splits, 2), k.split(splits, 2), v.split(splits, 2))], dim=2)
        return self.proj(out.transpose(1, 2).reshape(b, n, d))


class SwiGLU(nn.Module):
    def __init__(self, dim: int, hidden: int):
        super().__init__()
        self.w12 = nn.Linear(dim, 2 * hidden)
        self.w3 = nn.Linear(hidden, dim)

    def forward(self, x):
        a, b = self.w12(x).chunk(2, dim=-1)
        return self.w3(F.silu(a) * b)


def swiglu_hidden(dim: int, mlp_ratio: float) -> int:
    """Hidden size of a SwiGLU with the parameter count of a `mlp_ratio` MLP, rounded to a multiple of 8."""
    return int(round(2 / 3 * mlp_ratio * dim / 8)) * 8


class EqualLinear(nn.Module):
    """Linear layer with a learning rate multiplier (equalized learning rate, StyleGAN): the stored parameters are
    scaled by `1 / lr_mult`, so Adam moves the effective weights `lr_mult` times slower."""

    def __init__(self, in_dim: int, out_dim: int, lr_mult: float = 1.0, bias: bool = True):
        super().__init__()
        self.lr_mult = lr_mult
        self.weight = nn.Parameter(torch.empty(out_dim, in_dim))
        nn.init.xavier_uniform_(self.weight, gain=1 / lr_mult)
        self.bias = nn.Parameter(torch.zeros(out_dim)) if bias else None

    def forward(self, x):
        bias = None if self.bias is None else self.bias * self.lr_mult
        return F.linear(x, self.weight * self.lr_mult, bias)


def sincos_2d(dim: int, side: int) -> torch.Tensor:
    """Fixed 2D sine-cosine position embedding, `[side * side, dim]`: half of the channels encode the row."""
    if dim % 4:
        raise ValueError(f'width {dim} has to be a multiple of 4 for 2D sine-cosine embeddings')
    n = dim // 4
    omega = 1.0 / 10000 ** (torch.arange(n, dtype=torch.float64) / n)
    coords = grid_coords(side).double()
    parts = []
    for axis in range(2):
        angles = coords[:, axis:axis + 1] * omega
        parts += [angles.sin(), angles.cos()]
    return torch.cat(parts, dim=1).float()


def patchify(x: torch.Tensor, p: int) -> torch.Tensor:
    """`[B, C, H, W]` → `[B, (H / p) * (W / p), p * p * C]`, tokens in row-major order."""
    b, c, h, w = x.shape
    x = x.view(b, c, h // p, p, w // p, p).permute(0, 2, 4, 3, 5, 1)
    return x.reshape(b, (h // p) * (w // p), p * p * c)


def unpatchify(t: torch.Tensor, p: int, c: int, side: int) -> torch.Tensor:
    """Inverse of `patchify` for a square `side x side` token grid."""
    b = t.shape[0]
    x = t.view(b, side, side, p, p, c).permute(0, 5, 1, 3, 2, 4)
    return x.reshape(b, c, side * p, side * p)


def init_linear(module: nn.Module):
    """Xavier-uniform weights and zero biases for every `nn.Linear` in `module`."""
    for m in module.modules():
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if m.bias is not None:
                nn.init.zeros_(m.bias)


def normalize_2nd_moment(x: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    return x * (x.square().mean(dim=-1, keepdim=True) + eps).rsqrt()


__all__ = ['RMSNorm', 'grid_coords', 'rope_freqs', 'apply_rope', 'Attention', 'SwiGLU', 'swiglu_hidden',
           'EqualLinear', 'sincos_2d', 'patchify', 'unpatchify', 'init_linear', 'normalize_2nd_moment']
