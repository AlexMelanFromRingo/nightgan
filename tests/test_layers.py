import math

import pytest
import torch
import torch.nn.functional as F

from nightgan.layers import (Attention, EqualLinear, RMSNorm, SwiGLU, apply_rope, grid_coords, patchify,
                             rope_freqs, sincos_2d, unpatchify)

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'


def test_rmsnorm_keeps_bf16_under_autocast_and_matches_fp32():
    norm = RMSNorm(64).to(DEVICE)
    with torch.no_grad():
        norm.weight.uniform_(0.5, 1.5)
    x = torch.randn(4, 10, 64, device=DEVICE)
    with torch.autocast(DEVICE, torch.bfloat16):
        y = norm(x.bfloat16())
    assert y.dtype == torch.bfloat16
    ref = x * torch.rsqrt(x.square().mean(-1, keepdim=True) + 1e-6) * norm.weight
    torch.testing.assert_close(y.float(), ref, atol=2e-2, rtol=2e-2)


def test_rmsnorm_without_affine_has_no_parameters():
    assert len(list(RMSNorm(8, affine=False).parameters())) == 0


def test_split_attention_equals_block_diagonal_masked_attention():
    torch.manual_seed(0)
    attn = Attention(32, heads=4).double()
    x = torch.randn(2, 7, 32, dtype=torch.float64)
    splits = [5, 2]
    mask = torch.zeros(7, 7, dtype=torch.bool)
    mask[:5, :5] = True
    mask[5:, 5:] = True
    out = attn(x, splits=splits)

    b, n, d = x.shape
    q, k, v = attn.qkv(x).view(b, n, 3, 4, d // 4).permute(2, 0, 3, 1, 4)
    q, k = attn.q_norm(q), attn.k_norm(k)
    ref = F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
    ref = attn.proj(ref.transpose(1, 2).reshape(b, n, d))
    torch.testing.assert_close(out, ref)


def test_split_attention_segments_are_independent():
    torch.manual_seed(0)
    attn = Attention(16, heads=2)
    x = torch.randn(1, 6, 16)
    y = x.clone()
    y[:, 4:] += 1.0  # change only the second segment
    out_x, out_y = attn(x, splits=[4, 2]), attn(y, splits=[4, 2])
    torch.testing.assert_close(out_x[:, :4], out_y[:, :4])
    assert not torch.allclose(out_x[:, 4:], out_y[:, 4:])


def test_rope_preserves_norm_and_depends_on_relative_position():
    torch.manual_seed(0)
    hd = 16
    coords = torch.tensor([[0.0, 0.0], [2.0, 3.0], [5.0, 1.0], [7.0, 4.0]])
    cos, sin = rope_freqs(coords, hd)
    assert cos.shape == sin.shape == (4, hd // 2)
    q, k = torch.randn(hd), torch.randn(hd)
    rot = lambda v, i: apply_rope(v.view(1, hd), cos[i:i + 1], sin[i:i + 1]).view(hd)
    assert rot(q, 1).norm() == pytest.approx(q.norm().item(), rel=1e-5)
    # <R(p1) q, R(p2) k> == <R(p1 + d) q, R(p2 + d) k> with d = (5, 1)
    assert torch.dot(rot(q, 0), rot(k, 1)) == pytest.approx(torch.dot(rot(q, 2), rot(k, 3)).item(), rel=1e-4)
    # position (0, 0) is the identity
    torch.testing.assert_close(rot(q, 0), q)


def test_grid_coords_are_token_centres_in_finest_grid_units():
    fine = grid_coords(4, scale=1)
    assert fine.shape == (16, 2)
    torch.testing.assert_close(fine[:4], torch.tensor([[0., 0.], [0., 1.], [0., 2.], [0., 3.]]))
    coarse = grid_coords(2, scale=2)  # each token covers 2x2 fine tokens
    torch.testing.assert_close(coarse, torch.tensor([[0.5, 0.5], [0.5, 2.5], [2.5, 0.5], [2.5, 2.5]]))


def test_patchify_unpatchify_round_trip():
    x = torch.randn(2, 4, 8, 8)
    t = patchify(x, 2)
    assert t.shape == (2, 16, 16)
    torch.testing.assert_close(t[0, 0], x[0, :, :2, :2].permute(1, 2, 0).reshape(-1))
    torch.testing.assert_close(unpatchify(t, 2, 4, 4), x)


def test_equal_linear_scales_weight_and_bias_by_lr_mult():
    lin = EqualLinear(8, 4, lr_mult=0.01)
    with torch.no_grad():
        lin.bias.fill_(3.0)
    x = torch.randn(5, 8)
    torch.testing.assert_close(lin(x), F.linear(x, lin.weight * 0.01, lin.bias * 0.01))
    # the effective weight has the variance of a plain xavier initialization
    big = EqualLinear(512, 512, lr_mult=0.01)
    assert (big.weight * 0.01).std().item() == pytest.approx(math.sqrt(2 / 1024), rel=0.05)


def test_swiglu_shapes():
    assert SwiGLU(32, 48)(torch.randn(3, 5, 32)).shape == (3, 5, 32)


def test_sincos_2d_is_bounded_and_distinct():
    pe = sincos_2d(64, 16)
    assert pe.shape == (256, 64)
    assert pe.abs().max() <= 1.0
    assert torch.unique(pe, dim=0).shape[0] == 256
