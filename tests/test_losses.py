import pytest
import torch
import torch.nn.functional as F
from torch.nn.attention import SDPBackend, sdpa_kernel

from nightgan.config import ModelConfig
from nightgan.discriminator import Discriminator, real_pyramid
from nightgan.losses import finite_diff_penalty, repa_loss, rpgan_d, rpgan_g, stage_consistency


def test_rpgan_losses_are_softplus_of_the_logit_differences():
    real = torch.tensor([[1.0, 0.0], [2.0, -1.0]])
    fake = torch.tensor([[0.0, 0.5], [1.0, 1.0]])
    assert rpgan_d(real, fake).item() == pytest.approx(F.softplus(fake - real).mean().item())
    assert rpgan_g(real, fake).item() == pytest.approx(F.softplus(real - fake).mean().item())
    # only differences matter: shifting all logits changes nothing
    assert rpgan_d(real + 5, fake + 5).item() == pytest.approx(rpgan_d(real, fake).item())


def test_finite_difference_penalty_of_a_linear_critic_is_the_squared_weight_norm():
    torch.manual_seed(0)
    w = torch.randn(10, 1)
    x = torch.randn(20000, 10)
    sigma = 0.01
    clean, noisy = x @ w, (x + sigma * torch.randn_like(x)) @ w
    pen = finite_diff_penalty(clean, noisy, sigma)
    assert pen.shape == (20000,)
    assert pen.mean().item() == pytest.approx(w.square().sum().item(), rel=0.05)


def test_finite_difference_penalty_matches_r1_of_the_discriminator():
    torch.manual_seed(0)
    cfg = ModelConfig(latent_channels=4, latent_res=16, patch=1, dim=32, depth=4, heads=2, z_dim=16, repa_dim=0,
                      cmap_dim=32)
    D = Discriminator(cfg).double()
    with torch.no_grad():  # a less trivial critic than at initialization
        for blk in D.blocks:
            blk.ls1.fill_(0.5)
            blk.ls2.fill_(0.5)
    scales = [s.double() for s in real_pyramid(torch.randn(256, 4, 16, 16), 4)]
    inputs = [s.clone().requires_grad_(True) for s in scales]
    with sdpa_kernel(SDPBackend.MATH):
        logits, _ = D(inputs)
        # logit k depends only on scale k, so one backward pass gives every per-scale gradient
        grads = torch.autograd.grad(logits.sum(), inputs)
        r1 = sum(g.flatten(1).square().sum(1) for g in grads) / 4
    sigma = 0.01
    with torch.no_grad():
        clean, _ = D(scales)
        noisy, _ = D([s + sigma * torch.randn_like(s) for s in scales])
    pen = finite_diff_penalty(clean, noisy, sigma)
    assert pen.mean().item() == pytest.approx(r1.mean().item(), rel=0.15)


def test_stage_consistency_is_zero_for_equal_stages():
    last = torch.randn(2, 4, 8, 8)
    assert stage_consistency([last.clone() for _ in range(4)]).item() == 0.0
    stages = [torch.zeros(2, 4, 8, 8) for _ in range(3)] + [last]
    expected = sum(w * last.square().mean() for w in (1 / 3, 1 / 2, 1)) / 3
    assert stage_consistency(stages).item() == pytest.approx(expected.item())


def test_stage_consistency_of_accumulated_outputs_converges_without_drift():
    # the generator stages are partial sums of head outputs: h_k = o_1 + ... + o_k
    torch.manual_seed(0)
    heads = [torch.randn(1, 4, 8, 8, requires_grad=True) for _ in range(4)]
    def stages():
        out, acc = [], 0
        for o in heads:
            acc = acc + o
            out.append(acc)
        return out
    start_loss = stage_consistency(stages()).item()
    start_norm = stages()[-1].norm().item()
    opt = torch.optim.SGD(heads, lr=100.0)  # the loss is a mean over 256 elements
    for _ in range(300):
        opt.zero_grad()
        stage_consistency(stages()).backward()
        opt.step()
    assert stage_consistency(stages()).item() < 0.01 * start_loss
    assert stages()[-1].norm().item() < 1.5 * start_norm


def test_repa_loss_is_one_minus_cosine():
    cls, tok = torch.randn(3, 8), torch.randn(3, 5, 8)
    assert repa_loss(cls, tok, cls, tok).item() == pytest.approx(0.0, abs=1e-6)
    assert repa_loss(cls, tok, -cls, -tok).item() == pytest.approx(4.0)
    assert repa_loss(cls, tok, 2 * cls, 3 * tok).item() == pytest.approx(0.0, abs=1e-6)
