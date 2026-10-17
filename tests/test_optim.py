import copy

import pytest
import torch

from nightgan.optim import AdaBelief, make_optimizer


def test_adabelief_follows_the_published_update():
    # Zhuang et al. 2020, Algorithm 2, with bias correction and decoupled weight decay 0
    p = torch.nn.Parameter(torch.tensor([1.0, -2.0]))
    opt = AdaBelief([p], lr=0.1, betas=(0.5, 0.9), eps=1e-8)
    grads = [torch.tensor([0.3, -0.1]), torch.tensor([0.2, 0.4])]
    m = torch.zeros(2)
    s = torch.zeros(2)
    ref = p.detach().clone()
    for t, g in enumerate(grads, start=1):
        p.grad = g.clone()
        opt.step()
        m = 0.5 * m + 0.5 * g
        s = 0.9 * s + 0.1 * (g - m) ** 2 + 1e-8
        m_hat, s_hat = m / (1 - 0.5 ** t), s / (1 - 0.9 ** t)
        ref = ref - 0.1 * m_hat / (s_hat.sqrt() + 1e-8)
        torch.testing.assert_close(p.detach(), ref)


def test_adabelief_minimizes_a_quadratic():
    torch.manual_seed(0)
    target = torch.randn(10)
    p = torch.nn.Parameter(torch.zeros(10))
    opt = AdaBelief([p], lr=0.05, betas=(0.5, 0.99))
    for _ in range(500):
        opt.zero_grad()
        ((p - target) ** 2).sum().backward()
        opt.step()
    assert (p.detach() - target).abs().max() < 1e-2


def test_adabelief_needs_momentum():
    p = torch.nn.Parameter(torch.zeros(2))
    with pytest.raises(ValueError, match='beta1'):
        AdaBelief([p], lr=1e-3, betas=(0.0, 0.99))


def test_make_optimizer():
    p = [torch.nn.Parameter(torch.zeros(2))]
    assert isinstance(make_optimizer('adamw', p, 1e-3, (0.0, 0.99), fused=False), torch.optim.AdamW)
    assert isinstance(make_optimizer('adabelief', p, 1e-3, (0.5, 0.99), fused=False), AdaBelief)
    with pytest.raises(ValueError, match='sgd'):
        make_optimizer('sgd', p, 1e-3, (0.0, 0.99), fused=False)


def test_adabelief_state_round_trips():
    p = torch.nn.Parameter(torch.ones(3))
    opt = AdaBelief([p], lr=0.1, betas=(0.5, 0.99))
    p.grad = torch.ones(3)
    opt.step()
    state = copy.deepcopy(opt.state_dict())   # as a snapshot on disk: no tensors shared with `opt`
    q = torch.nn.Parameter(p.detach().clone())
    opt2 = AdaBelief([q], lr=0.1, betas=(0.5, 0.99))
    opt2.load_state_dict(state)
    p.grad = q.grad = torch.full((3,), 0.5)
    opt.step()
    opt2.step()
    torch.testing.assert_close(p.detach(), q.detach())
