"""Exponential moving average of the generator weights, parameterized by a half-life in images."""

import copy

import torch


def ema_decay(batch: int, cur_nimg: int, halflife_kimg: float, rampup: float | None) -> float:
    """Decay per optimizer step of `batch` images. With `rampup`, the half-life is at most `rampup * cur_nimg`, so
    early in training the average follows the quickly changing weights (StyleGAN)."""
    halflife = halflife_kimg * 1000
    if rampup is not None:
        halflife = min(halflife, cur_nimg * rampup)
    if halflife <= 0:
        return 0.0
    return 0.5 ** (batch / halflife)


class EMA:
    def __init__(self, model: torch.nn.Module):
        self.model = copy.deepcopy(model).eval().requires_grad_(False)

    @torch.no_grad()
    def update(self, model: torch.nn.Module, decay: float):
        ema_params = list(self.model.parameters())
        params = [p.detach() for p in model.parameters()]
        torch._foreach_lerp_(ema_params, params, 1.0 - decay)
        for b_ema, b in zip(self.model.buffers(), model.buffers()):
            b_ema.copy_(b)

    def state_dict(self):
        return self.model.state_dict()

    def load_state_dict(self, state):
        self.model.load_state_dict(state)
