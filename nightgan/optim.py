"""Optimizers: AdamW (the default of the GAT / CAT / R3GAN recipes) and AdaBelief, for comparison."""

import torch


class AdaBelief(torch.optim.Optimizer):
    """AdaBelief (Zhuang et al., NeurIPS 2020): Adam whose second moment is the variance of the gradient around its
    running mean, `(g - m)^2`, instead of `g^2`, with bias correction and decoupled weight decay.

    It needs momentum: with `beta1 = 0` the mean is the gradient itself, `(g - m)^2` is zero and every step is divided
    by `sqrt(eps)`. The GAN recipes nightgan follows use Adam with `beta1 = 0`; the AdaBelief paper uses 0.5 for GANs."""

    def __init__(self, params, lr=1e-3, betas=(0.5, 0.999), eps=1e-16, weight_decay=0.0):
        if not betas[0] > 0:
            raise ValueError(f'AdaBelief needs beta1 > 0 (got {betas[0]}): without momentum its second moment '
                             f'(g - m)^2 is zero')
        super().__init__(params, dict(lr=lr, betas=tuple(float(b) for b in betas), eps=eps, weight_decay=weight_decay))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group in self.param_groups:
            params = [p for p in group['params'] if p.grad is not None]
            if not params:
                continue
            beta1, beta2 = group['betas']
            for p in params:
                state = self.state[p]
                if not state:
                    state['step'] = torch.zeros((), dtype=torch.float32)
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_var'] = torch.zeros_like(p, memory_format=torch.preserve_format)
            grads = [p.grad for p in params]
            m = [self.state[p]['exp_avg'] for p in params]
            s = [self.state[p]['exp_avg_var'] for p in params]
            for p in params:
                self.state[p]['step'] += 1
            t = float(self.state[params[0]]['step'])
            if group['weight_decay']:
                torch._foreach_mul_(params, 1 - group['lr'] * group['weight_decay'])
            torch._foreach_lerp_(m, grads, 1 - beta1)
            diff = torch._foreach_sub(grads, m)
            torch._foreach_mul_(s, beta2)
            torch._foreach_addcmul_(s, diff, diff, value=1 - beta2)
            torch._foreach_add_(s, group['eps'])
            bc1, bc2 = 1 - beta1 ** t, 1 - beta2 ** t
            denom = torch._foreach_sqrt(s)
            torch._foreach_div_(denom, bc2 ** 0.5)
            torch._foreach_add_(denom, group['eps'])
            torch._foreach_addcdiv_(params, m, denom, value=-group['lr'] / bc1)
        return loss


OPTIMIZERS = ('adamw', 'adabelief')


def make_optimizer(name: str, params, lr: float, betas, fused: bool = False):
    betas = tuple(float(b) for b in betas)
    if name == 'adamw':
        return torch.optim.AdamW(params, lr=lr, betas=betas, eps=1e-8, weight_decay=0.0, fused=fused)
    if name == 'adabelief':
        return AdaBelief(params, lr=lr, betas=betas, eps=1e-16, weight_decay=0.0)
    raise ValueError(f'unknown optimizer {name!r}; available: {", ".join(OPTIMIZERS)}')
