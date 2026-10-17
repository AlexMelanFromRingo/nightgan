"""Adversarial loss, gradient penalties, stage consistency and representation alignment."""

import torch
import torch.nn.functional as F

STAGE_WEIGHTS = (1 / 3, 1 / 2, 1.0)  # CAT: earlier (coarser) stages are pulled less towards the final output


def rpgan_d(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    """Relativistic pairing loss of the discriminator (R3GAN), logits `[B, scales]` paired by index."""
    return F.softplus(fake.float() - real.float()).mean()


def rpgan_g(real: torch.Tensor, fake: torch.Tensor) -> torch.Tensor:
    return F.softplus(real.float() - fake.float()).mean()


def finite_diff_penalty(clean: torch.Tensor, noisy: torch.Tensor, sigma: float) -> torch.Tensor:
    """`((D(x + e) - D(x)) / sigma)^2` with `e ~ N(0, sigma^2 I)`, averaged over the scales: an unbiased estimate of
    `|grad D(x)|^2` (R1 for real inputs, R2 for fakes) that needs no double backward (GAT). Returns `[B]`."""
    return ((noisy.float() - clean.float()) / sigma).square().mean(dim=1)


def stage_consistency(stages: list[torch.Tensor], weights=STAGE_WEIGHTS) -> torch.Tensor:
    """Mean squared distance of the intermediate stage outputs to the final output (CAT).

    The final output is not detached. The stages are partial sums of the output heads, so this penalizes what the
    later heads add on top of each stage. With a detached target the gradient moves the early heads, and with them the
    target, by the same amount: the distance never shrinks and the outputs drift without bound."""
    target = stages[-1].float()
    terms = [w * (s.float() - target).square().mean() for w, s in zip(weights, stages[:-1])]
    return sum(terms) / len(terms)


def repa_loss(pred_cls, pred_tokens, target_cls, target_tokens) -> torch.Tensor:
    """`(1 - cos)` of the [cls] token plus the mean `(1 - cos)` over the patch tokens (GAT's REPA objective)."""
    cls = 1 - F.cosine_similarity(pred_cls.float(), target_cls.float(), dim=-1)
    tokens = 1 - F.cosine_similarity(pred_tokens.float(), target_tokens.float(), dim=-1)
    return (cls + tokens.mean(dim=1)).mean()
