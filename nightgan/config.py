"""Model and training configuration, presets and their JSON form."""

import dataclasses
from dataclasses import dataclass


@dataclass
class ModelConfig:
    """Architecture of the generator and the discriminator.

    The data the networks see is a `latent_channels x latent_res x latent_res` latent (or a small image in pixel
    mode), cut into `patch x patch` tokens. The discriminator sizes default to the generator sizes."""
    latent_channels: int = 4
    latent_res: int = 32
    patch: int = 2
    dim: int = 384              # generator width
    depth: int = 12             # generator blocks, a multiple of 4 (one output head per quarter)
    heads: int = 6
    d_dim: int | None = None
    d_depth: int | None = None
    d_heads: int | None = None
    z_dim: int = 64
    num_classes: int = 0        # 0: unconditional
    mlp_ratio: float = 4.0
    repa_dim: int = 768         # feature size of the REPA target encoder, 0: no projector
    cmap_dim: int = 2048        # size of the projection discriminator head
    num_scales: int = 4

    @property
    def tokens_side(self) -> int:
        return self.latent_res // self.patch

    @property
    def D_dim(self) -> int:
        return self.d_dim or self.dim

    @property
    def D_depth(self) -> int:
        return self.d_depth or self.depth

    @property
    def D_heads(self) -> int:
        return self.d_heads or self.heads


@dataclass
class TrainConfig:
    """Optimization, regularization, schedule and bookkeeping of a training run."""
    batch: int = 256                    # global batch size (images per optimizer step)
    micro_batch: int = 0                # images per forward pass on the GPU, 0: largest that fits
    lr_g: float | None = None           # None: 2e-4 * 768 / width
    lr_d: float | None = None
    optimizer: str = 'adamw'            # 'adamw' or 'adabelief' (needs beta1 > 0)
    betas: tuple[float, float] = (0.0, 0.99)
    grad_clip: float = 1.0
    gamma: float = 0.1                  # R1 and R2 weight, the loss is gamma / 2 * (R1 + R2)
    gp_sigma: float = 0.03              # noise of the finite difference gradient penalty
    gp_fraction: float = 0.25           # share of the batch the gradient penalty is computed on
    cons_weight: float = 0.1            # stage consistency
    repa_weight: float = 1.0
    repa_encoder: str = 'dinov2_vitb14'  # 'none' disables REPA
    aug_policy: str = 'color,translation,cutout'
    aug_p: float = 1.0
    ema_halflife_kimg: float = 350.0
    ema_rampup: float | None = 0.05
    kimg: float | None = None           # training length in thousands of images
    time_limit: float | None = None     # training length in seconds
    eval_every_kimg: float = 500.0
    eval_num: int = 10000
    sample_every_kimg: float = 100.0
    snapshot_minutes: float = 15.0
    keep_snapshots: int = 3
    compile: bool = True
    checkpointing: bool = False         # gradient checkpointing of the generator blocks
    seed: int = 0

    def resolved_lr(self, model: ModelConfig) -> tuple[float, float]:
        """Width-aware learning rates (GAT): 2e-4 for width 768, inversely proportional to the width."""
        lr_g = self.lr_g if self.lr_g is not None else 2e-4 * 768 / model.dim
        lr_d = self.lr_d if self.lr_d is not None else 2e-4 * 768 / model.D_dim
        return lr_g, lr_d


PRESETS = {
    'S': dict(dim=384, depth=12, heads=6),
    'B': dict(dim=768, depth=12, heads=12),
}


def model_config_for(preset: str, latent_res: int, latent_channels: int, num_classes: int,
                     patch: int | None = None, **overrides) -> ModelConfig:
    if preset not in PRESETS:
        raise ValueError(f'unknown preset {preset!r}, choose from {sorted(PRESETS)}')
    if patch is None:
        patch = max(latent_res // 16, 1)
    cfg = ModelConfig(latent_channels=latent_channels, latent_res=latent_res, patch=patch, num_classes=num_classes,
                      **{**PRESETS[preset], **overrides})
    validate(cfg)
    return cfg


def validate(cfg: ModelConfig):
    side = cfg.latent_res // cfg.patch
    if cfg.latent_res % cfg.patch or side % 2 ** (cfg.num_scales - 1):
        raise ValueError(f'patch {cfg.patch} on a {cfg.latent_res}x{cfg.latent_res} latent gives a {side}x{side} '
                         f'token grid, it has to be divisible by {2 ** (cfg.num_scales - 1)} for '
                         f'{cfg.num_scales} discriminator scales')
    if cfg.depth % 4:
        raise ValueError(f'depth {cfg.depth} has to be a multiple of 4')
    for name, width, heads in [('dim', cfg.dim, cfg.heads), ('d_dim', cfg.D_dim, cfg.D_heads)]:
        if width % heads or (width // heads) % 4:
            raise ValueError(f'{name} {width} with {heads} heads: the head size has to be a multiple of 4')


def to_dict(cfg) -> dict:
    return dataclasses.asdict(cfg)


def from_dict(cls, d: dict):
    names = {f.name: f for f in dataclasses.fields(cls)}
    unknown = sorted(set(d) - set(names))
    if unknown:
        raise ValueError(f'unknown {cls.__name__} keys: {", ".join(unknown)}')
    values = {}
    for key, value in d.items():
        if isinstance(names[key].default, tuple) and isinstance(value, list):
            value = tuple(value)
        values[key] = value
    return cls(**values)


__all__ = ['ModelConfig', 'TrainConfig', 'PRESETS', 'model_config_for', 'validate', 'to_dict', 'from_dict']
