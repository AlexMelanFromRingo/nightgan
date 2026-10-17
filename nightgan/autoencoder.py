"""Frozen autoencoders that define the space the GAN works in, and the normalization of their latents."""

import torch

SD_VAE = 'stabilityai/sd-vae-ft-mse'


def to_uint8(x: torch.Tensor) -> torch.Tensor:
    """Images in [-1, 1] → uint8 in [0, 255]."""
    return ((x.float() + 1) * 127.5).round().clamp(0, 255).to(torch.uint8)


def sample_latent(moments: torch.Tensor, generator=None) -> torch.Tensor:
    """Draws a latent from the diagonal Gaussian `moments = [mean | logvar]` along the channel dimension."""
    mean, logvar = moments.float().chunk(2, dim=1)
    noise = torch.randn(mean.shape, device=mean.device, generator=generator)
    return mean + (0.5 * logvar).exp() * noise


class LatentNorm:
    """Per-channel standardization of the latents with statistics of the training set."""

    def __init__(self, mean, std):
        self.mean = torch.as_tensor(mean, dtype=torch.float32)
        self.std = torch.as_tensor(std, dtype=torch.float32)

    def _view(self, t, like):
        return t.to(like.device).view(1, -1, 1, 1)

    def normalize(self, z):
        return (z - self._view(self.mean, z)) / self._view(self.std, z)

    def denormalize(self, z):
        return z * self._view(self.std, z) + self._view(self.mean, z)


class PixelAE:
    """Identity "autoencoder": the GAN works on the images themselves (small resolutions, tests)."""
    name = 'pixel'
    channels = 3
    factor = 1

    def encode_moments(self, images: torch.Tensor) -> torch.Tensor:
        mean = images.float() / 127.5 - 1
        return torch.cat([mean, torch.full_like(mean, -30.0)], dim=1)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        return z.float()


class SDVAE:
    """A diffusers `AutoencoderKL` (Stable Diffusion VAE by default). Encoding runs in float32 (once, during dataset
    preparation), decoding in bfloat16 in chunks."""

    def __init__(self, repo_id: str = SD_VAE, device='cuda'):
        from diffusers import AutoencoderKL
        self.name = repo_id
        self.device = torch.device(device)
        self.vae = AutoencoderKL.from_pretrained(repo_id).to(self.device).eval().requires_grad_(False)
        self.channels = self.vae.config.latent_channels
        self.factor = 2 ** (len(self.vae.config.block_out_channels) - 1)

    @torch.no_grad()
    def encode_moments(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device).float() / 127.5 - 1
        dist = self.vae.encode(x).latent_dist
        return torch.cat([dist.mean, dist.logvar], dim=1).float()

    @torch.no_grad()
    def decode(self, z: torch.Tensor, chunk: int = 16) -> torch.Tensor:
        out = []
        for part in z.to(self.device).split(chunk):
            with torch.autocast(self.device.type, torch.bfloat16, enabled=self.device.type == 'cuda'):
                out.append(self.vae.decode(part).sample.float())
        return torch.cat(out)


def load_autoencoder(name: str, device='cuda'):
    """`'pixel'` or a Hugging Face repository of an `AutoencoderKL`."""
    if name == 'pixel':
        return PixelAE()
    return SDVAE(name, device)
