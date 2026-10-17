"""Exported generators: saving, loading and generating images."""

import json
import math
import os
from pathlib import Path

import torch
from PIL import Image
from safetensors.torch import load_file, save_file

from .autoencoder import LatentNorm, load_autoencoder, to_uint8
from .config import ModelConfig, from_dict, to_dict
from .generator import Generator

MODEL_FORMAT = 1


def save_model(path, G: torch.nn.Module, model_cfg, meta: dict):
    """Writes the generator weights to `path` (.safetensors) and its configuration next to it (.json)."""
    path = Path(path)
    state = {k: v.detach().cpu().contiguous() for k, v in G.state_dict().items()}
    tmp = path.with_name(path.name + '.tmp')
    save_file(state, tmp)
    os.replace(tmp, path)
    info = dict(format=MODEL_FORMAT, model=to_dict(model_cfg), **meta)
    json_path = path.with_suffix('.json')
    tmp = json_path.with_name(json_path.name + '.tmp')
    tmp.write_text(json.dumps(info, indent=2))
    os.replace(tmp, json_path)


def slerp(a: torch.Tensor, b: torch.Tensor, t: float) -> torch.Tensor:
    """Spherical interpolation, which keeps Gaussian latents at a typical norm."""
    an, bn = a / a.norm(dim=-1, keepdim=True), b / b.norm(dim=-1, keepdim=True)
    omega = torch.acos((an * bn).sum(-1, keepdim=True).clamp(-1, 1))
    so = torch.sin(omega)
    if float(so.abs().min()) < 1e-6:
        return torch.lerp(a, b, t)
    return torch.sin((1 - t) * omega) / so * a + torch.sin(t * omega) / so * b


def grid_image(images: list[Image.Image], columns: int | None = None) -> Image.Image:
    cols = columns or math.ceil(math.sqrt(len(images)))
    rows = math.ceil(len(images) / cols)
    w, h = images[0].size
    grid = Image.new('RGB', (cols * w, rows * h))
    for i, im in enumerate(images):
        grid.paste(im, ((i % cols) * w, (i // cols) * h))
    return grid


class Pipeline:
    """A trained generator with its autoencoder: seeds (and classes) in, PIL images out."""

    def __init__(self, G, info: dict, autoencoder, device):
        self.G, self.info, self.autoencoder, self.device = G, info, autoencoder, torch.device(device)
        self.cfg = G.cfg
        self.norm = LatentNorm(info['latent_mean'], info['latent_std'])
        self.classes = info.get('classes') or []
        self.resolution = info['resolution']

    @classmethod
    def load(cls, path, device=None, autoencoder=None):
        """`path`: a `.safetensors` model file or a run folder (its `model.safetensors`)."""
        path = Path(path)
        if path.is_dir():
            path = path / 'model.safetensors'
        if not path.exists():
            raise FileNotFoundError(f'no model at {path}')
        info = json.loads(path.with_suffix('.json').read_text())
        if info.get('format') != MODEL_FORMAT:
            raise ValueError(f'{path} has model format {info.get("format")}, this version reads {MODEL_FORMAT}')
        device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        G = Generator(from_dict(ModelConfig, info['model']))
        G.load_state_dict(load_file(path))
        G = G.to(device).eval().requires_grad_(False)
        ae = autoencoder or load_autoencoder(info['autoencoder'], device)
        return cls(G, info, ae, device)

    def _check_class(self, class_idx):
        n = self.cfg.num_classes
        if class_idx is None:
            return
        if n == 0:
            raise ValueError('this model is unconditional, it has no classes')
        if not 0 <= class_idx < n:
            names = ', '.join(f'{i}: {c}' for i, c in enumerate(self.classes))
            raise ValueError(f'class {class_idx} is out of range, the model has {n} classes ({names})')

    def seed_inputs(self, seeds, class_idx=None):
        """The latent of each seed and its class (`class_idx`, or a class drawn from the seed)."""
        self._check_class(class_idx)
        zs, ys = [], []
        for s in seeds:
            g = torch.Generator().manual_seed(int(s))
            zs.append(torch.randn(self.cfg.z_dim, generator=g))
            if self.cfg.num_classes:
                ys.append(class_idx if class_idx is not None else int(torch.randint(self.cfg.num_classes, (1,),
                                                                                    generator=g)))
        y = torch.tensor(ys, dtype=torch.long) if ys else None
        return torch.stack(zs), y

    @torch.no_grad()
    def latents(self, z, y=None, truncation: float = 1.0) -> torch.Tensor:
        """Denormalized latents (the input of the autoencoder's decoder)."""
        with torch.autocast('cuda', torch.bfloat16, enabled=self.device.type == 'cuda'):
            lat = self.G(z.to(self.device), None if y is None else y.to(self.device), truncation)
        return self.norm.denormalize(lat.float())

    @torch.no_grad()
    def decode(self, latents) -> torch.Tensor:
        return to_uint8(self.autoencoder.decode(latents))

    def images(self, z, y=None, truncation: float = 1.0, batch: int = 16) -> torch.Tensor:
        """uint8 images `[B, 3, H, W]`."""
        out = []
        for i in range(0, z.shape[0], batch):
            out.append(self.decode(self.latents(z[i:i + batch], None if y is None else y[i:i + batch], truncation)))
        return torch.cat(out).cpu()

    def generate(self, seeds=None, n=None, class_idx=None, truncation: float = 1.0, batch: int = 16):
        if seeds is None:
            seeds = range(n or 1)
        z, y = self.seed_inputs(list(seeds), class_idx)
        return [Image.fromarray(im.permute(1, 2, 0).numpy()) for im in self.images(z, y, truncation, batch)]

    def interpolate(self, seeds, frames: int = 60, class_idx=None, truncation: float = 1.0, batch: int = 16):
        """A loop through the seeds, `frames` images per transition (spherical interpolation of the latents)."""
        z, y = self.seed_inputs(list(seeds), class_idx)
        path = [slerp(z[i], z[(i + 1) % len(z)], f / frames) for i in range(len(z)) for f in range(frames)]
        ys = None if y is None else torch.full((len(path),), int(y[0]), dtype=torch.long)
        imgs = self.images(torch.stack(path), ys, truncation, batch)
        return [Image.fromarray(im.permute(1, 2, 0).numpy()) for im in imgs]
