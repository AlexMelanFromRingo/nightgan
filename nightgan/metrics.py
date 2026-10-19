"""FID, KID and FD-DINOv2 against the reference statistics stored with a prepared dataset."""

from pathlib import Path
from typing import Iterable

import numpy as np
import torch

METRICS = {'fid': 'inception', 'kid': 'inception', 'fd_dinov2': 'dinov2'}  # metric → feature extractor


class MetricError(Exception):
    pass


def frechet_distance(mu1, sigma1, mu2, sigma2) -> float:
    """Fréchet distance of two Gaussians. `tr sqrt(S1 S2)` is computed from symmetric eigendecompositions
    (`sqrt(S1) S2 sqrt(S1)` has the eigenvalues of `S1 S2`), which is exact and faster than `scipy.linalg.sqrtm`."""
    mu1, mu2, s1, s2 = (torch.as_tensor(np.asarray(v), dtype=torch.float64) for v in (mu1, mu2, sigma1, sigma2))
    w, v = torch.linalg.eigh(s1)
    sqrt_s1 = (v * w.clamp_min(0).sqrt()) @ v.T
    m = sqrt_s1 @ s2 @ sqrt_s1
    eig = torch.linalg.eigvalsh((m + m.T) / 2).clamp_min(0)
    return float((mu1 - mu2).square().sum() + s1.trace() + s2.trace() - 2 * eig.sqrt().sum())


def feature_stats(features) -> tuple[np.ndarray, np.ndarray]:
    f = torch.as_tensor(np.asarray(features), dtype=torch.float64)
    mu = f.mean(0)
    centred = f - mu
    sigma = centred.T @ centred / max(f.shape[0] - 1, 1)
    return mu.numpy(), sigma.numpy()


def kid(f1, f2, subsets: int = 100, subset_size: int = 1000, seed: int = 0) -> tuple[float, float]:
    """Kernel inception distance (unbiased MMD² with a cubic polynomial kernel) over random subsets: (mean, std)."""
    x_all = torch.as_tensor(np.asarray(f1), dtype=torch.float64)
    y_all = torch.as_tensor(np.asarray(f2), dtype=torch.float64)
    d = x_all.shape[1]
    m = min(len(x_all), len(y_all), subset_size)
    g = torch.Generator().manual_seed(seed)
    values = []
    for _ in range(subsets):
        x = x_all[torch.randperm(len(x_all), generator=g)[:m]]
        y = y_all[torch.randperm(len(y_all), generator=g)[:m]]
        a = (x @ x.T / d + 1) ** 3 + (y @ y.T / d + 1) ** 3
        b = (x @ y.T / d + 1) ** 3
        values.append(((a.sum() - a.diagonal().sum()) / (m - 1) - b.sum() * 2 / m) / m)
    values = torch.stack(values)
    return float(values.mean()), float(values.std()) if subsets > 1 else 0.0


class InceptionFeatures:
    """The InceptionV3 of the original TensorFlow FID (pytorch-fid port), 2048-d pool features."""

    def __init__(self, device='cuda'):
        from pytorch_fid.inception import InceptionV3
        self.device = torch.device(device)
        self.model = InceptionV3([InceptionV3.BLOCK_INDEX_BY_DIM[2048]]).to(self.device).eval().requires_grad_(False)

    @torch.no_grad()
    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        x = images.to(self.device).float() / 255
        with torch.autocast(self.device.type, enabled=False):
            return self.model(x)[0].flatten(1).float()


class DinoFeatures:
    """[cls] features of DINOv2 ViT-L/14 (Stein et al. 2023, FD-DINOv2), computed in float32."""

    def __init__(self, device='cuda', name: str = 'dinov2_vitl14'):
        from .encoders import DinoV2
        self.encoder = DinoV2(name, device, bf16=False)

    def __call__(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)[0]


def default_extractors(device, names=('inception', 'dinov2')) -> dict:
    factories = {'inception': InceptionFeatures, 'dinov2': DinoFeatures}
    return {name: factories[name](device) for name in names}


def load_reference(data_dir, name: str) -> dict:
    path = Path(data_dir) / 'stats' / f'{name}.npz'
    if not path.exists():
        raise MetricError(f'{data_dir} has no {name} reference statistics (prepared without them): re-run nightgan '
                          f'prepare to compute them')
    with np.load(path) as f:
        return {k: f[k] for k in f.files}


def evaluate(image_batches: Iterable[torch.Tensor], data_dir, metrics=('fid', 'kid'), extractors: dict | None = None,
             device=None) -> dict:
    """Computes `metrics` for uint8 image batches `[B, 3, H, W]` against the dataset's reference statistics."""
    unknown = sorted(set(metrics) - set(METRICS))
    if unknown:
        raise ValueError(f'unknown metric {", ".join(unknown)}; available: {", ".join(METRICS)}')
    needed = sorted({METRICS[m] for m in metrics})
    refs = {name: load_reference(data_dir, name) for name in needed}
    if extractors is None:
        extractors = default_extractors(device or ('cuda' if torch.cuda.is_available() else 'cpu'), needed)
    feats = {name: [] for name in needed}
    n = 0
    for images in image_batches:
        n += images.shape[0]
        for name in needed:
            feats[name].append(extractors[name](images).double().cpu())
    if n == 0:
        raise MetricError('no images to evaluate')
    feats = {name: torch.cat(f) for name, f in feats.items()}
    result = {'num': n}
    for metric in metrics:
        name, ref = METRICS[metric], refs[METRICS[metric]]
        if metric in ('fid', 'fd_dinov2'):
            mu, sigma = feature_stats(feats[name])
            result[metric] = frechet_distance(mu, sigma, ref['mu'], ref['sigma'])
        else:
            result['kid'], result['kid_std'] = kid(feats[name], ref['features'])
    return result


def reconstruction_quality(data_dir, num: int = 10000, metrics=('fid', 'kid'), ae=None, extractors: dict | None = None,
                           device=None, batch: int = 32, seed: int = 0, progress: bool = False) -> dict:
    """`metrics` of the dataset's own images after the autoencoder round trip (their stored latents, sampled as in
    training, decoded): the best score a GAN in this latent space can reach in the same evaluation. `num` images are
    drawn at random, with repetition when the dataset is smaller."""
    from tqdm import tqdm

    from .autoencoder import load_autoencoder, sample_latent, to_uint8
    from .data import load_meta
    data_dir = Path(data_dir)
    meta = load_meta(data_dir)
    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ae = ae or load_autoencoder(meta['autoencoder'], device)
    latents = np.load(data_dir / 'latents.npy', mmap_mode='r')
    count = meta['count']
    idx = np.sort(np.random.default_rng(seed).choice(count, num, replace=num > count))
    g = torch.Generator(device).manual_seed(seed)

    def batches():
        with tqdm(total=num, desc='reconstruction', unit='img', disable=not progress) as bar:
            for i in range(0, num, batch):
                moments = torch.from_numpy(np.ascontiguousarray(latents[idx[i:i + batch], 0])).to(device)
                yield to_uint8(ae.decode(sample_latent(moments, g))).cpu()
                bar.update(min(batch, num - i))

    return evaluate(batches(), data_dir, metrics=metrics, extractors=extractors, device=device)
