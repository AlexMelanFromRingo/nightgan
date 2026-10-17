"""Dataset preparation (images → latents, REPA images, reference statistics) and the training-time sampler."""

import json
import os
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageOps
from tqdm import tqdm

from . import __version__
from .autoencoder import SD_VAE, LatentNorm, load_autoencoder, sample_latent

FORMAT = 1
IMAGE_EXTS = {'.jpg', '.jpeg', '.png', '.webp', '.bmp', '.tif', '.tiff', '.gif'}
REPA_SIZE = 224
MAX_STORED_FEATURES = 10000     # features kept for KID
GPU_LATENT_BYTES = 2 ** 30      # latent sets up to this size are kept on the GPU, larger ones in RAM


class DatasetError(Exception):
    pass


# --------------------------------------------------------------------------------------------------------------------
# Images

def scan_images(src, labels: str = 'none'):
    """All files under `src` (sorted). Returns `(items, skipped)`: `items` are `(path, class name or None)` of files
    with an image extension, `skipped` are `(path, reason)`. With `labels='dirs'` the class of an image is the name of
    the top level folder it is in."""
    src = Path(src)
    if not src.is_dir():
        raise DatasetError(f'{src} is not a folder')
    if labels not in ('none', 'dirs'):
        raise ValueError(f"labels has to be 'none' or 'dirs', not {labels!r}")
    items, skipped = [], []
    for path in sorted(p for p in src.rglob('*') if p.is_file()):
        if path.suffix.lower() not in IMAGE_EXTS:
            skipped.append((path, 'not an image'))
            continue
        cls = None
        if labels == 'dirs':
            parts = path.relative_to(src).parts
            if len(parts) < 2:
                raise DatasetError(f'with class labels from folders every image has to be inside a class folder, '
                                   f'{path} is not')
            cls = parts[0]
        items.append((path, cls))
    return items, skipped


def _to_rgb(im: Image.Image) -> Image.Image:
    if im.mode in ('I;16', 'I;16L', 'I;16B', 'I;16N', 'I'):
        a = np.asarray(im).astype(np.float64)
        if im.mode.startswith('I;16') or a.max() > 255:
            a = a / 257
        im = Image.fromarray(np.clip(np.round(a), 0, 255).astype(np.uint8), 'L')
    if im.mode in ('RGBA', 'LA', 'PA', 'RGBa', 'La') or (im.mode == 'P' and 'transparency' in im.info):
        im = im.convert('RGBA')
        white = Image.new('RGBA', im.size, (255, 255, 255, 255))
        im = Image.alpha_composite(white, im)
    return im.convert('RGB')


def load_square(path, res: int) -> np.ndarray:
    """An upright RGB `[res, res, 3]` uint8 image: EXIF orientation applied, transparency on white, other modes
    (palette, grayscale, CMYK, 16 bit) converted, centre square crop, Lanczos resampling."""
    with Image.open(path) as im:
        im = ImageOps.exif_transpose(im)
        im = _to_rgb(im)
        w, h = im.size
        s = min(w, h)
        left, top = (w - s) // 2, (h - s) // 2
        im = im.crop((left, top, left + s, top + s)).resize((res, res), Image.LANCZOS)
        return np.asarray(im, dtype=np.uint8)


def _load(item, res):
    path, cls = item
    try:
        return load_square(path, res), cls, None
    except Exception as e:  # unreadable or truncated files are skipped, not fatal
        return None, cls, f'{type(e).__name__}: {e}'


# --------------------------------------------------------------------------------------------------------------------
# Preparation

class _FeatureStats:
    """Running mean and covariance of features, plus a uniform random sample of at most `MAX_STORED_FEATURES` rows
    for KID (reservoir sampling: the folder order, e.g. sorted classes, does not bias the sample)."""

    def __init__(self, seed: int = 0):
        self.n, self.sum, self.outer, self.kept = 0, None, None, None
        self.rng = np.random.default_rng(seed)

    def add(self, feats: torch.Tensor):
        f = feats.detach().double().cpu()
        if self.sum is None:
            self.sum = torch.zeros(f.shape[1], dtype=torch.float64)
            self.outer = torch.zeros(f.shape[1], f.shape[1], dtype=torch.float64)
            self.kept = np.zeros((MAX_STORED_FEATURES, f.shape[1]), dtype=np.float32)
        self.sum += f.sum(0)
        self.outer += f.T @ f
        rows = f.float().numpy()
        for row in rows:
            if self.n < MAX_STORED_FEATURES:
                self.kept[self.n] = row
            else:
                j = self.rng.integers(0, self.n + 1)
                if j < MAX_STORED_FEATURES:
                    self.kept[j] = row
            self.n += 1

    def save(self, path):
        mu = self.sum / self.n
        sigma = (self.outer - self.n * torch.outer(mu, mu)) / max(self.n - 1, 1)
        np.savez(path, mu=mu.numpy(), sigma=sigma.numpy(), features=self.kept[:min(self.n, MAX_STORED_FEATURES)],
                 count=self.n)


def _resize_uint8(x: torch.Tensor, size: int) -> torch.Tensor:
    if x.shape[-1] == size:
        return x
    y = F.interpolate(x.float(), size=(size, size), mode='bicubic', antialias=True, align_corners=False)
    return y.round().clamp(0, 255).to(torch.uint8)


def _atomic_write_text(path: Path, text: str):
    tmp = path.with_name(path.name + '.tmp')
    tmp.write_text(text)
    os.replace(tmp, path)


def prepare(src, dst, resolution: int = 256, labels: str = 'none', autoencoder: str = SD_VAE, flip: bool = True,
            repa_images: bool = True, stats: bool = True, max_images: int | None = None, batch: int = 32,
            workers: int = 8, device=None, extractors: dict | None = None, ae=None, overwrite: bool = False,
            progress: bool = True) -> dict:
    """Encodes the images under `src` into the dataset folder `dst`.

    `extractors` maps a statistics name ('inception', 'dinov2') to a function of uint8 images `[B, 3, H, W]`
    returning features `[B, D]`; by default the metric networks are loaded. `meta.json` is written last: a folder
    without it is an interrupted preparation."""
    t0 = time.time()
    dst = Path(dst)
    meta_path = dst / 'meta.json'
    if meta_path.exists():
        if not overwrite:
            raise DatasetError(f'{dst} already contains a prepared dataset (use another folder or overwrite)')
        meta_path.unlink()
    items, skipped = scan_images(src, labels)
    if max_images:
        items = items[:max_images]
    if not items:
        raise DatasetError(f'no images found in {src}')
    classes = sorted({c for _, c in items}) if labels == 'dirs' else None
    class_index = {c: i for i, c in enumerate(classes or [])}

    device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
    ae = ae or load_autoencoder(autoencoder, device)
    if resolution % ae.factor:
        raise DatasetError(f'resolution {resolution} is not divisible by the autoencoder factor {ae.factor}')
    latent_res, channels = resolution // ae.factor, ae.channels
    if stats and extractors is None:
        from .metrics import default_extractors
        extractors = default_extractors(device)
    extractors = extractors if stats else {}

    dst.mkdir(parents=True, exist_ok=True)
    (dst / 'stats').mkdir(exist_ok=True)
    capacity, orientations = len(items), 2 if flip else 1
    latents = np.lib.format.open_memmap(dst / 'latents.npy', mode='w+', dtype=np.float16,
                                        shape=(capacity, orientations, 2 * channels, latent_res, latent_res))
    images = None
    if repa_images:
        images = np.lib.format.open_memmap(dst / 'images224.npy', mode='w+', dtype=np.uint8,
                                           shape=(capacity, REPA_SIZE, REPA_SIZE, 3))
    feature_stats = {name: _FeatureStats() for name in extractors}
    lat_sum = torch.zeros(channels, dtype=torch.float64)
    lat_sq = torch.zeros(channels, dtype=torch.float64)
    lat_var = torch.zeros(channels, dtype=torch.float64)
    lat_count = 0
    label_list, failed, n = [], [], 0

    chunks = [items[i:i + batch] for i in range(0, len(items), batch)]
    bar = tqdm(total=len(items), desc='prepare', unit='img', disable=not progress)
    with ThreadPoolExecutor(workers) as pool:
        pending = [pool.submit(_load, it, resolution) for it in chunks[0]]
        for k in range(len(chunks)):
            results = [f.result() for f in pending]
            pending = [pool.submit(_load, it, resolution) for it in chunks[k + 1]] if k + 1 < len(chunks) else []
            good = []
            for (path, _), (arr, cls, err) in zip(chunks[k], results):
                if err is None:
                    good.append((arr, cls))
                else:
                    failed.append((path, err))
            bar.update(len(results))
            if not good:
                continue
            x = torch.from_numpy(np.stack([a for a, _ in good])).permute(0, 3, 1, 2).to(device)
            moments = [ae.encode_moments(x)]
            if flip:
                moments.append(ae.encode_moments(x.flip(-1)))
            m = torch.stack(moments, dim=1).float()
            b = x.shape[0]
            latents[n:n + b] = m.half().cpu().numpy()
            mean, logvar = m.chunk(2, dim=2)
            lat_sum += mean.double().sum(dim=(0, 1, 3, 4)).cpu()
            lat_sq += mean.double().square().sum(dim=(0, 1, 3, 4)).cpu()
            lat_var += logvar.double().exp().sum(dim=(0, 1, 3, 4)).cpu()
            lat_count += mean[:, :, 0].numel()
            if images is not None:
                images[n:n + b] = _resize_uint8(x, REPA_SIZE).permute(0, 2, 3, 1).cpu().numpy()
            for name, fn in extractors.items():
                feature_stats[name].add(fn(x))
            label_list += [class_index.get(c, 0) for _, c in good]
            n += b
    bar.close()
    if n == 0:
        raise DatasetError(f'none of the {len(items)} images in {src} could be read')

    latents.flush()
    del latents
    if images is not None:
        images.flush()
        del images
    for name, acc in feature_stats.items():
        acc.save(dst / 'stats' / f'{name}.npz')
    if classes:
        np.save(dst / 'labels.npy', np.asarray(label_list, dtype=np.int64))
    lat_mean = lat_sum / lat_count
    lat_std = (lat_sq / lat_count - lat_mean.square() + lat_var / lat_count).clamp_min(1e-12).sqrt()
    skipped = sorted(skipped + failed, key=lambda s: str(s[0]))
    meta = dict(format=FORMAT, version=__version__, resolution=resolution, count=n, orientations=orientations,
                autoencoder=ae.name, latent=dict(channels=channels, resolution=latent_res, factor=ae.factor),
                latent_mean=lat_mean.tolist(), latent_std=lat_std.tolist(), classes=classes,
                repa_images=repa_images, stats=sorted(extractors), source=str(Path(src).resolve()),
                skipped=len(skipped))
    _atomic_write_text(meta_path, json.dumps(meta, indent=2))
    return dict(count=n, skipped=skipped, classes=classes, meta=meta, seconds=time.time() - t0)


def load_meta(path) -> dict:
    path = Path(path)
    if not path.is_dir():
        raise DatasetError(f'dataset folder {path} does not exist (create it with nightgan prepare)')
    meta_path = path / 'meta.json'
    if not meta_path.exists():
        raise DatasetError(f'{path} is not a complete dataset (no meta.json): the preparation was interrupted or '
                           f'never ran, re-run nightgan prepare')
    meta = json.loads(meta_path.read_text())
    if meta.get('format') != FORMAT:
        raise DatasetError(f'{path} was prepared in format {meta.get("format")}, this version reads {FORMAT}: '
                           f're-run nightgan prepare')
    return meta


# --------------------------------------------------------------------------------------------------------------------
# Training batches

@dataclass
class Batch:
    latents: torch.Tensor           # [B, C, h, w] normalized latent samples
    labels: torch.Tensor | None     # [B] class indices
    images224: torch.Tensor | None  # [B, 3, 224, 224] uint8, same orientation as the latents (REPA targets)
    indices: torch.Tensor           # [B] dataset indices
    orient: torch.Tensor            # [B] 0: original, 1: mirrored


class TrainSet:
    """Endless, deterministic stream of training batches.

    Each epoch is a permutation drawn from `seed` and the epoch number, and so are the orientations; the position in
    the stream is the whole state, so a resumed run sees exactly the batches it would have seen."""

    def __init__(self, path, device, seed: int = 0, repa: bool = True):
        self.path = Path(path)
        self.meta = meta = load_meta(self.path)
        self.device = torch.device(device)
        self.count = n = meta['count']
        lat = np.load(self.path / 'latents.npy', mmap_mode='r')[:n]
        self.latents = torch.from_numpy(np.array(lat))  # read into memory
        if self.device.type == 'cuda' and self.latents.nbytes <= GPU_LATENT_BYTES:
            self.latents = self.latents.to(self.device)
        # larger sets stay in pageable RAM; only each batch is pinned for the copy (WSL2 limits pinned memory)
        self.orientations = self.latents.shape[1]
        self.norm = LatentNorm(meta['latent_mean'], meta['latent_std'])
        self.class_names = meta['classes'] or []
        self.num_classes = len(self.class_names)
        self.labels = torch.from_numpy(np.load(self.path / 'labels.npy')[:n]) if self.num_classes else None
        self.images = None
        if repa:
            if not meta.get('repa_images'):
                raise DatasetError(f'{self.path} was prepared without REPA images: re-run nightgan prepare with them, '
                                   f'or train without REPA')
            self.images = np.load(self.path / 'images224.npy', mmap_mode='r')
        self.seed, self.epoch, self.position = seed, 0, 0
        self._order = None

    def _epoch_order(self):
        if self._order is None or self._order[0] != self.epoch:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + self.epoch)
            perm = torch.randperm(self.count, generator=g)
            orient = torch.randint(0, self.orientations, (self.count,), generator=g)
            self._order = (self.epoch, perm, orient)
        return self._order[1], self._order[2]

    def _draw(self, batch: int):
        idx, ori = [], []
        while batch > 0:
            perm, orient = self._epoch_order()
            take = min(batch, self.count - self.position)
            idx.append(perm[self.position:self.position + take])
            ori.append(orient[self.position:self.position + take])
            self.position += take
            batch -= take
            if self.position == self.count:
                self.epoch, self.position = self.epoch + 1, 0
        return torch.cat(idx), torch.cat(ori)

    def next(self, batch: int) -> Batch:
        idx, ori = self._draw(batch)
        dev = self.latents.device
        moments = self._to_device(self.latents[idx.to(dev), ori.to(dev)])
        z = self.norm.normalize(sample_latent(moments))
        labels = self.labels[idx].to(self.device) if self.labels is not None else None
        images = None
        if self.images is not None:
            arr = self.images[idx.numpy()]
            images = torch.from_numpy(arr).permute(0, 3, 1, 2)
            flip = ori.bool()
            if flip.any():
                images[flip] = images[flip].flip(-1)
            images = self._to_device(images)
        return Batch(z, labels, images, idx, ori)

    def _to_device(self, t: torch.Tensor) -> torch.Tensor:
        if t.device == self.device:
            return t
        if self.device.type == 'cuda':
            return t.pin_memory().to(self.device, non_blocking=True)
        return t.to(self.device)

    def state_dict(self) -> dict:
        return dict(seed=self.seed, epoch=self.epoch, position=self.position)

    def load_state_dict(self, state: dict):
        self.seed, self.epoch, self.position = state['seed'], state['epoch'], state['position']
        self._order = None
