"""Training throughput of a preset on a synthetic dataset (random latents), as the trainer runs it.

    python tools/bench.py --preset S --steps 20 --repeat 3
"""

import argparse
import json
import statistics
import tempfile
import time
from pathlib import Path

import numpy as np
import torch

from nightgan.config import TrainConfig, model_config_for
from nightgan.data import FORMAT
from nightgan.trainer import Trainer

REPA_DIMS = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024, 'none': 0}


def synthetic_dataset(root: Path, n: int, channels: int, latent_res: int, classes: int):
    lat = np.lib.format.open_memmap(root / 'latents.npy', mode='w+', dtype=np.float16,
                                    shape=(n, 2, 2 * channels, latent_res, latent_res))
    rng = np.random.default_rng(0)
    lat[:, :, :channels] = rng.standard_normal((n, 2, channels, latent_res, latent_res)).astype(np.float16)
    lat[:, :, channels:] = -8
    lat.flush()
    img = np.lib.format.open_memmap(root / 'images224.npy', mode='w+', dtype=np.uint8, shape=(n, 224, 224, 3))
    img[:] = rng.integers(0, 256, (n, 224, 224, 3), dtype=np.uint8)
    img.flush()
    names = [f'c{i}' for i in range(classes)] or None
    if names:
        np.save(root / 'labels.npy', rng.integers(0, classes, n).astype(np.int64))
    meta = dict(format=FORMAT, resolution=latent_res * 8, count=n, orientations=2, autoencoder='pixel',
                latent=dict(channels=channels, resolution=latent_res, factor=8), latent_mean=[0.0] * channels,
                latent_std=[1.0] * channels, classes=names, repa_images=True, stats=[])
    (root / 'meta.json').write_text(json.dumps(meta))


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--preset', default='S')
    p.add_argument('--resolution', type=int, default=256)
    p.add_argument('--batch', type=int, default=256)
    p.add_argument('--micro-batch', type=int, default=0)
    p.add_argument('--repa-encoder', default='dinov2_vitb14')
    p.add_argument('--no-compile', action='store_true')
    p.add_argument('--checkpointing', action='store_true')
    p.add_argument('--classes', type=int, default=0)
    p.add_argument('--steps', type=int, default=10)
    p.add_argument('--warmup', type=int, default=3)
    p.add_argument('--repeat', type=int, default=3)
    args = p.parse_args()

    latent_res = args.resolution // 8
    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        (root / 'data').mkdir()
        synthetic_dataset(root / 'data', 2048, 4, latent_res, args.classes)
        repa = args.repa_encoder != 'none'
        mc = model_config_for(args.preset, latent_res, 4, args.classes, repa_dim=REPA_DIMS[args.repa_encoder],
                              checkpointing=args.checkpointing)
        tc = TrainConfig(batch=args.batch, micro_batch=args.micro_batch, compile=not args.no_compile,
                         repa_encoder=args.repa_encoder, repa_weight=1.0 if repa else 0.0, eval_every_kimg=0,
                         sample_every_kimg=0, snapshot_minutes=1e9)
        t = Trainer(root / 'data', root / 'run', mc, tc, device='cuda', log=print)
        for _ in range(args.warmup):
            t.train_step()
        speeds = []
        torch.cuda.reset_peak_memory_stats()
        for _ in range(args.repeat):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(args.steps):
                t.train_step()
            torch.cuda.synchronize()
            speeds.append(args.steps * args.batch / (time.perf_counter() - t0))
        peak = torch.cuda.max_memory_reserved() / 2 ** 30
        print(f'{args.preset} {args.resolution}px batch {args.batch} (micro {t.micro_batch}), repa {args.repa_encoder}, '
              f'compile {not args.no_compile}: {statistics.median(speeds):.1f} img/s median of '
              f'{[round(s, 1) for s in speeds]}, peak reserved {peak:.1f} GB')


if __name__ == '__main__':
    main()
