"""Command line interface: nightgan prepare | train | generate | interpolate | eval."""

import argparse
import json
import re
import sys
from dataclasses import replace
from pathlib import Path

import torch

from . import __version__

DURATION_UNITS = {'s': 1, 'm': 60, 'h': 3600, 'd': 86400}
REPA_DIMS = {'dinov2_vits14': 384, 'dinov2_vitb14': 768, 'dinov2_vitl14': 1024, 'dinov2_vitg14': 1536,
             'dinov2_vits14_reg': 384, 'dinov2_vitb14_reg': 768, 'dinov2_vitl14_reg': 1024, 'dinov2_vitg14_reg': 1536,
             'random': 64, 'none': 0}


def parse_duration(text: str) -> float:
    """'10h', '90m', '2d', '3600s' or plain seconds."""
    m = re.fullmatch(r'\s*(\d+(?:\.\d+)?)\s*([smhd]?)\s*', str(text))
    if not m:
        raise ValueError(f'cannot read the duration {text!r}, use e.g. 10h, 90m, 2d or seconds')
    return float(m.group(1)) * DURATION_UNITS[m.group(2) or 's']


def parse_seeds(text: str) -> list[int]:
    """'0-7,12' → [0, 1, ..., 7, 12]."""
    seeds = []
    for part in str(text).split(','):
        part = part.strip()
        if '-' in part:
            a, b = part.split('-')
            seeds += list(range(int(a), int(b) + 1))
        elif part:
            seeds.append(int(part))
    return seeds


# --------------------------------------------------------------------------------------------------------------------

def cmd_prepare(args):
    from .data import prepare
    summary = prepare(args.src, args.dst, resolution=args.resolution, labels=args.labels, autoencoder=args.autoencoder,
                      flip=not args.no_flip, repa_images=not args.no_repa_images, stats=not args.no_stats,
                      max_images=args.max_images, batch=args.batch, workers=args.workers, device=args.device,
                      overwrite=args.overwrite)
    print(f'{summary["count"]} images prepared in {args.dst} ({summary["seconds"]:.0f} s)')
    if summary['classes']:
        print(f'{len(summary["classes"])} classes: {", ".join(summary["classes"])}')
    if summary['skipped']:
        print(f'{len(summary["skipped"])} files skipped:')
        for path, reason in summary['skipped'][:20]:
            print(f'  {path}: {reason}')
        if len(summary['skipped']) > 20:
            print(f'  ... and {len(summary["skipped"]) - 20} more')


def train_length(args, saved: dict | None):
    """`(kimg, time_limit)` of a run: the flags if given, else (when resuming) the length the run was started with."""
    if args.kimg is not None or args.time is not None:
        return args.kimg, parse_duration(args.time) if args.time else None
    if saved is not None:
        return saved['train'].get('kimg'), saved['train'].get('time_limit')
    return None, None


MODEL_FLAGS = ('dim', 'depth', 'heads', 'd_dim', 'd_depth', 'd_heads', 'z_dim', 'patch', 'cmap_dim')


def cmd_train(args):
    from .config import ModelConfig, TrainConfig, from_dict, model_config_for
    from .data import load_meta
    from .trainer import Trainer
    meta = load_meta(args.data)
    run = Path(args.run)
    overrides = {k: getattr(args, k) for k in MODEL_FLAGS if getattr(args, k) is not None}
    saved = json.loads((run / 'config.json').read_text()) if (run / 'config.json').exists() else None
    model_cfg = None
    repa_dim = REPA_DIMS.get(args.repa_encoder, 768) if args.repa_weight > 0 else 0
    if saved is None or args.preset:
        patch = overrides.pop('patch', None)
        model_cfg = model_config_for(args.preset or 'S', latent_res=meta['latent']['resolution'],
                                     latent_channels=meta['latent']['channels'], num_classes=len(meta['classes'] or []),
                                     patch=patch, repa_dim=repa_dim, **overrides)
    elif overrides:  # on resume, model flags apply on top of the saved configuration
        model_cfg = replace(from_dict(ModelConfig, saved['model']), **overrides)
    kimg, time_limit = train_length(args, saved)
    train_cfg = TrainConfig(
        batch=args.batch, micro_batch=args.micro_batch, lr_g=args.lr_g, lr_d=args.lr_d, gamma=args.gamma,
        gp_sigma=args.gp_sigma, cons_weight=args.cons_weight, repa_weight=args.repa_weight,
        repa_encoder=args.repa_encoder if args.repa_weight > 0 else 'none', aug_policy=args.aug_policy,
        aug_p=args.aug_p, ema_halflife_kimg=args.ema_halflife_kimg, kimg=kimg, time_limit=time_limit,
        eval_every_kimg=args.eval_every_kimg,
        eval_num=args.eval_num, sample_every_kimg=args.sample_every_kimg, snapshot_minutes=args.snapshot_minutes,
        compile=not args.no_compile, checkpointing=args.checkpointing, seed=args.seed)
    trainer = Trainer(args.data, run, model_cfg, train_cfg, device=args.device, force_config=args.force_config,
                      log=lambda msg: print(msg, flush=True))
    summary = trainer.run()
    state = 'finished' if summary['finished'] else 'interrupted' if summary['interrupted'] else 'stopped'
    best = f', best FID {summary["best_fid"]:.2f}' if summary['best_fid'] is not None else ''
    print(f'{state} at {summary["kimg"]:.1f} kimg ({summary["seconds"] / 3600:.2f} h of training){best}; '
          f'model: {run / "model.safetensors"}')


def _load_pipeline(args):
    from .pipeline import Pipeline
    return Pipeline.load(args.model, device=args.device)


def cmd_generate(args):
    from .pipeline import grid_image
    pipe = _load_pipeline(args)
    seeds = parse_seeds(args.seeds) if args.seeds else list(range(args.n))
    images = pipe.generate(seeds=seeds, class_idx=args.class_idx, truncation=args.truncation, batch=args.batch)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    if args.grid:
        grid_image(images).save(out / 'grid.png')
        print(f'grid of {len(images)} images: {out / "grid.png"}')
    else:
        for seed, im in zip(seeds, images):
            im.save(out / f'seed{seed:05d}.png')
        print(f'{len(images)} images in {out}')


def cmd_interpolate(args):
    pipe = _load_pipeline(args)
    frames = pipe.interpolate(parse_seeds(args.seeds), frames=args.frames, class_idx=args.class_idx,
                              truncation=args.truncation, batch=args.batch)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.suffix.lower() == '.gif':
        frames[0].save(out, save_all=True, append_images=frames[1:], duration=round(1000 / args.fps), loop=0)
    else:
        try:
            import imageio.v2 as imageio
            import numpy as np
            with imageio.get_writer(out, fps=args.fps, quality=8, macro_block_size=1) as writer:
                for f in frames:
                    writer.append_data(np.asarray(f))
        except ImportError as e:
            raise RuntimeError(f'writing {out.suffix} needs imageio with ffmpeg (pip install nightgan[video]); '
                               f'a .gif works without it') from e
    print(f'{len(frames)} frames: {out}')


def cmd_eval(args):
    import numpy as np
    from .data import load_meta, load_square, scan_images
    from .metrics import evaluate
    if args.images:
        if len(args.targets) != 1:
            raise ValueError('with --images give only the dataset folder')
        data = args.targets[0]
        res = load_meta(data)['resolution']
        items, _ = scan_images(args.images)
        if args.num:
            items = items[:args.num]

        def batches():
            for i in range(0, len(items), args.batch):
                arr = [load_square(p, res) for p, _ in items[i:i + args.batch]]
                yield torch.from_numpy(np.stack(arr)).permute(0, 3, 1, 2)
    else:
        if len(args.targets) != 2:
            raise ValueError('give the model and the dataset folder (or --images DIR and the dataset folder)')
        args.model, data = args.targets
        pipe = _load_pipeline(args)
        labels = None
        meta = load_meta(data)
        if pipe.cfg.num_classes and meta['classes']:
            labels = torch.from_numpy(np.load(Path(data) / 'labels.npy')[:meta['count']])
        num = args.num or 50000
        g = torch.Generator().manual_seed(args.seed)

        def batches():
            for i in range(0, num, args.batch):
                n = min(args.batch, num - i)
                z = torch.randn(n, pipe.cfg.z_dim, generator=g)
                y = labels[torch.randint(0, len(labels), (n,), generator=g)] if labels is not None else None
                yield pipe.images(z, y, args.truncation, args.batch)
    metrics = tuple(m.strip() for m in args.metrics.split(',') if m.strip())
    result = evaluate(batches(), data, metrics=metrics, device=args.device)
    for k, v in result.items():
        print(f'{k}: {v:.6g}' if isinstance(v, float) else f'{k}: {v}')


# --------------------------------------------------------------------------------------------------------------------

def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog='nightgan', description='A latent-space transformer GAN for a single GPU.')
    p.add_argument('--version', action='version', version=f'nightgan {__version__}')
    sub = p.add_subparsers(dest='command', metavar='command')

    s = sub.add_parser('prepare', help='encode a folder of images into a training dataset')
    s.add_argument('src', help='folder with images (searched recursively)')
    s.add_argument('dst', help='dataset folder to create')
    s.add_argument('--resolution', type=int, default=256, help='image size (default 256)')
    s.add_argument('--labels', choices=['none', 'dirs'], default='none',
                   help="'dirs': the top level subfolders are classes (default none)")
    s.add_argument('--autoencoder', default='stabilityai/sd-vae-ft-mse',
                   help="Hugging Face AutoencoderKL repository, or 'pixel' to train on the images themselves")
    s.add_argument('--no-flip', action='store_true', help='do not add mirrored copies')
    s.add_argument('--no-repa-images', action='store_true', help='do not store the 224px images REPA needs')
    s.add_argument('--no-stats', action='store_true', help='skip the FID / FD-DINOv2 reference statistics')
    s.add_argument('--max-images', type=int, help='use only the first N images')
    s.add_argument('--batch', type=int, default=32, help='images per encoding batch')
    s.add_argument('--workers', type=int, default=8, help='image decoding threads')
    s.add_argument('--overwrite', action='store_true', help='replace an existing dataset folder')
    s.add_argument('--device', help='cuda or cpu (default: cuda when available)')
    s.set_defaults(func=cmd_prepare)

    t = sub.add_parser('train', help='train (or resume training) on a prepared dataset')
    t.add_argument('data', help='dataset folder from nightgan prepare')
    t.add_argument('run', help='run folder; an existing run is resumed')
    t.add_argument('--preset', choices=['S', 'B'], help='model size: S (overnight, default) or B (several days)')
    t.add_argument('--time', help='training length, e.g. 10h (total over all sessions of the run, evaluations '
                                  'included; a resumed run keeps its length unless a new one is given)')
    t.add_argument('--kimg', type=float, help='training length in thousands of images')
    t.add_argument('--batch', type=int, default=256, help='images per optimizer step (default 256)')
    t.add_argument('--micro-batch', type=int, default=0, help='images per forward pass, 0: the largest that fits')
    t.add_argument('--lr-g', type=float, help='generator learning rate (default 2e-4 * 768 / width)')
    t.add_argument('--lr-d', type=float, help='discriminator learning rate (default 2e-4 * 768 / width)')
    t.add_argument('--gamma', type=float, default=0.1, help='R1 + R2 weight (default 0.1)')
    t.add_argument('--gp-sigma', type=float, default=0.03, help='noise of the approximated gradient penalty')
    t.add_argument('--cons-weight', type=float, default=0.1, help='stage consistency weight (default 0.1)')
    t.add_argument('--repa-weight', type=float, default=1.0, help='REPA weight, 0 disables it (default 1)')
    t.add_argument('--repa-encoder', default='dinov2_vitb14', help='DINOv2 model of the REPA targets, or none')
    t.add_argument('--aug-policy', default='color,translation,cutout', help='latent augmentations')
    t.add_argument('--aug-p', type=float, default=1.0, help='probability of each augmentation (default 1)')
    t.add_argument('--ema-halflife-kimg', type=float, default=350.0, help='EMA half-life in thousands of images')
    t.add_argument('--eval-every-kimg', type=float, default=500.0, help='FID / KID interval, 0 disables')
    t.add_argument('--eval-num', type=int, default=10000, help='images per evaluation (default 10000)')
    t.add_argument('--sample-every-kimg', type=float, default=100.0, help='sample grid interval')
    t.add_argument('--snapshot-minutes', type=float, default=30.0, help='snapshot interval in minutes')
    t.add_argument('--no-compile', action='store_true', help='do not use torch.compile')
    t.add_argument('--checkpointing', action='store_true', help='gradient checkpointing (less memory, slower)')
    t.add_argument('--seed', type=int, default=0)
    t.add_argument('--force-config', action='store_true',
                   help='start over in an existing run folder with a different model (old snapshots are kept aside)')
    t.add_argument('--device', help='cuda or cpu (default: cuda when available)')
    g = t.add_argument_group('model size (overrides the preset)')
    for flag in MODEL_FLAGS:
        g.add_argument('--' + flag.replace('_', '-'), type=int, dest=flag)
    t.set_defaults(func=cmd_train)

    def model_args(q):
        q.add_argument('--class', type=int, dest='class_idx', help='class index (default: random per seed)')
        q.add_argument('--truncation', type=float, default=1.0, help='1: none, lower: closer to the average image')
        q.add_argument('--batch', type=int, default=16)
        q.add_argument('--device', help='cuda or cpu (default: cuda when available)')

    q = sub.add_parser('generate', help='generate images from seeds')
    q.add_argument('model', help='model.safetensors or a run folder')
    q.add_argument('out', help='output folder')
    q.add_argument('--seeds', help='e.g. 0-63 or 1,5,9')
    q.add_argument('--n', type=int, default=16, help='number of images when no seeds are given (seeds 0..n-1)')
    q.add_argument('--grid', action='store_true', help='one grid image instead of separate files')
    model_args(q)
    q.set_defaults(func=cmd_generate)

    q = sub.add_parser('interpolate', help='a looping walk through the latents of some seeds (.gif or .mp4)')
    q.add_argument('model', help='model.safetensors or a run folder')
    q.add_argument('out', help='output file, .gif or .mp4')
    q.add_argument('--seeds', default='0,1,2,3', help='seeds to visit (default 0,1,2,3)')
    q.add_argument('--frames', type=int, default=60, help='frames per transition')
    q.add_argument('--fps', type=int, default=30)
    model_args(q)
    q.set_defaults(func=cmd_interpolate)

    q = sub.add_parser('eval', help='FID / KID / FD-DINOv2 of a model or of a folder of images')
    q.add_argument('targets', nargs='+', help='MODEL DATA, or DATA with --images')
    q.add_argument('--images', help='evaluate the images in this folder instead of a model')
    q.add_argument('--metrics', default='fid,kid', help='comma separated: fid, kid, fd_dinov2')
    q.add_argument('--num', type=int, help='number of images (default 50000 generated / all in the folder)')
    q.add_argument('--seed', type=int, default=0)
    model_args(q)
    q.set_defaults(func=cmd_eval)
    return p


def main(argv=None) -> int:
    from .data import DatasetError
    from .metrics import MetricError
    from .trainer import ConfigMismatch, TrainingDiverged
    parser = build_parser()
    args = parser.parse_args(argv)
    if not getattr(args, 'func', None):
        parser.print_help()
        return 1
    try:
        args.func(args)
    except (DatasetError, MetricError, ConfigMismatch, TrainingDiverged, ValueError, FileNotFoundError,
            RuntimeError) as e:
        print(f'error: {e}', file=sys.stderr)
        return 2
    return 0


if __name__ == '__main__':
    sys.exit(main())
