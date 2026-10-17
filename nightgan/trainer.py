"""The training loop: discriminator and generator steps, bookkeeping, snapshots and resuming."""

import contextlib
import json
import math
import os
import shutil
import signal
import threading
import time
from pathlib import Path

import torch
from PIL import Image

from . import __version__
from .augment import apply_aug, sample_aug, shift_images
from .autoencoder import load_autoencoder, to_uint8
from .config import ModelConfig, TrainConfig, from_dict, to_dict, validate
from .data import Batch, TrainSet
from .discriminator import Discriminator, pyramid, real_pyramid
from .ema import EMA, ema_decay
from .encoders import load_encoder
from .generator import Generator
from .losses import finite_diff_penalty, repa_loss, resize_tokens, rpgan_d, rpgan_g, stage_consistency
from .metrics import default_extractors, evaluate
from .pipeline import save_model

TICK_SECONDS = 60.0
MAX_BAD_STEPS = 3           # consecutive steps with non-finite losses before training stops
GRID = 8                    # the sample grid is GRID x GRID images
MEMORY_BUDGET = 0.85        # share of the free GPU memory the micro-batch may use
EVAL_BATCH = 32
STAT_KEYS = ('loss_d', 'r1', 'r2', 'repa', 'loss_g', 'cons', 'grad_d', 'grad_g')


class ConfigMismatch(Exception):
    pass


class TrainingDiverged(Exception):
    pass


class Trainer:
    """Trains a generator on a prepared dataset, in `run_dir`. A run directory that already holds snapshots is
    resumed from the latest one; its model configuration has to match unless `force_config` (which moves the old
    snapshots aside and starts over). Training options may change between sessions."""

    def __init__(self, data_dir, run_dir, model_cfg: ModelConfig | None, train_cfg: TrainConfig, device=None,
                 force_config: bool = False, encoder=None, extractors: dict | None = None, log=print):
        self.data_dir, self.run_dir, self.log = Path(data_dir), Path(run_dir), log
        self.device = torch.device(device or ('cuda' if torch.cuda.is_available() else 'cpu'))
        self.cuda = self.device.type == 'cuda'
        self.train_cfg = cfg = train_cfg

        saved = self._read_json(self.run_dir / 'config.json')
        resume = saved is not None
        if saved is not None:
            saved_model = from_dict(ModelConfig, saved['model'])
            if model_cfg is None:
                model_cfg = saved_model
            elif model_cfg != saved_model:
                if not force_config:
                    old, new = to_dict(saved_model), to_dict(model_cfg)
                    diff = ', '.join(f'{k}: {old[k]} -> {new[k]}' for k in new if old[k] != new[k])
                    raise ConfigMismatch(f'{self.run_dir} holds a run with another model configuration ({diff}); use '
                                         f'another run folder, or --force-config to start over in this one')
                self._move_old_snapshots()
                resume = False
        if model_cfg is None:
            raise ValueError('a new run needs a model configuration')
        validate(model_cfg)
        self.model_cfg = mc = model_cfg

        repa = cfg.repa_weight > 0 and cfg.repa_encoder != 'none'
        self.data = TrainSet(self.data_dir, self.device, seed=cfg.seed, repa=repa)
        meta = self.data.meta
        for key, expected in [('latent_channels', meta['latent']['channels']),
                              ('latent_res', meta['latent']['resolution']), ('num_classes', self.data.num_classes)]:
            if getattr(mc, key) != expected:
                raise ValueError(f'model {key} is {getattr(mc, key)} but the dataset {self.data_dir} has {expected}')
        self.encoder = None
        if repa:
            self.encoder = encoder if encoder is not None else load_encoder(cfg.repa_encoder, self.device)
            if mc.repa_dim != self.encoder.dim:
                raise ValueError(f'model repa_dim is {mc.repa_dim} but the REPA encoder {cfg.repa_encoder} has '
                                 f'{self.encoder.dim} features')
        self.extractors = extractors

        torch.manual_seed(cfg.seed)
        self.G = Generator(mc).to(self.device)
        self.G.checkpointing = cfg.checkpointing
        self.D = Discriminator(mc).to(self.device)
        self.ema = EMA(self.G)
        lr_g, lr_d = cfg.resolved_lr(mc)
        self.base_lr = (lr_g, lr_d)
        opt = dict(betas=tuple(float(b) for b in cfg.betas), eps=1e-8, weight_decay=0.0, fused=self.cuda)
        self.opt_g = torch.optim.AdamW(self.G.parameters(), lr=lr_g, **opt)
        self.opt_d = torch.optim.AdamW(self.D.parameters(), lr=lr_d, **opt)

        self.cur_nimg, self.step, self.train_seconds = 0, 0, 0.0
        self.best_fid, self.last_metrics, self.skipped_steps, self.bad_streak = math.inf, {}, 0, 0
        self._ae, self._stop = None, False
        g = torch.Generator().manual_seed(cfg.seed + 12345)
        self.grid_z = torch.randn(GRID * GRID, mc.z_dim, generator=g)
        self.grid_y = (torch.arange(GRID * GRID) // GRID) % mc.num_classes if mc.num_classes else None

        for sub in ('checkpoints', 'samples'):
            (self.run_dir / sub).mkdir(parents=True, exist_ok=True)
        if resume:
            self._resume()
        self._write_json(self.run_dir / 'config.json', dict(version=__version__, model=to_dict(mc),
                                                             train=to_dict(cfg), data=str(self.data_dir.resolve())))
        if self.cuda:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.backends.cudnn.benchmark = True
        if cfg.batch % max(cfg.micro_batch, 1):
            raise ValueError(f'batch {cfg.batch} is not a multiple of micro_batch {cfg.micro_batch}')
        self.micro_batch = cfg.micro_batch or (self._autotune() if self.cuda else cfg.batch)
        if cfg.compile and self.cuda:
            for block in list(self.G.blocks) + list(self.D.blocks):
                block.compile()

    # ----------------------------------------------------------------------------------------------------------------
    # One training step

    def _autocast(self):
        return torch.autocast('cuda', torch.bfloat16) if self.cuda else contextlib.nullcontext()

    def _aug(self, n):
        mc, cfg = self.model_cfg, self.train_cfg
        return sample_aug(n, mc.latent_res, cfg.aug_policy, cfg.aug_p, self.device)

    def _d_micro(self, mb: Batch, n_acc: int) -> dict:
        cfg, mc = self.train_cfg, self.model_cfg
        b, y = mb.latents.shape[0], mb.labels
        z = torch.randn(b, mc.z_dim, device=self.device)
        with torch.no_grad(), self._autocast():
            stages = self.G(z, y, return_stages=True)
        pr, pf = self._aug(b), self._aug(b)
        real = real_pyramid(apply_aug(mb.latents, pr), mc.num_scales)
        fake = pyramid([apply_aug(s, pf) for s in stages])
        n_gp = min(b, max(1, round(b * cfg.gp_fraction))) if cfg.gamma > 0 else 0
        sigma = cfg.gp_sigma
        # the perturbed copies go through the same call as the clean inputs: same kernels, same rounding
        inputs = [torch.cat([r, f, r[:n_gp] + sigma * torch.randn_like(r[:n_gp]),
                             f[:n_gp] + sigma * torch.randn_like(f[:n_gp])]) for r, f in zip(real, fake)]
        yy = None if y is None else torch.cat([y, y, y[:n_gp], y[:n_gp]])
        with self._autocast():
            logits, hidden = self.D(inputs, yy)
        l_real, l_fake, l_nreal, l_nfake = logits.split([b, b, n_gp, n_gp])
        adv = rpgan_d(l_real, l_fake)
        loss, stats = adv, {'loss_d': adv.detach()}
        if n_gp:
            r1 = finite_diff_penalty(l_real[:n_gp], l_nreal, sigma).mean()
            r2 = finite_diff_penalty(l_fake[:n_gp], l_nfake, sigma).mean()
            loss = loss + cfg.gamma / 2 * (r1 + r2)
            stats.update(r1=r1.detach(), r2=r2.detach())
        if self.encoder is not None:
            t_cls, t_tok = self.encoder(shift_images(mb.images224, pr, mc.latent_res))
            t_tok = resize_tokens(t_tok, mc.tokens_side)
            with self._autocast():
                p_cls, p_tok = self.D.repa_project(hidden[:b])
            align = repa_loss(p_cls, p_tok, t_cls, t_tok)
            loss = loss + cfg.repa_weight * align
            stats['repa'] = align.detach()
        (loss / n_acc).backward()
        return stats

    def _g_micro(self, mb: Batch, n_acc: int, update_avg: bool = True) -> dict:
        cfg, mc = self.train_cfg, self.model_cfg
        b, y = mb.latents.shape[0], mb.labels
        z = torch.randn(b, mc.z_dim, device=self.device)
        with self._autocast():
            stages = self.G(z, y, return_stages=True, update_avg=update_avg)
        pf, pr = self._aug(b), self._aug(b)
        fake = pyramid([apply_aug(s, pf) for s in stages])
        real = real_pyramid(apply_aug(mb.latents, pr), mc.num_scales)
        with torch.no_grad(), self._autocast():
            l_real, _ = self.D(real, y)
        with self._autocast():
            l_fake, _ = self.D(fake, y)
        adv = rpgan_g(l_real, l_fake)
        loss, stats = adv, {'loss_g': adv.detach()}
        if cfg.cons_weight > 0:
            cons = stage_consistency(stages)
            loss = loss + cfg.cons_weight * cons
            stats['cons'] = cons.detach()
        (loss / n_acc).backward()
        return stats

    def _lr_factor(self) -> float:
        """Constant, then a cosine decay to 25 % over the last third of a run of known length."""
        cfg = self.train_cfg
        if cfg.kimg:
            p = self.cur_nimg / (cfg.kimg * 1000)
        elif cfg.time_limit:
            p = self.train_seconds / cfg.time_limit
        else:
            return 1.0
        if p < 2 / 3:
            return 1.0
        t = min((p - 2 / 3) * 3, 1.0)
        return 0.25 + 0.75 * 0.5 * (1 + math.cos(math.pi * t))

    def _bad_step(self, phase: str):
        self.opt_d.zero_grad(set_to_none=True)
        self.opt_g.zero_grad(set_to_none=True)
        self.D.requires_grad_(True)
        self.skipped_steps += 1
        self.bad_streak += 1
        self._event(dict(type='skipped', phase=phase, step=self.step, kimg=self.cur_nimg / 1000))
        self.log(f'non-finite {phase} loss or gradient at step {self.step}: step skipped')
        if self.bad_streak >= MAX_BAD_STEPS:
            last = self._snapshots()
            where = f'the last snapshot is {last[-1]}' if last else 'no snapshot was written yet'
            raise TrainingDiverged(f'{self.bad_streak} consecutive steps with non-finite losses, training stopped; '
                                   f'{where}. Lower the learning rates or raise gamma.')
        return None

    def train_step(self) -> dict | None:
        """One discriminator and one generator update on `batch` images. Returns detached statistics, or None when
        the step had to be skipped (non-finite values)."""
        cfg = self.train_cfg
        n_acc = cfg.batch // self.micro_batch
        factor = self._lr_factor()
        for opt, lr in ((self.opt_g, self.base_lr[0]), (self.opt_d, self.base_lr[1])):
            for group in opt.param_groups:
                group['lr'] = lr * factor
        batches = [self.data.next(self.micro_batch) for _ in range(n_acc)]
        zero = torch.zeros((), device=self.device)
        stats = {k: zero for k in STAT_KEYS}

        self.D.requires_grad_(True)
        for mb in batches:
            for k, v in self._d_micro(mb, n_acc).items():
                stats[k] = stats[k] + v / n_acc
        clip = cfg.grad_clip if cfg.grad_clip > 0 else math.inf
        stats['grad_d'] = torch.nn.utils.clip_grad_norm_(self.D.parameters(), clip)
        if not bool(torch.isfinite(stats['loss_d'] + stats['r1'] + stats['r2'] + stats['repa'] + stats['grad_d'])):
            return self._bad_step('discriminator')
        self.opt_d.step()
        self.opt_d.zero_grad(set_to_none=True)

        self.D.requires_grad_(False)
        for mb in batches:
            for k, v in self._g_micro(mb, n_acc).items():
                stats[k] = stats[k] + v / n_acc
        stats['grad_g'] = torch.nn.utils.clip_grad_norm_(self.G.parameters(), clip)
        if not bool(torch.isfinite(stats['loss_g'] + stats['cons'] + stats['grad_g'])):
            return self._bad_step('generator')
        self.opt_g.step()
        self.opt_g.zero_grad(set_to_none=True)
        self.D.requires_grad_(True)

        self.cur_nimg += cfg.batch
        self.step += 1
        self.bad_streak = 0
        self.ema.update(self.G, ema_decay(cfg.batch, self.cur_nimg, cfg.ema_halflife_kimg, cfg.ema_rampup))
        return stats

    # ----------------------------------------------------------------------------------------------------------------
    # The loop

    def _done(self) -> bool:
        cfg = self.train_cfg
        return bool((cfg.kimg and self.cur_nimg >= cfg.kimg * 1000) or
                    (cfg.time_limit and self.train_seconds >= cfg.time_limit))

    def run(self, max_steps: int | None = None) -> dict:
        """Trains until the length of the run (`kimg` / `time_limit`), `max_steps` steps in this call, or Ctrl+C.
        Without a length the run goes on until interrupted. A snapshot is written at the end in every case."""
        cfg = self.train_cfg
        old_handler = None
        if threading.current_thread() is threading.main_thread():
            old_handler = signal.signal(signal.SIGINT, self._on_sigint)
        n_params = (sum(p.numel() for p in self.G.parameters()), sum(p.numel() for p in self.D.parameters()))
        self._event(dict(type='start', step=self.step, kimg=self.cur_nimg / 1000, micro_batch=self.micro_batch,
                         params_g=n_params[0], params_d=n_params[1], device=str(self.device)))
        self.log(f'training {self.run_dir}: G {n_params[0] / 1e6:.1f}M, D {n_params[1] / 1e6:.1f}M parameters, '
                 f'batch {cfg.batch} = {cfg.batch // self.micro_batch} x {self.micro_batch} on {self.device}, '
                 f'from {self.cur_nimg / 1000:.1f} kimg')
        acc, acc_steps, acc_imgs = {}, 0, 0
        tick_start = last_snapshot = time.time()
        next_sample = self._next_multiple(cfg.sample_every_kimg)
        next_eval = self._next_multiple(cfg.eval_every_kimg)
        steps, interrupted = 0, False
        try:
            while not self._done():
                if max_steps is not None and steps >= max_steps:
                    break
                if self._stop:
                    interrupted = True
                    break
                t0 = time.time()
                stats = self.train_step()
                self.train_seconds += time.time() - t0
                steps += 1
                if stats is None:
                    continue
                for k, v in stats.items():
                    acc[k] = acc.get(k, 0) + v
                acc_steps, acc_imgs = acc_steps + 1, acc_imgs + cfg.batch
                now = time.time()
                if now - tick_start >= TICK_SECONDS or self.step == 1 or self._done():
                    self._tick(acc, acc_steps, acc_imgs, now - tick_start)
                    acc, acc_steps, acc_imgs, tick_start = {}, 0, 0, time.time()
                if next_sample is not None and self.cur_nimg >= next_sample:
                    self.save_samples()
                    next_sample = self._next_multiple(cfg.sample_every_kimg)
                if next_eval is not None and self.cur_nimg >= next_eval:
                    self.evaluate()
                    next_eval = self._next_multiple(cfg.eval_every_kimg)
                if time.time() - last_snapshot >= cfg.snapshot_minutes * 60:
                    self.snapshot()
                    last_snapshot = time.time()
        finally:
            if old_handler is not None:
                signal.signal(signal.SIGINT, old_handler)
        if acc_steps:
            self._tick(acc, acc_steps, acc_imgs, time.time() - tick_start)
        finished = self._done()
        if finished and next_eval is not None and self.last_metrics.get('kimg') != self.cur_nimg / 1000:
            self.evaluate()
        self.snapshot()
        summary = dict(steps=self.step, kimg=self.cur_nimg / 1000, skipped_steps=self.skipped_steps,
                       interrupted=interrupted, finished=finished, seconds=self.train_seconds,
                       best_fid=None if math.isinf(self.best_fid) else self.best_fid, metrics=self.last_metrics)
        self._event(dict(type='end', **summary))
        return summary

    def _on_sigint(self, signum, frame):
        if self._stop:  # a second Ctrl+C aborts immediately
            raise KeyboardInterrupt
        self._stop = True
        self.log('stopping after this step and writing a snapshot (Ctrl+C again to abort)')

    def _next_multiple(self, every_kimg: float):
        if not every_kimg or every_kimg <= 0:
            return None
        every = every_kimg * 1000
        return (math.floor(self.cur_nimg / every + 1e-9) + 1) * every

    def _tick(self, acc: dict, steps: int, imgs: int, seconds: float):
        values = {k: float(v) / steps for k, v in acc.items()}
        mem = torch.cuda.max_memory_allocated(self.device) / 2 ** 30 if self.cuda else 0.0
        if self.cuda:
            torch.cuda.reset_peak_memory_stats(self.device)
        speed = imgs / max(seconds, 1e-9)
        lr_g, lr_d = (g.param_groups[0]['lr'] for g in (self.opt_g, self.opt_d))
        rec = dict(type='tick', step=self.step, kimg=self.cur_nimg / 1000, img_per_s=speed, mem_gb=mem,
                   lr_g=lr_g, lr_d=lr_d, train_hours=self.train_seconds / 3600, **values)
        self._event(rec)
        parts = [f'kimg {rec["kimg"]:9.2f}', f'{speed:7.1f} img/s', f'D {values.get("loss_d", 0):.3f}',
                 f'G {values.get("loss_g", 0):.3f}']
        for k in ('r1', 'r2', 'repa', 'cons'):
            if values.get(k):
                parts.append(f'{k} {values[k]:.3g}')
        parts += [f'|gD| {values.get("grad_d", 0):.2f}', f'|gG| {values.get("grad_g", 0):.2f}']
        if self.cuda:
            parts.append(f'{mem:.1f} GB')
        self.log(' | '.join(parts))

    # ----------------------------------------------------------------------------------------------------------------
    # Images and metrics

    @property
    def autoencoder(self):
        if self._ae is None:
            self._ae = load_autoencoder(self.data.meta['autoencoder'], self.device)
        return self._ae

    @torch.no_grad()
    def generate_uint8(self, z, y=None, truncation: float = 1.0) -> torch.Tensor:
        with self._autocast():
            lat = self.ema.model(z.to(self.device), None if y is None else y.to(self.device), truncation)
        return to_uint8(self.autoencoder.decode(self.data.norm.denormalize(lat.float())))

    def save_samples(self):
        imgs = torch.cat([self.generate_uint8(z, y) for z, y in zip(
            self.grid_z.split(16), self.grid_y.split(16) if self.grid_y is not None else [None] * 4)])
        n, c, h, w = imgs.shape
        grid = imgs.view(GRID, GRID, c, h, w).permute(0, 3, 1, 4, 2).reshape(GRID * h, GRID * w, c)
        path = self.run_dir / 'samples' / f'kimg-{int(self.cur_nimg // 1000):06d}-step-{self.step:07d}.jpg'
        Image.fromarray(grid.cpu().numpy()).save(path, quality=90)

    def evaluate(self) -> dict | None:
        """FID and KID of `eval_num` images of the EMA generator against the dataset statistics."""
        if 'inception' not in self.data.meta.get('stats', []):
            return None
        if self.extractors is None:
            self.extractors = default_extractors(self.device, ('inception',))
        cfg, mc = self.train_cfg, self.model_cfg
        g = torch.Generator().manual_seed(cfg.seed + 999)
        labels = self.data.labels

        def batches():
            for start in range(0, cfg.eval_num, EVAL_BATCH):
                n = min(EVAL_BATCH, cfg.eval_num - start)
                z = torch.randn(n, mc.z_dim, generator=g)
                y = labels[torch.randint(0, len(labels), (n,), generator=g)] if labels is not None else None
                yield self.generate_uint8(z, y)

        t0 = time.time()
        result = evaluate(batches(), self.data_dir, metrics=('fid', 'kid'), extractors=self.extractors)
        result.update(kimg=self.cur_nimg / 1000, step=self.step, seconds=time.time() - t0)
        self.last_metrics = result
        self._event(dict(type='eval', **result))
        self.log(f'eval at {result["kimg"]:.1f} kimg: FID {result["fid"]:.2f}, KID {result["kid"] * 1000:.2f} x1e-3 '
                 f'({result["num"]} images, {result["seconds"]:.0f} s)')
        if result['fid'] < self.best_fid:
            self.best_fid = result['fid']
            self.export(self.run_dir / 'model-best.safetensors')
        return result

    # ----------------------------------------------------------------------------------------------------------------
    # Files

    def export(self, path):
        meta = self.data.meta
        save_model(path, self.ema.model, self.model_cfg, dict(
            version=__version__, autoencoder=meta['autoencoder'], resolution=meta['resolution'],
            latent_mean=meta['latent_mean'], latent_std=meta['latent_std'], classes=meta['classes'],
            kimg=self.cur_nimg / 1000, step=self.step, metrics=self.last_metrics))

    def _snapshots(self) -> list[Path]:
        return sorted((self.run_dir / 'checkpoints').glob('state-*.pt'))

    def snapshot(self):
        state = dict(G=self.G.state_dict(), D=self.D.state_dict(), G_ema=self.ema.state_dict(),
                     opt_g=self.opt_g.state_dict(), opt_d=self.opt_d.state_dict(), data=self.data.state_dict(),
                     cur_nimg=self.cur_nimg, step=self.step, train_seconds=self.train_seconds,
                     best_fid=self.best_fid, skipped_steps=self.skipped_steps, last_metrics=self.last_metrics,
                     rng_cpu=torch.get_rng_state(), version=__version__)
        if self.cuda:
            state['rng_cuda'] = torch.cuda.get_rng_state_all()
        path = self.run_dir / 'checkpoints' / f'state-{self.step:08d}.pt'
        tmp = path.with_name(path.name + '.tmp')
        torch.save(state, tmp)
        os.replace(tmp, path)
        for old in self._snapshots()[:-max(self.train_cfg.keep_snapshots, 1)]:
            old.unlink()
        self.export(self.run_dir / 'model.safetensors')

    def _resume(self):
        snapshots = self._snapshots()
        if not snapshots:
            return
        state = torch.load(snapshots[-1], map_location='cpu', weights_only=True)
        self.G.load_state_dict(state['G'])
        self.D.load_state_dict(state['D'])
        self.ema.load_state_dict(state['G_ema'])
        self.opt_g.load_state_dict(state['opt_g'])
        self.opt_d.load_state_dict(state['opt_d'])
        self.data.load_state_dict(state['data'])
        self.cur_nimg, self.step = state['cur_nimg'], state['step']
        self.train_seconds, self.best_fid = state['train_seconds'], state['best_fid']
        self.skipped_steps, self.last_metrics = state['skipped_steps'], state['last_metrics']
        torch.set_rng_state(state['rng_cpu'])
        if self.cuda and 'rng_cuda' in state:
            torch.cuda.set_rng_state_all(state['rng_cuda'])
        self.log(f'resumed {snapshots[-1]} at {self.cur_nimg / 1000:.1f} kimg')

    def _move_old_snapshots(self):
        old = self.run_dir / 'checkpoints'
        if old.exists():
            target = self.run_dir / f'checkpoints-replaced-{time.strftime("%Y%m%d-%H%M%S")}'
            shutil.move(str(old), str(target))
            self.log(f'model configuration changed: previous snapshots moved to {target}')

    def _event(self, rec: dict):
        rec = dict(time=time.time(), **rec)
        with open(self.run_dir / 'log.jsonl', 'a') as f:
            f.write(json.dumps(rec, default=float) + '\n')

    @staticmethod
    def _read_json(path: Path):
        return json.loads(path.read_text()) if path.exists() else None

    @staticmethod
    def _write_json(path: Path, obj):
        tmp = path.with_name(path.name + '.tmp')
        tmp.write_text(json.dumps(obj, indent=2))
        os.replace(tmp, path)

    # ----------------------------------------------------------------------------------------------------------------
    # Micro-batch size

    def _trial(self, m: int) -> float:
        """Peak reserved GPU memory of one micro-batch of `m` images through both phases."""
        mc = self.model_cfg
        lat = torch.randn(m, mc.latent_channels, mc.latent_res, mc.latent_res, device=self.device)
        labels = torch.randint(0, mc.num_classes, (m,), device=self.device) if mc.num_classes else None
        imgs = torch.randint(0, 256, (m, 3, 224, 224), dtype=torch.uint8, device=self.device) if self.encoder else None
        mb = Batch(lat, labels, imgs, None, None)
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats(self.device)
        try:
            self.D.requires_grad_(True)
            self._d_micro(mb, 1)
            self.D.requires_grad_(False)
            self._g_micro(mb, 1, update_avg=False)
            torch.cuda.synchronize(self.device)
            peak = torch.cuda.max_memory_reserved(self.device)
        except torch.cuda.OutOfMemoryError:
            peak = math.inf
        finally:
            self.D.requires_grad_(True)
            self.opt_d.zero_grad(set_to_none=True)
            self.opt_g.zero_grad(set_to_none=True)
            torch.cuda.empty_cache()
        return peak

    def _autotune(self) -> int:
        cfg = self.train_cfg
        free, _ = torch.cuda.mem_get_info(self.device)
        budget = MEMORY_BUDGET * (free + torch.cuda.memory_reserved(self.device))
        n_params = sum(p.numel() for p in self.G.parameters()) + sum(p.numel() for p in self.D.parameters())
        optimizer_state = 0 if self.opt_g.state else 8 * n_params  # Adam moments are created at the first step
        candidates = [m for m in (1, 2, 4, 8, 16, 32, 64, 128) if m <= cfg.batch and cfg.batch % m == 0]
        rng = torch.get_rng_state(), torch.cuda.get_rng_state_all()
        small = candidates[min(2, len(candidates) - 1)]
        peaks = {small: self._trial(small) + optimizer_state}
        big = next((m for m in candidates if m > small), None)
        if big is not None:
            peaks[big] = self._trial(big) + optimizer_state
        if big is not None and math.isfinite(peaks[big]):
            per_image = (peaks[big] - peaks[small]) / (big - small)
            base = peaks[small] - per_image * small
            fits = [m for m in candidates if base + per_image * m <= budget]
        else:
            fits = [m for m in candidates if m <= small and peaks[small] <= budget]
        choice = fits[-1] if fits else 1
        while choice not in peaks and choice > 1:  # verify the extrapolation
            peaks[choice] = self._trial(choice) + optimizer_state
            if peaks[choice] <= budget:
                break
            choice = candidates[candidates.index(choice) - 1]
        torch.set_rng_state(rng[0])
        torch.cuda.set_rng_state_all(rng[1])
        self.log(f'micro-batch {choice}: ~{peaks.get(choice, 0) / 2 ** 30:.1f} GB of a {budget / 2 ** 30:.1f} GB budget')
        return choice


__all__ = ['Trainer', 'ConfigMismatch', 'TrainingDiverged']
