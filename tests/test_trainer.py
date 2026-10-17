import json

import pytest
import torch
from safetensors.torch import load_file

import nightgan.trainer as trainer_module
from conftest import mean_color, write_images
from nightgan.config import ModelConfig, TrainConfig
from nightgan.data import prepare
from nightgan.ema import EMA, ema_decay
from nightgan.encoders import RandomConvEncoder
from nightgan.generator import Generator
from nightgan.trainer import ConfigMismatch, Trainer

STEP_KIMG = 8 / 1000   # one optimizer step of batch 8


@pytest.fixture(scope='module')
def dataset(tmp_path_factory):
    root = tmp_path_factory.mktemp('ds')
    write_images(root / 'src', 32, seed=7)
    prepare(root / 'src', root / 'data', resolution=32, autoencoder='pixel', device='cpu', batch=16, workers=2,
            extractors={'inception': mean_color}, progress=False)
    return root / 'data'


def model_cfg(**kw):
    base = dict(latent_channels=3, latent_res=32, patch=2, dim=32, depth=4, heads=2, z_dim=16, repa_dim=64,
                cmap_dim=32)
    return ModelConfig(**{**base, **kw})


def train_cfg(**kw):
    base = dict(batch=8, micro_batch=4, compile=False, kimg=10 * STEP_KIMG, sample_every_kimg=5 * STEP_KIMG,
                eval_every_kimg=5 * STEP_KIMG, eval_num=16, snapshot_minutes=1e6, repa_encoder='random')
    return TrainConfig(**{**base, **kw})


def make(dataset, run_dir, mcfg=None, tcfg=None, **kw):
    return Trainer(dataset, run_dir, mcfg or model_cfg(), tcfg or train_cfg(), device='cpu', encoder=RandomConvEncoder(),
                   extractors={'inception': mean_color}, log=lambda *a: None, **kw)


def test_ema_decay_from_half_life():
    assert ema_decay(512, 10 ** 9, 350, 0.05) == pytest.approx(0.5 ** (512 / 350000))
    assert ema_decay(64, 1000, 350, 0.05) == pytest.approx(0.5 ** (64 / 50))   # ramp-up: half-life 5 % of images
    assert ema_decay(64, 0, 350, 0.05) == 0.0
    assert ema_decay(64, 0, 350, None) == pytest.approx(0.5 ** (64 / 350000))


def test_ema_update_moves_towards_the_model_and_copies_buffers():
    G = Generator(model_cfg())
    ema = EMA(G)
    with torch.no_grad():
        for p in G.parameters():
            p.add_(1.0)
        G.c_avg.fill_(3.0)
    before = [p.clone() for p in ema.model.parameters()]
    ema.update(G, 0.75)
    for b, e, p in zip(before, ema.model.parameters(), G.parameters()):
        torch.testing.assert_close(e, 0.75 * b + 0.25 * p)
    assert (ema.model.c_avg == 3.0).all()


def test_training_runs_and_writes_logs_samples_snapshots_and_the_model(dataset, tmp_path):
    summary = make(dataset, tmp_path / 'run').run()
    run = tmp_path / 'run'
    assert summary['steps'] == 10 and summary['kimg'] == pytest.approx(10 * STEP_KIMG)
    lines = [json.loads(l) for l in (run / 'log.jsonl').read_text().splitlines()]
    ticks = [l for l in lines if l['type'] == 'tick']
    assert ticks and all(torch.isfinite(torch.tensor(t['loss_d'])) for t in ticks)
    assert {'loss_g', 'r1', 'r2', 'repa', 'cons', 'grad_d', 'grad_g'} <= set(ticks[-1])
    evals = [l for l in lines if l['type'] == 'eval']
    assert evals and 'fid' in evals[-1] and 'kid' in evals[-1]
    assert sorted(p.name for p in (run / 'samples').iterdir())
    assert list((run / 'checkpoints').glob('state-*.pt'))
    meta = json.loads((run / 'model.json').read_text())
    G = Generator(ModelConfig(**meta['model']))
    G.load_state_dict(load_file(run / 'model.safetensors'))
    assert (run / 'model-best.safetensors').exists()
    assert meta['autoencoder'] == 'pixel' and len(meta['latent_mean']) == 3


def test_resumed_training_equals_uninterrupted_training(dataset, tmp_path):
    straight = make(dataset, tmp_path / 'a')
    straight.run()
    split = make(dataset, tmp_path / 'b')
    split.run(max_steps=4)
    assert split.step == 4
    resumed = make(dataset, tmp_path / 'b')
    assert resumed.step == 4
    resumed.run()
    for (name, a), b in zip(straight.ema.model.state_dict().items(), resumed.ema.model.state_dict().values()):
        torch.testing.assert_close(a, b, msg=name)


def test_a_different_model_config_is_refused_but_training_options_may_change(dataset, tmp_path):
    make(dataset, tmp_path / 'run').run(max_steps=1)
    with pytest.raises(ConfigMismatch, match='dim'):
        make(dataset, tmp_path / 'run', mcfg=model_cfg(dim=48))
    t = make(dataset, tmp_path / 'run', tcfg=train_cfg(time_limit=60.0, eval_every_kimg=1.0))
    assert t.step == 1 and t.train_cfg.time_limit == 60.0
    forced = make(dataset, tmp_path / 'run', mcfg=model_cfg(dim=48), force_config=True)
    assert forced.step == 0 and forced.model_cfg.dim == 48


def test_gradient_checkpointing_is_a_training_option(dataset, tmp_path):
    t = make(dataset, tmp_path / 'run', tcfg=train_cfg(checkpointing=True))
    assert t.G.checkpointing and not hasattr(t.model_cfg, 'checkpointing')
    t.run(max_steps=1)
    assert make(dataset, tmp_path / 'run').step == 1   # resuming without it is not a model change


@pytest.mark.parametrize('patch', [1, 4])
def test_repa_targets_follow_a_non_default_token_grid(dataset, tmp_path, patch):
    t = make(dataset, tmp_path / 'run', mcfg=model_cfg(patch=patch))
    assert t.run(max_steps=1)['steps'] == 1


def test_model_config_must_match_the_dataset(dataset, tmp_path):
    with pytest.raises(ValueError, match='latent_channels'):
        make(dataset, tmp_path / 'run', mcfg=model_cfg(latent_channels=4))


def test_a_non_finite_loss_skips_the_step(dataset, tmp_path, monkeypatch):
    calls = {'n': 0}
    real_rpgan_d = trainer_module.rpgan_d

    def flaky(real, fake):
        calls['n'] += 1
        loss = real_rpgan_d(real, fake)
        return loss * float('nan') if calls['n'] <= 2 else loss   # both micro-batches of the first step

    monkeypatch.setattr(trainer_module, 'rpgan_d', flaky)
    t = make(dataset, tmp_path / 'run', tcfg=train_cfg(kimg=3 * STEP_KIMG))
    summary = t.run()
    assert summary['skipped_steps'] == 1 and summary['steps'] == 3
    assert all(torch.isfinite(p).all() for p in t.D.parameters())


def test_repeated_non_finite_losses_stop_training(dataset, tmp_path, monkeypatch):
    monkeypatch.setattr(trainer_module, 'rpgan_d', lambda real, fake: (real - fake).mean() * float('nan'))
    with pytest.raises(trainer_module.TrainingDiverged, match='snapshot'):
        make(dataset, tmp_path / 'run').run()
