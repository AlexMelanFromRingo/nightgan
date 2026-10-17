import numpy as np
import pytest
from PIL import Image

import nightgan.metrics
import nightgan.trainer
from conftest import mean_color, write_images
from nightgan import load
from nightgan.cli import main, parse_duration, parse_seeds

TINY = ['--dim', '32', '--depth', '4', '--heads', '2', '--z-dim', '16', '--cmap-dim', '32', '--batch', '8',
        '--micro-batch', '4', '--no-compile', '--device', 'cpu', '--repa-encoder', 'none']


@pytest.fixture(autouse=True)
def fake_metric_networks(monkeypatch):
    fake = lambda device, names=('inception', 'dinov2'): {n: mean_color for n in names}
    monkeypatch.setattr(nightgan.metrics, 'default_extractors', fake)
    monkeypatch.setattr(nightgan.trainer, 'default_extractors', fake)


def test_parse_duration():
    assert parse_duration('10h') == 36000
    assert parse_duration('90m') == 5400
    assert parse_duration('2d') == 172800
    assert parse_duration('3600s') == parse_duration('3600') == 3600
    assert parse_duration('1.5h') == 5400
    with pytest.raises(ValueError):
        parse_duration('soon')


def test_parse_seeds():
    assert parse_seeds('0-3,7') == [0, 1, 2, 3, 7]
    assert parse_seeds('5') == [5]


@pytest.fixture(scope='module')
def trained(tmp_path_factory):
    import nightgan.metrics as m, nightgan.trainer as t
    fake = lambda device, names=('inception', 'dinov2'): {n: mean_color for n in names}
    saved = m.default_extractors, t.default_extractors
    m.default_extractors = t.default_extractors = fake
    try:
        root = tmp_path_factory.mktemp('cli')
        write_images(root / 'src' / 'a', 12, seed=1)
        write_images(root / 'src' / 'b', 12, seed=2)
        assert main(['prepare', str(root / 'src'), str(root / 'data'), '--resolution', '32', '--autoencoder', 'pixel',
                     '--labels', 'dirs', '--device', 'cpu', '--workers', '2']) == 0
        assert main(['train', str(root / 'data'), str(root / 'run'), '--kimg', '0.04', '--eval-every-kimg', '0.04',
                     '--eval-num', '16', '--sample-every-kimg', '0.04'] + TINY) == 0
    finally:
        m.default_extractors, t.default_extractors = saved
    return root


def test_generate_writes_one_image_per_seed(trained, tmp_path):
    assert main(['generate', str(trained / 'run' / 'model.safetensors'), str(tmp_path / 'out'), '--seeds', '0-3',
                 '--device', 'cpu']) == 0
    files = sorted((tmp_path / 'out').glob('*.png'))
    assert [f.name for f in files] == ['seed00000.png', 'seed00001.png', 'seed00002.png', 'seed00003.png']
    assert Image.open(files[0]).size == (32, 32)


def test_generate_a_grid_of_one_class(trained, tmp_path):
    assert main(['generate', str(trained / 'run'), str(tmp_path / 'out'), '--n', '6', '--class', '1', '--grid',
                 '--truncation', '0.7', '--device', 'cpu']) == 0
    assert Image.open(tmp_path / 'out' / 'grid.png').size == (3 * 32, 2 * 32)


def test_a_class_out_of_range_is_a_clear_error(trained, tmp_path, capsys):
    assert main(['generate', str(trained / 'run'), str(tmp_path / 'out'), '--class', '5', '--device', 'cpu']) == 2
    err = capsys.readouterr().err
    assert 'class 5' in err and '2 classes' in err


def test_interpolate_writes_a_gif(trained, tmp_path):
    out = tmp_path / 'walk.gif'
    assert main(['interpolate', str(trained / 'run'), str(out), '--seeds', '1,2', '--frames', '4',
                 '--device', 'cpu']) == 0
    assert Image.open(out).n_frames == 8


def test_eval_a_model_and_a_folder_of_images(trained, tmp_path, capsys):
    assert main(['eval', str(trained / 'run' / 'model.safetensors'), str(trained / 'data'), '--metrics', 'kid,fid',
                 '--num', '16', '--device', 'cpu']) == 0
    out = capsys.readouterr().out
    assert 'kid' in out and 'fid' in out
    main(['generate', str(trained / 'run'), str(tmp_path / 'imgs'), '--seeds', '0-7', '--device', 'cpu'])
    capsys.readouterr()
    assert main(['eval', str(trained / 'data'), '--images', str(tmp_path / 'imgs'), '--metrics', 'fid',
                 '--device', 'cpu']) == 0
    assert 'fid' in capsys.readouterr().out


def test_training_again_resumes_and_a_changed_model_is_refused(trained, capsys):
    run = trained / 'run'
    assert main(['train', str(trained / 'data'), str(run), '--kimg', '0.048'] + TINY) == 0
    assert 'resumed' in capsys.readouterr().out
    changed = [a if a != '32' else '48' for a in TINY]
    assert main(['train', str(trained / 'data'), str(run), '--kimg', '0.1'] + changed) == 2
    err = capsys.readouterr().err
    assert 'dim: 32 -> 48' in err and '--force-config' in err


def test_resuming_with_runtime_flags_only(trained, capsys):
    run = trained / 'run'
    runtime = ['--batch', '8', '--micro-batch', '4', '--no-compile', '--device', 'cpu', '--repa-encoder', 'none']
    # no model flags at all: the saved model configuration is used
    assert main(['train', str(trained / 'data'), str(run), '--kimg', '0.056', '--checkpointing'] + runtime) == 0
    assert 'resumed' in capsys.readouterr().out
    # one model flag on top of the saved configuration, equal to the saved value
    assert main(['train', str(trained / 'data'), str(run), '--kimg', '0.064', '--depth', '4'] + runtime) == 0


def test_python_api(trained):
    pipe = load(trained / 'run' / 'model.safetensors', device='cpu')
    assert pipe.classes == ['a', 'b'] and pipe.resolution == 32
    imgs = pipe.generate(seeds=[3, 4], class_idx=0)
    assert len(imgs) == 2 and imgs[0].size == (32, 32)
    again = pipe.generate(seeds=[3])
    assert np.array_equal(np.asarray(again[0]), np.asarray(pipe.generate(seeds=[3])[0]))  # seeds are reproducible


def test_missing_dataset_is_a_clear_error(tmp_path, capsys):
    assert main(['train', str(tmp_path / 'nothing'), str(tmp_path / 'run')] + TINY) == 2
    assert 'prepare' in capsys.readouterr().err


def test_a_resumed_run_keeps_its_length_unless_a_new_one_is_given():
    from argparse import Namespace
    from nightgan.cli import train_length
    saved = {'train': {'kimg': 500.0, 'time_limit': None}}
    assert train_length(Namespace(kimg=None, time=None), saved) == (500.0, None)
    assert train_length(Namespace(kimg=None, time='2h'), saved) == (None, 7200.0)
    assert train_length(Namespace(kimg=None, time=None), None) == (None, None)
