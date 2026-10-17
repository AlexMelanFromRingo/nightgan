import json

import numpy as np
import pytest
import torch
from PIL import Image

from conftest import mean_color, write_images
from nightgan.autoencoder import PixelAE
from nightgan.data import DatasetError, TrainSet, load_meta, load_square, prepare, scan_images

FAST = dict(autoencoder='pixel', batch=8, workers=2, device='cpu')


def prepare_fast(src, dst, **kw):
    return prepare(src, dst, resolution=kw.pop('resolution', 32),
                   extractors=kw.pop('extractors', {'inception': mean_color}), **{**FAST, **kw})


# --- image decoding (Review Focus 1) ---

def test_transparent_pixels_become_white(tmp_path):
    rgba = np.zeros((20, 20, 4), np.uint8)
    rgba[..., 0] = 200
    rgba[:10, :, 3] = 255     # top half opaque red, bottom half fully transparent
    Image.fromarray(rgba, 'RGBA').save(tmp_path / 'a.png')
    out = load_square(tmp_path / 'a.png', 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    assert tuple(out[2, 8]) == (200, 0, 0)
    assert tuple(out[13, 8]) == (255, 255, 255)


def test_palette_with_transparency_becomes_white(tmp_path):
    im = Image.new('P', (16, 16), 0)
    im.putpalette([0, 0, 255] + [0, 0, 0] * 255)
    im.info['transparency'] = 0
    im.save(tmp_path / 'p.png', transparency=0)
    assert tuple(load_square(tmp_path / 'p.png', 16)[8, 8]) == (255, 255, 255)


@pytest.mark.parametrize('mode', ['L', 'CMYK', 'I;16'])
def test_other_modes_become_rgb(tmp_path, mode):
    if mode == 'I;16':
        im = Image.fromarray(np.full((24, 24), 65535 // 2, np.uint16))
        path = tmp_path / 'g16.png'
    elif mode == 'CMYK':
        im = Image.new('CMYK', (24, 24), (0, 255, 255, 0))  # red
        path = tmp_path / 'c.jpg'
    else:
        im = Image.new('L', (24, 24), 77)
        path = tmp_path / 'l.png'
    im.save(path)
    out = load_square(path, 16)
    assert out.shape == (16, 16, 3) and out.dtype == np.uint8
    if mode == 'I;16':
        assert abs(int(out[8, 8, 0]) - 127) <= 1   # scaled from 16 bits, not clipped to 255
    if mode == 'CMYK':
        assert out[8, 8, 0] > 200 and out[8, 8, 1] < 60
    if mode == 'L':
        assert tuple(out[8, 8]) == (77, 77, 77)


def test_exif_orientation_is_applied(tmp_path):
    img = np.zeros((20, 40, 3), np.uint8)
    img[:, :20] = (255, 0, 0)   # stored: left red, right blue
    img[:, 20:] = (0, 0, 255)
    exif = Image.Exif()
    exif[0x0112] = 6           # display rotated 90 degrees clockwise: red on top
    Image.fromarray(img).save(tmp_path / 'r.jpg', exif=exif, quality=95)
    out = load_square(tmp_path / 'r.jpg', 16)
    assert out[2, 13, 0] > 200 and out[2, 13, 2] < 60     # top: red
    assert out[13, 2, 2] > 200 and out[13, 2, 0] < 60     # bottom: blue


def test_non_square_images_are_center_cropped(tmp_path):
    img = np.zeros((20, 60, 3), np.uint8)
    img[:, 20:40] = 255        # only the middle third is white
    Image.fromarray(img).save(tmp_path / 'w.png')
    assert load_square(tmp_path / 'w.png', 10).min() == 255


# --- scanning (Review Focus 2) ---

def test_non_images_and_corrupt_files_are_skipped(tmp_path, image_folder):
    (image_folder / 'notes.txt').write_text('hello')
    good = (image_folder / '0000.png').read_bytes()
    (image_folder / 'broken.png').write_bytes(good[:60])
    items, skipped = scan_images(image_folder)
    assert len(items) == 25 and [p.name for p, _ in skipped] == ['notes.txt']
    summary = prepare_fast(image_folder, tmp_path / 'data')
    assert summary['count'] == 24
    assert [p.name for p, _ in summary['skipped']] == ['broken.png', 'notes.txt']


def test_class_folders_give_sorted_labels(tmp_path):
    write_images(tmp_path / 'src' / 'dogs', 3, seed=1)
    write_images(tmp_path / 'src' / 'cats', 2, seed=2)
    summary = prepare_fast(tmp_path / 'src', tmp_path / 'data', labels='dirs')
    assert summary['classes'] == ['cats', 'dogs']
    labels = np.load(tmp_path / 'data' / 'labels.npy')
    assert sorted(labels.tolist()) == [0, 0, 1, 1, 1]
    meta = load_meta(tmp_path / 'data')
    assert meta['classes'] == ['cats', 'dogs']


def test_class_folders_require_files_inside_folders(tmp_path):
    write_images(tmp_path / 'src', 2)
    with pytest.raises(DatasetError, match='class'):
        prepare_fast(tmp_path / 'src', tmp_path / 'data', labels='dirs')


# --- prepared dataset ---

def test_prepared_arrays(tmp_path, image_folder):
    prepare_fast(image_folder, tmp_path / 'data')
    d = tmp_path / 'data'
    meta = json.loads((d / 'meta.json').read_text())
    lat = np.load(d / 'latents.npy')
    assert lat.dtype == np.float16 and lat.shape[1:] == (2, 6, 32, 32) and meta['count'] == 24
    np.testing.assert_array_equal(lat[:24, 1], lat[:24, 0, :, :, ::-1])  # second orientation is the mirror image
    imgs = np.load(d / 'images224.npy', mmap_mode='r')
    assert imgs.dtype == np.uint8 and imgs.shape[1:] == (224, 224, 3)
    stats = np.load(d / 'stats' / 'inception.npz')
    assert stats['mu'].shape == (3,) and stats['sigma'].shape == (3, 3) and stats['features'].shape == (24, 3)
    assert len(meta['latent_mean']) == 3 and all(s > 0 for s in meta['latent_std'])
    assert meta['latent'] == {'channels': 3, 'resolution': 32, 'factor': 1}


def test_existing_dataset_is_not_overwritten(tmp_path, image_folder):
    prepare_fast(image_folder, tmp_path / 'data')
    with pytest.raises(DatasetError, match='already'):
        prepare_fast(image_folder, tmp_path / 'data')
    prepare_fast(image_folder, tmp_path / 'data', overwrite=True)


class FailingAE(PixelAE):
    def __init__(self):
        self.calls = 0

    def encode_moments(self, images):
        self.calls += 1
        if self.calls > 2:
            raise KeyboardInterrupt
        return super().encode_moments(images)


def test_interrupted_preparation_is_not_usable(tmp_path, image_folder):
    with pytest.raises(KeyboardInterrupt):
        prepare_fast(image_folder, tmp_path / 'data', ae=FailingAE())
    assert not (tmp_path / 'data' / 'meta.json').exists()
    with pytest.raises(DatasetError, match='prepare'):
        TrainSet(tmp_path / 'data', 'cpu')


# --- training batches ---

@pytest.fixture
def small_set(tmp_path):
    write_images(tmp_path / 'src', 40, seed=3)
    prepare_fast(tmp_path / 'src', tmp_path / 'data')
    return tmp_path / 'data'


def test_dataset_smaller_than_the_batch(small_set):
    ts = TrainSet(small_set, 'cpu', seed=0)
    batch = ts.next(256)
    assert batch.latents.shape == (256, 3, 32, 32) and batch.images224.shape == (256, 3, 224, 224)
    assert set(batch.indices.tolist()) == set(range(40))
    assert batch.labels is None and ts.num_classes == 0


def test_latents_are_normalized_samples(small_set):
    ts = TrainSet(small_set, 'cpu', seed=0)
    z = ts.next(512).latents
    assert abs(z.mean().item()) < 0.1 and abs(z.std().item() - 1) < 0.1


def test_flipped_orientation_flips_the_repa_image(small_set):
    ts = TrainSet(small_set, 'cpu', seed=0)
    b = ts.next(64)
    raw = np.load(small_set / 'images224.npy', mmap_mode='r')
    for i in range(64):
        expected = torch.from_numpy(np.array(raw[b.indices[i]])).permute(2, 0, 1)
        if b.orient[i]:
            expected = expected.flip(-1)
        assert torch.equal(b.images224[i], expected)


def test_batches_are_reproducible_and_resumable(small_set):
    a, b = TrainSet(small_set, 'cpu', seed=5), TrainSet(small_set, 'cpu', seed=5)
    for _ in range(3):
        x, y = a.next(16), b.next(16)
        assert torch.equal(x.indices, y.indices) and torch.equal(x.orient, y.orient)
    state = a.state_dict()
    expected = a.next(16)
    c = TrainSet(small_set, 'cpu', seed=5)
    c.load_state_dict(state)
    got = c.next(16)
    assert torch.equal(got.indices, expected.indices) and torch.equal(got.orient, expected.orient)
    assert not torch.equal(TrainSet(small_set, 'cpu', seed=6).next(16).indices, TrainSet(small_set, 'cpu', seed=5).next(16).indices)


def test_repa_images_can_be_left_out(tmp_path, image_folder):
    prepare_fast(image_folder, tmp_path / 'data', repa_images=False)
    assert TrainSet(tmp_path / 'data', 'cpu', repa=False).next(4).images224 is None
    with pytest.raises(DatasetError, match='REPA'):
        TrainSet(tmp_path / 'data', 'cpu', repa=True)
