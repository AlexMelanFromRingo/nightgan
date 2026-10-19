import numpy as np
import pytest
import scipy.linalg
import torch

from conftest import GrayAE, left_right_color, mean_color, write_images
from nightgan.data import load_square, prepare
from nightgan.metrics import MetricError, evaluate, feature_stats, frechet_distance, kid, reconstruction_quality


def random_spd(d, rng):
    a = rng.normal(size=(d, 2 * d))
    return a @ a.T / (2 * d)


def scipy_fid(mu1, s1, mu2, s2):
    covmean = scipy.linalg.sqrtm(s1 @ s2).real
    return float(np.square(mu1 - mu2).sum() + np.trace(s1) + np.trace(s2) - 2 * np.trace(covmean))


def test_frechet_distance_matches_the_scipy_formula():
    rng = np.random.default_rng(0)
    mu1, mu2 = rng.normal(size=16), rng.normal(size=16)
    s1, s2 = random_spd(16, rng), random_spd(16, rng)
    assert frechet_distance(mu1, s1, mu2, s2) == pytest.approx(scipy_fid(mu1, s1, mu2, s2), rel=1e-6)
    assert frechet_distance(mu1, s1, mu1, s1) == pytest.approx(0.0, abs=1e-8)


def test_feature_stats():
    f = torch.randn(1000, 4, dtype=torch.float64) * torch.tensor([1.0, 2.0, 3.0, 4.0]) + 5
    mu, sigma = feature_stats(f)
    np.testing.assert_allclose(mu, f.mean(0).numpy())
    np.testing.assert_allclose(sigma, np.cov(f.numpy(), rowvar=False))


def test_kid_is_near_zero_for_one_distribution_and_positive_for_a_shift():
    g = torch.Generator().manual_seed(0)
    a, b = torch.randn(2000, 8, generator=g), torch.randn(2000, 8, generator=g)
    same, _ = kid(a, b, subsets=20, subset_size=500)
    shifted, _ = kid(a, b + 1.0, subsets=20, subset_size=500)
    assert abs(same) < 0.01
    assert shifted > 0.1


def test_evaluate_the_dataset_against_itself(tmp_path):
    paths = write_images(tmp_path / 'src', 64, seed=4)
    prepare(tmp_path / 'src', tmp_path / 'data', resolution=32, autoencoder='pixel', device='cpu', batch=16,
            workers=2, extractors={'inception': mean_color}, progress=False)
    imgs = torch.from_numpy(np.stack([load_square(p, 32) for p in paths])).permute(0, 3, 1, 2)
    result = evaluate(imgs.split(16), tmp_path / 'data', metrics=('fid', 'kid'), extractors={'inception': mean_color})
    assert result['num'] == 64
    assert result['fid'] == pytest.approx(0.0, abs=1e-6)
    assert np.isfinite(result['kid'])
    noise = torch.randint(0, 256, imgs.shape, dtype=torch.uint8)
    worse = evaluate(noise.split(16), tmp_path / 'data', metrics=('fid',), extractors={'inception': mean_color})
    assert worse['fid'] > 0.01


def test_missing_reference_statistics_are_reported(tmp_path):
    write_images(tmp_path / 'src', 8)
    prepare(tmp_path / 'src', tmp_path / 'data', resolution=32, autoencoder='pixel', device='cpu', batch=8,
            workers=1, extractors={'inception': mean_color}, progress=False)
    with pytest.raises(MetricError, match='dinov2'):
        evaluate([torch.zeros(2, 3, 32, 32, dtype=torch.uint8)], tmp_path / 'data', metrics=('fd_dinov2',),
                 extractors={'dinov2': mean_color})
    with pytest.raises(ValueError, match='unknown metric'):
        evaluate([], tmp_path / 'data', metrics=('psnr',), extractors={})


def test_a_lossless_autoencoder_reconstructs_the_dataset_exactly(tmp_path):
    write_images(tmp_path / 'src', 24, seed=5)
    prepare(tmp_path / 'src', tmp_path / 'data', resolution=32, autoencoder='pixel', device='cpu', batch=8,
            workers=2, extractors={'inception': left_right_color}, progress=False)
    result = reconstruction_quality(tmp_path / 'data', num=24, metrics=('fid',),
                                    extractors={'inception': left_right_color}, device='cpu')
    assert result['num'] == 24
    assert result['fid'] == pytest.approx(0.0, abs=1e-6)   # the stored originals, not their mirror images


def test_a_lossy_autoencoder_falls_short_of_the_dataset(tmp_path):
    write_images(tmp_path / 'src', 24, seed=5)
    prepare(tmp_path / 'src', tmp_path / 'data', resolution=32, device='cpu', batch=8, workers=2, ae=GrayAE(),
            extractors={'inception': mean_color}, progress=False)
    result = reconstruction_quality(tmp_path / 'data', num=24, metrics=('fid', 'kid'), ae=GrayAE(),
                                    extractors={'inception': mean_color}, device='cpu')
    assert result['fid'] > 0.01 and np.isfinite(result['kid'])   # KID: unbiased, noisy on 24 images


def test_a_dataset_smaller_than_the_sample_is_drawn_with_repetition(tmp_path):
    write_images(tmp_path / 'src', 8)
    prepare(tmp_path / 'src', tmp_path / 'data', resolution=32, autoencoder='pixel', device='cpu', batch=8,
            workers=1, extractors={'inception': mean_color}, progress=False)
    result = reconstruction_quality(tmp_path / 'data', num=50, metrics=('fid',), extractors={'inception': mean_color},
                                    device='cpu')
    assert result['num'] == 50


@pytest.mark.slow
@pytest.mark.skipif(not torch.cuda.is_available(), reason='needs a GPU')
def test_real_feature_extractors():
    from nightgan.metrics import default_extractors
    ex = default_extractors('cuda')
    imgs = torch.randint(0, 256, (2, 3, 256, 256), dtype=torch.uint8)
    assert ex['inception'](imgs).shape == (2, 2048)
    assert ex['dinov2'](imgs).shape == (2, 1024)
