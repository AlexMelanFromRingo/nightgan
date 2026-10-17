import pytest
import torch

from nightgan.augment import apply_aug, sample_aug, shift_images


def delta(side=16, at=(5, 5), channels=4):
    x = torch.zeros(1, channels, side, side)
    x[:, :, at[0], at[1]] = 1.0
    return x


def test_probability_zero_is_the_identity():
    x = torch.randn(8, 4, 16, 16)
    params = sample_aug(8, 16, 'color,translation,cutout', p=0.0, device='cpu')
    torch.testing.assert_close(apply_aug(x, params), x)


def test_translation_moves_content_by_the_shift():
    params = sample_aug(1, 16, 'translation', p=1.0, device='cpu')
    params.shift[:] = torch.tensor([2, -3])
    out = apply_aug(delta(), params)
    assert out[0, 0].nonzero().tolist() == [[7, 2]]


def test_translation_fills_with_zeros():
    params = sample_aug(1, 16, 'translation', p=1.0, device='cpu')
    params.shift[:] = torch.tensor([0, 2])
    out = apply_aug(torch.ones(1, 4, 16, 16), params)
    assert out[0, 0, :, :2].abs().sum() == 0 and out[0, 0, :, 2:].min() == 1


def test_shifts_are_bounded_by_an_eighth_of_the_side():
    params = sample_aug(1000, 32, 'translation', p=1.0, device='cpu')
    assert params.shift.abs().max() == 4


def test_cutout_zeroes_a_half_size_square():
    params = sample_aug(1, 16, 'cutout', p=1.0, device='cpu')
    params.cutout[:] = torch.tensor([8, 8])
    out = apply_aug(torch.ones(1, 4, 16, 16), params)
    zero = (out[0, 0] == 0)
    assert zero.sum() == 64
    assert zero[4:12, 4:12].all()


def test_color_ops_change_values_and_keep_shape():
    torch.manual_seed(0)
    x = torch.randn(16, 4, 8, 8)
    out = apply_aug(x, sample_aug(16, 8, 'color', p=1.0, device='cpu'))
    assert out.shape == x.shape and not torch.allclose(out, x)


def test_same_params_transform_stages_identically():
    torch.manual_seed(0)
    a = torch.randn(4, 4, 16, 16)
    params = sample_aug(4, 16, 'color,translation,cutout', p=1.0, device='cpu')
    torch.testing.assert_close(apply_aug(a, params), apply_aug(a.clone(), params))
    b = 2 * a
    assert not torch.allclose(apply_aug(a, params), apply_aug(b, params))


def test_shift_images_scales_the_latent_shift_to_pixels():
    params = sample_aug(1, 32, 'translation', p=1.0, device='cpu')
    params.shift[:] = torch.tensor([1, -2])
    img = torch.zeros(1, 3, 224, 224, dtype=torch.uint8)
    img[:, :, 100, 100] = 255
    out = shift_images(img, params, latent_side=32)
    assert out.dtype == torch.uint8
    assert out[0, 0].nonzero().tolist() == [[107, 86]]


def test_shift_images_without_translation_is_the_identity():
    img = torch.randint(0, 255, (2, 3, 224, 224), dtype=torch.uint8)
    params = sample_aug(2, 32, 'color,cutout', p=1.0, device='cpu')
    assert torch.equal(shift_images(img, params, latent_side=32), img)


def test_unknown_policy_is_rejected():
    with pytest.raises(ValueError, match='rotate'):
        sample_aug(1, 16, 'translation,rotate', p=1.0, device='cpu')


def test_color_is_applied_to_half_of_the_samples_as_in_gat():
    params = sample_aug(4000, 16, 'color', p=1.0, device='cpu')
    changed = (params.brightness.flatten() != 0).float().mean().item()
    assert 0.45 < changed < 0.55


def test_augmentation_does_not_synchronize_with_the_gpu(monkeypatch):
    # a data-dependent Python branch (tensor.any()) stalls the CPU until the GPU catches up
    calls = []
    original = torch.Tensor.__bool__
    monkeypatch.setattr(torch.Tensor, '__bool__', lambda self: calls.append(1) or original(self))
    x = torch.randn(4, 4, 16, 16)
    params = sample_aug(4, 16, 'color,translation,cutout', p=1.0, device='cpu')
    apply_aug(x, params)
    shift_images(torch.zeros(4, 3, 224, 224, dtype=torch.uint8), params, latent_side=16)
    assert calls == []
