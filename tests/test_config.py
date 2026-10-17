import json

import pytest

from nightgan.config import (PRESETS, ModelConfig, TrainConfig, from_dict, model_config_for, to_dict)


def test_presets_define_model_width_depth_heads():
    s = model_config_for('S', latent_res=32, latent_channels=4, num_classes=0)
    b = model_config_for('B', latent_res=32, latent_channels=4, num_classes=0)
    assert (s.dim, s.depth, s.heads) == (384, 12, 6)
    assert (b.dim, b.depth, b.heads) == (768, 12, 12)
    assert set(PRESETS) >= {'S', 'B'}


@pytest.mark.parametrize('latent_res, patch', [(16, 1), (32, 2), (64, 4)])
def test_default_patch_gives_16x16_tokens(latent_res, patch):
    cfg = model_config_for('S', latent_res=latent_res, latent_channels=4, num_classes=0)
    assert cfg.patch == patch
    assert cfg.tokens_side == 16


def test_patch_must_leave_room_for_four_scales():
    with pytest.raises(ValueError, match='patch'):
        model_config_for('S', latent_res=32, latent_channels=4, num_classes=0, patch=8)  # 4x4 tokens


def test_discriminator_size_falls_back_to_generator_size():
    cfg = ModelConfig(dim=256, depth=6, heads=4)
    assert (cfg.D_dim, cfg.D_depth, cfg.D_heads) == (256, 6, 4)
    cfg = ModelConfig(dim=256, depth=6, heads=4, d_dim=128, d_depth=3, d_heads=2)
    assert (cfg.D_dim, cfg.D_depth, cfg.D_heads) == (128, 3, 2)


def test_json_round_trip_preserves_configs():
    m = model_config_for('B', latent_res=64, latent_channels=4, num_classes=3)
    t = TrainConfig(batch=128, time_limit=3600.0, betas=(0.0, 0.9))
    m2 = from_dict(ModelConfig, json.loads(json.dumps(to_dict(m))))
    t2 = from_dict(TrainConfig, json.loads(json.dumps(to_dict(t))))
    assert m2 == m
    assert t2 == t


def test_unknown_keys_are_rejected_by_name():
    with pytest.raises(ValueError, match='bogus'):
        from_dict(ModelConfig, {'dim': 64, 'bogus': 1})


def test_width_aware_learning_rate():
    t = TrainConfig()
    assert t.resolved_lr(ModelConfig(dim=384)) == pytest.approx((4e-4, 4e-4))
    assert t.resolved_lr(ModelConfig(dim=768, d_dim=384)) == pytest.approx((2e-4, 4e-4))
    assert TrainConfig(lr_g=1e-4, lr_d=3e-4).resolved_lr(ModelConfig()) == (1e-4, 3e-4)
