import numpy as np
import pytest

from vae_training import train_fc_vae
from config import MODEL_CONFIG

DEMO_TYPES = ['continuous', 'categorical', 'continuous', 'continuous']


def make_data(n, d=60, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.standard_normal((n, d)).astype(np.float32)
    demo = [rng.normal(60, 10, n), rng.choice(['M', 'F'], n),
            rng.normal(24, 12, n), rng.normal(50, 15, n)]
    return x, demo


def cfg(**kw):
    return {**MODEL_CONFIG, 'nepochs': 3, 'use_cuda': False, **kw}


def test_default_config_is_trainable_on_small_cohort():
    x, demo = make_data(40)
    assert MODEL_CONFIG['latent_dim'] * 2 <= 40  # covariance loss needs n >> latent_dim
    vae, *_ = train_fc_vae(x, demo, DEMO_TYPES, cfg())
    assert vae.get_latents(x).shape == (40, MODEL_CONFIG['latent_dim'])


def test_batch_is_clamped_to_cohort_size():
    x, demo = make_data(40)
    vae, *_ = train_fc_vae(x, demo, DEMO_TYPES, cfg(bsize=1000))
    assert vae.bsize == 40


def test_rejects_latent_dim_not_below_batch_size():
    x, demo = make_data(20)
    with pytest.raises(ValueError, match='latent_dim'):
        train_fc_vae(x, demo, DEMO_TYPES, cfg(latent_dim=32, bsize=16))
