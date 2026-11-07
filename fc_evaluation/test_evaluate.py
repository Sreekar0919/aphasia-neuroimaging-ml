import numpy as np
import pytest

from evaluate import evaluate_cv

FAST = {'nepochs': 5, 'use_cuda': False, 'latent_dim': 4, 'bsize': 64}


def make_cohort(n=40, d=80, seed=0):
    rng = np.random.default_rng(seed)
    age, mpo = rng.normal(60, 10, n), rng.normal(24, 12, n)
    sex = rng.choice(['M', 'F'], n)
    wab = rng.normal(60, 20, n)
    signal = np.outer((wab - wab.mean()) / wab.std(), rng.standard_normal(d))
    x = (signal + 0.5 * rng.standard_normal((n, d))).astype(np.float32)
    return x, age, sex, mpo, wab


@pytest.fixture(scope='module')
def result():
    return evaluate_cv(*make_cohort(), n_splits=4, model_config=FAST)


def test_reports_every_baseline_and_method(result):
    assert {'mean_fc', 'group_mean_fc', 'vae_recon', 'vae_gen_mean'} <= set(result['fc_rmse'])
    assert {'train_mean', 'demo_only', 'latent+demo', 'fc_real', 'fc_gen_mean'} <= set(result['outcome'])


def test_real_fc_predicts_outcome_when_signal_is_planted(result):
    out = result['outcome']
    assert out['fc_real']['r2'] > 0.5
    assert out['fc_real']['rmse'] < out['train_mean']['rmse']


def test_wab_never_reaches_the_vae():
    x, age, sex, mpo, wab = make_cohort()
    shuffled = np.random.default_rng(1).permutation(wab)
    a = evaluate_cv(x, age, sex, mpo, wab, n_splits=4, model_config=FAST)
    b = evaluate_cv(x, age, sex, mpo, shuffled, n_splits=4, model_config=FAST)
    assert a['fc_rmse'] == b['fc_rmse']  # VAE output independent of WAB
    assert a['outcome']['fc_real'] != b['outcome']['fc_real']  # outcome side does use it


def test_low_capacity_model_beats_mean_fc_when_signal_is_planted():
    res = evaluate_cv(*make_cohort(), n_splits=4, model_config={'latent_dim': 3}, model='pca')
    assert res['fc_rmse']['vae_recon'] < res['fc_rmse']['mean_fc']
    assert res['outcome']['latent+demo']['r2'] > 0.5
