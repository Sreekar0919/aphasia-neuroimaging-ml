import numpy as np
import torch

from finetune import MPO_COL, FCModel, fit_for_cv
from pretrain import pretrain

D = 60
RNG = np.random.default_rng(0)
LOADING = RNG.standard_normal((4, D))


def cohort(n, seed):
    rng = np.random.default_rng(seed)
    age, sex = rng.normal(40, 12, n), rng.choice(['M', 'F'], n)
    x = rng.standard_normal((n, 4)) @ LOADING + 0.3 * rng.standard_normal((n, D))
    return x.astype(np.float32), [age, sex, rng.normal(24, 12, n)]


def heldout_rmse(model, x, demo):
    return float(np.sqrt(np.mean((model.transform(x, demo) - x) ** 2)))


def pretrained_checkpoint(tmp_path):
    x, (age, sex, _) = cohort(400, seed=1)
    model, val, base = pretrain(x, age, (sex == 'M').astype(float), latent_dim=8, epochs=60, bsize=64)
    assert val < base  # learns real structure, not just the mean
    path = tmp_path / 'pre.pt'
    model.save(path)
    return path


def test_pretraining_transfers_to_a_tiny_cohort(tmp_path):
    path = pretrained_checkpoint(tmp_path)
    x_tr, demo_tr = cohort(24, seed=2)
    x_te, demo_te = cohort(30, seed=3)
    torch.manual_seed(0)
    scratch = fit_for_cv('scratch', x_tr, demo_tr, epochs=60, latent_dim=8)
    zeroshot = fit_for_cv('zeroshot', x_tr, demo_tr, pretrained=path)
    finetuned = fit_for_cv('finetune', x_tr, demo_tr, pretrained=path, ft_epochs=20)
    assert heldout_rmse(zeroshot, x_te, demo_te) < heldout_rmse(scratch, x_te, demo_te)
    assert heldout_rmse(finetuned, x_te, demo_te) < heldout_rmse(scratch, x_te, demo_te)


def test_finetune_freezes_enc1_and_zeroes_the_new_demographic_column(tmp_path):
    path = pretrained_checkpoint(tmp_path)
    before = FCModel.load(path).vae.enc1.weight.detach().clone()
    x, demo = cohort(24, seed=2)
    model = fit_for_cv('finetune', x, demo, pretrained=path, ft_epochs=5)
    assert torch.equal(model.vae.enc1.weight, before)
    assert not torch.equal(model.vae.enc2.weight, FCModel.load(path).vae.enc2.weight)
    assert (model.vae.dec1.weight[:, model.latent_dim + MPO_COL] != 0).any()  # learned during fine-tuning
    zs = fit_for_cv('zeroshot', x, demo, pretrained=path)
    assert (zs.vae.dec1.weight[:, zs.latent_dim + MPO_COL] == 0).all()


def test_wab_conditioning_changes_the_output():
    x, (age, sex, mpo) = cohort(40, seed=2)
    wab = np.random.default_rng(5).uniform(10, 95, 40)
    model = FCModel(D, 8)
    model.wab_stats = (60.0, 25.0)
    model.fit(x, [age, sex, mpo, wab], nepochs=30, bsize=20)
    lo = model.transform(x, [age, sex, mpo, np.full(40, 10.0)])
    hi = model.transform(x, [age, sex, mpo, np.full(40, 95.0)])
    assert np.abs(lo - hi).max() > 0  # the WAB column reaches the decoder
