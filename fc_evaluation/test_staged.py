import numpy as np

import staged
from test_finetune import D, LOADING


def cohort(n, seed, wab_effect=True):
    rng = np.random.default_rng(seed)
    age, sex, mpo = rng.normal(55, 10, n), rng.choice(['M', 'F'], n), rng.uniform(3, 60, n)
    wab = rng.uniform(10, 95, n)
    x = rng.standard_normal((n, 4)) @ LOADING + 0.3 * rng.standard_normal((n, D))
    x = x + (wab[:, None] - 60) / 30 * LOADING[0] * 2 * wab_effect  # WAB shifts FC along one axis
    return x.astype(np.float32), age, sex, mpo, wab


def test_curriculum_runs_and_wab_conditioning_is_used(monkeypatch):
    monkeypatch.setattr(staged, 'EPOCHS', {'abide': 20, 'arc': 20, 'osf': 40})
    rng = np.random.default_rng(0)
    abide = (rng.standard_normal((200, D)).astype(np.float32), rng.uniform(5, 50, 200), rng.integers(0, 2, 200).astype(float))
    xa, aa, sa, ma, wa = cohort(80, 1)
    arc = (xa, aa, (sa == 'M').astype(float), ma, wa)
    s1 = staged.stage1(abide, latent_dim=8, epochs=20)
    bases = {'abide_wab': s1, 'arc_wab': staged.stage2(s1, arc, True, epochs=20)}
    x, age, sex, mpo, wab = cohort(30, 2)
    fc, wabres = staged.evaluate(x, age, sex, mpo, wab, bases)
    assert fc['arc_wab/recon'] < fc['mean_fc'] * 1.5 and np.isfinite(list(fc.values())).all()
    assert fc['arc_wab/recon'] <= fc['arc_wab/recon_wab_shuffled']  # true WAB helps vs a shuffled one
    assert set(wabres) == {'abide_wab', 'arc_wab', 'train_mean'}
