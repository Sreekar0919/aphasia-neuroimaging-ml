"""Curriculum fine-tuning for the FC VAE, evaluated by held-out CV on the OSFData aphasia cohort.

    stage 1  ABIDE-I adults (age >= 18)            conditioning: age, sex
    stage 2  Aphasia Recovery Cohort (ARC)         conditioning: age, sex, months post onset [, WAB-AQ]
    stage 3  OSFData training fold (the target)    same conditioning as stage 2

Stages 1-2 never see OSFData, so they are trained once; only stage 3 is refit per fold. Variants:
abide / abide_wab (skip stage 2), arc / arc_wab (full curriculum), '_wab' = WAB-AQ is a conditioning input.
WAB-AQ is a *given* label for the FC metrics; WAB prediction by analysis-by-synthesis (decode every
candidate WAB, keep the best match to the held-out FC) never sees the held-out WAB.

    python staged.py --npz data_aal.npz
"""
import argparse
import copy

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold

from abide import load_abide_fc
from arc import load_arc_fc
from finetune import MPO_COL, WAB_COL, FCModel

AGE_STATS, MPO_STATS, WAB_STATS = (45.0, 20.0), (36.0, 36.0), (65.0, 30.0)  # fixed so stages share a scale
LR = 1e-4
EPOCHS = {'abide': 300, 'arc': 150, 'osf': 100}
VARIANTS = {'abide': (False, False), 'abide_wab': (False, True), 'arc': (True, False), 'arc_wab': (True, True)}
WAB_GRID = np.arange(0, 101, 5.0)
N_DRAWS = 20


def new_model(dim, latent_dim=16):
    m = FCModel(dim, latent_dim)
    m.age_stats, m.mpo_stats, m.wab_stats = AGE_STATS, MPO_STATS, WAB_STATS
    return m


def stage1(abide, latent_dim=16, epochs=EPOCHS['abide'], seed=0):
    x, age, male = abide
    keep = age >= 18
    torch.manual_seed(seed)
    m = new_model(x.shape[1], latent_dim)
    sex = np.where(male[keep] == 1, 'M', 'F')
    return m.fit(x[keep], [age[keep], sex, np.zeros(keep.sum())], nepochs=epochs, bsize=128, decor_cols=(0, 1))


def stage_fit(model, x, demo, epochs, use_wab, new_cols):
    """Continue training with enc1 frozen; `new_cols` start at zero weight in the decoder."""
    model = copy.deepcopy(model)
    model.zero_columns(*new_cols)
    cols = (0, 1, 2, 3) if use_wab else (0, 1, 2)
    demo = demo if use_wab else demo[:3]
    return model.fit(x, demo, nepochs=epochs, lr=LR, bsize=min(64, len(x)), freeze_enc1=True, decor_cols=cols)


def stage2(base, arc, use_wab, epochs=EPOCHS['arc']):
    x, age, male, mpo, wab = arc
    sex = np.where(male == 1, 'M', 'F')
    return stage_fit(base, x, [age, sex, mpo, wab], epochs, use_wab, (MPO_COL, WAB_COL))


def rmse(a, b):
    return float(np.sqrt(np.mean((a - b) ** 2)))


def predict_wab(model, x, demo3, use_wab):
    """Pick, per subject, the WAB-AQ whose decoded FC best matches the observed FC."""
    z = torch.tensor(model.get_latents(x))
    errs = []
    for w in WAB_GRID:
        recon = model.transform(x, [*demo3, np.full(len(x), w)])
        errs.append(((recon - x) ** 2).mean(1))
    return WAB_GRID[np.argmin(np.stack(errs), 0)]


def fold_metrics(model, x_te, demo_te, use_wab, rng):
    age, sex, mpo, wab = demo_te
    d = [age, sex, mpo, wab] if use_wab else [age, sex, mpo]
    out = {'recon': rmse(model.transform(x_te, d), x_te)}
    out['gen_mean'] = rmse(np.mean([model.transform(len(x_te), d) for _ in range(N_DRAWS)], 0), x_te)
    if use_wab:
        shuffled = [age, sex, mpo, rng.permutation(wab)]
        out['recon_wab_shuffled'] = rmse(model.transform(x_te, shuffled), x_te)
        out['wab_pred'] = predict_wab(model, x_te, [age, sex, mpo], use_wab)
    return out


def evaluate(x, age, sex, mpo, wab, bases, n_splits=5, seed=0):
    """bases: {variant: model after stages 1-2}. Returns pooled FC RMSE and out-of-fold WAB results."""
    rng = np.random.default_rng(seed)
    res = {v: {'recon': [], 'gen_mean': [], 'recon_wab_shuffled': [], 'wab_pred': np.zeros(len(x))} for v in bases}
    base_err, pooled = {'mean_fc': [], 'group_mean_fc': []}, {}
    for tr, te in KFold(n_splits, shuffle=True, random_state=seed).split(x):
        mean = x[tr].mean(0)
        base_err['mean_fc'].append(((x[te] - mean) ** 2).ravel())
        gm = np.stack([x[tr][sex[tr] == s].mean(0) if (sex[tr] == s).any() else mean for s in sex[te]])
        base_err['group_mean_fc'].append(((x[te] - gm) ** 2).ravel())
        for v, base in bases.items():
            use_wab = VARIANTS[v][1]
            torch.manual_seed(seed)
            m = stage_fit(base, x[tr], [age[tr], sex[tr], mpo[tr], wab[tr]], EPOCHS['osf'], use_wab,
                          (WAB_COL,) if use_wab else ())
            for k, val in fold_metrics(m, x[te], [age[te], sex[te], mpo[te], wab[te]], use_wab, rng).items():
                if k == 'wab_pred':
                    res[v][k][te] = val
                else:
                    res[v][k].append(val)
    fc = {k: float(np.sqrt(np.concatenate(e).mean())) for k, e in base_err.items()}
    for v, r in res.items():
        fc[f'{v}/recon'] = float(np.mean(r['recon']))  # per-fold RMSE averaged
        fc[f'{v}/gen_mean'] = float(np.mean(r['gen_mean']))
        if VARIANTS[v][1]:
            fc[f'{v}/recon_wab_shuffled'] = float(np.mean(r['recon_wab_shuffled']))
    train_mean = np.mean(wab)
    r2 = lambda p: 1 - ((wab - p) ** 2).sum() / ((wab - train_mean) ** 2).sum()
    wabres = {v: {'rmse': rmse(wab, r['wab_pred']), 'r2': float(r2(r['wab_pred']))}
              for v, r in res.items() if VARIANTS[v][1]}
    wabres['train_mean'] = {'rmse': rmse(wab, np.full(len(wab), train_mean)), 'r2': 0.0}
    return fc, wabres


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--npz', default='data_aal.npz')
    ap.add_argument('--latent_dim', type=int, default=16)
    args = ap.parse_args()
    d = np.load(args.npz)
    keep = d['roi_keep']
    abide, arc = load_abide_fc(keep), load_arc_fc(keep)
    print(f'ABIDE adults: {(abide[1] >= 18).sum()}  ARC: {len(arc[0])} (WAB-AQ {arc[4].min():.0f}-{arc[4].max():.0f})')
    s1 = stage1(abide, args.latent_dim)
    bases = {'abide': s1, 'abide_wab': s1, 'arc': stage2(s1, arc, False), 'arc_wab': stage2(s1, arc, True)}
    fc, wabres = evaluate(d['X'], d['age'], d['sex'], d['mpo'], d['wab'], bases)
    print('FC RMSE (held-out OSF)')
    for k, v in fc.items():
        print(f'  {k:28s}{v:.4f}')
    print('WAB-AQ by analysis-by-synthesis (out-of-fold)')
    for k, v in wabres.items():
        print(f'  {k:12s}RMSE {v["rmse"]:.2f}  R2 {v["r2"]:+.3f}')
