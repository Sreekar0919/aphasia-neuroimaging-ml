"""Held-out FC reconstruction/generation accuracy on the OSFData cohort (5-fold CV, models never see the test subject).

Per subject, compared with that subject's real FC (Fisher-z upper triangle):
  Pearson r      edge-wise correlation of predicted vs real FC
  RMSE           root-mean-square edge error
  Euclid norm    Frobenius norm of (pred - real) over the full symmetric matrix (the poster's metric)
  skill          1 - SSE/SSE_meanFC : share of error removed relative to predicting the training-mean FC
'reconstruct' = encode the subject's own FC and decode; 'generate' = decode random latents (mean of 20 draws)
given only the subject's demographics, i.e. no access to their FC.

    python recon_accuracy.py --npz data_aal.npz
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import KFold

import staged
from abide import load_abide_fc
from arc import load_arc_fc
from lowcap import LowCapModel

N_DRAWS = 20


def summarise(pred, x, mean_pred):
    r = np.array([np.corrcoef(p, t)[0, 1] for p, t in zip(pred, x)])
    sse, sse_mean = ((pred - x) ** 2).sum(1), ((mean_pred - x) ** 2).sum(1)
    return {'r': r.mean(), 'rmse': np.sqrt(((pred - x) ** 2).mean()),
            'euclid': (np.sqrt(2 * sse)).mean(), 'skill': 1 - sse.sum() / sse_mean.sum()}


def run(d, bases, seed=0):
    x, age, sex, mpo = d['X'], d['age'], d['sex'], d['mpo']
    out = {k: np.zeros_like(x) for k in ['mean_fc', *[f'{b}/{m}' for b in [*bases, 'pca16'] for m in ('reconstruct', 'generate')]]}
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        out['mean_fc'][te] = x[tr].mean(0)
        demo_tr, demo_te = [age[tr], sex[tr], mpo[tr]], [age[te], sex[te], mpo[te]]
        models = {b: staged.stage_fit(base, x[tr], demo_tr, staged.EPOCHS['osf'], False, ()) for b, base in bases.items()}
        models['pca16'] = LowCapModel(16).fit(x[tr], demo_tr)
        for b, m in models.items():
            torch.manual_seed(seed)
            out[f'{b}/reconstruct'][te] = m.transform(x[te], demo_te)
            out[f'{b}/generate'][te] = np.mean([m.transform(len(te), demo_te) for _ in range(N_DRAWS)], 0)
    return {k: summarise(v, x, out['mean_fc']) for k, v in out.items()}


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--npz', default='data_aal.npz')
    args = ap.parse_args()
    d = np.load(args.npz)
    abide, arc = load_abide_fc(d['roi_keep']), load_arc_fc(d['roi_keep'])
    s1 = staged.stage1(abide)
    bases = {'abide_ft': s1, 'arc_ft': staged.stage2(s1, arc, False)}
    res = run(d, bases)
    print(f'{"model":24s}{"r":>8s}{"RMSE":>8s}{"Euclid":>9s}{"skill":>8s}')
    for k, v in res.items():
        print(f'{k:24s}{v["r"]:8.3f}{v["rmse"]:8.4f}{v["euclid"]:9.2f}{v["skill"]:+8.3f}')
