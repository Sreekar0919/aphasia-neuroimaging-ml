"""Held-out OSFData evaluation of weight-programmed FC VAEs (see programmed.py), unconditioned vs conditioned.

Every number is from the programmed network itself, on subjects outside the training fold; the
basis pool (ABIDE + ARC) and the demographic map (ARC + training fold) never contain test subjects.
Conditioning = [age, sex, months post onset, WAB-AQ].

    python programmed_eval.py --npz data_aal.npz
"""
import argparse

import numpy as np
import torch
from sklearn.model_selection import KFold

from abide import load_abide_fc
from arc import load_arc_fc
from programmed import OFFSET, demographic_map, pooled_basis, program
from staged import new_model

LATENT_SIZES = [16, 100, 400, 800, 996]
CONTROL_K = 800  # size used for the conditioning checks
WAB_GRID = np.arange(0, 101, 1.0)
N_DRAWS = 20


def accuracy(rec, x, mean_pred):
    r = np.array([np.corrcoef(a, b)[0, 1] for a, b in zip(rec, x)])
    r2 = np.array([1 - ((a - b) ** 2).sum() / ((b - b.mean()) ** 2).sum() for a, b in zip(rec, x)])
    skill = 1 - ((rec - x) ** 2).sum() / ((mean_pred - x) ** 2).sum()
    return {'r': r.mean(), 'r_min': r.min(), 'r2': r2.mean(), 'rmse': np.sqrt(((rec - x) ** 2).mean()), 'skill': skill}


def mean_abs_corr(z, v):
    return float(np.mean([abs(np.corrcoef(zj, v)[0, 1]) for zj in z.T]))


def fold_models(x_tr, demo_tr, broad, arc):
    """{k: (unconditioned model, conditioned model)} for one training fold."""
    ref = new_model(x_tr.shape[1], 1)  # only for its demographic standardisation
    B = demographic_map([(arc[0], ref._demo(arc[1:]).numpy()), (x_tr, ref._demo(demo_tr).numpy())])
    V0, s0 = pooled_basis([*broad, x_tr], max(LATENT_SIZES))
    V1, s1 = pooled_basis([*broad, x_tr], max(LATENT_SIZES), B)
    return {k: (program(new_model(x_tr.shape[1], k), x_tr, V0[:k], s0[:k]),
                program(new_model(x_tr.shape[1], k), x_tr, V1[:k], s1[:k], demo_tr, B)) for k in LATENT_SIZES}, B


def run(d, broad, arc, seed=0):
    x, sex = d['X'], d['sex']
    demo = [d['age'], sex, d['mpo'], d['wab']]
    rng = np.random.default_rng(seed)
    keys = ['mean_fc'] + [f'{c}_k{k}' for c in ('uncond', 'cond') for k in LATENT_SIZES]
    keys += ['cond_wab_shuffled', 'cond_demo_shuffled', 'generate_cond', 'generate_cond_wab_shuffled']
    recs = {k: np.zeros_like(x) for k in keys}
    wab_pred, min_pre, decor, wab_effect = np.zeros(len(x)), np.inf, {'uncond': [], 'cond': []}, []
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        dtr, dte = [a[tr] for a in demo], [a[te] for a in demo]
        models, B = fold_models(x[tr], dtr, broad, arc)
        recs['mean_fc'][te] = x[tr].mean(0)
        for k, (mu, mc) in models.items():
            recs[f'uncond_k{k}'][te] = mu.transform(x[te], dte[:3])
            recs[f'cond_k{k}'][te] = mc.transform(x[te], dte)
            min_pre = min(min_pre, float(mc.get_latents(x[te]).min() + OFFSET))
        mu, mc = models[CONTROL_K]
        wab_shuf = [*dte[:3], rng.permutation(dte[3])]
        perm = rng.permutation(len(te))
        recs['cond_wab_shuffled'][te] = mc.transform(x[te], wab_shuf)
        recs['cond_demo_shuffled'][te] = mc.transform(x[te], [a[perm] for a in dte])
        recs['generate_cond'][te] = np.mean([mc.transform(len(te), dte) for _ in range(N_DRAWS)], 0)
        recs['generate_cond_wab_shuffled'][te] = np.mean([mc.transform(len(te), wab_shuf) for _ in range(N_DRAWS)], 0)
        errs = [((mc.transform(x[te], [*dte[:3], np.full(len(te), w)]) - x[te]) ** 2).mean(1) for w in WAB_GRID]
        wab_pred[te] = WAB_GRID[np.argmin(errs, 0)]  # WAB whose decoded FC best matches the held-out FC
        for name, m in (('uncond', mu), ('cond', mc)):
            z = m.get_latents(x[tr])[:, :16]
            decor[name].append((mean_abs_corr(z, dtr[0]), mean_abs_corr(z, dtr[3])))
        wab_effect.append(np.sqrt(np.mean(((90 - 20) / mc.wab_stats[1] * B[:, 3]) ** 2)))
    acc = {k: accuracy(r, x, recs['mean_fc']) for k, r in recs.items()}
    wab = d['wab']
    wab_r2 = 1 - ((wab - wab_pred) ** 2).sum() / ((wab - wab.mean()) ** 2).sum()
    return acc, min_pre, {k: np.mean(v, 0) for k, v in decor.items()}, float(np.mean(wab_effect)), wab_pred, wab_r2


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', default='data_aal.npz')
    args = ap.parse_args()
    d = np.load(args.npz)
    torch.manual_seed(0)
    xa = load_abide_fc(d['roi_keep'])[0]
    xr, age_r, male_r, mpo_r, wab_r = load_arc_fc(d['roi_keep'])
    arc = (xr, age_r, np.where(male_r == 1, 'M', 'F'), mpo_r, wab_r)
    print(f'basis pool: ABIDE {len(xa)} + ARC {len(xr)} + fold; demographic map: ARC {len(xr)} + fold; '
          f'{d["X"].shape[1]} FC edges')
    acc, min_pre, decor, wab_effect, wab_pred, wab_r2 = run(d, [xa, xr], arc)
    print(f'\n{"held-out reconstruction":28s}{"r":>7s}{"min r":>7s}{"R2":>7s}{"RMSE":>8s}{"skill":>8s}')
    for k, a in acc.items():
        print(f'{k:28s}{a["r"]:7.3f}{a["r_min"]:7.3f}{a["r2"]:7.3f}{a["rmse"]:8.4f}{a["skill"]:+8.3f}')
    print(f'\nsmallest shifted hidden pre-activation: {min_pre:.1f} (> 0: every ReLU passed its value exactly)')
    print('mean |corr| of first 16 latents with (age, WAB-AQ), training folds:')
    for k, (a, w) in decor.items():
        print(f'  {k:8s} age {a:.3f}  WAB {w:.3f}')
    print(f'FC change (RMSE) from WAB-AQ 20 -> 90 at fixed latent: {wab_effect:.4f} '
          f'(between-subject spread, mean-FC RMSE: {acc["mean_fc"]["rmse"]:.4f})')
    wab = d['wab']
    print(f'WAB-AQ read back from held-out FC (analysis-by-synthesis, k={CONTROL_K}): '
          f'RMSE {np.sqrt(np.mean((wab - wab_pred) ** 2)):.1f} (predict-mean {wab.std():.1f}), R2 {wab_r2:+.3f}, '
          f'r {np.corrcoef(wab, wab_pred)[0, 1]:+.2f}')
