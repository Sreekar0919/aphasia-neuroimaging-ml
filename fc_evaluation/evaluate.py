"""Held-out evaluation of the FC DemoVAE.

The VAE is retrained inside every CV fold on the training subjects only. WAB-AQ is
the outcome, so it never enters the VAE (not conditioned on, not decorrelated
against, not a guidance target) or the predictors' inputs.

FC fidelity (pooled held-out RMSE): VAE reconstruction and demographic-conditioned
generation vs the train-fold mean FC and the train-fold same-sex mean FC.

Outcome prediction (out-of-fold WAB-AQ): ridge on train-fold real FC, applied to
held-out real / reconstructed / generated FC, vs demographics-only, VAE latents +
demographics, and predicting the train mean.

    python evaluate.py --npz data.npz   # arrays: X, age, sex, mpo, wab
"""
import argparse
import random

import numpy as np
import torch
from sklearn.linear_model import RidgeCV
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

from config import MODEL_CONFIG
from finetune import fit_for_cv
from lowcap import LowCapModel
from vae_training import train_fc_vae

DEMO_TYPES = ['continuous', 'categorical', 'continuous']  # age, sex, mpo
ALPHAS = np.logspace(0, 6, 13)
N_DRAWS = 20  # generated samples averaged for the "conditional mean" estimate


def _ridge_predict(train_x, train_y, *test_xs):
    model = RidgeCV(alphas=ALPHAS).fit(train_x, train_y)
    return [model.predict(x) for x in test_xs]


def _fc_predictions(vae, x_tr, sex_tr, x_te, demo_te):
    """Predicted held-out FC from each method, all (n_test, n_features)."""
    n = len(x_te)
    draws = np.stack([vae.transform(n, demo_te, DEMO_TYPES) for _ in range(N_DRAWS)])
    mean_fc = x_tr.mean(0)
    sex_means = {s: x_tr[sex_tr == s].mean(0) for s in np.unique(sex_tr)}
    return {
        'mean_fc': np.tile(mean_fc, (n, 1)),
        'group_mean_fc': np.stack([sex_means.get(s, mean_fc) for s in demo_te[1]]),
        'vae_recon': vae.transform(x_te, demo_te, DEMO_TYPES),
        'vae_gen_draw': draws[0],
        'vae_gen_mean': draws.mean(0),
    }


def _outcome_predictions(vae, x_tr, y_tr, d_tr, x_te, d_te, fc_pred):
    z_tr, z_te = vae.get_latents(x_tr), vae.get_latents(x_te)
    out = {'train_mean': np.full(len(x_te), y_tr.mean())}
    out['demo_only'], = _ridge_predict(d_tr, y_tr, d_te)
    out['latent+demo'], = _ridge_predict(np.hstack([z_tr, d_tr]), y_tr, np.hstack([z_te, d_te]))
    out['fc_real'], out['fc_recon'], out['fc_gen_draw'], out['fc_gen_mean'] = _ridge_predict(
        x_tr, y_tr, x_te, fc_pred['vae_recon'], fc_pred['vae_gen_draw'], fc_pred['vae_gen_mean'])
    return out


def evaluate_cv(x, age, sex, mpo, wab, n_splits=5, seed=0, model_config=None, model='demovae',
pretrained=None, ft_epochs=100, ft_lr=1e-4):
    """Return {'fc_rmse': {method: rmse}, 'outcome': {method: {'rmse','r2'}}}."""
    cfg = {**MODEL_CONFIG, **(model_config or {})}
    x, age, sex, mpo, wab = (np.asarray(a) for a in (x, age, sex, mpo, wab))
    sex_num = (sex == np.unique(sex)[0]).astype(float)
    d_num = np.column_stack([age.astype(float), sex_num, mpo.astype(float)])
    d_num = (d_num - d_num.mean(0)) / d_num.std(0)

    sq_err, count = {}, {}
    oof = {}
    for fold, (tr, te) in enumerate(KFold(n_splits, shuffle=True, random_state=seed).split(x)):
        random.seed(seed + fold)
        torch.manual_seed(seed + fold)
        demo_tr = [age[tr], sex[tr], mpo[tr]]
        demo_te = [age[te], sex[te], mpo[te]]
        if model == 'pca':  # latent_dim = number of PCA components
            vae = LowCapModel(cfg['latent_dim']).fit(x[tr], demo_tr)
        elif model.startswith('ft_'):  # ft_scratch | ft_zeroshot | ft_finetune
            vae = fit_for_cv(model[3:], x[tr], demo_tr, pretrained=pretrained, epochs=cfg['nepochs'],
                             ft_epochs=ft_epochs, ft_lr=ft_lr, weight_decay=cfg['weight_decay'],
                             bsize=cfg['bsize'], latent_dim=cfg['latent_dim'])
        else:
            vae, *_ = train_fc_vae(x[tr], demo_tr, DEMO_TYPES, cfg)

        fc_pred = _fc_predictions(vae, x[tr], sex[tr], x[te], demo_te)
        for name, pred in fc_pred.items():
            sq_err[name] = sq_err.get(name, 0.0) + float(((pred - x[te]) ** 2).sum())
            count[name] = count.get(name, 0) + x[te].size

        preds = _outcome_predictions(vae, x[tr], wab[tr], d_num[tr], x[te], d_num[te], fc_pred)
        for name, pred in preds.items():
            oof.setdefault(name, np.zeros(len(x)))[te] = pred

    return {
        'fc_rmse': {k: float(np.sqrt(sq_err[k] / count[k])) for k in sq_err},
        'outcome': {k: {'rmse': float(np.sqrt(np.mean((p - wab) ** 2))), 'r2': float(r2_score(wab, p))}
                    for k, p in oof.items()},
    }


def _print_report(res):
    print('\nHeld-out FC RMSE (lower is better)')
    for k, v in res['fc_rmse'].items():
        print(f'  {k:<16}{v:.4f}')
    print('\nOut-of-fold WAB-AQ prediction')
    for k, v in res['outcome'].items():
        print(f"  {k:<16}RMSE {v['rmse']:.2f}   R2 {v['r2']:+.3f}")


def _load_hf(data_dir):  # untested: needs nilearn/datasets and network
    from data_preprocessing import preprocess_fmri_to_fc
    from main import load_data
    _, demo, types = load_data(data_dir)
    x, demo, _ = preprocess_fmri_to_fc(data_dir, demo, types)
    age, sex, mpo, wab = demo
    return x, age, sex, mpo, wab


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--npz', help='npz with X, age, sex, mpo, wab (default: SreekarB/OSFData)')
    ap.add_argument('--folds', type=int, default=5)
    ap.add_argument('--seed', type=int, default=0)
    ap.add_argument('--latent_dim', type=int, default=MODEL_CONFIG['latent_dim'])
    ap.add_argument('--model', choices=['demovae', 'pca', 'ft_scratch', 'ft_zeroshot', 'ft_finetune'], default='demovae',
                    help="pca = frozen PCA encoder + ridge decoder (rows named vae_* are then this model)")
    ap.add_argument('--pretrained', default='pretrained.pt', help='checkpoint from pretrain.py (ft_zeroshot/ft_finetune)')
    ap.add_argument('--ft_epochs', type=int, default=100)
    ap.add_argument('--ft_lr', type=float, default=1e-4)
    ap.add_argument('--nepochs', type=int, default=MODEL_CONFIG['nepochs'])
    args = ap.parse_args()
    if args.npz:
        d = np.load(args.npz, allow_pickle=False)
        data = tuple(d[k] for k in ('X', 'age', 'sex', 'mpo', 'wab'))
    else:
        data = _load_hf('SreekarB/OSFData')
    _print_report(evaluate_cv(*data, n_splits=args.folds, seed=args.seed,
                                model_config={'latent_dim': args.latent_dim, 'nepochs': args.nepochs},
                                model=args.model, pretrained=args.pretrained,
                                ft_epochs=args.ft_epochs, ft_lr=args.ft_lr))
