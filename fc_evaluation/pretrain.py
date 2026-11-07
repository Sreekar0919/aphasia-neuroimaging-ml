"""Pretrain the FC model on ABIDE-I (AAL) over the ROIs valid in every aphasia subject.

    python pretrain.py --npz data_aal.npz --out pretrained.pt
"""
import argparse

import numpy as np
import torch

from abide import load_abide_fc
from finetune import FCModel


def pretrain(x, age, male, latent_dim=16, epochs=300, bsize=128, val_frac=0.1, seed=0):
    """Fit on a train split, return (model, val_rmse_model, val_rmse_mean_fc_baseline)."""
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    perm = rng.permutation(len(x))
    n_val = int(len(x) * val_frac)
    va, tr = perm[:n_val], perm[n_val:]
    sex = np.where(male == 1, 'M', 'F')
    model = FCModel(x.shape[1], latent_dim)
    model.age_stats = (age[tr].mean(), age[tr].std())
    model.fit(x[tr], [age[tr], sex[tr], np.zeros(len(tr))], nepochs=epochs, bsize=bsize, decor_cols=(0, 1))
    recon = model.transform(x[va], [age[va], sex[va], np.zeros(len(va))])
    rm = lambda p: float(np.sqrt(np.mean((p - x[va]) ** 2)))
    return model, rm(recon), rm(np.tile(x[tr].mean(0), (len(va), 1)))


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--npz', default='data_aal.npz', help='aphasia npz holding the roi_keep mask')
    ap.add_argument('--out', default='pretrained.pt')
    ap.add_argument('--latent_dim', type=int, default=16)
    ap.add_argument('--epochs', type=int, default=300)
    args = ap.parse_args()
    x, age, male = load_abide_fc(np.load(args.npz)['roi_keep'])
    print(f'ABIDE: {x.shape[0]} subjects, {x.shape[1]} FC edges, age {age.min():.0f}-{age.max():.0f}')
    model, val_rmse, base_rmse = pretrain(x, age, male, args.latent_dim, args.epochs)
    print(f'held-out ABIDE FC RMSE: model {val_rmse:.4f} vs mean-FC baseline {base_rmse:.4f}')
    model.save(args.out)
