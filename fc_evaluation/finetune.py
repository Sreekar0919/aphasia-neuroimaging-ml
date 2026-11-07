"""Pretrain a DemoVAE-style FC model on a broad cohort, then fine-tune it on aphasia.

Same network and losses as src/demovae (encoder 1000-unit ReLU layer, decoder fed
[z, demographics]; covariance + mean + reconstruction + decorrelation losses; the tiny
guidance loss is dropped). Demographics are always [age, male, months-post-onset, WAB-AQ]:
a stage whose cohort lacks a column gets a constant 0 there, and that decoder column is
zeroed again before the first stage that has it.
"""
import numpy as np
import torch

from src.demovae.model import VAE, decor_loss, latent_loss, rmse

N_DEMO = 4
MPO_COL, WAB_COL = 2, 3
W_C, W_MU, W_REC, W_DECOR = 1, 1, 100, 10  # reference defaults


class FCModel:
    def __init__(self, input_dim, latent_dim=16):
        self.latent_dim = latent_dim
        self.vae = VAE(input_dim, latent_dim, N_DEMO, use_cuda=False)
        self.age_stats = (0.0, 1.0)
        self.mpo_stats = (0.0, 1.0)
        self.wab_stats = (0.0, 1.0)

    def _demo(self, demo):
        age, sex, mpo, *wab = demo  # WAB-AQ is optional (absent = constant 0 column)
        wab = ((np.asarray(wab[0], float) - self.wab_stats[0]) / self.wab_stats[1]
               if wab else np.zeros(len(age)))
        return torch.tensor(np.column_stack([
            (np.asarray(age, float) - self.age_stats[0]) / self.age_stats[1],
            (np.asarray(sex) == 'M').astype(float),
            (np.asarray(mpo, float) - self.mpo_stats[0]) / self.mpo_stats[1],
            wab]), dtype=torch.float32)

    def fit(self, x, demo, nepochs=1000, lr=1e-4, weight_decay=1e-4, bsize=64,
            freeze_enc1=False, decor_cols=(0, 1, 2)):  # decor_cols: conditioning columns active this stage
        x, d = torch.tensor(x, dtype=torch.float32), self._demo(demo)
        for p in self.vae.enc1.parameters():
            p.requires_grad = not freeze_enc1
        opt = torch.optim.Adam([p for p in self.vae.parameters() if p.requires_grad],
                               lr=lr, weight_decay=weight_decay)
        for _ in range(nepochs):
            perm = torch.randperm(len(x))
            for s in range(0, len(x), bsize):
                idx = perm[s:s + bsize]
                if len(idx) <= self.latent_dim:  # covariance loss needs batch > latent_dim
                    continue
                xb, db = x[idx], d[idx]
                z = self.vae.enc(xb)
                y = self.vae.dec(z, db)
                loss_c, loss_mu, _, _ = latent_loss(z, False)
                loss_decor = decor_loss(z, db[:, list(decor_cols)], False)[0].sum()
                loss = W_C * loss_c + W_MU * loss_mu + W_REC * rmse(xb, y) + W_DECOR * loss_decor
                if torch.isfinite(loss):  # a batch with a constant demographic gives NaN decorrelation
                    opt.zero_grad()
                    loss.backward()
                    opt.step()
        return self

    def zero_columns(self, *cols):
        with torch.no_grad():
            for c in cols:
                self.vae.dec1.weight[:, self.latent_dim + c] = 0

    def get_latents(self, x):
        with torch.no_grad():
            return self.vae.enc(torch.tensor(x, dtype=torch.float32)).numpy()

    def transform(self, x, demo, demo_types=None):
        with torch.no_grad():
            z = torch.randn(x, self.latent_dim) if isinstance(x, int) else self.vae.enc(
                torch.tensor(x, dtype=torch.float32))
            return self.vae.dec(z, self._demo(demo)).numpy()

    def save(self, path):
        torch.save({'state': self.vae.state_dict(), 'input_dim': self.vae.input_dim,
                    'latent_dim': self.latent_dim, 'age_stats': self.age_stats,
                    'mpo_stats': self.mpo_stats, 'wab_stats': self.wab_stats}, path)

    @classmethod
    def load(cls, path):
        ck = torch.load(path, weights_only=False)
        model = cls(ck['input_dim'], ck['latent_dim'])
        model.vae.load_state_dict(ck['state'])
        model.age_stats = tuple(ck['age_stats'])
        model.mpo_stats, model.wab_stats = tuple(ck['mpo_stats']), tuple(ck['wab_stats'])
        return model


def fit_for_cv(kind, x, demo, pretrained=None, epochs=1000, ft_epochs=100, ft_lr=1e-4,
               weight_decay=1e-4, bsize=64, latent_dim=16, freeze_enc1=True):
    """Fit one CV fold: 'scratch' (random init), 'zeroshot' (pretrained as is) or 'finetune'."""
    age, _, mpo, *_ = demo
    if kind == 'scratch':
        model = FCModel(x.shape[1], latent_dim)
        model.age_stats = (np.mean(age), np.std(age))
    else:
        model = FCModel.load(pretrained)
        model.zero_columns(MPO_COL, WAB_COL)
    model.mpo_stats = (np.mean(mpo), np.std(mpo))
    bsize = min(bsize, len(x))
    if kind == 'scratch':
        model.fit(x, demo, nepochs=epochs, weight_decay=weight_decay, bsize=bsize)
    elif kind == 'finetune':
        model.fit(x, demo, nepochs=ft_epochs, lr=ft_lr, weight_decay=weight_decay, bsize=bsize,
                  freeze_enc1=freeze_enc1)
    return model
