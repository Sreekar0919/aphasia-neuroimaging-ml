"""Weight-space programming: write the FC VAE's weights in closed form instead of training them.

The network is enc: x -> ReLU(enc1) -> enc2 -> z, dec: [z, demo] -> ReLU(dec1) -> dec2 -> x.
A ReLU passes a value through unchanged when it is shifted to be positive: ReLU(u + C) - C = u
for u > -C. One hidden unit per latent (or demographic) dimension therefore carries a linear map
exactly, so the 1000-unit layers hold up to 1000 dimensions.

Conditioned version (DemoVAE's design, solved instead of trained):
  demographic map   B: FC change per unit of [age, male, months post onset, WAB-AQ] (standardised),
                    fit within each cohort (ARC + the training fold) so site offsets don't leak in
  encoder           z = V (x - mu) / s, with V a PCA basis orthogonal to B's columns, so z cannot
                    carry the demographic directions (the decorrelation DemoVAE trains for)
  decoder           x = mu + B (d - d_mean) + V^T (s * z)
With B = None the decoder ignores demographics (unconditioned). V comes from a broad pool
(ABIDE + ARC + training fold), each cohort centred on its own mean. Spare hidden units keep random
input weights and zero output weights, so they don't change the function.
"""
import numpy as np
import torch
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

OFFSET = 100.0  # C above; latents and standardised demographics of real data stay well inside +/- C
ALPHAS = np.logspace(-2, 4, 13)


def demographic_map(cohorts):
    """B (D, n_demo): ridge fit of FC on standardised demographics, centred within each (x, d) cohort."""
    x = np.vstack([c - c.mean(0) for c, _ in cohorts])
    d = np.vstack([c - c.mean(0) for _, c in cohorts])
    return RidgeCV(alphas=ALPHAS, fit_intercept=False).fit(d, x).coef_


def pooled_basis(cohorts, k, B=None):
    """PCA components (k, D) and score std (k,) of FC cohorts, each centred on its own mean,
    after removing the directions spanned by B's columns (if given)."""
    pool = np.vstack([c - c.mean(0) for c in cohorts])
    if B is not None:
        q = np.linalg.qr(B)[0]
        pool = pool - (pool @ q) @ q.T
    pca = PCA(k, random_state=0).fit(pool)
    return pca.components_, np.sqrt(pca.explained_variance_)


def program(model, x, components, s, demo=None, B=None):
    """Write enc/dec weights of `model` (finetune.FCModel, latent_dim == len(components)) from the
    training fold `x`; pass its demographics and B to condition the decoder."""
    vae, k = model.vae, model.latent_dim
    n_demo = vae.demo_dim if B is not None else 0
    assert len(components) == k and k + n_demo <= vae.dec1.out_features, 'one hidden unit per dimension'
    mu = x.mean(0)
    w_enc = components / s[:, None]
    w_dec = components.T * s
    bias = mu - w_dec.sum(1) * OFFSET
    if B is not None:
        d_mean = model._demo(demo).numpy().mean(0)
        w_dec = np.hstack([w_dec, B])
        bias = bias - B @ d_mean - B.sum(1) * OFFSET
    m = k + n_demo
    t = lambda a: torch.tensor(a, dtype=torch.float32)
    with torch.no_grad():
        vae.enc1.weight[:k] = t(w_enc)
        vae.enc1.bias[:k] = t(OFFSET - w_enc @ mu)
        vae.enc2.weight.zero_()
        vae.enc2.weight[:, :k] = torch.eye(k)
        vae.enc2.bias.fill_(-OFFSET)
        vae.dec1.weight[:m] = 0
        vae.dec1.weight[:m, :m] = torch.eye(m)
        vae.dec1.bias[:m] = OFFSET
        vae.dec2.weight.zero_()
        vae.dec2.weight[:, :m] = t(w_dec)
        vae.dec2.bias.copy_(t(bias))
    return model
