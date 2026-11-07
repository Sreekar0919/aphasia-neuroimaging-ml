"""Low-capacity stand-in for DemoVAE: frozen PCA encoder + ridge decoder.

Same interface as DemoVAE (fit / transform / get_latents) so evaluate.py can swap it in.
Only k PCA scores plus three demographics feed a single ridge-regularised linear decoder,
so there are (k + 3) * n_features parameters instead of ~35M.
"""
import numpy as np
from sklearn.decomposition import PCA
from sklearn.linear_model import RidgeCV

ALPHAS = np.logspace(-2, 4, 13)


class LowCapModel:
    def __init__(self, n_components=5, seed=0):
        self.n_components = n_components
        self.rng = np.random.default_rng(seed)

    def _demo(self, demo):
        age, sex, mpo = demo
        return np.column_stack([(np.asarray(age, float) - self.age_stats[0]) / self.age_stats[1],
                                (np.asarray(sex) == self.sex0).astype(float),
                                (np.asarray(mpo, float) - self.mpo_stats[0]) / self.mpo_stats[1]])

    def fit(self, x, demo, demo_types=None):
        age, sex, mpo = demo
        self.age_stats = (np.mean(age), np.std(age))
        self.mpo_stats = (np.mean(mpo), np.std(mpo))
        self.sex0 = np.unique(sex)[0]
        self.pca = PCA(self.n_components, random_state=0).fit(x)
        z = self.pca.transform(x)
        self.z_std = z.std(0)
        self.decoder = RidgeCV(alphas=ALPHAS).fit(np.hstack([z / self.z_std, self._demo(demo)]), x)
        return self

    def get_latents(self, x):
        return self.pca.transform(x) / self.z_std

    def transform(self, x, demo, demo_types=None):
        z = self.rng.standard_normal((x, self.n_components)) if isinstance(x, int) else self.get_latents(x)
        return self.decoder.predict(np.hstack([z, self._demo(demo)]))
