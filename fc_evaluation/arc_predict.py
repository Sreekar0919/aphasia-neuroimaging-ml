"""WAB-AQ prediction in ARC (n~140, CV): demographics vs lesion volume vs FC, alone and combined."""
import json
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from arc import CACHE, participants

COVERAGE = 0.9
ppl, lesion = participants(), json.load(open('arc_lesion_ml.json'))
subs = sorted(f.stem for f in CACHE.glob('*.npy') if f.stem in lesion)
fcs = np.stack([np.load(CACHE / f'{s}.npy') for s in subs])
ok = ~np.isnan(np.diagonal(fcs, axis1=1, axis2=2))
keep = ok.mean(0) >= COVERAGE  # ROIs valid in most scans; scans missing any of them are dropped
subs = [s for s, o in zip(subs, ok) if o[keep].all()]
fcs = fcs[[o[keep].all() for o in ok]]
iu = np.triu_indices(keep.sum(), 1)
X = np.stack([f[np.ix_(keep, keep)][iu] for f in fcs])
p = ppl.loc[subs]
age = (p.age_at_stroke + p.wab_days / 365.25).values
demo = np.c_[age, (p.sex == 'M').values, np.log(p.wab_days.values)]
les = np.log(np.array([lesion[s] for s in subs]))[:, None]
y = p.wab_aq.values
print(f'{len(subs)} subjects, {keep.sum()} ROIs valid in all, {X.shape[1]} edges')

r2 = lambda q: 1 - ((y - q) ** 2).sum() / ((y - y.mean()) ** 2).sum()
sets = {'demographics': demo, 'lesion': les, 'demo+lesion': np.c_[demo, les], 'FC': X,
        'demo+lesion+FC': np.c_[demo, les, X]}
for name, F in sets.items():
    scores = []
    for seed in range(5):  # repeated CV: n is small, one split is noisy
        m = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(0, 6, 13)))
        scores.append(r2(cross_val_predict(m, F, y, cv=KFold(5, shuffle=True, random_state=seed))))
    print(f'{name:16s} R2 {np.mean(scores):+.3f} +/- {np.std(scores):.3f}')

# FC on top of lesion: fit lesion line first, let FC ridge explain only the residual (nested inside each fold)
from sklearn.linear_model import LinearRegression
scores = []
for seed in range(5):
    q = np.zeros(len(y))
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(y):
        base = LinearRegression().fit(les[tr], y[tr])
        res = y[tr] - base.predict(les[tr])
        fc = make_pipeline(StandardScaler(), RidgeCV(alphas=np.logspace(2, 7, 11))).fit(X[tr], res)
        q[te] = base.predict(les[te]) + fc.predict(X[te])
    scores.append(r2(q))
print(f'{"lesion then FC":16s} R2 {np.mean(scores):+.3f} +/- {np.std(scores):.3f}')
