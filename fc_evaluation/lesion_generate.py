"""Can a conditioned model generate a patient's FC without their fMRI? Held-out test in ARC.

The conditioned decoder's expected output for a new patient (latent at its mean, z = 0) is
mean FC + B * conditions, so each condition set is scored by fitting B (ridge, inside the training
fold) and generating held-out patients' FC from their conditions alone. Condition sets:
demographics + WAB-AQ, lesion volume, lesion location (fraction of each JHU region lesioned, from
the ARC authors' MNI-normalised lesion maps), and everything together. A permutation null (conditions
shuffled across patients, whole CV refit) shows what 'better than mean FC' looks like by chance.

    python lesion_generate.py --npz data_aal.npz
"""
import argparse
import json
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.model_selection import KFold
from arc import load_arc_fc, participants

ALPHAS = np.logspace(-1, 5, 13)
SEEDS = range(5)
N_PERM = 200
JHU = Path('arc_lesion/merged_jhu_participants.tsv')
JHU_URL = ('https://raw.githubusercontent.com/neurolabusc/AphasiaRecoveryCohortDemo/main/Scripts/'
           'merged_jhu_participants.tsv')  # the ARC authors' lesion load per JHU region, MNI space (BSD-2)


def conditions(keep):
    x, age, male, mpo, wab, ids = load_arc_fc(keep, with_ids=True)
    if not JHU.exists():
        JHU.parent.mkdir(exist_ok=True)
        urllib.request.urlretrieve(JHU_URL, JHU)
    jhu = pd.read_csv(JHU, sep='\t').set_index('participant_id')
    vol = json.load(open('arc_lesion_ml.json'))
    ok = [i for i, s in enumerate(ids) if s in jhu.index and s in vol]
    ids = [ids[i] for i in ok]
    loc = jhu.loc[ids].drop(columns=['age_at_stroke', 'wab_aq']).values
    demo = zscore(np.c_[age[ok], male[ok], np.log(mpo[ok]), wab[ok]])
    dx = pd.get_dummies(participants().loc[ids, 'wab_type']).values.astype(float)  # aphasia type
    volume = zscore(np.log([vol[s] for s in ids])[:, None])
    # lesion loads stay as raw fractions: z-scoring regions that are almost never lesioned blows up
    sets = {'demographics + WAB': demo, 'diagnosis + demographics + WAB': np.c_[dx, demo],
            'lesion volume': volume, 'lesion location': loc,
            'location + volume + dx + demo + WAB': np.c_[loc, volume, dx, demo]}
    return x[ok], sets


def zscore(a):
    return (a - a.mean(0)) / a.std(0)


def generate_cv(x, c, seed):
    """Out-of-fold generated FC from conditions c (ridge decoder fit on the training fold only)."""
    out = np.zeros_like(x)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        out[te] = RidgeCV(alphas=ALPHAS).fit(c[tr], x[tr]).predict(c[te])
    return out


def mean_cv(x, seed):
    out = np.zeros_like(x)
    for tr, te in KFold(5, shuffle=True, random_state=seed).split(x):
        out[te] = x[tr].mean(0)
    return out


def scores(pred, x, base):
    r = np.mean([np.corrcoef(a, b)[0, 1] for a, b in zip(pred, x)])
    return r, 1 - ((pred - x) ** 2).sum() / ((base - x) ** 2).sum()


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--npz', default='data_aal.npz')
    args = ap.parse_args()
    x, sets = conditions(np.load(args.npz)['roi_keep'])
    print(f'ARC patients with FC + lesion maps: {len(x)}; {x.shape[1]} FC edges; '
          f'{sets["lesion location"].shape[1]} lesioned JHU regions')
    bases = {s: mean_cv(x, s) for s in SEEDS}
    r_mean = np.mean([scores(bases[s], x, bases[s])[0] for s in SEEDS])
    print(f'\n{"generated from":36s}{"r":>7s}{"skill vs mean FC":>18s}{"perm p":>8s}')
    print(f'{"nothing (mean FC)":36s}{r_mean:7.3f}{0:+18.3f}')
    rng = np.random.default_rng(0)
    for name, c in sets.items():
        r, sk = np.mean([scores(generate_cv(x, c, s), x, bases[s]) for s in SEEDS], 0)
        null = [scores(generate_cv(x, c[rng.permutation(len(c))], 0), x, bases[0])[1] for _ in range(N_PERM)]
        p = (1 + np.sum(np.array(null) >= sk)) / (1 + N_PERM)
        print(f'{name:36s}{r:7.3f}{sk:+18.3f}{p:8.3f}')
