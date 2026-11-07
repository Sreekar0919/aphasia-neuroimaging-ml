"""Stream Aphasia Recovery Cohort (OpenNeuro ds004884) resting-state scans into AAL FC, one at a time.

One rest run per subject (the largest), ~280 MB each, downloaded, registered to MNI with
extract_fc.subject_fc, reduced to a 116x116 Fisher-z matrix (NaN for ROIs outside the FOV) and deleted.
Re-running resumes. `load_arc_fc(keep)` returns X, age, male, mpo_months, wab_aq over ROI mask `keep`.

    python arc.py --index    # list rest runs (arc_rest.json) and lesion volumes (arc_lesion_ml.json)
    python arc.py [N]        # first N subjects (default all)
"""
import json
import re
import sys
import tempfile
import time
import urllib.parse
import urllib.request
from pathlib import Path

import nibabel as nib
import numpy as np
import pandas as pd

from extract_fc import subject_fc

S3 = 'https://s3.amazonaws.com/openneuro.org'
TR = 1.85
CACHE = Path('fc_cache_arc')
DAYS_PER_MONTH = 30.44
MIN_BYTES = 100_000_000


def s3_keys(prefix):
    """(key, size) for every object under `prefix` in the public OpenNeuro bucket."""
    keys, token = [], None
    while True:
        url = f'{S3}?list-type=2&prefix={prefix}&max-keys=1000'
        if token:
            url += '&continuation-token=' + urllib.parse.quote(token)
        page = urllib.request.urlopen(url, timeout=60).read().decode()
        keys += [(k, int(s)) for k, s in re.findall(r'<Key>([^<]+)</Key>.*?<Size>(\d+)</Size>', page)]
        token = (re.findall(r'<NextContinuationToken>([^<]+)', page) or [None])[0]
        if token is None:
            return keys


def build_index():
    rest = [(k, s) for k, s in s3_keys('ds004884/sub-') if 'task-rest' in k and k.endswith('_bold.nii.gz')]
    json.dump(rest, open('arc_rest.json', 'w'))
    vol = {}  # lesion volume (mL) from the lesion masks drawn on each subject's T2w scan
    with tempfile.TemporaryDirectory() as tmp:
        for key, _ in s3_keys('ds004884/derivatives/lesion_masks/'):
            if key.endswith('_mask.nii.gz'):
                urllib.request.urlretrieve(f'{S3}/{key}', f'{tmp}/m.nii.gz')
                img = nib.load(f'{tmp}/m.nii.gz')
                vol[key.split('/')[3]] = float((np.asarray(img.dataobj) > 0).sum()
                                               * abs(np.linalg.det(img.affine[:3, :3])) / 1000)
    json.dump(vol, open('arc_lesion_ml.json', 'w'))
    print(f'{len(rest)} rest runs, {len(vol)} lesion masks')


def participants():
    return pd.read_csv(f'{S3}/ds004884/participants.tsv', sep='\t').set_index('participant_id')


def first_rest_runs():
    runs = {}  # the longest (largest) rest run per subject; tiny files are truncated scans
    for key, size in sorted(json.load(open('arc_rest.json'))):
        sub = key.split('/')[1]
        if size >= MIN_BYTES and size > runs.get(sub, ('', 0))[1]:
            runs[sub] = (key, size)
    return {sub: key for sub, (key, _) in runs.items()}


def stream(n=None):
    CACHE.mkdir(exist_ok=True)
    for sub, key in list(first_rest_runs().items())[:n]:
        out = CACHE / f'{sub}.npy'
        if out.exists():
            continue
        t0 = time.time()
        with tempfile.TemporaryDirectory() as tmp:
            nii = Path(tmp) / 'rest.nii.gz'
            urllib.request.urlretrieve(f'{S3}/{key}', nii)
            fc, qc = subject_fc('aal', str(nii), tmp, t_r=TR)
        np.save(out, fc)
        print(sub, qc, f'{time.time() - t0:.0f}s', flush=True)


def load_arc_fc(keep, with_ids=False):
    ppl, fcs, rows, ids = participants(), [], [], []
    for f in sorted(CACHE.glob('*.npy')):
        fc = np.load(f)
        sub = f.stem
        if np.isnan(fc[np.ix_(keep, keep)]).any():
            continue  # a needed ROI fell outside this scan
        fcs.append(fc[np.ix_(keep, keep)][np.triu_indices(int(keep.sum()), k=1)])
        ids.append(sub)
        r = ppl.loc[sub]
        rows.append((r.age_at_stroke + r.wab_days / 365.25, float(r.sex == 'M'), r.wab_days / DAYS_PER_MONTH, r.wab_aq))
    age, male, mpo, wab = map(np.array, zip(*rows))
    out = (np.stack(fcs).astype(np.float32), age, male, mpo, wab)
    return (*out, ids) if with_ids else out


if __name__ == '__main__':
    if sys.argv[1:] == ['--index']:
        build_index()
    else:
        stream(int(sys.argv[1]) if len(sys.argv) > 1 else None)
