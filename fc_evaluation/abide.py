"""Download ABIDE-I (CPAC, band-pass filtered, no global signal regression) AAL ROI time series.

Public fcp-indi S3 bucket, ~230 KB per subject. Files land in abide/rois_aal/; re-running
resumes. `load_abide_fc(keep)` turns them into Fisher-z FC vectors over the ROI subset `keep`.

    python abide.py
"""
import urllib.error
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd

BASE = 'https://s3.amazonaws.com/fcp-indi/data/Projects/ABIDE_Initiative'
OUT = Path('abide')
MIN_TRS = 100  # shorter series give noisy FC


def _fetch(url, dest):
    if dest.exists():
        return True
    try:
        with urllib.request.urlopen(url, timeout=60) as r:
            dest.write_bytes(r.read())
        return True
    except urllib.error.HTTPError:
        return False


def download():
    (OUT / 'rois_aal').mkdir(parents=True, exist_ok=True)
    pheno = OUT / 'phenotypic.csv'
    _fetch(f'{BASE}/Phenotypic_V1_0b_preprocessed1.csv', pheno)
    df = pd.read_csv(pheno)
    df = df[df['FILE_ID'] != 'no_filename']
    urls = {f: f'{BASE}/Outputs/cpac/filt_noglobal/rois_aal/{f}_rois_aal.1D' for f in df['FILE_ID']}
    with ThreadPoolExecutor(8) as ex:
        ok = list(ex.map(lambda f: _fetch(urls[f], OUT / 'rois_aal' / f'{f}_rois_aal.1D'), urls))
    print(f'{sum(ok)}/{len(ok)} subject files available')


def load_abide_fc(keep):
    """Fisher-z upper-triangle FC over ROIs `keep` (bool mask of 116), plus age and sex (1 = male)."""
    df = pd.read_csv(OUT / 'phenotypic.csv')
    iu = np.triu_indices(int(keep.sum()), k=1)
    xs, age, male = [], [], []
    for row in df[df['FILE_ID'] != 'no_filename'].itertuples():
        path = OUT / 'rois_aal' / f'{row.FILE_ID}_rois_aal.1D'
        if not path.exists():
            continue
        ts = np.loadtxt(path, skiprows=1)[:, keep]
        if len(ts) < MIN_TRS or (ts.std(0) == 0).any():
            continue  # too short, or an ROI missing from this scan's field of view
        fc = np.corrcoef(ts.T)
        np.fill_diagonal(fc, 0)
        xs.append(np.arctanh(np.clip(fc, -0.999, 0.999))[iu])
        age.append(row.AGE_AT_SCAN)
        male.append(float(row.SEX == 1))
    return np.stack(xs).astype(np.float32), np.array(age), np.array(male)


if __name__ == '__main__':
    download()
