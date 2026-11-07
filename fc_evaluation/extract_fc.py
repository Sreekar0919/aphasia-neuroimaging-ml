"""Build data_<atlas>.npz (X, age, sex, mpo, wab) from SreekarB/OSFData, one subject at a time.

The scans are not in a common space (per-subject voxel sizes/origins, cropped FOV), so
atlas coordinates can't be sampled directly. For each subject we:
  1. average the 4D scan over time and mask the brain (nilearn EPI mask),
  2. register that mean EPI to the MNI152 brain with ANTs (affine + SyN),
  3. carry the atlas labels (Power-264 spheres or AAL-116) from MNI back into native space,
  4. average the scan inside each warped ROI, clean (detrend, 0.01-0.1 Hz, z-score),
     and compute Fisher-z FC.
ROIs with less than half their expected volume inside the native brain mask are NaN
for that subject; only ROIs valid in every subject go into X (mask saved as roi_keep).
Each ~265 MB NIfTI is downloaded, reduced and deleted, so disk use stays at one scan.
Re-running resumes from the cache. Registration QC goes to registration_qc_<atlas>.csv.

    python extract_fc.py [--atlas power|aal] [P01 P02 ...]   # ids for testing; npz needs all
"""
import argparse
import json
import tempfile
from pathlib import Path

import ants
import nibabel as nib
import numpy as np
import pandas as pd
from huggingface_hub import hf_hub_download
from nilearn import connectome, datasets, signal
from nilearn.image import resample_to_img
from nilearn.masking import compute_epi_mask

REPO = 'SreekarB/OSFData'
MNI_DIR = Path('mni_cache')
RADIUS = 5  # mm, Power spheres
MIN_ROI_FRAC = 0.5  # of expected ROI volume that must lie inside the native brain mask
CACHE_DIR = {'power': 'fc_cache_mni', 'aal': 'fc_cache_aal'}
OUT_NPZ = {'power': 'data_mni.npz', 'aal': 'data_aal.npz'}


def power_coords():
    power = datasets.fetch_coords_power_2011()
    return np.vstack((power.rois['x'], power.rois['y'], power.rois['z'])).T


def build_labels(tpl_img, coords):
    """Sphere labels 1..N (radius RADIUS mm) at MNI coords on the template grid."""
    lab = np.zeros(tpl_img.shape, np.int16)
    inv = np.linalg.inv(tpl_img.affine)
    r = int(np.ceil(RADIUS / np.abs(np.diag(tpl_img.affine)[:3]).min())) + 1
    steps = np.arange(-r, r + 1)
    offs = np.stack(np.meshgrid(steps, steps, steps, indexing='ij'), -1).reshape(-1, 3)
    for k, c in enumerate(coords):
        ijk = np.rint(nib.affines.apply_affine(inv, c)).astype(int) + offs
        ijk = ijk[((ijk >= 0) & (ijk < tpl_img.shape)).all(1)]
        near = ijk[np.linalg.norm(nib.affines.apply_affine(tpl_img.affine, ijk) - c, axis=1) <= RADIUS]
        lab[tuple(near.T)] = k + 1
    return lab


def aal_labels(tpl_img):
    """AAL-116 labels renumbered 1..116 (in AAL index order, as in ABIDE's rois_aal) on the template grid."""
    aal = datasets.fetch_atlas_aal(version='SPM12')
    maps = nib.load(aal.maps) if isinstance(aal.maps, (str, Path)) else aal.maps
    vals = np.rint(np.asarray(resample_to_img(maps, tpl_img, interpolation='nearest').dataobj)).astype(int)
    out = np.zeros(vals.shape, np.int16)
    for k, v in enumerate(sorted(int(i) for i in aal.indices if int(i) != 0)):
        out[vals == v] = k + 1
    return out


def mni_files(atlas):
    """Brain-extracted MNI152 template, atlas labels and brain mask (2 mm), cached on disk."""
    MNI_DIR.mkdir(exist_ok=True)
    brain, mask, labels = MNI_DIR / 'brain.nii.gz', MNI_DIR / 'mask.nii.gz', MNI_DIR / f'labels_{atlas}.nii.gz'
    tpl = datasets.load_mni152_template(resolution=2)
    if not (brain.exists() and mask.exists()):
        m = datasets.load_mni152_brain_mask(resolution=2)
        nib.save(nib.Nifti1Image(np.asarray(tpl.dataobj) * (np.asarray(m.dataobj) > 0), tpl.affine), brain)
        nib.save(m, mask)
    if not labels.exists():
        lab = build_labels(tpl, power_coords()) if atlas == 'power' else aal_labels(tpl)
        nib.save(nib.Nifti1Image(lab, tpl.affine), labels)
    return brain, labels, mask


def register(mean_img, epi_mask, tmp, brain_path):
    brain = ants.image_read(str(brain_path))
    path = Path(tmp) / 'epi.nii.gz'
    nib.save(nib.Nifti1Image(np.asarray(mean_img.dataobj) * epi_mask, mean_img.affine), path)
    epi = ants.image_read(str(path))
    return brain, epi, ants.registration(
        fixed=brain, moving=epi, type_of_transform='SyN', outprefix=str(Path(tmp) / 'reg_'), verbose=False)


def labels_to_native(reg, epi, labels_path, tmp):
    lab = ants.apply_transforms(
        fixed=epi, moving=ants.image_read(str(labels_path)), transformlist=reg['invtransforms'],
        whichtoinvert=[True, False], interpolator='genericLabel')
    out = Path(tmp) / 'labels_native.nii.gz'
    ants.image_write(lab, str(out))
    return np.rint(np.asarray(nib.load(out).dataobj)).astype(np.int16)


def mask_precision(reg, brain, epi_mask_img, mask_path, tmp):
    """Fraction of the subject's brain mask that lands inside the MNI brain mask."""
    p = Path(tmp) / 'epimask.nii.gz'
    nib.save(epi_mask_img, p)
    warped = ants.apply_transforms(
        fixed=brain, moving=ants.image_read(str(p)), transformlist=reg['fwdtransforms'],
        interpolator='nearestNeighbor').numpy() > 0.5
    mni = np.asarray(nib.load(mask_path).dataobj) > 0
    return float((warped & mni).sum() / max(warped.sum(), 1))


def subject_fc(atlas, nii_path, tmp, t_r=2.0):
    """(N, N) Fisher-z FC (NaN for ROIs outside the brain/FOV) and a QC dict."""
    brain_path, labels_path, mask_path = mni_files(atlas)
    tpl_labels = np.asarray(nib.load(labels_path).dataobj)
    n_rois = int(tpl_labels.max())
    expected_mm3 = np.bincount(tpl_labels.ravel(), minlength=n_rois + 1)[1:] * 8.0  # 2 mm voxels

    img = nib.load(nii_path)
    data = np.asarray(img.dataobj, dtype=np.float32)
    mean_img = nib.Nifti1Image(data.mean(-1), img.affine)
    mask_img = compute_epi_mask(mean_img)
    epi_mask = np.asarray(mask_img.dataobj) > 0

    brain, epi, reg = register(mean_img, epi_mask, tmp, brain_path)
    lab = labels_to_native(reg, epi, labels_path, tmp) * epi_mask

    vox_mm3 = abs(np.linalg.det(img.affine[:3, :3]))
    counts = np.bincount(lab.ravel(), minlength=n_rois + 1)[1:]
    ok = counts >= MIN_ROI_FRAC * expected_mm3 / vox_mm3
    flat, lab_flat = data.reshape(-1, data.shape[-1]), lab.ravel()
    ts = np.stack([flat[lab_flat == k + 1].mean(0) for k in np.where(ok)[0]], axis=1)
    ts = signal.clean(ts, detrend=True, standardize='zscore_sample', low_pass=0.1, high_pass=0.01, t_r=t_r)
    corr = connectome.ConnectivityMeasure(kind='correlation').fit_transform([ts])[0]
    np.fill_diagonal(corr, 0)
    full = np.full((n_rois, n_rois), np.nan)
    full[np.ix_(ok, ok)] = np.arctanh(corr)

    qc = {'rois_valid': int(ok.sum()),
          'mask_precision_after': mask_precision(reg, brain, mask_img, mask_path, tmp)}
    return full, qc


if __name__ == '__main__':
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument('--atlas', choices=['power', 'aal'], default='power')
    ap.add_argument('ids', nargs='*')
    args = ap.parse_args()
    cache = Path(CACHE_DIR[args.atlas])
    cache.mkdir(exist_ok=True)
    cov = pd.read_csv(hf_hub_download(REPO, 'FC_graph_covariate_data.csv', repo_type='dataset'))
    for sid in args.ids or list(cov['ID']):
        out = cache / f'{sid}.npy'
        if out.exists():
            continue
        with tempfile.TemporaryDirectory() as tmp:
            nii = hf_hub_download(REPO, f'{sid}_rs.nii', repo_type='dataset', cache_dir=tmp)
            fc, qc = subject_fc(args.atlas, nii, tmp)
        np.save(out, fc)
        (cache / f'{sid}.json').write_text(json.dumps(qc))
        print(sid, qc, flush=True)
    if args.ids:
        raise SystemExit
    pd.DataFrame({sid: json.loads((cache / f'{sid}.json').read_text()) for sid in cov['ID']}).T.to_csv(
        f'registration_qc_{args.atlas}.csv')
    fcs = np.stack([np.load(cache / f'{sid}.npy') for sid in cov['ID']])
    keep = ~np.isnan(np.diagonal(fcs, axis1=1, axis2=2)).any(axis=0)  # ROI valid in every subject
    iu = np.triu_indices(keep.sum(), k=1)
    X = np.stack([f[np.ix_(keep, keep)][iu] for f in fcs])
    assert np.isfinite(X).all(), 'non-finite FC values'
    np.savez(OUT_NPZ[args.atlas], X=X, age=cov['age'].values, sex=np.asarray(cov['gender'].tolist(), dtype='U1'),
             mpo=cov['mpo'].values, wab=cov['wab_aq'].values, roi_keep=keep)
    print(f'kept {keep.sum()}/{len(keep)} ROIs; wrote {OUT_NPZ[args.atlas]}', X.shape)
