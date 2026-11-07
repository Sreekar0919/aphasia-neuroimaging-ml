# Aphasia Neuroimaging ML

Experiments (January–March 2025) on predicting aphasia outcomes from brain imaging, using public
stroke-aphasia datasets.

## 1. Treatment-outcome prediction from functional connectivity

[`notebooks/aphasia_treatment_prediction.ipynb`](notebooks/aphasia_treatment_prediction.ipynb) +
[`preprocessing.py`](preprocessing.py)

- **Data:** three public sources, harmonized into a single table by `preprocessing.py`:
  - **ARC** (Aphasia Recovery Cohort, OpenNeuro ds004884): resting state, cross-sectional
  - **FigShare**: task fMRI, longitudinal
  - **OSF**: resting state, longitudinal
  
  Outcomes are put on one scale: Western Aphasia Battery AQ, with Aachen Aphasia Test subscores
  converted to WAB (`convert_aat_to_wab`). Connectivity matrices are vectorized
  (`mat2vec` / `vec2mat`) and standardized.
- **Model:** a variational autoencoder (1000 hidden units, 30-d latent space, demographic covariates
  fed to the decoder) compresses each connectome. Linear, ridge, random-forest and
  gradient-boosting regressors then predict WAB-AQ from the latent features plus covariates. Evaluation uses held-out R², MAE and feature importance, and
  the notebook also projects recovery trajectories.
- `functional_connectivity_extraction.ipynb`: remaps the parcellation and builds the matrices.
  `vae_test_sample.ipynb`: the same idea with DemoVAE (demographically conditioned VAE, 60-d latent).

## 2. Generating aphasia-conditioned brain MRI

[`notebooks/latent_diffusion_aphasia_conditions.ipynb`](notebooks/latent_diffusion_aphasia_conditions.ipynb)
adapts MONAI Generative Models' 3-D latent diffusion tutorial. A KL autoencoder (300 epochs, with
adversarial warm-up) is followed by a latent diffusion model. Scans are matched to participant
metadata, and the diffusion U-Net is conditioned on those aphasia characteristics through cross-attention.

## 3. Held-out re-evaluation of the connectivity models (2026)

[`fc_evaluation/`](fc_evaluation/) re-tests section 1 with held-out evaluation: every number below
comes from patients the model never saw, and is compared with a simple baseline. These results
replace the earlier in-sample numbers. In the original pipeline WAB-AQ was both a model input and the
prediction target, and nothing was compared with a baseline.

**Registration.** The OSF scans were not in a common space: voxel sizes, origins and field of view
differed between subjects. Each scan's mean EPI is now registered to MNI152 with ANTs (affine + SyN)
before atlas time series are extracted. After registration, 98–99.5% of each subject's brain mask
falls inside the MNI brain mask. Because the OSF field of view is cropped, 61 of the 116 AAL regions
are covered in every subject.

**Connectivity reconstruction (OSF, n = 30, 5-fold CV;
[`results/eval_recon_accuracy.txt`](fc_evaluation/results/eval_recon_accuracy.txt)).**

| Model | Correlation with real FC | RMSE |
|---|---|---|
| Group-average FC (baseline) | 0.623 | 0.276 |
| VAE, pretrained on ABIDE adults then fine-tuned | 0.610 | 0.271 |
| VAE, ABIDE adults → ARC → OSF staged fine-tuning | 0.619 | 0.269 |
| PCA (16 components) + ridge decoder | 0.684 | 0.244 |

The VAEs do no better than the group-average connectivity, and a 16-component PCA does better than
both. FC generated from demographics alone (no scan) is worse than the group average (r ≈ 0.45 vs 0.62).

**Predicting WAB-AQ.**
- *OSF (n = 30):* no model beats predicting the training-set mean, whether it uses real, reconstructed
  or generated FC (RMSE 25.4, R² −0.07; [`results/`](fc_evaluation/results/)).
- *ARC (n = 80 with 76 well-covered regions, 5 × 5-fold CV;
  [`results/eval_arc_predict.txt`](fc_evaluation/results/eval_arc_predict.txt)):*

| Predictors | R² |
|---|---|
| Age, sex, time since stroke | −0.05 |
| Resting-state FC | +0.14 |
| Lesion volume | +0.37 |
| Lesion volume, then FC fitted to the residual | **+0.42** |

  Lesion size carries most of the signal, and connectivity adds a small but real increment.

**Generating FC from clinical information (ARC, n = 93;
[`results/eval_lesion_generate.txt`](fc_evaluation/results/eval_lesion_generate.txt)).** This tests
whether a patient's FC can be produced from data they already have, without an fMRI scan.

| Generated from | Error removed vs group-average FC | Permutation p |
|---|---|---|
| Demographics + WAB-AQ | 1.3% | 0.010 |
| Aphasia type + demographics + WAB-AQ | 1.2% | 0.005 |
| Lesion location (ARC authors' MNI lesion maps) | 2.5% | 0.005 |

Clinical information beats chance, but most of what makes one person's FC differ from the average
is not captured. On these data, scores and demographics cannot replace a scan.

**Limits.** These are single runs with small samples. WAB-AQ here is a cross-sectional severity score,
not recovery over time. ABIDE (mostly young, autism and controls) is a large domain shift from stroke.

**Running it.** `pip install -r fc_evaluation/requirements.txt`, then from `fc_evaluation/`:
`python extract_fc.py --atlas aal` (OSF → `data_aal.npz`), `python abide.py`,
`python arc.py --index` then `python arc.py` (streams ARC rest scans, ~280 MB each, one at a time), and
`python evaluate.py`, `staged.py`, `recon_accuracy.py`, `arc_predict.py` or `lesion_generate.py`.
`pytest` runs the tests, which need no data. `src/demovae/` is DemoVAE (MIT, Anton Orlichenko).

## Data

None is included. Download ARC from OpenNeuro (ds004884), plus the FigShare and OSF releases. Place
them in `ARCData/`, `FigShare/` and `OSFData/` (see the `load_*` functions in `preprocessing.py`).

## Tech

PyTorch · MONAI Generative · ANTsPy · nilearn · nibabel · scikit-learn · pandas
