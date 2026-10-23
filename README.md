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

## Data

None is included. Download ARC from OpenNeuro (ds004884), plus the FigShare and OSF releases. Place
them in `ARCData/`, `FigShare/` and `OSFData/` (see the `load_*` functions in `preprocessing.py`).

## Tech

PyTorch · MONAI Generative · nibabel · scikit-learn · pandas
