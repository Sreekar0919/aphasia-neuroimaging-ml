import numpy as np
import torch

from src.demovae.sklearn import DemoVAE


def train_fc_vae(X, demo_data, demo_types, model_config):
    """Train a DemoVAE on upper-triangular functional connectivity vectors."""
    X = np.nan_to_num(np.asarray(X, dtype=np.float32))
    demo_data = [np.asarray(d) for d in demo_data]

    # The covariance loss needs more samples per batch than latent dims.
    bsize = min(model_config['bsize'], len(X))
    if model_config['latent_dim'] >= bsize:
        raise ValueError(
            f"latent_dim ({model_config['latent_dim']}) must be smaller than the "
            f"effective batch size ({bsize}, capped at the number of subjects)")

    params = {k: v for k, v in model_config.items() if k in DemoVAE.get_default_params()}
    params.update(bsize=bsize, use_cuda=model_config.get('use_cuda', torch.cuda.is_available()))
    vae = DemoVAE(**params)
    vae.fit(X, demo_data, demo_types)
    return vae, X, demo_data, demo_types
