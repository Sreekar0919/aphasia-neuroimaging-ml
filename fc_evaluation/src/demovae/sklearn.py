
from .model import VAE, train_vae, to_torch, to_cuda, to_numpy, demo_to_torch
import numpy as np

from sklearn.base import BaseEstimator

# For saving
import torch

class DemoVAE(BaseEstimator):
    def __init__(self, **params):
        self.set_params(**params)

    @staticmethod
    def get_default_params():
        return dict(latent_dim=60,      # Latent dimension
                use_cuda=True,          # GPU acceleration
                nepochs=3000,           # Training epochs
                pperiod=100,            # Epochs between printing updates 
                bsize=1000,             # Batch size
                loss_C_mult=1,          # Covariance loss (KL div)
                loss_mu_mult=1,         # Mean loss (KL div)
                loss_rec_mult=100,      # Reconstruction loss
                loss_decor_mult=10,     # Latent-demographic decorrelation loss
                loss_pred_mult=0.001,   # Classifier/regressor guidance loss
                alpha=100,              # Regularization for continuous guidance models
                LR_C=100,               # Regularization for categorical guidance models
                lr=1e-4,                # Learning rate
                weight_decay=0,         # L2 regularization for VAE model
                )

    def get_params(self, **params):
        return dict(latent_dim=self.latent_dim,
                use_cuda=self.use_cuda,
                nepochs=self.nepochs,
                pperiod=self.pperiod,
                bsize=self.bsize,
                loss_C_mult=self.loss_C_mult,
                loss_mu_mult=self.loss_mu_mult,
                loss_rec_mult=self.loss_rec_mult,
                loss_decor_mult=self.loss_decor_mult,
                loss_pred_mult=self.loss_pred_mult,
                alpha=self.alpha,
                LR_C=self.LR_C,
                lr=self.lr,
                weight_decay=self.weight_decay,
                )

    def set_params(self, **params):
        dft = DemoVAE.get_default_params()
        for key in dft:
            if key in params:
                setattr(self, key, params[key])
            else:
                setattr(self, key, dft[key])
        return self

    def fit(self, x, demo, demo_types, **kwargs):
        # Get demo_dim
        demo_dim = 0
        for d,t in zip(demo, demo_types):
            if t == 'continuous':
                demo_dim += 1
            elif t == 'categorical':
                ll = len(set(list(d)))
                if ll == 1:
                    print('Only one type of category for categorical variable')
                    raise Exception('Bad categorical')
                demo_dim += ll
            else:
                print(f'demographic type "{t}" not "continuous" or "categorical"')
                raise Exception('Bad demographic type')
        # Save parameters
        self.input_dim = x.shape[1]
        self.demo_dim = demo_dim
        # Create model
        self.vae = VAE(x.shape[1], self.latent_dim, demo_dim, self.use_cuda)
        # Train model
        train_vae(self.vae, x, demo, demo_types, 
                self.nepochs, self.pperiod, self.bsize, 
                self.loss_C_mult, self.loss_mu_mult, self.loss_rec_mult, self.loss_decor_mult, self.loss_pred_mult,
                self.lr, self.weight_decay, self.alpha, self.LR_C, 
                self)
        return self

    def transform(self, x, demo, demo_types, **kwargs):
        if isinstance(x, int):
            # Generate
            z = self.vae.gen(x)
        else:
            # Get latents for real data
            z = self.vae.enc(to_cuda(to_torch(x), self.vae.use_cuda))
        demo_t = demo_to_torch(demo, demo_types, self.pred_stats, self.vae.use_cuda)
        y = self.vae.dec(z, demo_t)
        return to_numpy(y)

    def fit_transform(self, x, demo, demo_types, **kwargs):
        self.fit(x, demo, demo_types)
        return self.transform(x, demo, demo_types)

    def get_latents(self, x):
        z = self.vae.enc(to_cuda(to_torch(x), self.vae.use_cuda))
        return to_numpy(z)

    def save(self, path):
        params = self.get_params()
        dct = dict(pred_stats=self.pred_stats, 
                   params=params, 
                   input_dim=self.input_dim,
                   demo_dim=self.demo_dim,
                   model_state_dict=self.vae.state_dict())
        torch.save(dct, path)

    def load(self, path):
        dct = torch.load(path)
        self.pred_stats = dct['pred_stats']
        self.set_params(**dct['params'])
        self.vae = VAE(dct['input_dim'], 
                       dct['params']['latent_dim'], 
                       dct['demo_dim'], 
                       dct['params']['use_cuda'])
        self.vae.load_state_dict(dct['model_state_dict'])


