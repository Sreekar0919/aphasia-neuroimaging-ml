# Model configuration
# The covariance loss compares z.T @ z against I * batch_size, so it can only be
# satisfied when batch_size > latent_dim (rank of z.T @ z <= batch_size). With a
# cohort of tens of subjects, keep latent_dim <= bsize / 2 and train near full-batch.
MODEL_CONFIG = {
    'latent_dim': 16,
    'nepochs': 1000,
    'bsize': 64,            # clamped to the number of subjects at fit time
    'loss_rec_mult': 100,
    'loss_decor_mult': 10,
    'lr': 1e-4,
    'weight_decay': 1e-4,   # L2 on the 35M-parameter encoder/decoder; untuned, sweep it
    'alpha': 100,           # Ridge strength for continuous guidance models
    'LR_C': 1,              # inverse L2 for categorical guidance (reference 100 ~ unregularised)
}

# Preprocessing configuration
PREPROCESS_CONFIG = {
    't_r': 2.0,
    'high_pass': 0.01,
    'low_pass': 0.1,
    'radius': 5
}

# Dataset configuration
DATASET_CONFIG = {
    'name': 'SreekarB/OSFData',
    'split': 'train'
}

# Prediction configuration
PREDICTION_CONFIG = {
    'n_estimators': 100,
    'max_depth': None,
    'cv_folds': 5,
    'prediction_type': 'regression',
    'default_outcome': 'wab_aq',
    'save_path': 'results/treatment_predictor.joblib'
}
