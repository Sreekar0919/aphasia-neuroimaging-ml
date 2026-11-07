import numpy as np

from programmed import demographic_map, pooled_basis, program
from staged import new_model
from test_staged import cohort


def test_programmed_network_is_exactly_pca_encode_decode_on_unseen_data():
    x, age, sex, mpo, _ = cohort(60, 1)
    x_new, age_n, sex_n, mpo_n, _ = cohort(20, 2)
    pool = np.random.default_rng(3).standard_normal((200, x.shape[1])).astype(np.float32)
    V, s = pooled_basis([pool, x], 40)
    model = program(new_model(x.shape[1], 40), x, V, s)
    mu = x.mean(0)
    expected = mu + ((x_new - mu) @ V.T) @ V
    assert np.allclose(model.get_latents(x_new), ((x_new - mu) @ V.T) / s, atol=1e-3)
    assert np.allclose(model.transform(x_new, [age_n, sex_n, mpo_n]), expected, atol=1e-3)


def test_conditioned_network_adds_the_demographic_map_and_its_latents_exclude_it():
    x, age, sex, mpo, wab = cohort(80, 1)  # WAB shifts FC along one axis in this cohort
    x_new, age_n, sex_n, mpo_n, wab_n = cohort(20, 2)
    model = new_model(x.shape[1], 30)
    demo = [age, sex, mpo, wab]
    B = demographic_map([(x, model._demo(demo).numpy())])
    V, s = pooled_basis([x], 30, B)
    program(model, x, V, s, demo, B)
    assert np.abs(V @ B).max() < 1e-4  # latent basis is orthogonal to every demographic direction
    d_new = model._demo([age_n, sex_n, mpo_n, wab_n]).numpy()
    mu, d_mean = x.mean(0), model._demo(demo).numpy().mean(0)
    expected = mu + (d_new - d_mean) @ B.T + ((x_new - mu) @ V.T) @ V
    assert np.allclose(model.transform(x_new, [age_n, sex_n, mpo_n, wab_n]), expected, atol=1e-3)
    lo = model.transform(x_new, [age_n, sex_n, mpo_n, np.full(20, 20.0)])
    hi = model.transform(x_new, [age_n, sex_n, mpo_n, np.full(20, 90.0)])
    assert np.allclose(hi - lo, (90 - 20) / model.wab_stats[1] * B[:, 3], atol=1e-3)
