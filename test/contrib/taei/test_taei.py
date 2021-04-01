import numpy as np
import torch
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, AE, VAE, StarOversampler


def test_latent_space_oversampler():
    # make torch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    d = np.genfromtxt("test/contrib/taei/data/data.csv", delimiter=",")
    categorical_features = [0, 1, 2]
    categorical_dims = [2, 2, 2]
    continuous_features = [3, 4, 5, 6, 7, 8, 9]

    star_fit_resample = StarOversampler(proportion=1.0).resample

    # Test AE+StarOversampler
    torch.manual_seed(0)
    ae_smote = LatentSpaceOversampler(
        model=AE(
            categorical_features=categorical_features,
            categorical_dims=categorical_dims,
            continuous_features=continuous_features,
            latent_dim=8,
            hidden_dim=[64, 32],
            nll_weight=0.5,
        ),
        base_oversampler=star_fit_resample,
    )
    # Train the model
    ae_smote.fit(d[:, :10], d[:, 10], max_epoch=5, validation_ratio=None)
    # Use the model for oversampling
    X_os, y_os = ae_smote.resample(d[:, :10], d[:, 10])
    np.testing.assert_almost_equal(
        X_os[-1, :],
        [1.0, 1.0, 0.0, 0.5661017, 0.7811485, 1.172961, 1.0983223, 1.5463793, 1.3487656, 0.605184],
        decimal=2,
    )

    # Test VAE+StarOversampler
    torch.manual_seed(0)
    vae_smote = LatentSpaceOversampler(
        model=VAE(
            categorical_features=categorical_features,
            categorical_dims=categorical_dims,
            continuous_features=continuous_features,
            latent_dim=16,
            hidden_dim=32,
            nll_weight=0.1,
            kld_weight=0.5,
        ),
        base_oversampler=star_fit_resample,
    )
    # Train and use the model in one function call
    X_os, y_os = vae_smote.fit_resample(d[:, :10], d[:, 10], max_epoch=50, early_stopping=1)
    np.testing.assert_almost_equal(
        X_os[-1, :],
        [0.0, 1.0, 1.0, 0.5926914, 0.4106686, 0.3133996, 0.0246359, 0.4813618, -0.1365427, -0.0096727],
        decimal=2,
    )

    # Test storing and loading models
    vae_smote.save_model("/tmp/vae_model.pth")
    vae_smote_loaded = LatentSpaceOversampler(model=None, base_oversampler=star_fit_resample)
    vae_smote_loaded.load_model("/tmp/vae_model.pth")
    X_os_loaded, y_os_loaded = vae_smote_loaded.resample(d[:, :10], d[:, 10])
    np.testing.assert_almost_equal(X_os, X_os_loaded)
