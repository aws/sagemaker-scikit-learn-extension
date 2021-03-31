import numpy as np
import torch
from imblearn.over_sampling import SMOTE
from sagemaker_sklearn_extension.contrib.taei.src.latent_space_oversampler import LatentSpaceOversampler
from sagemaker_sklearn_extension.contrib.taei.src.models import AE, VAE


def test_latent_space_oversampler():
    # make torch deterministic
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    d = np.genfromtxt("test/data/taei/data.csv", delimiter=",")
    categorical_features = [0, 1, 2]
    categorical_dims = [2, 2, 2]
    continuous_features = [3, 4, 5, 6, 7, 8, 9]

    smote_fit_resample = SMOTE(sampling_strategy=0.5, random_state=1).fit_resample

    # Test AE+SMOTE
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
        base_oversampler=smote_fit_resample,
    )
    # Train the model
    ae_smote.fit(d[:, :10], d[:, 10], max_epoch=5, validation_ratio=None)
    # Use the model for oversampling
    X_os, y_os = ae_smote.resample(d[:, :10], d[:, 10])
    np.testing.assert_almost_equal(
        X_os[-1, :], [1.0, 1.0, 0.0, 0.7807106, 1.5386977, 1.3619816, 1.5680989, 2.1116064, 1.6013482, 0.9756886]
    )

    # Test VAE+SMOTE
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
        base_oversampler=smote_fit_resample,
    )
    # Train and use the model in one function call
    X_os, y_os = vae_smote.fit_resample(d[:, :10], d[:, 10], max_epoch=50, early_stopping=1)
    np.testing.assert_almost_equal(
        X_os[-1, :], [0.0, 1.0, 1.0, 0.8581181, 0.7264875, 0.2218277, 0.1161354, 0.9728333, -0.1757372, -0.2257324]
    )

    # Test storing and loading models
    vae_smote.save_model("/tmp/vae_model.pth")
    vae_smote_loaded = LatentSpaceOversampler(model=None, base_oversampler=smote_fit_resample)
    vae_smote_loaded.load_model("/tmp/vae_model.pth")
    X_os_loaded, y_os_loaded = vae_smote_loaded.resample(d[:, :10], d[:, 10])
    np.testing.assert_almost_equal(X_os, X_os_loaded)
