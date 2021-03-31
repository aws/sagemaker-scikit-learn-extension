import numpy as np
import torch
import imblearn.datasets
from imblearn.over_sampling import SMOTE
from .smote_variants import polynom_fit_SMOTE
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, AE, VAE


def main():
    # make torch deterministic
    torch.manual_seed(0)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # load the datasets
    d = imblearn.datasets.fetch_datasets()["abalone"]
    # indexes of categorical features
    categorical_features = [0, 1, 2]
    # number of uniques for each categorical feature
    categorical_dims = [2, 2, 2]
    # indexes of continuous features
    continuous_features = [3, 4, 5, 6, 7, 8, 9]

    # Example of AE+SMOTE
    ae_smote = LatentSpaceOversampler(
        model=AE(
            categorical_features=categorical_features,
            categorical_dims=categorical_dims,
            continuous_features=continuous_features,
        ),
        base_oversampler=SMOTE(sampling_strategy=0.5).fit_resample,
    )
    # Train the model
    ae_smote.fit(d["data"], d["target"], verbose=True)
    # Oversample the minority class
    X_os, y_os = ae_smote.resample(d["data"], d["target"], verbose=True)
    print(
        f"Before oversampling:\n"
        f'X.shape={d["data"].shape}, y.shape={d["target"].shape}\n'
        f'y count majority / minority = {np.sum(d["target"] == -1)} / {np.sum(d["target"] == 1)}'
    )
    print(
        f"Oversampling using AE+SMOTE:\n"
        f"X-oversampled.shape={X_os.shape}, y-oversampled.shape={y_os.shape}\n"
        f"y count majority / minority = {np.sum(y_os == -1)} / {np.sum(y_os == 1)}"
    )

    # Example of VAE+Poly
    vae_poly = LatentSpaceOversampler(
        model=VAE(
            categorical_features=categorical_features,
            categorical_dims=categorical_dims,
            continuous_features=continuous_features,
        ),
        base_oversampler=polynom_fit_SMOTE(proportion=1.0).sample,
    )
    # Train the model and oversample in a single function call
    X_os, y_os = vae_poly.fit_resample(d["data"], d["target"], verbose=True)
    print(
        f"Oversampling using VAE+Poly:\n"
        f"X-oversampled.shape={X_os.shape}, y-oversampled.shape={y_os.shape}\n"
        f"y count majority / minority = {np.sum(y_os == -1)} / {np.sum(y_os == 1)}"
    )

    # Store the trained model to a file
    vae_poly.save_model("/tmp/vae_model.pth")

    # Use a stored model
    vae_poly_loaded = LatentSpaceOversampler(model=None, base_oversampler=polynom_fit_SMOTE(proportion=1.0).sample)
    vae_poly_loaded.load_model("/tmp/vae_model.pth")
    X_os, y_os = vae_poly_loaded.resample(d["data"], d["target"], verbose=True)
    print(
        f"\n"
        f"Oversampling using VAE+Poly from stored model:\n"
        f"X-oversampled.shape={X_os.shape}, y-oversampled.shape={y_os.shape}\n"
        f"y count majority / minority = {np.sum(y_os == -1)} / {np.sum(y_os == 1)}"
    )


if __name__ == "__main__":
    main()
