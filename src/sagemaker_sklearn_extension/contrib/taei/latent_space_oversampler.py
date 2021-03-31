import torch
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.utils import check_X_y


class LatentSpaceOversampler:
    """
    Implementation of the latent space minority oversampling techniques proposed in [1]. The model (autoencoder) is used
    to encode the samples to the latent space where the base oversampler is applied to generate new minority samples.
    The generated synthetic minority samples are decoded back to the original feature space using the decoder.
    Interpolation parameters such as the oversampling ratio are controlled by the base oversampler.

    Parameters
    ----------
    model : (autoencoder) pytorch model
        A model to be used to encode the samples into the latent space before interpolation and from the latent space
        after interpolation
    base_oversampler : oversampler
        oversampler used to interpolate samples in the latent space
    device : 'cpu' or 'gpu' (default = 'cpu')
        Device used by pytorch for training the model and using the trained model for encoding/decoding
    random_state : int (default = 0)
        Random number generation seed

    References
    ----------
    .. [1] S. Darabi and Y. Elor "Synthesising Multi-Modal Minority Samples for Tabular Data"

    """

    def __init__(self, model, base_oversampler, device="cpu", random_state=0):
        self.model = model
        self.base_oversampler = base_oversampler
        self.device = device
        self.random_state = random_state

    def fit(self, X, y, validation_ratio=0.2, **kwargs):
        """
        Train the model using gradient descent back propagation

        Parameters
        ----------
        X : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features matrix used to train the model
        y : vector-like of shape (n_samples, 1)
            The target vector used to train the model
        validation_ratio : float or None (default = 0.2)
            Ratio of samples to be used as validation set for early stopping in model training. If None then early
            stopping is not applied
        **kwargs:
            Additional arguments passed the the model internal fit function
        """
        X, y = check_X_y(X, y)
        if validation_ratio:
            X_train, X_validation, y_train, y_validation = train_test_split(
                X, y, test_size=validation_ratio, stratify=y, random_state=self.random_state
            )
        else:
            X_train = X
            y_train = y
            X_validation = None
            y_validation = None
        self.model.fit(
            X_train=X_train,
            y_train=y_train,
            X_validation=X_validation,
            y_validation=y_validation,
            device=self.device,
            **kwargs,
        )
        return self

    def resample(self, X, y, verbose=False):
        """
        Use the model and the base oversampler to generate synthetic minority samples
        """
        X, y = check_X_y(X, y)
        self.model.eval()
        X = torch.Tensor(X)
        X = X.to(self.device)
        with torch.no_grad():
            z = self.model.encode(X)
        z = z.cpu().numpy()
        if verbose:
            print(f"LatentSpaceOversampler: Shape before oversampling z:{z.shape}, y:{y.shape}")
        z_samples, y_samples = self.base_oversampler(z, y)
        if verbose:
            print(f"LatentSpaceOversampler: Shape after oversampling z:{z_samples.shape}, y:{y_samples.shape}")
        z_samples = z_samples[-(len(z_samples) - len(X)) :]
        y_samples = y_samples[-(len(y_samples) - len(y)) :].reshape(-1)
        z_samples = torch.Tensor(z_samples).to(self.device)
        with torch.no_grad():
            x_samples = self.model.decode_sample(z_samples)
        X = torch.cat([X, x_samples], dim=0).cpu().numpy()
        y = np.concatenate((y, y_samples), axis=0)
        return X, y

    def fit_resample(self, X, y, verbose=False, **kwargs):
        return self.fit(X, y, verbose=verbose, **kwargs).resample(X, y, verbose=verbose)

    def save_model(self, path):
        torch.save(self.model, path)

    def load_model(self, path):
        self.model = torch.load(path)
