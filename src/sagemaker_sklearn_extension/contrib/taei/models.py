"""Base class and autoencoders used for latent space oversampling"""

from abc import abstractmethod
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import TensorDataset
from torch.optim.lr_scheduler import MultiplicativeLR
from .nn_utils import GBN, LambdaLogSoftmax, weight_init, EmbeddingGenerator


class BaseModel(nn.Module):
    """
    Base class for all models
    """

    @abstractmethod
    def forward(self, *inputs):
        """
        Forward pass logic

        :return: Model output
        """
        raise NotImplementedError

    @abstractmethod
    def loss(self, output, target):
        raise NotImplementedError

    def decode_sample(self, z):
        x_hat = self.decode(z)
        x_cont, x_cat = [], []
        if hasattr(self, "cont_net"):
            x_cont = x_hat.pop(0)
        if hasattr(self, "cat_nets"):
            for _ in self.categorical_features:
                x_cat.append(torch.argmax(x_hat.pop(0), dim=1))
        x = []
        cont_c, cat_c = 0, 0
        for i in range(self.input_dim):
            if i in self.continuous_features:
                x.append(x_cont[:, cont_c].reshape(-1, 1))
                cont_c += 1
            elif i in self.categorical_features:
                x.append(x_cat[cat_c].reshape(-1, 1))
                cat_c += 1
        x = torch.cat(x, dim=1)
        return x

    def __str__(self):
        """
        Model prints with number of trainable parameters
        """
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + "\nTrainable parameters: {}".format(params)

    def fit(
        self,
        X_train,
        y_train,
        X_validation=None,
        y_validation=None,
        loss_key="opt",
        batch_size=128,
        num_workers=0,
        learning_rate=1e-3,
        learning_rate_lambda=0.995,
        max_epoch=10000,
        early_stopping=100,
        device="cpu",
        verbose=False,
    ):
        """
        Train the model using gradient descent back propagation

        Parameters
        ----------
        X_train : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features matrix used to train the model
        y_train : vector-like of shape (n_samples, 1)
            The target vector used to train the model
        X_validation : {array-like, sparse matrix} of shape (n_samples, n_features)
            Features matrix used for early stopping of the training
        y_validation : vector-like of shape (n_samples, 1)
            The target vector used for early stopping of the training
        loss_key: string (default = 'opt')
            Which field of the loss dictionary to optimize
        batch_size: int (default = 128)
            Batch size
        num_workers: int (default = 0)
            Number of cpus to use
        learning_rate: float (default = 1e-3)
            Gradient descent learning rate
        learning_rate_lambda: float (default = 0.995)
            The rate of decreasing learning_rate
        max_epoch: int (default = 10000)
            The maximum number of optimization epochs
        early_stopping: int (default = 100)
            The number of epochs without improving the bast validation loss allowed before stopping
        device : 'cpu' or 'gpu' (default = 'cpu')
            Device used by pytorch for training the model and using the trained model for encoding/decoding
        verbose: True or False (default = False)
            Verbosity
        """
        assert X_train.shape[1] == self.input_dim
        self.to(device)
        train_loader = torch.utils.data.DataLoader(
            TensorDataset(torch.Tensor(X_train), torch.Tensor(y_train)),
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
        )
        if X_validation is not None:
            validation_loader = torch.utils.data.DataLoader(
                TensorDataset(torch.Tensor(X_validation), torch.Tensor(y_validation)),
                batch_size=batch_size,
                shuffle=True,
                num_workers=num_workers,
            )
        else:
            validation_loader = None

        optimizer = torch.optim.Adam(self.parameters(), lr=learning_rate)
        scheduler = MultiplicativeLR(optimizer, lr_lambda=(lambda epoch: learning_rate_lambda))
        best_validation_loss = None
        iter_no_improve = 0
        for epoch in range(max_epoch):
            self.train()
            training_loss = 0
            for data in train_loader:
                Xb = data[0].to(device)
                optimizer.zero_grad()
                output = self(Xb)
                loss = self.loss(output, Xb)[loss_key]
                loss.backward()
                optimizer.step()
                training_loss += loss.detach().cpu().numpy()
            self.eval()
            validation_loss = 0
            if validation_loader:
                with torch.no_grad():
                    for data in validation_loader:
                        Xb = data[0].to(device)
                        output = self(Xb)
                        loss = self.loss(output, Xb)[loss_key]
                        validation_loss += loss.detach().cpu().numpy()
                    if best_validation_loss is None or validation_loss < best_validation_loss:
                        best_validation_loss = validation_loss
                        iter_no_improve = 0
                    else:
                        iter_no_improve += 1
                    if iter_no_improve > early_stopping:
                        if verbose:
                            print(f"Early stopping after {epoch} epochs")
                        break
            scheduler.step()
            if verbose:
                print(f"[{epoch}] training loss={training_loss}, validation loss={validation_loss}")
        return self


class AE(BaseModel):
    """
    Vanilla autoencoder based on 10.1126/science.1127647

    Parameters
    ----------
    categorical_features : list of integers
        The indexes of the categorical features in the input array
    categorical_features : list of integers
        The cardinality of the categorical features in the input array
    continuous_features : list of integers
        The indexes of the continuous features in the input array
    latent_dim : integer (default = 32)
        The size of the latent dimension of the autoencoder
    hidden_dim : list of integers (default = [128, 128, 128])
        The hidden layer sizes of the autoencoder. The sizes are used for both the encoder and the decoder
    nll_weight : float (default = 0.3)
        Weight of the nll component in the loss
    """

    def __init__(
        self,
        categorical_features,
        categorical_dims,
        continuous_features,
        latent_dim=32,
        hidden_dim=None,
        nll_weight=0.3,
    ):
        super().__init__()
        if not hidden_dim:
            hidden_dim = [128, 128, 128]
        elif not isinstance(hidden_dim, list):
            hidden_dim = [
                hidden_dim,
            ]

        assert len(categorical_features) == len(categorical_dims)
        self.categorical_features, self.categorical_dims = [], []
        if categorical_features and categorical_dims:
            self.categorical_features, self.categorical_dims = zip(*sorted(zip(categorical_features, categorical_dims)))
        self.continuous_features = sorted(continuous_features)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nll_weight = nll_weight

        self.input_dim = len(continuous_features) + len(categorical_features)
        self.embeddings = EmbeddingGenerator(self.input_dim, categorical_dims, categorical_features)
        self.post_embed_dim = self.embeddings.post_embed_dim
        hidden_dim = [self.post_embed_dim] + hidden_dim

        # Encoder
        self.encoder = [self.embeddings]
        for i in range(1, len(hidden_dim)):
            self.encoder.extend(
                (nn.Linear(hidden_dim[i - 1], hidden_dim[i]), GBN(hidden_dim[i]), nn.PReLU(hidden_dim[i]))
            )
        self.encoder.append(nn.Linear(hidden_dim[-1], latent_dim))
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        hidden_dim = hidden_dim + [latent_dim]
        self.decoder = []
        for i in range(len(hidden_dim) - 1, 1, -1):
            self.decoder.extend(
                (nn.Linear(hidden_dim[i], hidden_dim[i - 1]), GBN(hidden_dim[i - 1]), nn.PReLU(hidden_dim[i - 1]))
            )
        self.decoder = nn.Sequential(*self.decoder)

        if self.continuous_features:
            self.cont_net = nn.Sequential(nn.Linear(hidden_dim[1], len(self.continuous_features)),)

        if self.categorical_features:
            self.cat_nets = nn.ModuleList()
            for i, n_cats in zip(self.categorical_features, self.categorical_dims):
                self.cat_nets.append(nn.Sequential(nn.Linear(hidden_dim[1], n_cats), LambdaLogSoftmax(dim=1)))

        self.apply(weight_init)

    def decode(self, z):
        """note: order of decoding is important for loss function"""
        z = self.decoder(z)
        x_hat = []
        if hasattr(self, "cont_net"):
            x_hat.append(self.cont_net(z))
        if hasattr(self, "cat_nets"):
            for m in self.cat_nets:
                x_hat.append(m(z))
        return x_hat

    def encode(self, x):
        return self.encoder(x)

    def forward(self, x):
        z = self.encode(x)
        x_hat = self.decode(z)
        return x_hat, z

    def loss(self, output, target):
        loss = {"mse": 0.0, "nll": 0.0}

        x_hat = output[0]
        if self.continuous_features:
            out = x_hat.pop(0)
            loss["mse"] = nn.functional.mse_loss(target[:, self.continuous_features], out)
        if self.categorical_features:
            for idx in self.categorical_features:
                out = x_hat.pop(0)
                loss["nll"] += nn.functional.nll_loss(out, target[:, idx].long())

        loss["opt"] = loss["mse"] + self.nll_weight * loss["nll"]
        return loss


class VAE(BaseModel):
    """
    Variational autoencoder based on the vanilla autoencoder proposed in https://arxiv.org/abs/1312.6114

    Parameters
    ----------
    categorical_features : list of integers
        The indexes of the categorical features in the input array
    categorical_features : list of integers
        The cardinality of the categorical features in the input array
    continuous_features : list of integers
        The indexes of the continuous features in the input array
    latent_dim : integer (default = 32)
        The size of the latent dimension of the autoencoder
    hidden_dim : list of integers (default = [128, 128, 128])
        The hidden layer sizes of the autoencoder. The sizes are used for both the encoder and the decoder
    nll_weight : float (default = 0.3)
        Weight of the nll component in the loss
    kld_weight : float (default = 0.1)
        Weight of the kld component in the loss
    """

    def __init__(
        self,
        categorical_features,
        categorical_dims,
        continuous_features,
        latent_dim=32,
        hidden_dim=None,
        nll_weight=0.3,
        kld_weight=0.1,
    ):
        super().__init__()
        if not hidden_dim:
            hidden_dim = [128, 128, 128]
        elif not isinstance(hidden_dim, list):
            hidden_dim = [
                hidden_dim,
            ]

        assert len(categorical_features) == len(categorical_dims)
        self.categorical_features, self.categorical_dims = [], []
        if categorical_features and categorical_dims:
            self.categorical_features, self.categorical_dims = zip(*sorted(zip(categorical_features, categorical_dims)))
        self.continuous_features = sorted(continuous_features)
        self.latent_dim = latent_dim
        self.hidden_dim = hidden_dim
        self.nll_weight = nll_weight
        self.kld_weight = kld_weight

        self.input_dim = len(continuous_features) + len(categorical_features)
        self.embeddings = EmbeddingGenerator(self.input_dim, categorical_dims, categorical_features)
        self.post_embed_dim = self.embeddings.post_embed_dim
        hidden_dim = [self.post_embed_dim] + hidden_dim

        # Encoder
        self.encoder = [self.embeddings]
        for i in range(1, len(hidden_dim)):
            self.encoder.extend(
                (nn.Linear(hidden_dim[i - 1], hidden_dim[i]), GBN(hidden_dim[i]), nn.PReLU(hidden_dim[i]))
            )
        self.encoder.append(nn.Linear(hidden_dim[-1], 2 * latent_dim))
        self.encoder = nn.Sequential(*self.encoder)

        # Decoder
        hidden_dim = hidden_dim + [latent_dim]
        self.decoder = []
        for i in range(len(hidden_dim) - 1, 1, -1):
            self.decoder.extend(
                (nn.Linear(hidden_dim[i], hidden_dim[i - 1]), GBN(hidden_dim[i - 1]), nn.PReLU(hidden_dim[i - 1]))
            )
        self.decoder = nn.Sequential(*self.decoder)

        if self.continuous_features:
            self.cont_net = nn.Sequential(nn.Linear(hidden_dim[1], len(self.continuous_features)),)

        if self.categorical_features:
            self.cat_nets = nn.ModuleList()
            for i, n_cats in zip(self.categorical_features, self.categorical_dims):
                self.cat_nets.append(nn.Sequential(nn.Linear(hidden_dim[1], n_cats), LambdaLogSoftmax(dim=1)))

        self.apply(weight_init)

    def decode(self, z):
        """note: order of decoding is important for loss function"""
        z = self.decoder(z)
        x_hat = []
        if hasattr(self, "cont_net"):
            x_hat.append(self.cont_net(z))
        if hasattr(self, "cat_nets"):
            for m in self.cat_nets:
                x_hat.append(m(z))
        return x_hat

    def encode(self, x: torch.Tensor, is_forward=False) -> torch.Tensor:
        encoded = self.encoder(x)
        mu, log_var = torch.split(encoded, encoded.shape[-1] // 2, dim=1)
        if is_forward:
            eps = torch.randn_like(mu)
            std = torch.exp(0.5 * log_var)
            z = eps * std + mu
            return z, mu, log_var
        return mu

    def forward(self, x):
        z, mu, log_var = self.encode(x, is_forward=True)
        x_hat = self.decode(z)
        return x_hat, mu, log_var

    def loss(self, output, target):
        loss = {"mse": 0.0, "nll": 0.0, "kld": 0.0}

        mu = output[1]
        log_var = output[2]
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - mu ** 2 - log_var.exp(), dim=1), dim=0)
        loss["kld"] = kld_loss

        x_hat = output[0]
        if self.continuous_features:
            out = x_hat.pop(0)
            loss["mse"] = nn.functional.mse_loss(target[:, self.continuous_features], out)
        if self.categorical_features:
            for idx in self.categorical_features:
                out = x_hat.pop(0)
                loss["nll"] += nn.functional.nll_loss(out, target[:, idx].long())

        loss["opt"] = loss["mse"] + self.kld_weight * loss["kld"] + self.nll_weight * loss["nll"]
        return loss
