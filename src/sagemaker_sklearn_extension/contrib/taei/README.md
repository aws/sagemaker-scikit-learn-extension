# Tabular AutoEncoder Interpolator

## Overview
<img src="images/overview.png" alt="overview" height="200">

This library contains implementations of the latent space minority oversampling techniques proposed in [1] for 
multi-modal data. These oversamplers work by

1. Mapping the multi-modal samples to a dense continuous latent space using an autoencoder
2. Applying oversampling by interpolation in the latent space
3. Mapping the synthetic samples back to the original feature space

This framework was shown to be effective in generating high-quality multi-modal synthetic data which then resulted in 
better prediction quality for downstream tasks.

#### LatentSpaceOversampler
The interpolator is implemented by `LatentSpaceOversampler` which takes two inputs at initialization:
- `model` - The autoencoder used to map the samples to the latent space and back. Currently, two 
autoencoders are provided with the package: `AE` which is a vanilla autoencoder and `VAE` which is a variational 
autoencoder.
- `base_oversampler` function - The oversampling function applied in the latent space. We have experimented with
`SMOTE` from [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) and `StarOversampler` which is
our light weight implementation (provided with this package) of `polynom_fit_SMOTE`[2] based on the implementation of
[smote_variants](https://github.com/analyticalmindsltd/smote_variants)[3]

## Installation
It is recommended to install from PyPI
```
pip install sagemaker-scikit-learn-extension[taei]

# For Zsh users: 
pip install sagemaker-scikit-learn-extension\[taei]\
```

## Examples
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) is required to run the examples below as 
it provides the dataset and the base oversampler. Install
[imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) by
```
pip install imbalanced-learn==0.7
```

TAEI supports input of either a numpy.ndarray or a pandas.DataFrame object with two types of columns:
- Continuous columns: numeric values, can have very large cardinality
- Discrete (categorical) columns: numeric values with low cardinality. These columns need be encoded to ordinal integers
before using TAEI. This could be easily done using `sagemaker_sklearn_extension.preprocessing.OrdinalEncoder`

Next we load the dataset from [imbalanced-learn](https://github.com/scikit-learn-contrib/imbalanced-learn) and specify 
which columns are continuous and which are discrete
```python
import imblearn.datasets

# load the datasets
d = imblearn.datasets.fetch_datasets()["abalone"]
# indexes of categorical features
categorical_features = [0, 1, 2]
# number of uniques for each categorical feature
categorical_dims = [2, 2, 2]
# indexes of continuous features
continuous_features = [3, 4, 5, 6, 7, 8, 9]
```

### Vanilla autoencoder + SMOTE
We start with an example of wrapping SMOTE with a vanilla autoencoder
```python
from imblearn.over_sampling import SMOTE
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, AE

ae_smote = LatentSpaceOversampler(
    model=AE(
        categorical_features=categorical_features,
        categorical_dims=categorical_dims,
        continuous_features=continuous_features,
    ),
    base_oversampler=SMOTE(sampling_strategy=0.5).fit_resample,
)
```
We train the autoencoder on the training data before using the oversampler
```python
ae_smote.fit(X=d["data"], y=d["target"], verbose=True)
```

Finally, we can oversample the minority class
```python
# Oversample the minority class
X_oversampled, y_oversampled = ae_smote.resample(X=d["data"], y=d["target"], verbose=True)
```
Note that the base oversampler, SMOTE in our case, controls the number of minority samples generated

### Variational autoencoder + StarOversampler
We demonstrate PolynomFit using the "star" topology [2] wrapped by a variational autoencoder, a combination yielding
superior prediction quality in our experiments[1]. For PolynomFit, we use our light weight implementation,
`StarOversampler`, based on the implementation of
[smote_variants](https://github.com/analyticalmindsltd/smote_variants)[3]
```python
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, VAE, StarOversampler

vae_poly = LatentSpaceOversampler(
    model=VAE(
        categorical_features=categorical_features,
        categorical_dims=categorical_dims,
        continuous_features=continuous_features,
    ),
    base_oversampler=StarOversampler(proportion=1.0).resample
)
# Train the model and oversample in a single function call
X_oversampled, y_oversampled = vae_poly.fit_resample(X=d['data'], y=d['target'], verbose=True)
```

### Save and load trained models
First, store the model we trained in `vae_poly` to a file. Note that `base_oversampler` is not stored, only the trained
model
```python
vae_poly.save_model('/tmp/vae_model.pth')
```
We use the stored model by creating a new `LatentSpaceOversampler` and loading the trained model into it
```python
vae_poly_loaded = LatentSpaceOversampler(
    model=None,
    base_oversampler=StarOversampler(proportion=1.0).resample
)
vae_poly_loaded.load_model('/tmp/vae_model.pth')
# Oversample the minority class using the stored model
X_os, y_os = vae_poly_loaded.resample(d['data'], d['target'], verbose=True)
```


## Citing TAEI

If you use TAEI, please cite the following work:
- S. Darabi and Y. Elor "Synthesising Multi-Modal Minority Samples for Tabular Data"

## References
[1] S. Darabi and Y. Elor "Synthesising Multi-Modal Minority Samples for Tabular Data"

[2] Gazzah, S. and Amara, N. E. B., "New Oversampling Approaches Based on Polynomial Fitting for Imbalanced Data Sets", 
2008 The Eighth IAPR International Workshop on Document Analysis Systems, 2008, pp. 677-684

[3] Gy\"orgy Kov\'acs. "smote-variants: a Python Implementation of 85 Minority Oversampling Techniques", Neurocomputing
366, 2019