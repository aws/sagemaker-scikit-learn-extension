# TAEI

TAEI contains implementations of the latent space minority oversampling techniques proposed in [1]

## Installation

```
# install from pip
pip install sagemaker-scikit-learn-extension[taei]

# For Zsh users: 
pip install sagemaker-scikit-learn-extension\[taei]\
```

## Examples
imbalanced-learn is required to run the examples as it provides the dataset and the base oversampler, install it by
```
pip install imbalanced-learn==0.7
```
Load the dataset from imblearn and specify which columns are continuous (numeric) and which are discrete (categorical).
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
We start with and example of wrapping SMOTE with a vanilla autoencoder
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
Note that the base oversampler, SMOTE in our case, controls the number of minority samples generated

We train the autoencoder on the training data before using the oversampler 
```python
ae_smote.fit(d["data"], d["target"], verbose=True)
```

Finally, we can oversample the minority class 
```python
# Oversample the minority class
X_os, y_os = ae_smote.resample(d["data"], d["target"], verbose=True)
```

### Variational autoencoder + PolynomFit
TODO
We start with and example of wrapping SMOTE with a vanilla autoencoder
```python
from imblearn.over_sampling import SMOTE
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, VAE

ae_smote = LatentSpaceOversampler(
    model=VAE(
        categorical_features=categorical_features,
        categorical_dims=categorical_dims,
        continuous_features=continuous_features,
    ),
    base_oversampler=SMOTE(sampling_strategy=0.5).fit_resample,
)
```

## References
[1] S. Darabi and Y. Elor "Synthesising Multi-Modal Minority Samples for Tabular Data"