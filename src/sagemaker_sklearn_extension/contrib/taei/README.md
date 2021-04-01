# TAEI

TAEI contains implementations of the latent space minority oversampling techniques proposed in [1]

## Installation
Install from pip
```
pip install sagemaker-scikit-learn-extension[taei]

# For Zsh users: 
pip install sagemaker-scikit-learn-extension\[taei]\
```

## Examples
imbalanced-learn is required to run the examples as it provides the dataset and the base oversampler. smote-variants is
 used for polynom_fit_SMOTE which yield superior prediction quality in our experiments. Install these packages by
```
pip install imbalanced-learn==0.7 smote-variants
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
Note that the base oversampler, SMOTE in our case, controls the number of minority samples generated

We train the autoencoder on the training data before using the oversampler 
```python
ae_smote.fit(X=d["data"], y=d["target"], verbose=True)
```

Finally, we can oversample the minority class 
```python
# Oversample the minority class
X_os, y_os = ae_smote.resample(X=d["data"], y=d["target"], verbose=True)
```

### Variational autoencoder + PolynomFit
We demonstrate PolynomFit[2] wrapped by a variational autoencoder, a combination yielding yield superior prediction 
quality on our experiments[1]
```python
from smote_variants import polynom_fit_SMOTE
from sagemaker_sklearn_extension.contrib.taei import LatentSpaceOversampler, VAE

vae_poly = LatentSpaceOversampler(
    model=VAE(
        categorical_features=categorical_features,
        categorical_dims=categorical_dims,
        continuous_features=continuous_features,
    ),
    base_oversampler=polynom_fit_SMOTE(proportion=1.0).sample
)
# Train the model and oversample in a single function call
X_os, y_os = vae_poly.fit_resample(X=d['data'], y=d['target'], verbose=True)
```

## References
[1] S. Darabi and Y. Elor "Synthesising Multi-Modal Minority Samples for Tabular Data"

[2] Gazzah, S. and Amara, N. E. B., "New Oversampling Approaches Based on Polynomial Fitting for Imbalanced Data Sets", 
2008 The Eighth IAPR International Workshop on Document Analysis Systems, 2008, pp. 677-684