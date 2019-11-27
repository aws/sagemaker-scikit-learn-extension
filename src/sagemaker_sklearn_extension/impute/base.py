# Copyright 2019 Amazon.com, Inc. or its affiliates. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License"). You
# may not use this file except in compliance with the License. A copy of
# the License is located at
#
#      http://aws.amazon.com/apache2.0/
#
# or in the "license" file accompanying this file. This file is
# distributed on an "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF
# ANY KIND, either express or implied. See the License for the specific
# language governing permissions and limitations under the License.

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.impute import MissingIndicator, SimpleImputer
from sklearn.utils.validation import check_array, check_is_fitted


def is_finite_numeric(arr):
    """Helper function to check if values in an array can be converted to finite numeric
    """

    def _is_finite_numeric(val):
        try:
            f = float(val)
            return np.isfinite(f)
        except ValueError:
            return False

    return np.vectorize(_is_finite_numeric)(arr)


def _get_mask(X, vectorized_mask_function):
    """Compute boolean mask of X for vectorized_mask_function(X) == False
    """
    return np.logical_not(vectorized_mask_function(X).astype("bool"))


def _apply_mask(X, mask):
    X[mask] = np.nan
    return X


class RobustImputer(BaseEstimator, TransformerMixin):
    """Imputer for completing missing values.

    Similar to sklearn.impute.SimpleImputer with added functionality
    - RobustImputer uses a custom mask_function to determine values to impute.
      The default mask_function is sagemaker_sklearn_extension.impute.is_finite_numeric
      which checks if a value can be converted into a float.
    - RobustImputer can perform multi-column imputation with different values
      for each column (strategy=="constant")

    Parameters
    ----------
    dtype : string, type, list of types or None (default=None)
        Data type for output.

        - If left to default, numeric imputation strategies ("median" and "mean"),
        output array dtype will always be floating point dtype. Otherwise it will be
        np.dtype('O')

    strategy : string, optional (default='median')
        The imputation strategy.

        - If "mean", then replace missing values using the mean along
          each column. Can only be used with numeric data.
        - If "median", then replace missing values using the median along
          each column. Can only be used with numeric data.
        - If "most_frequent", then replace missing using the most frequent
          value along each column. Can be used with strings or numeric data.
        - If "constant", then replace missing values with fill_values.
          fill_values can be a singular value or a list of values equal to
          number of columns. Can be used with strings or numeric data.
          If fill_values is not set, fill_value will be 0 when imputing numerical
          data and "missing_value" for strings or object data types.

    fill_values : string, numerical value, or list, optional (default=None)
        When strategy=="constant", fill_values is used to replace all
        values that should be imputed.

        - If string or numerical value, that one value will be used to replace
          all values that should be imputed.
        - If list, fill_values must equal to number of columns of input. Each
          column will be imputed with the corresponding value in fill_values.
          fill_values[i] will replace ith column (X[:,i]).
        - If left to the default, fill_value will be 0 when imputing numerical
          data and "missing_value" for strings or object data types.

    mask_function : callable -> np.array, dtype('bool') (default=None)
        A vectorized python function, accepts np.array, returns np.array
        with dtype('bool')

        For each value, if mask_function(val) == False, that value will
        be imputed. mask_function is used to create a boolean mask that determines
        which values in the input to impute.

        Use np.vectorize to vectorize singular python functions.

        If left to default, mask_function will be
        sagemaker_sklearn_extension.impute.is_finite_numeric

    Notes
    -----
    only accepts 2D, non-sparse inputs
    """

    def __init__(self, dtype=None, strategy="median", fill_values=None, mask_function=None):
        self.dtype = dtype
        self.strategy = strategy
        self.fill_values = fill_values
        self.mask_function = mask_function

    def _validate_input(self, X):
        if self._is_constant_multicolumn_imputation():
            if len(self.fill_values) != X.shape[1]:
                raise ValueError(
                    "'fill_values' should have length equal to number of features in X {num_features}, "
                    "got {fill_values_length}".format(num_features=X.shape[1], fill_values_length=len(self.fill_values))
                )

        dtype = self.dtype or np.dtype("O")

        if hasattr(X, "dtype") and X.dtype is not None and hasattr(X.dtype, "kind") and X.dtype.kind == "c":
            raise ValueError("Complex data not supported\n{}\n".format(X))

        return check_array(X, dtype=dtype, copy=True, force_all_finite=False, ensure_2d=True)

    def _is_constant_multicolumn_imputation(self):
        return self.strategy == "constant" and isinstance(self.fill_values, (list, tuple, np.ndarray))

    def fit(self, X, y=None):
        """Fit the imputer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : RobustImputer
        """
        X = self._validate_input(X)

        self.vectorized_mask_function_ = self.mask_function or is_finite_numeric
        X = _apply_mask(X, _get_mask(X, self.vectorized_mask_function_))

        if self._is_constant_multicolumn_imputation():
            self.simple_imputer_ = SimpleImputer(strategy=self.strategy)
        else:
            self.simple_imputer_ = SimpleImputer(strategy=self.strategy, fill_value=self.fill_values)

        self.simple_imputer_.fit(X)

        # set "SimpleImputer.statistics_" for multicolumn imputations with different column fill values
        # SimpleImputer cannot preform multicolumn imputation with different column fill values
        if self._is_constant_multicolumn_imputation():
            self.simple_imputer_.statistics_ = np.asarray(self.fill_values)

        return self

    def transform(self, X):
        """Impute all missing values in X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray}, shape (n_samples, n_features)
            The imputed input data. The data type of ``Xt``
            will depend on your input dtype.
        """
        check_is_fitted(self, ["simple_imputer_", "vectorized_mask_function_"])
        X = self._validate_input(X)

        if X.shape[1] != self.simple_imputer_.statistics_.shape[0]:
            raise ValueError(
                "'transform' input X has {transform_dim} features per sample, "
                "expected {fit_dim} from 'fit' input".format(
                    transform_dim=X.shape[1], fit_dim=self.simple_imputer_.statistics_.shape[0]
                )
            )

        X = _apply_mask(X, _get_mask(X, self.vectorized_mask_function_))

        return self.simple_imputer_.transform(X).astype(self.dtype)

    def _more_tags(self):
        return {"allow_nan": True}


class RobustMissingIndicator(BaseEstimator, TransformerMixin):
    """Binary indicators for missing values.

    Note that this component typically should not be used in a vanilla
    :class:`sklearn.pipeline.Pipeline` consisting of transformers and a classifier,
    but rather could be added using a :class:`sklearn.pipeline.FeatureUnion` or
    :class:`sklearn.compose.ColumnTransformer`.

    Similar to sklearn.impute.MissingIndicator with added functionality
    - RobustMissingIndicator uses a custom mask_function to determine the boolean mask.
      The default mask_function is sagemaker_sklearn_extension.impute.is_finite_numeric
      which checks whether or not a value can be converted into a float.

    Parameters
    ----------
    features : str, optional (default="all")
        Whether the imputer mask should represent all or a subset of
        features.

        - If "missing-only", the imputer mask will only represent
          features containing missing values during fit time.
        - If "all" (default), the imputer mask will represent all features.

    error_on_new : boolean, optional (default=True)
        If True (default), transform will raise an error when there are
        features with missing values in transform that have no missing values
        in fit. This is applicable only when ``features="missing-only"``.

    mask_function : callable -> np.array, dtype('bool') (default=None)
        A vectorized python function, accepts np.array, returns np.array
        with dtype('bool')

        For each value, if mask_function(val) == False, that value will
        be imputed. mask_function is used to create a boolean mask that determines
        which values in the input to impute.

        Use np.vectorize to vectorize singular python functions.

        By default, mask_function will be
        sagemaker_sklearn_extension.impute.is_finite_numeric

    Notes
    -----
    only accepts 2D, non-sparse inputs
    """

    def __init__(self, features="all", error_on_new=True, mask_function=None):
        self.features = features
        self.error_on_new = error_on_new
        self.mask_function = mask_function

    def _validate_input(self, X):
        if hasattr(X, "dtype") and X.dtype is not None and hasattr(X.dtype, "kind") and X.dtype.kind == "c":
            raise ValueError("Complex data not supported\n{}\n".format(X))

        return check_array(X, dtype=np.dtype("O"), copy=True, force_all_finite=False, ensure_2d=True)

    def fit(self, X, y=None):
        """Fit the transformer on X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            Input data, where ``n_samples`` is the number of samples and
            ``n_features`` is the number of features.

        Returns
        -------
        self : RobustMissingIndicator
        """
        X = self._validate_input(X)

        self.vectorized_mask_function_ = self.mask_function or is_finite_numeric
        X = _apply_mask(X, _get_mask(X, self.vectorized_mask_function_))

        self.missing_indicator_ = MissingIndicator(features=self.features, error_on_new=self.error_on_new)
        self.missing_indicator_.fit(X)

        return self

    def transform(self, X):
        """Generate missing values indicator for X.

        Parameters
        ----------
        X : {array-like}, shape (n_samples, n_features)
            The input data to complete.

        Returns
        -------
        Xt : {ndarray}, shape (n_samples, n_features)
            The missing indicator for input data. The data type of ``Xt``
            will be boolean.
        """
        check_is_fitted(self, ["missing_indicator_", "vectorized_mask_function_"])
        X = self._validate_input(X)

        X = _apply_mask(X, _get_mask(X, self.vectorized_mask_function_))

        return self.missing_indicator_.transform(X)

    def _more_tags(self):
        return {"allow_nan": True}
