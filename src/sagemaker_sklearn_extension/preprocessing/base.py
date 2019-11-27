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
from sklearn.utils.validation import check_array, check_is_fitted
from sklearn.preprocessing import QuantileTransformer, quantile_transform


def log_transform(x):
    """Apply a log-like transformation.

    The transformation is log(x + 1) if all x >= 0, else it is a custom symmetric log transform: shifted log,
    mirrored around the origin, so that the domain is all real numbers and the sign of the input is preserved.
    It is a monotonic transformation.
    """
    if np.all(x >= 0):
        return np.log(x + 1)
    return np.sign(x) * np.log(np.abs(x) + 1)


def quantile_transform_nonrandom(x):
    """Apply ``sklearn.preprocessing.quantile_transoform``.

    Converts column with extreme values to a uniform distribution. random_state seed is always 0.
    """
    return quantile_transform(x.reshape((-1, 1)), random_state=0, copy=True)


def identity(x):
    """Identity function."""
    return x


class BaseExtremeValueTransformer(BaseEstimator, TransformerMixin):
    """Applies a transformation to columns which have "extreme" values.

    A value is considered "extreme" if it is greater than ``quantile`` or less than 100 - ``quantile`` percent of the
    data, and is more than ``threshold_std`` many standard deviations away from the mean. Heavy-tailed distributions are
    therefore more likely to have "extreme" values.

    Number of output columns is the same as number of input columns: each column is either transformed or not.
    The default transformation is the identity function.

    Parameters
    ----------
    quantile : int (default = 98)
        Used to calculate the lower and upper cutoff quantiles for a value to be considered "extreme".
        This must be an integer between 0 and 100.

    threshold_std : float (default = 4.0)
        Number of standard deviations away from the mean (in standard units). For a given column, if the magnitude of
        the quantile cutoffs is greater than the threshold_std cutoff, then that column contains an extreme value.
        ``threshold_std`` is converted to nonstandard units:
        ``nonstandard_thresholds = standard_threshold * np.std(X, axis=0) + np.mean(X, axis=0)``.

    transform_function : transform_function : callable -> 1D np.array (default = lambda x: x)
        The function to transform the columns with extreme values. transform_function is applied to an entire column
        if that column contains an "extreme" value. `transform_function` is applied during the `transform` stage.
        No state will be kept between ``fit`` and ``transform``. To keep state, create a child class of
        ``BaseExtremeValueTransformer``.

    Attributes
    ----------
    n_input_features_ : int
        The number of columns in the input dataset.

    quantiles_ : 2D array (2, n_input_features_)
        For each column j, ``quantiles_[0, j]`` is the valueof the ``(100 - quantile)`` percentile and
        ``quantiles_[1, j]`` is the value of the ``quantile`` percentile.

    cols_to_transform_ : list of int
        List of column indices to determine which columns to apply the transformation of ``transform_function``.

    Notes
    -----
    Accepts only two-dimensional, dense input arrays.

    This class can be called directly if inputs in ``fit`` and ``transform`` stages are the same.

    Users can also subclass this class and override the ``fit`` and ``_transform_function`` methods to store state as
    class attributes. To see examples of this implementation, see
    ``sagemaker_sklearn_extension.preprocessing.LogExtremeValueTransformer`` or
    ``sagemaker_sklearn_extension.preprocessing.QuantileExtremeValueTransformer``.
    """

    def __init__(self, quantile=98, threshold_std=4.0, transform_function=identity):
        self.quantile = quantile
        self.threshold_std = threshold_std
        self.transform_function = transform_function

    def fit(self, X, y=None):
        """Compute the lower and upper quantile cutoffs and which columns to transform.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data array to transform. Must be numeric, non-sparse, and two-dimensional.

        Returns
        -------
        self : BaseExtremeValueTransformer
        """
        if not 0 <= self.quantile <= 100:
            raise ValueError(
                "Parameter `quantile` {} is invalid. `quantile` must be an integer between 0 and 100".format(
                    self.quantile
                )
            )

        X = check_array(X)
        _, self.n_input_features_ = X.shape

        self.quantiles_ = np.percentile(X, [100 - self.quantile, self.quantile], axis=0)

        nonstandard_threshold_stds = self.threshold_std * np.std(X, axis=0)
        col_means = np.mean(X, axis=0)
        threshold_upper_bound = nonstandard_threshold_stds + col_means
        threshold_lower_bound = -nonstandard_threshold_stds + col_means

        self.cols_to_transform_ = [
            j
            for j in range(self.n_input_features_)
            if self.quantiles_[0, j] < threshold_lower_bound[j] or self.quantiles_[1, j] > threshold_upper_bound[j]
        ]

        return self

    def transform(self, X, y=None):
        """Transform columns that contain extreme values with ``transform_function``.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data array to transform. Must be numeric, non-sparse, and two-dimensional.

        Returns
        -------
        Xt : np.ndarray, shape (n_samples, n_features)
            The array of transformed input.
        """
        check_is_fitted(self, ["quantiles_", "cols_to_transform_"])
        X = check_array(X)
        _, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape.")

        return_cols = [
            self._transform_function(X[:, j], j) if j in self.cols_to_transform_ else X[:, j]
            for j in range(self.n_input_features_)
        ]

        return np.column_stack(return_cols)

    def _transform_function(self, x, idx=None):
        """Applies ``self.transform_function`` to a column x.

        Parameters
        ----------
        x : 1D column, array-like

        idx : index, int
            index of 1D column in relation to the 2D array.

        Returns
        -------
        xt : transformed x
        """
        return self.transform_function(x)


class LogExtremeValuesTransformer(BaseExtremeValueTransformer):
    """Applies a log transformation to columns which have "extreme" values.

    The transformation is log(x + 1) if all x >= 0, else it is a custom symmetric log transform: shifted log,
    mirrored around the origin, so that the domain is all real numbers and the sign of the input is preserved.
    Nonnegative columns are determined during ``fit`` and stored as state, which are then used in ``transform``.

    A value is considered "extreme" if it is greater than ``quantile`` or less than 100 - ``quantile`` percent of the
    data, and is more than ``threshold_std`` many standard deviations away from the mean. Heavy-tailed distributions are
    therefore more likely to have "extreme" values.

    Number of output columns is the same as number of input columns: each column is either transformed or not.

    Parameters
    ----------
    quantile : int (default = 98)
        Used to calculate the lower and upper cutoff quantiles for a value to be considered "extreme".
        This must be an integer between 0 and 100.

    threshold_std : float (default = 4.0)
        Number of standard deviations away from the mean (in standard units). For a given column, if the magnitude of
        the quantile cutoffs is greater than the threshold_std cutoff, then that column contains an extreme value.
        ``threshold_std`` is converted to nonstandard units:
        ``nonstandard_thresholds = standard_threshold * np.std(X, axis=0) + np.mean(X, axis=0)``.


    Attributes
    ----------
    n_input_features_ : int
        The number of columns in the input dataset.

    quantiles_ : 2D array (2, n_input_features_)
        For each column j, ``quantiles_[0, j]`` is the valueof the ``(100 - quantile)`` percentile and
        ``quantiles_[1, j]`` is the value of the ``quantile`` percentile.

    cols_to_transform_ : list of int
        List of column indices to determine which columns to apply the transformation of ``transform_function``.

    nonnegative_cols_ : list of int
        List of column indices that contain all non-negative values.

    Notes
    -----
    Accepts only two-dimensional, dense input arrays.

    This class inhereits from ``sagemaker_sklearn_extension.preprocessing.BaseExtremeValueTransformer``.
    """

    def __init__(self, quantile=98, threshold_std=4.0):
        super().__init__(quantile=quantile, threshold_std=threshold_std)

    def fit(self, X, y=None):
        """Compute the lower and upper quantile cutoffs, columns to transform, and nonnegative columns.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data array to transform. Must be numeric, non-sparse, and two-dimensional.

        Returns
        -------
        self : LogExtremeValueTransformer
        """
        super().fit(X)
        X = check_array(X)
        self.nonnegative_cols_ = [j for j in range(self.n_input_features_) if np.all(X[:, j] >= 0)]
        return self

    def _transform_function(self, x, idx=None):
        """Apply a log-like transformation.

        The transformation is log(x + 1) if all x >= 0, else it is a custom symmetric log transform: shifted log,
        mirrored around the origin. Uses ``nonnegative_cols_`` from ``fit`` to determine which columns are negative.
        """
        if idx in self.nonnegative_cols_:
            return np.log(x + 1)
        return np.sign(x) * np.log(np.abs(x) + 1)


class QuantileExtremeValuesTransformer(BaseExtremeValueTransformer):
    """Applies a quantile transformation to columns which have "extreme" values.

    The quantile transformation is ``sklearn.preprocessing.quantile_transform`` that converts columns with extreme
    values to a uniform distribution. Quantiles are computed during the ``fit`` stage and stored as state, which are
    then used in ``transform``.

    A value is considered "extreme" if it is greater than ``quantile`` or less than 100 - ``quantile`` percent of the
    data, and is more than ``threshold_std`` many standard deviations away from the mean. Heavy-tailed distributions are
    therefore more likely to have "extreme" values.

    Number of output columns is the same as number of input columns: each column is either transformed or not.

    Parameters
    ----------
    quantile : int (default = 98)
        Used to calculate the lower and upper cutoff quantiles for a value to be considered "extreme".
        This must be an integer between 0 and 100.

    threshold_std : float (default = 4.0)
        Number of standard deviations away from the mean (in standard units). For a given column, if the magnitude of
        the quantile cutoffs is greater than the threshold_std cutoff, then that column contains an extreme value.
        ``threshold_std`` is converted to nonstandard units:
        ``nonstandard_thresholds = standard_threshold * np.std(X, axis=0) + np.mean(X, axis=0)``.


    Attributes
    ----------
    n_input_features_ : int
        The number of columns in the input dataset.

    quantiles_ : 2D array (2, n_input_features_)
        For each column j, ``quantiles_[0, j]`` is the valueof the ``(100 - quantile)`` percentile and
        ``quantiles_[1, j]`` is the value of the ``quantile`` percentile.

    cols_to_transform_ : list of int
        List of column indices to determine which columns to apply the transformation of ``transform_function``.

    quantile_transformer_ : ``sklearn.preprocessing.QuantileTransformer``
        Instance of ``sklearn.preprocessing.QuantileTransformer``.

    Notes
    -----
    Accepts only two-dimensional, dense input arrays.

    This class inherits from ``sagemaker_sklearn_extension.preprocessing.BaseExtremeValueTransformer``.
    """

    def __init__(self, quantile=98, threshold_std=4.0):
        super().__init__(quantile=quantile, threshold_std=threshold_std)

    def fit(self, X, y=None):
        """Compute the lower and upper quantile cutoffs, columns to transform, and each column's quantiles.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data array to transform. Must be numeric, non-sparse, and two-dimensional.

        Returns
        -------
        self : QuantileExtremeValueTransformer
        """
        super().fit(X)
        X = check_array(X)
        self.quantile_transformer_ = QuantileTransformer(random_state=0, copy=True)
        self.quantile_transformer_.fit(X)
        return self

    def _transform_function(self, x, idx=None):
        """Applies single column quantile transform from ``sklearn.preprocessing.QuantileTransformer``.

        Uses ``quantile_transformer_.quantiles_`` calculated during ``fit`` if given an index, otherwise the quantiles
        will be calculated from input ``x``.
        """
        if idx:
            return self.quantile_transformer_._transform_col(  # pylint: disable=protected-access
                x, self.quantile_transformer_.quantiles_[:, idx], False
            )
        return quantile_transform_nonrandom(x)


class RemoveConstantColumnsTransformer(BaseEstimator, TransformerMixin):
    """Removes columns that only contain one value.

    Examples
    ----------
    >>> X = [[0, 1, 2, np.nan],[0, np.nan, 2, np.nan],[0, 1, 3, np.nan]]
    >>> selector = RemoveConstantColumnsTransformer()
    >>> selector.fit_transform(X)
    array([[1, 2],
           [np.nan, 2],
           [1, 3]])

    Attributes
    ----------
    n_input_features_ : int
        The number of columns in the input dataset.

    cols_to_transform_ : array of shape [n_input_features_, ]
        A mask that indicates which columns have only one value
    """

    def fit(self, X, y=None):
        """Learn empirical variances from X.
        Parameters
        ----------
        X : array of shape [n_samples, n_features]
            Input samples from which to check uniqueness.

        Returns
        -------
        self
        """
        X = check_array(X, force_all_finite=False)
        _, self.n_input_features_ = X.shape
        all_nan_cols = np.all(np.isnan(X), axis=0)
        self.cols_to_transform_ = np.logical_or(
            np.array([np.unique(X[:, j]).size == 1 for j in range(self.n_input_features_)]), all_nan_cols
        )
        return self

    def transform(self, X):
        """Reduce X to features with a non-zero variance.
        Parameters
        ----------
        X : array of shape [n_samples, n_input_features_]
            The input samples.
        Returns
        -------
        X_t : array of shape [n_samples, n_selected_features]
            The input samples with only features with a non-zero variance.
        """
        check_is_fitted(self, "cols_to_transform_")
        X = check_array(X, force_all_finite=False)
        _, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape.")

        # If all columns are constant return an empty array with shape (0, n_input_features_)
        if np.sum(self.cols_to_transform_) == self.n_input_features_:
            return np.empty((0, self.n_input_features_), dtype=X.dtype)

        return X[:, ~self.cols_to_transform_]

    def _more_tags(self):
        return {"allow_nan": True}
