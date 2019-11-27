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

from itertools import combinations

import numpy as np
from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import StandardScaler
from sklearn.utils import check_array
from sklearn.utils import check_random_state
from sklearn.utils.validation import check_is_fitted
from sklearn.utils.validation import FLOAT_DTYPES


class QuadraticFeatures(BaseEstimator, TransformerMixin):
    """Generate and add quadratic features to feature matrix.

    Generate a new feature matrix containing the original data, an optional bias column, a collection of squared
    features, and a collection of interaction terms. If ``max_n_features`` is not large enough to include all the
    squared features, then a random subset of them is added instead. If it is large enough to include all squared
    features, but not large enough to include all quadratic features, then all of the squared features and a random
    subset of the interaction features are added instead.

    This transformer is similar to ``PolynomialFeatures`` from the ``sklearn.preprocessing.data`` module.

    Parameters
    ----------
    include_bias : boolean (default = False)
        Whether to include a bias column -- the feature in which all entries are set to 1.0, and which acts as the
        intercept term in a linear model. Note that this parameter is False by default, in contrast to the corresponding
        parameter in ``sklearn``'s ``PolynomialFeatures``.

    interaction_only : boolean (default = False)
        Whether to produce only interaction features, and omit the squared features. For example, if the features are
        [a, b], then this will include ab, but not a^2 and b^2. The bias column is not affected by this parameter.

    max_n_features : int (default = 1000)
        The maximum number of features to include in the output data matrix. Squared features are prioritized over
        interaction features, unless ``interaction_only`` is ``True``. Must be larger than the number of input features
        (plus one, if ``include_bias`` is ``True``).

    order : str in {'C', 'F'} (default = 'C')
        Order of the input array: 'C' stands for C-contiguous order, and 'F' stands for Fortran-contiguous order.

    random_state : int, RandomState instance, or None (default = 0)
        If int, ``random_state`` is the seed used by the random number generator; if ``RandomState`` instance,
        ``random_state`` is the random number generator; if None, the random number generator is the ``RandomState``
        instance used by ``np.random``.  Used to determine which feature combinations to include in the output dataset
        when ``max_n_features`` is too small to fit all quadratic features.

    Examples
    --------
    >>> import numpy as np
    >>> from sagemaker_sklearn_extension.preprocessing import QuadraticFeatures
    >>> X = np.arange(1, 7).reshape((2, 3))
    >>> X
    array([[1, 2, 3],
           [4, 5, 6]])
    >>> QuadraticFeatures().fit_transform(X)
    array([[ 1,  2,  3,  1,  4,  9,  2,  3,  6],
           [ 4,  5,  6, 16, 25, 36, 20, 24, 30]])
    >>> QuadraticFeatures(interaction_only=True, max_n_features=5).fit_transform(X)
    array([[ 1,  2,  3,  2,  3],
           [ 4,  5,  6, 20, 24]])

    Attributes
    ----------
    combinations_ : list of tuples (i, j)
        List of tuples with two elements, each containing the indexes of the columns that are multiplied element-wise
        to form a single output column. Tuples appear in the same order as the corresponding output columns.
    n_input_features_ : int
        The number of columns in the input dataset.
    n_output_features_ : int
        The number of columns in the output dataset.

    Notes
    -----
    Accepts only two-dimensional, dense input arrays.
    """

    def __init__(self, include_bias=False, interaction_only=False, max_n_features=1000, order="C", random_state=0):
        self.include_bias = include_bias
        self.interaction_only = interaction_only
        self.max_n_features = max_n_features
        self.order = order
        self.random_state = random_state

    def _build_combinations(self, n_features, random_state):
        """Calculate the feature pairs to be added to the input data based on parameters and number of input columns.

        If ``interaction_only`` is ``True``, all squared features are omitted. Otherwise, they are added before
        interaction features. If there is enough space--as indicated by ``max_n_features``--to add all squared features,
        then do so. Otherwise, take a random sub-sample. Then, if there's enough space to add all interaction features,
        do so. Otherwise, return a random sub-sample of those.

        Parameters
        ----------
        n_features : int
            The number of columns in the input vector.
        random_state : RandomState
            The prepared (using ``check_random_state``) ``RandomState`` instance.
        """
        # First calculate how many new features of each kind (squared and interaction) we can add.
        added_feature_budget = self.max_n_features - n_features - int(self.include_bias)
        if added_feature_budget <= 0:
            message = "max_n_features must be large enough for the output to contain more than the original dataset"
            if self.include_bias:
                message += " and bias column"
            raise ValueError(message)
        squared_feature_budget = 0 if self.interaction_only else min(added_feature_budget, n_features)
        interaction_feature_budget = max(0, added_feature_budget - squared_feature_budget)

        # Produce squared feature pairs.
        squared_features = []
        if squared_feature_budget == n_features:
            # No need to reorder if we can fit all squared features.
            squared_features = [(i, i) for i in range(n_features)]
        elif squared_feature_budget > 0:
            # Otherwise, take a random sample of them.
            squared_features = [
                (i, i) for i in random_state.choice(range(n_features), size=squared_feature_budget, replace=False)
            ]

        # Produce interaction feature pairs.
        interaction_features = []
        if interaction_feature_budget > 0:
            interaction_features = list(combinations(range(n_features), 2))

            # Take a random sample of feature interactions if not all can fit.
            if len(interaction_features) > interaction_feature_budget:
                random_state.shuffle(interaction_features)

            interaction_features = interaction_features[:interaction_feature_budget]

        return squared_features + interaction_features

    def fit(self, X, y=None):
        """
        Compute the number of output features and the combination of input features to multiply.

        Parameters
        ----------
        X : array-like , shape (n_samples, n_features)
            The data array to transform. Must be a non-sparse two-dimensional numpy array.

        Returns
        -------
        self : instance
        """
        _, n_features = check_array(X).shape
        random_state = check_random_state(self.random_state)
        self.combinations_ = self._build_combinations(n_features, random_state)
        self.n_input_features_ = n_features
        self.n_output_features_ = n_features + len(self.combinations_) + int(self.include_bias)
        return self

    def transform(self, X):
        """
        Transform data to the chosen quadratic features.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data array to transform. Must be a non-sparse and two-dimensional.

        Returns
        -------
        XQ : np.ndarray, shape (n_samples, n_output_features_)
            The array of computed features.
        """
        check_is_fitted(self, ["n_input_features_", "n_output_features_", "combinations_"])
        X = check_array(X, order=self.order)
        n_samples, n_features = X.shape

        if n_features != self.n_input_features_:
            raise ValueError("X shape does not match training shape.")

        XQ = np.empty((n_samples, self.n_output_features_), dtype=X.dtype, order=self.order)

        if self.include_bias:
            XQ[:, 0] = 1.0
            X_col_range_start, X_col_range_end = 1, self.n_input_features_ + 1
        else:
            X_col_range_start, X_col_range_end = 0, self.n_input_features_

        XQ[:, X_col_range_start:X_col_range_end] = X
        XQ[:, X_col_range_end:] = np.column_stack([X[:, i] * X[:, j] for i, j in self.combinations_])

        return XQ


class RobustStandardScaler(BaseEstimator, TransformerMixin):
    """Scaler to adaptively scale dense and sparse inputs.

    RobustStandardScaler uses `sklearn.preprocessing.StandardScaler` to perform standardization, but adapts
    the centering based on the sparsity of the data.

    For dense inputs, the standard score of a sample `x` is calculated as:

        z = (x - u) / s

    where `u` is the mean of the training samples, and `s` is the standard deviation of the training samples.
    The mean `u` is a vector of means of each feature.  If the number of zeros for a feature is greater than or
    equal to 70% of the total number of samples, the corresponding value in `u` is set to `0` to avoid centering
    by mean.

    For sparse inputs, the standard score of a sample `x` is calculated as:

        z = x / s

    where `s` is the standard deviation of the training samples.

    Parameters
    ----------
    copy : boolean, optional, default True
        If False, try to avoid a copy and do inplace scaling instead.
        This is not guaranteed to always work inplace; e.g. if the data is
        not a NumPy array or scipy.sparse CSR matrix, a copy may still be
        returned.

     Attributes
    ----------
    self.scaler_ : ``sklearn.preprocessing.StandardScaler``
        - `scaler_` is instantiated inside the fit method used for computing the center and the standard deviation.

    """

    def __init__(self, copy=True):
        self.copy = copy

    def fit(self, X, y=None):
        """Fit RobustStandardScaler to X.

        If input is sparse, `fit` overrides `self.with_mean` to standardize without subtracting mean (avoids breaking
        for sparse matrix)

        If the data is dense, the mean is adjusted for sparse features and the scaled with mean.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to standardize.

        Returns
        -------
        self : RobustStandardScaler
        """
        X = check_array(
            X, accept_sparse=("csr", "csc"), estimator=self, dtype=FLOAT_DTYPES, force_all_finite="allow-nan"
        )

        with_mean = True
        if issparse(X):
            with_mean = False

        self.scaler_ = StandardScaler(with_mean=with_mean, with_std=True, copy=self.copy)
        self.scaler_.fit(X)

        if self.scaler_.with_mean:
            nnz_mean_mask = np.where(np.count_nonzero(X, axis=0) / X.shape[0] > 0.3, 1, 0)
            self.scaler_.mean_ = self.scaler_.mean_ * nnz_mean_mask

        return self

    def transform(self, X):
        """
        Standardize data by centering and scaling.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            The data array to transform.

        Returns
        -------
        Xt : array-like, shape (n_samples, n_features)
            The array of transformed input.
        """
        return self.scaler_.transform(X)

    def _more_tags(self):
        return {"allow_nan": True}
