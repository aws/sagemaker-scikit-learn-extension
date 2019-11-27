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
import pytest
from scipy.sparse import csr_matrix, issparse

from sagemaker_sklearn_extension.preprocessing import QuadraticFeatures, RobustStandardScaler
from sklearn.utils.testing import assert_array_almost_equal, assert_array_equal


def _n_choose_2(n):
    """Calculates the number of 2-combinations of n elements."""
    return (n * (n - 1)) // 2


X = np.array([[1.0, 5.0], [2.0, 3.0], [1.0, 1.0],])
X_sparse = csr_matrix(X)
X_standardized = (X - np.mean(X, axis=0)) / np.std(X, axis=0)

X_small = np.arange(6).reshape((2, 3))
X_small_n_rows, X_small_n_cols = X_small.shape
X_small_n_col_combinations = _n_choose_2(X_small_n_cols)

X_low_nnz = np.array(
    [[1.0, 5.0, 0], [2.0, 0.0, 0], [2.0, 1.0, 0], [1.0, 0.0, 1], [2.0, 3.0, 0], [3.0, 0.0, 3], [4.0, 5.0, 0],]
)
low_nnz_mask = np.where((np.count_nonzero(X_low_nnz, axis=0) / X_low_nnz.shape[0]) > 0.3, 1, 0)
X_low_nnz_standardized = (X_low_nnz - np.mean(X_low_nnz, axis=0) * low_nnz_mask) / np.std(X_low_nnz, axis=0)


def test_quadratic_features_explicit():
    """Explicitly test the return value for a small float-filled input matrix."""
    X_observed = QuadraticFeatures().fit_transform(X_standardized)
    X_expected = np.hstack(
        [
            X_standardized,
            (X_standardized[:, 0] * X_standardized[:, 0]).reshape((-1, 1)),
            (X_standardized[:, 1] * X_standardized[:, 1]).reshape((-1, 1)),
            (X_standardized[:, 0] * X_standardized[:, 1]).reshape((-1, 1)),
        ]
    )
    assert_array_equal(X_observed, X_expected)


def test_quadratic_features_max_n_features():
    """Test that small but valid ``max_n_features`` produces a non-complete set of combinations."""
    transformer = QuadraticFeatures(max_n_features=5)
    transformer.fit(X_small)
    assert len(transformer.combinations_) == 5 - X_small_n_cols


@pytest.mark.parametrize(
    ["include_bias", "max_n_features"],
    [
        # Exactly at limit of what's allowed.
        (False, X_small_n_col_combinations),
        (True, X_small_n_col_combinations + 1),
        # Smaller than limit of what's allowed.
        (False, X_small_n_col_combinations - 1),
        (True, X_small_n_col_combinations - 1),
    ],
)
def test_quadratic_features_max_n_features_too_small(include_bias, max_n_features):
    """Test that when the ``max_n_features`` parameter is too small, an exception is raised."""
    transformer = QuadraticFeatures(include_bias=include_bias, max_n_features=max_n_features,)
    with pytest.raises(ValueError):
        transformer.fit(X_small)


def test_quadratic_features_random_state_invariance():
    """Test that the exact same input is produced when using the same random seed."""
    transformer1 = QuadraticFeatures(random_state=0)
    transformer2 = QuadraticFeatures(random_state=0)
    X1 = transformer1.fit_transform(X_small)
    X2 = transformer2.fit_transform(X_small)
    assert np.all(X1 == X2)


@pytest.mark.parametrize(
    ["include_bias", "interaction_only", "n_output_features"],
    [
        (False, False, X_small_n_cols + 2 * X_small_n_col_combinations),
        (True, False, X_small_n_cols + 2 * X_small_n_col_combinations + 1),
        (False, True, X_small_n_cols + X_small_n_col_combinations),
        (True, True, X_small_n_cols + X_small_n_col_combinations + 1),
    ],
)
def test_quadratic_features_shape(include_bias, interaction_only, n_output_features):
    """Test that various parameter values produce expected resulting data shapes."""
    transformer = QuadraticFeatures(include_bias=include_bias, interaction_only=interaction_only,)
    XQ = transformer.fit_transform(X_small)
    assert XQ.shape == (X_small_n_rows, n_output_features)


def test_quadratic_features_single_column_input_explicit():
    """Test that using a single-column matrix as input produces the expected output."""
    X_observed = QuadraticFeatures().fit_transform(X_standardized[:, 0].reshape((-1, 1)))
    X_expected = np.hstack([X_standardized[:, [0]], (X_standardized[:, 0] * X_standardized[:, 0]).reshape((-1, 1)),])
    assert_array_equal(X_observed, X_expected)


def test_robust_standard_scaler_dense():
    scaler = RobustStandardScaler()
    X_observed = scaler.fit_transform(X)

    assert_array_equal(X_observed, X_standardized)


def test_robust_standard_scaler_sparse():
    scaler = RobustStandardScaler()
    X_observed = scaler.fit_transform(X_sparse)

    assert issparse(X_observed)
    assert_array_almost_equal(X_observed.toarray(), X / np.std(X, axis=0))


def test_robust_standard_dense_with_low_nnz_columns():
    scaler = RobustStandardScaler()
    X_observed = scaler.fit_transform(X_low_nnz)
    assert_array_almost_equal(X_observed, X_low_nnz_standardized)
