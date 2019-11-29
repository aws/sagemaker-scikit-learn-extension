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

from sklearn.utils.testing import assert_array_equal, assert_array_almost_equal

from sagemaker_sklearn_extension.preprocessing import (
    LogExtremeValuesTransformer,
    QuantileExtremeValuesTransformer,
    RemoveConstantColumnsTransformer,
    log_transform,
    quantile_transform_nonrandom,
)

np.random.seed(0)

X_zeros = np.zeros((10, 10))
X_extreme_vals = np.array(
    [
        [0.0, 0.0, 0.0],
        [-1.0, 1.0, 1.0],
        [-2.0, 2.0, 2.0],
        [-3.0, 3.0, 3.0],
        [-4.0, 4.0, 4.0],
        [-5.0, 5.0, 5.0],
        [-6.0, 6.0, 6.0],
        [-7.0, 7.0, 7.0],
        [-8.0, 8.0, 8.0],
        [-9.0, 9.0, 9.0],
        [-10.0, 10.0, 10.0],
        [-1e5, 1e6, 11.0],
    ]
)
X_log_extreme_vals = np.column_stack(
    [log_transform(X_extreme_vals.copy()[:, 0]), log_transform(X_extreme_vals.copy()[:, 1]), X_extreme_vals[:, 2]]
)
X_quantile_extreme_vals = np.column_stack(
    [
        quantile_transform_nonrandom(X_extreme_vals.copy()[:, 0]),
        quantile_transform_nonrandom(X_extreme_vals.copy()[:, 1]),
        X_extreme_vals[:, 2],
    ]
)
X_all_positive = 5 * np.random.random((100, 1)) + 20
X_extreme_all_positive = np.vstack([np.random.random((90, 1)) + 100, np.array(10 * [[5]], dtype=np.float64)])
X_log_extreme_all_positive = np.array([log_transform(X_extreme_all_positive.copy()[:, 0])]).reshape(-1, 1)
X_all_uniques = np.arange(20).reshape(4, 5)
X_one_val = np.column_stack([np.arange(20).reshape(4, 5), np.array([1, 1, 1, 1])])
X_nans = np.array([[np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan], [np.nan, np.nan, np.nan]])
X_no_uniques = np.zeros((4, 5))


@pytest.mark.parametrize(
    "X, X_expected",
    [
        (X_all_uniques, X_all_uniques),
        (X_one_val, X_one_val[:, :5]),
        (X_nans, np.empty((0, 3))),
        (X_no_uniques, np.empty((0, 5))),
    ],
)
def test_remove_constant_columns_transformer(X, X_expected):
    transformer = RemoveConstantColumnsTransformer()
    X_observed = transformer.fit_transform(X)

    assert_array_equal(X_observed, X_expected)


@pytest.mark.parametrize(
    ["X", "X_expected"],
    [
        (X_extreme_vals, X_log_extreme_vals,),
        (X_zeros, X_zeros),
        (X_all_positive, X_all_positive),
        (X_extreme_all_positive, X_log_extreme_all_positive),
    ],
)
def test_log_extreme_value_transformer(X, X_expected):
    transformer = LogExtremeValuesTransformer(threshold_std=2.0)
    X_observed = transformer.fit_transform(X)

    assert_array_almost_equal(X_observed, X_expected)


def test_log_extreme_value_transformer_state():
    t = LogExtremeValuesTransformer(threshold_std=2.0)
    X_observed = t.fit_transform(X_extreme_vals)

    assert_array_almost_equal(t.nonnegative_cols_, [1, 2])
    assert_array_almost_equal(X_observed, X_log_extreme_vals)


@pytest.mark.parametrize(
    ["X", "X_expected"],
    [(X_extreme_vals, X_quantile_extreme_vals), (X_zeros, X_zeros), (X_all_positive, X_all_positive),],
)
def test_extreme_value_transformer(X, X_expected):
    transformer = QuantileExtremeValuesTransformer(threshold_std=2.0)
    X_observed = transformer.fit_transform(X)

    assert_array_almost_equal(X_observed, X_expected)
