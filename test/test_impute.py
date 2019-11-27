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

from sklearn.utils.testing import assert_array_equal

from sagemaker_sklearn_extension.impute import RobustImputer, RobustMissingIndicator, is_finite_numeric

X_impute = np.array([[np.nan, 2, np.inf], [4, np.inf, 6], [10, np.nan, 10]])
X_impute_boolean_mask = np.array([[True, False, True], [False, True, False], [False, True, False]])
X_impute_string = X_impute.astype("O")
X_impute_mixed = np.array([["2", "a"], ["inf", "nan"], ["-1e2", "10.0"], ["0.0", "foobar"], ["-inf", "8"]])
X_impute_mixed_boolean_mask = np.array([[False, True], [True, True], [False, False], [False, True], [True, False]])
X_impute_categorical = np.array([["hot dog"], ["hot dog"], ["hot dog"], ["banana"]])
X_imputed_median = np.array([[7.0, 2.0, 8.0], [4.0, 2.0, 6.0], [10.0, 2.0, 10.0]])
X_imputed_constant = np.array([[1.0, 2.0, 13.0], [4.0, 7.0, 6.0], [10.0, 7.0, 10.0]])
X_imputed_mixed = np.array([[2.0, 9.0], [0.0, 9.0], [-1e2, 10.0], [0.0, 9.0], [0.0, 8.0]])
X_imputed_categorical = np.array([["hot dog"], ["hot dog"], ["hot dog"], ["not hot dog"]])

transform_error_msg = "'transform' input X has 4 features per sample, expected 3 from 'fit' input"
fill_values_error_msg = "'fill_values' should have length equal to number of features in X 3, got 5"


@pytest.mark.parametrize(
    "val, expected", [(np.array([1738, "10", np.inf, np.nan, "foobar"]), np.array([True, True, False, False, False]))]
)
def test_is_finite_numeric(val, expected):
    observed = is_finite_numeric(val)
    assert_array_equal(observed, expected)


@pytest.mark.parametrize(
    "X, X_expected, strategy, fill_values",
    [
        (X_impute_mixed, X_imputed_mixed, "median", None),
        (X_impute, X_imputed_median, "median", None),
        (X_impute_string, X_imputed_median, "median", None),
        (X_impute, X_imputed_constant, "constant", [1.0, 7.0, 13.0]),
        (X_impute_string, X_imputed_constant, "constant", [1.0, 7.0, 13.0]),
    ],
)
def test_robust_imputer(X, X_expected, strategy, fill_values):
    robust_imputer = RobustImputer(strategy=strategy, fill_values=fill_values)
    robust_imputer.fit(X)
    X_observed = robust_imputer.transform(X)

    assert_array_equal(X_observed, X_expected)


def test_robust_imputer_categorical_custom_function():
    robust_imputer = RobustImputer(
        dtype=np.dtype("O"), strategy="constant", fill_values="not hot dog", mask_function=lambda x: x == "hot dog"
    )
    robust_imputer.fit(X_impute_categorical)
    X_observed = robust_imputer.transform(X_impute_categorical)

    assert_array_equal(X_observed, X_imputed_categorical)


def test_robust_imputer_transform_dim_error():
    with pytest.raises(ValueError, match=transform_error_msg):
        robust_imputer = RobustImputer()
        robust_imputer.fit(X_impute)
        robust_imputer.transform(np.zeros((3, 4)))


def test_robust_imputer_fill_values_dim_error():
    with pytest.raises(ValueError, match=fill_values_error_msg):
        robust_imputer = RobustImputer(strategy="constant", fill_values=np.zeros(5))
        robust_imputer.fit(X_impute)


@pytest.mark.parametrize(
    "X, boolean_mask_X", [(X_impute_mixed, X_impute_mixed_boolean_mask), (X_impute, X_impute_boolean_mask)]
)
def test_robust_missing_indicator(X, boolean_mask_X):
    robust_indicator = RobustMissingIndicator()
    robust_indicator.fit(X)
    boolean_mask_X_observed = robust_indicator.transform(X)

    assert_array_equal(boolean_mask_X_observed, boolean_mask_X)
