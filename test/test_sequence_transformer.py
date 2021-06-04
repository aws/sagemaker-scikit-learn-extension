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

from sagemaker_sklearn_extension.feature_extraction.sequences import TSFlattener
from sagemaker_sklearn_extension.feature_extraction.sequences import TSFreshFeatureExtractor

# To test TSFlattener with and without missing values encoded in different ways
X_sequence = [["1, 2, 3, 44"], ["11, 12, 14, 111"], ["1, 1, 1, 2"]]
X_flat = [[1, 2, 3, 44], [11, 12, 14, 111], [1, 1, 1, 2]]
X_sequence_missing = [["1, NaN, 3, 44"], ["11, 12, NaN, 111"], ["NaN, NaN, 1, NaN"]]
X_flat_filled = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_sequence_varying_length = [["1"], ["11, 111"], ["2, 3, 1, 4"]]
X_flat_padded = [[1, np.nan, np.nan, np.nan], [11, 111, np.nan, np.nan], [2, 3, 1, 4]]
X_sequence_varying_length_missing = [["1"], ["NaN, 111"], ["2, NaN, 1, 4"]]
X_flat_padded_filled = [[1, np.nan, np.nan, np.nan], [np.nan, 111, np.nan, np.nan], [2, np.nan, 1, 4]]
X_sequence_gaps = [["1, , 3, 44"], ["11, 12, , 111"], [", , 1, "]]
X_sequence_variable_gaps = [[" 1,    , 3,   44"], ["11,  12,  , 111"], [",, 1, "]]
X_flat_gaps_filled = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_sequence_inf = [["1, inf, 3, 44"], ["11, 12, inf, 111"], ["inf, inf, 1, inf"]]
X_flat_inf = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_sequence_empty = [["1, 2, 3, 44"], [""], ["1, 1, 1, 2"]]
X_flat_full = [[1, 2, 3, 44], [np.nan, np.nan, np.nan, np.nan], [1, 1, 1, 2]]
X_multiple_sequences = [["1, 2, 3, 4", "1, 1, 3, 3"], ["11, 12, 14, 11", "10, 1, 1, 2"], ["10, 1, 1, 2", "1, 1, 1, 2"]]

# To test TSFreshFeatureExtractor with and without np.nans
X_input = np.array([[1, 2, 3], [4, 5, 6], [10, 10, 10]])
X_transformed = np.array([[1, 2, 3], [4, 5, 6], [10, 10, 10]])
X_impute = np.array([[np.nan, 2, np.nan], [4, np.nan, 6], [10, np.nan, 10]])
X_padded = np.array([[0, 2, 0], [4, 0, 6], [10, 0, 10]])
X_filled = np.array([[2, 2, 2], [4, 4, 6], [10, 10, 10]])
X_interpolated = np.array([[2, 2, 3], [4, 5, 6], [10, 10, 10]])
X_with_first_feature = np.array([[1.0, 2.0, 3.0, 44.0], [11.0, 12.0, 14.0, 111.0], [1.0, 1.0, 1.0, 2.0]])
X_padded_with_first_feature = np.array([[1.0, 0.0, 3.0, 44.0], [11.0, 12.0, 0.0, 111.0], [0.0, 0.0, 1.0, 0.0]])

flattener_error_msg = "TSFlattener can process a single sequence column at a time, but it was given 2 sequence columns."
tsfresh_error_msg = "The input dimension is 4 instead of the expected 3"


@pytest.mark.parametrize(
    "X, X_expected",
    [
        (X_sequence, X_flat),
        (X_sequence_missing, X_flat_filled),
        (X_sequence_varying_length, X_flat_padded),
        (X_sequence_varying_length_missing, X_flat_padded_filled),
        (X_sequence_gaps, X_flat_gaps_filled),
        (X_sequence_variable_gaps, X_flat_gaps_filled),
        (X_sequence_inf, X_flat_inf),
        (X_sequence_empty, X_flat_full),
    ],
)
def test_flattener(X, X_expected):
    ts_flattener = TSFlattener()
    X_observed = ts_flattener.transform(X)
    assert_array_equal(X_observed, X_expected)


def test_flattener_transform_input_error():
    with pytest.raises(AssertionError, match=flattener_error_msg):
        ts_flattener = TSFlattener()
        ts_flattener.transform(X_multiple_sequences)


@pytest.mark.parametrize(
    "X, num_expected_features, augment", [(X_input, 790, True), (X_input, 787, False)],
)
def test_tsfresh_feature_dimension(X, num_expected_features, augment):
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=augment)
    tsfresh_feature_extractor.fit(X_input)
    X_with_features = tsfresh_feature_extractor.transform(X)
    num_tsfresh_features = X_with_features.shape[1]
    assert num_tsfresh_features == num_expected_features


@pytest.mark.parametrize(
    "X, X_expected, augment, interpolation_method",
    [
        (X_input, X_input, True, None),
        (X_impute, X_padded, True, "zeroes"),
        (X_impute, X_filled, True, "fill"),
        (X_impute, X_interpolated, True, "linear"),
    ],
)
def test_tsfresh_interpolations(X, X_expected, augment, interpolation_method):
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=augment, interpolation_method=interpolation_method)
    tsfresh_feature_extractor.fit(X)
    X_with_features = tsfresh_feature_extractor.transform(X)
    num_dim = X.shape[1]
    X_observed = X_with_features[:, :num_dim]
    assert_array_equal(X_observed, X_expected)


def test_tsfresh_transform_dim_error():
    with pytest.raises(ValueError, match=tsfresh_error_msg):
        tsfresh_feature_extractor = TSFreshFeatureExtractor()
        tsfresh_feature_extractor.fit(X_impute)
        tsfresh_feature_extractor.transform(np.zeros((3, 4)))


@pytest.mark.parametrize(
    "X, X_expected", [(X_sequence, X_with_first_feature), (X_sequence_missing, X_padded_with_first_feature),],
)
def test_full_time_series_pipeline(X, X_expected):
    # Extract sequences from strings
    ts_flattener = TSFlattener()
    X_flattened = ts_flattener.transform(X)
    # Compute tsfresh features
    tsfresh_feature_extractor = TSFreshFeatureExtractor()
    tsfresh_feature_extractor.fit(X_flattened)
    X_with_features = tsfresh_feature_extractor.transform(X_flattened)
    X_observed = X_with_features[:4, :4]
    assert_array_equal(X_observed, X_expected)
