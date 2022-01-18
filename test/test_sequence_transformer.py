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
import tsfresh
import sagemaker_sklearn_extension.feature_extraction.sequences
import importlib
from sklearn.utils.testing import assert_array_almost_equal
from sagemaker_sklearn_extension.feature_extraction.sequences import TSFeatureExtractor
from sagemaker_sklearn_extension.feature_extraction.sequences import TSFlattener
from sagemaker_sklearn_extension.feature_extraction.sequences import TSFreshFeatureExtractor
from tsfresh.defaults import N_PROCESSES


# To test TSFlattener with and without missing values encoded in different ways
# with fixed-length inputs
X_sequence = [["1, 2, 3, 44"], ["11, 12, 14, 111"], ["1, 1, 1, 2"]]
X_flat = [[1, 2, 3, 44], [11, 12, 14, 111], [1, 1, 1, 2]]
X_sequence_missing = [["1, NaN, 3, 44"], ["11, 12, NaN, 111"], ["NaN, NaN, 1, NaN"]]
X_sequence_non_numeric = [["1, a, 3, 44"], ["11, 12, b, 111"], ["c, c, 1, c"]]
X_flat_filled = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_sequence_gaps = [["1, , 3, 44"], ["11, 12, , 111"], [", , 1, "]]
X_sequence_variable_gaps = [[" 1,    , 3,   44"], ["11,  12,  , 111"], [",, 1, "]]
X_flat_gaps_filled = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_sequence_inf = [["1, inf, 3, 44"], ["11, 12, -inf, 111"], ["-inf, inf, 1, inf"]]
X_flat_inf = [[1, np.nan, 3, 44], [11, 12, np.nan, 111], [np.nan, np.nan, 1, np.nan]]
X_multiple_sequences = [["1, NaN, 3, 44", "1, 1, , 3"], ["11, 12, NaN, 111", "10, 1, 1"], ["NaN, NaN, 1", "1, 1, 1, 2"]]
X_all_nan_column = [
    [",,,"],
    [",,,"],
    [",,,"],
    [",,,"],
]
X_sequence_5_columns = [
    ["1, 2", "11, 12", "1, 4", "11, 99", "7, 4"],
    ["1, 5", "11, 99", "7, 4", "71, 88", "7, 2"],
    ["1, 33", "11, 88", "1, 2", "1, 7", "11, 99"],
]

# with variable-length inputs
X_sequence_varying_length = [["1, 2"], ["11, 111"], ["2, 3, 1, 4"]]
X_flat_varying_length = [[1, 2], [11, 111], [2, 3, 1, 4]]
X_sequence_empty = [["1, 2, 3, 44"], [""], ["1, 1, 1, 2"]]
X_sequence_none = [["1, 2, 3, 44"], [None], ["1, 1, 1, 2"]]
X_flat_full = [[1, 2, 3, 44], [np.nan], [1, 1, 1, 2]]
# with sequences to trim
X_sequence_to_trim = [["1, 2, 3"], ["11, 12, 14, 111"], ["1, 1"]]
MAX_LENGTH = 2  # maximum allowed sequence length in test_flattener_with_truncation
X_flat_trimmed_start = [[2, 3], [14, 111], [1, 1]]
X_flat_trimmed_end = [[1, 2], [11, 12], [1, 1]]

flattener_error_msg = "TSFlattener can process a single sequence column at a time, but it was given 2 sequence columns."

# To test TSFreshFeatureExtractor
X_input = [[1], [4, 5, 6], [10, 10, 10, 10]]
X_nans = [[1, np.nan, np.nan, np.nan], [4, 5, 6, np.nan], [10, 10, 10, 10]]
# to test its interpolation strategies
X_impute = np.array([[np.nan, 2, np.nan], [4, np.nan, 6], [10, np.nan, 10]])
X_padded = np.array([[0, 2, 0], [4, 0, 6], [10, 0, 10]])
X_filled = np.array([[2, 2, 2], [4, 4, 6], [10, 10, 10]])
X_interpolated = np.array([[2, 2, 3], [4, 5, 6], [10, 10, 10]])
X_hybrid = np.array([[0, 2, 0], [4, 4, 6], [10, 10, 10]])
# to test the case of observations with only np.nans
X_all_nan = [[np.nan, np.nan, np.nan, np.nan], [np.nan], [10, 10, 10, 10], [10, 20, 30, 40]]
X_all_nan_imputed = [[0, 0, 0, 0], [0, 0, 0, 0], [10, 10, 10, 10], [10, 20, 30, 40]]
# to test that the first tsfresh feature is computed correctly
X_with_first_feature = np.array(
    [[1.0, 2.0, 3.0, 44.0, 50.0], [11.0, 12.0, 14.0, 111.0, 148.0], [1.0, 1.0, 1.0, 2.0, 5.0]]
)

X_filled_with_first_feature = np.array(
    [[1.0, 1.0, 3.0, 44.0, 48.0], [11.0, 12.0, 12.0, 111.0, 134.0], [0.0, 0.0, 1.0, 0.0, 1.0]]
)
X_padded_with_first_feature = np.array(
    [[1.0, 2.0, 0.0, 0.0, 3.0], [11.0, 111.0, 0.0, 0.0, 122.0], [2.0, 3.0, 1.0, 4.0, 10.0]]
)


@pytest.mark.parametrize(
    "X, X_expected",
    [
        (X_sequence, X_flat),
        (X_sequence_missing, X_flat_filled),
        (X_sequence_non_numeric, X_flat_filled),
        (X_sequence_gaps, X_flat_gaps_filled),
        (X_sequence_variable_gaps, X_flat_gaps_filled),
        (X_sequence_inf, X_flat_inf),
    ],
)
def test_flattener_fixed_length(X, X_expected):
    ts_flattener = TSFlattener()
    X_observed = ts_flattener.transform(X)
    assert_array_almost_equal(X_observed, X_expected)


@pytest.mark.parametrize(
    "X, X_expected",
    [
        (X_sequence_varying_length, X_flat_varying_length),
        (X_sequence_empty, X_flat_full),
        (X_sequence_none, X_flat_full),
    ],
)
def test_flattener_varying_length(X, X_expected):
    ts_flattener = TSFlattener()
    X_observed = ts_flattener.transform(X)
    [assert_array_almost_equal(x, y) for x, y in zip(X_observed, X_expected)]


@pytest.mark.parametrize(
    "X, X_expected, trim_beginning",
    [(X_sequence_to_trim, X_flat_trimmed_start, True), (X_sequence_to_trim, X_flat_trimmed_end, False),],
)
def test_flattener_with_truncation(X, X_expected, trim_beginning):
    ts_flattener = TSFlattener(max_allowed_length=MAX_LENGTH, trim_beginning=trim_beginning)
    X_observed = ts_flattener.transform(X)
    assert_array_almost_equal(X_observed, X_expected)


def test_flattener_transform_input_error():
    with pytest.raises(ValueError, match=flattener_error_msg):
        ts_flattener = TSFlattener()
        ts_flattener.transform(X_multiple_sequences)


@pytest.mark.parametrize(
    "X, num_expected_features, augment, extraction_type",
    [
        (X_input, 791, True, "all"),
        (X_input, 787, False, "all"),
        (X_input, 785, True, "efficient"),
        (X_input, 781, False, "efficient"),
        (X_input, 13, True, "minimal"),
        (X_input, 9, False, "minimal"),
    ],
)
def test_tsfresh_feature_dimension(X, num_expected_features, augment, extraction_type):
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=augment, extraction_type=extraction_type)
    tsfresh_feature_extractor.fit(X_input)
    X_with_features = tsfresh_feature_extractor.transform(X)
    num_tsfresh_features = X_with_features.shape[1]
    assert num_tsfresh_features == num_expected_features


@pytest.mark.parametrize(
    "X, X_expected, interpolation_method",
    [
        (X_input, X_nans, None),
        (X_impute, X_padded, "zeroes"),
        (X_impute, X_filled, "fill"),
        (X_impute, X_interpolated, "linear"),
        (X_impute, X_hybrid, "hybrid"),
    ],
)
def test_tsfresh_interpolations(X, X_expected, interpolation_method):
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=True, interpolation_method=interpolation_method)
    tsfresh_feature_extractor.fit(X)
    X_with_features = tsfresh_feature_extractor.transform(X)
    max_dim = np.max([len(x) for x in X])
    X_observed = X_with_features[:, :max_dim]
    assert_array_almost_equal(X_observed, X_expected)


def test_tsfresh_all_nan():
    X = X_all_nan
    X_expected = X_all_nan_imputed
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=True, extraction_type="minimal")
    X_with_features = tsfresh_feature_extractor.fit_transform(X)
    max_dim = np.max([len(x) for x in X])
    X_observed = X_with_features[:, :max_dim]
    computed_tsfresh_features = X_with_features[2:, max_dim:]
    imputed_tsfresh_features_observed = X_with_features[:2, max_dim:]
    tsfresh_features_median = np.median(computed_tsfresh_features, axis=0)
    imputed_tsfresh_features_expected = np.repeat(tsfresh_features_median.reshape((1, -1)), 2, axis=0)
    assert_array_almost_equal(X_observed, X_expected)
    # 13 features --- 4 from augmentation, 9 generated by minimal settings of tsfresh
    assert X_with_features.shape == (4, 13)
    assert_array_almost_equal(imputed_tsfresh_features_observed, imputed_tsfresh_features_expected)


def test_tsfresh_transform_variable_input_dim():
    tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=False, extraction_type="all")
    tsfresh_feature_extractor.fit(X_impute)
    tsfresh_features_first_set = tsfresh_feature_extractor.transform(X_impute.tolist())
    tsfresh_features_second_set = tsfresh_feature_extractor.transform(np.ones((5, 6)).tolist())
    num_tsfresh_features_first_set = tsfresh_features_first_set.shape[1]
    num_tsfresh_features_second_set = tsfresh_features_second_set.shape[1]
    assert num_tsfresh_features_first_set == 787
    assert num_tsfresh_features_second_set == 787


@pytest.mark.parametrize(
    "X, X_expected",
    [
        (X_sequence, X_with_first_feature),
        (X_sequence_missing, X_filled_with_first_feature),
        (X_sequence_varying_length, X_padded_with_first_feature),
    ],
)
def test_full_time_series_pipeline(X, X_expected):
    # Extract sequences from strings
    ts_flattener = TSFlattener()
    X_flattened = ts_flattener.transform(X)
    # Compute tsfresh features
    tsfresh_feature_extractor = TSFreshFeatureExtractor(extraction_type="minimal", augment=True)
    tsfresh_feature_extractor.fit(X_flattened)
    X_with_features_combined_transformers = tsfresh_feature_extractor.transform(X_flattened)
    X_observed_combined_transformers = X_with_features_combined_transformers[:5, :5]
    # Repeat the above in a single step with the TSFeatureExtractor wrapper
    time_series_feature_extractor = TSFeatureExtractor(
        extraction_type="minimal", augment=True, sequences_lengths_q25=[4]
    )
    X_with_features_time_series_transformer = time_series_feature_extractor.fit_transform(X)
    X_observed_time_series_transformer = X_with_features_time_series_transformer[:5, :5]
    # Compare the two outputs with each other and with the expected result

    assert_array_almost_equal(X_observed_combined_transformers, X_expected)
    assert_array_almost_equal(X_observed_time_series_transformer, X_expected)
    assert_array_almost_equal(X_with_features_combined_transformers, X_with_features_time_series_transformer)


@pytest.mark.parametrize(
    "X, X_expected, sequences_lengths_q25",
    [
        (X_multiple_sequences, X_filled_with_first_feature, [4, 4]),
        (X_sequence_missing, X_filled_with_first_feature, [4]),
    ],
)
def test_time_series_feature_extractor_multiple_columns(X, X_expected, sequences_lengths_q25):
    time_series_feature_extractor = TSFeatureExtractor(
        augment=True, extraction_type="minimal", sequences_lengths_q25=sequences_lengths_q25
    )
    X_with_features_time_series_transformer = time_series_feature_extractor.fit_transform(X)
    num_sequence_columns = np.array(X).shape[1]
    # The output is expected to have 787 features (extracted from tsfresh) for each sequence column in the input,
    # plus the 4 raw features from each original sequence column (since augment = True)
    num_expected_features = 13 * num_sequence_columns
    assert X_with_features_time_series_transformer.shape == (3, num_expected_features)
    X_observed = X_with_features_time_series_transformer[:5, :5]
    assert_array_almost_equal(X_observed, X_expected)


@pytest.mark.parametrize("sequences_lengths_q25, feats_num", [([10], 27), ([100], 252), ([1000], 781), ([10000], 781)])
def test_time_series_expansion_control(sequences_lengths_q25, feats_num):
    time_series_feature_extractor = TSFeatureExtractor(
        extraction_type="efficient", augment=False, sequences_lengths_q25=sequences_lengths_q25
    )
    X_out = time_series_feature_extractor.fit_transform(X_sequence)
    assert X_out.shape[1] == feats_num


def test_time_series_expansion_control_seed():
    time_series_feature_extractor = TSFeatureExtractor(
        extraction_type="efficient", augment=False, sequences_lengths_q25=[5], extraction_seed=27
    )
    time_series_feature_extractor.fit(X_sequence)
    X_out = time_series_feature_extractor.tsfresh_feature_extractors_[0].transform(
        TSFlattener().transform(enumerate(np.array(X_sequence).T).__next__()[1].reshape(-1, 1))
    )
    assert (
        list(X_out.columns.values).sort()
        == [
            "0__sum_values",
            "0__median",
            "0__mean",
            "0__length",
            "0__standard_deviation",
            "0__variance",
            "0__root_mean_square",
            "0__maximum",
            "0__minimum",
            "0__cwt_coefficients__coeff_14__w_5__widths_(2, 5, 10, 20)",
            '0__fft_coefficient__attr_"imag"__coeff_85',
            '0__fft_coefficient__attr_"abs"__coeff_19',
            '0__fft_coefficient__attr_"abs"__coeff_72',
            '0__fft_coefficient__attr_"angle"__coeff_63',
            "0__energy_ratio_by_chunks__num_segments_10__segment_focus_4",
        ].sort()
    )


@pytest.mark.parametrize(
    "sequences_lengths_q25, expansion_thresholds",
    [([500, 200, 1000, 2000, 100], [329, 132, 658, 1316, 66]), ([5, 10, 20, 15, 25], [15, 27, 52, 40, 65])],
)
def test_time_series_expansion_control_across_columns(sequences_lengths_q25, expansion_thresholds):
    time_series_feature_extractor = TSFeatureExtractor(
        extraction_type="efficient", augment=False, sequences_lengths_q25=sequences_lengths_q25
    )
    time_series_feature_extractor.fit(X_sequence_5_columns)
    for i, extractor in enumerate(time_series_feature_extractor.tsfresh_feature_extractors_):
        assert expansion_thresholds[i] == extractor.expansion_threshold


@pytest.mark.parametrize(
    "settings, expansion_threshold, expected_settings",
    [
        ({"k1": None, "k2": ["v1", "v2"], "k3": None}, 1, {"k2": ["v2"]}),
        ({"k2": ["v1", "v2", "v3"], "k3": None}, 2, {"k2": ["v3"], "k3": None}),
        ({"k1": None, "k2": ["v1", "v2"], "k3": None}, 15, {"k1": None, "k2": ["v1", "v2"], "k3": None}),
    ],
)
def test_apply_feature_threshold(settings, expansion_threshold, expected_settings):
    time_series_feature_extractor = TSFreshFeatureExtractor(expansion_threshold=expansion_threshold)
    time_series_feature_extractor.min_settings_card = 0
    time_series_feature_extractor._apply_feature_threshold(settings)
    assert settings == expected_settings


def test_time_series_all_nan_column():
    time_series_feature_extractor = TSFeatureExtractor(extraction_type="efficient", augment=False)
    X_out = time_series_feature_extractor.fit_transform(X_all_nan_column)
    assert X_out.shape[0] == 4


@pytest.mark.parametrize(
    "env, n_jobs",
    [
        (["SAGEMAKER_PROGRAM", "sagemaker_serve"], 0),
        (["SAGEMAKER_PROGRAM", "train"], N_PROCESSES),
        (["key", "value"], N_PROCESSES),
    ],
)
def test_tsfresh_extractor_njobs_is_ncpus_when_non_sagemaker_serve_env(monkeypatch, env, n_jobs):
    monkeypatch.setenv(env[0], env[1])

    def mocked_extract(*args, **kwargs):
        assert (kwargs["n_jobs"]) == n_jobs
        return tsfresh.extract_features(*args, **kwargs)

    importlib.reload(sagemaker_sklearn_extension.feature_extraction.sequences)
    monkeypatch.setattr("sagemaker_sklearn_extension.feature_extraction.sequences.extract_features", mocked_extract)
    TSFeatureExtractor(extraction_type="efficient", augment=False).fit_transform(X_sequence)
