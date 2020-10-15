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
import scipy.sparse as sp

from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sagemaker_sklearn_extension.preprocessing import RobustOrdinalEncoder
from sagemaker_sklearn_extension.preprocessing import WOEEncoder


X = np.array([["hot dog", 1], ["hot dog", 1], ["apple", 2], ["hot dog", 3], ["hot dog", 1], ["banana", 3]])
X_expected_categories_ = [np.array(["hot dog"], dtype=object), np.array(["1", "3"], dtype=object)]
X_expected_max_one_categories_ = [np.array(["hot dog"], dtype=object), np.array(["1"], dtype=object)]
X_expected_dense = [
    [1.0, 1.0, 0.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 0.0],
    [1.0, 0.0, 1.0],
    [1.0, 1.0, 0.0],
    [0.0, 0.0, 1.0],
]
X_expected_max_one_dense = [[1.0, 1.0], [1.0, 1.0], [0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 0.0]]


ordinal_data = np.array(
    [
        ["hot dog", 1, "?"],
        ["hot dog", 1, "a"],
        ["apple", 2, "b"],
        ["hot dog", 3, "a"],
        ["hot dog", 1, "b"],
        ["banana", 3, "a"],
    ]
)
ordinal_expected_categories_ = [{"hot dog", "apple", "banana"}, {"1", "2", "3"}, {"?", "a", "b"}]


@pytest.mark.parametrize(
    ["X", "X_expected_categories", "X_expected", "max_categories", "threshold"],
    [
        (X, X_expected_categories_, X_expected_dense, 100, 2),
        (X, X_expected_categories_, X_expected_dense, 100, 1.5 / X.shape[0]),
        (X, X_expected_max_one_categories_, X_expected_max_one_dense, 1, 2),
    ],
)
def test_threshold_encoder(X, X_expected_categories, X_expected, max_categories, threshold):
    enc = ThresholdOneHotEncoder(threshold=threshold, max_categories=max_categories)
    X_observed_sparse = enc.fit_transform(X)
    assert isinstance(X_observed_sparse, sp.csr_matrix)

    assert len(enc.categories_) == len(X_expected_categories)
    for observed_category, expected_category in zip(enc.categories_, X_expected_categories):
        np.testing.assert_array_equal(observed_category, expected_category)

    X_observed_dense = X_observed_sparse.toarray()
    np.testing.assert_array_equal(X_observed_dense, X_expected)


X_int_column_under_threshold = np.array([[1, "1"], [2, "1"], [3, "1"],])
X_str_column_under_threshold = np.array([["1", "1"], ["2", "1"], ["3", "1"],])
X_float_column_under_threshold = np.array([[1.1, "1"], [2.1, "1"], [3.1, "1"],])
X_mixed_column_under_threshold = np.array([[1, "1"], ["2", "1"], [np.nan, "1"],])
X_column_under_threshold_expected_dense = np.array([[0.0, 1.0], [0.0, 1.0], [0.0, 1.0],])


@pytest.mark.parametrize(
    ["X", "X_transformed_expected"],
    [
        # Test that when one column contains no classes over threshold, entries are encoded as zeros.
        (X_int_column_under_threshold, X_column_under_threshold_expected_dense),
        (X_str_column_under_threshold, X_column_under_threshold_expected_dense),
        (X_float_column_under_threshold, X_column_under_threshold_expected_dense),
        (X_mixed_column_under_threshold, X_column_under_threshold_expected_dense),
        # Test that when whole matrix contains no classes over threshold, all entries are encoded as zeros.
        (np.array([[1, "a"], [2, "b"]]), np.array([[0.0, 0.0], [0.0, 0.0]])),
        (np.array([1, 2]).reshape(-1, 1), np.array([0.0, 0.0]).reshape(-1, 1)),
    ],
)
def test_threshold_encoder_with_a_column_under_threshold(X, X_transformed_expected):
    encoder = ThresholdOneHotEncoder(threshold=2)
    encoder.fit(X)
    np.testing.assert_array_equal(encoder.transform(X).todense(), X_transformed_expected)


def test_threshold_encoder_with_all_columns_under_threshold():
    encoder = ThresholdOneHotEncoder(threshold=0.01)
    Xt = encoder.fit_transform(np.array([[1, 2], [1, 3]])).todense()
    np.testing.assert_array_equal(np.array([[1, 1, 0], [1, 0, 1],]), Xt)


def test_threshold_encoder_with_no_columns_over_threshold():
    encoder = ThresholdOneHotEncoder(threshold=1000)
    Xt = encoder.fit_transform(np.array([[1, 2], [1, 3]])).todense()
    np.testing.assert_array_equal(np.array([[0, 0], [0, 0],]), Xt)


def test_robust_label_encoder():
    enc = RobustLabelEncoder()
    enc.fit(X[:, 0])

    np.testing.assert_array_equal(enc.classes_, ["apple", "banana", "hot dog"])
    np.testing.assert_array_equal(enc.get_classes(), ["apple", "banana", "hot dog"])
    np.testing.assert_array_equal(enc.transform([]), [])
    np.testing.assert_array_equal(enc.transform(["hot dog", "banana", "hot dog"]), [2, 1, 2])
    np.testing.assert_array_equal(enc.transform(["hot dog", "llama"]), [2, 3])
    np.testing.assert_array_equal(enc.inverse_transform([0, 2]), ["apple", "hot dog"])
    np.testing.assert_array_equal(enc.inverse_transform([0, 10]), ["apple", "<unseen_label>"])

    np.testing.assert_array_equal(enc.fit_transform(X[:, 0]), [2, 2, 0, 2, 2, 1])


def test_robust_label_encoder_error_unknown():
    with pytest.raises(ValueError):
        enc = RobustLabelEncoder(fill_unseen_labels=False)
        enc.fit(X[:, 0])
        np.testing.assert_array_equal(enc.get_classes(), ["apple", "banana", "hot dog"])
        enc.transform(["eggplant"])


def test_robust_label_encoder_inverse_transform_dtype():
    enc = RobustLabelEncoder()
    enc.fit(X[:, 0])

    np.testing.assert_array_equal(enc.inverse_transform(["1.0", "2.0"]), ["banana", "hot dog"])

    with pytest.raises(ValueError):
        enc.inverse_transform(["0.", "2.b"])


@pytest.mark.parametrize("labels", (["-12", "3", "9"], ["-12.", "3.", "9."]))
def test_robust_label_encoder_sorted_labels(labels):
    enc = RobustLabelEncoder(labels=labels)
    enc.fit([labels[1], labels[0]])

    np.testing.assert_array_equal(list(enc.classes_), labels)
    np.testing.assert_array_equal(enc.get_classes(), labels)
    np.testing.assert_array_equal(enc.transform([labels[2], labels[1], "173"]), [2, 1, 3])

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=labels)
    y_transformed = enc.fit_transform([labels[2], labels[1], "173"])

    np.testing.assert_array_equal(list(enc.classes_), labels)
    np.testing.assert_array_equal(y_transformed, [2, 1, 3])


@pytest.mark.parametrize("labels", (["-12", "9", "3"], ["-12.", "9.", "3."]))
def test_robust_label_encoder_unsorted_labels_warning(labels):
    enc = RobustLabelEncoder(labels=labels)
    with pytest.warns(UserWarning):
        enc.fit([labels[2], labels[0]])

    np.testing.assert_array_equal(list(enc.classes_), sorted(labels))
    np.testing.assert_array_equal(enc.get_classes(), sorted(labels))
    np.testing.assert_array_equal(enc.transform([labels[1], labels[2], "173"]), [2, 1, 3])

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=labels)
    with pytest.warns(UserWarning):
        y_transformed = enc.fit_transform([labels[1], labels[2], "173"])

    np.testing.assert_array_equal(list(enc.classes_), sorted(labels))
    np.testing.assert_array_equal(y_transformed, [2, 1, 3])

    # Test fill_label_value is not sorted when include_unseen_class is True
    enc = RobustLabelEncoder(labels=labels, fill_label_value="-99", include_unseen_class=True)
    with pytest.warns(UserWarning):
        enc.fit([labels[2], labels[0]])
    np.testing.assert_array_equal(enc.get_classes(), sorted(labels) + ["-99"])


def test_robust_label_encoder_fill_label_value():
    y = np.array([1, 1, 0, 1, 1])
    enc = RobustLabelEncoder(labels=[1], fill_label_value=0, include_unseen_class=True)
    enc.fit(y)
    np.testing.assert_array_equal(enc.get_classes(), [1, 0])
    y_transform = enc.transform(y)
    np.testing.assert_array_equal(y_transform, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(enc.inverse_transform(y_transform), y)

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=[1], fill_label_value=0)
    y_transform = enc.fit_transform(y)
    np.testing.assert_array_equal(enc.get_classes(), [1])
    np.testing.assert_array_equal(y_transform, [0, 0, 1, 0, 0])
    np.testing.assert_array_equal(enc.inverse_transform(y_transform), y)


@pytest.mark.parametrize(
    "y, y_expected",
    [
        (np.array([np.inf, 0.1, 0.2]), np.array([np.nan, 0.1, 0.2])),
        (np.array(["1.1", "hello", "1"]), np.array([1.1, np.nan, 1])),
    ],
)
def test_na_label_encoder(y, y_expected):
    na_label_encoder = NALabelEncoder()
    na_label_encoder.fit(y)
    y_transform = na_label_encoder.transform(y)
    np.testing.assert_array_equal(y_transform, y_expected)


@pytest.mark.parametrize(
    "threshold, expected", ([1, ordinal_expected_categories_], [2, [{"hot dog"}, {"1", "3"}, {"a", "b"}]])
)
def test_robust_ordinal_encoding_categories(threshold, expected):
    encoder = RobustOrdinalEncoder(threshold=threshold)
    encoder.fit(ordinal_data)
    for i, cat in enumerate(encoder.categories_):
        assert set(cat) == set(expected[i])


@pytest.mark.parametrize("unknown_as_nan", (True, False))
def test_robust_ordinal_encoding_transform(unknown_as_nan):
    encoder = RobustOrdinalEncoder(unknown_as_nan=unknown_as_nan)
    encoder.fit(ordinal_data)
    test_data = np.concatenate([ordinal_data, np.array([["waffle", 1213, np.nan]])], axis=0)
    encoded = encoder.transform(test_data)
    assert all(list((encoded[:-1] < 3).reshape((-1,))))
    if unknown_as_nan:
        assert all(list(np.isnan(encoded[-1])))
    else:
        assert all(list(encoded[-1] == 3))


def test_robust_ordinal_encoding_transform_threshold():
    # Test where some categories are below the threshold
    encoder = RobustOrdinalEncoder(threshold=2)
    encoder.fit(ordinal_data)
    encoded = encoder.transform(ordinal_data)
    assert all(list(encoded[:, 0] < 2))
    assert all(list((encoded[:, 1:] < 3).reshape((-1,))))

    # Test where some categories are below the threshold and new categories are introduced in transformation
    test_data = np.concatenate([ordinal_data, np.array([["waffle", 1213, np.nan]])], axis=0)
    encoded = encoder.transform(test_data)
    assert all(list(encoded[:, 0] < 2))
    assert all(list((encoded[:, 1:] < 3).reshape((-1,))))

    # Test where all categories are below the threshold
    encoder = RobustOrdinalEncoder(threshold=10)
    encoder.fit(ordinal_data)
    assert len(encoder.feature_idxs_no_categories_) == 3
    encoded = encoder.transform(test_data)
    assert np.all(encoded == 0)


def test_robust_ordinal_encoding_transform_max_categories():
    # Test where number of categories is much larger than max_categories
    data = np.array([[i for i in range(200)] + [i for i in range(150)] + [i for i in range(100)]]).T
    encoder = RobustOrdinalEncoder(max_categories=100)
    encoder.fit(data)
    assert len(encoder.categories_[0]) == 100
    assert all(list(encoder.categories_[0] <= 100))
    encoded = encoder.transform(data)
    cats, frequencies = np.unique(encoded, return_counts=True)
    assert len(cats) == encoder.max_categories + 1
    assert sum(frequencies == 3) == 100

    # Test where number of categories is equal to max categories
    encoder = RobustOrdinalEncoder(max_categories=2)
    encoder.fit(np.array([["x", "y"], ["y", "x"]]))
    assert len(encoder.categories_[0]) == 2
    assert len(encoder.categories_[1]) == 2
    encoded = encoder.transform([["x", "y"], ["z", "z"]])
    assert np.all(encoded[1] == 2)
    assert np.all(encoded[0] == [0, 1])


@pytest.mark.parametrize("unknown_as_nan", (True, False))
def test_robust_ordinal_encoding_inverse_transform(unknown_as_nan):
    encoder = RobustOrdinalEncoder(unknown_as_nan=unknown_as_nan)
    encoder.fit(ordinal_data)
    test_data = np.concatenate([ordinal_data, np.array([["waffle", 1213, None]])], axis=0)
    encoded = encoder.transform(test_data)
    reverse = encoder.inverse_transform(encoded)
    assert np.array_equal(ordinal_data, reverse[:-1])
    assert all([x is None for x in reverse[-1]])

    # Test where some categories are below the threshold
    encoder = RobustOrdinalEncoder(unknown_as_nan=unknown_as_nan, threshold=2)
    encoder.fit(ordinal_data)
    encoded = encoder.transform(test_data)
    reverse = encoder.inverse_transform(encoded)
    assert sum([i is None for i in reverse[:, 0]]) == 3
    assert sum([i is None for i in reverse[:, 1]]) == 2
    assert sum([i is None for i in reverse[:, 2]]) == 2

    # Test where all categories are below the threshold
    encoder = RobustOrdinalEncoder(unknown_as_nan=unknown_as_nan, threshold=10)
    encoder.fit(ordinal_data)
    encoded = encoder.transform(test_data)
    reverse = encoder.inverse_transform(encoded)
    assert sum(([i is None for i in reverse.flatten()])) == reverse.size


def test_robust_ordinal_encoding_inverse_transform_floatkeys():
    encoder = RobustOrdinalEncoder()
    data = np.arange(9).astype(np.float32).reshape((3, 3))
    encoder.fit(data)
    test_data = data + 3
    encoded = encoder.transform(test_data)
    reverse = encoder.inverse_transform(encoded)
    assert reverse.dtype == object
    assert np.array_equal(data[1:], reverse[:-1])
    assert all([x is None for x in reverse[-1]])


# first 50 rows of titanic dataset
titanic_y = np.array(
    [
        0,
        1,
        1,
        1,
        0,
        0,
        0,
        0,
        1,
        1,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        1,
        0,
        1,
        0,
        1,
        1,
        1,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        1,
        1,
        0,
        0,
        0,
        1,
        0,
        0,
        1,
        0,
        0,
        0,
        1,
        1,
        0,
        0,
        1,
        0,
        0,
    ]
)
titanic_pclass = np.array(
    [
        3,
        1,
        3,
        1,
        3,
        3,
        1,
        3,
        3,
        2,
        3,
        1,
        3,
        3,
        3,
        2,
        3,
        2,
        3,
        3,
        2,
        2,
        3,
        1,
        3,
        3,
        3,
        1,
        3,
        3,
        1,
        1,
        3,
        2,
        1,
        1,
        3,
        3,
        3,
        3,
        3,
        2,
        3,
        2,
        3,
        3,
        3,
        3,
        3,
        3,
    ]
)
titanic_fare = np.array(
    [
        7.25,
        71.2833,
        7.925,
        53.1,
        8.05,
        8.4583,
        51.8625,
        21.075,
        11.1333,
        30.0708,
        16.7,
        26.55,
        8.05,
        31.275,
        7.8542,
        16.0,
        29.125,
        13.0,
        18.0,
        7.225,
        26.0,
        13.0,
        8.0292,
        35.5,
        21.075,
        31.3875,
        7.225,
        263.0,
        7.8792,
        7.8958,
        27.7208,
        146.5208,
        7.75,
        10.5,
        82.1708,
        52.0,
        7.2292,
        8.05,
        18.0,
        11.2417,
        9.475,
        21.0,
        7.8958,
        41.5792,
        7.8792,
        8.05,
        15.5,
        7.75,
        21.6792,
        17.8,
    ]
)
titanic_age = np.array(
    [
        22.0,
        38.0,
        26.0,
        35.0,
        35.0,
        np.nan,
        54.0,
        2.0,
        27.0,
        14.0,
        4.0,
        58.0,
        20.0,
        39.0,
        14.0,
        55.0,
        2.0,
        np.nan,
        31.0,
        np.nan,
        35.0,
        34.0,
        15.0,
        28.0,
        8.0,
        38.0,
        np.nan,
        19.0,
        np.nan,
        np.nan,
        40.0,
        np.nan,
        np.nan,
        66.0,
        28.0,
        42.0,
        np.nan,
        21.0,
        18.0,
        14.0,
        40.0,
        27.0,
        np.nan,
        3.0,
        19.0,
        np.nan,
        np.nan,
        np.nan,
        np.nan,
        18.0,
    ]
)


def test_woe_basic_comparison_skcontrib_no_smoothing():
    SK_CONTRIB_0 = np.array([0.24116206, 0.26966357, 0.75198768])
    enc = WOEEncoder(alpha=0)
    xe = enc.fit_transform(titanic_pclass.reshape(-1, 1), titanic_y)
    uv = np.sort(np.abs(np.unique(xe)))
    assert np.allclose(uv, SK_CONTRIB_0)


def test_woe_basic_comparison_skcontrib_laplace():
    SK_CONTRIB_L = np.array([0.23180161, 0.26289463, 0.68378674])
    enc = WOEEncoder(alpha=0.5, laplace=True)
    xe = enc.fit_transform(titanic_pclass.reshape(-1, 1), titanic_y)
    uv = np.sort(np.abs(np.unique(xe)))
    assert np.allclose(uv, SK_CONTRIB_L)


def test_woe_binning_quantile():
    enc = WOEEncoder(binning="quantile", n_bins=4)
    age = titanic_age.copy()
    age[np.isnan(age)] = np.median(age[~np.isnan(age)])
    xe = enc.fit_transform(age.reshape(-1, 1), titanic_y)
    print(np.unique(xe))
    assert len(np.unique(xe)) == 4


def test_woe_binning_uniform():
    enc = WOEEncoder(binning="uniform", n_bins=5, alpha=0.5)
    xe = enc.fit_transform(titanic_fare.reshape(-1, 1), titanic_y)
    assert len(np.unique(xe)) > 3


def test_woe_multi_cols():
    age = titanic_age.copy()
    age[np.isnan(age)] = np.median(age[~np.isnan(age)])
    X = np.vstack((age, titanic_fare)).T
    enc = WOEEncoder(binning="quantile", n_bins=4)
    Xe = enc.fit_transform(X, titanic_y)
    assert len(np.unique(Xe[:, 0])) == 4
    assert len(np.unique(Xe[:, 1])) == 4
