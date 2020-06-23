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

from sklearn.utils.testing import assert_array_equal

from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
from sagemaker_sklearn_extension.preprocessing import ThresholdOneHotEncoder
from sagemaker_sklearn_extension.preprocessing import RobustOrdinalEncoder


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
        assert_array_equal(observed_category, expected_category)

    X_observed_dense = X_observed_sparse.toarray()
    assert_array_equal(X_observed_dense, X_expected)


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
    assert_array_equal(encoder.transform(X).todense(), X_transformed_expected)


def test_threshold_encoder_with_all_columns_under_threshold():
    encoder = ThresholdOneHotEncoder(threshold=0.01)
    Xt = encoder.fit_transform(np.array([[1, 2], [1, 3]])).todense()
    assert_array_equal(np.array([[1, 1, 0], [1, 0, 1],]), Xt)


def test_threshold_encoder_with_no_columns_over_threshold():
    encoder = ThresholdOneHotEncoder(threshold=1000)
    Xt = encoder.fit_transform(np.array([[1, 2], [1, 3]])).todense()
    assert_array_equal(np.array([[0, 0], [0, 0],]), Xt)


def test_robust_label_encoder():
    enc = RobustLabelEncoder()
    enc.fit(X[:, 0])

    assert_array_equal(enc.classes_, ["apple", "banana", "hot dog"])
    assert_array_equal(enc.transform([]), [])
    assert_array_equal(enc.transform(["hot dog", "banana", "hot dog"]), [2, 1, 2])
    assert_array_equal(enc.transform(["hot dog", "llama"]), [2, 3])
    assert_array_equal(enc.inverse_transform([0, 2]), ["apple", "hot dog"])
    assert_array_equal(enc.inverse_transform([0, 10]), ["apple", "<unseen_label>"])

    assert_array_equal(enc.fit_transform(X[:, 0]), [2, 2, 0, 2, 2, 1])


def test_robust_label_encoder_error_unknown():
    with pytest.raises(ValueError):
        enc = RobustLabelEncoder(fill_unseen_labels=False)
        enc.fit(X[:, 0])
        enc.transform(["eggplant"])


def test_robust_label_encoder_inverse_transform_dtype():
    enc = RobustLabelEncoder()
    enc.fit(X[:, 0])

    assert_array_equal(enc.inverse_transform(["1.0", "2.0"]), ["banana", "hot dog"])

    with pytest.raises(ValueError):
        enc.inverse_transform(["0.", "2.b"])


@pytest.mark.parametrize("labels", (["-12", "3", "9"], ["-12.", "3.", "9."]))
def test_robust_label_encoder_sorted_labels(labels):
    enc = RobustLabelEncoder(labels=labels)
    enc.fit([labels[1], labels[0]])

    assert_array_equal(list(enc.classes_), labels)
    assert_array_equal(enc.transform([labels[2], labels[1], "173"]), [2, 1, 3])

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=labels)
    y_transformed = enc.fit_transform([labels[2], labels[1], "173"])

    assert_array_equal(list(enc.classes_), labels)
    assert_array_equal(y_transformed, [2, 1, 3])


@pytest.mark.parametrize("labels", (["-12", "9", "3"], ["-12.", "9.", "3."]))
def test_robust_label_encoder_unsorted_labels_warning(labels):
    enc = RobustLabelEncoder(labels=labels)
    with pytest.warns(UserWarning):
        enc.fit([labels[2], labels[0]])

    assert_array_equal(list(enc.classes_), sorted(labels))
    assert_array_equal(enc.transform([labels[1], labels[2], "173"]), [2, 1, 3])

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=labels)
    with pytest.warns(UserWarning):
        y_transformed = enc.fit_transform([labels[1], labels[2], "173"])

    assert_array_equal(list(enc.classes_), sorted(labels))
    assert_array_equal(y_transformed, [2, 1, 3])


def test_robust_label_encoder_fill_label_value():
    y = np.array([1, 1, 0, 1, 1])
    enc = RobustLabelEncoder(labels=[1], fill_label_value=0)
    enc.fit(y)
    y_transform = enc.transform(y)
    assert_array_equal(y_transform, [0, 0, 1, 0, 0])
    assert_array_equal(enc.inverse_transform(y_transform), y)

    # Test that fit_transform has the same behavior
    enc = RobustLabelEncoder(labels=[1], fill_label_value=0)
    y_transform = enc.fit_transform(y)
    assert_array_equal(y_transform, [0, 0, 1, 0, 0])
    assert_array_equal(enc.inverse_transform(y_transform), y)


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
    assert_array_equal(y_transform, y_expected)


def test_robust_ordinal_encoding_categories():
    encoder = RobustOrdinalEncoder()
    encoder.fit(ordinal_data)
    for i, cat in enumerate(encoder.categories_):
        assert set(cat) == set(ordinal_expected_categories_[i])


def test_robust_ordinal_encoding_transform():
    encoder = RobustOrdinalEncoder()
    encoder.fit(ordinal_data)
    test_data = np.concatenate([ordinal_data, np.array([["waffle", 1213, None]])], axis=0)
    encoded = encoder.transform(test_data)
    assert all(list((encoded[:-1] < 3).reshape((-1,))))
    assert all(list(encoded[-1] == 3))


def test_robust_ordinal_encoding_inverse_transform():
    encoder = RobustOrdinalEncoder()
    encoder.fit(ordinal_data)
    test_data = np.concatenate([ordinal_data, np.array([["waffle", 1213, None]])], axis=0)
    encoded = encoder.transform(test_data)
    reverse = encoder.inverse_transform(encoded)
    assert np.array_equal(ordinal_data, reverse[:-1])
    assert all([x is None for x in reverse[-1]])


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
