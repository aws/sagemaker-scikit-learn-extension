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

import pytest

from sagemaker_sklearn_extension.externals import Header


@pytest.mark.parametrize("names, col_idx, feature_idx", [(["a"], [0], [0]), (["a", "c"], [0, 2], [0, 1])])
def test_header_happy(names, col_idx, feature_idx):
    h = Header(column_names=["a", "b", "c"], target_column_name="b")
    assert h.target_column_index == 1
    assert h.as_feature_indices(names) == feature_idx
    assert h.as_column_indices(names) == col_idx
    assert h.num_features == 2
    assert h.num_columns == 3


def test_header_errors_target_missing():
    with pytest.raises(ValueError):
        Header(column_names=["a", "b"], target_column_name="c")


@pytest.mark.parametrize("column_names, target_column", [(["a", "b", "b", "c"], "c"), (["a", "b", "c", "c"], "c")])
def test_header_errors_duplicate_columns(column_names, target_column):
    with pytest.raises(ValueError):
        Header(column_names=column_names, target_column_name=target_column)


@pytest.mark.parametrize(
    "names, error_regex",
    [(["unknown"], "'unknown' is an unknown feature name"), (["b"], "'b' is the target column name.")],
)
def test_header_error_as_feature_indices(names, error_regex):
    h = Header(column_names=["a", "b", "c"], target_column_name="b")
    assert h.target_column_index == 1
    with pytest.raises(ValueError) as err:
        h.as_feature_indices(names)
        err.match(error_regex)


def test_header_error_as_column_index():
    h = Header(column_names=["a", "b", "c"], target_column_name="b")
    assert h.target_column_index == 1
    with pytest.raises(ValueError):
        h.as_column_indices(["unknown"])


def test_header_feature_column_index_order():
    h = Header(column_names=["a", "b", "c", "d"], target_column_name="c")
    assert h.feature_column_indices == [0, 1, 3]
