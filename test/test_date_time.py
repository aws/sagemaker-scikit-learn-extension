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
from dateutil import parser

from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer, DateTimeDefinition


data_array = [
    [parser.parse("Jan 5th, 2012, 12:34am")],
    [parser.parse("Feb 2, 2011, 2:34:04am")],
    [parser.parse("Jan 1st, 2012, 11:59:59pm")],
    [parser.parse("Dec 2th, 2012, 12:00am")],
    [parser.parse("Jan 3th, 2012, 12:34am")],
    [parser.parse("Jan 3th, 2018, 1:34am")],
]

data = np.array(data_array)


@pytest.mark.parametrize("data_shape", [(2, 3), (2, 3, 4), (2,)])
def test_cyclic_transform_outputs_correct_shape(data_shape):
    size = int(np.prod(data_shape))
    data = np.arange(size).reshape(data_shape)
    ret = DateTimeVectorizer._cyclic_transform(data, low=0, high=size - 1)

    new_shape = list(data_shape)
    new_shape[-1] *= 2
    new_shape = tuple(new_shape)
    assert ret.shape == new_shape

    ret = ret.reshape((-1, 2))
    ret = ret ** 2
    assert np.linalg.norm(np.sum(ret, axis=1) - 1) < 1e-8


@pytest.mark.parametrize("mode", ["ordinal", "cyclic"])
def test_fit_transform_works_with_non_np_input(mode):
    dtv = DateTimeVectorizer(
        mode=mode,
        extract=[
            DateTimeDefinition.HOUR.value,
            DateTimeDefinition.SECOND.value,
            DateTimeDefinition.YEAR.value,
            DateTimeDefinition.MONTH.value,
        ],
    )
    output = dtv.fit_transform(data_array)
    assert output.shape[0] == len(data_array)
    assert output.shape[1] > 1


@pytest.mark.parametrize("data_shape", [(2, 3), (2, 3, 4), (2,)])
def test_cyclic_transform_outputs_correct_cyclic_values(data_shape):
    size = int(np.prod(data_shape))
    data = np.arange(size).reshape(data_shape)
    ret = DateTimeVectorizer._cyclic_transform(data, low=0, high=size - 1)
    ret = ret.reshape((-1, 2))
    ret = ret ** 2
    assert np.linalg.norm(np.sum(ret, axis=1) - 1) < 1e-8


def test_fit_eliminates_constant_columns():
    dtv = DateTimeVectorizer(
        mode="ordinal",
        extract=[
            DateTimeDefinition.HOUR.value,
            DateTimeDefinition.SECOND.value,
            DateTimeDefinition.YEAR.value,
            DateTimeDefinition.MONTH.value,
        ],
    )
    # taking only odd items. Year and month are always the same.
    cur_data = data.reshape((-1, 2))[:, 0].reshape((-1, 1))
    dtv = dtv.fit(cur_data)
    # Year and month are constants, make sure they are out
    assert dtv.extract_ == [DateTimeDefinition.HOUR.value, DateTimeDefinition.SECOND.value]


@pytest.mark.parametrize("mode", ["ordinal", "cyclic"])
def test_fit_eliminates_constant_columns_multicol_input(mode):
    # set up data. Properties:
    # Hour: Constant thrghout - eliminate
    # Year: Constant in both, but has different value accross columns - should eliminate
    # Month: Constant in column 2, not in 1 - should not eliminate
    # Day of month: not constant in both columns - should not eliminate
    col1 = [
        parser.parse("Jan 5th, 2012"),
        parser.parse("Feb 2, 2012"),
        parser.parse("Jan 1st, 2012"),
    ]
    col2 = [
        parser.parse("Dec 2th, 2013"),
        parser.parse("Dec 3th, 2013"),
        parser.parse("Dec 3th, 2013"),
    ]

    cur_data = np.array([col1, col2]).T

    dtv = DateTimeVectorizer(
        mode=mode,
        extract=[
            DateTimeDefinition.HOUR.value,
            DateTimeDefinition.DAY_OF_MONTH.value,
            DateTimeDefinition.YEAR.value,
            DateTimeDefinition.MONTH.value,
        ],
    )
    # taking only odd items. Year and month are always the same.
    dtv = dtv.fit(cur_data)
    # Year and month are constants, make sure they are out
    assert dtv.extract_ == [DateTimeDefinition.DAY_OF_MONTH.value, DateTimeDefinition.MONTH.value]


def test_transform_categorical():
    extract_keys = [k for k in dir(DateTimeDefinition) if not k.startswith("_")]
    extract = [DateTimeDefinition.__dict__[k].value for k in extract_keys]
    dtv = DateTimeVectorizer(mode="ordinal", extract=extract, ignore_constant_columns=False)
    dtv.fit(data)
    output = dtv.transform(data)

    assert np.all(output >= 0)

    loc_year = extract_keys.index("YEAR")
    assert_array_equal(output[:, loc_year], np.array([2012, 2011, 2012, 2012, 2012, 2018]))

    loc_month = extract_keys.index("MONTH")
    assert_array_equal(output[:, loc_month], np.array([0, 1, 0, 11, 0, 0]))


def test_transform_cyclic_leaves_year():
    extract_keys = [k for k in dir(DateTimeDefinition) if not k.startswith("_")]
    extract = [DateTimeDefinition.__dict__[k].value for k in extract_keys]

    dtv = DateTimeVectorizer(mode="cyclic", extract=extract, ignore_constant_columns=False)
    dtv.fit(data)
    output = dtv.transform(data)

    loc_year = extract_keys.index("YEAR")
    loc_year *= 2
    assert_array_equal(output[:, loc_year], np.array([2012, 2011, 2012, 2012, 2012, 2018]))

    assert output.shape[1] == len(extract) * 2 - 1


def test_fit_transform_cyclic_leaves_year():
    extract_keys = [k for k in dir(DateTimeDefinition) if not k.startswith("_")]
    extract = [DateTimeDefinition.__dict__[k].value for k in extract_keys]

    dtv = DateTimeVectorizer(mode="cyclic", extract=extract, ignore_constant_columns=False)
    output = dtv.fit_transform(data)

    loc_year = extract_keys.index("YEAR")
    loc_year *= 2
    assert_array_equal(output[:, loc_year], np.array([2012, 2011, 2012, 2012, 2012, 2018]))

    assert output.shape[1] == len(dtv.extract_) * 2 - 1


def test_fit_transform_accepts_mixed_str_datetime():
    cur_data_array = data_array + [["Feb 12th, 15:33, 2011"], ["Nov 5th, 1am, 1975"], [432], [None], ["Feb 45th, 2018"]]

    dtv = DateTimeVectorizer(mode="ordinal")
    processed = dtv.fit_transform(cur_data_array)
    year_location = dtv.extract_.index(DateTimeDefinition.YEAR.value)
    assert processed[0, year_location] == 2012
    assert processed[-4, year_location] == 1975
    assert np.isnan(processed[-3, year_location])
    assert np.isnan(processed[-2, year_location])
    assert np.isnan(processed[-1, year_location])

    dtv = DateTimeVectorizer(mode="cyclic")
    processed = dtv.fit_transform(cur_data_array)
    assert all(np.isnan(processed[-1]))
    assert not any(np.isnan(processed[-4]))
    assert not any(np.isnan(processed[0]))
