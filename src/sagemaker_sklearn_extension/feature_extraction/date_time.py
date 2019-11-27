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

from datetime import datetime
from enum import Enum

from dateutil import parser
import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted


class DateTimeProperty:
    def __init__(self, extract_func, max_, min_):
        """Contains information about a property of a datetime object

        Parameters
        ----------
        extract_func: function
            function mapping a datetime object to the property
        max_: int
            maximum value for the property
        min_: int
            minimum value for the property
        """
        self.min = min_
        self.max = max_
        self.extract_func = extract_func


class DateTimeDefinition(Enum):
    WEEK_OF_YEAR = DateTimeProperty(lambda t: t.isocalendar()[1] if isinstance(t, datetime) else np.nan, 53, 1)
    WEEKDAY = DateTimeProperty(lambda t: t.isocalendar()[2] if isinstance(t, datetime) else np.nan, 7, 1)
    YEAR = DateTimeProperty(lambda t: t.year if isinstance(t, datetime) else np.nan, None, None)
    HOUR = DateTimeProperty(lambda t: t.hour if isinstance(t, datetime) else np.nan, 23, 0)
    MONTH = DateTimeProperty(lambda t: t.month if isinstance(t, datetime) else np.nan, 12, 1)
    MINUTE = DateTimeProperty(lambda t: t.minute if isinstance(t, datetime) else np.nan, 59, 0)
    QUARTER = DateTimeProperty(lambda t: (t.month - 1) // 3 + 1 if isinstance(t, datetime) else np.nan, 4, 1)
    SECOND = DateTimeProperty(lambda t: t.second if isinstance(t, datetime) else np.nan, 59, 0)
    DAY_OF_YEAR = DateTimeProperty(lambda t: t.timetuple().tm_yday if isinstance(t, datetime) else np.nan, 366, 1)
    DAY_OF_MONTH = DateTimeProperty(lambda t: t.day if isinstance(t, datetime) else np.nan, 31, 1)


class DateTimeVectorizer(BaseEstimator, TransformerMixin):
    def __init__(self, extract=None, mode="cyclic", ignore_constant_columns=True):
        """Converts array-like data with datetime.datetime or strings describing datetime objects into numeric features

        A datetime item contains categorical information: year, month, hour, day of week, etc. This information is given
        as the output features. The encoding of these categories can be ordinal or cyclic. The cyclic encoding of an
        integer i between 0 and k consists of two floats: sin(i/k), cos(i/k). This makes sure for example that the
        months Decembers and January are encoded to vectors that are close in Euclidean distance.

        Parameters
        ----------
        extract: list of DateTimeProperty, default None
            Types of data to extract. See DateTimeDefinition class for options. If given None,
            defaults to DateTimeVectorizer.default_data
        mode: str, default cyclic
            'ordinal': each data type is outputted to a non-negative integer, as in ordinal encoding for categorical
                       data
            'cyclic': each data type is converted to two numbers in [-1,1] so that the distance between these numbers
                      is small for close items in the cyclic order (for example hour=23 is close to hour=0)
        ignore_constant_columns: bool, default True
            If True, fit will make sure the output columns are not constant in the training set.

        Attributes
        ----------
        extract_ : list of DateTimeProperty
            List of DateTimeProperty objects, each providing the necessary information for extracting a single property
            from a datetime object. The properties corresponding to this list describe the different columns of the
            output of the transform function


        Examples
        --------
        >>> from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer
        >>> import numpy as np
        >>> data = np.array([
        ...     'Jan 3th, 2018, 1:34am',
        ...     'Feb 11th, 2012, 11:34:59pm',
        ...     ]).reshape((-1, 1))
        >>> date_time = DateTimeVectorizer(mode='ordinal', ignore_constant_columns=False)
        >>> X = date_time.fit_transform(data)
        >>> print(X.shape)
        (2, 7)
        >>> print(X[0].astype(np.int))
        [   2 2018    1   34    0    0    0]
        >>> date_time = DateTimeVectorizer(mode='ordinal')
        >>> # with ignore_constant_columns=True, the minute field, which is 34 in both examples, will be filtered
        >>> X = date_time.fit_transform(data)
        >>> print(X.shape)
        (2, 6)
        >>> print(X[0].astype(np.int))
        [   2 2018    1    0    0    0]



        """
        self.extract = extract
        self.mode = mode
        self.ignore_constant_columns = ignore_constant_columns

    @staticmethod
    def _cyclic_transform(data, low, high):
        """
        Converts numeric data into 2d-cyclic.

        The conversion of a single integer into two floats makes sure that the Euclidian distance between two (output)
        values is similar to the cyclic distance between the integers. For example, hour of day is a number between 0
        and 23. The cyclic distance between the hours 0 and 23 is 1 (and not 23). After the cyclic transform, the
        transformed hour 0 will be a vector very close to that of the hour 23, and far away from that of 12.

        Parameters
        ----------
        data: np.array of numbers
        low: lower bound of the data values
        high: upper bound of the data values

        Returns
        -------
        np.array with double the dimension in the last axis

        Examples
        --------
        >>> from sagemaker_sklearn_extension.feature_extraction.date_time import DateTimeVectorizer
        >>> output = DateTimeVectorizer._cyclic_transform(np.array([[1],[2],[3],[4]]), low=1, high=4)
        >>> # up to numeric precision, the outputs should be [[0,1], [1,0], [0,-1], [-1,0]]
        >>> print(output)
        [[ 0.0000000e+00  1.0000000e+00]
         [ 1.0000000e+00  6.1232340e-17]
         [ 1.2246468e-16 -1.0000000e+00]
         [-1.0000000e+00 -1.8369702e-16]]
        >>> output = DateTimeVectorizer._cyclic_transform(np.array([[1],[2],[3],[4],[5],[6],[7],[8]]), low=1, high=8)
        >>> print(output)
        [[ 0.00000000e+00  1.00000000e+00]
         [ 7.07106781e-01  7.07106781e-01]
         [ 1.00000000e+00  6.12323400e-17]
         [ 7.07106781e-01 -7.07106781e-01]
         [ 1.22464680e-16 -1.00000000e+00]
         [-7.07106781e-01 -7.07106781e-01]
         [-1.00000000e+00 -1.83697020e-16]
         [-7.07106781e-01  7.07106781e-01]]
        """
        normalized = (data - low) * 2 * np.pi / (1 + high - low)
        sin_values = np.sin(normalized)
        cos_values = np.cos(normalized)

        shape = list(sin_values.shape)

        tmp_shape = tuple(shape + [1])
        sin_values = sin_values.reshape(tmp_shape)
        cos_values = cos_values.reshape(tmp_shape)
        ret = np.concatenate((sin_values, cos_values), axis=len(tmp_shape) - 1)

        shape[-1] *= 2
        return ret.reshape(tuple(shape))

    default_data = [
        DateTimeDefinition.WEEKDAY.value,
        DateTimeDefinition.YEAR.value,
        DateTimeDefinition.HOUR.value,
        DateTimeDefinition.MINUTE.value,
        DateTimeDefinition.SECOND.value,
        DateTimeDefinition.MONTH.value,
        DateTimeDefinition.WEEK_OF_YEAR.value,
    ]

    @staticmethod
    def _to_datetime_single(item):
        if isinstance(item, datetime):
            return item
        try:
            return parser.parse(item)
        except ValueError:
            pass
        except TypeError:
            pass

    @staticmethod
    def _to_datetime_array(X):
        """Converts np array with string or datetime into datetime or None

        Parameters
        ----------
        X : np.array
            numpy array containing data representing datetime objects

        Returns
        -------
        X : np.array
            np.array with datetime objects of the same shape of the input. Items that could not be parsed become None

        """
        X = np.vectorize(DateTimeVectorizer._to_datetime_single)(X)
        return X

    def fit(self, X, y=None):
        """Filter the extracted field so as not to contain constant columns.

        Parameters
        ----------
        X : {array-like}, datetime.datetime or str

        Notes
        -----
        If fitting with a 2d array with more than one column, any data type that is not constant in any column will
        remain. If for example, column 1 has year=1999 for all rows but column 2 has two or more possible year values,
        we will still produce an output with the year information from column 1. To avoid this, run fit on each column
        separately, and obtain a separate DateTimeVectorizer for each column

        Returns
        -------
        self : DateTimeVectorizer
        """

        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        X = np.array(X)
        X = self._to_datetime_array(X)

        if self.mode not in ["cyclic", "ordinal"]:
            raise ValueError("mode must be either cyclic or ordinal. Current value is {}".format(self.mode))

        self.extract_ = self.extract or self.default_data

        if self.ignore_constant_columns:
            new_extract = []
            for col in range(X.shape[1]):
                # convert the current column to get the different property values
                transformed = self._convert(X[:, col].reshape((-1, 1)), mode="ordinal")
                # check for constant columns
                transformed_var = np.nanvar(transformed, axis=0)
                for i, cur_var in enumerate(transformed_var):
                    if cur_var > 0 and self.extract_[i] not in new_extract:
                        new_extract.append(self.extract_[i])
            if not new_extract:
                new_extract = [self.extract_[0]]
            self.extract_ = new_extract

        return self

    def _convert(self, X, mode):
        n_cols = X.shape[1]

        cols = []

        for datetime_property in self.extract_:
            # apply the function on the datetime values in the input array, create a python list. To iterate over all
            # items we view the input as a 1d vector
            cur_conversions = list(map(datetime_property.extract_func, X.reshape((-1,))))
            # convert the list to a float32 numpy array
            cur_extract = np.array(cur_conversions, dtype=np.float32).reshape((-1, 1))
            if datetime_property.min is None:
                # the output isn't cyclic. Leave it as is
                pass
            elif mode == "ordinal":
                # the output is ordinal - shift it so the minimum value is 0
                cur_extract -= datetime_property.min
            elif mode == "cyclic":
                # the output is cyclic - need to apply the cyclic transform
                cur_extract = self._cyclic_transform(cur_extract, low=datetime_property.min, high=datetime_property.max)

            cols.append(cur_extract)

        ret = np.concatenate(cols, axis=1)
        # the return array is in 1d form. We need to reshape it to bring it back to the correct 2d form
        ret = ret.reshape((-1, n_cols * ret.shape[1]))
        return ret

    def transform(self, X, y=None):
        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        check_is_fitted(self, "extract_")

        X = np.array(X)
        X = self._to_datetime_array(X)

        return self._convert(X, self.mode)

    def _more_tags(self):
        return {"X_types": ["datetime.datetime", "string"]}
