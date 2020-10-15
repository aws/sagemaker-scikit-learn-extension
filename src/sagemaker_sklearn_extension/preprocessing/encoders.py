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

import warnings
from math import ceil
from enum import Enum

import numpy as np

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, OrdinalEncoder, KBinsDiscretizer
from sklearn.preprocessing.label import _encode, _encode_check_unknown
from sklearn.utils.validation import check_is_fitted, column_or_1d, _num_samples, check_array, check_X_y
from sagemaker_sklearn_extension.impute import RobustImputer


class ThresholdOneHotEncoder(OneHotEncoder):
    """Encode categorical integer features as a one-hot numeric array, with optional restrictions on feature encoding.

    This adds functionality to encode only if a feature appears more than ``threshold`` number of times. It also adds
    functionality to bound the number of categories per feature to ``max_categories``.

    This transformer is an extension of ``OneHotEncoder`` from the ``sklearn.preprocessing`` module.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values (default = 'auto')
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith column. The passed categories should not
          mix strings and numeric values within a single feature, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    drop : 'first' or a list/array of shape (n_features,) (default = None)
        Specifies a methodology to use to drop one of the categories per feature. This is useful in situations where
        perfectly collinear features cause problems, such as when feeding the resulting data into a neural network or
        an unregularized regression.

        - None : retain all features (the default).
        - 'first' : drop the first category in each feature. If only one category is present, the feature will be
          dropped entirely.
        - array : ``drop[i]`` is the category in feature ``X[:, i]`` that should be dropped.

    sparse : boolean (default = True)
        Will return sparse matrix if set True else will return an array.

    dtype : number type (default = np.float64)
        Desired dtype of output.

    threshold : float (default = max(10, n_features / 1000))
        The threshold for including a value in the encoding of the result. Default value is the maximum of `10` or
        `n_features / 1000` where `n_features` is the number of columns of input X. How this parameter is interpreted
        depends on whether it is more than or equal to or less than 1.

        - If `threshold` is more than or equal to one, it represents the number of times a value must appear to be
          one hot encoded in the result.

        - If `threshold` is less than one, it represents the fraction of rows which must contain the value for it to be
          one hot encoded in the result. The values is rounded up, so if `threshold` is 0.255 and there are 100 rows, a
          value must appear at least 26 times to be included.

    max_categories : int (default = 100)
        Maximum number of categories to encode per feature. If the number of observed categories is greater than
        ``max_categories``, the encoder will take the top ``max_categories`` observed categories, sorted by count.

    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting (in order of the features in X and corresponding with
        the output of ``transform``). This includes the category specified in ``drop`` (if any).

    drop_idx_ : array of shape (n_features,)
        ``drop_idx_[i]`` is the index in ``categories_[i]`` of the category to be dropped for each feature. None if all
        the transformed features will be retained.
    """

    def __init__(self, categories="auto", drop=None, sparse=True, dtype=np.float64, threshold=None, max_categories=100):
        super().__init__(categories=categories, drop=drop, sparse=sparse, dtype=dtype, handle_unknown="ignore")
        self.threshold = threshold
        self.max_categories = max_categories

    def fit(self, X, y=None):
        """Fit ThresholdOneHotEncoder to X.

        Overrides self.categories_ under the following conditions:
         - include values that appear at least ``threshold`` number of times
         - include the top ``self.max_categories`` number of categories to encode

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature.

        Returns
        -------
        self : ThresholdOneHotEncoder
        """
        super().fit(X, y)
        assert self.max_categories >= 1

        n_samples, n_features = X.shape

        if not self.threshold:
            threshold = max(10, n_samples / 1000)
        elif self.threshold >= 1:
            threshold = self.threshold
        else:
            threshold = ceil(self.threshold * n_samples)

        n_features_completely_under_threshold = 0

        for j in range(n_features):
            # get unique values and their counts
            items, counts = np.unique([X[:, j]], return_counts=True)

            # add items that appear more than threshold times
            self.categories_[j] = items[counts >= threshold].astype("O")

            if self.categories_[j].size == 0:
                n_features_completely_under_threshold += 1
                # If no category is above the threshold, then create an unknown category to prevent
                # self.transform() from raising an IndexError.
                items.sort()
                unknown_category = "{}___".format(items[-1])
                # It's important to keep the dtype of `self.categories_[j]` as 'U' here because our `unknown_category`
                # might end up being longer than any of the seen categories, and that changes the behavior of
                # the `self._transform` method.
                self.categories_[j] = np.asarray([unknown_category], dtype="U")
            elif len(self.categories_[j]) > self.max_categories:
                items_and_counts = dict(zip(items, counts))
                self.categories_[j] = np.asarray(
                    sorted(items_and_counts, key=items_and_counts.get, reverse=True)[: self.max_categories], dtype="O"
                )

        if n_features_completely_under_threshold > 0:
            times = "time" if self.threshold == 1 else "times"
            warnings.warn(
                "{} out of {} features do not have any categories appearing more than threshold={} {}.".format(
                    n_features_completely_under_threshold, n_features, self.threshold, times
                )
            )

        return self

    def fit_transform(self, X, y=None):
        self._validate_keywords()

        return super().fit_transform(X, y)

    def _more_tags(self):
        return {"X_types": ["categorical"]}


class RobustLabelEncoder(LabelEncoder):
    """Encode labels for seen and unseen labels.

    Seen labels are encoded with value between 0 and n_classes-1.  Unseen labels are encoded with
        ``self.fill_encoded_label_value`` with a default value of n_classes.

    Similar to ``sklearn.preprocessing.LabelEncoder`` with additional features.
    - ``RobustLabelEncoder`` encodes unseen values with ``fill_encoded_label_value`` or ``fill_label_value``
      if ``fill_unseen_labels=True`` for ``transform`` or ``inverse_transform`` respectively
    - ``RobustLabelEncoder`` can use predetermined labels with the parameter``labels``.

    Examples
    --------
    >>> from sagemaker_sklearn_extension.preprocessing import RobustLabelEncoder
    >>> rle = RobustLabelEncoder()
    >>> rle.fit([1, 2, 2, 6])
    RobustLabelEncoder(fill_encoded_label_value=None,
                   fill_label_value='<unseen_label>', fill_unseen_labels=True,
                   labels=None)
    >>> rle.classes_
    array([1, 2, 6])
    >>> rle.transform([1, 1, 2, 6])
    array([0, 0, 1, 2])
    >>> rle.transform([1, 1, 2, 6, 1738])
    array([ 0,  0,  1,  2, 3])
    >>> rle.inverse_transform([0, 0, 1, 2])
    array([1, 1, 2, 6])
    >>> rle.inverse_transform([-1738, 0, 0, 1, 2])
    ['<unseen_label>', 1, 1, 2, 6]

    It can also be used to transform non-numerical labels (as long as they are
    hashable and comparable) to numerical labels.

    >>> rle = RobustLabelEncoder()
    >>> rle.fit(["hot dog", "hot dog", "banana"])
    RobustLabelEncoder(fill_encoded_label_value=None,
                   fill_label_value='<unseen_label>', fill_unseen_labels=True,
                   labels=None)
    >>> list(rle.classes_)
    ['banana', 'hot dog']
    >>> rle.transform(["hot dog", "hot dog"])
    array([1, 1])
    >>> rle.transform(["banana", "llama"])
    array([0, 2])
    >>> list(rle.inverse_transform([2, 2, 1]))
    ['<unseen_label>', '<unseen_label>', 'hot dog']

    Parameters
    ----------
    labels : list of values (default = None)
        List of unique values for label encoding. Overrides ``self.classes_``.
        If ``labels`` is None, RobustLabelEncoder will automatically determine the labels.

    fill_unseen_labels : boolean (default = True)
        Whether or not to fill unseen values during transform or inverse_transform.

    fill_encoded_label_value : int (default = n_classes)
        Replacement value for unseen labels during ``transform``.
        Default value is n_classes.

    fill_label_value : str (default = '<unseen_label>')
        Replacement value for unseen encoded labels during ``inverse_transform``.

    include_unseen_class: boolean (default = False)
        Whether or not ``fill_label_value`` should be included as a class.

    Attributes
    ----------
    classes_ : array of shape (n_classes,)
        Holds the label for each class that is seen when the encoder is ``fit``.
    """

    def __init__(
        self,
        labels=None,
        fill_unseen_labels=True,
        fill_encoded_label_value=None,
        fill_label_value="<unseen_label>",
        include_unseen_class=False,
    ):
        super().__init__()
        self.labels = labels
        self.fill_unseen_labels = fill_unseen_labels
        self.fill_encoded_label_value = fill_encoded_label_value
        self.fill_label_value = fill_label_value
        self.include_unseen_class = include_unseen_class

    def fit(self, y):
        """Fit label encoder.

        Parameters
        ----------
        y : array-like of shape (n_samples,)
            Label values.

        Returns
        -------
        self : RobustLabelEncoder.
        """
        y = column_or_1d(y, warn=True)
        self.classes_ = self._check_labels_and_sort() or _encode(y)
        return self

    def _check_labels_and_sort(self):
        if not self.labels:
            return None
        if self._is_sorted(self.labels):
            return self.labels
        warnings.warn("`labels` parameter is expected to be sorted. Sorting `labels`.")
        return sorted(self.labels)

    def _is_sorted(self, iterable):
        return all(iterable[i] <= iterable[i + 1] for i in range(len(iterable) - 1))

    def fit_transform(self, y):
        """Fit label encoder and return encoded labels.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Label values.

        Returns
        -------
        y_encoded : array-like of shape [n_samples]
                    Encoded label values.
        """
        return self.fit(y).transform(y)

    def transform(self, y):
        """Transform labels to normalized encoding.

        If ``self.fill_unseen_labels`` is ``True``, use ``self.fill_encoded_label_value`` for unseen values.
        Seen labels are encoded with value between 0 and n_classes-1.  Unseen labels are encoded with
        ``self.fill_encoded_label_value`` with a default value of n_classes.

        Parameters
        ----------
        y : array-like of shape [n_samples]
            Label values.

        Returns
        -------
        y_encoded : array-like of shape [n_samples]
                    Encoded label values.
        """
        check_is_fitted(self, "classes_")
        y = column_or_1d(y, warn=True)

        # transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        if self.fill_unseen_labels:
            _, mask = _encode_check_unknown(y, self.classes_, return_mask=True)
            y_encoded = np.searchsorted(self.classes_, y)
            fill_encoded_label_value = self.fill_encoded_label_value or len(self.classes_)
            y_encoded[~mask] = fill_encoded_label_value
        else:
            _, y_encoded = _encode(y, uniques=self.classes_, encode=True)

        return y_encoded

    def inverse_transform(self, y):
        """Transform labels back to original encoding.

        If ``self.fill_unseen_labels`` is ``True``, use ``self.fill_label_value`` for unseen values.

        Parameters
        ----------
        y : numpy array of shape [n_samples]
            Encoded label values.

        Returns
        -------
        y_decoded : numpy array of shape [n_samples]
                    Label values.
        """
        check_is_fitted(self, "classes_")
        y = column_or_1d(y, warn=True)

        if y.dtype.kind not in ("i", "u"):
            try:
                y = y.astype(np.float).astype(np.int)
            except ValueError:
                raise ValueError("`y` contains values not convertible to integer.")

        # inverse transform of empty array is empty array
        if _num_samples(y) == 0:
            return np.array([])

        labels = np.arange(len(self.classes_))
        diff = np.setdiff1d(y, labels)

        if diff.size > 0 and not self.fill_unseen_labels:
            raise ValueError("y contains previously unseen labels: %s" % str(diff))

        y_decoded = [self.classes_[idx] if idx in labels else self.fill_label_value for idx in y]
        return y_decoded

    def get_classes(self):
        """Returns the values of the unencoded classes.
        If ``self.include_unseen_class`` is ``True`` include ``self.fill_label_value`` as a class.

        Returns
        -------
        classes : array of shape (n_classes,)
        """
        if self.include_unseen_class and self.fill_unseen_labels:
            return np.append(self.classes_, [self.fill_label_value])

        return self.classes_


class NALabelEncoder(BaseEstimator, TransformerMixin):
    """Encoder for transforming labels to NA values.

       Uses `RobustImputer` on 1D inputs of labels
       - Uses `is_finite_numeric` mask for encoding by default
       - Only uses the `RobustImputer` strategy `constant` and fills using `np.nan`
       - Default behavior encodes non-float and non-finite values as nan values in
          the target column of a given regression dataset

       Parameters
       ----------

       mask_function : callable -> np.array, dtype('bool') (default=None)
           A vectorized python function, accepts np.array, returns np.array
           with dtype('bool')

           For each value, if mask_function(val) == False, that value will
           be imputed. mask_function is used to create a boolean mask that determines
           which values in the input to impute.

           Use np.vectorize to vectorize singular python functions.

    """

    def __init__(self, mask_function=None):
        self.mask_function = mask_function

    def fit(self, y):
        """Fit the encoder on y.

        Parameters
        ----------
        y : {array-like}, shape (n_samples,)
            Input column, where `n_samples` is the number of samples.

        Returns
        -------
        self : NALabelEncoder
        """
        self.model_ = RobustImputer(strategy="constant", fill_values=np.nan, mask_function=self.mask_function)
        y = y.reshape(-1, 1)
        self.model_.fit(X=y)
        return self

    def transform(self, y):
        """Encode all non-float and non-finite values in y as NA values.

        Parameters
        ----------
        y : {array-like}, shape (n_samples)
            The input column to encode.

        Returns
        -------
        yt : {ndarray}, shape (n_samples,)
            The encoded input column.
        """
        check_is_fitted(self, "model_")
        y = y.reshape(-1, 1)
        return self.model_.transform(y).flatten()

    def inverse_transform(self, y):
        """Returns input column"""
        return y

    def _more_tags(self):
        return {"X_types": ["1dlabels"]}


class RobustOrdinalEncoder(OrdinalEncoder):
    """Encode categorical features as an integer array.

    The input should be a 2D, array-like input of categorical features. Each column of categorical features will be
    converted to ordinal integers. For a given column of n unique values, seen values will be mapped to integers 0 to
    n-1 and unseen values will be mapped to the integer n (or to np.nan when unknown_as_nan is True). An unseen value
    is a value that was passed in during the transform step, but not present in the fit step input.
    This encoder supports inverse_transform, transforming ordinal integers back into categorical features. Unknown
    integers are transformed to None.

    Similar to ``sklearn.preprocessing.OrdinalEncoder`` with the additional feature of handling unseen values.

    Parameters
    ----------
    categories : 'auto' or a list of lists/arrays of values.
        Categories (unique values) per feature:

        - 'auto' : Determine categories automatically from the training data.
        - list : ``categories[i]`` holds the categories expected in the ith
          column. The passed categories should not mix strings and numeric
          values, and should be sorted in case of numeric values.

        The used categories can be found in the ``categories_`` attribute.

    dtype : number type, default np.float32
        Desired dtype of output.

    unknown_as_nan : boolean, default False
        When unknown_as_nan is false, unknown values are transformed to n, where n-1 is the last category
        When unknown_as_nan is true, unknown values are transformed to np.nan

    threshold : 'auto' or float, default = 1
        The threshold for encoding a value as its own label in the result. Default value 1. If `threshold='auto'`, the
        maximum of `10` or`n_features / 1000` where `n_features` is the number of columns of input X is calculated as
        the `threshold`. How this parameter is interpreted depends on whether it is more than or equal to or less
        than 1.

        - If `threshold` is more than or equal to one, it represents the number of times a value must appear to be
          one hot encoded in the result.

        - If `threshold` is less than one, it represents the fraction of rows which must contain the value for it to be
          one hot encoded in the result. The values is rounded up, so if `threshold` is 0.255 and there are 100 rows, a
          value must appear at least 26 times to be included.

    max_categories : int or np.inf, default = np.inf
        Maximum number of categories to encode per feature. Default value is np.inf and does not place an upper bound on
        the number of categories. If the number of observed categories is greater than ``max_categories``, the encoder
        will take the top ``max_categories`` observed categories, sorted by count. All remaining values will be
        encoded as the last category. Note this means that the number of categories will be ``max_categories + 1``.
        In the case of a tie between categories, the category whose label is higher will be chosen.


    Attributes
    ----------
    categories_ : list of arrays
        The categories of each feature determined during fitting
        (in order of the features in X and corresponding with the output
        of ``transform``).

    feature_idxs_no_categories_ : list of ints
        A list of indexes of features who have no categories with a frequency
        greater than or equal to the value of ``threshold``.

    Examples
    --------
    Given a dataset with two features, we let the encoder find the unique
    values per feature and transform the data to an ordinal encoding.

    >>> from sagemaker_sklearn_extension.preprocessing import RobustOrdinalEncoder
    >>> enc = RobustOrdinalEncoder()
    >>> X = [['Male', 1], ['Female', 3], ['Female', 2]]
    >>> enc.fit(X)
    RobustOrdinalEncoder(categories='auto', dtype=<class 'numpy.float32'>)
    >>> enc.categories_
    [array(['Female', 'Male'], dtype=object), array([1, 2, 3], dtype=object)]
    >>> enc.transform([['Female', 3], ['Male', 1], ['Other', 15]])
    array([[0., 2.],
           [1., 0.],
           [2., 3.]], dtype=float32)

    >>> enc.inverse_transform([[1, 0], [0, 1]])
    array([['Male', 1],
           ['Female', 2]], dtype=object)

    >>> enc.inverse_transform([[1, 0], [0, 1], [2, 3]])
    array([['Male', 1],
           ['Female', 2],
           [None, None]], dtype=object)

    """

    def __init__(self, categories="auto", dtype=np.float32, unknown_as_nan=False, threshold=1, max_categories=np.inf):
        super(RobustOrdinalEncoder, self).__init__(categories=categories, dtype=dtype)
        self.categories = categories
        self.dtype = dtype
        self.unknown_as_nan = unknown_as_nan
        self.threshold = threshold
        self.max_categories = max_categories

    def fit(self, X, y=None):
        """Fit the RobustOrdinalEncoder to X.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to determine the categories of each feature, assuming the input parameter categories equals 'auto'

        Returns
        -------
        self

        """
        self._fit(X, handle_unknown="ignore")

        assert self.max_categories >= 1

        self.feature_idxs_no_categories_ = []

        if isinstance(self.max_categories, int) or self.threshold != 1:
            X_columns, n_samples, n_features = self._check_X(X)

            if self.threshold == "auto":
                threshold = max(10, n_samples / 1000)
            elif self.threshold >= 1:
                threshold = self.threshold
            else:
                threshold = ceil(self.threshold * n_samples)

            for i in range(n_features):
                dtype = X_columns[i].dtype
                items, counts = np.unique(X_columns[i].astype(str), return_counts=True)
                categories_to_encode = items[counts >= threshold].astype("O")
                if categories_to_encode.size == 0:
                    warnings.warn(
                        "feature at index {} does not have any categories appearing more than {} {}".format(
                            i, threshold, "time" if threshold == 1 else "times"
                        )
                    )
                    # If no category is above the threshold, create an unknown category to prevent
                    # self._transform() from raising an IndexError
                    categories_to_encode = np.array(["unknown"])
                    self.feature_idxs_no_categories_.append(i)
                if len(categories_to_encode) > self.max_categories:
                    most_freq_idxs = np.argsort(counts)[len(counts) - self.max_categories :]
                    categories_to_encode = items[most_freq_idxs]
                self.categories_[i] = np.sort(categories_to_encode.astype(dtype))

        return self

    def transform(self, X):
        """Transform X to ordinal integers.

        Parameters
        ----------
        X : array-like, shape [n_samples, n_features]
            The data to encode.

        Returns
        -------
        X_out : sparse matrix or a 2-d array
            Transformed input.

        """
        X_int, X_mask = self._transform(X, handle_unknown="ignore")
        if self.unknown_as_nan:
            # assign the unknowns np.nan
            X_int = X_int.astype(self.dtype, copy=False)
            X_int[~X_mask] = np.nan
            X_int[:, self.feature_idxs_no_categories_] = np.nan
        else:
            # assign the unknowns an integer indicating they are unknown. The largest integer is always reserved for
            # unknowns
            for col in range(X_int.shape[1]):
                mask = X_mask[:, col]
                X_int[~mask, col] = self.categories_[col].shape[0]
            X_int = X_int.astype(self.dtype, copy=False)
            X_int[:, self.feature_idxs_no_categories_] = 0
        return X_int

    def inverse_transform(self, X):
        """Convert the data back to the original representation.
        In slots where the encoding is that of an unrecognised category, the output of the inverse transform is np.nan
        for float or complex arrays, and None otherwise

        Parameters
        ----------
        X : array-like or sparse matrix, shape [n_samples, n_encoded_features]
            The transformed data.

        Returns
        -------
        X_tr : array-like, shape [n_samples, n_features]
            Inverse transformed array.

        Notes
        -----
        Most of the logic is copied from sklearn.preprocessing.OrdinalEncoder.inverse_transform. The difference is in
        handling unknown values.

        """
        check_is_fitted(self, "categories_")
        X = check_array(X, dtype="numeric", force_all_finite="allow-nan" if self.unknown_as_nan else True)

        n_samples, _ = X.shape
        n_features = len(self.categories_)

        # validate shape of passed X
        msg = "Shape of the passed X data is not correct. Expected {0} " "columns, got {1}."
        if X.shape[1] != n_features:
            raise ValueError(msg.format(n_features, X.shape[1]))

        # create resulting array of appropriate dtype
        dt = np.find_common_type([cat.dtype for cat in self.categories_], [])
        X_tr = np.empty((n_samples, n_features), dtype=dt)

        found_unknown = {}
        for i in range(n_features):
            labels = X[:, i].astype("int64", copy=False)
            known_mask = np.isfinite(X[:, i]) if self.unknown_as_nan else (labels != self.categories_[i].shape[0])
            labels *= known_mask
            X_tr[:, i] = self.categories_[i][labels]
            if not np.all(known_mask):
                found_unknown[i] = ~known_mask

        # if unknown are found cast to an object array and transform the missing values to None
        if found_unknown:
            if X_tr.dtype != object:
                X_tr = X_tr.astype(object)

            for idx, unknown_mask in found_unknown.items():
                X_tr[unknown_mask, idx] = None

        X_tr[:, self.feature_idxs_no_categories_] = None

        return X_tr


class WOEAsserts(Enum):
    ALPHA = "Regularization parameter `alpha` must be non-negative."
    BINARY = "Weight-of-Evidence encoder is only supported for binary targets."
    BINNING = "Binning strategy must be in {'uniform', 'quantile', 'kmeans'}."
    NBINS = "Number of bins must be larger than 2."
    UNSEEN_CAT = "One or more category in the data was not seen at training time."


class WOEEncoder(BaseEstimator, TransformerMixin):
    """Weight of Evidence (WoE) encoder: encodes categorical features as a numerical vector
    using weight of evidence encoding. This is only supported with binary targets. Both the
    features and the target are assumed to be free of missing values, missing values should
    be handled separately before the encoding. A binning function can be provided to handle
    numerical features which are then binned first then encoded.
    Note that the sign of the weight of evidence values depends on the order in which the
    categories of the target column are detected. This does not affect the performance of
    any supervised model applied thereafter.
    See [1] for more details on WoE.

    Parameters
    ----------
    binning: {'uniform', 'quantile', 'kmeans', None}, default=None
        What binning method to apply, no binning applied if set to None.
        This uses ScikitLearn's KBinsDiscretizer (see [2]).
        uniform: all bins in each feature have identical width.
        quantile: all bins in each feature have the same number of points.
        kmeans: values in each bin have the same nearest center of a 1D kmeans cluster.

    n_bins: int (default=10), greater than 2
        Number of bins to use when binning is applied.

    alpha: float (default = 0.5), non-negative
        Regularization value to avoid numerical errors due to division by zero in the
        computation of the weight of evidence (e.g. in case the data points corresponding
        to one category of a feature all have the same target value).

    laplace: boolean (default = False)
        If alpha is positive, adds Laplace smoothing to the computation of the weight of
        evidence.

    Example
    -------
    >>> import numpy as np
    >>> from sagemaker_sklearn_extension.preprocessing import WOEEncoder
    >>> np.random.seed(112)
    >>> N = 10
    >>> y = np.random.choice([0, 1], size=N)
    >>> y
    array([0, 1, 0, 0, 1, 0, 1, 0, 1, 0])
    >>> sex = np.random.choice(['m', 'f'], size=N)
    >>> sex
    array(['m', 'f', 'm', 'm', 'f', 'm', 'f', 'm', 'm', 'm'], dtype='<U1')
    >>> WOEEncoder().fit_transform(sex.reshape(-1, 1), y)
    array([[ 1.06087196],
           [-2.35137526],
           [ 1.06087196],
           [ 1.06087196],
           [-2.35137526],
           [ 1.06087196],
           [-2.35137526],
           [ 1.06087196],
           [ 1.06087196],
           [ 1.06087196]])
    >>> age = np.random.randint(low=25, high=95, size=N)
    >>> age
    array([54, 73, 76, 30, 53, 33, 28, 51, 62, 43])
    >>> WOEEncoder(binning='quantile', n_bins=2).fit_transform(age.reshape(-1, 1), y)
    array([[-0.74193734],
           [-0.74193734],
           [-0.74193734],
           [ 0.69314718],
           [-0.74193734],
           [ 0.69314718],
           [ 0.69314718],
           [ 0.69314718],
           [-0.74193734],
           [ 0.69314718]])

    Attributes
    ----------
    binner_: estimator trained to bin numerical data if binning is not None.

    woe_pairs_: list of pairs (codex, woe) of size n_encoded_features
        The codex has the mapping feature_value => woe_index and woe has the weight of
        evidence values.

    References
    ----------
    [1] https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html
    [2] https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.KBinsDiscretizer.html
    [3] https://en.wikipedia.org/wiki/Additive_smoothing
    """

    def __init__(self, binning=None, n_bins=10, alpha=0.5, laplace=False):
        self.binning = binning
        self.n_bins = n_bins
        self.alpha = alpha
        self.laplace = laplace

    def _woe(self, x, count_y_0, mask_y_0, beta):
        """Return the categories for a feature vector `x` as well as the corresponding
        weight of evidence value for each of those categories.

        Parameters
        ----------
        x: vector, shape (n_samples,)
            Feature vector to encode.
        count_y: list of length 2
            List of counts for the number of observations with the first (resp. the second)
            target category.
        mask_y_0: vector, shape (n_samples,)
            Mask of observations with the first target category.
        beta: float
            Value to use for Laplace Smoothing (0 if laplace is False).
        """
        cat_x = np.unique(x)
        mask_y_1 = np.logical_not(mask_y_0)
        count_y_1 = len(mask_y_0) - count_y_0

        # Computation of the Weight of Evidence for a category c in cat_x and with
        # regularization α
        #
        #   woe_c = log( { #(y==0 | c) + α / #(y==1 | c)  + α } *
        #                { #(y==1) + β / #(y==0) + β } )
        #
        # where β = 2α if laplace == True, 0 otherwise. The second factor can be computed
        # once, call it `r10` then
        #
        #   woe_c = log( r10 * ratio(c) )
        #
        # where
        #
        #   ratio(c) = { #(y==0 | c) + α } / { #(y==1 | x==c) + α }
        #

        def ratio(c):
            x_c = x == c
            # retrieve the number of (y == 0 | x == c) and same for y == 1
            y_0_c = sum(np.logical_and(mask_y_0, x_c))
            y_1_c = sum(np.logical_and(mask_y_1, x_c))
            # compute the ratio with regularization for 0 events
            return (y_0_c + self.alpha) / (y_1_c + self.alpha)

        # computation of woe possibly using Laplace smoothing (beta factor)
        r10 = (count_y_1 + beta) / (count_y_0 + beta)
        woe = np.log(r10 * np.array([ratio(c) for c in cat_x]))
        # encoder from unique values of x to index
        codex = {c: i for (i, c) in enumerate(cat_x)}
        return (codex, woe)

    def fit(self, X, y):
        """Fit Weight of Evidence encoder to `X` and `y`.

        Parameters
        ----------
        X: array-like, shape (n_samples, n_features)
            The data to encode.

        y: array-like, shape (n_samples,)
            The binary target vector.

        Returns
        -------
        self: WOEEncoder.
        """
        # Validate parameters
        if self.binning:
            assert self.binning in ("uniform", "quantile", "kmeans"), WOEAsserts.BINNING
            assert self.n_bins >= 2, WOEAsserts.NBINS
        assert self.alpha >= 0, WOEAsserts.ALPHA
        # Validate data
        X, y = check_X_y(X, y)
        # Keep track of number of features encoded
        self._dim = X.shape[1]
        # recover the target categories and check there's only two
        cat_y = np.unique(y)
        # it should be == 2 but relax to <= 2 for a single-sample test by check_estimator
        assert len(cat_y) <= 2, WOEAsserts.BINARY

        # value for laplace smoothing
        beta = 2 * self.alpha * self.laplace

        # count the number of occurrences per target class and form the mask
        # for the rows for which y==0
        mask_y_0 = y == cat_y[0]
        count_y_0 = sum(mask_y_0)

        if self.binning:
            self.binner_ = KBinsDiscretizer(n_bins=self.n_bins, strategy=self.binning, encode="ordinal")
            Xp = self.binner_.fit_transform(X)
        else:
            Xp = X
        # go over each column and compute the woe
        self.woe_pairs_ = list(map(lambda x: self._woe(x, count_y_0, mask_y_0, beta), Xp.T))
        return self

    def transform(self, X):
        """Transform each column of `X` using the Weight-of-Evidence encoding.

        Returns
        -------
        X_encoded: array, shape (n_samples, n_encoded_features)
            Array with each of the encoded columns.
        """
        # check is fitted
        check_is_fitted(self, "woe_pairs_")
        # check input
        X = check_array(X)

        if X.shape[1] != self._dim:
            raise ValueError(f"The input dimension is {X.shape[1]} instead of the expected {self._dim}")

        if self.binning:
            Xp = self.binner_.transform(X)
        else:
            Xp = X

        Xe = np.zeros(Xp.shape)
        for (i, x) in enumerate(Xp.T):
            codex, woe = self.woe_pairs_[i]
            # check that the data to encode doesn't have classes yet unseen
            assert all([e in codex.keys() for e in np.unique(x)]), WOEAsserts.UNSEEN_CAT
            # construct the encoded column by inverting the codex, if the category
            # is not recognised (not a key of the codex), a np.nan is inputted
            Xe[:, i] = np.array([woe[codex[xi]] for xi in x])

        return Xe

    def fit_transform(self, X, y):
        return self.fit(X, y).transform(X)

    def _more_tags(self):
        return {"X_types": ["categorical"], "binary_only": True, "requires_y": True}
