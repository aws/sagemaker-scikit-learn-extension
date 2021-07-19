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
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute

from sagemaker_sklearn_extension.preprocessing.data import RobustStandardScaler


class TSFeatureExtractor(BaseEstimator, TransformerMixin):
    """Wrap TSFlattener and TSFreshFeatureExtractor to extract time series features from multiple sequence columns.

    The input is an array where rows are observations and columns are sequence features. Each sequence feature is a
    string containing a sequence of comma-separate values.

    For each column, TSFlattener extracts numerical values from the strings and returns a list of np.arrays as output,
    and then TSFreshFeatureExtractor extracts time series features from each list. The outputs from each column are then
    stacked horizontally into a single array.

    Examples of features are the mean, median, kurtosis, and autocorrelation of each sequence. The full list of
    extracted features can be found at https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

    Any value in the input strings that can't be turned into a finite float is converted to a np.nan.

    See TSFlattener and TSFreshFeatureExtractor for more details.

    Parameters
    ----------
    max_allowed_length : int (default = 10000)
        Maximum allowed length of an input sequence. If the length of a sequence is greater than ``max_allowed_length``,
        the transformer will truncate its beginning or end as indicated by ``trim_beginning``.

    trim_beginning : bool (default = True)
        If a sequence length exceeds ``max_allowed_length``, trim its start and only keep the last `max_allowed_length``
        values if ``trim_beginning`` = True, otherwise trim the end and only keep the first max_allowed_length values.

    augment : boolean (default=False):
        Whether to append the tsfresh features to the original data (if True),
        or output only the extracted tsfresh features (if False).
        If True, also pad shorter sequences (if any) in the original data with np.nans, so that all sequences
        match the length of the longest sequence, and interpolate them as indicated by ``interpolation_method``.

    interpolation_method : {'linear', 'fill', 'zeroes', 'hybrid', None} (default='hybrid')
        'linear': linear interpolation
        'fill': forward fill to complete the sequences; then, backfill in case of NaNs at the start of the sequence.
        'zeroes': pad with zeroes
        'hybrid': replace with zeroes any NaNs at the end or start of the sequences, and forward fill NaNs in between
         None: no interpolation

    extraction_type : {'minimal', 'efficient', 'all'} (default='efficient')
        Control the number of features extracted from tsfresh.
        'minimal': most of the feature extractors are disabled and only a small subset is used
        'efficient': extract 781 tsfresh features, namely all of them except for the expensive to compute ones
        'all': extract all 787 tsfresh features

    Examples
    --------
    >>> from sagemaker_sklearn_extension.feature_extraction.sequences import TSFeatureExtractor
    >>> import numpy as np
    >>> data = [["1,, 3, 44", "3, 4, 5, 6"], ["11, 111", "1, 1, 2, 2"], ["NaN, , 1, NaN", "2, 1, 3, 2"]]
    >>> ts_pipeline = TSFeatureExtractor(augment=True)
    >>> X = ts_pipeline.fit_transform(data)
    >>> print(X.shape)
    (3, 1570)
    >>> ts_pipeline = TSFeatureExtractor(augment=False)
    >>> X = ts_pipeline.fit_transform(data)
    >>> print(X.shape)
    (3, 1562)
    """

    def __init__(
        self,
        max_allowed_length=10000,
        trim_beginning=True,
        augment=False,
        interpolation_method="hybrid",
        extraction_type="efficient",
    ):
        super().__init__()
        if max_allowed_length <= 0:
            raise ValueError(f"{max_allowed_length} must be positive.\n")
        self.max_allowed_length = max_allowed_length
        self.trim_beginning = trim_beginning
        self.augment = augment
        self.interpolation_method = interpolation_method
        self.extraction_type = extraction_type

    def fit(self, X, y=None):
        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        ts_flattener = TSFlattener(max_allowed_length=self.max_allowed_length, trim_beginning=self.trim_beginning)
        tsfresh_feature_extractors = []
        for sequence_column in X.T:
            numeric_sequences = ts_flattener.transform(sequence_column.reshape(-1, 1))
            tsfresh_feature_extractor = TSFreshFeatureExtractor(
                augment=self.augment,
                interpolation_method=self.interpolation_method,
                extraction_type=self.extraction_type,
            )
            tsfresh_feature_extractor.fit(numeric_sequences)
            tsfresh_feature_extractors.append(tsfresh_feature_extractor)
        self.tsfresh_feature_extractors_ = tsfresh_feature_extractors
        return self

    def transform(self, X, y=None):
        """Apply TSFlattener followed by TSFreshFeatureExtractor to each sequence column in X.

        Parameters
        ----------
        X : np.array (each column is a list of strings)

        Returns
        -------
        X : np.array (all values are numerical)

        """
        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        check_is_fitted(self, "tsfresh_feature_extractors_")
        ts_flattener = TSFlattener(max_allowed_length=self.max_allowed_length, trim_beginning=self.trim_beginning)
        sequences_with_features = []
        for id_column, sequence_column in enumerate(X.T):
            numeric_sequences = ts_flattener.transform(sequence_column.reshape(-1, 1))
            X_with_features = self.tsfresh_feature_extractors_[id_column].transform(numeric_sequences)
            sequences_with_features.append(X_with_features)
        X = np.hstack(sequences_with_features)
        return X

    def _more_tags(self):
        return {"X_types": ["string"], "allow_nan": True}


class TSFlattener(BaseEstimator, TransformerMixin):
    """Convert lists of strings of varying length into an np.array.

    The input is a collection of lists of strings, with each string containing a sequence of comma-separate numbers.
    TSFlattener extracts numerical values from these strings and returns a list of np.arrays as output.

    Any missing values (represented either by a NaN, inf or empty string) are converted to np.nans.
    Any entire sequence that cannot be parsed (represented by a None) is converted to a np.nan.

    Parameters
    ----------
    max_allowed_length : int (default = 10000)
        Maximum allowed length of an input sequence. If the length of a sequence is greater than ``max_allowed_length``,
        the transformer will truncate its beginning or end as indicated by ``trim_beginning``.

    trim_beginning : bool (default = True)
        If a sequence length exceeds ``max_allowed_length``, trim its start and only keep the last `max_allowed_length``
        values if ``trim_beginning`` = True, otherwise trim the end and only keep the first max_allowed_length values.


    Examples
    --------
    >>> from sagemaker_sklearn_extension.feature_extraction.sequences import TSFlattener
    >>> import numpy as np
    >>> data = [["1,, 3, 44"], ["11, 111"], ["NaN, , 1, NaN"]]
    >>> ts_flattener = TSFlattener()
    >>> X = ts_flattener.transform(data)
    >>> print(len(X))
    3
    >>> print(X)
    [array([ 1., nan,  3., 44.]), array([ 11., 111.]), array([nan, nan,  1., nan])]
    >>> ts_flattener = TSFlattener(max_allowed_length=2, trim_beginning=True)
    >>> X = ts_flattener.transform(data)
    >>> print(X)
    [array([ 3., 44.]), array([ 11., 111.]), array([ 1., nan])]
    """

    def __init__(self, max_allowed_length=10000, trim_beginning=True):
        super().__init__()
        if max_allowed_length <= 0:
            raise ValueError(f"{max_allowed_length} must be positive.\n")
        self.max_allowed_length = max_allowed_length
        self.trim_beginning = trim_beginning

    def fit(self, X, y=None):
        check_array(X, dtype=None, force_all_finite="allow-nan")
        return self

    def transform(self, X, y=None):
        """Extract numerical values from strings of comma-separated numbers and returns an np.array.

        Anything that can't be turned into a finite float is converted to a np.nan.

        Parameters
        ----------
        X : lists of strings

        Returns
        -------
        X : List of np.arrays

        """
        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        # Parse the input strings
        numeric_sequences = self._convert_to_numeric(X)
        return numeric_sequences

    def _convert_to_numeric(self, X):
        numeric_sequences = []
        for string_sequence in X:
            if len(string_sequence) != 1:
                raise ValueError(
                    f"TSFlattener can process a single sequence column at a time, "
                    f"but it was given {len(string_sequence)} sequence columns.\n"
                )
            numeric_sequence = []
            if string_sequence[0] is not None:
                for s in string_sequence[0].split(","):
                    # Turn anything that can't be converted to a finite float to np.nan
                    try:
                        s = float(s)
                    except ValueError:
                        s = np.nan
                    if np.isinf(s):
                        s = np.nan
                    numeric_sequence.append(s)
            else:
                numeric_sequence.append(np.nan)
            numeric_sequence = self._truncate_sequence(numeric_sequence)
            numeric_sequences.append(numeric_sequence)
        # Convert to list of np.arrays
        numeric_sequences = [np.array(sequence) for sequence in numeric_sequences]
        return numeric_sequences

    def _truncate_sequence(self, numeric_sequence):
        if self.trim_beginning:
            numeric_sequence = numeric_sequence[-self.max_allowed_length :]
        else:
            numeric_sequence = numeric_sequence[: self.max_allowed_length]
        return numeric_sequence

    def _more_tags(self):
        return {"X_types": ["string"], "allow_nan": True}


class TSFreshFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features computed by tsfresh from each input array or list and append them to the input
    array (if augment = True) or return the extracted features alone (if augment = False).

    Examples of these features are the mean, median, kurtosis, and autocorrelation of each sequence. The full list
    of extracted features can be found at https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

    Any np.nans in the input arrays/lists are dropped before extracting the tsfresh features.

    Parameters
    ----------
    augment : boolean (default=False):
        Whether to append the tsfresh features to the original data (if True),
        or output only the extracted tsfresh features (if False).
        If True, also pad shorter sequences (if any) in the original data with np.nans, so that all sequences
        match the length of the longest sequence, and interpolate them as indicated by ``interpolation_method``.

    interpolation_method : {'linear', 'fill', 'zeroes', 'hybrid', None} (default='hybrid')
        'linear': linear interpolation
        'fill': forward fill to complete the sequences; then, backfill in case of NaNs at the start of the sequence.
        'zeroes': pad with zeroes
        'hybrid': replace with zeroes any NaNs at the end or start of the sequences, and forward fill NaNs in between
         None: no interpolation

    extraction_type : {'minimal', 'efficient', 'all'} (default='efficient')
        Control the number of features extracted from tsfresh.
        'minimal': most of the feature extractors are disabled and only a small subset is used
        'efficient': extract 781 tsfresh features, namely all of them except for the expensive to compute ones
        'all': extract all 787 tsfresh features

    Attributes
    ----------
    self.robust_standard_scaler_ : ``sagemaker_sklearn_extension.preprocessing.data.RobustStandardScaler``
        - `robust_standard_scaler_` is instantiated inside the fit method used for computing the mean and
        the standard deviation.


    Examples
    --------
    >>> from sagemaker_sklearn_extension.feature_extraction.sequences import TSFreshFeatureExtractor
    >>> import numpy as np
    >>> data = [np.array([ 3., np.nan,  4.]), np.array([ 5, 6]), np.array([8, np.nan,  np.nan, 10])]
    >>> tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=True, interpolation_method="hybrid")
    >>> X = tsfresh_feature_extractor.fit_transform(data)
    >>> print(X.shape)
    (3, 785)
    >>> print(X[:4, :4])
    [[ 3.  3.  4.  0.]
     [ 5.  6.  0.  0.]
     [ 8.  8.  8. 10.]]
    >>> tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=False)
    >>> X = tsfresh_feature_extractor.fit_transform(data)
    >>> print(X.shape)
    (3, 781)
    """

    def __init__(self, augment=False, interpolation_method="hybrid", extraction_type="efficient"):
        super().__init__()
        self.augment = augment
        self.interpolation_method = interpolation_method
        self.extraction_type = extraction_type

    def fit(self, X, y=None):
        tsfresh_features, _ = self._extract_tsfresh_features(X)
        robust_standard_scaler = RobustStandardScaler()
        robust_standard_scaler.fit(tsfresh_features)
        self.robust_standard_scaler_ = robust_standard_scaler
        return self

    def transform(self, X, y=None):
        """Extract features computed by tsfresh from each input array/list.

        Parameters
        ----------
        X : list of np.arrays or list of lists

        Returns
        -------
        tsfresh_features : np.array

        """
        check_is_fitted(self, "robust_standard_scaler_")
        tsfresh_features, X_df = self._extract_tsfresh_features(X)
        tsfresh_features = self.robust_standard_scaler_.transform(tsfresh_features)
        if self.augment:
            # Stack the extracted features to the original sequences in X, after padding with np.nans any shorter
            # input sequences in X to match the length of the longest sequence, and imputing missing values as
            # specified by interpolation_method
            X_df_padded = self._interpolate(X_df)
            X = X_df_padded.groupby("id").agg(lambda x: x.tolist())[0].to_numpy()
            X = np.stack(X, axis=0)
            tsfresh_features = np.hstack((X, tsfresh_features))
        return tsfresh_features

    def _impute_ts(self, X, interpolation_method):
        """Impute time series missing values by linear interpolation,
        forward/backward filling, or padding with zeroes.
        """
        if interpolation_method == "linear":
            X[0] = X[0].interpolate(method=interpolation_method, limit_direction="both")
        elif interpolation_method == "fill":
            # Forward fill to complete the sequences. Then, backfill in case of NaNs at the start of the sequence.
            X[0] = X[0].interpolate(method="ffill", axis=0).interpolate(method="bfill", axis=0)
        elif interpolation_method == "zeroes":
            X[0] = X[0].fillna(0)
        elif interpolation_method == "hybrid":
            X = X.groupby("id").apply(self._hybrid_interpolation)
        else:
            raise ValueError(
                f"{interpolation_method} is not a supported interpolation method. Please choose one from "
                f"the following options: [linear, fill, zeroes, hybrid]."
            )
        return X

    @staticmethod
    def _convert_to_df(X):
        """Convert the list of np.arrays X into a dataframe compatible with extract_features."""
        X_df = pd.DataFrame(data=X)
        X_df = X_df.stack(dropna=False)
        X_df.index.rename(["id", "time"], inplace=True)
        X_df = X_df.reset_index()
        return X_df

    def _interpolate(self, X_df):
        """Impute missing values through the selected interpolation_method."""
        if self.interpolation_method is not None:
            X_df = self._impute_ts(X_df, self.interpolation_method)
        return X_df

    @staticmethod
    def _hybrid_interpolation(x):
        """Replace with zeroes any NaNs at the end or start of the sequences and forward fill the remaining NaNs."""
        # Compute the index of the first and last non-NaN value
        # In case of all NaNs, both first_valid and last_valid are None, and all values get replaced with zeroes
        first_valid = x[0].first_valid_index()
        last_valid = x[0].last_valid_index()
        x.loc[first_valid:last_valid, 0] = x.loc[first_valid:last_valid, 0].interpolate(method="ffill", axis=0)
        x[0] = x[0].fillna(0)
        return x

    def _extract_tsfresh_features(self, X):
        X_df = self._convert_to_df(X)
        X_df_no_nans = X_df.dropna()
        if self.extraction_type == "minimal":
            extraction_setting = MinimalFCParameters()
        elif self.extraction_type == "efficient":
            extraction_setting = EfficientFCParameters()
        elif self.extraction_type == "all":
            extraction_setting = ComprehensiveFCParameters()
        else:
            raise ValueError(
                f"{self.extraction_type} is not a supported feature extraction option. Please choose one from "
                f"the following options: [minimal, efficient, all]."
            )
        # Extract time series features from the dataframe
        # Replace any ``NaNs`` and ``infs`` in the extracted features with median/extreme values for that column
        tsfresh_features = extract_features(
            X_df_no_nans,
            default_fc_parameters=extraction_setting,
            column_id="id",
            column_sort="time",
            impute_function=impute,
        )
        # If X_df.dropna() dropped some observations entirely (i.e., due to all NaNs),
        # impute each tsfresh feature for those observations with the median of that tsfresh feature
        tsfresh_features_imputed = impute(tsfresh_features.reindex(pd.RangeIndex(X_df["id"].max() + 1)))
        return tsfresh_features_imputed, X_df

    def _more_tags(self):
        return {"_skip_test": True, "allow_nan": True}
