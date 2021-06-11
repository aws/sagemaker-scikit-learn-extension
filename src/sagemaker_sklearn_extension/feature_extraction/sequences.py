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
from tsfresh.utilities.dataframe_functions import impute

from sagemaker_sklearn_extension.preprocessing.data import RobustStandardScaler


class TSFlattener(BaseEstimator, TransformerMixin):
    """Convert lists of strings of varying length into an np.array.

    The input is a collection of lists of strings, with each string containing a sequence of comma-separate numbers.
    TSFlattener extracts numerical values from these strings and pads shorter sequences (if any) with np.nans,
    so that all sequences match the length of the longest sequence. TSFlattener returns a single np.array as output.

    Any missing values (represented either by a NaN, inf or empty string) are converted to np.nans.
    Any entire sequence that cannot be parsed (represented by a None) is filled with np.nans to match the length
    of the longest sequence.

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
    [[  1.  nan   3.  44.]
     [ 11. 111.  nan  nan]
     [ nan  nan   1.  nan]]
    """

    def __init__(self, max_allowed_length=10000, trim_beginning=True):
        super().__init__()
        assert max_allowed_length > 0, f"{max_allowed_length} must be positive."
        self.max_allowed_length = max_allowed_length
        self.trim_beginning = trim_beginning

    def fit(self, X, y=None):
        check_array(X, dtype=None, force_all_finite="allow-nan")
        return self

    def transform(self, X, y=None):
        """Extract numerical values from strings of comma-separated numbers and returns an np.array.

        - If some sequences are shorter than the longest sequence, these are padded with np.nans.
        - Anything that can't be turned into a finite float is converted to a np.nan.

        Parameters
        ----------
        X : lists of strings

        Returns
        -------
        X : np.array

        """
        X = check_array(X, dtype=None, force_all_finite="allow-nan")
        # Parse the input strings
        numeric_sequences = self._convert_to_numeric(X)
        # Pad shorter sequences with np.nans to reach the length of the longest sequence
        X = self._pad_sequences(numeric_sequences)
        return X

    @staticmethod
    def _convert_to_numeric(X):
        numeric_sequences = []
        for string_sequence in X:
            assert len(string_sequence) == 1, (
                f"TSFlattener can process a single sequence column at a time, "
                f"but it was given {len(string_sequence)} sequence columns."
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
            numeric_sequences.append(numeric_sequence)
        return numeric_sequences

    def _pad_sequences(self, numeric_sequences):
        max_observed_length = np.max([len(numeric_sequence) for numeric_sequence in numeric_sequences])
        max_length = min(max_observed_length, self.max_allowed_length)
        num_observations = len(numeric_sequences)
        X = np.empty((num_observations, max_length))
        X.fill(np.nan)
        for id_sequence, sequence in enumerate(numeric_sequences):
            if len(sequence) > max_length:
                # If a sequence exceeds the maximum length allowed, truncate it
                if self.trim_beginning:
                    sequence = sequence[-max_length:]
                else:
                    sequence = sequence[:max_length]
            X[id_sequence, : len(sequence)] = sequence
        return X

    def _more_tags(self):
        return {"X_types": ["string"], "allow_nan": True}


class TSFreshFeatureExtractor(BaseEstimator, TransformerMixin):
    """Extract features computed by tsfresh from each row of an array and append them to the input
    array (if augment = True) or return the extracted features alone (if augment = False).

    Examples of these features are the mean, median, kurtosis, and autocorrelation of each sequence. The full list
    of extracted features can be found at https://tsfresh.readthedocs.io/en/latest/text/list_of_features.html.

    Parameters
    ----------
    augment : boolean (default=True):
        Whether to append the tsfresh features to the original raw data (if True),
        or output only the extracted tsfresh features (if False)

    interpolation_method : {'linear', 'fill', 'zeroes', None} (default='zeroes')
        'linear': linear interpolation
        'fill': forward fill to complete the sequences; then, backfill in case of NaNs at the start of the sequence.
        'zeroes': pad with zeroes
         None: no interpolation

    Attributes
    ----------
    self.robust_standard_scaler_ : ``sagemaker_sklearn_extension.preprocessing.data.RobustStandardScaler``
        - `robust_standard_scaler_` is instantiated inside the fit method used for computing the mean and
        the standard deviation.


    Examples
    --------
    >>> from sagemaker_sklearn_extension.feature_extraction.sequences import TSFreshFeatureExtractor
    >>> import numpy as np
    >>> data = np.array([[1, 2, np.nan], [4, 5, 6], [10, 10, 10]])
    >>> tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=True)
    >>> X = tsfresh_feature_extractor.fit_transform(data)
    >>> print(X.shape)
    (3, 790)
    >>> tsfresh_feature_extractor = TSFreshFeatureExtractor(augment=False)
    >>> X = tsfresh_feature_extractor.fit_transform(data)
    >>> print(X.shape)
    (3, 787)
    """

    def __init__(self, augment=True, interpolation_method="zeroes"):
        super().__init__()
        self.augment = augment
        self.interpolation_method = interpolation_method

    def fit(self, X, y=None):
        X = check_array(X, force_all_finite="allow-nan")
        self._dim = X.shape[1]
        tsfresh_features, _ = self._extract_tsfresh_features(X)
        robust_standard_scaler = RobustStandardScaler()
        robust_standard_scaler.fit(tsfresh_features)
        self.robust_standard_scaler_ = robust_standard_scaler
        return self

    def transform(self, X, y=None):
        """Extract features computed by tsfresh from each row of the input array after imputing missing values.

        Parameters
        ----------
        X : np.array

        Returns
        -------
        tsfresh_features : np.array

        """
        X = check_array(X, force_all_finite="allow-nan")
        check_is_fitted(self, "robust_standard_scaler_")
        if X.shape[1] != self._dim:
            raise ValueError(f"The input dimension is {X.shape[1]} instead of the expected {self._dim}")
        tsfresh_features, X_df = self._extract_tsfresh_features(X)
        tsfresh_features = self.robust_standard_scaler_.transform(tsfresh_features)
        if self.augment:
            # Append the extracted features to the original dataset X, after converting X
            # from a DataFrame back to a np.array (with missing values imputed)
            X = X_df.groupby("id").agg(lambda x: x.tolist())[0].to_numpy()
            X = np.stack(X, axis=0)
            tsfresh_features = np.hstack((X, tsfresh_features))
        return tsfresh_features

    @staticmethod
    def _impute_ts(X, interpolation_method):
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
        else:
            raise ValueError(
                f"{interpolation_method} is not a supported interpolation method. Please choose one from "
                f"the following options: [linear, fill, zeroes]."
            )
        return X

    @staticmethod
    def _convert_to_df(X):
        """Convert the np.array X into a dataframe compatible with extract_features."""
        X_df = pd.DataFrame(data=X.astype("float64"))
        X_df = X_df.stack(dropna=False)  # dropna=True would drop value for all sequences if at least one has a np.nan
        X_df.index.rename(["id", "time"], inplace=True)
        X_df = X_df.reset_index()
        return X_df

    def _interpolate(self, X_df):
        """Impute missing values through the selected interpolation_method."""
        if self.interpolation_method is not None:
            X_df = self._impute_ts(X_df, self.interpolation_method)
        return X_df

    def _extract_tsfresh_features(self, X):
        X_df = self._convert_to_df(X)
        X_df = self._interpolate(X_df)
        # Extract time series features from the dataframe
        # Replace any ``NaNs`` and ``infs`` in the extracted features with average/extreme values for that column
        extraction_settings = ComprehensiveFCParameters()
        tsfresh_features = extract_features(
            X_df, default_fc_parameters=extraction_settings, column_id="id", column_sort="time", impute_function=impute
        )
        return tsfresh_features, X_df

    def _more_tags(self):
        return {"allow_nan": True}
