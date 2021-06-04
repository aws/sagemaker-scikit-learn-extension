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
    TSFlattener extracts the numerical values contained in these strings and pads shorter sequences (if any) with NaNs,
    so that all sequences match the length of the longest sequence. TSFlattener returns a single np.array as output.

    Any missing values (represented either by a NaN, inf or empty string) are converted to np.nans.

    Example:
    Input = [["1,, 3, 44"],
             ["11, 111"],
             ["NaN, , 1, NaN"]]
    Output = [[1, np.nan, 3, 44],
              [11, 111, np.nan, np.nan],
              [np.nan, np.nan, 1, np.nan]
              ]
    """

    def fit(self, X, y=None):
        check_array(X, dtype=None, force_all_finite="allow-nan")
        return self

    def transform(self, X, y=None):
        """Extract numerical values from strings of comma-separated numbers and returns an np.array.

        - If some sequences are shorter than the longest sequence, these are padded with np.nans.
        - Any NaN, inf or empty string is converted to a np.nan.

        Parameters
        ----------
        X : lists of strings

        Returns
        -------
        X : np.array

        """
        X = check_array(X, dtype=None, force_all_finite="allow-nan")

        numeric_sequences = []
        for string_sequence in X:
            assert len(string_sequence) == 1, (
                f"TSFlattener can process a single sequence column at a time, "
                f"but it was given {len(string_sequence)} sequence columns."
            )
            numeric_sequence = [float(s) if (s and not s.isspace()) else np.nan for s in string_sequence[0].split(",")]
            # Convert inf (if any) to np.nan
            numeric_sequence = [value if (not np.isinf(value)) else np.nan for value in numeric_sequence]
            numeric_sequences.append(numeric_sequence)
        # Pad shorter sequences with np.nan to reach the length of the longest sequence
        max_length = np.max([len(numeric_sequence) for numeric_sequence in numeric_sequences])
        padded_sequences = []
        for sequence in numeric_sequences:
            if len(sequence) < max_length:
                length_difference = max_length - len(sequence)
                padding_nans = np.array([np.nan] * length_difference)
                sequence = np.hstack([sequence, padding_nans])
            padded_sequences.append(sequence)
        X = np.array(padded_sequences)
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
        X = check_array(X, force_all_finite="allow-nan")
        check_is_fitted(self, "robust_standard_scaler_")
        if X.shape[1] != self._dim:
            raise ValueError(f"The input dimension is {X.shape[1]} instead of the expected {self._dim}")
        tsfresh_features, X_df = self._extract_tsfresh_features(X)
        tsfresh_features = pd.DataFrame(self.robust_standard_scaler_.transform(tsfresh_features))
        if self.augment:
            # Append the extracted features to the original dataset X, after converting X
            # from a DataFrame back to a np.array (with missing values imputed)
            X = X_df.groupby("id").agg(lambda x: x.tolist())[0].to_numpy()
            X = np.stack(X, axis=0)
            tsfresh_features = np.hstack((X, tsfresh_features.to_numpy()))
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
        # Replace all ``NaNs`` and ``infs`` in the extracted features with average/extreme values for that column
        extraction_settings = ComprehensiveFCParameters()
        tsfresh_features = extract_features(
            X_df, default_fc_parameters=extraction_settings, column_id="id", column_sort="time", impute_function=impute
        )
        return tsfresh_features, X_df

    def _more_tags(self):
        return {"allow_nan": True}
