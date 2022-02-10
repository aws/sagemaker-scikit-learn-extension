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
import os
from math import ceil

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.utils.validation import check_array, check_is_fitted
from tsfresh import extract_features
from tsfresh.feature_extraction import ComprehensiveFCParameters
from tsfresh.feature_extraction import EfficientFCParameters
from tsfresh.feature_extraction import MinimalFCParameters
from tsfresh.utilities.dataframe_functions import impute
from tsfresh.defaults import N_PROCESSES  # the default number of processes used by TSFresh, equals to n_vcores/2

TOTAL_EXPANSION_THRESHOLD = 2500
DEFAULT_INPUT_SEQUENCE_LENGTH = 1000
SEQUENCE_EXPANSION_FACTOR = 2.5
# do not use TSFresh parallelism in container serve(transform), does not work with server's workers
N_TSFRESH_JOBS = 0 if os.environ.get("SAGEMAKER_PROGRAM") == "sagemaker_serve" else N_PROCESSES


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

    extraction_seed : int (default = 0)
        Random seed used to choose subset of features, when expansion control does not allow to include all features

    sequences_lengths_q25 : list of ints (default = None)
        List contianing 25th percentile of sequence lengths for each column at the train step.
        Length of the list should correspond to total number of columns in the input.
        If not provided, default value will be assigned at the fit stage.

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
        extraction_seed=0,
        sequences_lengths_q25=None,
    ):
        super().__init__()
        if max_allowed_length <= 0:
            raise ValueError(f"{max_allowed_length} must be positive.\n")
        self.max_allowed_length = max_allowed_length
        self.trim_beginning = trim_beginning
        self.augment = augment
        self.interpolation_method = interpolation_method
        self.extraction_type = extraction_type
        self.extraction_seed = extraction_seed
        self.sequences_lengths_q25 = sequences_lengths_q25

    def fit(self, X, y=None):
        X = check_array(X, dtype=None, force_all_finite="allow-nan")

        if self.sequences_lengths_q25 is None:
            self.sequences_lengths_q25 = [DEFAULT_INPUT_SEQUENCE_LENGTH] * X.shape[1]

        if len(self.sequences_lengths_q25) != X.shape[1]:
            raise ValueError(
                f"length of sequences_lengths_q25 should be equal to number of columns in X (={X.shape[1]})."
            )
        # cap total expansion for all columns
        expansion_thresholds = np.ceil(
            (self.sequences_lengths_q25 / np.sum(self.sequences_lengths_q25)) * TOTAL_EXPANSION_THRESHOLD
        )
        ts_flattener = TSFlattener(max_allowed_length=self.max_allowed_length, trim_beginning=self.trim_beginning)
        tsfresh_feature_extractors = []
        for sequence_column_i, sequence_column in enumerate(X.T):
            numeric_sequences = ts_flattener.transform(sequence_column.reshape(-1, 1))
            tsfresh_feature_extractor = TSFreshFeatureExtractor(
                augment=self.augment,
                interpolation_method=self.interpolation_method,
                extraction_type=self.extraction_type,
                extraction_seed=self.extraction_seed,
                sequence_length_q25=self.sequences_lengths_q25[sequence_column_i],
                expansion_threshold=int(expansion_thresholds[sequence_column_i]),
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

    extraction_seed : int (default = 0)
        Random seed used to choose subset of features, when expansion control does not allow to include all features

    sequence_length_q25 : list of ints (default = None)
        List contianing 25th percentile of sequence lengths for each column at the train step.
        If not provided, default value will be assigned (DEFAULT_INPUT_SEQUENCE_LENGTH).


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

    def __init__(
        self,
        augment=False,
        interpolation_method="hybrid",
        extraction_type="efficient",
        extraction_seed=0,
        sequence_length_q25=None,
        expansion_threshold=None,
    ):
        super().__init__()
        self.augment = augment
        self.interpolation_method = interpolation_method
        self.extraction_type = extraction_type
        self.feature_sampling_seed = extraction_seed
        self.sequence_length_q25 = sequence_length_q25 or DEFAULT_INPUT_SEQUENCE_LENGTH
        expansion_threshold = expansion_threshold or self._compute_expansion_threshold(self.sequence_length_q25)
        self.expansion_threshold = min(expansion_threshold, self._compute_expansion_threshold(self.sequence_length_q25))
        # expansion_threshold will be the stricter between the one computed for this column and the one respecting
        # the total expansion for all columns

    def fit(self, X, y=None):
        # Nothing to learn during fit.
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
        tsfresh_features, X_df = self._extract_tsfresh_features(X)
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
        # covering corner case when all nans
        if X_df_no_nans.shape[0] == 0:
            X_df_no_nans = X_df.loc[[0]].fillna(0)
        if self.extraction_type not in ["minimal", "efficient", "all"]:
            raise ValueError(
                f"{self.extraction_type} is not a supported feature extraction option. Please choose one from "
                f"the following options: [minimal, efficient, all]."
            )
        min_settings = MinimalFCParameters()
        # Extract time series features from the dataframe
        # Replace any ``NaNs`` and ``infs`` in the extracted features with median/extreme values for that column
        tsfresh_features = extract_features(
            X_df_no_nans,
            default_fc_parameters=min_settings,
            column_id="id",
            column_sort="time",
            impute_function=impute,
            n_jobs=N_TSFRESH_JOBS,
        )
        self.min_settings_card = tsfresh_features.shape[1]
        # Minimal features computed indepdently to ensure they go first in the output,
        # this is needed to ensure their survival when filtering features
        if self.extraction_type in ["efficient", "all"]:
            if self.extraction_type == "efficient":
                settings = EfficientFCParameters()
            else:
                settings = ComprehensiveFCParameters()
            settings = {k: v for k, v in settings.items() if k not in min_settings}

            self._apply_feature_threshold(settings)
            if settings:
                # check that efficient strategies are not emptied when applying expansion threshold
                tsfresh_features_extra = extract_features(
                    X_df_no_nans,
                    default_fc_parameters=settings,
                    column_id="id",
                    column_sort="time",
                    impute_function=impute,
                    n_jobs=N_TSFRESH_JOBS,
                )
                tsfresh_features = pd.concat([tsfresh_features, tsfresh_features_extra], axis=1)

        # If X_df.dropna() dropped some observations entirely (i.e., due to all NaNs),
        # impute each tsfresh feature for those observations with the median of that tsfresh feature
        tsfresh_features_imputed = impute(tsfresh_features.reindex(pd.RangeIndex(X_df["id"].max() + 1)))
        return tsfresh_features_imputed, X_df

    def _apply_feature_threshold(self, settings):
        """Accepts a settings dictionary, with all the possible generated features,
        and filters features if needed until their count matches the given "self.expansion_threshold"
        (minus minimal features).
        Does that in a reproducible "random" way, controlled by "self.feature_sampling_seed".
        Draws Random indexes to be filtered, then iterates over the settings dictionary assigning an index to each value
         and performs the filtering based on that index.
        """
        settings.pop("linear_trend_timewise", None)  # remove these 5 features that need dateTime indexes for sequences
        max_available_features = self._get_features_count(settings)
        if self.expansion_threshold >= max_available_features + self.min_settings_card:
            return  # no need to limit

        filter_order = np.arange(max_available_features)
        random_state = np.random.get_state()
        np.random.seed(self.feature_sampling_seed)
        np.random.shuffle(filter_order)
        np.random.set_state(random_state)
        removed_indices = list(filter_order[max(0, self.expansion_threshold - self.min_settings_card) :])
        removed_indices.sort()

        feature_idx = 0
        for k in list(settings.keys()):
            if isinstance(settings[k], list):
                survived_list = []
                # case the value is a list, each list element is counted separately
                for index, _ in enumerate(settings[k]):
                    if removed_indices and removed_indices[0] == feature_idx:
                        del removed_indices[0]
                    else:
                        survived_list.append(settings[k][index])
                    feature_idx += 1
                # copy the "survived", features to the final list. if no one survived, delete the settings key.
                if survived_list:
                    settings[k] = survived_list
                else:
                    del settings[k]
            else:
                # case the value is None, count it as one feature
                if removed_indices and removed_indices[0] == feature_idx:
                    del removed_indices[0]
                    del settings[k]
                feature_idx += 1

    def _compute_expansion_threshold(self, input_len):
        return int(max(ceil(SEQUENCE_EXPANSION_FACTOR * input_len + 1) + 1, 10))

    def _more_tags(self):
        return {"_skip_test": True, "allow_nan": True}

    def _get_features_count(self, settings):
        return sum([len(v) if isinstance(v, list) else 1 for v in settings.values()])
