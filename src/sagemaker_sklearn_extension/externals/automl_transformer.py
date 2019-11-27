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

from scipy.sparse import isspmatrix
from sklearn.base import BaseEstimator
from sklearn.base import TransformerMixin


class AutoMLTransformer(BaseEstimator, TransformerMixin):
    """Utility class encapsulating feature and target transformation functionality used in AutoML pipelines.

    Parameters
    ----------
    header : Header instance
        Instance of the ``Header`` class from ``sagemaker_sklearn_extension.externals``. Contains indices of the
        features and response in the corresponding dataset.

    feature_transformer : transformer instance
        A Scikit-Learn transformer used on the feature columns in the dataset. Should have ``fit`` and ``transform``
        methods which accept 2-dimensional inputs.

    target_transformer : transformer instance
        A Scikit-Learn transformer used on the target column in the dataset. Should have ``fit``, ``transform``, and
        optionally ``inverse_transform`` methods which accept 1-dimensional inputs.
    """

    def __init__(self, header, feature_transformer, target_transformer):
        self.header = header
        self.feature_transformer = feature_transformer
        self.target_transformer = target_transformer

    def fit(self, X, y):
        """Fit and transform target, then fit feature data using the underlying transformers.

        Parameters
        ----------
        X : numpy array of shape [n_samples, n_features]
            The feature-only dataset.

        y : numpy array of shape [n_samples]
            The target column.

        Returns
        -------
        self : AutoMLTransformer
        """
        y_transformed = y

        if self.target_transformer:
            y_transformed = self.target_transformer.fit_transform(y)

        self.feature_transformer.fit(X, y_transformed)
        return self

    def transform(self, X):
        """Transform the dataset using the underlying transformers.

        Depending on the shape of the input, it transforms either the feature data, or the feature data and the target
        column and then concatenates them back into a single dataset.

        Parameters
        ----------
        X : numpy array
            The array to transform whose shape should be either:
            - [n_samples, n_features], if it only contains the features; or
            - [n_samples, n_features + 1], if it contains the feature columns and the target column.

        Returns
        -------
        array-like of shape [n_samples, n_transformed_features] or [n_samples, n_transformed_features + 1]
        """
        n_columns = X.shape[1]
        n_features = len(self.header.feature_column_indices)

        # X contains both features and response.
        if n_columns == n_features + 1:
            y = X[:, self.header.target_column_index]
            y_transformed = self.label_transform(y)
            non_nan_indices = np.arange(y_transformed.shape[0])[~np.isnan(y_transformed)]
            feature_indices = np.array(self.header.feature_column_indices)
            X_transformed = self.feature_transformer.transform(
                X[non_nan_indices[:, np.newaxis], feature_indices[np.newaxis, :]]
            )
            y_transformed_no_nans = y_transformed[non_nan_indices]
            return np.column_stack((y_transformed_no_nans, self._dense_array(X_transformed)))

        # X contains only the features.
        if n_columns == n_features:
            return self.feature_transformer.transform(X)

        raise ValueError(
            f"Received data of unknown size. Expected number of columns is {n_features}. "
            f"Number of columns in the received data is {n_columns}."
        )

    def label_transform(self, y):
        """Apply transformation, if ``target_transformer`` has been specified.

        Parameters
        ----------
        y : array-like, 1-dimensional

        Returns
        -------
        array-like
            The transformed data. If target transformer has not been specified, simply returns the input.
        """
        if self.target_transformer:
            return self.target_transformer.transform(y)

        return y.astype("float32")

    def inverse_label_transform(self, yt):
        """Apply inverse target transformation, if ``target_transformer`` has been specified set.

        Parameters
        ----------
        yt : array-like, 1-dimensional

        Returns
        -------
        array-like
            The inverse-transformed target. If target transformer has not been specified, simply returns the input.
        """
        if not self.target_transformer:
            return yt

        return self.target_transformer.inverse_transform(yt)

    @staticmethod
    def _dense_array(arr):
        """Converts the input array to dense array.

        Parameters
        ----------
        arr : numpy array or csr_matrix
            The array to be densified.

        Returns
        -------
        array-like
            Dense numpy array representing arr.

        """
        if isspmatrix(arr):
            return arr.todense()
        return arr
