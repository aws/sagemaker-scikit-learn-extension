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
from scipy.sparse import csr_matrix

from sklearn.decomposition import PCA
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import FunctionTransformer
from sagemaker_sklearn_extension.externals import AutoMLTransformer
from sagemaker_sklearn_extension.externals import Header
from sagemaker_sklearn_extension.externals import read_csv_data
from sagemaker_sklearn_extension.preprocessing import NALabelEncoder
from sagemaker_sklearn_extension.impute import RobustImputer


def to_csr(X):
    return csr_matrix(X.shape, dtype=np.int8)


impute_pca_pipeline = Pipeline(steps=[("impute", SimpleImputer()), ("pca", PCA(n_components=2))])


@pytest.mark.parametrize(
    "feature_transformer, target_transformer, " "expected_X_transformed_shape, expected_Xy_transformed_shape",
    [
        (impute_pca_pipeline, LabelEncoder(), (10, 2), (10, 3)),
        (impute_pca_pipeline, NALabelEncoder(), (10, 2), (9, 3)),
        (FunctionTransformer(to_csr, validate=False), None, (10, 3), (9, 4)),
    ],
)
def test_automl_transformer(
    feature_transformer, target_transformer, expected_X_transformed_shape, expected_Xy_transformed_shape
):
    X = np.arange(0, 3 * 10).reshape((10, 3)).astype(np.str)
    y = np.array([0] * 5 + [1] * 4 + [np.nan]).astype(np.str)

    header = Header(column_names=["x1", "x2", "x3", "class"], target_column_name="class")
    automl_transformer = AutoMLTransformer(
        header=header, feature_transformer=feature_transformer, target_transformer=target_transformer,
    )

    model = automl_transformer.fit(X, y)

    X_transformed = model.transform(X)
    assert X_transformed.shape == expected_X_transformed_shape

    Xy = np.column_stack([X, y])

    Xy_transformed = model.transform(Xy)
    assert Xy_transformed.shape == expected_Xy_transformed_shape

    with pytest.raises(ValueError):
        model.transform(X[:, 2:])


def test_automl_transformer_regression():
    """Tests that rows in a regression dataset where the target column is not a finite numeric are imputed"""
    data = read_csv_data(source="test/data/csv/regression_na_labels.csv")
    X = data[:, :3]
    y = data[:, 3]
    header = Header(column_names=["x1", "x2", "x3", "class"], target_column_name="class")
    automl_transformer = AutoMLTransformer(
        header=header,
        feature_transformer=RobustImputer(strategy="constant", fill_values=0),
        target_transformer=NALabelEncoder(),
    )
    model = automl_transformer.fit(X, y)
    X_transformed = model.transform(X)
    assert X_transformed.shape == X.shape

    Xy = np.concatenate((X, y.reshape(-1, 1)), axis=1)

    Xy_transformed = model.transform(Xy)
    assert Xy_transformed.shape == (3, 4)
    assert np.array_equal(
        Xy_transformed, np.array([[1.1, 1.0, 2.0, 3.0], [2.2, 4.0, 0.0, 5.0], [3.3, 12.0, 13.0, 14.0]])
    )
