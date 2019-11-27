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
from scipy.sparse import csr_matrix

from sklearn import datasets
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.testing import assert_array_almost_equal

from sagemaker_sklearn_extension.decomposition import RobustPCA


X_iris = datasets.load_iris().data
X_iris_sparse = csr_matrix(X_iris)


@pytest.mark.parametrize(
    ["X", "n_components", "X_expected"],
    [
        # Dense input
        (X_iris, 2, PCA(n_components=2).fit_transform(X_iris)),
        # Sparse input
        (X_iris_sparse, 2, TruncatedSVD().fit_transform(X_iris_sparse)),
        # n_components > X.shape[1], no dimension reduction
        (X_iris, 1000, X_iris),
    ],
)
def test_svd(X, n_components, X_expected):
    svd = RobustPCA(n_components=n_components)
    X_observed = svd.fit_transform(X)

    assert_array_almost_equal(X_observed, X_expected)
