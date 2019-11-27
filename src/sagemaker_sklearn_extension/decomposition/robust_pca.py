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

from scipy.sparse import issparse

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.utils.validation import check_array, check_is_fitted


class RobustPCA(BaseEstimator, TransformerMixin):
    """RobustPCA dimension reduction for dense and sparse matrices.

    RobustPCA uses a different implementation of singular value decomposition depending on the input.
    - ``sklearn.decomposition.PCA`` for dense inputs
    - ``sklearn.decomposition.TruncatedSVD`` for sparse inputs

    Please see ``sklearn.decomposition.PCA`` or ``sklearn.decomposition.TruncatedSVD`` for more details.

    If input number of features (input dimension) is less than n_components (target dimension), then no dimension
    reduction will be performed. The output will be the same as the input.

    Parameters
    ----------
    n_components : int, optional (default=1000)
        Desired dimensionality of output data.
        Must be strictly less than the number of features. If n_components is greater than than the number of features,
        no dimension reduction will be performed.

    svd_solver : string, optional (default='auto')

        - If 'auto', the solver is selected by a default policy based on `X.shape` and `n_components`: if the input
          data is larger than 500x500 and the number of components to extract is lower than 80% of the smallest
          dimension of the data, then the more efficient 'randomized' method is enabled. Otherwise the exact full
          RobustPCA is computed and optionally truncated afterwards.
          Note: 'auto' option only available for dense inputs. If 'auto' and input is sparse, svd_solver will use
          'randomized'
        - If 'full', run exact full RobustPCA calling the standard LAPACK solver via `scipy.linalg.svd` and select the
          components by postprocessing.
          Note: 'full' option only available for dense inputs. If 'full' and input is sparse, svd_solver will use
          'randomized'
        - If 'arpack', run RobustPCA truncated to n_components calling ARPACK solver via `scipy.sparse.linalg.svds`.
          'arpack' requires strictly 0 < n_components < n_components
        - If 'randomized', run randomized RobustPCA by the method of Halko et al.

    iterated_power : int >= 0 or 'auto', optional (default='auto')
        Number of iterations for the power method computed by
        svd_solver == 'randomized'.
        Note: If 'auto' and input is sparse, default for `iterated_power` is 5.

    tol : float >= 0, optional (default=0.)
        Tolerance for singular values computed by svd_solver == 'arpack'. 0 means machine precision. Ignored by
        randomized RobustPCA solver.

    random_state : int, RandomState instance, or None, optional (default=None)
        - If int, random_state is the seed used by the random number generator;
        - If RandomState instance, random_state is the random number generator;
        - If None, the random number generator is the RandomState instance used
          by np.random. Used when svd_solver == 'arpack' or 'randomized'.


    Attributes
    ----------
    robust_pca_ : ``sklearn.decomposition.PCA``, ``sklearn.decomposition.TruncatedSVD``, or None
        - If input number of features (input dimension) is less than n_components (target dimension), then `svd_` will
          be set to None and no dimension reduction will be performed. The output will be the same as the input.

        Assuming number of features is more than n_components:
        - If input is sparse, `svd_` is ``sklearn.decomposition.TruncatedSVD``.
        - If input is dense, `svd_` is ``sklearn.decomposition.PCA``

    Notes
    -----
    For dense inputs, ``sklearn.decomposition.PCA`` will center the input data by per-feature mean subtraction before
    RobustPCA. Sparse inputs will not center data.
    """

    def __init__(self, n_components=1000, svd_solver="auto", iterated_power="auto", tol=0.0, random_state=None):
        self.n_components = n_components
        self.svd_solver = svd_solver
        self.iterated_power = iterated_power
        self.tol = tol
        self.random_state = random_state

    def fit(self, X, y=None):
        """Fit the model with X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data.

        Returns
        -------
        self : RobustPCA
        """
        X = check_array(X, accept_sparse=True, dtype=None)

        # if input dimension is less than or equal to target dimension, no reduction will be performed
        if X.shape[1] <= self.n_components:
            self.robust_pca_ = None
            return self

        # fit for sparse or dense input
        if issparse(X):
            algorithm = self.svd_solver if self.svd_solver == "arpack" else "randomized"
            n_iter = self.iterated_power if self.iterated_power != "auto" else 5

            self.robust_pca_ = TruncatedSVD(
                n_components=self.n_components,
                algorithm=algorithm,
                n_iter=n_iter,
                random_state=self.random_state,
                tol=self.tol,
            )
        else:
            self.robust_pca_ = PCA(
                n_components=self.n_components,
                svd_solver=self.svd_solver,
                tol=self.tol,
                iterated_power=self.iterated_power,
                random_state=self.random_state,
            )

        self.robust_pca_.fit(X)
        return self

    def transform(self, X, y=None):
        """Fit the model with X and apply the dimensionality reduction on X.

        Parameters
        ----------
        X : array-like, shape (n_samples, n_features)
            Training data

        Returns
        -------
        X : array-like, shape (n_samples, n_features)
        or
        X_new : array-like, shape (n_samples, n_components)

        """
        check_is_fitted(self, "robust_pca_")

        if self.robust_pca_:
            return self.robust_pca_.transform(X)
        return X
