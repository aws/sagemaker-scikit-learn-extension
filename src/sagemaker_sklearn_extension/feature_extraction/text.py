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
import scipy.sparse as sp

from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.feature_extraction.text import VectorizerMixin, TfidfVectorizer
from sklearn.utils.validation import check_array, check_is_fitted


class MultiColumnTfidfVectorizer(BaseEstimator, VectorizerMixin, TransformerMixin):
    """Applies ``sklearn.feature_extraction.text.TfidfVectorizer`` to each column in an array.

    Each column of text is treated separately with a unique TfidfVectorizer. The vectorizers are applied sequentially.

    Parameters
    ----------
    strip_accents : {'ascii', 'unicode', None} (default=None)
        Remove accents and perform other character normalization during the preprocessing step.
        'ascii' is a fast method that only works on characters that have an direct ASCII mapping.
        'unicode' is a slightly slower method that works on any characters.
        None (default) does nothing.

        Both 'ascii' and 'unicode' use NFKD normalization from :func:`unicodedata.normalize`.

    lowercase : boolean (default=True)
        Convert all characters to lowercase before tokenizing.

    preprocessor : callable or None (default=None)
        Override the preprocessing (string transformation) stage while preserving the tokenizing and n-grams
        generation steps.

    tokenizer : callable or None (default=None)
        Override the string tokenization step while preserving the preprocessing and n-grams generation steps.
        Only applies if ``analyzer == 'word'``.

    stop_words : string {'english'}, list, or None (default)
        If 'english', a built-in stop word list for English is used.
        There are several known issues with 'english' and you should consider an alternative (see :ref:`stop_words`).

        If a list, that list is assumed to contain stop words, all of which will be removed from the resulting tokens.
        Only applies if ``analyzer == 'word'``.

        If None, no stop words will be used. max_df can be set to a value in the range [0.7, 1.0) to automatically
        detect and filter stop words based on intra corpus document frequency of terms.

    token_pattern : string
        Regular expression denoting what constitutes a "token", only used if ``analyzer == 'word'``. The default regexp
        select tokens of 2 or more alphanumeric characters (punctuation is completely ignored and always treated as a
        token separator).

    ngram_range : tuple (min_n, max_n) (default=(1, 1))
        The lower and upper boundary of the range of n-values for different n-grams to be extracted. All values of n
        such that min_n <= n <= max_n will be used.

    analyzer : string, {'word', 'char', 'char_wb'} or callable
        Whether the feature should be made of word or character n-grams.
        Option 'char_wb' creates character n-grams only from text inside word boundaries; n-grams at the edges of words
        are padded with space.

        If a callable is passed it is used to extract the sequence of features out of the raw, unprocessed input.

    max_df : float in range [0.0, 1.0] or int (default=1.0)
        When building the vocabulary ignore terms that have a document frequency strictly higher than the given
        threshold (corpus-specific stop words).
        If float, the parameter represents a proportion of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    min_df : float in range [0.0, 1.0] or int (default=1)
        When building the vocabulary ignore terms that have a document frequency strictly lower than the given
        threshold. This value is also called cut-off in the literature.
        If float, the parameter represents a proportion of documents, integer absolute counts.
        This parameter is ignored if vocabulary is not None.

    max_features : int or None (default=1000)
        If not None, build a vocabulary that only consider the top max_features ordered by term frequency across
        the corpus.
        This parameter is ignored if vocabulary is not None.

    vocabulary : Mapping or iterable, optional (default=None)
        Either a Mapping (e.g., a dict) where keys are terms and values are indices in the feature matrix, or an
        iterable over terms. If not given, a vocabulary is determined from the input.

    dtype : type, optional (default=float64)
        Type of the matrix returned by fit_transform() or transform().

    norm : 'l1', 'l2' or None, optional (default='l2')
        Each output row will have unit norm, either:
        * 'l2': Sum of squares of vector elements is 1. The cosine similarity between two vectors is their dot product
        when l2 norm has been applied.
        * 'l1': Sum of absolute values of vector elements is 1.
        See :func:`preprocessing.normalize`

    use_idf : boolean (default=True)
        Enable inverse-document-frequency reweighting.

    smooth_idf : boolean (default=True)
        Smooth idf weights by adding one to document frequencies, as if an extra document was seen containing every
        term in the collection exactly once. Prevents zero divisions.

    sublinear_tf : boolean (default=False)
        Apply sublinear tf scaling, i.e. replace tf with 1 + log(tf).

    vocabulary_sizes : list(int) (default=None)
        Specify the exact vocabulary size to use while encoding each column in the input dataset. The vocabulary size
        of a column corresponds to the number of features in its TF-IDF encoding, before the feature matrices are
        concatenated. If the feature matrix of column ``i`` has more features than the corresponding vocabulary size,
        only the first ``vocabulary_sizes[i]`` features are kept. If the feature matrix of column ``i`` has fewer
        features than the corresponding vocabulary size, zero columns are added to the feature matrix until it has
        ``vocabulary_sizes[i]`` features. This parameter is useful if the total number of features of the encoding
        has to be constant.

    ignore_columns_with_zero_vocabulary_size : boolean (default=True)
        Allow ValueErrors thrown by ``sklearn.feature_extraction.text.TfidfVectorizer`` because of over-pruning
        of terms to be ignored and an empty ``scipy.sparse.csr_matrix`` to be used in place of the given columns
        TF-IDF document-term matrix.

    Attributes
    ----------
    vectorizers_ : list of ``sklearn.feature_extraction.text.TfidfVectorizers``
        List of ``sklearn.feature_extraction.text.TfidfVectorizers``. Each TfidfVectorizer is separately instantiated
        on an input column. len(self.vectorizers_) should equal to the number of input columns.

    Notes
    -----
    MultiColumnTfidfVectorizer should be used with 2D arrays of text strings, for 1D arrays of text data, use
    ``sklearn.feature_extraction.text.TfidfVectorizer`` or reshape using array.reshape(-1, 1)
    """

    def __init__(
        self,
        strip_accents=None,
        lowercase=True,
        preprocessor=None,
        tokenizer=None,
        stop_words=None,
        token_pattern=r"(?u)\b\w\w+\b",
        ngram_range=(1, 1),
        analyzer="word",
        max_df=1.0,
        min_df=1,
        max_features=1000,
        vocabulary=None,
        dtype=np.float64,
        norm="l2",
        use_idf=True,
        smooth_idf=True,
        sublinear_tf=False,
        vocabulary_sizes=None,
        ignore_columns_with_zero_vocabulary_size=True,
    ):
        self.strip_accents = strip_accents
        self.lowercase = lowercase
        self.preprocessor = preprocessor
        self.tokenizer = tokenizer
        self.stop_words = stop_words
        self.token_pattern = token_pattern
        self.ngram_range = ngram_range
        self.analyzer = analyzer
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features
        self.vocabulary = vocabulary
        self.dtype = dtype
        self.norm = norm
        self.use_idf = use_idf
        self.smooth_idf = smooth_idf
        self.sublinear_tf = sublinear_tf
        self.vocabulary_sizes = vocabulary_sizes
        self.ignore_columns_with_zero_vocabulary_size = ignore_columns_with_zero_vocabulary_size

    def fit(self, X, y=None):
        """Build the list of TfidfVectorizers for each column.

        Parameters
        ----------
        X : {array-like}, text data

        Returns
        -------
        self : MultiColumnTfidfVectorizer
        """
        X = check_array(X, dtype=None)
        n_columns = X.shape[1]

        # If specified, vocabulary size must be given for each column of the input dataset.
        if self.vocabulary_sizes and len(self.vocabulary_sizes) != n_columns:
            raise ValueError("If specified, vocabulary_sizes has to have exactly one entry per data column.")

        self.vectorizers_ = []
        for col_idx in range(n_columns):
            max_features = self.max_features

            # Override max_features for the current column in order to enforce the vocabulary size.
            if self.max_features and self.vocabulary_sizes:
                max_features = min(self.max_features, self.vocabulary_sizes[col_idx])
            elif self.vocabulary_sizes:
                max_features = self.vocabulary_sizes[col_idx]

            try:
                vectorizer = TfidfVectorizer(
                    strip_accents=self.strip_accents,
                    lowercase=self.lowercase,
                    preprocessor=self.preprocessor,
                    tokenizer=self.tokenizer,
                    stop_words=self.stop_words,
                    token_pattern=self.token_pattern,
                    ngram_range=self.ngram_range,
                    analyzer=self.analyzer,
                    max_df=self.max_df,
                    min_df=self.min_df,
                    max_features=max_features,
                    vocabulary=self.vocabulary,
                    dtype=self.dtype,
                    norm=self.norm,
                    use_idf=self.use_idf,
                    smooth_idf=self.smooth_idf,
                    sublinear_tf=self.sublinear_tf,
                )
                vectorizer.fit(X[:, col_idx])
            except ValueError as err:
                zero_vocab_errors = [
                    "After pruning, no terms remain. Try a lower min_df or a higher max_df.",
                    "max_df corresponds to < documents than min_df",
                    "empty vocabulary; perhaps the documents only contain stop words",
                ]
                if str(err) in zero_vocab_errors and self.ignore_columns_with_zero_vocabulary_size:
                    vectorizer = None
                else:
                    raise

            self.vectorizers_.append(vectorizer)

        return self

    def transform(self, X, y=None):
        """Transform documents to document term-matrix.

        Parameters
        ----------
        X : 2D array of text data

        Returns
        -------
        tfidf_matrix : sparse matrix, [n_samples, n_features]
                       Tf-idf-weighted document-term matrix.
        """
        check_is_fitted(self, "vectorizers_")
        X = check_array(X, dtype=None)

        ret = []
        for col_idx in range(X.shape[1]):
            if self.vectorizers_[col_idx]:
                tfidf_features = self.vectorizers_[col_idx].transform(X[:, col_idx])
                # If the vocabulary size is specified and there are too few features, then pad the output with zeros.
                if self.vocabulary_sizes and tfidf_features.shape[1] < self.vocabulary_sizes[col_idx]:
                    tfidf_features = sp.csr_matrix(
                        (tfidf_features.data, tfidf_features.indices, tfidf_features.indptr),
                        shape=(tfidf_features.shape[0], self.vocabulary_sizes[col_idx]),
                    )
            else:
                # If ``TfidfVectorizer`` threw a value error, add an empty TF-IDF document-term matrix for the column
                tfidf_features = sp.csr_matrix((X.shape[0], 0))
            ret.append(tfidf_features)

        return sp.hstack(ret)

    def _more_tags(self):
        return {"X_types": ["string"]}
