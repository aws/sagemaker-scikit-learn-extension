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
import scipy.sparse as sp

from sagemaker_sklearn_extension.feature_extraction.text import MultiColumnTfidfVectorizer

from sklearn.feature_extraction.text import TfidfVectorizer


corpus = np.array(
    [
        ["Cats eat rats.", "Rats are mammals."],
        ["Dogs chase cats.", "Cats have ears."],
        ["People like dogs.", "People are mammals."],
        ["People hate rats.", "Rats are quite smart."],
    ]
)


def test_multi_column_tfidf_vectorizer():
    vec = MultiColumnTfidfVectorizer()
    output = vec.fit_transform(corpus)

    assert isinstance(output, sp.coo.coo_matrix)

    observed = output.todense()
    expected = np.hstack(
        [
            TfidfVectorizer().fit_transform(corpus[:, 0]).todense(),
            TfidfVectorizer().fit_transform(corpus[:, 1]).todense(),
        ]
    )

    np.testing.assert_array_equal(observed, expected)


def test_multi_column_tfidf_vectorizer_fit_dim_error():
    with pytest.raises(ValueError):
        vec = MultiColumnTfidfVectorizer()
        vec.fit(corpus[0])


def test_multi_column_tfidf_vectorizer_transform_dim_error():
    with pytest.raises(ValueError):
        vec = MultiColumnTfidfVectorizer()
        vec.fit(corpus)
        vec.transform(corpus[0])


def test_multi_column_tfidf_vectorizer_vocabulary_sizes_large():
    vocabulary_sizes = [TfidfVectorizer().fit_transform(corpus[:, i]).shape[1] + 1 for i in range(corpus.shape[1])]
    vectorizer = MultiColumnTfidfVectorizer(vocabulary_sizes=vocabulary_sizes)
    observed = vectorizer.fit_transform(corpus)
    assert observed.shape[1] == sum(vocabulary_sizes)
    assert sp.issparse(observed)


def test_multi_column_tfidf_vectorizer_vocabulary_sizes_small():
    vocabulary_sizes = [TfidfVectorizer().fit_transform(corpus[:, i]).shape[1] - 1 for i in range(corpus.shape[1])]
    vectorizer = MultiColumnTfidfVectorizer(vocabulary_sizes=vocabulary_sizes)
    observed = vectorizer.fit_transform(corpus)
    assert observed.shape[1] == sum(vocabulary_sizes)
    assert sp.issparse(observed)


def test_multi_column_tfidf_vectorizer_vocabulary_sizes_error():
    with pytest.raises(ValueError):
        vectorizer = MultiColumnTfidfVectorizer(vocabulary_sizes=[1])
        vectorizer.fit(corpus)


@pytest.mark.parametrize(
    "kwargs, data, shape",
    [
        ({"min_df": 0.9}, corpus, (4, 0)),
        ({"max_df": 0.1}, corpus, (4, 0)),
        ({"max_df": 0.9941}, np.array([[""], [""], [""]]), (3, 0)),
    ],
)
def test_multi_column_tfidf_vectorizer_zero_output_tokens_ignore_zero_vocab_on(kwargs, data, shape):
    """Tests for empty matrix when no terms remain after pruning"""
    vec = MultiColumnTfidfVectorizer(**kwargs)
    output = vec.fit_transform(data)
    assert output.shape == shape


@pytest.mark.parametrize(
    "kwargs, data",
    [
        ({"min_df": 0.9, "ignore_columns_with_zero_vocabulary_size": False}, corpus),
        ({"max_df": 0.1, "ignore_columns_with_zero_vocabulary_size": False}, corpus),
        ({"max_df": 0.9941, "ignore_columns_with_zero_vocabulary_size": False}, np.array([[""], [""], [""]])),
    ],
)
def test_multi_column_tfidf_vectorizer_zero_output_tokens_ignore_zero_vocab_off(kwargs, data):
    """Tests for ValueError when no terms remain after pruning and `ignore_overpruned_columns=False`"""
    with pytest.raises(ValueError):
        vec = MultiColumnTfidfVectorizer(**kwargs)
        vec.fit_transform(data)


@pytest.mark.parametrize("kwargs, output_shape", [({"min_df": 0.9}, (4, 3)), ({"max_df": 0.9}, (4, 8))])
def test_multi_column_tfidf_vectorizer_one_column_zero_output_tokens(kwargs, output_shape):
    """Tests that a TF-IDF document-term matrix is still returned when only one column breaks"""
    corpus = np.array(
        [
            ["Cats eat rats.", "Rats are mammals."],
            ["Dogs chase cats.", "Rats are mammals."],
            ["People like dogs.", "Rats are mammals."],
            ["People hate rats.", "Rats are mammals."],
        ]
    )

    vec = MultiColumnTfidfVectorizer(**kwargs)
    output = vec.fit_transform(corpus)
    assert output.shape == output_shape
