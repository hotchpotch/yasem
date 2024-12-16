# test_splade_embedder.py
import numpy as np
import pytest
import scipy.sparse

from yasem import SpladeEmbedder
from yasem.splade_embedder import RankResult, RankResultWithText

SPLADE_MODEL = "naver/splade-v3"


def test_splade_embedder_np():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences)  # default return type is numpy array

    assert isinstance(embeddings, np.ndarray), "Embeddings should be a numpy array"

    similarity = embedder.similarity(embeddings, embeddings)
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values: dict[str, float] = embedder.get_token_values(embeddings[0])  # type: ignore
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values

    token_values_list: list[dict[str, float]] = embedder.get_token_values(embeddings)  # type: ignore
    token_values = token_values_list[0]
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values


def test_splade_embedder_csr_matrix():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences, convert_to_csr_matrix=True)

    assert isinstance(
        embeddings, scipy.sparse.csr_matrix
    ), "Embeddings should be a csr_matrix"

    similarity = embedder.similarity(embeddings, embeddings)
    similarity = similarity.toarray()  # type: ignore
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values: dict[str, float] = embedder.get_token_values(embeddings[0])  # type: ignore
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values

    token_values_list: list[dict[str, float]] = embedder.get_token_values(embeddings)  # type: ignore
    token_values = token_values_list[0]
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values


def test_encode_args_error():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    with pytest.raises(ValueError):
        embedder.encode(
            ["Hello, my dog is cute"], convert_to_csr_matrix=True, convert_to_numpy=True
        )


def test_rank_without_documents():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    query = "What programming language is best for machine learning?"
    documents = [
        "Python is widely used in machine learning due to its extensive libraries like TensorFlow and PyTorch",
        "JavaScript is primarily used for web development and front-end applications",
        "SQL is essential for database management and data manipulation",
    ]

    results: list[RankResult] = embedder.rank(query, documents)  # type: ignore

    assert len(results) == 3
    assert isinstance(results[0]["corpus_id"], int)
    assert isinstance(results[0]["score"], float)
    assert "text" not in results[0]

    # First result should be about Python and ML
    assert results[0]["corpus_id"] == 0
    # Score ordering should be maintained
    assert results[0]["score"] > results[1]["score"]
    assert results[1]["score"] > results[2]["score"]


def test_rank_with_documents():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    query = "What programming language is best for machine learning?"
    documents = [
        "Python is widely used in machine learning due to its extensive libraries like TensorFlow and PyTorch",
        "JavaScript is primarily used for web development and front-end applications",
        "SQL is essential for database management and data manipulation",
    ]

    results: list[RankResultWithText] = embedder.rank(
        query, documents, return_documents=True
    )  # type: ignore

    assert len(results) == 3
    assert isinstance(results[0]["corpus_id"], int)
    assert isinstance(results[0]["score"], float)
    assert isinstance(results[0]["text"], str)
    assert results[0]["text"] == documents[results[0]["corpus_id"]]

    assert results[0]["corpus_id"] == 0
    assert "PyTorch" in results[0]["text"]
    # Score ordering should be maintained
    assert results[0]["score"] > results[1]["score"]
    assert results[1]["score"] > results[2]["score"]
