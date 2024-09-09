# test_splade_embedder.py
import numpy as np
import torch

from yasem import SpladeEmbedder

SPLADE_MODEL = "naver/splade-v3"


def test_splade_embedder_np():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    assert isinstance(embeddings, np.ndarray), "Embeddings should be a numpy array"

    similarity = embedder.similarity(embeddings, embeddings)
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values = embedder.get_token_values(embeddings[0])
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values


def test_splade_embedder_torch():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences, convert_to_numpy=False)

    assert isinstance(embeddings, torch.Tensor), "Embeddings should be a torch tensor"

    similarity = embedder.similarity(embeddings, embeddings)
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values = embedder.get_token_values(embeddings[0])
    assert "dog" in token_values
    assert token_values["dog"] > 0.0
    assert "ramen" not in token_values
