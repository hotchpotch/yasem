from scipy import sparse

from yasem.splade_embedder import SpladeEmbedder

SPLADE_MODEL = "naver/splade-v3"


def test_splade_embedder_np():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences, convert_to_numpy=True)

    # Check if embeddings is a sparse matrix
    assert sparse.issparse(embeddings), "Embeddings should be a scipy sparse matrix"
    assert isinstance(
        embeddings, sparse.csr_matrix
    ), "Embeddings should be a CSR matrix"

    # is sparse matrix check
    similarity = embedder.similarity(embeddings, embeddings)
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values = embedder.get_token_values(embeddings[0])
    assert token_values["dog"]
    assert token_values["dog"] > 0.0
    assert token_values.get("ramen") is None


def test_splade_embedder_torch():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = [
        "Hello, my dog is cute",
        "Hello, my cat is cute",
        "Hello, I like a ramen",
    ]
    embeddings = embedder.encode(sentences, convert_to_numpy=False)
    # embeddings is sparse tensor
    assert embeddings.is_sparse  # type: ignore

    similarity = embedder.similarity(embeddings, embeddings)
    assert similarity.shape == (3, 3)
    assert similarity[0][1] > similarity[0][2]
    assert similarity[0][1] > similarity[1][2]

    token_values = embedder.get_token_values(embeddings[0])
    assert token_values["dog"]
    assert token_values["dog"] > 0.0
    assert token_values.get("ramen") is None
