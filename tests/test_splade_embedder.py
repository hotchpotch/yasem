

from yasem.splade_embedder import SpladeEmbedder

SPLADE_MODEL = "naver/splade-v3"

def test_splade_embedder():
    embedder = SpladeEmbedder(SPLADE_MODEL)
    sentences = ["Hello, my dog is cute", "Hello, my cat is cute"]
    embeddings = embedder.encode(sentences)
    print(embeddings)
