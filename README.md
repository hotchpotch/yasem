# YASEM (Yet Another Splade|Sparse Embedder)

YASEM is a simple and efficient library for executing SPLADE (Sparse Lexical and Expansion Model for Information Retrieval) and creating sparse vectors. It provides a straightforward interface inspired by SentenceTransformers for easy integration into your projects.

## Why YASEM?

- **Simplicity**: YASEM focuses on providing a clean and simple implementation of SPLADE without unnecessary complexity.
- **Efficiency**: Generate sparse embeddings quickly and easily.
- **Flexibility**: Works with both NumPy and PyTorch backends.
- **Convenience**: Includes helpful utilities like `get_token_values` for inspecting feature representations.

## Installation

You can install YASEM using pip:

```bash
pip install yasem
```

## Quick Start

Here's a simple example of how to use YASEM:

```python
from yasem import SpladeEmbedder

# Initialize the embedder
embedder = SpladeEmbedder("naver/splade-v3")

# Prepare some sentences
sentences = [
    "Hello, my dog is cute",
    "Hello, my cat is cute",
    "Hello, I like ramen",
]

# Generate sparse embeddings
embeddings = embedder.encode(sentences)

# Compute similarity
similarity = embedder.similarity(embeddings, embeddings)
print(similarity)
# [[148.46356781 106.77744783  17.91641146]
#  [106.77744783 122.72133482  16.49506931]
#  [ 17.91641146  16.49506931  48.50924806]]


# Inspect token values for the first sentence
token_values = embedder.get_token_values(embeddings[0])
print(token_values)
# {'hello': 6.89453125, 'dog': 6.48828125, 'cute': 4.6015625,
#  'message': 2.38671875, 'greeting': 2.259765625,
#    ...
```

## Features

- Easy-to-use API inspired by [SentenceTransformers](https://sbert.net/)
- Support for both NumPy (scipy.sparse) and PyTorch sparse tensors
- Customizable similarity functions (dot product and cosine similarity)
- Utility function to inspect token values in embeddings

## License

This project is licensed under the MIT License. See the [LICENSE](LICENSE) file for the full license text. Copyright (c) 2024 Yuichi Tateno (@hotchpotch)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This library is inspired by the SPLADE model and aims to provide a simple interface for its usage. Special thanks to the authors of the original SPLADE paper and the developers of the model.