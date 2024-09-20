## YASEM (Yet Another Splade|Sparse Embedder)

YASEM is a simple and efficient library for executing SPLADE (Sparse Lexical and Expansion Model for Information Retrieval) and creating sparse vectors. It provides a straightforward interface inspired by [SentenceTransformers](https://sbert.net/) for easy integration into your projects.

## Why YASEM?

- Simplicity: YASEM focuses on providing a clean and simple implementation of SPLADE without unnecessary complexity.
- Efficiency: Generate sparse embeddings quickly and easily.
- Flexibility: Works with both NumPy and PyTorch backends.
- Convenience: Includes helpful utilities like get_token_values for inspecting feature representations.

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
    "Hello, I like a ramen",
    "Hello, I like a sushi",
]

# Generate embeddings
embeddings = embedder.encode(sentences)
# or sparse csr matrix
# embeddings = embedder.encode(sentences, convert_to_csr_matrix=True)

# Compute similarity
similarity = embedder.similarity(embeddings, embeddings)
print(similarity)
# [[148.62903569 106.88184372  18.86930016  22.87525314]
#  [106.88184372 122.79656474  17.45339064  21.44758757]
#  [ 18.86930016  17.45339064  61.00272733  40.92700849]
#  [ 22.87525314  21.44758757  40.92700849  73.98511539]]


# Inspect token values for the first sentence
token_values = embedder.get_token_values(embeddings[0])
print(token_values)
# {'hello': 6.89453125, 'dog': 6.48828125, 'cute': 4.6015625,
#  'message': 2.38671875, 'greeting': 2.259765625,
#    ...

token_values = embedder.get_token_values(embeddings[3])
print(token_values)
# {'##shi': 3.63671875, 'su': 3.470703125, 'eat': 3.25,
#  'hello': 2.73046875, 'you': 2.435546875, 'like': 2.26953125, 'taste': 1.8203125,
```

## Features

- Easy-to-use API inspired by SentenceTransformers
- Support for both NumPy and scipy.sparse.csr_matrix
- Efficient dot product similarity computation
- Utility function to inspect token values in embeddings

## License

This project is licensed under the MIT License. See the LICENSE file for the full license text. Copyright (c) 2024 Yuichi Tateno (@hotchpotch)

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## Acknowledgements

This library is inspired by the SPLADE model and aims to provide a simple interface for its usage. Special thanks to the authors of the original SPLADE paper and the developers of the model.