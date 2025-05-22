## YASEM: Yet Another Splade|Sparse Embedder ✨

Ever wished for a straightforward way to leverage the power of SPLADE? Look no further! YASEM (Yet Another Splade|Sparse Embedder) is your go-to library for seamlessly executing SPLADE, a cutting-edge technique that helps understand text by focusing on the most important words and smartly expanding on them for superior matching 🚀. We've designed YASEM with an easy-to-use interface, taking inspiration from the popular [SentenceTransformers](https://sbert.net/), to ensure you can integrate sparse vector creation into your projects with minimal fuss.

## Why YASEM?

- ✨ Streamlined Experience: Get started quickly with a clean and focused SPLADE implementation, free of unnecessary complexity.
- ⚡️ Peak Performance: Generate sparse embeddings rapidly for your demanding tasks.
- 🤸 Backend Agility: Seamlessly switch between NumPy and PyTorch to suit your workflow.
- 🛠️ Insightful Utilities: Easily inspect and understand your feature representations with tools like `get_token_values`.

## Installation

Getting YASEM up and running is a breeze! Simply install it using pip:

```bash
pip install yasem
```

## Quick Start

Let's dive in with a quick example to see YASEM in action:

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

## rank API

Effortlessly rank documents against a query to find the most relevant information. Here’s how you can use the `rank` API:

```python
# Rank documents based on query
query = "What programming language is best for machine learning?"
documents = [
   "Python is widely used in machine learning due to its extensive libraries like TensorFlow and PyTorch",
   "JavaScript is primarily used for web development and front-end applications", 
   "SQL is essential for database management and data manipulation"
]

# Get ranked results with relevance scores
results = embedder.rank(query, documents)
print(results)
# [
#   {'corpus_id': 0, 'score': 12.453},  # Python/ML document ranks highest
#   {'corpus_id': 2, 'score': 5.234},
#   {'corpus_id': 1, 'score': 3.123}
# ]

# Get ranked results including document text
results = embedder.rank(query, documents, return_documents=True)
print(results)  
# [
#   {
#     'corpus_id': 0,
#     'score': 12.453,
#     'text': 'Python is widely used in machine learning due to its extensive libraries like TensorFlow and PyTorch'
#   },
#   {
#     'corpus_id': 2, 
#     'score': 5.234,
#     'text': 'SQL is essential for database management and data manipulation'
#   },
#   ...
# ]
```

## 🎯 Features

- User-friendly API, thoughtfully inspired by SentenceTransformers for a familiar feel.
- Flexible output formats: Works with both NumPy arrays and `scipy.sparse.csr_matrix` for your convenience.
- Blazing-fast similarity scores: Utilizes efficient dot product computations.
- Deeper insights: Comes with a handy utility function to inspect token values within your sparse embeddings.

## License

This project is licensed under the MIT License. See the LICENSE file for the full license text. Copyright (c) 2024 Yuichi Tateno (@hotchpotch)

## Contributing 🤝

We warmly welcome contributions! If you have ideas for improvements or new features, please feel free to submit a Pull Request. We appreciate your help in making YASEM even better!

## Acknowledgements

YASEM draws its inspiration from the innovative SPLADE model and strives to offer a user-friendly interface for its powerful capabilities. Our heartfelt thanks go out to the brilliant authors of the original SPLADE paper and the talented developers behind the model.