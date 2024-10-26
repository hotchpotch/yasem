from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from scipy.sparse import csr_matrix
from tqdm import tqdm
from transformers import AutoConfig, AutoModelForMaskedLM, AutoTokenizer


class SpladeEmbedder:
    @staticmethod
    def splade_max(logits: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        """
        Compute SPLADE max pooling.

        Args:
            logits (torch.Tensor): The output logits from the model.
            attention_mask (torch.Tensor): The attention mask for the input.

        Returns:
            torch.Tensor: The SPLADE embedding.
        """
        if logits.dim() != 3 or attention_mask.dim() != 2:
            raise ValueError("Invalid input dimensions")

        if logits.size(0) != attention_mask.size(0) or logits.size(
            1
        ) != attention_mask.size(1):
            raise ValueError("Mismatched batch size or sequence length")

        embeddings = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        # Perform max pooling instead of sum pooling
        return torch.max(embeddings, dim=1).values

    def __init__(
        self,
        model_name_or_path: str,
        device: Optional[Literal["cuda", "cpu", "mps", "npu"]] = None,
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        model_kwargs: Optional[Dict] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        config_kwargs: Optional[Dict] = None,
        max_seq_length: int | None = 512,
        use_fp16: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device: Literal["cuda", "cpu", "mps", "npu"] = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )

        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            **(tokenizer_kwargs or {}),
        )

        if config_kwargs is not None:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                token=token,
                **config_kwargs,
            )
            self.model = AutoModelForMaskedLM.from_config(config).to(self.device)
        else:
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                token=token,
                **(model_kwargs or {}),
            ).to(self.device)

        if use_fp16:
            try:
                self.model = self.model.half()
            except Exception:
                print("Warning: Could not convert model to FP16. Continuing with FP32.")

        self.max_seq_length = max_seq_length
        self.vocab_size = self.tokenizer.vocab_size
        # Precompute token mapping for performance
        self.id_to_token = self.tokenizer.convert_ids_to_tokens(
            list(range(self.vocab_size))
        )

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_csr_matrix: Optional[bool] = None,
        convert_to_numpy: Optional[bool] = None,
        device: Optional[Literal["cuda", "cpu", "mps", "npu"]] = None,
    ) -> Union[csr_matrix, np.ndarray]:
        if convert_to_csr_matrix is None and convert_to_numpy is None:
            convert_to_numpy = True
        if convert_to_csr_matrix is True and convert_to_numpy is True:
            raise ValueError(
                "Only one of convert_to_csr_matrix or convert_to_numpy can be True"
            )

        if device is None:
            device = self.device  # type: ignore

        data = []
        rows = []
        cols = []
        current_row = 0
        vocab_size = None

        # Create iterator with tqdm if show_progress_bar is True
        iterator = tqdm(
            range(0, len(sentences), batch_size),
            desc="Encoding",
            disable=not show_progress_bar,
        )

        for i in iterator:
            batch = sentences[i : i + batch_size]

            # Tokenize and prepare input
            inputs = self.tokenizer(
                batch,
                padding=True,
                truncation=True,
                return_tensors="pt",
                max_length=self.max_seq_length,
            ).to(device)

            # Get SPLADE embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.splade_max(outputs.logits, inputs["attention_mask"])  # type: ignore

            embeddings = embeddings.cpu()

            # Use torch operations to find non-zero elements
            non_zero = embeddings > 0
            rows_batch, cols_batch = torch.nonzero(non_zero, as_tuple=True)
            data_batch = embeddings[non_zero].tolist()

            # Adjust row indices
            rows.extend((rows_batch + current_row).tolist())
            cols.extend(cols_batch.tolist())
            data.extend(data_batch)

            if vocab_size is None:
                vocab_size = embeddings.size(1)

            current_row += embeddings.size(0)

        if vocab_size is None:
            vocab_size = 0

        all_embeddings = csr_matrix(
            (data, (rows, cols)), shape=(len(sentences), vocab_size)
        )

        if convert_to_csr_matrix:
            # no-op
            pass
        elif convert_to_numpy:
            all_embeddings = all_embeddings.toarray()
        return all_embeddings

    def similarity(
        self,
        embeddings1: Union[np.ndarray, csr_matrix],
        embeddings2: Union[np.ndarray, csr_matrix],
    ) -> Union[np.ndarray, csr_matrix]:
        """
        Compute similarity between two sets of embeddings.

        Args:
            embeddings1 (Union[np.ndarray, csr_matrix]): The first set of embeddings.
            embeddings2 (Union[np.ndarray, csr_matrix]): The second set of embeddings.

        Returns:
            Union[np.ndarray, csr_matrix]: The similarity matrix.
        """
        # Check if both embeddings are numpy arrays
        if isinstance(embeddings1, np.ndarray) and isinstance(embeddings2, np.ndarray):
            return np.dot(embeddings1, embeddings2.T)

        # Check if both embeddings are csr_matrix
        elif isinstance(embeddings1, csr_matrix) and isinstance(
            embeddings2, csr_matrix
        ):
            return embeddings1.dot(embeddings2.T)

        else:
            raise ValueError(
                "Both inputs must be of the same type (either numpy.ndarray or csr_matrix)"
            )

    def __call__(self, sentences: List[str], **kwargs):
        return self.encode(sentences, **kwargs)

    def get_token_values(
        self,
        embedding: Union[np.ndarray, csr_matrix],
        top_k: Optional[int] = None,
    ) -> Union[Dict[str, float], List[Dict[str, float]]]:
        """
        Extract token values from embeddings.

        Args:
            embedding (Union[np.ndarray, csr_matrix]): The embedding(s) from which to extract token values.
            top_k (Optional[int], optional): The number of top tokens to return. If None, all tokens with non-zero values are returned.

        Returns:
            Union[Dict[str, float], List[Dict[str, float]]]: A dictionary for single embeddings or a list of dictionaries for multiple embeddings.
        """

        def extract_token_values(
            tokens: List[str], values: List[float], top_k: Optional[int]
        ) -> Dict[str, float]:
            token_values = {
                token: float(val) for token, val in zip(tokens, values) if val > 0
            }
            sorted_tokens = sorted(
                token_values.items(), key=lambda x: x[1], reverse=True
            )
            if top_k is not None:
                sorted_tokens = sorted_tokens[:top_k]
            return dict(sorted_tokens)

        def process_csr_matrix(emb: csr_matrix) -> Dict[str, float]:
            tokens = [self.id_to_token[idx] for idx in emb.indices]
            values = emb.data
            return extract_token_values(tokens, values, top_k)  # type: ignore

        def process_dense_array(emb: np.ndarray) -> Dict[str, float]:
            non_zero_indices = np.nonzero(emb > 0)[0]
            tokens = [self.id_to_token[idx] for idx in non_zero_indices]
            values = emb[non_zero_indices]
            return extract_token_values(tokens, values, top_k)  # type: ignore

        # Ensure embedding is 2D for uniform processing
        if isinstance(embedding, csr_matrix):
            if embedding.ndim == 1:
                embedding = embedding.reshape(1, -1)
            elif embedding.ndim != 2:
                raise ValueError("csr_matrix input must be 1D or 2D")

            results = [process_csr_matrix(emb) for emb in embedding]

        elif isinstance(embedding, np.ndarray):
            if embedding.ndim == 1:
                embedding = embedding[np.newaxis, :]
            elif embedding.ndim != 2:
                raise ValueError("numpy.ndarray input must be 1D or 2D")

            results = [process_dense_array(emb) for emb in embedding]

        else:
            raise TypeError(
                "Embedding must be either a numpy.ndarray or a scipy.sparse.csr_matrix"
            )

        if len(results) == 1:
            return results[0]
        return results
