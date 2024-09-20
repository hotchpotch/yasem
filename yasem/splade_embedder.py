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
        embeddings = torch.log(1 + torch.relu(logits)) * attention_mask.unsqueeze(-1)
        return embeddings.sum(dim=1)

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
        use_fp16: bool = True,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = (
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

    def encode(
        self,
        sentences: List[str],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_csr_matrix: Optional[bool] = None,
        convert_to_numpy: Optional[bool] = None,
    ) -> Union[csr_matrix, np.ndarray]:
        if convert_to_csr_matrix is None and convert_to_numpy is None:
            convert_to_numpy = True
        if convert_to_csr_matrix is True and convert_to_numpy is True:
            raise ValueError(
                "Only one of convert_to_csr_matrix or convert_to_numpy can be True"
            )

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
                batch, padding=True, truncation=True, return_tensors="pt"
            ).to(self.device)

            # Get SPLADE embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = self.splade_max(outputs.logits, inputs["attention_mask"])  # type: ignore

            embeddings = embeddings.cpu()

            # Vectorized extraction of non-zero elements
            rows_batch, cols_batch = embeddings.nonzero(as_tuple=True)
            data_batch = embeddings[rows_batch, cols_batch]

            # Adjust row indices
            rows.extend((rows_batch + current_row).tolist())
            cols.extend(cols_batch.tolist())
            data.extend(data_batch.tolist())

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

    def __call__(self, sentences: list[str], **kwargs):
        return self.encode(sentences, **kwargs)

    def get_token_values(
        self,
        embedding: Union[np.ndarray, csr_matrix],
        top_k: Optional[int] = None,
    ) -> dict[str, float]:
        if isinstance(embedding, csr_matrix):
            embedding = embedding.toarray()  # type: ignore

        if embedding.ndim > 1:
            embedding = embedding.squeeze()

        token_values = {
            self.tokenizer.convert_ids_to_tokens(idx): float(val)
            for idx, val in enumerate(embedding)
            if val > 0
        }

        sorted_token_values = sorted(
            token_values.items(), key=lambda x: x[1], reverse=True
        )
        if top_k is not None:
            sorted_token_values = sorted_token_values[:top_k]

        return dict(sorted_token_values)  # type: ignore
