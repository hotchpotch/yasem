from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
from scipy import sparse
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
        similarity_fn_name: Literal["dot", "cosine"] = "dot",
        trust_remote_code: bool = False,
        revision: Optional[str] = None,
        token: Optional[str] = None,
        model_kwargs: Optional[Dict] = None,
        tokenizer_kwargs: Optional[Dict] = None,
        config_kwargs: Optional[Dict] = None,
        use_fp16: bool = True,
        sparsity_threshold: float = 1e-8,
    ):
        self.model_name_or_path = model_name_or_path
        self.device = (
            device if device else ("cuda" if torch.cuda.is_available() else "cpu")
        )
        self.similarity_fn_name = similarity_fn_name
        self.sparsity_threshold = sparsity_threshold

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
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False,
    ) -> Union[sparse.csr_matrix, torch.Tensor]:
        # Ensure sentences is a list
        is_sentence_str = isinstance(sentences, str)
        if is_sentence_str:
            sentences = [sentences]

        all_embeddings = []

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

            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)

            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            all_embeddings = all_embeddings.cpu().numpy()
            all_embeddings = all_embeddings.astype(np.float64)
            all_embeddings[np.abs(all_embeddings) < self.sparsity_threshold] = 0.0
            all_embeddings = sparse.csr_matrix(all_embeddings)
        else:
            all_embeddings = all_embeddings.to_sparse()
            mask = torch.abs(all_embeddings.values()) >= self.sparsity_threshold
            all_embeddings = torch.sparse_coo_tensor(
                all_embeddings.indices()[:, mask],
                all_embeddings.values()[mask],
                all_embeddings.size(),
            )

        if is_sentence_str:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def similarity(
        self,
        embeddings1: Union[sparse.csr_matrix, torch.Tensor],
        embeddings2: Union[sparse.csr_matrix, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(embeddings1, sparse.csr_matrix) and isinstance(
            embeddings2, sparse.csr_matrix
        ):
            if self.similarity_fn_name == "dot":
                return self._similarity_dot_scipy(embeddings1, embeddings2)
            elif self.similarity_fn_name == "cosine":
                return self._similarity_cosine_scipy(embeddings1, embeddings2)
        elif isinstance(embeddings1, torch.Tensor) and isinstance(
            embeddings2, torch.Tensor
        ):
            if self.similarity_fn_name == "dot":
                return self._similarity_dot_torch(embeddings1, embeddings2)
            elif self.similarity_fn_name == "cosine":
                return self._similarity_cosine_torch(embeddings1, embeddings2)
        else:
            raise ValueError(
                "Both inputs must be of the same type (either scipy.sparse.csr_matrix or torch.Tensor)"
            )

        raise ValueError(f"Unknown similarity function: {self.similarity_fn_name}")

    def _similarity_dot_scipy(
        self, embeddings1: sparse.csr_matrix, embeddings2: sparse.csr_matrix
    ) -> np.ndarray:
        return embeddings1.dot(embeddings2.T).toarray()

    def _similarity_cosine_scipy(
        self, embeddings1: sparse.csr_matrix, embeddings2: sparse.csr_matrix
    ) -> np.ndarray:
        norm1 = np.sqrt(embeddings1.multiply(embeddings1).sum(axis=1))
        norm2 = np.sqrt(embeddings2.multiply(embeddings2).sum(axis=1))
        dot_product = embeddings1.dot(embeddings2.T).toarray()
        return dot_product / (norm1.reshape(-1, 1) * norm2.reshape(1, -1))

    def _similarity_dot_torch(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        if embeddings1.is_sparse and embeddings2.is_sparse:
            return torch.sparse.mm(embeddings1, embeddings2.T).to_dense()
        else:
            return torch.mm(embeddings1, embeddings2.T)

    def _similarity_cosine_torch(
        self, embeddings1: torch.Tensor, embeddings2: torch.Tensor
    ) -> torch.Tensor:
        if embeddings1.is_sparse and embeddings2.is_sparse:
            norm1 = torch.sparse.sum(embeddings1.pow(2), dim=1).sqrt().unsqueeze(1)
            norm2 = torch.sparse.sum(embeddings2.pow(2), dim=1).sqrt().unsqueeze(0)
            dot_product = torch.sparse.mm(embeddings1, embeddings2.T).to_dense()
            return dot_product / (norm1 * norm2)
        else:
            return torch.nn.functional.cosine_similarity(
                embeddings1.unsqueeze(1), embeddings2.unsqueeze(0), dim=2
            )

    def __call__(
        self, sentences: Union[str, List[str]], **kwargs
    ) -> Union[sparse.csr_matrix, torch.Tensor]:
        return self.encode(sentences, **kwargs)

    def get_token_values(
        self,
        embedding: Union[sparse.csr_matrix, torch.Tensor, np.ndarray],
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get the token-value pairs from a SPLADE embedding.

        Args:
            embedding (Union[sparse.csr_matrix, torch.Tensor, np.ndarray]): The SPLADE embedding.
            top_k (Optional[int]): If specified, return only the top k token-value pairs.

        Returns:
            Dict[str, float]: A dictionary mapping tokens to their corresponding values.
        """
        if isinstance(embedding, sparse.csr_matrix):
            indices = embedding.indices
            values = embedding.data
        elif isinstance(embedding, torch.Tensor):
            if embedding.is_sparse:
                indices = embedding._indices().squeeze().cpu().numpy()
                values = embedding._values().cpu().numpy()
            else:
                indices = embedding.nonzero().squeeze().cpu().numpy()
                values = embedding[indices].cpu().numpy()
        elif isinstance(embedding, np.ndarray):
            if embedding.ndim > 1:
                embedding = embedding.squeeze()
            indices = np.nonzero(embedding)[0]
            values = embedding[indices]
        else:
            raise ValueError(
                "Embedding must be either scipy.sparse.csr_matrix, torch.Tensor, or np.ndarray"
            )

        token_values = {
            self.tokenizer.convert_ids_to_tokens(int(idx)): float(val)
            for idx, val in zip(indices, values)
        }

        if top_k is not None:
            token_values = dict(
                sorted(token_values.items(), key=lambda x: x[1], reverse=True)[:top_k]
            )
        else:
            token_values = dict(
                sorted(token_values.items(), key=lambda x: x[1], reverse=True)
            )

        return token_values  # type: ignore
