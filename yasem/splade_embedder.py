from typing import Dict, List, Literal, Optional, Union

import numpy as np
import torch
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
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
    ) -> Union[np.ndarray, torch.Tensor]:
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

            all_embeddings.append(embeddings)

        all_embeddings = torch.cat(all_embeddings, dim=0)

        if convert_to_numpy:
            all_embeddings = all_embeddings.cpu().numpy()

        if is_sentence_str:
            all_embeddings = all_embeddings[0]

        return all_embeddings

    def similarity(
        self,
        embeddings1: Union[np.ndarray, torch.Tensor],
        embeddings2: Union[np.ndarray, torch.Tensor],
    ) -> Union[np.ndarray, torch.Tensor]:
        if isinstance(embeddings1, np.ndarray) and isinstance(embeddings2, np.ndarray):
            return np.dot(embeddings1, embeddings2.T)
        elif isinstance(embeddings1, torch.Tensor) and isinstance(
            embeddings2, torch.Tensor
        ):
            return torch.matmul(embeddings1, embeddings2.T)
        else:
            raise ValueError(
                "Both inputs must be of the same type (either numpy.ndarray or torch.Tensor)"
            )

    def __call__(
        self, sentences: Union[str, List[str]], **kwargs
    ) -> Union[np.ndarray, torch.Tensor]:
        return self.encode(sentences, **kwargs)

    def get_token_values(
        self,
        embedding: Union[np.ndarray, torch.Tensor],
        top_k: Optional[int] = None,
    ) -> Dict[str, float]:
        """
        Get the token-value pairs from a SPLADE embedding.

        Args:
            embedding (Union[np.ndarray, torch.Tensor]): The SPLADE embedding.
            top_k (Optional[int]): If specified, return only the top k token-value pairs.

        Returns:
            Dict[str, float]: A dictionary mapping tokens to their corresponding values.
        """
        if isinstance(embedding, torch.Tensor):
            embedding = embedding.cpu().numpy()

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
