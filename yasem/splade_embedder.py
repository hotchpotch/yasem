import torch
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoConfig
from typing import Literal, Union, List, Dict, Optional
from tqdm import tqdm
import numpy as np

class SpladeEmbedder:
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
        use_fp16: bool = True
    ):
        self.model_name_or_path = model_name_or_path
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.similarity_fn_name = similarity_fn_name
        
        # Initialize tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            trust_remote_code=trust_remote_code,
            revision=revision,
            token=token,
            **(tokenizer_kwargs or {})
        )

        if config_kwargs is not None:
            config = AutoConfig.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                token=token,
                **config_kwargs
            )
            self.model = AutoModelForMaskedLM.from_config(config).to(self.device)
        else:                
            self.model = AutoModelForMaskedLM.from_pretrained(
                model_name_or_path,
                trust_remote_code=trust_remote_code,
                revision=revision,
                token=token,
                **(model_kwargs or {})
            ).to(self.device)

        if use_fp16:
            try:
                self.model = self.model.half()
            except:
                print("Warning: Could not convert model to FP16. Continuing with FP32.")
        

    def encode(
        self,
        sentences: Union[str, List[str]],
        batch_size: int = 32,
        show_progress_bar: bool = False,
        convert_to_numpy: bool = True,
        normalize_embeddings: bool = False
    ) -> Union[np.ndarray, torch.Tensor]:
        # Ensure sentences is a list
        is_sentence_str = isinstance(sentences, str)
        if is_sentence_str:
            sentences = [sentences]
        
        all_embeddings = []
        
        # Create iterator with tqdm if show_progress_bar is True
        iterator = tqdm(range(0, len(sentences), batch_size), desc="Encoding", disable=not show_progress_bar)
        
        for i in iterator:
            batch = sentences[i:i+batch_size]
            
            # Tokenize and prepare input
            inputs = self.tokenizer(batch, padding=True, truncation=True, return_tensors="pt").to(self.device)
            
            # Get SPLADE embeddings
            with torch.no_grad():
                outputs = self.model(**inputs)
                embeddings = torch.log(1 + torch.relu(outputs.logits)) * inputs['attention_mask'].unsqueeze(-1)
                embeddings = embeddings.sum(dim=1)
            
            if normalize_embeddings:
                embeddings = torch.nn.functional.normalize(embeddings, p=2, dim=1)
            
            all_embeddings.append(embeddings)
        
        all_embeddings = torch.cat(all_embeddings, dim=0)
        
        if convert_to_numpy:
            all_embeddings = all_embeddings.cpu().numpy()
        
        if is_sentence_str:
            all_embeddings = all_embeddings[0]
        
        return all_embeddings

    def similarity(self, embeddings1: torch.Tensor | np.ndarray, embeddings2: torch.Tensor | np.ndarray) -> torch.Tensor:
        if isinstance(embeddings1, np.ndarray):
            embeddings1 = torch.from_numpy(embeddings1)
        if isinstance(embeddings2, np.ndarray):
            embeddings2 = torch.from_numpy(embeddings2)
        
        if self.similarity_fn_name == "dot":
            return torch.mm(embeddings1, embeddings2.T)
        elif self.similarity_fn_name == "cosine":
            return torch.nn.functional.cosine_similarity(embeddings1, embeddings2, dim=1)
        else:
            raise ValueError(f"Unknown similarity function: {self.similarity_fn_name}")

    def __call__(self, sentences: Union[str, List[str]], **kwargs) -> Union[np.ndarray, torch.Tensor]:
        return self.encode(sentences, **kwargs)