import torch
import torch.nn.functional as F
from typing import Dict, List

from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


EMBEDDING_MODEL_PATH = "sentence-transformers/all-mpnet-base-v2"


class Embedder:
    def __init__(self, embedding_model: PreTrainedModel, tokenizer: PreTrainedTokenizer) -> None:
        self.embedding_model = embedding_model
        self.tokenizer = tokenizer
        self.dimension = embedding_model.config.hidden_size

    def __call__(self, sentences: List[str]) -> torch.Tensor:
        encoded_input = self.tokenizer(sentences, padding=True, truncation=True, return_tensors='pt')
        encoded_input = {k: v.to(self.embedding_model.device) for k, v in encoded_input.items()}
        # Compute token embeddings
        with torch.no_grad():
            model_output = self.embedding_model(**encoded_input)
        
        # Perform pooling
        sentence_embeddings = self.mean_pooling(model_output, encoded_input['attention_mask'])

        # Normalize embeddings
        sentence_embeddings = F.normalize(sentence_embeddings, p=2, dim=1)

        return sentence_embeddings

    # Mean Pooling - Take attention mask into account for correct averaging
    @classmethod
    def mean_pooling(cls, model_output: Dict[str, torch.Tensor], attention_mask: torch.Tensor) -> torch.Tensor:
        token_embeddings = model_output[0]  # First element of model_output contains all token embeddings
        input_mask_expanded = attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
        return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(input_mask_expanded.sum(1), min=1e-9)


def load_embedder():
    embedding_model_tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODEL_PATH)
    embedding_model = AutoModel.from_pretrained(EMBEDDING_MODEL_PATH, device_map={"": 0}, trust_remote_code=True)
    embedder = Embedder(embedding_model, embedding_model_tokenizer)
    return embedder
