import torch
import torch.nn.functional as F
from typing import Dict, List

from sentence_transformers import SentenceTransformer
from transformers import (
    AutoModel,
    AutoTokenizer,
    PreTrainedModel,
    PreTrainedTokenizer
)


class TextEmbedder:
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


class CLIPViTEmbedder:
    def __init__(self, embedding_model) -> None:
        self.model = embedding_model
        self.dimension = 768 # TODO

    def __call__(self, query_or_image) -> torch.Tensor:
        embedding = self.model.encode(query_or_image)
        embedding_tensor = torch.from_numpy(embedding).to(self.model.device)
        if embedding_tensor.dim() == 1:
            # If the embedding is a single vector, unsqueeze to make it a batch of size 1
            # This is necessary for consistency with batch processing
            # and to ensure the output is always a 2D tensor
            embedding_tensor = embedding_tensor.unsqueeze(0)
        return embedding_tensor


def load_text_embedder(model_name="sentence-transformers/all-mpnet-base-v2"):
    embedding_model_tokenizer = AutoTokenizer.from_pretrained(model_name)
    embedding_model = AutoModel.from_pretrained(model_name, device_map="auto", trust_remote_code=True)
    embedder = TextEmbedder(embedding_model, embedding_model_tokenizer)
    return embedder


def load_clip_vit_embedder(model_name="sentence-transformers/clip-ViT-L-14"):
    embedding_model = SentenceTransformer(model_name).to("cuda" if torch.cuda.is_available() else "cpu")
    embedder = CLIPViTEmbedder(embedding_model)
    return embedder


def get_multimodal_embedding(
    text_embedding: torch.tensor,
    image_embedding: torch.tensor
) -> torch.tensor:
    text_embedding = text_embedding.view(1, -1)
    image_embedding = image_embedding.view(1, -1)
    # Concatenate along the last dimension
    combined_embedding = torch.cat((text_embedding, image_embedding), dim=1)
    return combined_embedding
