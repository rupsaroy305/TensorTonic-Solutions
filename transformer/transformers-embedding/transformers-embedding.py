import torch
import torch.nn as nn

def create_embedding_layer(vocab_size: int, d_model: int) -> nn.Embedding:
        return torch.nn.Embedding(vocab_size, d_model)


def embed_tokens(embedding: nn.Embedding, tokens: torch.Tensor, d_model: int) -> torch.Tensor:
        return embedding(tokens) * math.sqrt(embedding.embedding_dim)