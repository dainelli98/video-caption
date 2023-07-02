# -*- coding: utf-8 -*-
"""Positional Encoding."""
import numpy as np
import torch
from torch import nn


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer.

    :param embedding_dim: Embedding dimension.
    :param dropout: Dropout rate. Defaults to 0.1.
    :param max_len: Maximum length of sequence. Defaults to 5000.
    """

    def __init__(self, embedding_dim: int, dropout: float = 0.1, max_len: int = 5000) -> None:
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, embedding_dim)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, embedding_dim, 2).float() * (-np.log(10000.0) / embedding_dim)
        )
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer("pe", pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.pe[: x.size(0), :]
        return self.dropout(x)
