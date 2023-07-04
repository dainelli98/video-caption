# -*- coding: utf-8 -*-
"""Model Builder - decoder."""
import torch
from torch import nn

from ._pos_enc import PositionalEncoding


class TransformerNet(nn.Module):
    """Transformer decoder for video captioning.

    :param vocab_size: Size of vocabulary.
    :param embedding_dim: Embedding dimension.
    :param nheads: Number of attention heads.
    :param n_layers: Number of layers.
    :param max_seq_len: Maximum length of sequence.
    """

    def __init__(
        self, vocab_size: int, embedding_dim: int, nheads: int, n_layers: int, max_seq_len: int
    ) -> None:
        super().__init__()
        # embedding layer
        self.dec_embedding = nn.Embedding(vocab_size, embedding_dim)

        # positional encoding layer
        self.dec_pe = PositionalEncoding(embedding_dim, max_len=max_seq_len)

        # encoder/decoder layer
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, activation="gelu", batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # final dense layer
        self.dense = nn.Linear(embedding_dim, vocab_size)

    def forward(self, src: torch.Tensor, tgt: torch.Tensor) -> torch.Tensor:
        tgt = self.dec_embedding(tgt)
        tgt = self.dec_pe(tgt)
        src = self.dec_pe(src)
        transformer_out = self.decoder(tgt, src)
        return self.dense(transformer_out)
