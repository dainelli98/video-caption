# -*- coding: utf-8 -*-
"""Model Builder - decoder."""

import torch
from loguru import logger
from torch import nn

from ._pos_enc import PositionalEncoding


class TransformerNet(nn.Module):
    """Transformer decoder for video captioning.

    :param vocab_size: Size of vocabulary.
    :param embedding_dim: Embedding dimension.
    :param nheads: Number of attention heads.
    :param n_layers: Number of layers.
    :param max_seq_len: Maximum length of sequence.
    :param dropout: Dropout rate.
    :param d_ff: Feedforward dimension.
    """

    def __init__(
        self,
        vocab_size: int,
        embedding_dim: int,
        nheads: int,
        n_layers: int,
        max_seq_len: int,
        dropout: float = 0.1,
        d_ff: int = 2048,
    ) -> None:
        super().__init__()

        logger.debug(f"vocab_size: {vocab_size}")
        logger.debug(f"embedding_dim: {embedding_dim}")
        logger.debug(f"max_seq_len: {max_seq_len}")

        # embedding layer
        self.dec_embedding = nn.Embedding(vocab_size, embedding_dim)

        # positional encoding layer
        self.dec_pe = PositionalEncoding(embedding_dim, max_len=max_seq_len, dropout=dropout)

        # encoder/decoder layer
        dec_layer = nn.TransformerDecoderLayer(
            embedding_dim,
            nheads,
            activation="gelu",
            batch_first=True,
            dropout=dropout,
            dim_feedforward=d_ff,
        )
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # target sequence mask
        self.tgt_mask = None

        # final dense layer
        self.dense = nn.Linear(embedding_dim, vocab_size)

        initrange = 0.1
        self.dec_embedding.weight.data.uniform_(-initrange, initrange)

        self.embedding_dim = embedding_dim

    def _generate_square_subsequent_mask(self, sz: int) -> torch.Tensor:
        """Generate mask for target sequence.

        :param sz: Size of target sequence.
        :return: Mask for target sequence.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        return mask.float().masked_fill(mask == 0, float("-inf")).masked_fill(mask == 1, float(0.0))

    def forward(self, src: torch.Tensor, tgt: torch.Tensor, has_mask: bool = False) -> torch.Tensor:
        tgt = self.dec_embedding(tgt)
        tgt = self.dec_pe(tgt)
        src = self.dec_pe(src)

        if has_mask:
            device = tgt.device
            if self.tgt_mask is None or self.tgt_mask.size(0) != len(
                tgt[0]
            ):  # as we're working with batch
                mask = self._generate_square_subsequent_mask(len(tgt[0])).to(device)
                self.tgt_mask = mask
        else:
            self.tgt_mask = None

        transformer_out = self.decoder(tgt=tgt, memory=src, tgt_mask=self.tgt_mask)
        return self.dense(transformer_out)
