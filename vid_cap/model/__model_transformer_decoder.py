# -*- coding: utf-8 -*-
"""Model Builder - decoder."""
from torch import nn

from vid_cap.model.__positional_encoding import PositionalEncoding


class TransformerNet(nn.Module):
    def __init__(
        self,
        num_src_vocab,
        num_tgt_vocab,
        embedding_dim,
        hidden_size,
        nheads,
        n_layers,
        max_src_len,
        max_tgt_len,
        dropout,
    ) -> None:
        super().__init__()
        # embedding layer
        self.dec_embedding = nn.Embedding(len(unique_token_dictionary), embedding_dim)

        # positional encoding layer
        self.dec_pe = PositionalEncoding(embedding_dim, max_len=max_tgt_len)

        # encoder/decoder layer
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, hidden_size, dropout)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # final dense layer
        self.dense = nn.Linear(embedding_dim, num_tgt_vocab)
        self.log_softmax = nn.LogSoftmax()

    def forward(self, src, tgt):
        tgt = self.dec_embedding(tgt).permute(1, 0, 2)
        tgt = self.dec_pe(tgt)
        memory = self.encoder(src)
        transformer_out = self.decoder(tgt, memory)
        final_out = self.dense(transformer_out)
        return self.log_softmax(final_out)

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)
        return None
