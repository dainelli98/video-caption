# -*- coding: utf-8 -*-
"""Model Builder - decoder."""
from torch import nn

from vid_cap.model.__positional_encoding import PositionalEncoding


class TransformerNet(nn.Module):
    def __init__(
        self,
        num_tgt_vocab,
        embedding_dim,
        vocab_size,
        nheads,
        n_layers,
        max_tgt_len,
    ) -> None:
        super().__init__()
        # embedding layer
        self.dec_embedding = nn.Embedding(vocab_size, embedding_dim)

        # positional encoding layer
        self.dec_pe = PositionalEncoding(embedding_dim, max_len=max_tgt_len)

        # encoder/decoder layer
        dec_layer = nn.TransformerDecoderLayer(embedding_dim, nheads, batch_first=True)
        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)

        # final dense layer
        self.dense = nn.Linear(embedding_dim, num_tgt_vocab)

    def forward(self, src, tgt):
        tgt = self.dec_embedding(tgt)
        tgt = self.dec_pe(tgt)
        src = self.dec_pe(src)
        transformer_out = self.decoder(tgt, src)
        final_out = self.dense(transformer_out)

        return final_out

        self.decoder = nn.TransformerDecoder(dec_layer, num_layers=n_layers)