# -*- coding: utf-8 -*-
"""Utils for modelling module."""
import re

import torch


def convert_tokens_to_ids(tokens: str, vocab: dict[str, int]) -> list[int]:
    """Convert a string of tokens to a list of token indices.

    :param tokens: String of tokens.
    :param vocab: Vocabulary.
    :return: List of token indices.
    """
    return [vocab.get(token, 1) for token in tokens.split()]


def convert_tensor_to_caption(
    caption_indices: torch.Tensor | list[int], id2word: dict[int, str], use_bpe: bool = False
) -> str:
    """Decode a caption from token indices to words using the vocabulary.

    :param caption_indices: Tensor of token indices representing a caption.
    :param id2word: Dictionary mapping token indices to words.
    :param use_bpe: Whether to use BPE. Defaults to ``False``.
    :return: Decoded caption as a string.
    """
    if isinstance(caption_indices, torch.Tensor):
        caption_indices = caption_indices.cpu().numpy()
    words = [id2word.get(idx_, "<unk>") for idx_ in caption_indices]

    if "<eos>" in words:
        words = words[: words.index("<eos>")]

    words = [word for word in words if word not in ["<sos>", "<eos>", "<pad>"]]

    caption = " ".join(words)

    if not use_bpe:
        return caption

    return _decode_bpe(caption)


def _decode_bpe(caption_to_decode: str) -> str:
    """Decode a caption from BPE to words.

    :param caption_to_decode: Caption to decode.
    """
    return re.sub(r"(@@ )|(@@ ?$)", "", caption_to_decode)
