# -*- coding: utf-8 -*-
"""Function to test models."""

import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torcheval.metrics.functional.text import bleu_score

from .model import TransformerNet


def test_model(
    model: TransformerNet,
    test_loader: DataLoader,
    data_captions: pd.DataFrame,
    vocab: dict[str, int],
    device: torch.device,
) -> float:
    """Test the model.

    :param model: Model to test.
    :param test_loader: Test data loader.
    :param data_captions: Data Captions.
    :param vocab: Vocabulary.
    :param device: Device to use.
    :return: Test loss and BLEU score.
    """
    model.eval()

    decoded_predictions = []
    decoded_targets = []

    id2word = {id_: word for word, id_ in vocab.items()}

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]
    max_len = 50

    with torch.no_grad():
        for inputs, vid_ids in tqdm.tqdm(test_loader, "Testing"):
            for vid_id in vid_ids:
                vid_id = vid_id.item()
                targets = [
                    _convert_tensor_to_caption(_convert_tokens_to_ids(cap, vocab), id2word)
                    for cap in data_captions[data_captions["video"] == vid_id]["caption"]
                ]

                decoded_targets.append(targets)

            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            captions = torch.full(
                (batch_size, max_len), fill_value=eos_id, dtype=torch.long, device=device
            )
            captions[:, 0] = sos_id

            for t in range(1, max_len):
                outputs = model(inputs, captions[:, :t])
                next_word_logits = outputs[:, t - 1, :]
                captions[:, t] = next_word_logits.argmax(-1)

            [
                decoded_predictions.append(_convert_tensor_to_caption(output, id2word))
                for output in captions
            ]

    score = bleu_score(decoded_predictions, decoded_targets, 4).item()

    logger.info("Test BLEU score: {}", score)

    for example_idx in np.random.default_rng().integers(0, len(decoded_predictions), 5):
        logger.info("Trgt: {}", decoded_targets[example_idx][:2])
        logger.info("Pred: {}", decoded_predictions[example_idx])

    return score


def _convert_captions_to_tensor(
    captions: list[str], vocab: dict[str, int]
) -> tuple[torch.Tensor, torch.Tensor]:
    padded_captions_str = pad_sequence(
        [torch.tensor(_convert_tokens_to_ids("<sos> " + tokens, vocab)) for tokens in captions],
        batch_first=True,
    )

    padded_captions_end = pad_sequence(
        [torch.tensor(_convert_tokens_to_ids(tokens + " <eos>", vocab)) for tokens in captions],
        batch_first=True,
    )

    return padded_captions_str, padded_captions_end


def _convert_tokens_to_ids(tokens: str, vocab: dict[str, int]) -> list[int]:
    return [vocab.get(token, 1) for token in tokens.split()]


def _convert_tensor_to_caption(
    caption_indices: torch.Tensor | list[int], id2word: dict[int, str]
) -> str:
    """Decode a caption from token indices to words using the vocabulary.

    :param caption_indices: Tensor of token indices representing a caption.
    :param id2word: Dictionary mapping token indices to words.
    :return: Decoded caption as a string.
    """
    if isinstance(caption_indices, torch.Tensor):
        caption_indices = caption_indices.cpu().numpy()
    words = [id2word.get(idx_, "<unk>") for idx_ in caption_indices]

    if "<eos>" in words:
        words = words[: words.index("<eos>")]

    words = [word for word in words if word not in ["<sos>", "<eos>", "<pad>"]]

    return " ".join(words)
