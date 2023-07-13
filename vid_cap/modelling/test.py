# -*- coding: utf-8 -*-
"""Function to test models."""

import numpy as np
import torch
import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torcheval.metrics import BLEUScore

from .model import TransformerNet


def test_model(
    model: TransformerNet,
    test_loader: DataLoader,
    vocab: dict[str, int],
    device: torch.device,
) -> float:
    """Test the model.

    :param model: Model to test.
    :param test_loader: Test data loader.
    :param vocab: Vocabulary.
    :param device: Device to use.
    :return: Test loss and BLEU score.
    """
    model.eval()

    decoded_predictions = []
    decoded_targets = []
    bleu_metric = BLEUScore(n_gram=4)
    bleu_scores = []

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]
    max_len = 50

    with torch.no_grad():
        for data in tqdm.tqdm(test_loader, "Testing"):
            inputs, captions = data

            _, captions_end = _convert_captions_to_tensor(list(captions), vocab)

            inputs = inputs.to(device)
            batch_size = inputs.size(0)
            captions = torch.full(
                (batch_size, max_len), fill_value=eos_id, dtype=torch.long, device=device
            )
            captions[:, 0] = sos_id

            captions_end = captions_end.to(device)

            for t in range(1, max_len):
                outputs = model(inputs, captions[:, :t])
                next_word_logits = outputs[:, t - 1, :]
                captions[:, t] = next_word_logits.argmax(-1)

            decoded_predictions = [_convert_tensor_to_caption(output, vocab) for output in captions]

            # Compute BLEU score
            decoded_targets = [
                _convert_tensor_to_caption(caption, vocab) for caption in captions_end
            ]

            bleu_metric.update(decoded_targets, decoded_predictions)
            bleu_scores.append(bleu_metric.compute())

    avg_bleu_metric = torch.mean(torch.stack(bleu_scores))
    logger.info("Test BLEU score: {}", avg_bleu_metric)

    for example_idx in np.random.default_rng().integers(0, len(decoded_predictions), 5):
        logger.info("Trgt: {}", decoded_targets[example_idx])
        logger.info("Pred: {}", decoded_predictions[example_idx])

    return avg_bleu_metric


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


def _convert_tensor_to_caption(caption_indices: torch.Tensor, vocab: dict[str, int]) -> str:
    """Decode a caption from token indices to words using the vocabulary.

    :param caption_indices: Tensor of token indices representing a caption.
    :param vocab: Vocabulary mapping token indices to words.
    :return: Decoded caption as a string.
    """
    caption_indices = caption_indices.cpu().numpy()
    words = []
    for idx in caption_indices:
        word = next((key for key, val in vocab.items() if val == idx), "<unk>")
        words.append(word)

    words = [word for word in words if word not in ["<sos>", "<eos>", "<pad>"]]
    return " ".join(words)
