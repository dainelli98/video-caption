# -*- coding: utf-8 -*-
"""Function to test models."""
import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from torch.utils.data import DataLoader
from torcheval.metrics.functional.text import bleu_score

from . import _utils as utils
from .model import TransformerNet


def test_model(
    model: TransformerNet,
    test_loader: DataLoader,
    data_captions: pd.DataFrame,
    vocab: dict[str, int],
    device: torch.device,
    use_bpe: bool = False,
) -> float:
    """Test the model.

    :param model: Model to test.
    :param test_loader: Test data loader.
    :param data_captions: Data Captions.
    :param vocab: Vocabulary.
    :param device: Device to use.
    :param use_bpe: Whether to use BPE.
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
                    (
                        utils.convert_tensor_to_caption(
                            utils.convert_tokens_to_ids(cap, vocab), id2word, use_bpe
                        )
                        if not use_bpe
                        else cap
                    )
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
                decoded_predictions.append(
                    utils.convert_tensor_to_caption(output, id2word, use_bpe)
                )
                for output in captions
            ]

    score = bleu_score(decoded_predictions, decoded_targets, 4).item()

    logger.info("Test BLEU score: {}", score)

    for example_idx in np.random.default_rng().integers(0, len(decoded_predictions), 5):
        logger.info("Trgt: {}", decoded_targets[example_idx][:5])
        logger.info("Pred: {}", decoded_predictions[example_idx])

    return score
