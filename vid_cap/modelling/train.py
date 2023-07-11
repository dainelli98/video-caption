# -*- coding: utf-8 -*-
# ruff: noqa: PLR0913
"""Train - decoder."""
import random
from pathlib import Path

import torch
import torch.nn.functional as F  # ruff: noqa: N812
import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BLEUScore

from vid_cap.modelling.scheduler import NoamOptimizer
from vid_cap.utils.early_stopper import EarlyStopper
from vid_cap.utils.model_saver import ModelSaver

from .model import TransformerNet

_LOSS_FN = torch.nn.modules.loss._Loss  # noqa: SLF001


def train(
    model: TransformerNet,
    train_loader: DataLoader | list[tuple[torch.Tensor, list[str]]],
    valid_loader: DataLoader,
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer | NoamOptimizer,
    loss_fn: _LOSS_FN,
    num_epochs: int,
    device: torch.device,
    data_dir: Path,
    model_name: str,
    tb_writer: SummaryWriter | None = None,
    label_smoothing: float = 0.0,
) -> tuple[TransformerNet, list[float], list[float], list[float]]:
    """Train model.

    :param model: Model to train.
    :param train_loader: Training data loader.
    :param valid_loader: Validation data loader.
    :param vocab: Vocabulary.
    :param optimizer: Optimizer.
    :param loss_fn: Loss function.
    :param num_epochs: Number of epochs.
    :param device: Device to use.
    :param tb_writer: Tensorboard writer.
    :param label_smoothing: Label smoothing. Defaults to ``0.0``.
    :return: Trained model and metrics.
    """
    early_stopper = EarlyStopper(patience=5, min_delta=0)
    model_saver = ModelSaver()

    train_losses = []
    val_losses = []
    bleu_scores = []

    for epoch in range(num_epochs):
        logger.info("Starting epoch {}/{}", epoch + 1, num_epochs)
        train_loss = _train_one_epoch(
            model,
            train_loader,
            vocab,
            optimizer,
            loss_fn,
            epoch,
            device,
            tb_writer,
            label_smoothing,
        )
        train_losses.append(train_loss)

        logger.info("End of epoch {}/{}. Train loss: {}", epoch + 1, num_epochs, train_loss)
        val_loss, bleu = _validate_one_epoch(
            model, valid_loader, vocab, loss_fn, epoch, device, tb_writer, label_smoothing
        )

        val_losses.append(val_loss)
        bleu_scores.append(bleu)

        logger.info("End of epoch {}/{}. Validation loss: {}", epoch + 1, num_epochs, val_loss)
        logger.info("End of epoch {}/{}. Validation BLEU: {}", epoch + 1, num_epochs, bleu)
        if isinstance(optimizer, NoamOptimizer):
            logger.info("Learning rate: {}", optimizer.lr)

        model_saver.save_if_best_model(
            val_loss, model, data_dir, f"{model_name}-last_epoch_{epoch + 1}"
        )

        if early_stopper.early_stop(val_loss):
            return model, train_losses, val_losses, bleu_scores

    return model, train_losses, val_losses, bleu_scores


def _train_one_epoch(
    model: TransformerNet,
    training_loader: DataLoader | list[tuple[torch.Tensor, list[str]]],
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer | NoamOptimizer,
    loss_fn: _LOSS_FN,
    epoch: int,
    device: torch.device,
    tb_writer: SummaryWriter | None,
    label_smoothing: float = 0.0,
) -> float:
    """Train one epoch.

    :param model: Model to train.
    :param training_loader: Training data loader.
    :param vocab: Vocabulary.
    :param optimizer: Optimizer.
    :param loss_fn: Loss function.
    :param epoch: Epoch number.
    :param device: Device to use.
    :param tb_writer: Tensorboard writer. Defaults to ``None``.
    :return: Loss.
    """
    model.train()
    running_loss = 0.0

    for data in tqdm.tqdm(training_loader, f"Train epoch {epoch + 1}"):
        inputs, captions = data
        captions_str, captions_end = _convert_captions_to_tensor(list(captions), vocab)

        inputs = inputs.to(device)
        captions_str = captions_str.to(device)
        captions_end = captions_end.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, captions_str)

        flatten_outputs = outputs.view(-1, len(vocab))

        flatten_captions_end = captions_end.flatten()
        one_hot = F.one_hot(flatten_captions_end, num_classes=len(vocab)).float()
        if label_smoothing > 0.0:  # ruff: noqa: PLR2004
            one_hot = _smooth_labels(one_hot, label_smoothing)

        loss = loss_fn(flatten_outputs, one_hot)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    loss = running_loss / len(training_loader)

    decoded_targets = [_convert_tensor_to_caption(caption, vocab) for caption in captions_end]

    outputs_normalized = torch.argmax(outputs, dim=2)

    decoded_predictions = [
        _convert_tensor_to_caption(output, vocab) for output in outputs_normalized
    ]

    example_idx = random.randint(0, len(decoded_predictions) - 1)

    logger.info("Trgt: {}", decoded_targets[example_idx])
    logger.info("Pred: {}", decoded_predictions[example_idx])

    if tb_writer is not None:
        tb_writer.add_scalar("Loss/train", loss, epoch)

    return loss


def _validate_one_epoch(
    model: TransformerNet,
    val_loader: DataLoader,
    vocab: dict[str, int],
    loss_fn: _LOSS_FN,
    epoch: int,
    device: torch.device,
    tb_writer: SummaryWriter | None,
    label_smoothing: float = 0.0,
) -> tuple[float, float]:
    """Validate one epoch.

    :param model: Model to validate.
    :param val_loader: Validation data loader.
    :param vocab: Vocabulary.
    :param loss_fn: Loss function.
    :param epoch: Epoch number.
    :param device: Device to use.
    :param tb_writer: Tensorboard writer. Defaults to ``None``.
    :param label_smoothing: Label smoothing. Defaults to ``0.0``.
    :return: Validation loss and accuracy.
    """
    model.eval()

    running_loss = 0.0
    decoded_predictions = []
    decoded_targets = []
    bleu_metric = BLEUScore(n_gram=4)
    bleu_scores = []

    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, f"Validation epoch {epoch + 1}"):
            inputs, captions = data

            captions_str, captions_end = _convert_captions_to_tensor(list(captions), vocab)

            inputs = inputs.to(device)
            captions_str = captions_str.to(device)
            captions_end = captions_end.to(device)

            outputs = model(inputs, captions_str)

            # compute loss
            flatten_outputs = outputs.view(-1, len(vocab))
            flatten_captions_end = captions_end.flatten()

            one_hot = F.one_hot(flatten_captions_end, num_classes=len(vocab)).float()
            if label_smoothing > 0.0:  # ruff: noqa: PLR2004
                one_hot = _smooth_labels(one_hot, label_smoothing)

            loss = loss_fn(flatten_outputs, one_hot)
            running_loss += loss.item()

            # Compute BLEU score
            [
                decoded_targets.append(_convert_tensor_to_caption(caption, vocab))
                for caption in captions_end
            ]
            outputs_normalized = torch.argmax(outputs, dim=2)

            [
                decoded_predictions.append(_convert_tensor_to_caption(output, vocab))
                for output in outputs_normalized
            ]

            bleu_metric.update(decoded_targets, decoded_predictions)
            bleu_scores.append(bleu_metric.compute())

    avg_loss = running_loss / len(val_loader)
    avg_bleu_metric = torch.mean(torch.stack(bleu_scores))
    logger.info("Validation loss: {}", avg_loss)
    logger.info("Validation BLEU score: {}", avg_bleu_metric)

    if tb_writer is not None:
        tb_writer.add_scalar("Loss/validation", avg_loss, epoch)
        tb_writer.add_scalar("BLEU/validation", avg_bleu_metric, epoch)

    example_idx = random.randint(0, len(decoded_predictions) - 1)

    logger.info("Trgt: {}", decoded_targets[example_idx])
    logger.info("Pred: {}", decoded_predictions[example_idx])

    return avg_loss, avg_bleu_metric


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


def _smooth_labels(y: torch.Tensor, smooth_factor: float) -> torch.Tensor:
    """Convert a matrix of one-hot row-vector labels into smoothed versions.

    :param y: matrix of one-hot row-vector labels.
    :param smooth_factor: label smoothing factor (between 0 and 1).
    :return: A new matrix of smoothed labels.
    """
    return y * (1 - smooth_factor) + smooth_factor / y.size(1)
