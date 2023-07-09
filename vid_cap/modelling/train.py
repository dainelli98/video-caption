# -*- coding: utf-8 -*-
"""Train - decoder."""

import random

import torch
import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torcheval.metrics import BLEUScore

from vid_cap.modelling.scheduler import NoamOptimizer

from .model import TransformerNet
import torch.nn.functional as F

_LOSS_FN = torch.nn.modules.loss._Loss  # noqa: SLF001


def train(
    model: TransformerNet,
    shuffle: bool,
    train_loader: DataLoader | list[tuple[torch.Tensor, list[str]]],
    valid_loader: DataLoader,
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer | NoamOptimizer,
    loss_fn: _LOSS_FN,
    num_epochs: int,
    device: torch.device,
    tb_writer: SummaryWriter | None = None,
) -> TransformerNet:
    """Train model.

    :param model: Model to train.
    :param shuffle: Whether to shuffle train dataset at each epoch.
    :param train_loader: Training data loader.
    :param valid_loader: Validation data loader.
    :param vocab: Vocabulary.
    :param optimizer: Optimizer.
    :param loss_fn: Loss function.
    :param num_epochs: Number of epochs.
    :param device: Device to use.
    :param tb_writer: Tensorboard writer.
    :return: Trained model.
    """
    train_losses = []
    val_losses=[]

    for epoch in range(num_epochs):
        logger.info("Starting epoch {}/{}", epoch + 1, num_epochs)
        train_loss = _train_one_epoch(
            model, train_loader, shuffle, vocab, optimizer, loss_fn, epoch, device, tb_writer
        )
        train_losses.append(train_loss)
        logger.info("End of epoch {}/{}. Train loss: {}", epoch + 1, num_epochs, train_loss)
        val_loss, bleu = _validate_one_epoch(
            model, valid_loader, vocab, loss_fn, epoch, device, tb_writer
        )
        val_losses.append(val_loss)
        logger.info("End of epoch {}/{}. Validation loss: {}", epoch + 1, num_epochs, val_loss)
        logger.info("End of epoch {}/{}. Validation BLEU: {}", epoch + 1, num_epochs, bleu)
        if isinstance(optimizer, NoamOptimizer):
            logger.info("Learning rate: {}", optimizer.lr)

    return model, train_losses, val_losses


def _train_one_epoch(
    model: TransformerNet,
    training_loader: DataLoader | list[tuple[torch.Tensor, list[str]]],
    shuffle: bool,
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer | NoamOptimizer,
    loss_fn: _LOSS_FN,
    epoch: int,
    device: torch.device,
    tb_writer: SummaryWriter | None,
) -> float:
    """Train one epoch.

    :param model: Model to train.
    :param training_loader: Training data loader.
    :param shuffle: Whether to shuffle train dataset batches at each epoch.
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
    if shuffle:
        logger.info("Shuffling training data")
        random.shuffle(training_loader)

    for data in tqdm.tqdm(training_loader, f"Train epoch {epoch + 1}"):
        inputs, captions = data
        captions_str, captions_end = _convert_captions_to_tensor(list(captions), vocab)

        inputs = inputs.to(device)
        captions_str = captions_str.to(device)
        captions_end = captions_end.to(device)

        optimizer.zero_grad()

        
        outputs = model(inputs, captions_str)
        captions_end = captions_end.view(-1)

        outputs = outputs.view(-1, len(vocab))

        x = captions_end.flatten()
        one_hot = F.one_hot(x, num_classes=len(vocab)).float()

        loss = loss_fn(outputs, one_hot)
        #loss = loss_fn(outputs, captions_end)
        loss.backward()

        optimizer.step()

        running_loss += loss.item()

    loss = running_loss / len(training_loader)

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
) -> tuple[float, float]:
    """Validate one epoch.

    :param model: Model to validate.
    :param val_loader: Validation data loader.
    :param vocab: Vocabulary.
    :param loss_fn: Loss function.
    :param epoch: Epoch number.
    :param device: Device to use.
    :param tb_writer: Tensorboard writer. Defaults to ``None``.
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
            flatten_outputs = outputs.view(-1, vocab.__len__())
            flatten_captions_end = captions_end.view(-1)

            x = flatten_captions_end.flatten()
            one_hot = F.one_hot(x, num_classes=len(vocab)).float()

            loss = loss_fn(flatten_outputs, one_hot)

            #loss = loss_fn(flatten_outputs, flatten_captions_end)
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

    words = [word for word in words if word not in ["<sos>", "<eos>"]]
    return " ".join(words)
