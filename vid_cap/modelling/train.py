# -*- coding: utf-8 -*-
"""Train - decoder."""
import torch
import tqdm
from loguru import logger
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from .model import TransformerNet

_LOSS_FN = torch.nn.modules.loss._Loss  # noqa: SLF001


def train(
    model: TransformerNet,
    train_loader: DataLoader,
    valid_loader: DataLoader,
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer,
    loss_fn: _LOSS_FN,
    num_epochs: int,
    device: torch.device,
    tb_writer: SummaryWriter | None = None,
) -> TransformerNet:
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
    :return: Trained model.
    """
    model.train()

    for epoch in range(num_epochs):
        logger.info("Starting epoch {}/{}", epoch + 1, num_epochs)
        train_loss = _train_one_epoch(
            model, train_loader, vocab, optimizer, loss_fn, epoch, device, tb_writer
        )
        logger.info("End of epoch {}/{}. Train loss: {}", epoch + 1, num_epochs, train_loss)
        val_loss, _ = _validate_one_epoch(
            model, valid_loader, vocab, loss_fn, epoch, device, tb_writer
        )
        logger.info("End of epoch {}/{}. Validation loss: {}", epoch + 1, num_epochs, val_loss)

    return model


def _train_one_epoch(
    model: TransformerNet,
    training_loader: DataLoader,
    vocab: dict[str, int],
    optimizer: torch.optim.Optimizer,
    loss_fn: _LOSS_FN,
    epoch: int,
    device: torch.device,
    tb_writer: SummaryWriter | None,
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
    running_loss = 0.0

    for data in tqdm.tqdm(training_loader, f"Train epoch {epoch + 1}"):
        inputs, captions = data
        captions_str, captions_end = _convert_captions_to_tensor(list(captions), vocab)
        inputs = inputs.to(device)
        captions_str = captions_str.to(device)
        captions_end = captions_end.to(device)

        optimizer.zero_grad()

        outputs = model(inputs, captions_str)

        captions_end = (
            torch.nn.functional.one_hot(captions_end, num_classes=1000).float().to(device)
        )

        loss = loss_fn(outputs, captions_end)
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
    correct_predictions = 0
    total_predictions = 0

    with torch.no_grad():
        for data in tqdm.tqdm(val_loader, f"Validation epoch {epoch + 1}"):
            inputs, captions = data
            captions_str, captions_end = _convert_captions_to_tensor(list(captions), vocab)

            inputs = inputs.to(device)
            captions_str = captions_str.to(device)
            captions_end = captions_end.to(device)

            outputs = model(inputs, captions_str)

            captions_end = (
                torch.nn.functional.one_hot(captions_end, num_classes=1000).float().to(device)
            )

            loss = loss_fn(outputs, captions_end)
            running_loss += loss.item()

            # Compute accuracy
            _, predicted = torch.max(outputs, 1)
            total_predictions += len(captions)
            correct_predictions += (predicted == captions).sum().item()

    model.train()

    avg_loss = running_loss / len(val_loader)
    accuracy = correct_predictions / total_predictions

    logger.info("Validation loss: {}", avg_loss)
    logger.info("Validation accuracy: {}", accuracy)

    if tb_writer is not None:
        tb_writer.add_scalar("Loss/validation", avg_loss, epoch)
        tb_writer.add_scalar("Accuracy/validation", accuracy, epoch)

    return avg_loss, accuracy


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
