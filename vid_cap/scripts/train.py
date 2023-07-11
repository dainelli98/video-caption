# -*- coding: utf-8 -*-
# ruff: noqa: PLR0913
"""Script to train decoder."""
import platform
from pathlib import Path

import click
import joblib
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vid_cap import DATA_DIR
from vid_cap.dataset import VideoFeatDataset
from vid_cap.modelling import train
from vid_cap.modelling.model import TransformerNet
from vid_cap.modelling.scheduler import NoamOptimizer
from vid_cap.utils.loss_plot import plot_traing_losses, plot_bleu_scores

_MAX_TGT_LEN = 100


@click.command("train")
@click.option("--loss-smoothing", default=0.0, type=click.FloatRange(0, 1), help="Loss smoothing.")
@click.option("--data-dir", default=DATA_DIR, type=click.Path(exists=True), help="Data directory")
@click.option("--shuffle", is_flag=True, default=False, type=bool, help="Shuffle datasets")
@click.option("--batch-size", default=64, type=click.IntRange(1, 512), help="Batch size.")
@click.option("--n-heads", default=8, type=click.IntRange(1, 128), help="Number of heads.")
@click.option(
    "--n-layers", default=4, type=click.IntRange(1, 128), help="Number of decoder layers."
)
@click.option("--use-gpu", is_flag=True, type=bool, help="Try to train with GPU")
@click.option("--epochs", default=50, type=click.IntRange(1, 10000), help="Number of epochs.")
@click.option("--vocab-len", default=8000, type=click.IntRange(1, 100000), help="Vocab length.")
@click.option(
    "--caps-per-vid",
    default=1,
    type=click.IntRange(1, 20),
    help="Captions per video used in the dataset.",
)
@click.option("--dropout", default=0.1, type=click.FloatRange(0, 1), help="Dropout rate.")
def main(
    loss_smoothing: float,
    data_dir: Path,
    shuffle: bool,
    batch_size: int,
    n_heads: int,
    n_layers: int,
    use_gpu: bool,
    epochs: int,
    vocab_len: int,
    caps_per_vid: int,
    dropout: float,
) -> None:
    """Train decoder.

    \f

    :param loss_smoothing: Loss smoothing.
    :param data_dir: Path to data directory.
    :param shuffle: Whether to shuffle datasets.
    :param batch_size: Batch size.
    :param n_heads: Number of heads.
    :param n_layers: Number of decoder layers.
    :param use_gpu: Whether to try to use GPU.
    :param epochs: Number of epochs.
    :param vocab_len: Vocab length.
    :param caps_per_vid: Number of captions per video.
    :param dropout: Dropout rate.
    """
    gpu_model = "cpu"

    if use_gpu:
        if platform.processor() == "arm":
            gpu_model = "mps"
        elif torch.cuda.is_available():
            gpu_model = "cuda"

    device = torch.device(gpu_model)

    logger.info(f"Training with device : {device}")
    hparams = {
        "data_dir": data_dir,
        "batch_size": batch_size,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "use_gpu": use_gpu,
        "epochs": epochs,
        "vocab_len": vocab_len,
        "caps_per_vid": caps_per_vid,
    }

    [logger.debug(f"hparams::{k} : {v}") for k, v in hparams.items()]

    train_dataset = VideoFeatDataset(
        data_dir / "train" / "videos",
        data_dir / "train" / "captions.parquet",
        caps_per_vid,
        vocab_len,
    )
    valid_dataset = VideoFeatDataset(
        data_dir / "val" / "videos", data_dir / "val" / "captions.parquet", caps_per_vid
    )

    train_loader = DataLoader(
        train_dataset, batch_size, shuffle, pin_memory=True, prefetch_factor=True, num_workers=4
    )

    valid_loader = DataLoader(valid_dataset, batch_size)

    embed_dim = train_dataset.shape[1]

    model = TransformerNet(
        train_dataset.vocab_len, embed_dim, n_heads, n_layers, _MAX_TGT_LEN, dropout
    ).to(device)

    writer = SummaryWriter()

    optimizer = torch.optim.Adam(model.parameters(), lr=0, betas=(0.9, 0.98), eps=1e-9)
    optimizer = NoamOptimizer(model.embedding_dim, 2, 4000, optimizer, writer)

    criterion = nn.CrossEntropyLoss()

    model_name = (
        f"MODEL-batch_size_{batch_size}-n_heads_{n_heads}-n_layers_"
        f"{n_layers}-vocab_len_{vocab_len}-caps_per_vid_{caps_per_vid}"
    )

    model, train_loss, val_loss, bleu_scores = train.train(
        model,
        train_loader,
        valid_loader,
        train_dataset.vocab,
        optimizer,
        criterion,
        epochs,
        device,
        data_dir,
        model_name,
        writer,
        loss_smoothing,
    )

    joblib.dump(train_dataset.vocab, data_dir / "output" / f"{model_name}_vocab.pkl")
    plot_traing_losses(train_loss=train_loss, val_loss=val_loss)
    plot_bleu_scores(bleu_scores)
    
if __name__ == "__main__":
    main.main(standalone_mode=False)
