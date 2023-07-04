# -*- coding: utf-8 -*-
"""Script to train decoder."""
from pathlib import Path

import click
from loguru import logger
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vid_cap import DATA_DIR
from vid_cap.dataset import VideoFeatDataset
from vid_cap.modelling import train
from vid_cap.modelling.model import TransformerNet

"""Identify if arch is x86_64 or ARM"""
import platform

_MAX_TGT_LEN = 100


@click.command("train")
@click.option("--data-dir", default=DATA_DIR, type=click.Path(exists=True), help="Data directory")
@click.option("--shuffle", is_flag=True, default=True, type=bool, help="Shuffle datasets")
@click.option("--batch_size", default=64, type=click.IntRange(1, 512), help="Batch size.")
@click.option("--n-heads", default=8, type=click.IntRange(1, 128), help="Number of heads.")
@click.option(
    "--n-layers", default=4, type=click.IntRange(1, 128), help="Number of decoder layers."
)
@click.option("--use-gpu", is_flag=True, type=bool, help="Try to train with GPU")
@click.option("--epochs", default=10, type=click.IntRange(1, 10000), help="Number of epochs.")
@click.option("--lr", default=1e-3, type=click.FloatRange(1e-6, 1e-1), help="Learning rate.")
@click.option("--vocab-len", default=1000, type=click.IntRange(1, 100000), help="Vocab length.")
def main(
    data_dir: Path,
    shuffle: bool,
    batch_size: int,
    n_heads: int,
    n_layers: int,
    use_gpu: bool,
    epochs: int,
    lr: float,
    vocab_len: int,
) -> None:
    """Train decoder.

    \f

    :param data_dir: Path to data directory.
    :param shuffle: Whether to shuffle datasets.
    :param batch_size: Batch size.
    :param n_heads: Number of heads.
    :param n_layers: Number of decoder layers.
    :param use_gpu: Whether to try to use GPU.
    :param epochs: Number of epochs.
    :param lr: Learning rate.
    :param vocab_len: Vocab length.
    """
    GPU_MODEL = torch.device('mps') if platform.processor()=='arm' else torch.device("cuda") if use_gpu and torch.cuda.is_available() else torch.device("cpu")
    device = GPU_MODEL

    logger.info(f"Training with device : {device}")
    hparams = { "data_dir": data_dir,
               "shuffle": shuffle,
               "batch_size": batch_size,
               "n_heads": n_heads,
               "n_layers": n_layers,
               "use_gpu": use_gpu,
               "epochs": epochs,
               "lr": lr,
               "vocab_len": vocab_len}
    [logger.debug(f'hparams::{k} : {v}') for k,v in hparams.items()]


    train_dataset = VideoFeatDataset(
        data_dir / "train" / "videos", data_dir / "train" / "captions.parquet", vocab_len=vocab_len
    )
    valid_dataset = VideoFeatDataset(
        data_dir / "val" / "videos", data_dir / "val" / "captions.parquet", vocab_len=vocab_len
    )

    train_loader = DataLoader(train_dataset, batch_size, shuffle, pin_memory=True, num_workers=3, prefetch_factor=True)
    valid_loader = DataLoader(valid_dataset, batch_size)

    embed_dim = train_dataset[0][0].shape[1]

    model = TransformerNet(train_dataset.vocab_len, embed_dim, n_heads, n_layers, _MAX_TGT_LEN).to(
        device
    )

    optimizer = optim.Adam(model.parameters(), lr)
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    model = train.train(
        model,
        train_loader,
        valid_loader,
        train_dataset.vocab,
        optimizer,
        criterion,
        epochs,
        device,
        writer,
    )

    torch.save(model.state_dict(), data_dir / "output" / "model.joblib")


if __name__ == "__main__":
    main.main(standalone_mode=False)
