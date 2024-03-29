# -*- coding: utf-8 -*-
# ruff: noqa: PLR0913
"""Script to train decoder."""
import platform
import random
import shutil
from pathlib import Path

import click
import joblib
import numpy as np
import pandas as pd
import torch
from loguru import logger
from torch import nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vid_cap import DATA_DIR
from vid_cap.dataset import DatasetType, VideoFeatDataset
from vid_cap.modelling import train
from vid_cap.modelling.model import TransformerNet
from vid_cap.modelling.scheduler import NoamOptimizer
from vid_cap.utils import loss_plot

_MAX_TGT_LEN = 100
_SEED = 1234


@click.command("train")
@click.option("--warmup-steps", default=4000, type=click.IntRange(1, 100000), help="Warmup steps.")
@click.option(
    "--label-smoothing", default=0.1, type=click.FloatRange(0, 1), help="Label smoothing."
)
@click.option("--data-dir", default=DATA_DIR, type=click.Path(exists=True), help="Data directory")
@click.option("--shuffle", is_flag=True, default=False, type=bool, help="Shuffle datasets")
@click.option("--batch-size", default=64, type=click.IntRange(1, 512), help="Batch size.")
@click.option("--n-heads", default=8, type=click.IntRange(1, 128), help="Number of heads.")
@click.option(
    "--n-layers", default=4, type=click.IntRange(1, 128), help="Number of decoder layers."
)
@click.option("--use-gpu", is_flag=True, type=bool, help="Try to train with GPU")
@click.option("--epochs", default=50, type=click.IntRange(1, 10000), help="Number of epochs.")
@click.option(
    "--vocab-len",
    default=8000,
    type=click.IntRange(1, 100000),
    help="Vocab length. If BPE is used, this is ignored.",
)
@click.option(
    "--caps-per-vid",
    default=1,
    type=click.IntRange(1, 20),
    help="Captions per video used in the dataset.",
)
@click.option("--dropout", default=0.1, type=click.FloatRange(0, 1), help="Dropout rate.")
@click.option(
    "--bpe-num-operations",
    type=click.IntRange(1, 100000),
    help="Number of BPE operations. If not provided, BPE will not be used.",
)
def main(
    warmup_steps: int,
    label_smoothing: float,
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
    bpe_num_operations: int | None,
) -> None:
    """Train decoder.

    \f

    :param warmup_steps: Warmup steps.
    :param label_smoothing: Label smoothing.
    :param data_dir: Path to data directory.
    :param shuffle: Whether to shuffle datasets.
    :param batch_size: Batch size.
    :param n_heads: Number of heads.
    :param n_layers: Number of decoder layers.
    :param use_gpu: Whether to try to use GPU.
    :param epochs: Number of epochs.
    :param vocab_len: Vocab length. If BPE is used, this is ignored.
    :param caps_per_vid: Number of captions per video.
    :param dropout: Dropout rate.
    :param bpe_num_operations: Nuber of BPE operations. If not provided, BPE will not be
        used.
    """
    use_bpe = bpe_num_operations is not None

    if use_bpe:
        bpe_dir = data_dir / "bpe"
        shutil.rmtree(bpe_dir, ignore_errors=True)
        bpe_dir.mkdir(exist_ok=True, parents=True)

    else:
        bpe_dir = None

    torch.manual_seed(_SEED)
    np.random.seed(_SEED)
    random.seed(_SEED)

    gpu_model = "cpu"

    if use_gpu:
        if platform.processor() == "arm":
            gpu_model = "mps"
        elif torch.cuda.is_available():
            gpu_model = "cuda"

    device = torch.device(gpu_model)

    logger.info(f"Training with device : {device}")

    train_dataset = VideoFeatDataset(
        DatasetType.TRAIN,
        data_dir / "train" / "videos",
        data_dir / "train" / "captions.parquet",
        caps_per_vid,
        vocab_len,
        bpe_dir,
        bpe_num_operations,
    )
    valid_dataset = VideoFeatDataset(
        DatasetType.VALID,
        data_dir / "val" / "videos",
        data_dir / "val" / "captions.parquet",
        caps_per_vid,
        bpe_dir=bpe_dir,
        bpe_num_operations=bpe_num_operations,
    )

    hparams = {
        "batch_size": batch_size,
        "shuffle": shuffle,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "vocab_len": train_dataset.vocab_len,
        "caps_per_vid": caps_per_vid,
        "label_smoothing": label_smoothing,
        "dropout": dropout,
        "embeddings": train_dataset.shape[0],
        "warmup_steps": warmup_steps,
        "bpe_num_operations": bpe_num_operations,
        "use_bpe": use_bpe,
    }

    [logger.debug(f"hparams::{k} : {v}") for k, v in hparams.items()]

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
    optimizer = NoamOptimizer(model.embedding_dim, 2, warmup_steps, optimizer, writer)

    criterion = nn.CrossEntropyLoss()

    out_dir = data_dir / "output" / f"{pd.Timestamp.now().strftime('%Y%m%d%H%M%S')}"

    model, train_loss, val_loss, bleu_scores, lrs = train.train(
        model,
        train_loader,
        valid_loader,
        train_dataset.vocab,
        optimizer,
        criterion,
        epochs,
        device,
        out_dir,
        writer,
        label_smoothing,
        use_bpe,
    )

    joblib.dump(train_dataset.vocab, out_dir / "vocab.pkl")
    if use_bpe:
        shutil.copyfile(train_dataset.bpe_codes_file, out_dir / "bpe_codes.codes")

    loss_plot.plot_and_store_graphs(train_loss, val_loss, bleu_scores, lrs, out_dir)


if __name__ == "__main__":
    main.main(standalone_mode=False)
