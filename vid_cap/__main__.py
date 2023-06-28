# -*- coding: utf-8 -*-
# ruff: noqa: D401
"""Entry point."""
import torch
from torch import nn, optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from vid_cap import DATA_DIR
from vid_cap.model.__model_transformer_decoder import TransformerNet
from vid_cap.model.train import DecoderTrainer
from vid_cap.modelling.preprocessing import VideoFeatDataset

device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")


def _main() -> None:
    """Run main function for entrypoint."""
    # @click.group(chain=True)
    # @click.version_option(__version__)
    # def entry_point() -> None:
    """Package entry point."""

    train_dataset = VideoFeatDataset(
        DATA_DIR / "train" / "videos", DATA_DIR / "train" / "captions.parquet"
    )
    valid_dataset = VideoFeatDataset(
        DATA_DIR / "val" / "videos", DATA_DIR / "train" / "captions.parquet"
    )
    test_dataset = VideoFeatDataset(
        DATA_DIR / "test" / "videos", DATA_DIR / "train" / "captions.parquet"
    )

    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"], shuffle=True)
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"], shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"], shuffle=True)

    my_model = TransformerNet(
        config["num_tgt_vocab"],
        config["embedding_dim"],
        config["vocab_size"],
        config["nheads"],
        config["n_layers"],
        config["max_tgt_len"],
    ).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    criterion = nn.CrossEntropyLoss()
    writer = SummaryWriter()

    dec_trainer = DecoderTrainer(
        optimizer, criterion, train_loader, valid_loader, my_model, train_dataset._vocab
    )
    dec_trainer.train(config["epochs"], writer)


if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "num_tgt_vocab": 1000,
        "embedding_dim": 768,
        "vocab_size": 1000,
        "nheads": 8,
        "n_layers": 4,
        "max_tgt_len": 100,
    }
    _main()
