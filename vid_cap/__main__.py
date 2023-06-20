# -*- coding: utf-8 -*-
# ruff: noqa: D401
"""Entry point."""
# from . import __version__
from vid_cap.model.train import DecoderTrainer
from vid_cap.model.__model_transformer_decoder import TransformerNet
from vid_cap.modelling.preprocessing.dataset import VideoFeatDataset
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader

import torch.optim as optim
import torch
import torch.nn as nn


device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def _main() -> None:
    """Run main function for entrypoint."""

    # @click.group(chain=True)
    # @click.version_option(__version__)
    # def entry_point() -> None:
    """Package entry point."""

    # entry_point.add_command(dummy.main)

    # entry_point()

    train_dataset = VideoFeatDataset.load("../data/Dataset/Train/train.pickle")
    valid_dataset = VideoFeatDataset.load("../data/Dataset/Valid/val.pickle")# Change path
    test_dataset = VideoFeatDataset.load("../data/Dataset/Train/train.pickle") # Change path
    
    train_loader = DataLoader(train_dataset, batch_size=config["batch_size"])
    valid_loader = DataLoader(valid_dataset, batch_size=config["batch_size"])
    test_loader = DataLoader(test_dataset, batch_size=config["batch_size"])

    my_model = TransformerNet(
        config["num_tgt_vocab"], 
        config["embedding_dim"],
        config["vocab_size"], 
        config["nheads"], 
        config["n_layers"],
        config["max_tgt_len"]
    ).to(device)

    optimizer = optim.Adam(my_model.parameters(), config["lr"])
    criterion = nn.NLLLoss()
    writer = SummaryWriter()

    dec_trainer = DecoderTrainer(optimizer, criterion, train_loader, valid_loader, my_model)
    dec_trainer.train(writer, config["epochs"])

if __name__ == "__main__":
    config = {
        "lr": 1e-3,
        "batch_size": 64,
        "epochs": 5,
        "num_tgt_vocab": 100,
        "embedding_dim": 1568,
        "vocab_size": 1000,
        "nheads": 8,
        "n_layers": 4,
        "max_tgt_len": 100,
    }
    _main()
