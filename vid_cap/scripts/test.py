# -*- coding: utf-8 -*-
"""Script to test decoder."""
import os
import platform
from pathlib import Path

import click
import joblib
import torch
from loguru import logger
from torch.utils.data import DataLoader

from vid_cap import DATA_DIR
from vid_cap.dataset import VideoEvalDataset
from vid_cap.modelling import test
from vid_cap.modelling.model import TransformerNet

_MAX_TGT_LEN = 100


@click.command("test")
@click.option("--n-heads", default=8, type=click.IntRange(1, 128), help="Number of heads.")
@click.option("--data-dir", default=DATA_DIR, type=click.Path(exists=True), help="Data directory")
@click.option(
    "--n-layers", default=4, type=click.IntRange(1, 128), help="Number of decoder layers."
)
@click.option("--batch-size", default=64, type=click.IntRange(1, 512), help="Batch size.")
@click.option("--use-gpu", is_flag=True, type=bool, help="Try to test with GPU")
@click.option(
    "--experiment-number",
    required=True,
    type=str,
    help="Number timestamp name on experiment folder.",
)
def main(
    n_heads: int,
    data_dir: Path,
    n_layers: int,
    batch_size: int,
    use_gpu: bool,
    experiment_number: str,
) -> None:
    """Test decoder.

    \f

    :param n_heads: Number of heads.
    :param data_dir: Path to data directory.
    :param model_path: Path to trained model.
    :param vocab_path: Path to vocabulary file.
    :param n_layers: Number of decoder layers.
    :param batch_size: Batch size.
    :param use_gpu: Whether to try to use GPU.
    :param bpe_codes_path: Path to bpe codes file. If not provided, BPE will not be used.
    """
    experiment_folder_path: Path = data_dir / "output" / experiment_number
    model_path: Path = experiment_folder_path / "model"
    use_bpe = os.path.isfile(experiment_folder_path / "bpe.codes")

    gpu_model = "cpu"

    if use_gpu:
        if platform.processor() == "arm":
            gpu_model = "mps"
        elif torch.cuda.is_available():
            gpu_model = "cuda"

    device = torch.device(gpu_model)

    logger.info(f"Testing with device : {device}")

    vocab = joblib.load(experiment_folder_path / "vocab.pkl")

    test_dataset = VideoEvalDataset(
        data_dir / "test" / "videos",
        data_dir / "test" / "captions.parquet",
    )

    test_loader = DataLoader(test_dataset, batch_size, shuffle=False)

    model = TransformerNet(len(vocab), test_dataset.shape[1], n_heads, n_layers, _MAX_TGT_LEN).to(
        device
    )

    model.load_state_dict(torch.load(model_path, map_location=device))

    bleu_score = test.test_model(model, test_loader, test_dataset.captions, vocab, device, use_bpe)

    logger.info(f"Test BLEU score : {bleu_score}")


if __name__ == "__main__":
    main.main(standalone_mode=False)
