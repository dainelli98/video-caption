# -*- coding: utf-8 -*-
"""Script to test decoder."""
import platform
from pathlib import Path

import click
import joblib
import numpy as np
import torch
from loguru import logger
from torch.utils.data import DataLoader

from vid_cap import DATA_DIR
from vid_cap.dataset import VideoEvalDataset
from vid_cap.modelling import test
from vid_cap.modelling.model import TransformerNet
from vid_cap.modelling import preprocessing as pre
from vid_cap.modelling import _utils as utils

_MAX_TGT_LEN = 100
_SAMPLE_PERIOD = 16

@click.command("inference")
@click.option("--n-heads", default=4, type=click.IntRange(1, 128), help="Number of heads.")
@click.option(
    "--n-layers", default=2, type=click.IntRange(1, 128), help="Number of decoder layers."
)
@click.option("--use-gpu", is_flag=True, type=bool, help="Try to test with GPU")
@click.option("--video-path", type=click.Path(exists=True), help="Video Input")
@click.option("--inference-model", type=click.Path(exists=True), help="Model to be used on inference")
@click.option("--inference-model-vocab", type=click.Path(exists=True),help="Vocab to be used on inference")
def main(
    n_heads: int,
    n_layers: int,
    use_gpu: bool,
    video_path:str,
    inference_model:str,
    inference_model_vocab:str
) -> list[str]:

    
    model_path: Path = inference_model
    gpu_model = "cpu"

    if use_gpu:
        if platform.processor() == "arm":
            gpu_model = "mps"
        elif torch.cuda.is_available():
            gpu_model = "cuda"

    device = torch.device(gpu_model)

    logger.info(f"Inference with device : {device}")

    vocab = joblib.load(inference_model_vocab)
    id2word = {id_: word for word, id_ in vocab.items()}

    input = torch.tensor(load_video(video_path=video_path),device=device)
    input = input.unsqueeze(0)


    ##TODO
    model = TransformerNet(len(vocab), input.shape[2], n_heads, n_layers, _MAX_TGT_LEN).to(
        device
    )
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.eval()

    sos_id = vocab["<sos>"]
    eos_id = vocab["<eos>"]

    with torch.no_grad():
        captions = torch.full(
            (1,_MAX_TGT_LEN), fill_value=eos_id, dtype=torch.long, device=device
        )
        captions[:, 0] = sos_id

        for t in range(1, _MAX_TGT_LEN):
            outputs = model(input, captions[:, :t])
            next_word_logits = outputs[:, t - 1, :]
            captions[:, t] = next_word_logits.argmax(-1)

        test = [utils.convert_tensor_to_caption(output, id2word, use_bpe=False)           
            for output in captions
        ]
        print(test)
    
    return test

def load_video(video_path: Path) -> np.ndarray:
    
    feat_vec = pre.gen_feat_vecs(video_path, 16, True)[0, :, :]
    feat_vec = feat_vec[::_SAMPLE_PERIOD, :]
    feat_vec = feat_vec.astype(np.float16)
    return feat_vec

if __name__ == "__main__":
    main.main(standalone_mode=False)
