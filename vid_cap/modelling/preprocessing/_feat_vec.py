# -*- coding: utf-8 -*-
"""Functions used to generate feature vectors from videos."""
from collections.abc import Callable, Iterable
from pathlib import Path

import av
import numpy as np
import torch
from av.container.input import InputContainer
from transformers import AutoImageProcessor, VideoMAEForVideoClassification, VideoMAEModel

_IMG_PROCESSOR_NAME: str = "MCG-NJU/videomae-base-finetuned-kinetics"
_ENC_MODEL_NAME: str = "MCG-NJU/videomae-base-finetuned-kinetics"
_ENC_MODEL: VideoMAEForVideoClassification = VideoMAEForVideoClassification.from_pretrained(
    _ENC_MODEL_NAME
)

IMG_PROCESSOR: Callable = AutoImageProcessor.from_pretrained(_IMG_PROCESSOR_NAME)
ENCODER: VideoMAEModel = _ENC_MODEL.videomae


def gen_feat_vecs(
    filepaths: Path | str | Iterable[Path | str],
    n_frames: int,
    img_processor: Callable = IMG_PROCESSOR,
    encoder: VideoMAEModel = ENCODER,
) -> np.ndarray:
    """Generate feature vectors from a video files.

    :param filepaths: Paths to the video files.
    :param n_frames: Number of frames to sample from the videos.
    :param img_processor: Image processor used to prepare de videos. Defaults to ``IMG_PROCESSOR``.
    :param encoder: Encoder used to generate feature vectors. Defaults to ``ENCODER``.
    :return: Feature vector for videos.
    """
    if not isinstance(filepaths, list | tuple | set | np.ndarray):
        filepaths = [filepaths]

    videos = [
        img_processor(list(get_video_frames(filepath, n_frames)), return_tensors="pt")
        for filepath in filepaths
    ]

    with torch.no_grad():
        return np.array([encoder(**video)[0][0].numpy() for video in videos])


def get_video_frames(filepath: Path | str, n_frames: int) -> np.ndarray:
    container = av.open(str(filepath.absolute()))
    indices = _sample_frame_indices(n_frames, seg_len=container.streams.video[0].frames)
    return _read_video_pyav(container, indices)


def _read_video_pyav(container: InputContainer, indices: np.ndarray | list[int]) -> np.ndarray:
    """Decode the video with PyAV decoder.

    :param container: PyAV container.
    :param indices: List of frame indices to decode.
    :return: Decoded frames of shape (num_frames, height, width, 3).
    """
    frames = []
    container.seek(0)
    start_index = indices[0]
    end_index = indices[-1]
    for i, frame in enumerate(container.decode(video=0)):
        if i > end_index:
            break
        if i >= start_index and i in indices:
            frames.append(frame)
    return np.stack([x.to_ndarray(format="rgb24") for x in frames])


def _sample_frame_indices(n_frames: int, seg_len: int) -> np.ndarray:
    """Sample evenly distributed frame indices from a video.

    :param n_frames: Number of frames to sample.
    :param seg_len: Length of the video in frames.
    :return: Array of evenly distributed frame indices.
    """
    end_idx = seg_len - 1
    start_idx = 0
    indices = np.linspace(start_idx, end_idx, num=n_frames)
    return np.clip(indices, start_idx, end_idx).astype(np.int64)
