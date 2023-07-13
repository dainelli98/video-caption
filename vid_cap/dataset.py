# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import collections
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions.

    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
    :param caps_per_vid: Amount of captions per video, min=1, max=20.
        Defaults to ``None``.
    :param vocab_len: Amount of words for vocabulary. If ``None`` no vocab is built.
    """

    _captions: pd.DataFrame
    _videos: dict[int, torch.Tensor]
    _vocab: dict[str, int] | None

    def __init__(
        self,
        video_dir: Path | str,
        caps_file: Path | str,
        caps_per_vid: int | None = None,
        vocab_len: int | None = None,
    ) -> None:
        if not caps_per_vid:
            caps_per_vid = 20

        caps_per_vid = min(max(caps_per_vid, 1), 20)

        self._captions = pd.read_parquet(caps_file, dtype_backend="pyarrow")[
            lambda x: x["n_cap"] <= caps_per_vid
        ]

        self._captions = self._captions.assign(
            length=lambda x: x["caption"].apply(lambda x: len(x.split()))
        )

        self._captions = self._captions.sort_values("length")

        if vocab_len is not None:
            self.build_vocab(vocab_len)

        else:
            self._vocab = None

        if not isinstance(video_dir, Path):
            video_dir = Path(video_dir)

        self._videos = {
            video: torch.tensor(np.load(video_dir / f"{video}.npy"), dtype=torch.float16)
            for video in self._captions["video"].unique()
        }

    @property
    def vocab(self) -> dict[str, int]:
        """Get vocabulary.

        :return: Vocabulary.
        """
        if self._vocab is None:
            raise AttributeError("Vocabulary not built.")

        return self._vocab

    @property
    def vocab_len(self) -> int:
        """Get vocabulary length.

        :return: Vocabulary length.
        """
        if self._vocab is None:
            raise AttributeError("Vocabulary not built.")

        return len(self._vocab)

    def build_vocab(self, vocab_len: int) -> None:
        """Build vocabulary from the captions in the dataset.

        :param vocab_len: Amount of words for vocabulary.
        """
        captions = self._captions["caption"].to_list()
        words = [word for caption in captions for word in caption.split()]
        vocab = collections.Counter(words)
        sorted_vocab = sorted(vocab, key=lambda x: vocab[x], reverse=True)

        sorted_vocab.insert(0, "<eos>")
        sorted_vocab.insert(0, "<sos>")
        sorted_vocab.insert(0, "<unk>")
        sorted_vocab.insert(0, "<pad>")

        if vocab_len < len(sorted_vocab):
            truncated_sorted_vocab = sorted_vocab[:vocab_len]

        else:
            truncated_sorted_vocab = sorted_vocab

        self._vocab = {token: idx for idx, token in enumerate(truncated_sorted_vocab)}

    def __getitem__(self, index: int) -> tuple[torch.Tensor, str]:
        """Get item from dataset.

        :param index: Index of item.
        :return: Feature vector and caption.
        """
        caption_row = self._captions.iloc[index]
        return (
            self._videos[caption_row["video"]],
            caption_row["caption"],
        )

    def __len__(self) -> int:
        """Get length of dataset.

        :return: Length of dataset.
        """
        return self._captions.shape[0]

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of dataset.

        :return: Shape of dataset.
        """
        return self._videos[1].shape
