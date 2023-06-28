# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions."""

    _captions: pd.DataFrame
    _video_dir: Path
    _vocab: list[str]

    def __init__(
        self,
        video_dir: Path | str,
        caps_file: Path | str,
        captions_amount_per_video: int = 1,
        vocab_len: int = 10000,
    ) -> None:
        """Initialize dataset with data dictionary.

        :param data_dict: Dictionary with data.
        :param captions_amount_per_video: Amount of captions per video, min=1, max=20. Defaults to 1.
        :param vocab_len: Amount of words for vocabulary. Defaults to 10000.
        """
        captions_amount_per_video = min(max(captions_amount_per_video, 1), 20)

        self._captions = pd.read_parquet(caps_file)[
            lambda x: x["n_cap"] <= captions_amount_per_video
        ]
        self._video_dir = Path(video_dir)

        order = self._captions["caption"].str.len().sort_values().index
        self._captions = self._captions.reindex(order).reset_index(drop=True)

        self._vocab = self._build_vocab(vocab_len)

    def _build_vocab(self, vocab_len: int) -> list[str]:
        """Build vocabulary from the captions in the dataset.

        :param vocab_len: Amount of words for vocabulary.
        :return: Truncated sorted by frecuency vocab.
        """
        captions = self._captions["caption"].to_list()
        words = [word for caption in captions for word in caption.split()]
        vocab = collections.Counter(words)
        sorted_vocab = sorted(vocab, key=lambda x: vocab[x], reverse=True)

        sorted_vocab.insert(0, "<eos>")
        sorted_vocab.insert(0, "<sos>")
        sorted_vocab.insert(0, "<unk>")
        sorted_vocab.insert(0, "<pad>")

        truncated_sorted_vocab = sorted_vocab[:vocab_len]
        return truncated_sorted_vocab

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        """Get item from dataset.

        :param index: Index of item.
        :return: Feature vector and caption.
        """
        caption_row = self._captions.iloc[index]
        return np.load(self._video_dir / f"{caption_row['video']}.npy"), caption_row["caption"]

    def __len__(self) -> int:
        """Get length of dataset.

        :return: Length of dataset.
        """
        return self._captions.shape[0]
