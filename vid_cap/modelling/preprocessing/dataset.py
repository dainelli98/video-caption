# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import random
from pathlib import Path
import collections
import joblib
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Self


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions."""

    _data: list[tuple[np.ndarray, str]]
    _vocab: dict[str, int]

    def __init__(
        self, data_dict: dict[str, dict[str, int | np.ndarray | list[str]]], captions_amount_per_video: int = 1, vocab_len: int = 1000
    ) -> None:
        """Initialize dataset with data dictionary.

        :param data_dict: Dictionary with data.
        :param captions_amount_per_video: Amount of captions per video, min=1, max=20. Defaults to 1.
        :param vocab_len: Amount of words for vocabulary. Defaults to 1000.
        """
        captions_amount_per_video = min(max(captions_amount_per_video, 1), 20)

        self._data = self._sort_data_by_caption_lenght(data_dict, captions_amount_per_video)
        self._vocab = self._build_vocab(vocab_len)

    def _sort_data_by_caption_lenght(self, data_dict: dict[str, dict[str, int | np.ndarray | list[str]]], captions_amount_per_video: int) -> list[tuple[np.ndarray, str]]:
        """Build vocabulary from the captions in the dataset.

        :param data_dict: Dictionary with data.
        :param captions_amount_per_video: Amount of captions per video.
        :return: Truncated sorted by frecuency vocab.
        """
        data = [
            (data["features"], caption)
            for data in data_dict.values()
            for caption in data["captions"][:captions_amount_per_video]
        ]

        sorted_data = sorted(data, key=lambda x: len(x[1]))
        return sorted_data
    
    def _build_vocab(self, vocab_len: int) -> dict[str, int]:
        """Build vocabulary from the captions in the dataset.

        :param vocab_len: Amount of words for vocabulary.
        :return: Truncated sorted by frecuency vocab.
        """
        captions = [caption for _, caption in self._data]
        words = [word for caption in captions for word in caption.split()]
        vocab = collections.Counter(words)
        sorted_vocab = sorted(vocab, key=lambda x: vocab[x], reverse=True)

        sorted_vocab.insert(0, "<eos>")
        sorted_vocab.insert(0, "<sos>")
        sorted_vocab.insert(0, "<unk>")
        sorted_vocab.insert(0, "<pad>")

        truncated_sorted_vocab = sorted_vocab[:vocab_len]
        truncated_sorted_vocab = {token: idx for idx, token in enumerate(truncated_sorted_vocab)}

        return truncated_sorted_vocab

    @classmethod
    def load(cls, file_path: Path | str, captions_amount_per_video: int = 1, vocab_len: int = 1000) -> Self:
        """Load dataset from file.

        :param file_path: Path to file with data.
        :param captions_amount_per_video: Amount of captions per video, min=1, max=20. Defaults to 1.
        :param vocab_len: Amount of words for vocabulary. Defaults to 1000.
        :return: Dataset with data from file.
        """
        captions_amount_per_video = min(max(captions_amount_per_video, 1), 20)

        data_dict = joblib.load(file_path)
        
        return cls(data_dict, captions_amount_per_video, vocab_len)

    def __getitem__(self, index: int) -> tuple[np.ndarray, str]:
        """Get item from dataset.

        :param index: Index of item.
        :return: Feature vector and caption.
        """
        return self._data[index]

    def __len__(self) -> int:
        """Get length of dataset.

        :return: Length of dataset.
        """
        return len(self._data)