# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import random
from pathlib import Path

import joblib
import numpy as np
from torch.utils.data import Dataset
from typing_extensions import Self


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions."""

    _data: list[tuple[np.ndarray, str]]

    def __init__(
        self, data_dict: dict[str, dict[str, int | np.ndarray | list[str]]], only_one: bool = False
    ) -> None:
        """Initialize dataset with data dictionary.

        :param data_dict: Dictionary with data.
        :param only_one: Whether to only use one caption per video. Defaults to ``False``.
        """
        if only_one:
            self_data = [(data["features"], data["captions"][0]) for data in data_dict.values()]

        else:
            self_data = [
                (data["features"], caption)
                for data in data_dict.values()
                for caption in data["captions"]
            ]
            random.shuffle(self_data)

    @classmethod
    def load(cls, file_path: Path | str, only_one: bool = False) -> Self:
        """Load dataset from file.

        :param file_path: Path to file with data.
        :param only_one: Whether to only use one caption per video. Defaults to ``False``.
        :return: Dataset with data from file.
        """
        data_dict = joblib.load(file_path)
        return cls(data_dict, only_one)

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
