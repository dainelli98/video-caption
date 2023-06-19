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
        self, data_dict: dict[str, dict[str, int | np.ndarray | list[str]]], captions_amount_per_video: int = 1
    ) -> None:
        """Initialize dataset with data dictionary.

        :param data_dict: Dictionary with data.
        :param captions_amount_per_video: Amount of captions per video, min=1, max=20. Defaults to 1.
        """
        captions_amount_per_video = min(max(captions_amount_per_video, 1), 20)

        data = [
            (data["features"], caption)
            for data in data_dict.values()
            for caption in data["captions"][:captions_amount_per_video]
        ]

        # Sort the data by caption length
        sorted_data = sorted(data, key=lambda x: len(x[1]))
    
        self._data = sorted_data

    @classmethod
    def load(cls, file_path: Path | str, captions_amount_per_video: int = 1) -> Self:
        """Load dataset from file.

        :param file_path: Path to file with data.
        :param captions_amount_per_video: Amount of captions per video, min=1, max=20. Defaults to 1.
        :return: Dataset with data from file.
        """
        captions_amount_per_video = min(max(captions_amount_per_video, 1), 20)

        data_dict = joblib.load(file_path)
        
        return cls(data_dict, captions_amount_per_video)

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
