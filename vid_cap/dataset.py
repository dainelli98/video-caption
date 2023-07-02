# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import collections
from pathlib import Path

import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions.

    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
    :param caps_per_vid: Amount of captions per video, min=1, max=20.
        Defaults to 1.
    :param vocab_len: Amount of words for vocabulary. If ``None`` no vocab is built.
    """

    _captions: pd.DataFrame
    _video_dir: Path
    _vocab: dict[str, int] | None

    def __init__(
        self,
        video_dir: Path | str,
        caps_file: Path | str,
        caps_per_vid: int = 1,
        vocab_len: int | None = None,
    ) -> None:
        caps_per_vid = min(max(caps_per_vid, 1), 20)

        self._captions = pd.read_parquet(caps_file)[lambda x: x["n_cap"] <= caps_per_vid]
        self._video_dir = Path(video_dir)

        order = self._captions["caption"].str.len().sort_values().index
        self._captions = self._captions.reindex(order).reset_index(drop=True)

        if vocab_len is not None:
            self._build_vocab(vocab_len)

    @property
    def vocab(self) -> dict[str, int]:
        """Get vocabulary.

        :return: Vocabulary.
        """
        if self._vocab is None:
            raise AttributeError("Vocabulary not built.")

        return self._vocab

    def get_vocab_len(self) -> int:
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

        truncated_sorted_vocab = sorted_vocab[:vocab_len]
        self._vocab = {token: idx for idx, token in enumerate(truncated_sorted_vocab)}

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
