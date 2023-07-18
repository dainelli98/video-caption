# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import collections
import enum
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import tqdm
from loguru import logger
from subword_nmt.apply_bpe import BPE
from subword_nmt.learn_bpe import learn_bpe
from torch.utils.data import Dataset


class DatasetType(enum.Enum):
    """Dataset type."""

    TRAIN = "train"
    VALID = "valid"
    TEST = "test"


class VideoEvalDataset(Dataset):
    """Dataset with video feature vectors and associated captions.

    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
    :param bpe_codes_file: Path to BPE codes file. If ``None`` no BPE is used.
    """

    _data: list[tuple[torch.Tensor, int]]
    _captions: pd.DataFrame

    def __init__(self, video_dir: Path | str, caps_file: Path | str) -> None:
        if not isinstance(video_dir, Path):
            video_dir = Path(video_dir)

        if not isinstance(caps_file, Path):
            caps_file = Path(caps_file)

        self._captions = pd.read_parquet(caps_file, dtype_backend="pyarrow")

        self._data = [
            (torch.tensor(np.load(video_dir / f"{video}.npy"), dtype=torch.float16), video)
            for video in self._captions["video"].unique()
        ]

    @property
    def captions(self) -> pd.DataFrame:
        """Get captions.

        :return: Captions.
        """
        return self._captions

    def __getitem__(self, index: int) -> tuple[torch.Tensor, int]:
        """Get item.

        :param index: Index.
        :return: Video feature vector and associated captions.
        """
        return self._data[index]

    def __len__(self) -> int:
        """Get length.

        :return: Length.
        """
        return len(self._data)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of dataset.

        :return: Shape of dataset.
        """
        return self._data[1][0].shape


class VideoFeatDataset(Dataset):
    """Dataset with video feature vectors and associated captions.

    :type: Dataset type.
    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
    :param caps_per_vid: Amount of captions per video, min=1, max=20.
        Defaults to ``None``.
    :param vocab_len: Amount of words for vocabulary. If ``None`` no word vocab is built.
        If bpe_num_operations is not ``None`` this is ignored.
    :param bpe_dir: Directory with BPE codes file. Is necessary if ``bpe_num_operations``
        is not ``None``.
    :param bpe_num_operations: Number of BPE merge operations. If ``None`` no BPE is used.
    """

    _type: DatasetType
    _captions: pd.DataFrame
    _videos: dict[int, torch.Tensor]
    _vocab: dict[str, int] | None
    _coverage: float
    _bpe_codes_file: Path | None

    def __init__(
        self,
        type_: DatasetType,
        video_dir: Path | str,
        caps_file: Path | str,
        caps_per_vid: int | None = None,
        vocab_len: int | None = None,
        bpe_dir: Path | None = None,
        bpe_num_operations: int | None = None,
    ) -> None:
        self._type = type_

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

        if bpe_num_operations is not None or vocab_len is not None:
            self.build_vocab(vocab_len, bpe_dir, bpe_num_operations)

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
    def bpe_codes_file(self) -> Path | str | None:
        """Get vocabulary.

        :return: Vocabulary.
        """
        return self._bpe_codes_file

    @property
    def vocab_len(self) -> int:
        """Get vocabulary length.

        :return: Vocabulary length.
        """
        if self._vocab is None:
            raise AttributeError("Vocabulary not built.")

        return len(self._vocab)

    def build_vocab(
        self, vocab_len: int | None, bpe_dir: Path | None, bpe_num_operations: int | None
    ) -> None:
        """Build vocabulary from the captions in the dataset using BPE encoding.

        :param vocab_len: Amount of words for vocabulary. If ``None`` no word vocab is built.
        :param bpe_codes_file: Path to the BPE codes file. Is necessary if ``bpe_num_operations``
            is not ``None``.
        :param bpe_num_operations: Number of BPE merge operations. If ``None`` no BPE is used.
        """
        captions = self._captions["caption"].to_list()

        if bpe_num_operations is None and vocab_len is not None:
            words = [word for caption in captions for word in caption.split()]
            self._vocab = self.build_vocab_without_bpe(vocab_len, words)
            self._bpe_codes_file = None
        elif bpe_num_operations is not None:
            self._vocab = self.build_vocab_with_bpe(bpe_dir, bpe_num_operations)

        logger.info("Vocabulary has {} unique words", len(self._vocab))
        logger.info("Vocabulary coverage is {}%", self._coverage * 100)

    def build_vocab_without_bpe(self, vocab_len: int, words: list[str]) -> dict[str, int]:
        """Build vocabulary without BPE encoding.

        :param vocab_len: Amount of words for vocabulary.
        :param words: List of words from the captions.
        :return: Vocabulary without BPE encoding.
        """
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

        voc = {token: idx for idx, token in enumerate(truncated_sorted_vocab)}
        self._coverage = self.__calculate_coverage__(words, voc)
        return voc

    def build_vocab_with_bpe(self, bpe_dir: Path, bpe_num_operations: int) -> None:
        """Build vocabulary with BPE encoding.

        :param vocab_len: Amount of words for vocabulary.
        :param bpe_codes_file: Path to the BPE codes file.
        :param bpe_num_operations: Number of BPE merge operations.
        """
        self._bpe_codes_file = bpe_dir / "bpe.codes"
        bpe_captions_file = bpe_dir / f"captions-{self._type}.txt"

        captions = self._captions["caption"].to_list()
        with bpe_captions_file.open("w", encoding="utf-8") as f:
            f.write("\n".join(captions))

        if self._type == DatasetType.TRAIN:
            logger.info("Generating train BPE codes")
            with self._bpe_codes_file.open("w", encoding="utf-8") as codes_file:
                learn_bpe(
                    bpe_captions_file.open(encoding="utf-8"),
                    codes_file,
                    bpe_num_operations,
                    min_frequency=1,
                )

        bpe = BPE(self._bpe_codes_file.open(encoding="utf-8"))
        for idx, row in tqdm.tqdm(
            self._captions.iterrows(), f"Processing {self._type} captions with BPE codes"
        ):
            encoded_line = bpe.process_line(row["caption"])
            self._captions.at[idx, "caption"] = encoded_line

        # Build vocabulary from BPE-encoded codes
        bpe_words = []
        for caption in self._captions["caption"]:
            words = caption.split()
            bpe_words.extend(words)

        vocab = collections.Counter(bpe_words)
        sorted_vocab = sorted(vocab, key=lambda x: vocab[x], reverse=True)

        sorted_vocab.insert(0, "<eos>")
        sorted_vocab.insert(0, "<sos>")
        sorted_vocab.insert(0, "<unk>")
        sorted_vocab.insert(0, "<pad>")

        voc = {token: idx for idx, token in enumerate(sorted_vocab)}
        self._coverage = self.__calculate_coverage__(bpe_words, voc)
        return voc

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

    def __calculate_coverage__(self, split: list, voc: dict[str, int]) -> float:
        """Calculate the coverage of the vocabulary in the captions.

        :return: Coverage percentage.
        """
        total = 0.0
        unk = 0.0
        for token in split:
            if token not in voc:
                unk += 1.0
            total += 1.0
        return 1.0 - (unk / total)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of dataset.

        :return: Shape of dataset.
        """
        return self._videos[1].shape
