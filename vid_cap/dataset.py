# -*- coding: utf-8 -*-
"""Dataset class for loading video feature vector data."""
import collections
from pathlib import Path
import tqdm

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from subword_nmt.learn_bpe import learn_bpe
from subword_nmt.apply_bpe import BPE
import calendar
import time

class VideoEvalDataset(Dataset):
    """Dataset with video feature vectors and associated captions.

    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
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

    :param video_dir: Directory with videos.
    :param caps_file: Path to captions file.
    :param caps_per_vid: Amount of captions per video, min=1, max=20.
        Defaults to ``None``.
    :param vocab_len: Amount of words for vocabulary. If ``None`` no vocab is built.
    """

    _type: str
    _captions: pd.DataFrame
    _videos: dict[int, torch.Tensor]
    _vocab: dict[str, int] | None
    _coverage: float
    _bpe_codes_file: Path | str | None = None

    def __init__(
        self,
        type: str,
        video_dir: Path | str,
        caps_file: Path | str,
        caps_per_vid: int | None = None,
        vocab_len: int | None = None,
        data_dir: Path | str | None = None,
        bpe_num_operations: int | None = None
    ) -> None:
        self._type = type

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

        self.build_vocab(vocab_len, data_dir, bpe_num_operations)

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

    def build_vocab(self, vocab_len: int, data_dir: Path, bpe_num_operations: int | None) -> None:
        """Build vocabulary from the captions in the dataset using BPE encoding.

        :param vocab_len: Amount of words for vocabulary.
        :param bpe_codes_file: Path to the BPE codes file.
        :param bpe_num_operations: Number of BPE merge operations.
        """
        captions = self._captions["caption"].to_list()
        
        
        if bpe_num_operations is None:
            words = [word for caption in captions for word in caption.split()]
            self._vocab = self.build_vocab_without_bpe(vocab_len, words)
        else:
            self._vocab = self.build_vocab_with_bpe(data_dir, bpe_num_operations)

        print('* Unique words', len(self._vocab))
        print('* Coverage', self._coverage)

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

        truncated_sorted_vocab = sorted_vocab[:vocab_len]
        voc = {token: idx for idx, token in enumerate(truncated_sorted_vocab)}
        self._coverage = self.__calculate_coverage__(words, voc)
        return voc

    def build_vocab_with_bpe(self, data_dir: Path, bpe_num_operations: int) -> None:
        """Build vocabulary with BPE encoding.

        :param vocab_len: Amount of words for vocabulary.
        :param bpe_codes_file: Path to the BPE codes file.
        :param bpe_num_operations: Number of BPE merge operations.
        """
        self._bpe_codes_file = data_dir / "bpe" / f"bpe.codes"
        bpe_captions_file = data_dir / "bpe" / f"captions-{self._type}.txt"

        captions = self._captions["caption"].to_list()
        with open(bpe_captions_file, "w", encoding="utf-8") as f:
            f.write('\n'.join(captions))

        if self._type == "train":
            # Generate BPE codes
            print("Generating BPE codes")
            with open(self._bpe_codes_file, "w", encoding="utf-8") as codes_file:
                learn_bpe(open(bpe_captions_file, "r", encoding="utf-8"), codes_file, bpe_num_operations, min_frequency=1)

        bpe = BPE(open(self._bpe_codes_file, "r", encoding="utf-8"))
        for idx, row in tqdm.tqdm( self._captions.iterrows(), f"Processing captions with BPE codes"):
            encoded_line = bpe.process_line(row["caption"])
            self._captions.at[idx, "caption"] = encoded_line


        # Build vocabulary from BPE-encoded codes
        bpe_words = []
        for caption in self._captions['caption']:
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
    
    def __calculate_coverage__(self, split: list, voc) -> float:
        """Calculate the coverage of the vocabulary in the captions.

        :return: Coverage percentage.
        """
        total = 0.0
        unk = 0.0
        for token in split:
            if not token in voc:
                unk += 1.0
            total += 1.0
        return 1.0 - (unk/total)

    @property
    def shape(self) -> tuple[int, int]:
        """Get shape of dataset.

        :return: Shape of dataset.
        """
        return self._videos[1].shape
