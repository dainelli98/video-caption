# -*- coding: utf-8 -*-
"""Data setup script."""

from torch.utils.data import Dataset


class VideoCaptionDataset(Dataset):
    def __init__(
        self, video_embeddings_with_captions_file, transform=None, target_transform=None
    ) -> None:
        # TODO: Load video embeddings with captions into the iterable.
        self._video_embeddings_with_captions = []

    def __len__(self) -> int:
        return len(self._video_embeddings_with_captions)

    def __getitem__(self, idx):
        # TODO: Return embedding and caption
        video_embedding, caption = self._video_embeddings_with_captions[idx]
        return video_embedding, caption
