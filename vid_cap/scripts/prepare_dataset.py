# -*- coding: utf-8 -*-
"""Script to prepare dataset with VideoMAE."""
from pathlib import Path

import click
import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from loguru import logger

from vid_cap import DATA_DIR
from vid_cap.modelling import preprocessing as pre

_TRAIN_MAX = 488.92505
_TRAIN_MIN = -96.46513

_FLOAT16_MAX = np.finfo(np.float16).max
_FLOAT16_MIN = np.finfo(np.float16).min

_SAMPLE_PERIOD = 32

@click.command("prepare-dataset")
@click.option("--data-dir", default=DATA_DIR, type=click.Path(exists=True), help="Data directory")
def main(data_dir: Path) -> None:
    """Prepare dataset with ``VideoMAE``.

    \f

    :param data_dir: Path to data directory.
    """
    logger.info("Preparing dataset")

    for split in ("test", "val", "train"):
        logger.info(f"Preparing {split} set")

        captions_path = data_dir / split / "captions.csv"
        videos_path = data_dir / split / "videos"

        captions = pd.read_csv(captions_path, dtype_backend="pyarrow")[["video_id", "caption"]]

        total = len(list(videos_path.iterdir()))

        (data_dir / "output" / split / "videos").mkdir(parents=True, exist_ok=True)

        out_caps = Parallel(n_jobs=1, verbose=1000)(
            delayed(_process_video)(data_dir, split, videos_path, captions, counter, total, video)
            for counter, video in enumerate(videos_path.iterdir(), 1)
        )

        pd.concat(out_caps).to_parquet(data_dir / "output" / split / "captions.parquet")


def _process_video(data_dir, split, videos_path, captions, counter, total, video):
    logger.info(f"Processing video {counter+1}/{total} from {split} set")

    video_path = videos_path / video
    stem = video.stem

    feat_vec = pre.gen_feat_vecs(video_path, 16)[0, :, :]
    
    feat_vec = feat_vec[::_SAMPLE_PERIOD, :]

    feat_vec = (feat_vec - _TRAIN_MIN) / (_TRAIN_MAX - _TRAIN_MIN)

    feat_vec = feat_vec * (_FLOAT16_MAX - _FLOAT16_MIN) + _FLOAT16_MIN

    feat_vec = feat_vec.round().clip(_FLOAT16_MIN, _FLOAT16_MAX).astype(np.float16)

    np.save(data_dir / "output" / split / "videos" / f"{counter}.npy", feat_vec)

    vid_caps = captions.loc[lambda x: x["video_id"] == stem, "caption"].to_list()

    return pd.DataFrame(
        {
            "video": counter,
            "n_cap": range(1, len(vid_caps) + 1),
            "caption": vid_caps,
        }
    )


if __name__ == "__main__":
    main.main(standalone_mode=False)
