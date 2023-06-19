# -*- coding: utf-8 -*-
"""Script to prepare dataset with VideoMAE."""
import click
import joblib
import pandas as pd
from loguru import logger

from vid_cap import DATA_DIR
from vid_cap.modelling import preprocessing as pre


@click.command("prepare-dataset")
def main() -> None:
    """Prepare dataset with VideoMAE."""
    logger.info("Preparing dataset")

    for split in ("test", "val", "train"):
        logger.info(f"Preparing {split} set")

        dataset = {}

        captions_path = DATA_DIR / split / "captions.csv"
        videos_path = DATA_DIR / split / "videos"

        captions = pd.read_csv(captions_path, dtype_backend="pyarrow")[["video_id", "caption"]]

        counter = 0
        total = len(list(videos_path.iterdir()))

        for video in videos_path.iterdir():
            logger.info(f"Processing video {counter}/{total} from {split} set")

            video_path = videos_path / video
            stem = video.stem
            vid_id = int(stem.lstrip("video"))

            feat_vec = pre.gen_feat_vecs(video_path, 16)[0, :, :]

            dataset[vid_id] = {
                "id": vid_id,
                "features": feat_vec,
                "captions": captions.loc[lambda x: x["video_id"] == stem, "caption"].to_list(),
            }

            counter += 1

        joblib.dump(dataset, DATA_DIR / f"{split}.pickle")


if __name__ == "__main__":
    main()
