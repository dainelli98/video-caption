# -*- coding: utf-8 -*-
"""Script to prepare dataset with VideoMAE."""
import click
import numpy as np
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

        captions_path = DATA_DIR / split / "captions.csv"
        videos_path = DATA_DIR / split / "videos"

        captions = pd.read_csv(captions_path, dtype_backend="pyarrow")[["video_id", "caption"]]

        counter = 0
        total = len(list(videos_path.iterdir()))

        out_caps = []

        (DATA_DIR / "output" / split / "videos").mkdir(parents=True, exist_ok=True)

        for video in videos_path.iterdir():
            logger.info(f"Processing video {counter+1}/{total} from {split} set")

            video_path = videos_path / video
            stem = video.stem

            feat_vec = pre.gen_feat_vecs(video_path, 16)[0, :, :]

            np.save(DATA_DIR / "output" / split / "videos" / f"{counter}.npy", feat_vec)

            vid_caps = captions.loc[lambda x: x["video_id"] == stem, "caption"].to_list()

            out_caps.append(
                pd.DataFrame(
                    {
                        "video": counter,
                        "n_cap": range(1, len(vid_caps) + 1),
                        "caption": vid_caps,
                    }
                )
            )

            counter += 1

        pd.concat(out_caps).to_parquet(DATA_DIR / "output" / split / "captions.parquet")


if __name__ == "__main__":
    main()
