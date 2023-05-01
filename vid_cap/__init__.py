# -*- coding: utf-8 -*-
"""Top level package for Video Caption Generator."""
import os
from importlib import metadata
from pathlib import Path

__version__ = metadata.version("vid_cap")


# Base path of vid_cap module
# (to be used when accessing non .py files in Video Caption Generator/)
WORKDIR = Path(os.getenv("WORKDIR", Path.cwd()))
BASEPATH = Path(__file__).parent
ASSET_DIR = BASEPATH / "assets"
