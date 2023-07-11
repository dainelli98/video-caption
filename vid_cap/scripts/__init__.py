# -*- coding: utf-8 -*-
"""Scripts."""
from .prepare_dataset import main as prepare_dataset
from .train import main as train
from .test import main as test

__all__: list[str] = ["prepare_dataset", "train", "test"]
