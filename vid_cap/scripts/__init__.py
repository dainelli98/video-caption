# -*- coding: utf-8 -*-
"""Scripts."""
from .prepare_dataset import main as prepare_dataset
from .train import main as train

__all__: list[str] = ["prepare_dataset", "train"]
