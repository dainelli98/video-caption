# -*- coding: utf-8 -*-
"""Scripts."""
from .experiment import main as experiment
from .prepare_dataset import main as prepare_dataset
from .test import main as test
from .train import main as train

__all__: list[str] = ["experiment", "prepare_dataset", "train", "test"]
