# -*- coding: utf-8 -*-
"""Model saver for training loop."""
from pathlib import Path

import numpy as np
import torch


class ModelSaver:
    """Model saver for training loop."""

    _min_validation_loss: float

    def __init__(self) -> None:
        self._min_validation_loss = np.inf

    def save_if_best_model(
        self, validation_loss: float, model: torch.nn.Module, data_dir: Path, model_name: str
    ) -> None:
        """Save model if it is the best model so far.

        :param validation_loss: Validation loss.
        :param model: Model to save.
        :param data_dir: Data directory.
        :param model_name: Model name.
        """
        if validation_loss < self._min_validation_loss:
            self._min_validation_loss = validation_loss
            data_dir.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), data_dir / model_name)
