# -*- coding: utf-8 -*-
"""Early stopper for training loop."""
import numpy as np


class EarlyStopper:
    """Early stopper for training loop.

    :param patience: Number of epochs to wait for improvement. Defaults to ``1``.
    :param min_delta: Minimum change in the monitored quantity to qualify as an improvement. Defaults to ``0``.
    """

    _patience: int
    _min_delta: float
    _counter: int
    _min_validation_loss: float

    def __init__(self, patience=1, min_delta=0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_validation_loss = np.inf

    def early_stop(self, validation_loss: float) -> bool:
        """Check if training should be stopped.

        :param validation_loss: Validation loss.
        :return: Whether training should be stopped.
        """
        if validation_loss < self._min_validation_loss:
            self._min_validation_loss = validation_loss
            self._counter = 0
        elif validation_loss > (self._min_validation_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True
        return False
