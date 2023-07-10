import numpy as np

class EarlyStopper:
    _patience: int
    _min_delta: float
    _counter: int
    _min_validation_loss: float
    
    def __init__(self, patience=1, min_delta=0) -> None:
        self._patience = patience
        self._min_delta = min_delta
        self._counter = 0
        self._min_validation_loss = np.inf

    def early_stop(self, validation_loss):
        if validation_loss < self._min_validation_loss:
            self._min_validation_loss = validation_loss
            self._counter = 0
        elif validation_loss > (self._min_validation_loss + self._min_delta):
            self._counter += 1
            if self._counter >= self._patience:
                return True
        return False