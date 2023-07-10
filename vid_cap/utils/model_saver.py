import numpy as np
import torch

class ModelSaver:
    _min_validation_loss: float
    
    def __init__(self)-> None:
        self._min_validation_loss = np.inf

    def save_if_best_model(self, validation_loss, model, data_dir, model_name):
        if validation_loss < self._min_validation_loss:
            self._min_validation_loss = validation_loss
            torch.save(model.state_dict(), data_dir / "output" / model_name)