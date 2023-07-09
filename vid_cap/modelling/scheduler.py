# -*- coding: utf-8 -*-
"""Learning rate scheduler for transformer training."""
import torch
from loguru import logger


class NoamOptimizer:
    """Optim wrapper that implements learning rate schedule.

    :params model_size: Model size.
    :params factor: Factor.
    :params warmup: Warmup.
    :params optimizer: Optimizer.
    """

    def __init__(
        self, model_size: int, factor: float, warmup: int, optimizer: torch.optim.Optimizer
    ) -> None:
        self.optimizer = optimizer
        self._step = 0
        self.warmup = warmup
        self.factor = factor
        self.model_size = model_size
        self._rate = 0

    def step(self) -> None:
        """Update parameters and rate."""
        self._step += 1
        rate = self.rate()
        logger.info(f"Learning rate : {rate}")
        for p in self.optimizer.param_groups:
            p["lr"] = rate
        self._rate = rate
        self.optimizer.step()

    def rate(self, step: int | None = None) -> None:
        """Implement `lrate` above.

        :param step: Step. Defaults to ``None``.
        """
        if step is None:
            step = self._step
        return self.factor * (
            self.model_size ** (-0.5) * min(step ** (-0.5), step * self.warmup ** (-1.5))
        )

    def zero_grad(self) -> None:
        """Zero gradients."""
        self.optimizer.zero_grad()
