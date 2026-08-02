"""Global reproducibility utilities for training scripts."""

import logging
import os
import random

import numpy as np
import torch

logger = logging.getLogger(__name__)


class ReproducibilityManager:
    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def seed_everything(self) -> None:
        random.seed(self.seed)
        np.random.seed(self.seed)
        os.environ["PYTHONHASHSEED"] = str(self.seed)
        torch.manual_seed(self.seed)

        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(self.seed)
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        if torch.backends.mps.is_available():
            torch.mps.manual_seed(self.seed)

        logger.info("Seeds fixed to %d (random, numpy, torch)", self.seed)
