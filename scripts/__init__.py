"""FisherAdapTune — Fisher-guided Adaptive Fine-Tuning.

Public API
----------
FisherAdapTuneTrainer : main trainer class
AdaFisher            : diagonal FIM optimizer (used internally for Fisher collection)
EarlyStopping        : helper for early stopping
save_checkpoint      : save model + optimizer state
"""

from .adafisher import AdaFisher
from .trainer import FisherAdapTuneTrainer
from .utils import EarlyStopping, save_checkpoint

__all__ = [
    "FisherAdapTuneTrainer",
    "AdaFisher",
    "EarlyStopping",
    "save_checkpoint",
]
