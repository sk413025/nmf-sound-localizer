"""
NMF Sound Localizer

A modular toolkit for NMF-based sound source localization with fixed group sparsity.
Supports separate datasets workflow to eliminate data leakage.

"""

from .pipeline.full_pipeline import NMFLocalizationPipeline
from .pipeline.experiment_runner import ExperimentRunner
from .config.defaults import NMFConfig
from .core.data_processor import DataProcessor
from .core.usm_trainer import USMTrainer
from .core.stft_unified_processor import STFTUnifiedProcessor
from .core.localizer import NMFSoundLocalizer
from .core.evaluator import Evaluator

__version__ = "0.1.0"
__author__ = "Speech Processing Lab"

__all__ = [
    "NMFLocalizationPipeline",
    "ExperimentRunner", 
    "NMFConfig",
    "DataProcessor", 
    "USMTrainer",
    "STFTUnifiedProcessor",
    "NMFSoundLocalizer",
    "Evaluator",
]