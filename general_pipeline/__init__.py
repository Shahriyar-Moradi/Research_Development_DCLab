"""General Tabular Pipeline: Unified execution, benchmarking, and modeling."""

from .dataset import load_dataset
from .models import MODEL_REGISTRY, get_model
from .runner import run_experiment, run_quadrant
from .transformer import FTTransformerClassifier

__all__ = [
    "load_dataset",
    "get_model",
    "run_experiment",
    "run_quadrant",
    "MODEL_REGISTRY",
    "FTTransformerClassifier",
]
