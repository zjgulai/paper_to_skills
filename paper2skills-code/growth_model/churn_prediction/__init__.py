"""Churn Prediction package.

Two implementations are bundled:
- model_baseline: sklearn (RandomForest / GradientBoosting) — fast baseline
- model_dnn: PyTorch deep neural network — higher accuracy on large datasets
"""

from .model_baseline import ChurnPredictor, generate_sample_data
from . import model_dnn

__all__ = ['ChurnPredictor', 'generate_sample_data', 'model_dnn']
