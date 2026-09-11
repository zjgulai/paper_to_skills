"""Open World Class Incremental Learning

开放世界增量文本分类：在运行时发现新类别并扩展分类能力。
"""

from .openworld_classifier import (
    ClassPrototype,
    IncrementResult,
    OpenWorldClassifier,
    PredictionResult,
)

__all__ = [
    "OpenWorldClassifier",
    "ClassPrototype",
    "PredictionResult",
    "IncrementResult",
]
