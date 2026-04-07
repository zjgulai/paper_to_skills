"""
智能预测 - 双重稳健估计 (Doubly Robust Estimation)

用于母婴出海电商促销效果预测和新产品上架时机决策
"""

from .model import (
    DoublyRobustEstimator,
    PromotionEffectAnalyzer,
    generate_promotion_data
)

__all__ = [
    'DoublyRobustEstimator',
    'PromotionEffectAnalyzer',
    'generate_promotion_data'
]
