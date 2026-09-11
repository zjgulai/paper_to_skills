"""
Customer Journey Prototype Detection with Counterfactual Explanations
客户旅程序列原型检测与反事实解释

论文: Analysis of Customer Journeys Using Prototype Detection and Counterfactual Explanations
arXiv: 2505.11086 (2025)

主要组件:
- SequenceDistance: 序列距离计算
- PrototypeDetector: 原型序列检测
- PurchasePredictor: 购买概率预测
- CounterfactualRecommender: 反事实推荐
- CustomerJourneyAnalyzer: 整合分析系统
"""

from .model import (
    JourneyEvent,
    CustomerJourney,
    SequenceDistance,
    PrototypeDetector,
    PurchasePredictor,
    CounterfactualRecommender,
    CustomerJourneyAnalyzer,
    create_sample_journeys,
    test_customer_journey_analysis
)

__all__ = [
    'JourneyEvent',
    'CustomerJourney',
    'SequenceDistance',
    'PrototypeDetector',
    'PurchasePredictor',
    'CounterfactualRecommender',
    'CustomerJourneyAnalyzer',
    'create_sample_journeys',
    'test_customer_journey_analysis'
]
