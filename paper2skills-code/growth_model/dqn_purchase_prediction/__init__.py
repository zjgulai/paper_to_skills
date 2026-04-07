"""
DQN-Inspired Deep Learning for Purchase Intent Prediction
DQN深度强化学习购买意图预测

论文: Predicting E-commerce Purchase Behavior using a DQN-Inspired Deep Learning Model
arXiv: 2506.17543 (2025)

主要组件:
- ExperienceReplayBuffer: 经验回放缓冲区
- DQNLSTMNetwork: DQN+LSTM网络
- DQNPurchasePredictor: 购买意图预测器
- AIPLPurchaseScorer: AIPL购买评分器
"""

from .model import (
    UserSession,
    ExperienceReplayBuffer,
    DQNLSTMNetwork,
    DQNPurchasePredictor,
    AIPLPurchaseScorer,
    create_sample_sessions,
    test_dqn_purchase_prediction
)

__all__ = [
    'UserSession',
    'ExperienceReplayBuffer',
    'DQNLSTMNetwork',
    'DQNPurchasePredictor',
    'AIPLPurchaseScorer',
    'create_sample_sessions',
    'test_dqn_purchase_prediction'
]
