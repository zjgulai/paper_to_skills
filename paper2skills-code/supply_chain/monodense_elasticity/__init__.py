"""
Monodense Deep Neural Model for Item Price Elasticity

基于论文: Monodense Deep Neural Model for Determining Item Price Elasticity
(arXiv:2603.29261, Walmart Inc.)

核心创新:
- Monodense 层: 在神经网络中强制价格→需求的单调递减约束
- 无需对照实验即可从大规模交易数据中学习单品价格弹性
- 经济学一致性保证: 输出的弹性始终为负
"""

from .model import (
    ElasticityEstimator,
    MonodenseDLM,
    MonodenseLayer,
    generate_momcozy_pricing_data,
)

__all__ = [
    "ElasticityEstimator",
    "MonodenseDLM",
    "MonodenseLayer",
    "generate_momcozy_pricing_data",
]
