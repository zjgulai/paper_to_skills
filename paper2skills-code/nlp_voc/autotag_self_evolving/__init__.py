"""VOC AutoTag Self-Evolving Label System

基于 InsightNet 思想的层级标签自动分类与自进化系统。

主要模块:
    - model.py: 多任务层级分类器 + 新标签发现
    - label_system.py: 标签体系管理（层级关系、CRUD）
    - evolution.py: 标签进化引擎

使用示例:
    >>> from autotag_self_evolving import LabelSystem, MultiTaskClassifier, LabelEvolution
    >>> system = LabelSystem.from_yaml("labels.yaml")
    >>> classifier = MultiTaskClassifier(system)
    >>> result = classifier.predict("纸尿裤晚上侧漏严重，宝宝睡不好")
    >>> print(result)
    {
        'l1': '纸尿裤',
        'l2': '质量',
        'l3': '漏尿问题',
        'l4': '夜间侧漏',
        'sentiment': -1,
        'verbatim': '侧漏严重',
        'confidence': 0.92
    }
"""

from .label_system import LabelSystem, LabelNode
from .model import MultiTaskClassifier, PredictionResult
from .evolution import LabelEvolution, EvolutionTrigger

__all__ = [
    "LabelSystem",
    "LabelNode",
    "MultiTaskClassifier",
    "PredictionResult",
    "LabelEvolution",
    "EvolutionTrigger",
]
