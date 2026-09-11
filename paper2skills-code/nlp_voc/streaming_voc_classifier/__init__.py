"""Streaming VOC Classifier

基于 AdaNEN (Adaptive Neural Ensemble Network) 思想简化实现：
  1. 多个分类器组成集成，各基于不同时间窗口的数据
  2. 滑动窗口检测概念漂移（分布变化）
  3. 根据验证窗口上的准确率动态调整集成权重
  4. 漂移时自动添加新分类器，旧分类器权重衰减

参考: A Novel Neural Ensemble Architecture for On-the-fly Classification
      of Evolving Text Streams (ACM TKDD 2024)
"""

from .streaming_classifier import (
    AdaNENClassifier,
    DriftDetector,
    EnsembleClassifier,
)

__all__ = [
    "AdaNENClassifier",
    "DriftDetector",
    "EnsembleClassifier",
]
