"""Review Quality Scoring: 评论质量评分

基于 AutoQual (EMNLP 2025) + BHeIPCoRT (Applied Intelligence 2025) 的工程实现。

模块结构:
    - feature_engine.py: 4维度可解释特征提取
    - quality_scorer.py: 质量评分引擎（加权综合）
    - spam_detector.py: 虚假评论检测（模板/矛盾/极端）
    - pipeline.py: 完整流水线 + AutoTag 集成接口

核心流程:
    1. 特征提取: 信息丰富度 / 评分一致性 / 语言真实性 / 实用性
    2. 质量评分: 加权综合 → 0-100 分
    3. 虚假检测: 多规则组合 → 虚假概率
    4. 综合决策: 质量分 + 虚假概率 → 过滤决策
    5. 报告输出: 维度分解 + 行动建议
"""

from .feature_engine import FeatureExtractor, QualityFeatures
from .quality_scorer import QualityReport, QualityScore, ReviewQualityScorer
from .spam_detector import SpamDetectionResult, SpamDetector
from .pipeline import (
    PipelineResult,
    PipelineReport,
    ReviewQualityPipeline,
    review_quality_pipeline,
)

__all__ = [
    "FeatureExtractor",
    "QualityFeatures",
    "ReviewQualityScorer",
    "QualityScore",
    "QualityReport",
    "SpamDetector",
    "SpamDetectionResult",
    "ReviewQualityPipeline",
    "PipelineResult",
    "PipelineReport",
    "review_quality_pipeline",
]
