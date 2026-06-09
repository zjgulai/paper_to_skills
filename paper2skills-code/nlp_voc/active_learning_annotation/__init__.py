"""ALCHEmist: 低成本 LLM 主动标注

模块结构:
    - annotator.py: LLM-as-Annotator 核心实现（含模拟模式）
    - active_learner.py: 主动学习循环 + 不确定性采样
    - batch_pipeline.py: 批量标注流水线

生产环境建议:
    - 接入真实 LLM API (OpenAI / Anthropic / 本地 vLLM)
    - 使用 sentence-transformers 做嵌入
    - 使用 sklearn 训练轻量级分类器
"""

from .annotator import LLMAnnotator, AnnotationResult
from .active_learner import ActiveLearner, UncertaintySampler
from .batch_pipeline import BatchAnnotationPipeline

__all__ = [
    "LLMAnnotator",
    "AnnotationResult",
    "ActiveLearner",
    "UncertaintySampler",
    "BatchAnnotationPipeline",
]
