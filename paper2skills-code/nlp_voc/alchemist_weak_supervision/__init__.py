"""ALCHEmist: 弱监督自动标注 — LLM 生成标注程序

模块结构:
    - label_function.py: Label Function 定义与执行
    - program_generator.py: LLM 程序生成器
    - aggregator.py: 多程序投票聚合
    - pipeline.py: 完整流水线

核心差异（vs 主动学习标注）:
    - 生成可复用的标注程序，而非逐条标注
    - 程序可审计、可修改、零后续运行成本
    - 多程序投票 + 置信度加权聚合
"""

from .label_function import LabelFunction, LFRegistry
from .program_generator import ProgramGenerator
from .aggregator import MajorityVoteAggregator, ProbabilisticAggregator
from .pipeline import ALCHEmistPipeline

__all__ = [
    "LabelFunction",
    "LFRegistry",
    "ProgramGenerator",
    "MajorityVoteAggregator",
    "ProbabilisticAggregator",
    "ALCHEmistPipeline",
]
