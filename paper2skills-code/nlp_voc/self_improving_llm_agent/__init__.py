"""
Self-Improving LLM Agent Pipeline
自迭代 LLM Agent 管线

核心组件:
- ReflexionEngine: 反思引擎，从执行结果生成结构化反思报告
- SelfRefineEngine: 自精炼引擎，将反思转化为可执行的编辑指令
- DPOTrainer: 轻量级 DPO 训练器，管理偏好对比数据
- SelfImprovingAgent: 完整的 GRO (Generate-Review-Optimize) 闭环 Agent

业务适配:
- CopyOptimizationAgent: 电商文案优化
- IntelligenceExtractionAgent: 竞品情报萃取
"""

from .model import (
    ExecutionResult,
    ReflectionReport,
    SelfEditInstruction,
    ReflexionEngine,
    SelfRefineEngine,
    DPOTrainer,
    SelfImprovingAgent,
    CopyOptimizationAgent,
    IntelligenceExtractionAgent,
)

__all__ = [
    "ExecutionResult",
    "ReflectionReport",
    "SelfEditInstruction",
    "ReflexionEngine",
    "SelfRefineEngine",
    "DPOTrainer",
    "SelfImprovingAgent",
    "CopyOptimizationAgent",
    "IntelligenceExtractionAgent",
]
