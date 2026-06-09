"""VOC Proxy NPS × AIPL 全旅程指标落地 Skills 工作流

端到端工作流：异构 VOC 数据 → 质量筛选 → 自动打标 → 画像×AIPL矩阵 → 三层价值洞察

核心组件:
- UnifiedLabelExtraction: 统一标签萃取引擎（376标签 + 55画像 + AIPL 7节点）
- PersonaAIPLMatrixBuilder: 画像×AIPL矩阵构建器
- ValueInsightEngine: 三层价值洞察引擎（监控/分析/决策）
- VOCProxyNPSWorkflow: 工作流编排器
- DashboardGenerator: 指标看板生成器

Usage:
    from proxy_nps_aipl_workflow import (
        VOCProxyNPSWorkflow,
        PersonaAIPLMatrixBuilder,
        ValueInsightEngine,
    )

    # 1. 运行萃取
    workflow = VOCProxyNPSWorkflow(tag_dict_path="tag_seeds.csv")
    results = workflow.run(voc_records)

    # 2. 构建画像×AIPL矩阵
    matrix_builder = PersonaAIPLMatrixBuilder()
    matrix = matrix_builder.build(results.extractions)

    # 3. 运行三层洞察
    engine = ValueInsightEngine()
    report = engine.run(matrix, results.extractions)
"""

from unified_label_extraction import (
    AIPLTagMatch,
    AtomicPersonaTag,
    BrandDetector,
    DashboardData,
    DashboardGenerator,
    PersonaTagMatcher,
    ProxyNPSCalculator,
    SentimentCalibrator,
    TagSeed,
    TagSeedDictionary,
    UnifiedLabelingPipeline,
    VOCRecord,
    VOCLabelExtraction,
    VOCLabelExtractor,
    create_demo_tag_dictionary,
    derive_business_persona,
)
from workflow import VOCProxyNPSWorkflow, WorkflowResults, run_voc_proxy_nps_workflow
from persona_aipl_matrix import (
    PersonaAIPLMatrix,
    PersonaAIPLMatrixBuilder,
    MatrixCell,
    PERSONA_DIMENSIONS,
    AIPL_NODES,
    DIMENSION_NAMES,
    AIPL_NODE_NAMES,
    CELL_METRICS,
)
from insight_engine import (
    ValueInsightEngine,
    MonitoringLayer,
    AnalysisLayer,
    DecisionLayer,
    MonitoringReport,
    AnalysisReport,
    DecisionReport,
)

__all__ = [
    # 工作流
    "VOCProxyNPSWorkflow",
    "WorkflowResults",
    "run_voc_proxy_nps_workflow",
    # 萃取引擎
    "VOCLabelExtractor",
    "VOCLabelExtraction",
    "UnifiedLabelingPipeline",
    # 画像×AIPL矩阵
    "PersonaAIPLMatrix",
    "PersonaAIPLMatrixBuilder",
    "MatrixCell",
    "PERSONA_DIMENSIONS",
    "AIPL_NODES",
    "DIMENSION_NAMES",
    "AIPL_NODE_NAMES",
    "CELL_METRICS",
    # 价值洞察引擎
    "ValueInsightEngine",
    "MonitoringLayer",
    "AnalysisLayer",
    "DecisionLayer",
    "MonitoringReport",
    "AnalysisReport",
    "DecisionReport",
    # 数据模型
    "VOCRecord",
    "TagSeed",
    "TagSeedDictionary",
    "AIPLTagMatch",
    "AtomicPersonaTag",
    # 子组件
    "PersonaTagMatcher",
    "SentimentCalibrator",
    "BrandDetector",
    "ProxyNPSCalculator",
    "DashboardGenerator",
    "DashboardData",
    # 工具函数
    "derive_business_persona",
    "create_demo_tag_dictionary",
]
