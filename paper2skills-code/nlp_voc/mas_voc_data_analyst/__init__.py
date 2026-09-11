"""Multi-Agent VOC Data Analyst

LLM-based multi-agent system for qualitative VOC data analysis.
Based on arXiv:2402.01386.

Core pipeline (7 Agents):
- DataIngestionAgent: Clean and preprocess VOC records
- ThematicAnalysisAgent: Extract themes and sentiment
- CodebookAgent: Generate structured codebook
- PatternRecognitionAgent: Detect co-occurrence, trends, anomalies
- InsightSynthesisAgent: Generate actionable insights
- QualityVerificationAgent: Check consistency and coverage
- ReportGeneratorAgent: Produce final structured report

Usage:
    from mas_voc_data_analyst import MultiAgentVOCPipeline, create_demo_voc_data

    pipeline = MultiAgentVOCPipeline()
    report = pipeline.run(raw_data, dataset_name="momcozy_voc")
"""

from model import (
    AnalysisResult,
    BaseAgent,
    CodebookAgent,
    DataIngestionAgent,
    Insight,
    InsightSynthesisAgent,
    MultiAgentVOCPipeline,
    Pattern,
    PatternRecognitionAgent,
    QualityVerificationAgent,
    ReportGeneratorAgent,
    ThemeCluster,
    ThematicAnalysisAgent,
    VOCAnalysisReport,
    VOCRecord,
    create_demo_voc_data,
    print_report,
)

__all__ = [
    "AnalysisResult",
    "BaseAgent",
    "CodebookAgent",
    "DataIngestionAgent",
    "Insight",
    "InsightSynthesisAgent",
    "MultiAgentVOCPipeline",
    "Pattern",
    "PatternRecognitionAgent",
    "QualityVerificationAgent",
    "ReportGeneratorAgent",
    "ThemeCluster",
    "ThematicAnalysisAgent",
    "VOCAnalysisReport",
    "VOCRecord",
    "create_demo_voc_data",
    "print_report",
]
