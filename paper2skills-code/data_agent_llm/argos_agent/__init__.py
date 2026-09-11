"""
Argos-inspired Agentic Time-Series Anomaly Detection
基于三Agent协作的简化版实现

论文: ARGOS: Agentic Time-Series Anomaly Detection with Autonomous Rule Generation via LLMs
arXiv: 2501.14170
"""

from .anomaly_detector import ArgosAnomalyDetector, DetectionRule

__all__ = ["ArgosAnomalyDetector", "DetectionRule"]
