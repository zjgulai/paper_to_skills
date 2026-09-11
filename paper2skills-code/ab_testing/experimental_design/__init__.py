"""
A/B Experimental Design Basics
A/B 实验设计基础工具包

涵盖样本量计算、统计功效分析、最小可检测效应(MDE)、
分层随机分配与 CUPED 方差缩减。

主要参考:
- Zhou et al. (2023). All about Sample-Size Calculations for A/B Testing. CIKM.
- Deng et al. (2013). Improving the Sensitivity of Online Controlled Experiments. WSDM.
"""

from .design import (
    ABTestDesigner,
    cuped_adjustment,
    mde_calculator,
    power_analysis,
    sample_size_binary,
    sample_size_continuous,
    sample_size_relative_lift,
    stratified_allocation,
)

__all__ = [
    "sample_size_continuous",
    "sample_size_binary",
    "sample_size_relative_lift",
    "power_analysis",
    "mde_calculator",
    "stratified_allocation",
    "cuped_adjustment",
    "ABTestDesigner",
]
