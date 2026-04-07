"""
智能归因 - 因果森林 (Causal Forest)

用于母婴出海电商多市场智能广告归因和促销时机优化
"""

from .model import (
    CausalForestAttribution,
    MultiMarketAttributionAnalyzer,
    generate_multimarket_data
)

__all__ = [
    'CausalForestAttribution',
    'MultiMarketAttributionAnalyzer',
    'generate_multimarket_data'
]
