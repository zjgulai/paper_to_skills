"""TaxoAdapt: Taxonomy 动态演化

模块结构:
    - taxonomy_builder.py: 种子 Taxonomy 生成 + 多维 Taxonomy 管理
    - iterative_classifier.py: 迭代层级分类器
    - expander.py: 宽度/深度自适应扩展引擎
    - pipeline.py: 完整流水线

核心流程:
    1. 种子 Taxonomy 生成 (LLM / 人工预定义)
    2. 迭代层级分类 (L1 → L2 → L3 → L4)
    3. 未覆盖文本聚类分析
    4. 宽度/深度自适应扩展
    5. 一致性校验与输出
"""

from .taxonomy_builder import TaxonomyNode, TaxonomyTree, MultidimensionalTaxonomy
from .iterative_classifier import IterativeClassifier
from .expander import WidthExpander, DepthExpander
from .pipeline import TaxoAdaptPipeline

__all__ = [
    "TaxonomyNode",
    "TaxonomyTree",
    "MultidimensionalTaxonomy",
    "IterativeClassifier",
    "WidthExpander",
    "DepthExpander",
    "TaxoAdaptPipeline",
]
