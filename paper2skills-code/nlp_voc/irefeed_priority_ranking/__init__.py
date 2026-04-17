"""
iReFeed: interconnected ReFeed Pipeline

基于论文: Enhancing User-Feedback Driven Requirements Prioritization
(arXiv:2603.28677)

将用户反馈驱动的需求优先级排序从"单需求独立评估"升级为"需求簇互联评估"，
通过 LDA 主题聚类、簇级反馈关联、D-value 依赖价值和 NSGA-II 三目标优化，
生成最优产品迭代路线图。
"""

from .model import (
    CandidateRequirement,
    FeedbackCluster,
    iReFeedPrioritizer,
    TopicModeler,
    UserFeedback,
)

__all__ = [
    "CandidateRequirement",
    "FeedbackCluster",
    "iReFeedPrioritizer",
    "TopicModeler",
    "UserFeedback",
]
