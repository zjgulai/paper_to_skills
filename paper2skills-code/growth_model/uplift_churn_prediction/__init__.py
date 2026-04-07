"""
Uplift Modeling for Churn Prediction
Uplift建模在用户流失预测中的应用

论文: A churn prediction dataset from the telecom sector: a new benchmark for uplift modeling
arXiv: 2312.07206 (2023)
会议: ECML PKDD 2023 Workshop

主要组件:
- BaseUpliftModel: Uplift建模基类
- TLearner: 分别训练处理组和对照组模型
- SLearner: 将干预作为特征输入单一模型
- XLearner: 结合T-Learner和S-Learner优势
- UpliftMetrics: Qini曲线和AUUC评估指标
- CustomerUpliftAnalyzer: 客户Uplift分析器

使用示例:
    from model import CustomerUpliftAnalyzer, create_sample_customers

    customers = create_sample_customers(1000)
    analyzer = CustomerUpliftAnalyzer(model_type='xlearner')
    analyzer.fit(customers)
    result = analyzer.analyze_customer(customers[0])
"""

from .model import (
    CustomerData,
    BaseUpliftModel,
    TLearner,
    SLearner,
    XLearner,
    UpliftMetrics,
    CustomerUpliftAnalyzer,
    create_sample_customers,
    test_uplift_modeling
)

__all__ = [
    'CustomerData',
    'BaseUpliftModel',
    'TLearner',
    'SLearner',
    'XLearner',
    'UpliftMetrics',
    'CustomerUpliftAnalyzer',
    'create_sample_customers',
    'test_uplift_modeling'
]
