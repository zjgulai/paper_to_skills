"""
CSK: Cuckoo Search + K-means for Customer Sentiment Clustering
CSK客户情感聚类分析

论文: Customer Sentiment Analysis with Cuckoo Search and K-means Clustering
arXiv: 2311.11250 (2023)

主要组件:
- TextFeatureExtractor: 文本特征提取
- CuckooSearch: 布谷鸟搜索优化
- CSKClustering: CSK聚类算法
- CustomerSentimentAnalyzer: 客户情感分析器
"""

from .model import (
    CustomerReview,
    TextFeatureExtractor,
    CuckooSearch,
    CSKClustering,
    CustomerSentimentAnalyzer,
    create_sample_reviews,
    test_csk_sentiment_clustering
)

__all__ = [
    'CustomerReview',
    'TextFeatureExtractor',
    'CuckooSearch',
    'CSKClustering',
    'CustomerSentimentAnalyzer',
    'create_sample_reviews',
    'test_csk_sentiment_clustering'
]
