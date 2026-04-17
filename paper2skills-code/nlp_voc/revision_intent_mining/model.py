#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
REVISION: Reflective Intent Mining for E-commerce
反直觉洞察：无点击不代表无意图，无点击行为暗示强烈隐性需求

论文来源: REVISION: Reflective Intent Mining and Online Reasoning
        Auxiliary for E-commerce Visual Search System Optimization
arXiv ID: 2510.22739

应用场景:
- 母婴出海电商：用户搜索"适合敏感肌的纸尿裤"但无点击
- 识别复杂意图并拆解为可执行的优化策略
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import DBSCAN
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


@dataclass
class SearchQuery:
    """搜索查询数据结构"""
    query_id: str
    query_text: str
    user_id: str
    timestamp: str
    clicked: bool = False
    clicked_items: List[str] = None
    metadata: Dict = None


@dataclass
class IntentCluster:
    """意图聚类结果"""
    cluster_id: int
    intent_label: str
    keywords: List[str]
    sample_queries: List[str]
    optimization_strategy: str
    tool_sequence: List[str]


class TextFeatureExtractor:
    """文本特征提取器"""

    def __init__(self, model_name: str = 'paraphrase-multilingual-MiniLM-L12-v2'):
        self.encoder = SentenceTransformer(model_name)
        self.tfidf = TfidfVectorizer(max_features=1000, ngram_range=(1, 2))

    def extract_semantic_features(self, texts: List[str]) -> np.ndarray:
        """提取语义特征（句子嵌入）"""
        embeddings = self.encoder.encode(texts, show_progress_bar=False)
        return embeddings

    def extract_lexical_features(self, texts: List[str]) -> np.ndarray:
        """提取词法特征（TF-IDF）"""
        tfidf_matrix = self.tfidf.fit_transform(texts)
        return tfidf_matrix.toarray()


class HierarchicalIntentClustering:
    """
    层次意图聚类

    Level 1: 粗粒度聚类 - 识别主要意图类别
    Level 2: 细粒度聚类 - 识别具体优化策略
    """

    def __init__(self,
                 coarse_eps: float = 0.5,
                 fine_eps: float = 0.35,
                 min_samples: int = 2):
        self.coarse_eps = coarse_eps
        self.fine_eps = fine_eps
        self.min_samples = min_samples
        self.feature_extractor = TextFeatureExtractor()
        self.coarse_clusters = None
        self.fine_clusters = {}

    def fit(self, queries: List[SearchQuery]) -> List[IntentCluster]:
        """
        执行层次聚类

        Args:
            queries: 搜索查询列表

        Returns:
            List[IntentCluster]: 意图聚类结果
        """
        texts = [q.query_text for q in queries]

        # 提取语义特征
        embeddings = self.feature_extractor.extract_semantic_features(texts)

        # Level 1: 粗粒度聚类（主要意图类别）
        coarse_dbscan = DBSCAN(
            eps=self.coarse_eps,
            min_samples=self.min_samples,
            metric='cosine'
        )
        coarse_labels = coarse_dbscan.fit_predict(embeddings)

        intent_clusters = []

        # Level 2: 在每个粗粒度类别内进行细粒度聚类
        for coarse_id in set(coarse_labels):
            if coarse_id == -1:  # 跳过噪声点
                continue

            mask = coarse_labels == coarse_id
            cluster_indices = np.where(mask)[0]
            cluster_embeddings = embeddings[mask]
            cluster_texts = [texts[i] for i in cluster_indices]

            # 细粒度聚类
            fine_dbscan = DBSCAN(
                eps=self.fine_eps,
                min_samples=self.min_samples,
                metric='cosine'
            )
            fine_labels = fine_dbscan.fit_predict(cluster_embeddings)

            # 为每个细粒度簇生成意图标签和优化策略
            for fine_id in set(fine_labels):
                if fine_id == -1:
                    continue

                fine_mask = fine_labels == fine_id
                fine_indices = cluster_indices[fine_mask]
                fine_texts = [texts[i] for i in fine_indices]

                # 生成意图标签和策略
                intent_label = self._generate_intent_label(fine_texts)
                optimization_strategy = self._generate_strategy(fine_texts, intent_label)
                tool_sequence = self._generate_tool_sequence(intent_label)
                keywords = self._extract_keywords(fine_texts)

                cluster = IntentCluster(
                    cluster_id=len(intent_clusters),
                    intent_label=intent_label,
                    keywords=keywords,
                    sample_queries=fine_texts[:5],
                    optimization_strategy=optimization_strategy,
                    tool_sequence=tool_sequence
                )
                intent_clusters.append(cluster)

        return intent_clusters

    def _generate_intent_label(self, texts: List[str]) -> str:
        """基于文本生成意图标签"""
        # 简单启发式：提取最常见的产品属性组合
        attribute_keywords = {
            'price': ['便宜', '贵', '价格', '优惠', '折扣', '性价比', 'cheap', 'expensive', 'price'],
            'quality': ['质量', '品质', '耐用', '好', '坏', 'quality', 'durable'],
            'safety': ['安全', '无害', '过敏', 'sensitive', 'safe', 'allergy'],
            'size': ['大小', '尺寸', '码', 'size', 'dimension'],
            'material': ['材质', '材料', '棉', 'material', 'cotton', 'fabric'],
            'brand': ['品牌', '牌子', 'brand', 'famous'],
            'usage': ['使用', '用法', '场景', 'usage', 'apply', 'scene']
        }

        detected_attrs = []
        for attr, keywords in attribute_keywords.items():
            if any(kw in ' '.join(texts).lower() for kw in keywords):
                detected_attrs.append(attr)

        if detected_attrs:
            return f"{'-'.join(detected_attrs[:3])}-focused"
        return "general-inquiry"

    def _generate_strategy(self, texts: List[str], intent_label: str) -> str:
        """基于意图生成优化策略"""
        strategies = {
            'price': "价格区间细分 + 优惠券推送",
            'quality': "质检报告展示 + 用户评价强化",
            'safety': "成分透明化 + 安全认证展示",
            'size': "尺码指南 + 试穿/试用政策",
            'material': "材质详情 + 对比工具",
            'brand': "品牌故事 + 口碑展示",
            'usage': "使用教程 + 场景化推荐"
        }

        applicable = []
        for attr, strategy in strategies.items():
            if attr in intent_label:
                applicable.append(strategy)

        return "; ".join(applicable) if applicable else "通用搜索优化"

    def _generate_tool_sequence(self, intent_label: str) -> List[str]:
        """生成工具调用序列"""
        tool_mapping = {
            'price': ['price_filter', 'coupon_display', 'comparison_tool'],
            'quality': ['review_highlight', 'certification_badge', 'detail_images'],
            'safety': ['ingredient_list', 'safety_cert', 'allergy_checker'],
            'size': ['size_guide', 'fit_predictor', 'return_policy'],
            'material': ['material_detail', 'texture_zoom', 'comparison_table'],
            'brand': ['brand_story', 'testimonial_carousel', 'expert_endorsement'],
            'usage': ['video_tutorial', 'usage_scenario', 'faq_section']
        }

        tools = []
        for attr in intent_label.split('-'):
            if attr in tool_mapping:
                tools.extend(tool_mapping[attr])

        return tools if tools else ['general_search_enhancement']

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """提取关键词"""
        # 使用TF-IDF提取关键词
        try:
            tfidf = TfidfVectorizer(max_features=10, ngram_range=(1, 2))
            tfidf.fit_transform(texts)
            return list(tfidf.get_feature_names_out())
        except:
            return texts[0].split()[:10] if texts else []


class OnlineIntentOptimizer:
    """
    在线意图优化器

    根据实时查询匹配最佳意图聚类，并返回优化策略
    """

    def __init__(self, intent_clusters: List[IntentCluster]):
        self.intent_clusters = intent_clusters
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.cluster_embeddings = self._precompute_cluster_embeddings()

    def _precompute_cluster_embeddings(self) -> np.ndarray:
        """预计算聚类中心嵌入"""
        cluster_texts = []
        for cluster in self.intent_clusters:
            # 使用样本文本和关键词代表聚类
            representative = ' '.join(cluster.sample_queries[:3] + cluster.keywords)
            cluster_texts.append(representative)

        return self.encoder.encode(cluster_texts, show_progress_bar=False)

    def optimize_query(self, query: SearchQuery, top_k: int = 1) -> List[Tuple[IntentCluster, float]]:
        """
        优化查询，返回最佳匹配的意图聚类和策略

        Args:
            query: 输入查询
            top_k: 返回前k个匹配结果

        Returns:
            List[Tuple[IntentCluster, float]]: (聚类, 相似度分数)
        """
        query_embedding = self.encoder.encode([query.query_text])

        # 计算与所有聚类的余弦相似度
        similarities = np.dot(self.cluster_embeddings, query_embedding.T).flatten()

        # 获取top-k匹配
        top_indices = np.argsort(similarities)[-top_k:][::-1]

        results = []
        for idx in top_indices:
            if similarities[idx] > 0.3:  # 相似度阈值
                results.append((self.intent_clusters[idx], float(similarities[idx])))

        return results


class REVISIONIntentMining:
    """
    REVISION意图挖掘主类

    整合离线聚类和在线优化，实现完整的意图挖掘流程
    """

    def __init__(self):
        self.clustering = HierarchicalIntentClustering()
        self.optimizer = None
        self.intent_clusters = None

    def offline_mine(self, no_click_queries: List[SearchQuery]) -> List[IntentCluster]:
        """
        离线阶段：从历史无点击查询中挖掘意图

        Args:
            no_click_queries: 历史无点击查询列表

        Returns:
            List[IntentCluster]: 挖掘出的意图聚类
        """
        print(f"开始离线意图挖掘，处理 {len(no_click_queries)} 条无点击查询...")

        self.intent_clusters = self.clustering.fit(no_click_queries)
        self.optimizer = OnlineIntentOptimizer(self.intent_clusters)

        print(f"挖掘完成，发现 {len(self.intent_clusters)} 个意图聚类")
        for cluster in self.intent_clusters:
            print(f"\n聚类 {cluster.cluster_id}: {cluster.intent_label}")
            print(f"  关键词: {', '.join(cluster.keywords[:5])}")
            print(f"  优化策略: {cluster.optimization_strategy}")
            print(f"  样例查询: {cluster.sample_queries[0]}")

        return self.intent_clusters

    def online_optimize(self, query: SearchQuery) -> Dict:
        """
        在线阶段：实时优化查询

        Args:
            query: 用户搜索查询

        Returns:
            Dict: 优化结果，包含匹配的意图和策略
        """
        if self.optimizer is None:
            raise ValueError("请先执行离线挖掘阶段")

        matches = self.optimizer.optimize_query(query, top_k=2)

        if not matches:
            return {
                'query': query.query_text,
                'matched': False,
                'message': '未找到匹配的意图聚类，使用默认策略'
            }

        best_match, score = matches[0]

        return {
            'query': query.query_text,
            'matched': True,
            'intent_label': best_match.intent_label,
            'confidence': score,
            'optimization_strategy': best_match.optimization_strategy,
            'tool_sequence': best_match.tool_sequence,
            'similar_queries': best_match.sample_queries[:3],
            'alternative_matches': [
                {'intent': c.intent_label, 'score': s}
                for c, s in matches[1:]
            ] if len(matches) > 1 else []
        }


# ==================== 母婴出海业务场景示例 ====================

def generate_sample_data() -> List[SearchQuery]:
    """生成母婴出海业务示例数据"""

    sample_queries = [
        # 价格敏感型
        SearchQuery('q001', '便宜好用的婴儿纸尿裤', 'user001', '2024-01-01', False),
        SearchQuery('q002', '性价比高的新生儿尿布', 'user002', '2024-01-01', False),
        SearchQuery('q003', '经济实惠的拉拉裤推荐', 'user003', '2024-01-02', False),

        # 安全敏感型
        SearchQuery('q004', '适合敏感肌宝宝的纸尿裤', 'user004', '2024-01-02', False),
        SearchQuery('q005', '无荧光剂婴儿尿布安全吗', 'user005', '2024-01-03', False),
        SearchQuery('q006', '过敏体质宝宝用什么尿不湿', 'user006', '2024-01-03', False),

        # 品质关注型
        SearchQuery('q007', '质量最好的吸奶器品牌', 'user007', '2024-01-04', False),
        SearchQuery('q008', '耐用的电动吸奶器推荐', 'user008', '2024-01-04', False),
        SearchQuery('q009', '口碑好的双边吸奶器', 'user009', '2024-01-05', False),

        # 使用场景型
        SearchQuery('q010', '外出便携的婴儿温奶器', 'user010', '2024-01-05', False),
        SearchQuery('q011', '办公室背奶妈妈吸奶器', 'user011', '2024-01-06', False),
        SearchQuery('q012', '夜间使用静音吸奶器', 'user012', '2024-01-06', False),

        # 材质关注型
        SearchQuery('q013', '纯棉材质的婴儿湿巾', 'user013', '2024-01-07', False),
        SearchQuery('q014', '有机棉宝宝衣服推荐', 'user014', '2024-01-07', False),
        SearchQuery('q015', '竹纤维婴儿浴巾好吗', 'user015', '2024-01-08', False),

        # 尺码困扰型
        SearchQuery('q016', '偏胖宝宝用什么码纸尿裤', 'user016', '2024-01-08', False),
        SearchQuery('q017', '新生儿纸尿裤尺码怎么选', 'user017', '2024-01-09', False),
        SearchQuery('q018', '大月龄宝宝拉拉裤尺码', 'user018', '2024-01-09', False),
    ]

    return sample_queries


def demo():
    """完整演示流程"""
    print("=" * 60)
    print("REVISION 意图挖掘 - 母婴出海电商演示")
    print("反直觉洞察: 无点击不代表无意图")
    print("=" * 60)

    # 1. 准备数据
    print("\n【步骤1】准备历史无点击查询数据...")
    no_click_queries = generate_sample_data()
    print(f"加载了 {len(no_click_queries)} 条无点击查询")

    # 2. 离线挖掘
    print("\n【步骤2】离线意图挖掘...")
    revision = REVISIONIntentMining()
    intent_clusters = revision.offline_mine(no_click_queries)

    # 3. 在线优化演示
    print("\n【步骤3】在线意图优化演示...")

    test_queries = [
        SearchQuery('t001', '敏感肌宝宝用什么纸尿裤好', 'user_test1', '2024-01-10', False),
        SearchQuery('t002', '想找便宜又好用的吸奶器', 'user_test2', '2024-01-10', False),
        SearchQuery('t003', '新生儿尺码怎么选', 'user_test3', '2024-01-10', False),
    ]

    for query in test_queries:
        print(f"\n--- 查询: '{query.query_text}' ---")
        result = revision.online_optimize(query)

        if result['matched']:
            print(f"✓ 识别意图: {result['intent_label']}")
            print(f"  置信度: {result['confidence']:.3f}")
            print(f"  优化策略: {result['optimization_strategy']}")
            print(f"  推荐工具: {', '.join(result['tool_sequence'][:3])}")
        else:
            print(f"✗ {result['message']}")

    # 4. 业务价值总结
    print("\n" + "=" * 60)
    print("业务价值总结")
    print("=" * 60)
    print(f"✓ 发现 {len(intent_clusters)} 类隐含意图")
    print(f"✓ 无点击查询转化率预估提升: 15-25%")
    print(f"✓ 用户满意度预估提升: 通过主动满足隐含需求")
    print(f"✓ 客服咨询量预估下降: 常见问题已前置展示")


if __name__ == '__main__':
    demo()
