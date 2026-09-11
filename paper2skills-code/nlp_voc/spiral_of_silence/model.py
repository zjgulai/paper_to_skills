#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Spiral of Silence: Mining Unspoken Opinions in Customer Reviews
沉默螺旋：发现评论中未被表达的少数派意见

论文来源: Mapping the Spiral of Silence: Surveying Unspoken Opinions
        in Online Communities (CHI 2026)
arXiv ID: 2502.00952

反直觉洞察:
- 72.1%的少数派选择沉默
- 少数派意见被分享的概率只有多数派的一半(27.9% vs 47.2%)
- 主动挖掘沉默用户的不满，发现被好评掩盖的产品问题

应用场景:
- 母婴出海电商：好评如潮的吸奶器，可能掩盖了真实的少数派不满
- 识别被主流声音掩盖的产品改进点
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import Counter, defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sentence_transformers import SentenceTransformer
import warnings
warnings.filterwarnings('ignore')


@dataclass
class Review:
    """用户评论数据结构"""
    review_id: str
    user_id: str
    product_id: str
    text: str
    rating: int
    timestamp: str
    helpful_votes: int = 0
    is_verified: bool = False
    metadata: Dict = None


@dataclass
class OpinionGroup:
    """意见群体"""
    group_id: int
    stance: str  # 'majority' or 'minority'
    representative_texts: List[str]
    keywords: List[str]
    size_estimate: float
    silence_likelihood: float
    concern_areas: List[str]


class OpinionDivergenceDetector:
    """
    意见分歧检测器

    检测评论中的观点分歧，识别潜在的少数派意见
    """

    def __init__(self, n_clusters: int = 5):
        self.n_clusters = n_clusters
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
        self.kmeans = None
        self.reviews = []
        self.clusters = None

    def fit(self, reviews: List[Review]):
        """
        对评论进行聚类，识别不同的观点群体

        Args:
            reviews: 评论列表
        """
        self.reviews = reviews
        texts = [r.text for r in reviews]

        # 提取语义嵌入
        embeddings = self.encoder.encode(texts, show_progress_bar=False)

        # K-means聚类识别观点群体
        self.kmeans = KMeans(n_clusters=self.n_clusters, random_state=42)
        self.clusters = self.kmeans.fit_predict(embeddings)

        return self

    def analyze_divergence(self) -> Dict[int, Dict]:
        """
        分析每个聚类的特征，识别少数派观点

        Returns:
            Dict: 每个聚类的分析结果
        """
        cluster_analysis = {}

        for cluster_id in range(self.n_clusters):
            mask = self.clusters == cluster_id
            cluster_reviews = [self.reviews[i] for i in range(len(self.reviews)) if mask[i]]

            if len(cluster_reviews) < 2:
                continue

            # 统计特征
            avg_rating = np.mean([r.rating for r in cluster_reviews])
            helpfulness = np.mean([r.helpful_votes for r in cluster_reviews])
            size = len(cluster_reviews)
            size_ratio = size / len(self.reviews)

            # 提取关键词
            keywords = self._extract_keywords([r.text for r in cluster_reviews])

            # 判断是多数派还是少数派
            stance = 'minority' if size_ratio < 0.3 else 'majority'

            # 估算沉默概率（基于论文发现：少数派更可能沉默）
            silence_likelihood = self._estimate_silence_likelihood(
                size_ratio, avg_rating, helpfulness
            )

            cluster_analysis[cluster_id] = {
                'size': size,
                'size_ratio': size_ratio,
                'avg_rating': avg_rating,
                'avg_helpfulness': helpfulness,
                'keywords': keywords,
                'stance': stance,
                'silence_likelihood': silence_likelihood,
                'sample_reviews': cluster_reviews[:5],
                'concern_areas': self._identify_concern_areas(cluster_reviews)
            }

        return cluster_analysis

    def _extract_keywords(self, texts: List[str]) -> List[str]:
        """提取关键词"""
        try:
            tfidf = TfidfVectorizer(max_features=15, ngram_range=(1, 2),
                                   stop_words='english')
            tfidf.fit_transform(texts)
            return list(tfidf.get_feature_names_out())
        except:
            return []

    def _estimate_silence_likelihood(self, size_ratio: float,
                                     avg_rating: float,
                                     helpfulness: float) -> float:
        """
        估算该观点群体的沉默概率

        基于论文发现：
        - 少数派更可能沉默 (72.1%沉默率)
        - 低helpfulness的评论更可能来自沉默边缘的用户
        """
        # 基础沉默概率
        base_silence = 0.72 if size_ratio < 0.3 else 0.40

        # 调整因子
        rating_factor = 0.1 if avg_rating >= 4 else -0.1  # 低评分更可能表达
        helpfulness_factor = -0.1 if helpfulness > 10 else 0.1  # 低helpfulness更可能沉默

        return min(0.95, max(0.1, base_silence + rating_factor + helpfulness_factor))

    def _identify_concern_areas(self, reviews: List[Review]) -> List[str]:
        """识别该群体的关注点/抱怨点"""
        concern_keywords = {
            'quality': ['质量', '耐用', '坏', '问题', ' defective', 'broke', 'quality', 'durability'],
            'safety': ['安全', '过敏', '危险', 'safety', 'allergy', 'dangerous', 'hazard'],
            'price': ['贵', '不值', '价格', 'expensive', 'overpriced', 'costly'],
            'usability': ['难用', '不方便', '复杂', 'difficult', 'complicated', 'hard to use'],
            'size': ['尺码', '不合', '大小', 'size', 'fit', 'too small', 'too big'],
            'material': ['材质', '粗糙', '不舒服', 'material', 'uncomfortable', 'rough'],
            'service': ['客服', '售后', '退货', 'service', 'support', 'return']
        }

        all_text = ' '.join([r.text.lower() for r in reviews])
        concerns = []

        for area, keywords in concern_keywords.items():
            if any(kw in all_text for kw in keywords):
                # 计算提及频率
                count = sum(all_text.count(kw) for kw in keywords)
                if count > 0:
                    concerns.append((area, count))

        # 按频率排序
        concerns.sort(key=lambda x: x[1], reverse=True)
        return [c[0] for c in concerns[:3]]


class SilentMinorityMiner:
    """
    沉默少数派挖掘器

    主动发现被主流好评掩盖的真实问题
    """

    def __init__(self):
        self.divergence_detector = OpinionDivergenceDetector()
        self.encoder = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')

    def mine_unspoken_opinions(self, reviews: List[Review]) -> List[OpinionGroup]:
        """
        挖掘未被表达的少数派意见

        Args:
            reviews: 产品评论列表

        Returns:
            List[OpinionGroup]: 发现的意见群体
        """
        # 1. 检测意见分歧
        self.divergence_detector.fit(reviews)
        cluster_analysis = self.divergence_detector.analyze_divergence()

        # 2. 识别高价值少数派意见（可能被沉默的真实问题）
        opinion_groups = []

        for cluster_id, analysis in cluster_analysis.items():
            if analysis['stance'] == 'minority' and analysis['silence_likelihood'] > 0.6:
                # 这可能是被沉默的真实不满
                group = OpinionGroup(
                    group_id=cluster_id,
                    stance='minority',
                    representative_texts=[r.text[:200] for r in analysis['sample_reviews'][:3]],
                    keywords=analysis['keywords'][:8],
                    size_estimate=analysis['size_ratio'],
                    silence_likelihood=analysis['silence_likelihood'],
                    concern_areas=analysis['concern_areas']
                )
                opinion_groups.append(group)

        # 3. 按沉默概率排序（最可能被沉默的意见优先）
        opinion_groups.sort(key=lambda x: x.silence_likelihood, reverse=True)

        return opinion_groups

    def generate_insights(self, opinion_groups: List[OpinionGroup],
                         product_name: str) -> Dict:
        """
        生成业务洞察报告

        Args:
            opinion_groups: 发现的意见群体
            product_name: 产品名称

        Returns:
            Dict: 洞察报告
        """
        if not opinion_groups:
            return {
                'product': product_name,
                'findings': '未发现显著的沉默少数派意见',
                'recommendations': []
            }

        insights = {
            'product': product_name,
            'total_minority_groups': len(opinion_groups),
            'high_silence_risk_groups': sum(1 for g in opinion_groups if g.silence_likelihood > 0.7),
            'key_concerns': self._aggregate_concerns(opinion_groups),
            'findings': [],
            'recommendations': []
        }

        for group in opinion_groups:
            finding = {
                'group_id': group.group_id,
                'estimated_size': f"{group.size_estimate*100:.1f}%",
                'silence_probability': f"{group.silence_likelihood*100:.1f}%",
                'concern_areas': group.concern_areas,
                'keywords': group.keywords[:5],
                'sample_complaints': group.representative_texts[:2]
            }
            insights['findings'].append(finding)

        # 生成改进建议
        insights['recommendations'] = self._generate_recommendations(opinion_groups)

        return insights

    def _aggregate_concerns(self, opinion_groups: List[OpinionGroup]) -> Dict[str, int]:
        """聚合所有少数派的关注点"""
        all_concerns = []
        for group in opinion_groups:
            all_concerns.extend(group.concern_areas)

        return dict(Counter(all_concerns).most_common())

    def _generate_recommendations(self, opinion_groups: List[OpinionGroup]) -> List[Dict]:
        """基于发现的问题生成改进建议"""
        recommendations = []

        concern_recommendations = {
            'quality': {
                'issue': '质量耐用性问题',
                'action': '加强质检流程，提供延保服务',
                'priority': '高'
            },
            'safety': {
                'issue': '安全/过敏隐患',
                'action': '提供成分透明化，增加安全认证',
                'priority': '高'
            },
            'price': {
                'issue': '价格敏感度高',
                'action': '推出不同价位产品线，增加试用装',
                'priority': '中'
            },
            'usability': {
                'issue': '使用体验不佳',
                'action': '优化产品设计，提供更详细使用教程',
                'priority': '高'
            },
            'size': {
                'issue': '尺码选择困扰',
                'action': '完善尺码指南，提供试穿/退换政策',
                'priority': '中'
            },
            'material': {
                'issue': '材质不适',
                'action': '提供更多材质选项，强调亲肤材质',
                'priority': '中'
            },
            'service': {
                'issue': '客服/售后问题',
                'action': '优化客服响应，简化退换流程',
                'priority': '高'
            }
        }

        seen_concerns = set()
        for group in opinion_groups:
            for concern in group.concern_areas:
                if concern in concern_recommendations and concern not in seen_concerns:
                    recommendations.append(concern_recommendations[concern])
                    seen_concerns.add(concern)

        return recommendations


# ==================== 母婴出海业务场景示例 ====================

def generate_sample_reviews() -> List[Review]:
    """生成母婴产品评论示例数据"""

    # 多数派好评（70%）- 高评分、高频出现
    majority_reviews = [
        Review(f'r{i:03d}', f'user{i:03d}', 'p001',
               '很好用，宝宝穿着很舒服，吸水性强，会一直回购！', 5,
               '2024-01-01', 45, True)
        for i in range(35)
    ] + [
        Review(f'r{i:03d}', f'user{i:03d}', 'p001',
               '质量特别好，不红屁股，强烈推荐！', 5,
               '2024-01-02', 38, True)
        for i in range(35, 70)
    ]

    # 少数派真实不满（30%但分散）- 被沉默的意见
    minority_reviews = [
        # 尺码问题（被沉默）
        Review('r101', 'user101', 'p001',
               '尺码偏小，宝宝用着紧，但已经拆了包装退不了', 3,
               '2024-01-03', 2, True),
        Review('r102', 'user102', 'p001',
               '建议买大一码，尺码不太准', 3,
               '2024-01-03', 1, False),

        # 过敏问题（重要但被淹没）
        Review('r103', 'user103', 'p001',
               '我家敏感肌宝宝用了有点红，可能不适合所有宝宝', 2,
               '2024-01-04', 0, True),
        Review('r104', 'user104', 'p001',
               '成分表里有香精，敏感肌慎买', 2,
               '2024-01-04', 1, False),

        # 材质问题（少数反馈）
        Review('r105', 'user105', 'p001',
               '材质没有想象中柔软，外层有点硬', 3,
               '2024-01-05', 2, True),

        # 价格问题（低helpfulness）
        Review('r106', 'user106', 'p001',
               '性价比一般，同价位有更好选择', 3,
               '2024-01-05', 0, False),
        Review('r107', 'user107', 'p001',
               '活动价买的还可以，原价有点贵', 3,
               '2024-01-06', 1, True),

        # 使用体验问题
        Review('r108', 'user108', 'p001',
               '魔术贴设计不太合理，粘不牢', 2,
               '2024-01-06', 3, True),
        Review('r109', 'user109', 'p001',
               '腰封部分有点勒，胖宝宝可能不适合', 2,
               '2024-01-07', 2, False),

        # 质量不一致问题
        Review('r110', 'user110', 'p001',
               '这次买的和上次不一样，感觉变薄了', 3,
               '2024-01-07', 4, True),
    ]

    return majority_reviews + minority_reviews


def demo():
    """完整演示流程"""
    print("=" * 70)
    print("沉默少数派挖掘 - 母婴出海电商演示")
    print("反直觉洞察: 72.1%的少数派选择沉默")
    print("=" * 70)

    # 1. 准备数据
    print("\n【步骤1】准备产品评论数据...")
    reviews = generate_sample_reviews()

    rating_dist = Counter([r.rating for r in reviews])
    print(f"评论总数: {len(reviews)}")
    print(f"评分分布: {dict(rating_dist)}")
    print(f"好评率(4-5星): {sum(rating_dist[i] for i in [4,5])/len(reviews)*100:.1f}%")

    # 2. 挖掘沉默少数派
    print("\n【步骤2】挖掘沉默少数派意见...")
    miner = SilentMinorityMiner()
    opinion_groups = miner.mine_unspoken_opinions(reviews)

    # 3. 生成洞察报告
    print("\n【步骤3】生成业务洞察报告...")
    insights = miner.generate_insights(opinion_groups, "XX品牌婴儿纸尿裤")

    print(f"\n发现 {insights['total_minority_groups']} 个少数派意见群体")
    print(f"高风险沉默群体: {insights['high_silence_risk_groups']} 个")

    print("\n关键关注点分布:")
    for concern, count in insights['key_concerns'].items():
        print(f"  - {concern}: {count}个群体提及")

    print("\n详细发现:")
    for finding in insights['findings']:
        print(f"\n  群体 {finding['group_id']}:")
        print(f"    预估占比: {finding['estimated_size']}")
        print(f"    沉默概率: {finding['silence_probability']}")
        print(f"    关注点: {', '.join(finding['concern_areas'])}")
        print(f"    关键词: {', '.join(finding['keywords'])}")
        print(f"    典型反馈: {finding['sample_complaints'][0][:80]}...")

    # 4. 改进建议
    print("\n【步骤4】产品改进建议:")
    for i, rec in enumerate(insights['recommendations'], 1):
        print(f"\n  建议{i} [{rec['priority']}优先级]:")
        print(f"    问题: {rec['issue']}")
        print(f"    行动: {rec['action']}")

    # 5. 业务价值总结
    print("\n" + "=" * 70)
    print("业务价值总结")
    print("=" * 70)
    print("✓ 发现被好评掩盖的真实问题")
    print("✓ 识别潜在的用户流失风险（尺码、过敏等问题）")
    print("✓ 为产品迭代提供数据支撑")
    print("✓ 降低差评爆发风险（提前解决沉默用户的不满）")


if __name__ == '__main__':
    demo()
