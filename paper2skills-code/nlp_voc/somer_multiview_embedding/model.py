#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SoMeR: Multi-View User Representation Learning
SoMeR多视角用户表示学习 - Momcozy吸奶器场景

论文来源: SoMeR: A Multi-View Social Media User Representation Learning Framework
arXiv ID: 2405.05275

应用场景:
- Momcozy吸奶器多维度用户画像
- 融合搜索/评论/行为/社交数据
- 相似用户发现与人群聚类
"""

import numpy as np
import pandas as pd
from typing import List, Dict, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict
from sklearn.cluster import DBSCAN
from sklearn.metrics.pairwise import cosine_similarity


@dataclass
class UserBehavior:
    """用户行为数据"""
    user_id: str
    view_type: str  # 'search', 'review', 'purchase', 'social'
    timestamp: str
    content: str
    value: float = 1.0


@dataclass
class UserProfile:
    """用户档案数据"""
    user_id: str
    age_group: Optional[str] = None
    role: Optional[str] = None  # '职场妈妈', '全职妈妈', etc.
    baby_age: Optional[int] = None
    location: Optional[str] = None


class TripletEncoder:
    """
    Triplet编码器

    将(timestamp, feature, value)编码为嵌入向量
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        # 简化的编码（实际应使用FFN和Lookup table）
        np.random.seed(42)
        self.feature_embeddings = {}

    def encode(self, timestamp: str, feature: str, value: float) -> np.ndarray:
        """编码单个triplet"""
        # 时间嵌入（简化为hash）
        time_emb = np.random.randn(self.embedding_dim) * 0.1

        # 特征嵌入
        if feature not in self.feature_embeddings:
            self.feature_embeddings[feature] = np.random.randn(self.embedding_dim)
        feature_emb = self.feature_embeddings[feature]

        # 值嵌入
        value_emb = np.ones(self.embedding_dim) * value * 0.1

        # 求和融合
        return time_emb + feature_emb + value_emb


class TemporalTransformer:
    """
    时间序列Transformer编码器（简化版）

    实际应使用PyTorch/TensorFlow实现
    """

    def __init__(self, input_dim: int = 64, num_layers: int = 2):
        self.input_dim = input_dim
        self.num_layers = num_layers

    def encode(self, triplet_embeddings: List[np.ndarray]) -> np.ndarray:
        """
        编码triplet序列

        简化为平均池化（实际应用Self-Attention）
        """
        if not triplet_embeddings:
            return np.zeros(self.input_dim)

        # 平均池化（简化）
        return np.mean(triplet_embeddings, axis=0)


class MultiViewEncoder:
    """
    多视角编码器

    融合四视角数据：时间活动、文本内容、个人资料、网络互动
    """

    def __init__(self, embedding_dim: int = 64):
        self.embedding_dim = embedding_dim
        self.triplet_encoder = TripletEncoder(embedding_dim)
        self.temporal_transformer = TemporalTransformer(embedding_dim)

    def encode_temporal(self, behaviors: List[UserBehavior]) -> np.ndarray:
        """编码时间活动视角"""
        triplets = []
        for b in behaviors:
            emb = self.triplet_encoder.encode(b.timestamp, b.view_type, b.value)
            triplets.append(emb)

        return self.temporal_transformer.encode(triplets)

    def encode_textual(self, reviews: List[str]) -> np.ndarray:
        """编码文本内容视角（简化）"""
        # 实际应使用Sentence-BERT
        if not reviews:
            return np.zeros(self.embedding_dim)

        # 简化为词袋编码
        all_text = ' '.join(reviews)
        words = set(all_text.split())

        # Hash编码
        emb = np.zeros(self.embedding_dim)
        for word in words:
            idx = hash(word) % self.embedding_dim
            emb[idx] += 1

        # 归一化
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def encode_profile(self, profile: UserProfile) -> np.ndarray:
        """编码个人资料视角"""
        emb = np.zeros(self.embedding_dim)

        # 角色编码
        role_map = {
            '职场妈妈': 0,
            '全职妈妈': 1,
            '新手妈妈': 2,
            '经验妈妈': 3
        }
        if profile.role in role_map:
            emb[role_map[profile.role]] = 1.0

        # 宝宝月龄编码（离散化）
        if profile.baby_age:
            baby_idx = min(profile.baby_age // 3, 10)  # 0-3月, 3-6月, etc.
            if 20 + baby_idx < self.embedding_dim:
                emb[20 + baby_idx] = 1.0

        return emb

    def encode_network(self, interactions: List[str]) -> np.ndarray:
        """编码网络互动视角（简化）"""
        # 实际应使用Graph Neural Network
        emb = np.zeros(self.embedding_dim)

        # 统计互动类型
        for interaction in interactions:
            idx = hash(interaction) % self.embedding_dim
            emb[idx] += 1

        # 归一化
        norm = np.linalg.norm(emb)
        return emb / norm if norm > 0 else emb

    def fuse_views(self, embeddings: List[np.ndarray]) -> np.ndarray:
        """
        融合多视角嵌入

        使用加权平均（实际应用Attention机制）
        """
        if not embeddings:
            return np.zeros(self.embedding_dim)

        # 等权重融合
        fused = np.mean(embeddings, axis=0)

        # 归一化
        norm = np.linalg.norm(fused)
        return fused / norm if norm > 0 else fused


class SoMeRProfiler:
    """
    SoMeR多视角用户画像生成器
    """

    def __init__(self, embedding_dim: int = 64):
        self.encoder = MultiViewEncoder(embedding_dim)
        self.user_embeddings = {}
        self.embedding_dim = embedding_dim

    def encode_user(self,
                   user_id: str,
                   behaviors: List[UserBehavior],
                   reviews: List[str],
                   profile: UserProfile,
                   interactions: List[str]) -> np.ndarray:
        """
        为单个用户生成多视角嵌入

        Args:
            user_id: 用户ID
            behaviors: 行为序列
            reviews: 评论文本列表
            profile: 个人资料
            interactions: 社交互动

        Returns:
            np.ndarray: 用户嵌入向量
        """
        # 四视角编码
        temporal_emb = self.encoder.encode_temporal(behaviors)
        textual_emb = self.encoder.encode_textual(reviews)
        profile_emb = self.encoder.encode_profile(profile)
        network_emb = self.encoder.encode_network(interactions)

        # 融合
        fused_emb = self.encoder.fuse_views([
            temporal_emb, textual_emb, profile_emb, network_emb
        ])

        self.user_embeddings[user_id] = fused_emb
        return fused_emb

    def find_similar_users(self, user_id: str, top_k: int = 5) -> List[Tuple[str, float]]:
        """
        查找相似用户

        Args:
            user_id: 查询用户ID
            top_k: 返回数量

        Returns:
            List[Tuple[str, float]]: (用户ID, 相似度)
        """
        if user_id not in self.user_embeddings:
            return []

        query_emb = self.user_embeddings[user_id].reshape(1, -1)

        similarities = []
        for uid, emb in self.user_embeddings.items():
            if uid == user_id:
                continue
            sim = cosine_similarity(query_emb, emb.reshape(1, -1))[0][0]
            similarities.append((uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:top_k]

    def cluster_users(self, eps: float = 0.3, min_samples: int = 2) -> Dict[int, List[str]]:
        """
        用户聚类

        Args:
            eps: DBSCAN邻域半径
            min_samples: 最小样本数

        Returns:
            Dict: 聚类结果
        """
        if not self.user_embeddings:
            return {}

        user_ids = list(self.user_embeddings.keys())
        embeddings = np.array([self.user_embeddings[uid] for uid in user_ids])

        # DBSCAN聚类
        clustering = DBSCAN(eps=eps, min_samples=min_samples, metric='cosine')
        labels = clustering.fit_predict(embeddings)

        # 组织结果
        clusters = defaultdict(list)
        for uid, label in zip(user_ids, labels):
            clusters[label].append(uid)

        return dict(clusters)

    def get_persona_attributes(self, user_id: str) -> Dict:
        """
        获取可解释的画像属性

        Args:
            user_id: 用户ID

        Returns:
            Dict: 画像属性
        """
        if user_id not in self.user_embeddings:
            return {}

        emb = self.user_embeddings[user_id]

        # 基于嵌入向量推断属性（简化）
        attributes = {
            'user_id': user_id,
            'embedding_norm': float(np.linalg.norm(emb)),
            'top_dimensions': np.argsort(emb)[-5:].tolist(),
            'interpretation': {}
        }

        # 简化的维度解释（实际应训练解释器）
        if emb[0] > 0.3:
            attributes['interpretation']['人群类型'] = '职场背奶型'
        elif emb[1] > 0.3:
            attributes['interpretation']['人群类型'] = '全职新手型'
        else:
            attributes['interpretation']['人群类型'] = '经验妈妈型'

        return attributes


# ==================== Momcozy业务场景示例 ====================

def generate_sample_data():
    """生成Momcozy业务示例数据"""

    # 用户行为数据
    behaviors = [
        # U001 - 职场背奶妈妈
        UserBehavior('U001', 'search', '2024-01-01', '吸奶器推荐', 1.0),
        UserBehavior('U001', 'search', '2024-01-02', '静音吸奶器', 1.0),
        UserBehavior('U001', 'purchase', '2024-01-03', 'S12', 1.0),
        UserBehavior('U001', 'review', '2024-01-10', '吸力很强但噪音大', 4.0),

        # U002 - 出差妈妈
        UserBehavior('U002', 'search', '2024-01-01', '便携吸奶器', 1.0),
        UserBehavior('U002', 'purchase', '2024-01-02', 'S9', 1.0),
        UserBehavior('U002', 'review', '2024-01-08', '便携性很好', 5.0),

        # U003 - 新手妈妈
        UserBehavior('U003', 'search', '2024-01-01', '新手吸奶器推荐', 1.0),
        UserBehavior('U003', 'search', '2024-01-02', '吸奶器使用方法', 1.0),
        UserBehavior('U003', 'purchase', '2024-01-03', 'M5', 1.0),
        UserBehavior('U003', 'review', '2024-01-10', '很容易上手', 5.0),
    ]

    # 用户评论
    reviews = {
        'U001': ['吸力很强，10分钟能吸150ml', '就是噪音有点大，在公司用有点尴尬'],
        'U002': ['便携性很好，放包里不占地方', '续航能力不错'],
        'U003': ['新手妈妈很容易上手', '操作简单'],
    }

    # 用户档案
    profiles = {
        'U001': UserProfile('U001', '28-35', '职场妈妈', 6),
        'U002': UserProfile('U002', '28-35', '职场妈妈', 4),
        'U003': UserProfile('U003', '25-30', '新手妈妈', 2),
    }

    # 社交互动
    interactions = {
        'U001': ['关注背奶群', '点赞便携装备'],
        'U002': ['关注出差妈妈群', '分享旅行装备'],
        'U003': ['关注新手妈妈群', '点赞育儿知识'],
    }

    return behaviors, reviews, profiles, interactions


def demo():
    """完整演示流程"""
    print("=" * 70)
    print("SoMeR多视角用户表示 - Momcozy吸奶器演示")
    print("=" * 70)

    # 1. 准备数据
    print("\n【步骤1】准备多视角数据...")
    behaviors, reviews, profiles, interactions = generate_sample_data()
    print("数据准备完成")

    # 2. 初始化SoMeR
    print("\n【步骤2】初始化SoMeR...")
    somer = SoMeRProfiler(embedding_dim=64)

    # 3. 编码用户
    print("\n【步骤3】多视角用户编码...")

    # 组织用户数据
    user_ids = ['U001', 'U002', 'U003']
    for uid in user_ids:
        user_behaviors = [b for b in behaviors if b.user_id == uid]
        user_reviews = reviews.get(uid, [])
        user_profile = profiles.get(uid, UserProfile(uid))
        user_interactions = interactions.get(uid, [])

        embedding = somer.encode_user(
            uid, user_behaviors, user_reviews, user_profile, user_interactions
        )
        print(f"  用户 {uid}: 嵌入维度 {len(embedding)}, L2范数 {np.linalg.norm(embedding):.3f}")

    # 4. 相似用户发现
    print("\n【步骤4】相似用户发现...")
    for uid in user_ids:
        similar = somer.find_similar_users(uid, top_k=2)
        print(f"\n  与 {uid} 最相似的用户:")
        for sim_uid, sim_score in similar:
            print(f"    - {sim_uid}: 相似度 {sim_score:.3f}")

    # 5. 用户聚类
    print("\n【步骤5】用户聚类...")
    clusters = somer.cluster_users(eps=0.5, min_samples=1)

    print(f"发现 {len(clusters)} 个用户群体:")
    for cluster_id, user_list in clusters.items():
        print(f"  群体 {cluster_id}: {user_list}")

    # 6. 画像属性
    print("\n【步骤6】可解释画像属性...")
    for uid in user_ids:
        attrs = somer.get_persona_attributes(uid)
        print(f"\n  用户 {uid}:")
        print(f"    画像解读: {attrs.get('interpretation', {})}")

    # 7. 业务价值总结
    print("\n" + "=" * 70)
    print("Momcozy业务价值")
    print("=" * 70)
    print("✓ 多视角融合: 综合搜索+评论+档案+社交")
    print("✓ 相似用户发现: 支持精准推荐")
    print("✓ 用户聚类: 发现自然群体")
    print("✓ 360°画像: 比单一维度更准确")


if __name__ == '__main__':
    demo()
