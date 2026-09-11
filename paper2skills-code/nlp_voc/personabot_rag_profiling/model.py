#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
PERSONABOT: RAG-based User Persona Generation
PERSONABOT RAG用户画像生成 - Momcozy吸奶器场景

论文来源: PERSONABOT: Bringing Customer Personas to Life with LLMs and RAG
arXiv ID: 2505.17156

应用场景:
- Momcozy吸奶器用户画像生成
- RAG检索真实评论生成结构化画像
- 群体画像生成与营销策略
"""

import numpy as np
import json
from typing import List, Dict, Optional
from dataclasses import dataclass, asdict
from collections import defaultdict


@dataclass
class Review:
    """评论数据结构"""
    review_id: str
    user_id: str
    product_id: str
    text: str
    rating: int
    timestamp: str


@dataclass
class PersonaSchema:
    """用户画像结构"""
    demographics: Dict[str, str]      # 人口统计
    needs: List[str]                   # 需求
    pain_points: List[str]             # 痛点
    preferences: List[str]             # 偏好
    usage_scenarios: List[str]         # 使用场景
    persona_summary: str               # 画像总结


class MockLLM:
    """模拟LLM用于演示（实际使用OpenAI/Claude API）"""

    def generate_persona(self, reviews: List[str], user_info: Dict = None) -> PersonaSchema:
        """
        基于评论生成用户画像

        实际实现应调用LLM API with Few-Shot + CoT prompting
        """
        # 模拟画像生成逻辑（基于规则）
        review_text = ' '.join(reviews)

        # 人群识别
        demographics = {}
        if '上班' in review_text or '公司' in review_text or '背奶' in review_text:
            demographics['role'] = '职场背奶妈妈'
        elif '新手' in review_text or '第一次' in review_text:
            demographics['role'] = '新手妈妈'
        else:
            demographics['role'] = '经验妈妈'

        if '奶多' in review_text or '产量' in review_text:
            demographics['milk_supply'] = '奶量充足型'
        else:
            demographics['milk_supply'] = '标准型'

        # 需求提取
        needs = []
        if '效率' in review_text or '快' in review_text or '时间' in review_text:
            needs.append('高效吸奶')
        if '静音' in review_text or '噪音' in review_text:
            needs.append('静音体验')
        if '便携' in review_text or '轻便' in review_text:
            needs.append('便携性')

        # 痛点提取
        pain_points = []
        if '噪音' in review_text or '吵' in review_text:
            pain_points.append('噪音困扰')
        if '清洗' in review_text and ('麻烦' in review_text or '难' in review_text):
            pain_points.append('清洗不便')
        if '配件' in review_text and ('丢' in review_text or '坏' in review_text):
            pain_points.append('配件管理')

        # 偏好提取
        preferences = []
        if '静音' in review_text:
            preferences.append('静音优先')
        if '吸力' in review_text and ('强' in review_text or '好' in review_text):
            preferences.append('吸力强劲')
        if '价格' in review_text or '性价比' in review_text:
            preferences.append('性价比关注')

        # 场景提取
        scenarios = []
        if '上班' in review_text or '公司' in review_text:
            scenarios.append('工作日公司背奶')
        if '家用' in review_text or '家里' in review_text:
            scenarios.append('居家使用')

        # 生成总结
        summary = f"{demographics.get('role', '妈妈')}，"
        if needs:
            summary += f"关注{'、'.join(needs[:2])}，"
        if pain_points:
            summary += f"受{'、'.join(pain_points[:2])}困扰"

        return PersonaSchema(
            demographics=demographics,
            needs=needs,
            pain_points=pain_points,
            preferences=preferences,
            usage_scenarios=scenarios,
            persona_summary=summary
        )


class ReviewRetriever:
    """
    RAG检索器（简化版）

    实际应使用向量数据库（如FAISS/Chroma）
    """

    def __init__(self, reviews: List[Review]):
        self.reviews = reviews
        self.user_reviews = defaultdict(list)
        for r in reviews:
            self.user_reviews[r.user_id].append(r)

    def retrieve_by_user(self, user_id: str) -> List[Review]:
        """检索特定用户的所有评论"""
        return self.user_reviews.get(user_id, [])

    def retrieve_similar_users(self, user_id: str, top_k: int = 5) -> List[str]:
        """
        检索相似用户（基于评论内容相似度）

        实际应使用向量相似度
        """
        user_reviews = self.retrieve_by_user(user_id)
        if not user_reviews:
            return []

        user_text = ' '.join([r.text for r in user_reviews])

        # 简化的相似度计算
        similarities = []
        for uid, reviews in self.user_reviews.items():
            if uid == user_id:
                continue
            other_text = ' '.join([r.text for r in reviews])
            # Jaccard相似度
            user_words = set(user_text)
            other_words = set(other_text)
            sim = len(user_words & other_words) / len(user_words | other_words) if user_words | other_words else 0
            similarities.append((uid, sim))

        similarities.sort(key=lambda x: x[1], reverse=True)
        return [uid for uid, _ in similarities[:top_k]]

    def retrieve_by_segment(self, segment_keywords: List[str]) -> List[Review]:
        """基于群体关键词检索评论"""
        results = []
        for r in self.reviews:
            if any(kw in r.text for kw in segment_keywords):
                results.append(r)
        return results


class PERSONABOTProfiler:
    """
    PERSONABOT画像生成器

    整合RAG检索和LLM生成，创建结构化用户画像
    """

    def __init__(self, reviews: List[Review]):
        self.retriever = ReviewRetriever(reviews)
        self.llm = MockLLM()

    def generate_individual_persona(self, user_id: str) -> Dict:
        """
        为单个用户生成画像

        Args:
            user_id: 用户ID

        Returns:
            Dict: 结构化用户画像
        """
        # RAG检索用户评论
        user_reviews = self.retriever.retrieve_by_user(user_id)
        if not user_reviews:
            return {'error': f'No reviews found for user {user_id}'}

        review_texts = [r.text for r in user_reviews]

        # 检索相似用户评论作为上下文增强
        similar_users = self.retriever.retrieve_similar_users(user_id, top_k=3)
        similar_reviews = []
        for sim_uid in similar_users:
            sim_revs = self.retriever.retrieve_by_user(sim_uid)
            similar_reviews.extend([r.text for r in sim_revs[:2]])  # 取前2条

        # LLM生成画像
        persona = self.llm.generate_persona(review_texts + similar_reviews)

        # 组装输出
        result = {
            'user_id': user_id,
            'based_on_reviews': len(user_reviews),
            'similar_users_referenced': similar_users,
            'persona': asdict(persona)
        }

        return result

    def generate_segment_persona(self, segment_name: str,
                                  keywords: List[str]) -> Dict:
        """
        为特定群体生成群体画像

        Args:
            segment_name: 群体名称
            keywords: 群体关键词

        Returns:
            Dict: 群体画像
        """
        # 检索群体相关评论
        segment_reviews = self.retriever.retrieve_by_segment(keywords)
        if not segment_reviews:
            return {'error': f'No reviews found for segment {segment_name}'}

        # 抽样代表性评论（实际应使用聚类选择）
        sample_reviews = [r.text for r in segment_reviews[:20]]

        # 分析群体特征
        all_text = ' '.join(sample_reviews)

        # 统计分析
        topic_keywords = {
            '吸力': ['吸力', '强度'],
            '噪音': ['噪音', '静音', '声音'],
            '便携': ['便携', '轻便'],
            '清洗': ['清洗', '清洁'],
            '价格': ['价格', '性价比']
        }

        topic_freq = {}
        for topic, kws in topic_keywords.items():
            count = sum(all_text.count(kw) for kw in kws)
            topic_freq[topic] = count

        # 生成群体画像
        segment_persona = {
            'segment_name': segment_name,
            'segment_size': len(set(r.user_id for r in segment_reviews)),
            'sample_reviews': len(segment_reviews),
            'core_needs': [],
            'pain_points': [],
            'topic_distribution': topic_freq,
            'marketing_insights': {}
        }

        # 识别核心需求（基于频率和情感）
        if topic_freq.get('噪音', 0) > 5:
            segment_persona['core_needs'].append({
                'need': '静音体验',
                'importance': 8.8,
                'evidence': '高频提及噪音困扰'
            })

        if topic_freq.get('便携', 0) > 5:
            segment_persona['core_needs'].append({
                'need': '便携性',
                'importance': 8.5,
                'evidence': '关注携带便利性'
            })

        # 营销策略建议
        segment_persona['marketing_insights'] = {
            'key_message': f'针对{segment_name}，强调{"、".join([n["need"] for n in segment_persona["core_needs"][:2]])}',
            'bundle_suggestion': '主机+便携包+配件套装'
        }

        return segment_persona


# ==================== Momcozy业务场景示例 ====================

def generate_sample_reviews() -> List[Review]:
    """生成Momcozy吸奶器评论示例数据"""

    reviews = [
        # 用户U001 - 职场背奶妈妈
        Review('R001', 'U001', 'S12',
               '吸力很强，10分钟能吸150ml，适合我这种奶多的妈妈', 5, '2024-01-01'),
        Review('R002', 'U001', 'S12',
               '就是噪音有点大，在公司用有点尴尬', 3, '2024-01-05'),
        Review('R003', 'U001', 'S12',
               '配件清洗还算方便，就是小零件容易丢', 4, '2024-01-10'),

        # 用户U002 - 出差妈妈
        Review('R004', 'U002', 'S9',
               '便携性很好，放包里不占地方，出差带着方便', 5, '2024-01-02'),
        Review('R005', 'U002', 'S9',
               '续航能力不错，一天够用', 5, '2024-01-06'),

        # 用户U003 - 新手妈妈
        Review('R006', 'U003', 'M5',
               '新手妈妈很容易上手，操作简单', 5, '2024-01-03'),
        Review('R007', 'U003', 'M5',
               '说明书很详细，第一次用也不慌', 5, '2024-01-07'),

        # 用户U004 - 经验妈妈
        Review('R008', 'U004', 'S12',
               '吸力一般，不如我之前用的那个牌子', 2, '2024-01-04'),
        Review('R009', 'U004', 'S12',
               '电池续航不行，充一次只能用两次', 2, '2024-01-08'),

        # 用户U005 - 另一职场妈妈（与U001相似）
        Review('R010', 'U005', 'S9',
               '上班背奶神器，静音模式很安静', 5, '2024-01-05'),
        Review('R011', 'U005', 'S9',
               '配件质量一般，用了一个月就坏了', 3, '2024-01-09'),
    ]

    return reviews


def demo():
    """完整演示流程"""
    print("=" * 70)
    print("PERSONABOT RAG用户画像生成 - Momcozy吸奶器演示")
    print("=" * 70)

    # 1. 准备数据
    print("\n【步骤1】准备Momcozy评论数据...")
    reviews = generate_sample_reviews()
    print(f"加载了 {len(reviews)} 条评论")

    # 2. 初始化PERSONABOT
    print("\n【步骤2】初始化PERSONABOT...")
    profiler = PERSONABOTProfiler(reviews)

    # 3. 生成个体用户画像
    print("\n【步骤3】生成个体用户画像...")
    for user_id in ['U001', 'U002', 'U003']:
        print(f"\n--- 用户 {user_id} ---")
        persona = profiler.generate_individual_persona(user_id)

        if 'error' not in persona:
            print(f"基于 {persona['based_on_reviews']} 条评论")
            print(f"参考相似用户: {persona['similar_users_referenced']}")
            print(f"画像总结: {persona['persona']['persona_summary']}")
            print(f"核心需求: {persona['persona']['needs']}")
            print(f"痛点: {persona['persona']['pain_points']}")
            print(f"偏好: {persona['persona']['preferences']}")

    # 4. 生成群体画像
    print("\n" + "=" * 70)
    print("【步骤4】生成群体画像 - 职场背奶妈妈")
    print("=" * 70)

    segment = profiler.generate_segment_persona(
        '职场背奶妈妈',
        ['上班', '公司', '背奶', '职场']
    )

    print(f"\n群体名称: {segment['segment_name']}")
    print(f"群体规模: {segment['segment_size']} 用户")
    print(f"样本评论: {segment['sample_reviews']} 条")
    print(f"\n核心需求:")
    for need in segment['core_needs']:
        print(f"  - {need['need']} (重要度: {need['importance']})")
    print(f"\n营销策略:")
    print(f"  核心信息: {segment['marketing_insights']['key_message']}")
    print(f"  推荐套装: {segment['marketing_insights']['bundle_suggestion']}")

    # 5. 业务价值总结
    print("\n" + "=" * 70)
    print("Momcozy业务应用")
    print("=" * 70)
    print("✓ 个体画像: 支持个性化推荐")
    print("✓ 群体画像: 支持精准营销")
    print("✓ RAG增强: 画像可溯源到真实评论")
    print("✓ 营销策略: 从数据驱动到洞察驱动")


if __name__ == '__main__':
    demo()
