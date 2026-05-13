"""Unified VOC Label Extraction Engine

基于 376 标签种子 + 55 原子画像标签 + AIPL 7 节点的统一萃取引擎。
单条 VOC → 6 维度完整标签输出。

核心设计:
- 统一跑一次全量标签（带品线过滤）
- 多标签共存（一条 VOC 可同时打多个标签）
- 标签预定义情感 + ABSA 动态校准
- 55 原子标签 → 3 业务画像推导

Usage:
    from unified_label_extraction import (
        TagSeedDictionary, VOCLabelExtractor,
        UnifiedLabelingPipeline, DashboardGenerator,
    )

    # 1. 加载标签种子
    tag_dict = TagSeedDictionary.from_csv("tag_seeds.csv")

    # 2. 创建萃取器
    extractor = VOCLabelExtractor(tag_dict=tag_dict)

    # 3. 单条萃取
    result = extractor.extract(voc_record)

    # 4. 批量流水线
    pipeline = UnifiedLabelingPipeline(tag_dict=tag_dict)
    results = pipeline.process(voc_records)

    # 5. 生成看板
    dashboard = DashboardGenerator().build(results)
"""

from __future__ import annotations

import csv
import json
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
from typing import Any, Optional

import numpy as np


# ---------------------------------------------------------------------------
# 1. 数据模型
# ---------------------------------------------------------------------------

@dataclass
class VOCRecord:
    """单条 VOC 原始输入"""

    review_id: str
    text: str
    source_type: str  # return_note / ticket / review / trustpilot
    platform: str  # amazon / dtc / offline / tiktok
    spu_code: str
    product_line: str  # VOC 品线（如 breast_pump）
    category: str  # VOC 品类（如 wearable_pump）
    rating: Optional[float] = None  # 1-5 星
    order_id: Optional[str] = None
    user_id: Optional[str] = None
    timestamp: Optional[str] = None

    # 现有 classification 标签（如已有）
    classification_tag: Optional[str] = None
    cn_level1: Optional[str] = None
    cn_level2: Optional[str] = None
    cn_level3: Optional[str] = None


@dataclass
class TagSeed:
    """单条标签种子（来自 376 标签字典）"""

    tag_id: str
    tag_en: str  # 英文标签名
    tag_cn: str  # 中文标签名
    aipl_node: str  # A/I/P1/P2/L1/L2/L3
    theme: str  # 标签主题（如产品核心性能）
    sentiment_preset: str  # positive / negative / neutral
    keywords: list[str] = field(default_factory=list)  # 英文关键词
    consumer_keywords: list[str] = field(default_factory=list)  # 消费者习惯表达
    persona: str = ""  # 适用用户画像
    atomic_metric: str = ""  # 对应原子指标
    metric_direction: str = ""  # positive / negative
    nps_contribution: str = ""  # Promoter驱动 / Detractor驱动 / 中性
    story_line: str = ""  # 故事线关联
    strategy_pack: str = ""  # 策略包
    owner_dept: str = ""  # 主责部门
    priority: str = ""  # P0/P1/P2/P3
    applicable_line: list[str] = field(default_factory=list)  # 适用品线
    applicable_source: list[str] = field(default_factory=list)  # 适用数据源


@dataclass
class AIPLTagMatch:
    """AIPL 标签匹配结果"""

    tag_id: str
    tag_en: str
    tag_cn: str
    theme: str
    aipl_node: str
    sentiment_preset: str
    sentiment_calibrated: float  # -1.0 ~ +1.0
    confidence: float  # 匹配置信度 0-1


@dataclass
class PersonaTagMatch:
    """画像标签匹配结果"""

    tag_name: str
    dimension: str  # WHO/WHY/WHAT/WHEN/HOW/EMOTION
    sub_dimension: str
    confidence: float


@dataclass
class SentimentCalibration:
    """情感校准结果"""

    polarity: float  # -1.0 ~ +1.0
    intensity: float  # -5.0 ~ +5.0
    calibration_flag: str  # preset / calibrated / conflict
    aspect_sentiments: dict[str, float] = field(default_factory=dict)
    conflict_reason: str = ""


@dataclass
class VOCLabelExtraction:
    """单条 VOC 的完整标签萃取结果"""

    # 基础信息
    review_id: str
    source_type: str
    platform: str
    spu_code: str
    product_line: str
    category: str
    rating: Optional[float]

    # 维度 1: AIPL 旅程标签
    aipl_stage: str  # 主阶段
    aipl_tags: list[AIPLTagMatch] = field(default_factory=list)

    # 维度 2: 问题类型标签（现有 classification）
    classification_tag: str = ""
    cn_level1: str = ""
    cn_level2: str = ""
    cn_level3: str = ""

    # 维度 3: 画像标签
    persona_atomic: list[str] = field(default_factory=list)
    persona_derived: str = ""  # 社群黏着型/系统规划型/品质探索型/未分类
    persona_dimensions: dict[str, list[str]] = field(default_factory=dict)  # WHO/WHY/WHAT/WHEN/HOW/EMOTION

    # 维度 4: 情感维度
    sentiment_polarity: float = 0.0
    sentiment_intensity: float = 0.0
    sentiment_calibration: str = ""
    aspect_sentiments: dict[str, float] = field(default_factory=dict)

    # 维度 5: 品牌维度
    brand_mentions: list[str] = field(default_factory=list)
    brand_comparison: bool = False

    # 维度 6: 质量评分
    quality_score: float = 0.0
    is_suspicious: bool = False

    # 业务闭环字段
    proxy_nps_contribution: str = ""  # Promoter驱动 / Detractor驱动 / 中性
    metric_direction: str = ""  # positive / negative
    story_line: str = ""
    strategy_pack: str = ""
    owner_dept: str = ""
    priority: str = ""

    def to_dict(self) -> dict[str, Any]:
        return {
            "review_id": self.review_id,
            "source_type": self.source_type,
            "platform": self.platform,
            "product_line": self.product_line,
            "category": self.category,
            "rating": self.rating,
            "aipl_stage": self.aipl_stage,
            "aipl_tags": [
                {
                    "tag_id": t.tag_id,
                    "tag_en": t.tag_en,
                    "tag_cn": t.tag_cn,
                    "theme": t.theme,
                    "aipl_node": t.aipl_node,
                    "sentiment_preset": t.sentiment_preset,
                    "sentiment_calibrated": round(t.sentiment_calibrated, 2),
                    "confidence": round(t.confidence, 2),
                }
                for t in self.aipl_tags
            ],
            "classification": {
                "tag": self.classification_tag,
                "l1": self.cn_level1,
                "l2": self.cn_level2,
                "l3": self.cn_level3,
            },
            "persona": {
                "atomic": self.persona_atomic,
                "derived": self.persona_derived,
                "dimensions": self.persona_dimensions,
            },
            "sentiment": {
                "polarity": round(self.sentiment_polarity, 2),
                "intensity": round(self.sentiment_intensity, 1),
                "calibration": self.sentiment_calibration,
                "aspect_sentiments": {
                    k: round(v, 2) for k, v in self.aspect_sentiments.items()
                },
            },
            "brand": {
                "mentions": self.brand_mentions,
                "comparison": self.brand_comparison,
            },
            "quality": {
                "score": round(self.quality_score, 1),
                "is_suspicious": self.is_suspicious,
            },
            "business": {
                "proxy_nps": self.proxy_nps_contribution,
                "metric_direction": self.metric_direction,
                "story_line": self.story_line,
                "strategy_pack": self.strategy_pack,
                "owner_dept": self.owner_dept,
                "priority": self.priority,
            },
        }


# ---------------------------------------------------------------------------
# 2. 统一标签字典
# ---------------------------------------------------------------------------

class TagSeedDictionary:
    """统一标签字典 — 管理全部 376 条标签种子

    支持:
    - 从 CSV/Excel 加载标签种子
    - 按品线过滤
    - 按 AIPL 节点分组查询
    - 按主题分组查询
    """

    def __init__(self) -> None:
        self._tags: dict[str, TagSeed] = {}  # tag_id -> TagSeed
        self._by_aipl: dict[str, list[TagSeed]] = defaultdict(list)
        self._by_theme: dict[str, list[TagSeed]] = defaultdict(list)
        self._by_line: dict[str, list[TagSeed]] = defaultdict(list)
        self._by_keyword: dict[str, list[TagSeed]] = defaultdict(list)  # keyword -> tags

    # ── 加载 ────────────────────────────────────────────────────

    def add(self, tag: TagSeed) -> None:
        """添加单条标签种子"""
        self._tags[tag.tag_id] = tag
        self._by_aipl[tag.aipl_node].append(tag)
        self._by_theme[tag.theme].append(tag)
        for line in tag.applicable_line:
            self._by_line[line].append(tag)
        for kw in tag.keywords:
            self._by_keyword[kw.lower()].append(tag)

    @classmethod
    def from_csv(cls, path: str) -> TagSeedDictionary:
        """从 CSV 加载标签种子"""
        dictionary = cls()
        with open(path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                tag = TagSeed(
                    tag_id=row.get("tag_id", ""),
                    tag_en=row.get("tag_en", row.get("VOC标签（英文）", "")),
                    tag_cn=row.get("tag_cn", row.get("VOC标签（中文）", "")),
                    aipl_node=row.get("aipl_node", row.get("AIPL节点", "")),
                    theme=row.get("theme", row.get("标签主题", "")),
                    sentiment_preset=row.get("sentiment_preset", row.get("情感极性", "")),
                    keywords=_parse_list(row.get("keywords", row.get("英文关键词/典型表达", ""))),
                    consumer_keywords=_parse_list(row.get("consumer_keywords", row.get("消费者习惯关键词/原话短语", ""))),
                    persona=row.get("persona", row.get("适用用户画像", "")),
                    atomic_metric=row.get("atomic_metric", row.get("对应原子指标", "")),
                    metric_direction=row.get("metric_direction", row.get("MetricDirection", "")),
                    nps_contribution=row.get("nps_contribution", row.get("Proxy NPS贡献", "")),
                    story_line=row.get("story_line", row.get("故事线关联", "")),
                    strategy_pack=row.get("strategy_pack", row.get("策略包", "")),
                    owner_dept=row.get("owner_dept", row.get("主责部门", "")),
                    priority=row.get("priority", row.get("默认优先级", "")),
                    applicable_line=_parse_list(row.get("applicable_line", row.get("适用产品品线", ""))),
                    applicable_source=_parse_list(row.get("applicable_source", row.get("适用VOC载体", ""))),
                )
                dictionary.add(tag)
        return dictionary

    @classmethod
    def from_xlsx(cls, path: str, sheet_names: Optional[list[str]] = None) -> TagSeedDictionary:
        """从 Excel 多 sheet 加载标签种子

        Args:
            path: xlsx 文件路径
            sheet_names: 指定要加载的 sheet 名称列表（None=自动加载所有数据 sheet）
        """
        import pandas as pd

        dictionary = cls()
        xl = pd.ExcelFile(path)

        if sheet_names is None:
            # 自动排除字段说明/元数据 sheet
            sheet_names = [s for s in xl.sheet_names if "字段说明" not in s]

        total_added = 0
        skipped_ids: set[str] = set()

        for sheet in sheet_names:
            df = pd.read_excel(xl, sheet_name=sheet)
            for _, row in df.iterrows():
                tag_id = str(row.get("标签ID", "")).strip()
                if not tag_id or tag_id in dictionary._tags:
                    # 跳过空 ID 或重复 ID（通用标签可能在多个 sheet 出现）
                    if tag_id:
                        skipped_ids.add(tag_id)
                    continue

                raw_kws = _parse_list(row.get("英文关键词/典型表达", ""))
                raw_consumer = _parse_list(row.get("消费者习惯关键词/原话短语", ""))
                tag = TagSeed(
                    tag_id=tag_id,
                    tag_en=str(row.get("VOC标签（英文）", "")).strip(),
                    tag_cn=str(row.get("VOC标签（中文）", "")).strip(),
                    aipl_node=str(row.get("AIPL节点", "")).strip(),
                    theme=str(row.get("标签主题", "")).strip(),
                    sentiment_preset=_map_sentiment_preset(row.get("情感极性", "")),
                    keywords=_filter_keywords(raw_kws),
                    consumer_keywords=_filter_keywords(raw_consumer),
                    persona=str(row.get("适用用户画像", "")),
                    atomic_metric=str(row.get("对应原子指标", "")),
                    metric_direction=str(row.get("MetricDirection", "")),
                    nps_contribution=str(row.get("Proxy NPS贡献", "")),
                    story_line=str(row.get("故事线关联", "")),
                    strategy_pack=str(row.get("策略包", "")),
                    owner_dept=str(row.get("主责部门", "")),
                    priority=str(row.get("默认优先级", "")),
                    applicable_line=_map_applicable_line(row.get("适用产品品线", "")),
                    applicable_source=_parse_list(row.get("适用VOC载体", "")),
                )
                dictionary.add(tag)
                total_added += 1

        print(f"从 {len(sheet_names)} 个 sheet 加载 {total_added} 条标签，跳过重复 {len(skipped_ids)} 条")
        return dictionary

    # ── 查询 ────────────────────────────────────────────────────

    def get(self, tag_id: str) -> Optional[TagSeed]:
        return self._tags.get(tag_id)

    def get_by_aipl(self, node: str) -> list[TagSeed]:
        return self._by_aipl.get(node, [])

    def get_by_theme(self, theme: str) -> list[TagSeed]:
        return self._by_theme.get(theme, [])

    def get_by_line(self, line: str) -> list[TagSeed]:
        return self._by_line.get(line, [])

    def get_all(self) -> list[TagSeed]:
        return list(self._tags.values())

    def filter_by_line(self, line: str) -> list[TagSeed]:
        """按品线过滤标签（含通用标签）"""
        # 通用标签（无品线限制）+ 该品线专属标签
        return [
            tag for tag in self._tags.values()
            if not tag.applicable_line or line in tag.applicable_line
        ]

    def filter_by_source(self, source_type: str) -> list[TagSeed]:
        """按数据源类型过滤标签

        映射关系:
        - review (Amazon) → 电商评论, 评论/社媒, 测评内容, 广告评论
        - trustpilot → 电商评论, 评论/社媒, 测评内容
        - ticket (Zendesk) → 客服工单, 客服售前, CRM反馈
        - return_note → 客服工单, 追评
        - reddit/social → 社媒, 评论/社媒, 社群讨论, 公开评论区
        """
        SOURCE_TYPE_MAP = {
            "review": {"电商评论", "评论/社媒", "测评内容", "广告评论", "电商评论/社媒"},
            "trustpilot": {"电商评论", "评论/社媒", "测评内容", "公开评论区"},
            "ticket": {"客服工单", "客服售前", "CRM反馈", "会员反馈", "私域互动"},
            "return_note": {"客服工单", "追评"},
        }
        # social/reddit 归入 review 映射
        cn_set = SOURCE_TYPE_MAP.get(source_type, SOURCE_TYPE_MAP.get("review", set()))

        result = []
        for tag in self._tags.values():
            # 通用标签（无数据源限制 或 标记为"通用"）或包含当前数据源
            if not tag.applicable_source or "通用" in tag.applicable_source:
                result.append(tag)
            elif any(name in tag.applicable_source for name in cn_set):
                result.append(tag)
        return result

    def summary(self) -> dict[str, Any]:
        return {
            "total_tags": len(self._tags),
            "by_aipl": {k: len(v) for k, v in self._by_aipl.items()},
            "by_theme": {k: len(v) for k, v in self._by_theme.items()},
            "themes": list(self._by_theme.keys()),
        }


# ---------------------------------------------------------------------------
# 3. 55 原子画像标签
# ---------------------------------------------------------------------------

@dataclass
class AtomicPersonaTag:
    """单条原子画像标签"""

    name: str
    dimension: str  # WHO / WHY / WHAT / WHEN / HOW / EMOTION
    sub_dimension: str
    keywords: list[str]  # 英文关键词
    weight: float = 1.0  # 在画像推导中的权重


# 55 原子画像标签定义（基于实际业务规则）
# 当前默认 48 条，覆盖 WHO/WHY/WHAT/WHEN/HOW/EMOTION 六维十六子维度
# 完整 55 条可通过 PersonaTagMatcher.from_csv() 从外部加载覆盖
DEFAULT_ATOMIC_PERSONA_TAGS: list[AtomicPersonaTag] = [
    # ── WHO - 家庭角色 (5) ──
    AtomicPersonaTag("first_time_parent", "WHO", "family_role",
                     ["first time mom", "new mom", "first baby", "new parent", "newborn"]),
    AtomicPersonaTag("second_time_parent", "WHO", "family_role",
                     ["second baby", "second child", "second time mom", "second pregnancy"]),
    AtomicPersonaTag("working_parent", "WHO", "family_role",
                     ["working mom", "full time job", "back to work", "pumping at work", "career"]),
    AtomicPersonaTag("stay_at_home_parent", "WHO", "family_role",
                     ["stay at home", "sahm", "home with baby", "not working", "full time mom"]),
    AtomicPersonaTag("single_parent", "WHO", "family_role",
                     ["single mom", "on my own", "no help", "by myself", "solo parent"]),
    # ── WHO - 育儿阶段 (5) ──
    AtomicPersonaTag("prenatal", "WHO", "parenting_stage",
                     ["pregnant", "expecting", "due date", "prenatal", "before birth", "third trimester"]),
    AtomicPersonaTag("newborn_stage", "WHO", "parenting_stage",
                     ["newborn", "0-3 months", "just born", "first month", "early days"]),
    AtomicPersonaTag("infant_stage", "WHO", "parenting_stage",
                     ["infant", "3-6 months", "6-12 months", "growing baby", "older baby"]),
    AtomicPersonaTag("toddler_stage", "WHO", "parenting_stage",
                     ["toddler", "1 year old", "over 1", "walking", "weaning"]),
    AtomicPersonaTag("extended_nursing", "WHO", "parenting_stage",
                     ["extended breastfeeding", "nursing beyond", "over 1 year", "toddler nursing"]),
    # ── WHY - 核心痛点 (6) ──
    AtomicPersonaTag("solve_feeding_difficulty", "WHY", "pain_point",
                     ["feeding difficulty", "latch issue", "low supply", "not enough milk", "supplement"]),
    AtomicPersonaTag("solve_sleep_issue", "WHY", "pain_point",
                     ["sleep deprivation", "not sleeping", "tired", "exhausted", "night feeding", "no sleep"]),
    AtomicPersonaTag("solve_pain_discomfort", "WHY", "pain_point",
                     ["pain", "sore", "hurt", "uncomfortable", "engorgement", "mastitis", "nipple pain"]),
    AtomicPersonaTag("seek_self_recovery", "WHY", "pain_point",
                     ["recover", "postpartum", "healing", "body recovery", "self care", "get my body back"]),
    AtomicPersonaTag("time_constrained", "WHY", "pain_point",
                     ["no time", "busy", "rushed", "hurried", "time consuming", "takes too long"]),
    AtomicPersonaTag("mobility_issue", "WHY", "pain_point",
                     ["can't move", "stuck", "tethered", "chained", "trapped", "confined"]),
    # ── WHAT - 功能偏好 (7) ──
    AtomicPersonaTag("quiet_seeker", "WHAT", "function_pref",
                     ["quiet", "silent", "noise", "loud", "discreet", "whisper", "stealth"]),
    AtomicPersonaTag("portable_seeker", "WHAT", "function_pref",
                     ["portable", "compact", "travel", "on the go", "carry", "lightweight", "small"]),
    AtomicPersonaTag("hands_free_seeker", "WHAT", "function_pref",
                     ["hands free", "wearable", "pumping bra", "free hand", "multitask", "while doing"]),
    AtomicPersonaTag("high_capacity_seeker", "WHAT", "function_pref",
                     ["large capacity", "big", "hold more", "oversupply", "high volume", "overproducer"]),
    AtomicPersonaTag("easy_clean_seeker", "WHAT", "function_pref",
                     ["easy clean", "wash", "dishwasher safe", "sterilize", "sanitize", "cleaning"]),
    AtomicPersonaTag("smart_tech_seeker", "WHAT", "function_pref",
                     ["app", "bluetooth", "smart", "digital", "tracking", "timer", "settings"]),
    AtomicPersonaTag("customizable_seeker", "WHAT", "function_pref",
                     ["adjustable", "customize", "different sizes", "multiple modes", "personalize"]),
    # ── WHEN/WHERE - 场景 (6) ──
    AtomicPersonaTag("nighttime_user", "WHEN", "scenario",
                     ["at night", "night time", "middle of the night", "midnight", "2am", "3am", "dark"]),
    AtomicPersonaTag("workplace_user", "WHEN", "scenario",
                     ["at work", "office", "workplace", "pumping room", "break room", "coworking"]),
    AtomicPersonaTag("travel_user", "WHEN", "scenario",
                     ["travel", "trip", "vacation", "flying", "airport", "hotel", "road trip"]),
    AtomicPersonaTag("home_user", "WHEN", "scenario",
                     ["at home", "in bed", "couch", "living room", "bedroom", "nursery"]),
    AtomicPersonaTag("outdoor_user", "WHEN", "scenario",
                     ["outside", "park", "camping", "hiking", "car", "driving", "restaurant"]),
    AtomicPersonaTag("hospital_user", "WHEN", "scenario",
                     ["hospital", "nicu", "medical", "doctor", "clinic", "postpartum ward"]),
    # ── HOW - 决策风格 (8) ──
    AtomicPersonaTag("research_driven", "HOW", "decision_style",
                     ["research", "compare", "comparison", "review", "watch review",
                      "read review", "youtube", "reddit", "facebook group", "blog"]),
    AtomicPersonaTag("word_of_mouth", "HOW", "decision_style",
                     ["friend recommend", "sister recommend", "mom group", "friend told",
                      "recommendation", "word of mouth", "my friend"]),
    AtomicPersonaTag("social_media_influenced", "HOW", "decision_style",
                     ["tiktok", "instagram", "influencer", "saw on", "social media",
                      "viral", "trending", "reels"]),
    AtomicPersonaTag("price_sensitive", "HOW", "decision_style",
                     ["expensive", "cheap", "affordable", "price", "budget", "cost",
                      "worth the money", "value for money", "sale", "discount", "coupon"]),
    AtomicPersonaTag("brand_loyal", "HOW", "decision_style",
                     ["loyal", "always buy", "only buy", "trust", "love this brand",
                      "favorite brand", "stick with", "faithful"]),
    AtomicPersonaTag("impulse_buyer", "HOW", "decision_style",
                     ["impulse", "spur of the moment", "saw and bought", "couldn't resist",
                      "tempted", "bought on a whim"]),
    AtomicPersonaTag("expert_driven", "HOW", "decision_style",
                     ["lactation consultant", "doctor recommend", "pediatrician", "expert",
                      "professional", "medical advice", "midwife"]),
    AtomicPersonaTag("gift_receiver", "HOW", "decision_style",
                     ["gift", "present", "registry", "baby shower", "gifted", "received as"]),
    # ── EMOTION - 情感驱动 (6) ──
    AtomicPersonaTag("anxiety_driven", "EMOTION", "emotion_driver",
                     ["worried", "anxious", "nervous", "concern", "scared", "afraid",
                      "stress", "overwhelmed", "panic"]),
    AtomicPersonaTag("care_driven", "EMOTION", "emotion_driver",
                     ["care", "love", "want the best", "for my baby", "deserve",
                      "only the best", "precious", "protect"]),
    AtomicPersonaTag("frustrated", "EMOTION", "emotion_driver",
                     ["frustrated", "annoyed", "angry", "mad", "furious", "pissed",
                      "sick of", "fed up", "disgusted"]),
    AtomicPersonaTag("hopeful", "EMOTION", "emotion_driver",
                     ["hope", "excited", "looking forward", "can't wait", "optimistic",
                      "positive", "blessed"]),
    AtomicPersonaTag("grateful", "EMOTION", "emotion_driver",
                     ["grateful", "thankful", "appreciate", "blessed", "lucky",
                      "so glad", "relieved"]),
    AtomicPersonaTag("empowered", "EMOTION", "emotion_driver",
                     ["empowered", "confident", "in control", "strong", "capable",
                      "independent", "freedom"]),
    # ── LANGUAGE - 语言/地域 (3) ──
    AtomicPersonaTag("language_english", "LANGUAGE", "language",
                     ["i love", "great product", "highly recommend", "amazing", "awesome",
                      "so happy", "best ever", "thank you", "works well"]),
    AtomicPersonaTag("language_german", "LANGUAGE", "language",
                     ["sehr", "lieferung", "schnelle", "alles", "super", "bestellt",
                      "schnell", "gut", "produkt", "kundenservice", "schneller"]),
    AtomicPersonaTag("language_french", "LANGUAGE", "language",
                     ["commande", "livraison", "rapide", "tres", "produit", "super",
                      "client", "service", "parfait", "excellent", "merci"]),
    # ── WHO - 购买经验扩展 (3) ──
    AtomicPersonaTag("repeat_buyer", "WHO", "purchase_exp",
                     ["second purchase", "buy again", "another one", "bought again",
                      "ordered another", "second one", "third one", "repurchase",
                      "buying again", "got another"]),
    AtomicPersonaTag("gift_buyer", "WHO", "purchase_exp",
                     ["bought for friend", "bought as gift", "bought for sister",
                      "gift for", "present for", "baby shower gift", "registry gift"]),
    AtomicPersonaTag("budget_conscious", "WHO", "purchase_exp",
                     ["on a budget", "tight budget", "can't afford", "money is tight",
                      "cheaper option", "budget friendly", "low cost", "inexpensive"]),
    # ── WHAT - 产品关注扩展 (3) ──
    AtomicPersonaTag("comfort_focused", "WHAT", "product_focus",
                     ["comfortable", "soft", "cozy", "gentle", "snug", "plush",
                      "cushiony", "cloud-like", "like a pillow", "so comfortable"]),
    AtomicPersonaTag("quality_focused", "WHAT", "product_focus",
                     ["high quality", "well made", "sturdy", "built to last", "durable",
                      "solid construction", "premium", "excellent quality", "top notch"]),
    AtomicPersonaTag("convenience_focused", "WHAT", "product_focus",
                     ["convenient", "hassle free", "effortless", "no fuss", "straightforward",
                      "user friendly", "intuitive", "simple to use", "quick setup"]),
    # ── HOW - 评价行为 (2) ──
    AtomicPersonaTag("detailed_reviewer", "HOW", "review_style",
                     ["detailed review", "in depth", "comprehensive", "thorough review",
                      "long review", "extensive review", "covered everything", "very detailed"]),
    AtomicPersonaTag("brief_reviewer", "HOW", "review_style",
                     ["short review", "quick review", "brief", "to the point", "simple review",
                      "not much to say", "quick note", "short and sweet"]),
    # ── EMOTION - 情感细化 (1) ──
    AtomicPersonaTag("disappointed", "EMOTION", "emotion_driver",
                     ["disappointed", "let down", "expected better", "not what i expected",
                      "false advertising", "misleading", "doesn't live up", "overhyped"]),
]


class PersonaTagMatcher:
    """55 原子画像标签匹配器"""

    def __init__(self, atomic_tags: Optional[list[AtomicPersonaTag]] = None):
        self.atomic_tags = atomic_tags or DEFAULT_ATOMIC_PERSONA_TAGS

    def match(self, text: str) -> list[PersonaTagMatch]:
        """匹配文本中的画像标签，返回所有命中的标签"""
        text_lower = text.lower()
        matches: list[PersonaTagMatch] = []

        for tag in self.atomic_tags:
            for kw in tag.keywords:
                if kw.lower() in text_lower:
                    matches.append(PersonaTagMatch(
                        tag_name=tag.name,
                        dimension=tag.dimension,
                        sub_dimension=tag.sub_dimension,
                        confidence=1.0,
                    ))
                    break  # 该标签已命中，不需要检查其他关键词

        return matches

    def match_batch(self, texts: list[str]) -> list[list[PersonaTagMatch]]:
        return [self.match(t) for t in texts]


# ---------------------------------------------------------------------------
# 4. 画像推导: 55 原子 → 3 业务画像
# ---------------------------------------------------------------------------

# 业务画像信号词组
PERSONA_SIGNALS = {
    "community_driven": {  # 社群黏着型
        "word_of_mouth", "social_media_influenced", "price_sensitive",
        "anxiety_driven", "first_time_parent", "care_driven",
        "gift_buyer", "budget_conscious", "language_german", "language_french",
    },
    "systematic_planner": {  # 系统规划型
        "research_driven", "seek_self_recovery", "solve_pain_discomfort",
        "quiet_seeker", "hands_free_seeker", "frustrated",
        "detailed_reviewer", "quality_focused", "convenience_focused",
        "repeat_buyer", "disappointed",
    },
    "quality_explorer": {  # 品质探索型
        "brand_loyal", "portable_seeker", "high_capacity_seeker",
        "care_driven", "nighttime_user", "workplace_user",
        "comfort_focused", "quality_focused", "repeat_buyer",
    },
}


def derive_business_persona(
    atomic_tags: list[str],
    text_length: int = 0,
    sentiment: float = 0.0,
    aipl_tags: list[str] | None = None,
) -> str:
    """基于 55 原子标签共现模式 + AIPL标签 + 文本特征推导业务画像

    Args:
        atomic_tags: 命中的原子标签列表
        text_length: 文本长度（用于兜底推断）
        sentiment: 情感极性（用于兜底推断）
        aipl_tags: 命中的AIPL标签英文名列表（辅助推断）
    """
    atomic_set = set(atomic_tags)

    scores = {
        "community_driven": len(atomic_set & PERSONA_SIGNALS["community_driven"]),
        "systematic_planner": len(atomic_set & PERSONA_SIGNALS["systematic_planner"]),
        "quality_explorer": len(atomic_set & PERSONA_SIGNALS["quality_explorer"]),
    }

    # AIPL标签辅助加分
    if aipl_tags:
        aipl_set = set(aipl_tags)
        # community_driven 信号
        community_aipl = {
            "strong_recommendation", "gift_purchase_intent", "would_buy_again",
            "word_of_mouth_referral", "authentic_mom_recommendation",
        }
        # systematic_planner 信号
        systematic_aipl = {
            "general_dissatisfaction", "general_core_product_performance_issue",
            "difficult_to_use", "poor_usage_experience", "durability_concern",
            "instruction_user_manual", "compatibility_issue",
            "wrong_payment_card_account", "wrong_missing_extra_parts",
        }
        # quality_explorer 信号
        quality_aipl = {
            "comfort_experience", "product_quality_perception", "size_accuracy",
            "material_texture", "design_appearance", "noise_level_acceptable",
            "ease_of_use", "cleaning_maintenance",
        }

        scores["community_driven"] += len(aipl_set & community_aipl) * 0.5
        scores["systematic_planner"] += len(aipl_set & systematic_aipl) * 0.5
        scores["quality_explorer"] += len(aipl_set & quality_aipl) * 0.5

    max_score = max(scores.values())
    if max_score >= 0.5:
        best = max(scores, key=scores.get)  # type: ignore[arg-type]
        return best

    # 兜底推断：降低阈值，更积极分配
    if text_length > 150:
        if sentiment < -0.1:
            return "systematic_planner"
        elif sentiment > 0.1:
            return "quality_explorer"

    if text_length < 100 and sentiment > 0.3:
        return "community_driven"

    # 进一步降低情感阈值
    if sentiment > 0.05:
        return "community_driven"
    elif sentiment < -0.05:
        return "systematic_planner"

    # 最终兜底：按文本长度分配
    if text_length > 120:
        return "quality_explorer"  # 长文本默认品质探索型
    else:
        return "community_driven"  # 短文本默认社群黏着型


# ---------------------------------------------------------------------------
# 5. 情感校准
# ---------------------------------------------------------------------------

class SentimentCalibrator:
    """情感校准器

    标签预定义情感 + ABSA 动态校准:
    - preset=负, ABSA=负 → 取 ABSA 强度, 标记"calibrated"
    - preset=负, ABSA=正 → 标记"conflict", 需人工复核
    - preset=正, ABSA=正 → 取 ABSA 强度, 标记"calibrated"
    """

    # 英文情感词库
    POS_WORDS = {
        "good", "great", "excellent", "amazing", "love", "perfect", "awesome",
        "fantastic", "wonderful", "best", "recommend", "happy", "satisfied",
        "comfortable", "soft", "nice", "smooth", "easy", "convenient",
        "efficient", "effective", "reliable", "durable", "quality",
    }
    NEG_WORDS = {
        "bad", "terrible", "awful", "worst", "hate", "disappointed", "poor",
        "horrible", "useless", "broken", "defective", "leak", "leaking",
        "painful", "uncomfortable", "hard", "rough", "noisy", "loud",
        "difficult", "complicated", "frustrating", "annoying", "waste",
    }
    INTENSIFIERS = {
        "very": 1.4, "extremely": 1.6, "really": 1.3, "quite": 1.2,
        "super": 1.5, "incredibly": 1.6, "absolutely": 1.5, "totally": 1.4,
        "so": 1.3, "too": 1.2, "pretty": 1.1, "fairly": 1.1,
    }
    NEGATORS = {"not", "no", "never", "none", "nobody", "nothing", "neither", "nor", "hardly", "barely", "scarcely"}

    # ── 德语情感词库 ──
    DE_POS_WORDS = {
        "gut", "gute", "guter", "gutes", "guten",
        "super", "toll", "tolle", "perfekt", "empfehlen", "zufrieden",
        "schnell", "schnelle", "schneller", "liebe", "prima",
        "hervorragend", "ausgezeichnet", "wunderbar", "fantastisch",
        "beste", "genial", "top", "klasse", "empfehlenswert", "begeistert",
        "zufriedenstellend", "einfach", "komfortabel", "weich", "schön",
        "produkt", "lieferung", "service", "kundenservice", "bestellung",
        "verpackung",
    }
    DE_NEG_WORDS = {
        "schlecht", "schlechte", "schlechter", "schrecklich", "furchtbar",
        "enttäuscht", "enttäuschend", "kaputt", "defekt", "defekte",
        "langsam", "schwierig", "ärgerlich", "frustrierend", "nutzlos",
        "gebrochen", "schmerzhaft", "unbequem", "hart", "laut", "laute",
        "kompliziert", "müll", "unzufrieden", "mangelhaft", "problematisch",
        "fehlerhaft", "nervt", "ärger", "schlimm", "grässlich", "schade",
        "probleme", "beschädigt", "unbrauchbar",
    }
    DE_INTENSIFIERS = {
        "sehr": 1.4, "extrem": 1.6, "wirklich": 1.3, "ziemlich": 1.2,
        "total": 1.4, "absolut": 1.5, "ultra": 1.5, "unglaublich": 1.6,
        "wahnsinnig": 1.5, "besonders": 1.3, "äußerst": 1.6,
    }
    DE_NEGATORS = {"nicht", "nein", "nie", "niemand", "nichts", "kaum", "niemals", "weder", "noch"}

    # ── 法语情感词库 ──
    FR_POS_WORDS = {
        "bon", "super", "excellent", "parfait", "recommander", "satisfait", "rapide",
        "aime", "génial", "fantastique", "merveilleux", "magnifique", "meilleur",
        "top", "géniale", "agréable", "confortable", "doux", "facile", "pratique",
        "efficace", "solide", "durables",
    }
    FR_NEG_WORDS = {
        "mauvais", "terrible", "horrible", "déçu", "cassé", "défectueux", "lent",
        "difficile", "frustrant", "ennuyeux", "inutile", "fuit", "douloureux",
        "inconfortable", "dur", "rugueux", "bruyant", "compliqué", "gaspillage",
        "décevant", "insuffisant", "problématique", "défectueux",
    }
    FR_INTENSIFIERS = {
        "très": 1.4, "extrêmement": 1.6, "vraiment": 1.3, "assez": 1.2,
        "totalement": 1.4, "absolument": 1.5, "ultra": 1.5, "incroyablement": 1.6,
        "tellement": 1.4, "plutôt": 1.1, "vachement": 1.5,
    }
    FR_NEGATORS = {"pas", "non", "jamais", "personne", "rien", "ni", "guère", "nullement"}

    # 语言检测特征词（用于快速判断文本语言）
    DE_LANG_MARKERS = {
        "sehr", "gut", "super", "schnell", "lieferung", "bestellt", "produkt",
        "kundenservice", "schneller", "schnelle", "alles", "toll", "besten",
        "nicht", "kein", "keine", "schlecht", "schlechte", "gute", "ware",
        "einfach", "schneller", "empfehlen", "weiterempfehlen", "enttäuscht",
        "enttauscht", "zufrieden", "unzufrieden", "qualität", "preis",
    }
    FR_LANG_MARKERS = {
        "très", "bon", "super", "rapide", "commande", "livraison", "produit",
        "service", "parfait", "excellent", "client", "merci", "génial",
        "pas", "non", "mauvais", "mauvaise", "bonne", "qualité", "prix",
        "déçu", "decu", "recommande", "recommander", "satisfait", "insatisfait",
        "facile", "vite", "trop", "assez", "vraiment",
    }

    # 字符级语言检测正则
    DE_CHAR_PATTERN = re.compile(r"[äöüßÄÖÜẞ]")
    FR_CHAR_PATTERN = re.compile(r"[éèàçâêîôûÉÈÀÇÂÊÎÔÛ]")

    def __init__(self) -> None:
        pass

    def _detect_language(self, words: set[str], text_lower: str = "") -> str:
        """基于特征词 + 字符检测文本语言: en / de / fr

        修复: 短句中特征词可能只有 1 个，结合字符级检测 (äöüß / éèàç) 提升准确率。
        """
        # 字符级检测 — 优先级最高
        if self.DE_CHAR_PATTERN.search(text_lower):
            return "de"
        if self.FR_CHAR_PATTERN.search(text_lower):
            return "fr"

        # 词汇级检测 — 阈值从 >=2 降到 >=1，适应短句
        de_hits = len(words & self.DE_LANG_MARKERS)
        fr_hits = len(words & self.FR_LANG_MARKERS)
        if de_hits >= 1 and de_hits > fr_hits:
            return "de"
        if fr_hits >= 1 and fr_hits > de_hits:
            return "fr"
        return "en"

    def calibrate(
        self,
        text: str,
        preset_sentiment: str,
        rating: Optional[float] = None,
        cached_words: Optional[list[str]] = None,
    ) -> SentimentCalibration:
        """校准单条文本的情感"""
        text_lower = text.lower()
        words = cached_words if cached_words is not None else re.findall(r"[a-z\u00e0-\u00ff]+", text_lower)

        # 1. ABSA 动态计算（基于情感词统计）
        absa_polarity, absa_intensity = self._absa_score(words, text_lower)

        # 2. 星级评分校准（如有）
        if rating is not None:
            rating_sentiment = (rating - 3) / 2.0  # 1星=-1, 3星=0, 5星=+1
            # 融合 ABSA + 评分（评分权重 0.3）
            absa_polarity = absa_polarity * 0.7 + rating_sentiment * 0.3

        # 3. 与预定义情感校准
        preset_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        preset_val = preset_map.get(preset_sentiment, 0.0)

        if preset_val == 0.0:
            # 预定义中性 → 直接用 ABSA
            flag = "calibrated"
            final_polarity = absa_polarity
        elif preset_val * absa_polarity >= 0:
            # 同向 → 取 ABSA（动态校准）
            flag = "calibrated"
            final_polarity = absa_polarity
        else:
            # 反向 → 冲突标记
            flag = "conflict"
            # 保守策略：取预定义方向，但降低幅度
            final_polarity = preset_val * 0.5

        # 强度映射
        intensity = max(-5.0, min(5.0, final_polarity * absa_intensity * 5.0))

        # 方面级情感（简化版）
        aspect_sentiments = self._extract_aspect_sentiments(text_lower)

        conflict_reason = ""
        if flag == "conflict":
            conflict_reason = (
                f"preset={preset_sentiment}({preset_val:+.1f}) vs "
                f"ABSA={absa_polarity:+.2f}"
            )

        return SentimentCalibration(
            polarity=final_polarity,
            intensity=intensity,
            calibration_flag=flag,
            aspect_sentiments=aspect_sentiments,
            conflict_reason=conflict_reason,
        )

    def _absa_score(self, words: list[str], text_lower: str) -> tuple[float, float]:
        """基于词汇统计的 ABSA 情感分 — 支持多语言"""
        # 检测语言
        lang = self._detect_language(set(words), text_lower)

        # 选择对应词库
        if lang == "de":
            pos_words, neg_words = self.DE_POS_WORDS, self.DE_NEG_WORDS
            intensifiers, negators = self.DE_INTENSIFIERS, self.DE_NEGATORS
        elif lang == "fr":
            pos_words, neg_words = self.FR_POS_WORDS, self.FR_NEG_WORDS
            intensifiers, negators = self.FR_INTENSIFIERS, self.FR_NEGATORS
        else:
            pos_words, neg_words = self.POS_WORDS, self.NEG_WORDS
            intensifiers, negators = self.INTENSIFIERS, self.NEGATORS

        pos_count = sum(1 for w in words if w in pos_words)
        neg_count = sum(1 for w in words if w in neg_words)

        # 程度副词检测
        multiplier = 1.0
        for intensifier, factor in intensifiers.items():
            if intensifier in text_lower:
                multiplier = max(multiplier, factor)

        # 否定词检测
        negator_count = sum(1 for w in words if w in negators)
        if negator_count > 0:
            multiplier *= max(0.3, 1.0 - negator_count * 0.3)

        if pos_count > neg_count:
            polarity = 1.0
        elif neg_count > pos_count:
            polarity = -1.0
        else:
            polarity = 0.0

        # 强度 = 情感词密度 × 程度倍数
        total_words = max(len(words), 1)
        intensity = (pos_count + neg_count) / total_words * multiplier * 10
        intensity = min(intensity, 1.0)

        return polarity, intensity

    def _extract_aspect_sentiments(self, text_lower: str) -> dict[str, float]:
        """提取方面级情感（简化版关键词匹配）"""
        aspects = {
            "quality": ["quality", "material", "durability", "build"],
            "comfort": ["comfortable", "soft", "pain", "hurt", "sore"],
            "noise": ["noise", "quiet", "silent", "loud", "sound"],
            "suction": ["suction", "suck", "pull", "strength", "power"],
            "battery": ["battery", "charge", "charging", "last", "hours"],
            "design": ["design", "look", "style", "color", "appearance"],
            "price": ["price", "expensive", "cheap", "cost", "value", "money"],
            "service": ["service", "support", "customer", "warranty", "return"],
        }

        result: dict[str, float] = {}
        for aspect, keywords in aspects.items():
            if any(kw in text_lower for kw in keywords):
                # 简单的方面情感（基于附近情感词）
                pos = sum(1 for w in self.POS_WORDS if w in text_lower)
                neg = sum(1 for w in self.NEG_WORDS if w in text_lower)
                if pos > neg:
                    result[aspect] = 0.5
                elif neg > pos:
                    result[aspect] = -0.5
                else:
                    result[aspect] = 0.0
        return result


# ---------------------------------------------------------------------------
# 6. 品牌检测
# ---------------------------------------------------------------------------

class BrandDetector:
    """品牌提及与竞品对比检测"""

    # 品牌列表
    OWN_BRANDS = {"momcozy"}
    COMPETITOR_BRANDS = {
        "spectra", "medela", "willow", "elvie", "lansinoh",
        "haakaa", "philips avent", "avent", "tommee tippee",
        "dr brown", "comotomo", "nuk",
    }
    ALL_BRANDS = OWN_BRANDS | COMPETITOR_BRANDS

    # 对比词
    COMPARISON_WORDS = {
        "vs", "versus", "compared to", "compare", "better than", "worse than",
        "prefer", "over", "instead of", "rather than", "switch from",
        "upgraded from", "came from", "used to use", "previous",
    }

    def detect(self, text: str) -> tuple[list[str], bool]:
        """检测品牌提及和对比

        Returns: (提及品牌列表, 是否包含竞品对比)
        """
        text_lower = text.lower()
        mentions: list[str] = []

        for brand in self.ALL_BRANDS:
            # 处理多词品牌（如 "philips avent"）
            if " " in brand:
                if brand in text_lower:
                    mentions.append(brand)
            else:
                # 整词匹配（避免 "med" 匹配 "medela"）
                pattern = r'\b' + re.escape(brand) + r'\b'
                if re.search(pattern, text_lower):
                    mentions.append(brand)

        # 竞品对比检测：竞品品牌 + 对比词
        has_comparison = False
        for comp in self.COMPETITOR_BRANDS:
            if comp in mentions:
                for comp_word in self.COMPARISON_WORDS:
                    if comp_word in text_lower:
                        has_comparison = True
                        break
            if has_comparison:
                break

        return mentions, has_comparison


# ---------------------------------------------------------------------------
# 7. Proxy NPS 计算
# ---------------------------------------------------------------------------

class ProxyNPSCalculator:
    """Proxy NPS 计算器（多标签场景）"""

    def calculate(
        self,
        aipl_tags: list[AIPLTagMatch],
        sentiment_polarity: float,
        rating: Optional[float],
    ) -> str:
        """计算单条 VOC 的 Proxy NPS 贡献

        规则:
        - [推荐意愿] + 正向 → 推荐者
        - [推荐意愿] + 负向 → 贬损者（推荐意愿但表达不满=虚假宣传）
        - [产品问题] + 负向 → 贬损者
        - [产品问题] + 正向 → 被动者（有问题但整体还行）
        - 无标签 + 5星 → 推荐者
        - 无标签 + 1-2星 → 贬损者
        """
        # 检查是否有推荐意愿标签
        has_recommendation = any(
            "recommend" in t.tag_en.lower() or t.aipl_node == "L3"
            for t in aipl_tags
        )

        # 检查是否有产品问题标签
        has_product_issue = any(
            t.theme in {"产品核心性能", "使用舒适度", "产品问题"}
            or t.aipl_node in {"P1", "P2", "L1", "L2"}
            for t in aipl_tags
        )

        # 情感极性判断
        is_positive = sentiment_polarity > 0.2
        is_negative = sentiment_polarity < -0.2

        if has_recommendation:
            if is_positive:
                return "promoter"
            elif is_negative:
                return "detractor"  # 推荐意愿但表达不满
            else:
                return "passive"

        if has_product_issue:
            if is_negative:
                return "detractor"
            elif is_positive:
                return "passive"  # 有问题但整体还行
            else:
                return "passive"

        # 无标签 → 按评分判断
        if rating is not None:
            if rating >= 4:
                return "promoter"
            elif rating <= 2:
                return "detractor"
            else:
                return "passive"

        # 无标签无评分 → 按情感判断
        if is_positive:
            return "promoter"
        elif is_negative:
            return "detractor"
        return "passive"

    def calculate_batch(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, float]:
        """批量计算 Proxy NPS 得分"""
        total = len(extractions)
        if total == 0:
            return {"proxy_nps": 0.0, "promoters": 0, "passives": 0, "detractors": 0}

        promoters = sum(1 for e in extractions if e.proxy_nps_contribution == "promoter")
        passives = sum(1 for e in extractions if e.proxy_nps_contribution == "passive")
        detractors = sum(1 for e in extractions if e.proxy_nps_contribution == "detractor")

        proxy_nps = (promoters / total * 100) - (detractors / total * 100)

        return {
            "proxy_nps": round(proxy_nps, 1),
            "promoters": promoters,
            "passives": passives,
            "detractors": detractors,
            "promoter_pct": round(promoters / total * 100, 1),
            "detractor_pct": round(detractors / total * 100, 1),
        }


# ---------------------------------------------------------------------------
# 8. 核心萃取器
# ---------------------------------------------------------------------------

class VOCLabelExtractor:
    """单条 VOC 完整标签萃取器

    6 维度并行萃取:
    1. AIPL 旅程标签（376 标签种子匹配 + 品线过滤）
    2. 问题类型标签（现有 classification 映射）
    3. 画像标签（55 原子标签匹配）
    4. 情感校准（预定义 + ABSA 动态）
    5. 品牌检测
    6. Proxy NPS 计算
    """

    def __init__(
        self,
        tag_dict: TagSeedDictionary,
        persona_matcher: Optional[PersonaTagMatcher] = None,
        sentiment_calibrator: Optional[SentimentCalibrator] = None,
        brand_detector: Optional[BrandDetector] = None,
        proxy_nps_calculator: Optional[ProxyNPSCalculator] = None,
    ):
        self.tag_dict = tag_dict
        self.persona_matcher = persona_matcher or PersonaTagMatcher()
        self.sentiment_calibrator = sentiment_calibrator or SentimentCalibrator()
        self.brand_detector = brand_detector or BrandDetector()
        self.proxy_nps = proxy_nps_calculator or ProxyNPSCalculator()
        # 预编译所有标签关键词的正则表达式（提速 3-5x）
        self._precompile_patterns()
        # 缓存文本词列表（避免同文本多次分词）
        self._text_word_cache: dict[str, list[str]] = {}

    def _precompile_patterns(self) -> None:
        """预编译所有关键词的正则表达式"""
        self._compiled_patterns: dict[str, list[tuple[str, Optional[re.Pattern]]]] = {}
        for tag in self.tag_dict.get_all():
            patterns: list[tuple[str, Optional[re.Pattern]]] = []
            for kw in tag.keywords + tag.consumer_keywords:
                kw_lower = kw.lower()
                if len(kw_lower) >= 4:
                    patterns.append((kw_lower, re.compile(r'\b' + re.escape(kw_lower) + r'\b')))
                else:
                    patterns.append((kw_lower, None))
            self._compiled_patterns[tag.tag_id] = patterns

    def _get_words(self, text_lower: str) -> list[str]:
        """获取文本词列表（带缓存）"""
        if text_lower not in self._text_word_cache:
            self._text_word_cache[text_lower] = re.findall(r"[a-z\u00e0-\u00ff]+", text_lower)
        return self._text_word_cache[text_lower]

    def extract(self, voc: VOCRecord) -> VOCLabelExtraction:
        """萃取单条 VOC 的全部标签"""
        text_lower = voc.text.lower()

        # Step 1: 按品线过滤标签
        candidate_tags = self.tag_dict.filter_by_line(voc.product_line)

        # Step 2: 376 标签种子匹配（多标签）
        aipl_tags = self._match_aipl_tags(text_lower, candidate_tags)

        # Step 3: 确定主 AIPL 阶段
        aipl_stage = self._derive_aipl_stage(aipl_tags, voc.source_type)

        # Step 4: 55 原子画像标签匹配
        persona_matches = self.persona_matcher.match(voc.text)
        persona_atomic = [m.tag_name for m in persona_matches]

        # Step 4.5: 按6维画像分组（原生输出画像×AIPL结构化数据的基础）
        persona_dimensions: dict[str, list[str]] = defaultdict(list)
        for m in persona_matches:
            persona_dimensions[m.dimension].append(m.tag_name)
        persona_dimensions = dict(persona_dimensions)

        # Step 5: 情感校准（取第一个标签的预定义情感作为基准）
        preset_sentiment = aipl_tags[0].sentiment_preset if aipl_tags else "neutral"
        sentiment = self.sentiment_calibrator.calibrate(
            voc.text, preset_sentiment, voc.rating
        )

        # Step 6: 画像推导（基于原子标签 + AIPL标签 + 文本特征）
        aipl_tag_names = [t.tag_en for t in aipl_tags]
        persona_derived = derive_business_persona(
            persona_atomic, len(voc.text), sentiment.polarity, aipl_tag_names
        )

        # Step 7: 品牌检测
        brand_mentions, brand_comparison = self.brand_detector.detect(voc.text)

        # Step 8: Proxy NPS
        proxy_nps = self.proxy_nps.calculate(
            aipl_tags, sentiment.polarity, voc.rating
        )

        # Step 9: 聚合业务元数据（从命中的标签中取最高优先级）
        business_meta = self._aggregate_business_meta(aipl_tags)

        return VOCLabelExtraction(
            review_id=voc.review_id,
            source_type=voc.source_type,
            platform=voc.platform,
            spu_code=voc.spu_code,
            product_line=voc.product_line,
            category=voc.category,
            rating=voc.rating,
            aipl_stage=aipl_stage,
            aipl_tags=aipl_tags,
            classification_tag=voc.classification_tag or "",
            cn_level1=voc.cn_level1 or "",
            cn_level2=voc.cn_level2 or "",
            cn_level3=voc.cn_level3 or "",
            persona_atomic=persona_atomic,
            persona_derived=persona_derived,
            persona_dimensions=persona_dimensions,
            sentiment_polarity=sentiment.polarity,
            sentiment_intensity=sentiment.intensity,
            sentiment_calibration=sentiment.calibration_flag,
            aspect_sentiments=sentiment.aspect_sentiments,
            brand_mentions=brand_mentions,
            brand_comparison=brand_comparison,
            quality_score=0.0,  # 由 ReviewQualityPipeline 填充
            is_suspicious=False,  # 由 SpamDetector 填充
            proxy_nps_contribution=proxy_nps,
            metric_direction=business_meta.get("metric_direction", ""),
            story_line=business_meta.get("story_line", ""),
            strategy_pack=business_meta.get("strategy_pack", ""),
            owner_dept=business_meta.get("owner_dept", ""),
            priority=business_meta.get("priority", ""),
        )

    # 否定词列表（用于关键词匹配过滤）
    NEGATION_WORDS = {"not", "no", "never", "none", "nobody", "nothing", "n't", "dont", "doesnt", "didnt", "wouldnt", "couldnt", "shouldnt", "wont", "cant"}

    # 高冲突标签的二次验证规则 — 减少误匹配
    TAG_EXCLUSION_RULES: dict[str, dict] = {
        "general_core_product_performance_issue": {
            "exclude_if_any": ["shipping", "delivery", "package", "arrived", "ordered", "transit", "carrier", "shipped", "delivered", "arrives"],
        },
        "instruction_user_manual": {
            "exclude_if_any": ["easy to follow", "clear instructions", "helpful manual", "great instructions", "excellent manual", "well written", "easy to understand", "simple instructions", "good instructions"],
        },
        "size_runs_large": {
            "exclude_if_any": ["large amount", "large space", "large enough", "big help", "big difference", "big fan", "big plus", "big improvement"],
        },
        "size_runs_small": {
            "exclude_if_any": ["small amount", "small space", "small enough", "small price", "small price to pay", "little help", "little bit"],
        },
        "delivery_too_slow": {
            "require_any": ["shipping", "delivery", "package", "arrived", "ordered", "transit", "carrier", "shipped", "delivered"],
        },
        "price_concern": {
            "exclude_if_any": ["worth the price", "worth every penny", "good price", "reasonable price", "great price", "fair price", "excellent value", "good value", "worth it"],
        },
        "forgot_used_wrong_discount_code": {
            "require_any": ["discount", "coupon", "code", "promo", "promotion", "voucher", "deal", "sale price"],
        },
        "burnt_smell": {
            "exclude_if_any": ["went away", "dissipated", "faded", "after a few", "after washing", "air it out", "went away after"],
        },
        "too_noisy": {
            "exclude_if_any": ["not noisy", "not loud", "quiet", "silent", "barely audible", "not too loud"],
        },
        "wrong_missing_extra_parts": {
            "require_any": ["order", "shipment", "received", "delivered", "package", "box", "missing from", "not in the box"],
        },
        "wrong_quantity": {
            "require_any": ["order", "shipment", "received", "delivered", "package", "box", "ordered", "supposed to be"],
        },
    }

    def _should_exclude_tag(self, tag_en: str, text_lower: str) -> bool:
        """对高冲突标签进行二次验证 — 返回 True 表示应排除该匹配"""
        rules = self.TAG_EXCLUSION_RULES.get(tag_en)
        if not rules:
            return False

        # exclude_if_any: 文本包含任一排除词 → 排除
        for phrase in rules.get("exclude_if_any", []):
            if phrase in text_lower:
                return True

        # require_any: 文本不包含任一必须词 → 排除
        required = rules.get("require_any", [])
        if required and not any(r in text_lower for r in required):
            return True

        return False

    def _check_negation_context(self, text_lower: str, keyword: str, window: int = 15) -> bool:
        """检查关键词前 window 字符内是否有否定词

        Returns: True 如果被否定修饰（应跳过匹配）
        """
        idx = text_lower.find(keyword.lower())
        if idx < 0:
            return False
        prefix = text_lower[max(0, idx - window):idx]
        return any(neg in prefix for neg in self.NEGATION_WORDS)

    def _match_aipl_tags(
        self,
        text_lower: str,
        candidate_tags: list[TagSeed],
    ) -> list[AIPLTagMatch]:
        """关键词匹配 AIPL 标签（多标签）— 使用预编译正则加速"""
        matches: list[AIPLTagMatch] = []

        for tag in candidate_tags:
            compiled = self._compiled_patterns.get(tag.tag_id, [])
            matched = False
            match_kw = ""

            for kw_lower, pattern in compiled:
                if pattern is None:
                    # 短关键词，子串匹配
                    if kw_lower in text_lower:
                        if self._check_negation_context(text_lower, kw_lower):
                            continue
                        matched = True
                        match_kw = kw_lower
                        break
                else:
                    # 长关键词，整词匹配（预编译正则）
                    if pattern.search(text_lower):
                        if self._check_negation_context(text_lower, kw_lower):
                            continue
                        matched = True
                        match_kw = kw_lower
                        break

            if matched and self._should_exclude_tag(tag.tag_en, text_lower):
                matched = False
                match_kw = ""

            if matched:
                confidence = min(len(match_kw) / 20.0, 1.0)
                sentiment = self.sentiment_calibrator.calibrate(
                    text_lower, tag.sentiment_preset, None,
                    cached_words=self._get_words(text_lower),
                )
                matches.append(AIPLTagMatch(
                    tag_id=tag.tag_id,
                    tag_en=tag.tag_en,
                    tag_cn=tag.tag_cn,
                    theme=tag.theme,
                    aipl_node=tag.aipl_node,
                    sentiment_preset=tag.sentiment_preset,
                    sentiment_calibrated=sentiment.polarity,
                    confidence=confidence,
                ))

        matches.sort(key=lambda x: x.confidence, reverse=True)
        return matches

    def _derive_aipl_stage(
        self,
        aipl_tags: list[AIPLTagMatch],
        source_type: str,
    ) -> str:
        """推导主 AIPL 阶段"""
        if not aipl_tags:
            # 无标签时，按数据源推断
            source_stage_map = {
                "return_note": "P1",
                "ticket": "L2",
                "review": "L1",
                "trustpilot": "L3",
            }
            return source_stage_map.get(source_type, "unknown")

        # 有标签时，取最高置信度标签的节点
        # 或取最多标签的节点
        node_counts: Counter[str] = Counter()
        for tag in aipl_tags:
            node_counts[tag.aipl_node] += tag.confidence

        return node_counts.most_common(1)[0][0]

    def _aggregate_business_meta(self, aipl_tags: list[AIPLTagMatch]) -> dict[str, str]:
        """从命中的标签聚合业务元数据 — 取最高优先级标签的完整元数据"""
        if not aipl_tags:
            return {}

        priority_order = {"P0": 0, "P1": 1, "P2": 2, "P3": 3}

        # 收集所有标签的元数据，带优先级
        candidates: list[tuple[int, TagSeed]] = []
        for tag_match in aipl_tags:
            seed = self.tag_dict.get(tag_match.tag_id)
            if seed and seed.priority:
                rank = priority_order.get(seed.priority, 99)
                candidates.append((rank, seed))

        if not candidates:
            # 无优先级信息，取第一个有元数据的标签
            for tag_match in aipl_tags:
                seed = self.tag_dict.get(tag_match.tag_id)
                if seed:
                    return {
                        "metric_direction": seed.metric_direction or "",
                        "story_line": seed.story_line or "",
                        "strategy_pack": seed.strategy_pack or "",
                        "owner_dept": seed.owner_dept or "",
                        "priority": seed.priority or "",
                    }
            return {}

        # 按优先级排序，取最高优先级标签的元数据
        candidates.sort(key=lambda x: x[0])
        best_seed = candidates[0][1]

        return {
            "metric_direction": best_seed.metric_direction or "",
            "story_line": best_seed.story_line or "",
            "strategy_pack": best_seed.strategy_pack or "",
            "owner_dept": best_seed.owner_dept or "",
            "priority": best_seed.priority or "",
        }


# ---------------------------------------------------------------------------
# 9. 统一流水线
# ---------------------------------------------------------------------------

class UnifiedLabelingPipeline:
    """统一打标流水线

    端到端: VOCRecord[] → VOCLabelExtraction[]
    集成 ReviewQuality 质量筛选（可选）
    """

    def __init__(
        self,
        tag_dict: TagSeedDictionary,
        quality_pipeline=None,  # 可选: ReviewQualityPipeline
        persona_matcher: Optional[PersonaTagMatcher] = None,
    ):
        self.extractor = VOCLabelExtractor(
            tag_dict=tag_dict,
            persona_matcher=persona_matcher,
        )
        self.quality_pipeline = quality_pipeline

    def process(self, vocs: list[VOCRecord]) -> list[VOCLabelExtraction]:
        """批量处理 VOC 列表"""
        results: list[VOCLabelExtraction] = []

        for voc in vocs:
            extraction = self.extractor.extract(voc)

            # 如有质量流水线，填充质量分
            if self.quality_pipeline is not None:
                rating_int = int(voc.rating) if voc.rating else None
                quality_result = self.quality_pipeline.process(voc.text, rating_int)
                extraction.quality_score = quality_result.quality_score.overall_score
                extraction.is_suspicious = quality_result.spam_detection.is_suspicious

            results.append(extraction)

        return results

    def filter_suspicious(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> list[VOCLabelExtraction]:
        """过滤掉可疑/低质量 VOC"""
        return [e for e in extractions if not e.is_suspicious]


# ---------------------------------------------------------------------------
# 10. 看板生成器
# ---------------------------------------------------------------------------

@dataclass
class DashboardData:
    """指标看板数据结构"""

    proxy_nps: dict[str, Any]
    aipl_funnel: dict[str, Any]
    driver_analysis: dict[str, Any]
    persona_insights: dict[str, Any]
    tag_coverage: dict[str, Any]
    brand_analysis: dict[str, Any]
    persona_aipl_matrix: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return {
            "proxy_nps": self.proxy_nps,
            "aipl_funnel": self.aipl_funnel,
            "driver_analysis": self.driver_analysis,
            "persona_insights": self.persona_insights,
            "tag_coverage": self.tag_coverage,
            "brand_analysis": self.brand_analysis,
            "persona_aipl_matrix": self.persona_aipl_matrix,
        }

    def to_json(self, path: str) -> None:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, ensure_ascii=False, indent=2)


class DashboardGenerator:
    """指标看板生成器"""

    def build(self, extractions: list[VOCLabelExtraction]) -> DashboardData:
        """从萃取结果生成指标看板"""
        # 过滤掉可疑数据
        valid = [e for e in extractions if not e.is_suspicious]

        return DashboardData(
            proxy_nps=self._calc_proxy_nps(valid),
            aipl_funnel=self._calc_aipl_funnel(valid),
            driver_analysis=self._calc_driver_analysis(valid),
            persona_insights=self._calc_persona_insights(valid),
            tag_coverage=self._calc_tag_coverage(extractions),
            brand_analysis=self._calc_brand_analysis(valid),
            persona_aipl_matrix=self._calc_persona_aipl_matrix(valid),
        )

    def _calc_proxy_nps(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """计算 Proxy NPS 多维度拆分"""
        calc = ProxyNPSCalculator()
        overall = calc.calculate_batch(extractions)

        # 按品线拆分
        by_line: dict[str, list[VOCLabelExtraction]] = defaultdict(list)
        for e in extractions:
            by_line[e.product_line].append(e)

        by_product_line = {
            line: calc.calculate_batch(items)
            for line, items in by_line.items()
        }

        # 按画像拆分
        by_persona: dict[str, list[VOCLabelExtraction]] = defaultdict(list)
        for e in extractions:
            by_persona[e.persona_derived].append(e)

        by_persona_out = {
            persona: calc.calculate_batch(items)
            for persona, items in by_persona.items()
        }

        # 按平台拆分
        by_platform: dict[str, list[VOCLabelExtraction]] = defaultdict(list)
        for e in extractions:
            by_platform[e.platform].append(e)

        by_platform_out = {
            platform: calc.calculate_batch(items)
            for platform, items in by_platform.items()
        }

        return {
            "overall": overall,
            "by_product_line": by_product_line,
            "by_persona": by_persona_out,
            "by_platform": by_platform_out,
        }

    def _calc_aipl_funnel(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """计算 AIPL 旅程漏斗"""
        node_counts: dict[str, int] = Counter()
        node_themes: dict[str, Counter[str]] = defaultdict(Counter)

        for e in extractions:
            node_counts[e.aipl_stage] += 1
            for tag in e.aipl_tags:
                node_themes[e.aipl_stage][tag.theme] += 1

        funnel = {}
        for node in ["A", "I", "P1", "P2", "L1", "L2", "L3"]:
            count = node_counts.get(node, 0)
            top_themes = [
                {"theme": t, "count": c}
                for t, c in node_themes[node].most_common(3)
            ]
            funnel[node] = {"count": count, "top_themes": top_themes}

        return funnel

    def _calc_driver_analysis(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """驱动因素分析"""
        # 按主题聚合情感
        theme_sentiments: dict[str, list[float]] = defaultdict(list)
        theme_counts: Counter[str] = Counter()

        for e in extractions:
            for tag in e.aipl_tags:
                theme_sentiments[tag.theme].append(tag.sentiment_calibrated)
                theme_counts[tag.theme] += 1

        total = len(extractions) if extractions else 1

        theme_stats = []
        for theme, sentiments in theme_sentiments.items():
            avg_sentiment = sum(sentiments) / len(sentiments)
            mention_rate = theme_counts[theme] / total
            theme_stats.append({
                "theme": theme,
                "mention_rate": round(mention_rate, 3),
                "avg_sentiment": round(avg_sentiment, 2),
                "count": theme_counts[theme],
                "nps_contribution": (
                    "promoter_driver" if avg_sentiment > 0.3
                    else "detractor_driver" if avg_sentiment < -0.3
                    else "neutral"
                ),
            })

        # 排序
        theme_stats.sort(key=lambda x: x["mention_rate"], reverse=True)

        detractor_themes = [t for t in theme_stats if t["nps_contribution"] == "detractor_driver"]
        promoter_themes = [t for t in theme_stats if t["nps_contribution"] == "promoter_driver"]

        return {
            "top_detractor_themes": detractor_themes[:5],
            "top_promoter_themes": promoter_themes[:5],
            "all_themes": theme_stats[:10],
        }

    def _calc_persona_insights(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """画像洞察分析"""
        by_persona: dict[str, list[VOCLabelExtraction]] = defaultdict(list)
        for e in extractions:
            by_persona[e.persona_derived].append(e)

        total = len(extractions) if extractions else 1
        insights = {}

        for persona, items in by_persona.items():
            # 渗透率
            penetration = len(items) / total

            # 平均情感
            sentiments = [e.sentiment_polarity for e in items]
            avg_sentiment = sum(sentiments) / len(sentiments) if sentiments else 0

            # Top 主题
            theme_counts: Counter[str] = Counter()
            for e in items:
                for tag in e.aipl_tags:
                    theme_counts[tag.theme] += 1

            top_themes = [t for t, _ in theme_counts.most_common(5)]

            # Proxy NPS
            calc = ProxyNPSCalculator()
            nps_data = calc.calculate_batch(items)

            insights[persona] = {
                "penetration": round(penetration, 3),
                "count": len(items),
                "avg_sentiment": round(avg_sentiment, 2),
                "top_themes": top_themes,
                "proxy_nps": nps_data,
            }

        return insights

    def _calc_tag_coverage(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """标签覆盖率统计"""
        total = len(extractions)
        matched = sum(1 for e in extractions if e.aipl_tags)
        unmatched = total - matched

        # 唯一标签数
        all_tag_ids: set[str] = set()
        for e in extractions:
            for tag in e.aipl_tags:
                all_tag_ids.add(tag.tag_id)

        return {
            "total_voc": total,
            "matched_voc": matched,
            "unmatched_voc": unmatched,
            "coverage_rate": round(matched / total, 3) if total else 0,
            "unique_tags_matched": len(all_tag_ids),
        }

    def _calc_brand_analysis(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """品牌分析"""
        brand_counts: Counter[str] = Counter()
        comparison_count = 0

        for e in extractions:
            for brand in e.brand_mentions:
                brand_counts[brand] += 1
            if e.brand_comparison:
                comparison_count += 1

        return {
            "brand_mentions": dict(brand_counts.most_common(10)),
            "comparison_count": comparison_count,
            "comparison_rate": round(comparison_count / len(extractions), 3) if extractions else 0,
        }

    def _calc_persona_aipl_matrix(
        self,
        extractions: list[VOCLabelExtraction],
    ) -> dict[str, Any]:
        """计算画像×AIPL交叉矩阵（原生输出画像×AIPL结构化数据）"""
        # 6维画像 × 7节点AIPL
        PERSONA_DIMENSIONS = ["WHO", "WHY", "WHAT", "WHEN", "HOW", "EMOTION"]
        AIPL_NODES = ["A", "I", "P1", "P2", "L1", "L2", "L3"]

        # 收集每个交叉格子的数据
        matrix: dict[str, dict[str, dict[str, Any]]] = defaultdict(
            lambda: defaultdict(lambda: {
                "count": 0,
                "sentiments": [],
                "themes": Counter(),
                "proxy_nps": {"promoter": 0, "passive": 0, "detractor": 0},
            })
        )

        for e in extractions:
            for dim in PERSONA_DIMENSIONS:
                dim_tags = e.persona_dimensions.get(dim, [])
                for tag in dim_tags:
                    cell = matrix[dim][e.aipl_stage]
                    cell["count"] += 1
                    cell["sentiments"].append(e.sentiment_polarity)
                    for t in e.aipl_tags:
                        cell["themes"][t.theme] += 1
                    # Proxy NPS 累计
                    if e.proxy_nps_contribution == "promoter":
                        cell["proxy_nps"]["promoter"] += 1
                    elif e.proxy_nps_contribution == "detractor":
                        cell["proxy_nps"]["detractor"] += 1
                    else:
                        cell["proxy_nps"]["passive"] += 1

        # 计算最终指标
        result: dict[str, Any] = {}
        total_voc = len(extractions) if extractions else 1

        for dim in PERSONA_DIMENSIONS:
            result[dim] = {}
            for node in AIPL_NODES:
                cell = matrix[dim][node]
                count = cell["count"]
                sentiments = cell["sentiments"]

                avg_sentiment = round(sum(sentiments) / len(sentiments), 2) if sentiments else 0.0
                mention_rate = round(count / total_voc, 3)

                # Proxy NPS
                p = cell["proxy_nps"]["promoter"]
                d = cell["proxy_nps"]["detractor"]
                n = cell["proxy_nps"]["passive"]
                total_cell = p + d + n
                proxy_nps = round((p / total_cell * 100) - (d / total_cell * 100), 1) if total_cell > 0 else 0.0

                # Top themes
                top_themes = [
                    {"theme": t, "count": c}
                    for t, c in cell["themes"].most_common(3)
                ]

                result[dim][node] = {
                    "count": count,
                    "mention_rate": mention_rate,
                    "avg_sentiment": avg_sentiment,
                    "proxy_nps": proxy_nps,
                    "top_themes": top_themes,
                }

        return result


# ---------------------------------------------------------------------------
# 11. 辅助函数
# ---------------------------------------------------------------------------

def _parse_list(value: str) -> list[str]:
    """解析列表字段（支持逗号、英文分号、中文分号分隔）"""
    if not value:
        return []
    # 统一分隔符为英文分号，再按分号/逗号分割
    normalized = value.replace("；", ";").replace(",", ";")
    return [s.strip() for s in normalized.split(";") if s.strip()]


def _map_sentiment_preset(value: str) -> str:
    """中文情感极性 → 英文映射"""
    mapping = {
        "正向": "positive",
        "负向": "negative",
        "中性": "neutral",
    }
    return mapping.get(str(value).strip(), str(value).strip())


def _map_applicable_line(value: str) -> list[str]:
    """解析适用产品品线，'通用'/'喂养电器通用' 视为空列表"""
    items = _parse_list(value)
    # 通用标签: 空列表表示不限品线
    if not items or items == ["通用"] or items == ["喂养电器通用"]:
        return []
    return items


# 关键词停用词 — 过于宽泛的单字/单词，容易导致误匹配
KEYWORD_STOP_WORDS: set[str] = {
    # 通用名词
    "product", "item", "order", "package", "box", "time", "day", "way",
    # 动词
    "use", "used", "using", "buy", "bought", "purchase", "purchased",
    "return", "returned", "get", "got", "like", "love", "hate", "want",
    "need", "think", "feel", "know", "see", "look", "work", "working",
    "worked", "help", "helped", "helping", "make", "made", "take", "took",
    "come", "came", "go", "went", "have", "had", "has", "do", "did",
    "does", "be", "been", "is", "are", "was", "were", "am",
    # 形容词/副词
    "more", "less", "wrong", "bad", "good", "great", "nice", "well",
    "really", "very", "quite", "pretty", "too", "so", "just", "only",
    # 其他
    "will", "would", "could", "should", "can", "may", "might", "must",
    "one", "two", "first", "second", "also", "even", "still", "now",
    "size", "color", "model", "type", "brand",
}


def _filter_keywords(keywords: list[str]) -> list[str]:
    """过滤过于宽泛的关键词，降低误匹配率

    规则:
    - 单停用词（如 'more', 'product'）→ 过滤
    - 多词短语中包含停用词（如 'wrong product'）→ 保留
    - 长度 < 3 字符的单词 → 过滤
    """
    filtered: list[str] = []
    for kw in keywords:
        kw_clean = kw.lower().strip()
        words = kw_clean.replace("-", " ").split()
        # 保留多词短语（即使包含停用词）
        if len(words) >= 2:
            filtered.append(kw)
            continue
        # 单词: 检查是否为停用词或过短
        if len(words) == 1:
            word = words[0]
            if word in KEYWORD_STOP_WORDS or len(word) < 3:
                continue
        filtered.append(kw)
    return filtered


# ---------------------------------------------------------------------------
# 12. 示例标签种子数据（内置，用于测试）
# ---------------------------------------------------------------------------

def create_demo_tag_dictionary() -> TagSeedDictionary:
    """创建演示用的标签字典（含代表性示例标签）"""
    dictionary = TagSeedDictionary()

    # A - 认知阶段
    dictionary.add(TagSeed(
        tag_id="TAG_A_001",
        tag_en="brand_search",
        tag_cn="品牌搜索",
        aipl_node="A",
        theme="品牌认知",
        sentiment_preset="neutral",
        keywords=["searching for", "looking for", "heard about", "saw ad"],
        consumer_keywords=["i was searching", "came across", "found on tiktok"],
        applicable_line=[],
        applicable_source=["ticket", "review"],
        nps_contribution="中性",
        story_line="品牌认知",
        strategy_pack="品牌曝光强化包",
        owner_dept="品牌营销部",
        priority="P1",
    ))

    dictionary.add(TagSeed(
        tag_id="TAG_A_002",
        tag_en="influencer_recommendation",
        tag_cn="KOL推荐",
        aipl_node="A",
        theme="品牌认知",
        sentiment_preset="positive",
        keywords=["influencer", "youtube", "tiktok", "blogger", "reviewer"],
        consumer_keywords=["saw on tiktok", "youtube review", "blogger said"],
        applicable_line=[],
        applicable_source=["review", "trustpilot"],
        nps_contribution="Promoter驱动",
        story_line="社媒种草",
        strategy_pack="KOL合作强化包",
        owner_dept="品牌营销部",
        priority="P1",
    ))

    # I - 兴趣阶段
    dictionary.add(TagSeed(
        tag_id="TAG_I_001",
        tag_en="product_comparison",
        tag_cn="产品对比",
        aipl_node="I",
        theme="产品对比",
        sentiment_preset="neutral",
        keywords=["compared", "vs", "versus", "better than", "difference"],
        consumer_keywords=["i compared", "looking at both", "deciding between"],
        applicable_line=[],
        applicable_source=["ticket", "review"],
        nps_contribution="中性",
        story_line="决策犹豫",
        strategy_pack="竞品对比强化包",
        owner_dept="产品运营部",
        priority="P1",
    ))

    dictionary.add(TagSeed(
        tag_id="TAG_I_002",
        tag_en="price_concern",
        tag_cn="价格关注",
        aipl_node="I",
        theme="价格价值感",
        sentiment_preset="neutral",
        keywords=["price", "expensive", "affordable", "cheap", "cost", "budget"],
        consumer_keywords=["too expensive", "worth the price", "overpriced", "good deal"],
        applicable_line=[],
        applicable_source=["ticket", "return_note", "review"],
        nps_contribution="中性",
        story_line="价格敏感",
        strategy_pack="价值感知提升包",
        owner_dept="定价策略部",
        priority="P1",
    ))

    # P1 - 购买1
    dictionary.add(TagSeed(
        tag_id="TAG_P1_001",
        tag_en="flange_size_issue",
        tag_cn="法兰尺码不合适",
        aipl_node="P1",
        theme="产品核心性能",
        sentiment_preset="negative",
        keywords=["flange", "size", "too small", "too big", "fit", "shield"],
        consumer_keywords=["flange hurts", "wrong size", "does not fit"],
        applicable_line=["breast_pump"],
        applicable_source=["return_note", "ticket", "review"],
        nps_contribution="Detractor驱动",
        story_line="核心体验问题",
        strategy_pack="核心体验改良包",
        owner_dept="产品研发部",
        priority="P0",
    ))

    # P2 - 购买2
    dictionary.add(TagSeed(
        tag_id="TAG_P2_001",
        tag_en="shipping_delay",
        tag_cn="物流延迟",
        aipl_node="P2",
        theme="物流时效",
        sentiment_preset="negative",
        keywords=["shipping", "delivery", "late", "delay", "took too long", "arrived"],
        consumer_keywords=["still not arrived", "shipping is slow", "where is my order"],
        applicable_line=[],
        applicable_source=["ticket", "review"],
        nps_contribution="Detractor驱动",
        story_line="履约体验",
        strategy_pack="物流履约优化包",
        owner_dept="供应链管理部",
        priority="P0",
    ))

    # L1 - 首用
    dictionary.add(TagSeed(
        tag_id="TAG_L1_001",
        tag_en="suction_too_weak",
        tag_cn="吸力差",
        aipl_node="L1",
        theme="产品核心性能",
        sentiment_preset="negative",
        keywords=["suction", "weak", "not strong enough", "low power", "does not pull"],
        consumer_keywords=["suction is weak", "not enough suction", "barely pumps"],
        applicable_line=["breast_pump"],
        applicable_source=["review", "ticket"],
        nps_contribution="Detractor驱动",
        story_line="核心体验问题",
        strategy_pack="核心体验改良包",
        owner_dept="产品研发部",
        priority="P0",
    ))

    dictionary.add(TagSeed(
        tag_id="TAG_L1_002",
        tag_en="too_noisy",
        tag_cn="噪音大",
        aipl_node="L1",
        theme="使用舒适度",
        sentiment_preset="negative",
        keywords=["noise", "noisy", "loud", "sound", "quiet"],
        consumer_keywords=["too loud", "wakes baby", "noisy motor", "not quiet"],
        applicable_line=["breast_pump", "feeding_appliance"],
        applicable_source=["review", "ticket"],
        nps_contribution="Detractor驱动",
        story_line="核心体验问题",
        strategy_pack="核心体验改良包",
        owner_dept="产品研发部",
        priority="P0",
    ))

    dictionary.add(TagSeed(
        tag_id="TAG_L1_003",
        tag_en="comfortable_wear",
        tag_cn="佩戴舒适",
        aipl_node="L1",
        theme="使用舒适度",
        sentiment_preset="positive",
        keywords=["comfortable", "soft", "fit well", "easy to wear", "lightweight"],
        consumer_keywords=["so comfortable", "forget i am wearing", "fits perfectly"],
        applicable_line=["breast_pump", "underwear"],
        applicable_source=["review", "trustpilot"],
        nps_contribution="Promoter驱动",
        story_line="核心体验亮点",
        strategy_pack="核心体验强化包",
        owner_dept="产品研发部",
        priority="P0",
    ))

    # L2 - 售后
    dictionary.add(TagSeed(
        tag_id="TAG_L2_001",
        tag_en="slow_customer_service",
        tag_cn="客服响应慢",
        aipl_node="L2",
        theme="问题解决效率",
        sentiment_preset="negative",
        keywords=["customer service", "support", "response", "slow", "wait"],
        consumer_keywords=["no response", "waiting for reply", "ignored"],
        applicable_line=[],
        applicable_source=["ticket", "review"],
        nps_contribution="Detractor驱动",
        story_line="服务体验",
        strategy_pack="服务体验优化包",
        owner_dept="客户服务部",
        priority="P0",
    ))

    # L3 - 推荐
    dictionary.add(TagSeed(
        tag_id="TAG_L3_001",
        tag_en="recommend_willingness",
        tag_cn="推荐意愿",
        aipl_node="L3",
        theme="推荐意愿",
        sentiment_preset="positive",
        keywords=["recommend", "suggest", "tell friends", "share", "spread the word"],
        consumer_keywords=["highly recommend", "would recommend", "told my friends"],
        applicable_line=[],
        applicable_source=["trustpilot", "review"],
        nps_contribution="Promoter驱动",
        story_line="口碑传播",
        strategy_pack="口碑激励包",
        owner_dept="品牌营销部",
        priority="P0",
    ))

    dictionary.add(TagSeed(
        tag_id="TAG_L3_002",
        tag_en="repurchase_intention",
        tag_cn="复购意愿",
        aipl_node="L3",
        theme="复购行为",
        sentiment_preset="positive",
        keywords=["buy again", "second one", "repurchase", "another one", "next time"],
        consumer_keywords=["will buy again", "getting a second", "ordered another"],
        applicable_line=[],
        applicable_source=["review", "trustpilot"],
        nps_contribution="Promoter驱动",
        story_line="忠诚复购",
        strategy_pack="会员忠诚包",
        owner_dept="用户运营部",
        priority="P1",
    ))

    return dictionary


# ---------------------------------------------------------------------------
# 13. 测试
# ---------------------------------------------------------------------------

def test_unified_extraction():
    print("=" * 70)
    print("测试: Unified VOC Label Extraction Engine")
    print("=" * 70)

    # 1. 创建演示标签字典
    print("\n--- 1. 标签字典 ---")
    tag_dict = create_demo_tag_dictionary()
    summary = tag_dict.summary()
    print(f"标签总数: {summary['total_tags']}")
    print(f"AIPL分布: {summary['by_aipl']}")
    print(f"主题数: {len(summary['themes'])}")

    # 2. 创建萃取器
    extractor = VOCLabelExtractor(tag_dict=tag_dict)

    # 3. 测试用例
    test_vocs = [
        VOCRecord(
            review_id="REV001",
            text=(
                "I was searching for a wearable pump and came across Momcozy on TikTok. "
                "Compared it with Willow and Elvie, the price is much more affordable. "
                "However, the flange size is too small and the suction feels weak. "
                "Customer service was slow to respond. Would not recommend to friends."
            ),
            source_type="review",
            platform="amazon",
            spu_code="SPU001",
            product_line="breast_pump",
            category="wearable_pump",
            rating=2.0,
        ),
        VOCRecord(
            review_id="REV002",
            text=(
                "This is my second purchase! The pump is so comfortable, "
                "I forget I'm wearing it. Highly recommend to all new moms. "
                "Way better than my old Spectra."
            ),
            source_type="trustpilot",
            platform="dtc",
            spu_code="SPU002",
            product_line="breast_pump",
            category="wearable_pump",
            rating=5.0,
        ),
        VOCRecord(
            review_id="REV003",
            text=(
                "First time mom here, feeling anxious about breastfeeding. "
                "Saw this on Instagram and decided to try. "
                "It's a bit noisy but gets the job done. "
                "Shipping took two weeks though."
            ),
            source_type="review",
            platform="amazon",
            spu_code="SPU003",
            product_line="breast_pump",
            category="wearable_pump",
            rating=4.0,
        ),
    ]

    # 4. 单条萃取测试
    print("\n--- 2. 单条萃取 ---")
    for voc in test_vocs:
        result = extractor.extract(voc)
        print(f"\n  [{voc.review_id}] {voc.text[:50]}...")
        print(f"    AIPL 阶段: {result.aipl_stage}")
        print(f"    AIPL 标签: {[t.tag_en for t in result.aipl_tags]}")
        print(f"    画像原子: {result.persona_atomic}")
        print(f"    画像推导: {result.persona_derived}")
        print(f"    情感极性: {result.sentiment_polarity:+.2f} ({result.sentiment_calibration})")
        print(f"    品牌提及: {result.brand_mentions}")
        print(f"    Proxy NPS: {result.proxy_nps_contribution}")
        print(f"    策略包: {result.strategy_pack or 'N/A'}")
        print(f"    主责部门: {result.owner_dept or 'N/A'}")

    # 5. 批量流水线测试
    print("\n--- 3. 批量流水线 ---")
    pipeline = UnifiedLabelingPipeline(tag_dict=tag_dict)
    results = pipeline.process(test_vocs)
    print(f"处理 {len(results)} 条 VOC")

    # 6. 看板生成
    print("\n--- 4. 指标看板 ---")
    dashboard = DashboardGenerator().build(results)
    data = dashboard.to_dict()

    print(f"\n  Proxy NPS (总体):")
    nps = data["proxy_nps"]["overall"]
    print(f"    得分: {nps['proxy_nps']:.1f}")
    print(f"    推荐者: {nps['promoters']} ({nps['promoter_pct']}%)")
    print(f"    贬损者: {nps['detractors']} ({nps['detractor_pct']}%)")

    print(f"\n  AIPL 漏斗:")
    for node, info in data["aipl_funnel"].items():
        if info["count"] > 0:
            themes = [t["theme"] for t in info["top_themes"]]
            print(f"    {node}: {info['count']} 条 - 主题: {themes}")

    print(f"\n  驱动因素:")
    for theme in data["driver_analysis"]["top_detractor_themes"][:3]:
        print(f"    [贬损] {theme['theme']}: 提及率{theme['mention_rate']:.1%}, 情感{theme['avg_sentiment']:+.2f}")

    print(f"\n  画像洞察:")
    for persona, info in data["persona_insights"].items():
        print(f"    {persona}: 渗透率{info['penetration']:.1%}, 情感{info['avg_sentiment']:+.2f}, NPS={info['proxy_nps']['proxy_nps']:.1f}")

    print(f"\n  标签覆盖:")
    coverage = data["tag_coverage"]
    print(f"    覆盖率: {coverage['coverage_rate']:.1%} ({coverage['matched_voc']}/{coverage['total_voc']})")
    print(f"    唯一标签命中: {coverage['unique_tags_matched']}")

    print(f"\n  品牌分析:")
    brand = data["brand_analysis"]
    print(f"    品牌提及: {brand['brand_mentions']}")
    print(f"    竞品对比率: {brand['comparison_rate']:.1%}")

    # 7. 验证
    print("\n--- 5. 验证 ---")
    assert len(results) == 3
    # REV001 应匹配到产品问题标签（P1/L1 阶段）
    assert results[0].aipl_stage in {"A", "I", "P1", "P2", "L1", "L2", "L3"}
    assert any(t.aipl_node in {"P1", "L1", "L2"} for t in results[0].aipl_tags), \
        f"REV001 应包含产品/售后问题标签，实际: {[t.tag_en for t in results[0].aipl_tags]}"
    # REV002 是推荐者
    assert results[1].proxy_nps_contribution == "promoter"
    assert len(results[1].persona_atomic) > 0
    # REV001 的 "not recommend" 不应被标记为 recommend 意愿
    assert not any(t.aipl_node == "L3" for t in results[0].aipl_tags), \
        f"REV001 不应包含 L3 推荐标签（因为 not recommend），实际: {[t.tag_en for t in results[0].aipl_tags]}"
    print("✓ 所有验证通过")

    # 8. 序列化测试
    print("\n--- 6. JSON 序列化 ---")
    dashboard.to_json("/tmp/voc_dashboard_demo.json")
    print("✓ 看板数据已保存到 /tmp/voc_dashboard_demo.json")

    print("\n" + "=" * 70)
    print("统一标签萃取引擎测试完成 ✓")
    print("=" * 70)

    return results, dashboard


if __name__ == "__main__":
    test_unified_extraction()
