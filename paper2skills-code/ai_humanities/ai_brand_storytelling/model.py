"""
AI Brand Storytelling — AI 辅助品牌故事创作：情感连接与文化适应
paper2skills-code: 11-AI人文 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Optional


# ──────────────────────────────────────────────
# 数据类：叙事结构
# ──────────────────────────────────────────────

@dataclass
class NarrativeStructure:
    """品牌故事四段情感弧线结构"""
    hook: str
    conflict: str
    resolution: str
    cta: str
    market: str = "US"
    emotional_arc: list[str] = field(default_factory=lambda: ["担忧", "探索", "信任", "行动"])

    def render(self) -> str:
        return f"{self.hook}\n\n{self.conflict}\n\n{self.resolution}\n\n{self.cta}"

    def word_count(self) -> int:
        return len(self.render().split())


# ──────────────────────────────────────────────
# 文化适配器：US / EU / JP / CN 价值观规则
# ──────────────────────────────────────────────

CULTURAL_RULES: dict[str, dict] = {
    "US": {
        "core_values": ["个人选择", "科学支持", "透明度"],
        "hook_template": "当你第一次看到宝宝配方罐上的成分表，是否也曾困惑……",
        "conflict_template": "市面上 {feature_count} 种配方，哪个成分表才真正透明？",
        "resolution_template": "{certifications} 认证 + {key_feature}，让选择有据可依。",
        "cta_template": "查看完整成分溯源",
        "avoid": ["权威指令", "你必须", "所有妈妈都应该"],
        "emphasis_keywords": ["FDA", "independent testing", "transparency", "your choice"],
    },
    "EU": {
        "core_values": ["自然", "可持续", "监管合规"],
        "hook_template": "自然来自土地，安心传给宝宝。",
        "conflict_template": "工业化农业 vs 有机认证：真正的区别，藏在这里。",
        "resolution_template": "{certifications} + {key_feature}，严格超越欧盟标准。",
        "cta_template": "了解我们的有机认证",
        "avoid": ["夸大功效", "最好的", "无可匹敌"],
        "emphasis_keywords": ["EU Organic", "sustainable", "traceable", "regulated"],
    },
    "JP": {
        "core_values": ["精准", "安心", "集体认同"],
        "hook_template": "{age_range} 是宝宝味觉发育的关键窗口。",
        "conflict_template": "成长阶段的细微差异，决定了不同的营养需求配比。",
        "resolution_template": "精确到月龄的配方设计，参考 {certifications} 品质标准。",
        "cta_template": "按月龄查找适合的配方",
        "avoid": ["个性化强调", "与众不同", "激进营销"],
        "emphasis_keywords": ["精密", "安心", "品质管理", "段阶配方"],
    },
    "CN": {
        "core_values": ["权威背书", "营养科学", "阶段匹配"],
        "hook_template": "科学育儿，从选对每个阶段的营养开始。",
        "conflict_template": "奶粉成分复杂，{certifications} 认证才是可靠的底线。",
        "resolution_template": "儿科营养学参考标准 + {key_feature}，给宝宝科学的营养支持。",
        "cta_template": "查看科学配方详情",
        "avoid": ["情感虚化", "模糊表达"],
        "emphasis_keywords": ["科学配方", "营养均衡", "阶段匹配", "专业认可"],
    },
}


class CulturalAdapter:
    """根据目标市场文化价值观适配品牌叙事框架"""

    SUPPORTED_MARKETS = list(CULTURAL_RULES.keys())

    def get_rules(self, market: str) -> dict:
        if market not in CULTURAL_RULES:
            raise ValueError(f"不支持的市场: {market}，支持: {self.SUPPORTED_MARKETS}")
        return CULTURAL_RULES[market]

    def adapt(
        self,
        product_info: dict,
        market: str,
    ) -> NarrativeStructure:
        rules = self.get_rules(market)
        certs = "/".join(product_info.get("certifications", ["认证"]))
        feature_count = product_info.get("feature_count", "200+")
        key_feature = product_info.get("key_feature", "全程溯源系统")
        age_range = product_info.get("age_range", "6-12 个月")

        hook = rules["hook_template"].format(
            age_range=age_range,
            certifications=certs,
        )
        conflict = rules["conflict_template"].format(
            feature_count=feature_count,
            certifications=certs,
        )
        resolution = rules["resolution_template"].format(
            certifications=certs,
            key_feature=key_feature,
        )
        cta = rules["cta_template"]

        return NarrativeStructure(
            hook=hook,
            conflict=conflict,
            resolution=resolution,
            cta=cta,
            market=market,
        )

    def check_avoid_words(self, text: str, market: str) -> list[str]:
        rules = self.get_rules(market)
        triggered = [word for word in rules["avoid"] if word in text]
        return triggered


# ──────────────────────────────────────────────
# 真实性评分器：检测 AI "过度抛光"
# ──────────────────────────────────────────────

EMOTIONAL_WORDS = [
    "最好", "完美", "无与伦比", "卓越", "绝佳", "最优质", "最纯净",
    "最安全", "最放心", "无可挑剔", "最值得信赖",
    "perfect", "best", "outstanding", "superior", "ultimate",
    "unmatched", "exceptional", "flawless",
]


class AuthenticityScorer:
    """
    检测 AI 生成文本的"过度抛光"程度，输出真实性评分 [0, 1]。

    评分维度：
    1. 情感词密度（emotional density）— 占比超过 15% 扣分
    2. 句式长度分布方差（length variance）— 方差过低（均匀）扣分
    3. 词汇重复率（repetition）— 高频词重复扣分
    """

    EMOTIONAL_THRESHOLD = 0.12
    LENGTH_VARIANCE_MIN = 5.0

    def score(self, text: str) -> float:
        sentences = [s.strip() for s in re.split(r"[。！？\.\!\?]", text) if s.strip()]
        if not sentences:
            return 0.0

        words = text.split()
        total_words = max(len(words), 1)

        emotional_count = sum(1 for w in EMOTIONAL_WORDS if w in text)
        emotional_density = emotional_count / total_words
        emotional_score = max(0.0, 1.0 - emotional_density / self.EMOTIONAL_THRESHOLD)

        lengths = [len(s.split()) for s in sentences]
        if len(lengths) > 1:
            mean_len = sum(lengths) / len(lengths)
            variance = sum((l - mean_len) ** 2 for l in lengths) / len(lengths)
            length_score = min(1.0, variance / self.LENGTH_VARIANCE_MIN)
        else:
            length_score = 0.5

        word_freq: dict[str, int] = {}
        for w in words:
            w_lower = w.lower().strip("，。！？,.!?")
            word_freq[w_lower] = word_freq.get(w_lower, 0) + 1
        max_freq = max(word_freq.values()) if word_freq else 1
        repetition_score = max(0.0, 1.0 - (max_freq - 1) / max(total_words * 0.1, 1))

        final_score = emotional_score * 0.4 + length_score * 0.35 + repetition_score * 0.25
        return round(min(1.0, max(0.0, final_score)), 3)

    def is_authentic(self, text: str, threshold: float = 0.7) -> bool:
        return self.score(text) >= threshold


# ──────────────────────────────────────────────
# 品牌故事生成器
# ──────────────────────────────────────────────

class BrandStoryGenerator:
    """
    结合产品信息 + 文化规则生成品牌故事，
    并通过 AuthenticityScorer 验证真实性。
    """

    def __init__(self, authenticity_threshold: float = 0.7) -> None:
        self._adapter = CulturalAdapter()
        self._scorer = AuthenticityScorer()
        self._threshold = authenticity_threshold

    def generate(
        self,
        product_info: dict,
        market: str = "US",
        max_retries: int = 3,
    ) -> NarrativeStructure:
        story = self._adapter.adapt(product_info, market)
        score = self._scorer.score(story.render())

        if score < self._threshold and max_retries > 0:
            enriched = dict(product_info)
            enriched["key_feature"] = enriched.get("key_feature", "") + "（独立检测报告）"
            story = self._adapter.adapt(enriched, market)

        return story

    def generate_multi_market(
        self,
        product_info: dict,
        markets: Optional[list[str]] = None,
    ) -> dict[str, NarrativeStructure]:
        if markets is None:
            markets = CulturalAdapter.SUPPORTED_MARKETS
        return {m: self.generate(product_info, market=m) for m in markets}


# ──────────────────────────────────────────────
# 测试：婴儿奶粉场景 3 个市场版本
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("AI Brand Storytelling — 品牌故事生成测试")
    print("=" * 60)

    product_info = {
        "name": "WF-B Stage 2 Organic Formula",
        "certifications": ["FDA", "EU Organic", "Non-GMO"],
        "key_feature": "12 项成分溯源平台",
        "age_range": "6-12 个月",
        "feature_count": "200+",
    }

    generator = BrandStoryGenerator(authenticity_threshold=0.5)
    scorer = AuthenticityScorer()

    markets = ["US", "EU", "JP"]
    results = {}

    for market in markets:
        story = generator.generate(product_info, market=market)
        text = story.render()
        auth_score = scorer.score(text)
        results[market] = {"story": story, "score": auth_score}

        print(f"\n[{market}] 市场版本")
        print(f"  钩子: {story.hook}")
        print(f"  冲突: {story.conflict}")
        print(f"  解决: {story.resolution}")
        print(f"  CTA:  {story.cta}")
        print(f"  真实性评分: {auth_score:.3f}")

    us_story = results["US"]["story"]
    eu_story = results["EU"]["story"]
    jp_story = results["JP"]["story"]

    assert us_story.hook != eu_story.hook, "US 和 EU 钩子应不同"
    assert us_story.hook != jp_story.hook, "US 和 JP 钩子应不同"
    assert eu_story.cta != jp_story.cta, "EU 和 JP CTA 应不同"
    print("\n[✓] 3 个市场版本文化适配差异验证通过")

    adapter = CulturalAdapter()
    avoid_result = adapter.check_avoid_words("这款奶粉无与伦比，所有妈妈都应该选择", "US")
    assert len(avoid_result) > 0, "应检测到 US 市场禁用词"
    print(f"[✓] 禁用词检测: {avoid_result}")

    assert scorer.is_authentic("平凡的奶粉，普通的一天，宝宝吃了。", threshold=0.5), \
        "普通文本应通过真实性检查"

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — AI Brand Storytelling")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
