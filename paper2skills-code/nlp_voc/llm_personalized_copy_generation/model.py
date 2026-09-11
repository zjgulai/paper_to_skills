"""
LLM-Driven Personalized Marketing Copy Generation
LLM 驱动个性化营销文案生成 - 母婴出海跨境电商场景

论文来源:
1. LLM-Driven E-Commerce Marketing Content Optimization
   arXiv: 2505.23809, Haowei Yang, 2025
2. LLMs for Customized Marketing Content Generation and Evaluation at Scale
   arXiv: 2506.17863, MarketingFM, 2025

核心框架: 可控属性 Prompt Engineering → 多目标生成 → 后处理筛选
"""

from __future__ import annotations

import random
import re
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from collections import Counter


# =============================================================================
# 数据模型
# =============================================================================

@dataclass
class UserPersona:
    """用户画像"""
    persona_id: str
    name: str
    segment: str
    core_needs: List[str]
    pain_points: List[str]
    decision_factors: List[str]
    preferred_tone: str
    language: str = "zh"


@dataclass
class ProductFeature:
    """产品特性"""
    feature_id: str
    name: str
    description: str
    benefit: str
    keywords: List[str]


@dataclass
class ProductProfile:
    """产品档案"""
    product_id: str
    name: str
    category: str
    features: List[ProductFeature]
    price_tier: str
    target_market: str


@dataclass
class CopyAttributes:
    """文案可控属性"""
    tone: str = "professional"
    length: str = "medium"
    language: str = "zh"
    cta_type: str = "buy_now"
    emoji_level: str = "moderate"
    urgency_level: str = "low"

    def validate(self):
        assert self.tone in {"professional", "friendly", "urgent", "warm"}
        assert self.length in {"short", "medium", "long"}
        assert self.language in {"zh", "en", "es", "ja"}
        assert self.cta_type in {"buy_now", "learn_more", "limited_offer", "join_community"}
        assert self.emoji_level in {"none", "moderate", "heavy"}
        assert self.urgency_level in {"low", "medium", "high"}


@dataclass
class GeneratedCopy:
    """生成的文案"""
    copy_id: str
    headline: str
    body: str
    cta: str
    attributes: CopyAttributes
    scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0


# =============================================================================
# Prompt 构建器
# =============================================================================

class PromptBuilder:
    """结构化 Prompt 构建器"""

    TONE_INSTRUCTIONS = {
        "professional": "使用专业、可信的语气，强调产品认证和科学依据",
        "friendly": "使用亲切、对话式的语气，像朋友推荐一样",
        "urgent": "使用紧迫、促行动的语气，强调限时和稀缺",
        "warm": "使用温暖、关怀的语气，强调对妈妈和宝宝的呵护",
    }

    LENGTH_INSTRUCTIONS = {
        "short": "文案控制在30-50字，精炼有力",
        "medium": "文案控制在80-120字，信息完整",
        "long": "文案控制在150-200字，详细展开",
    }

    CTA_TEMPLATES = {
        "zh": {
            "buy_now": "立即购买",
            "learn_more": "了解更多",
            "limited_offer": "限时特惠",
            "join_community": "加入妈妈社群",
        },
        "en": {
            "buy_now": "Shop Now",
            "learn_more": "Learn More",
            "limited_offer": "Limited Offer",
            "join_community": "Join Our Community",
        },
        "es": {
            "buy_now": "Comprar Ahora",
            "learn_more": "Saber Más",
            "limited_offer": "Oferta Limitada",
            "join_community": "Únete a la Comunidad",
        },
        "ja": {
            "buy_now": "今すぐ購入",
            "learn_more": "詳細を見る",
            "limited_offer": "期間限定",
            "join_community": "コミュニティに参加",
        },
    }

    URGENCY_PHRASES = {
        "zh": {"low": [], "medium": ["限时优惠"], "high": ["仅限今日", "库存告急"]},
        "en": {"low": [], "medium": ["Limited Time"], "high": ["Today Only", "Selling Fast"]},
    }

    EMOJI_MAP = {
        "zh": {"baby": "👶", "heart": "❤️", "star": "⭐", "fire": "🔥", "gift": "🎁"},
        "en": {"baby": "👶", "heart": "❤️", "star": "⭐", "fire": "🔥", "gift": "🎁"},
    }

    @classmethod
    def build(
        cls,
        persona: UserPersona,
        product: ProductProfile,
        attributes: CopyAttributes,
    ) -> str:
        """构建结构化生成 Prompt"""

        # 角色设定
        role = f"你是一位专业的跨境电商文案策划，擅长为{product.target_market}市场撰写营销文案。"

        # 用户画像上下文
        user_ctx = (
            f"目标用户: {persona.name} ({persona.segment})\n"
            f"核心需求: {', '.join(persona.core_needs)}\n"
            f"痛点: {', '.join(persona.pain_points)}\n"
            f"决策因素: {', '.join(persona.decision_factors)}"
        )

        # 产品上下文
        features_text = "\n".join(
            f"- {f.name}: {f.description} (益处: {f.benefit})"
            for f in product.features
        )
        product_ctx = (
            f"产品: {product.name}\n"
            f"品类: {product.category}\n"
            f"价格带: {product.price_tier}\n"
            f"核心特性:\n{features_text}"
        )

        # 风格控制
        style_ctx = (
            f"语气: {cls.TONE_INSTRUCTIONS.get(attributes.tone, attributes.tone)}\n"
            f"长度: {cls.LENGTH_INSTRUCTIONS.get(attributes.length, attributes.length)}\n"
            f"语言: {attributes.language}\n"
            f"CTA类型: {attributes.cta_type}\n"
            f"紧迫度: {attributes.urgency_level}"
        )

        # 输出格式
        output_format = (
            "请按以下格式输出:\n"
            "[标题] 一句吸引眼球的标题\n"
            "[正文] 2-3句产品描述，针对目标用户痛点和需求\n"
            "[CTA] 行动号召语"
        )

        return f"{role}\n\n{user_ctx}\n\n{product_ctx}\n\n{style_ctx}\n\n{output_format}"


# =============================================================================
# Mock LLM 生成器 (模板驱动)
# =============================================================================

class MockLLMGenerator:
    """
    Mock LLM 文案生成器

    实际部署应替换为 OpenAI / Claude / 本地 LLM API 调用。
    本实现使用模板 + 属性控制演示核心 pipeline。
    """

    # 按品类和画像预设模板库
    TEMPLATE_LIBRARY = {
        "吸奶器": {
            "职场背奶妈妈": {
                "headlines": [
                    "{emoji}10分钟高效背奶，职场妈妈的时间救星",
                    "{emoji}静音不尴尬，办公室背奶也能从容",
                    "{emoji}吸力强劲续航持久，背奶妈妈的可靠搭档",
                ],
                "bodies": {
                    "professional": "专为职场妈妈设计的智能吸奶系统。{feature1}，{feature2}，{feature3}。让背奶成为高效从容的日常。",
                    "friendly": "姐妹们！这个吸奶器真的太懂我们了！{feature1}，午休时间就能搞定。{feature2}，隔壁工位完全听不到～{feature3}，出差一周不用愁！",
                    "warm": "每一位背奶妈妈都值得被温柔以待。{feature1}，让你多陪宝宝一会。{feature2}，守护你的职场体面。{feature3}，安心工作每一天。",
                    "urgent": "{urgency}还在用手动吸奶器浪费时间？{feature1}，{feature2}，{feature3}。升级你的背奶装备，现在就行动！",
                },
            },
            "新手妈妈": {
                "headlines": [
                    "{emoji}第一次喂奶不慌张，新手妈妈的安心选择",
                    "{emoji}操作简单零学习，生完就能用的吸奶器",
                    "{emoji}温和吸力保护乳腺，新手妈妈的第一台吸奶器",
                ],
                "bodies": {
                    "professional": "经儿科医生推荐的入门款吸奶器。{feature1}，{feature2}，{feature3}。新手也能轻松上手。",
                    "friendly": "刚当妈妈手忙脚乱？这个吸奶器超简单！{feature1}，完全不会痛～{feature2}，婆婆也能帮你操作。{feature3}，有问题随时问！",
                    "warm": "初为人母，一切都值得最好的开始。{feature1}，温柔呵护你。{feature2}，让你少一分焦虑。{feature3}，我们陪你一起走过。",
                    "urgent": "{urgency}新手妈妈别走弯路！{feature1}，{feature2}，{feature3}。 thousands of 新手妈妈的选择，别再犹豫了！",
                },
            },
            "价格敏感妈妈": {
                "headlines": [
                    "{emoji}千元品质百元价格，高性价比吸奶器",
                    "{emoji}大牌平替！功能不打折的平价吸奶器",
                    "{emoji}花小钱办大事，精明妈妈的选择",
                ],
                "bodies": {
                    "professional": "同厂同源的高品质吸奶器。{feature1}，{feature2}，{feature3}。品质不打折，价格更亲民。",
                    "friendly": "谁说便宜没好货？这个吸奶器性价比绝了！{feature1}，跟大牌一样好用。{feature2}，{feature3}。省钱麻麻必入！",
                    "warm": "养娃已经很费钱了，吸奶器不该是负担。{feature1}，{feature2}，{feature3}。每一分钱都花在刀刃上。",
                    "urgent": "{urgency}大牌平替库存有限！{feature1}，{feature2}，{feature3}。错过这波就要等双11了！",
                },
            },
        },
        "纸尿裤": {
            "新手妈妈": {
                "headlines": [
                    "{emoji}一夜一片不漏尿，新手妈妈终于能睡整觉",
                    "{emoji}0.2cm超薄透气，宝宝屁屁不再红",
                    "{emoji}医护级标准，给新生儿最安心的保护",
                ],
                "bodies": {
                    "professional": "通过皮肤科安全测试的婴儿纸尿裤。{feature1}，{feature2}，{feature3}。给新生儿最安心的呵护。",
                    "friendly": "终于找到宝宝不哭的纸尿裤！{feature1}，一觉到天亮～{feature2}，夏天也不闷。{feature3}，新手妈妈闭眼入！",
                    "warm": "宝宝的每一寸肌肤都值得温柔守护。{feature1}，{feature2}，{feature3}。给你和宝宝最好的开始。",
                    "urgent": "{urgency}红屁屁反复发作？{feature1}，{feature2}，{feature3}。 thousands of 妈妈验证，今晚就用上！",
                },
            },
            "价格敏感妈妈": {
                "headlines": [
                    "{emoji}单片不到1元，性价比之王纸尿裤",
                    "{emoji}大牌品质工厂价，精明妈妈囤货首选",
                    "{emoji}月省200元，好用不贵的国货纸尿裤",
                ],
                "bodies": {
                    "professional": "国产高品质纸尿裤。{feature1}，{feature2}，{feature3}。省钱又省心，品质不打折。",
                    "friendly": "养娃太费钱了！这个纸尿裤帮我省了一大笔～{feature1}，跟大牌一样。{feature2}，{feature3}。按箱囤就对了！",
                    "warm": "爱宝宝，也要爱自己。{feature1}，{feature2}，{feature3}。每一包都是满满的爱。",
                    "urgent": "{urgency}限时工厂直供价！{feature1}，{feature2}，{feature3}。下个月就要涨价了，现在囤最划算！",
                },
            },
        },
        "辅食机": {
            "经验妈妈": {
                "headlines": [
                    "{emoji}一机搞定全阶段辅食，二胎妈妈省心之选",
                    "{emoji}从泥到颗粒一键切换，辅食添加不踩坑",
                    "{emoji}FDA认证材质，经验妈妈认可的辅食神器",
                ],
                "bodies": {
                    "professional": "经FDA认证的多功能辅食机。{feature1}，{feature2}，{feature3}。覆盖4-36个月全阶段，省时省力。",
                    "friendly": "二胎了才知道辅食机有多重要！{feature1}，不用买一堆工具。{feature2}，宝宝爱吃。{feature3}，本懒人妈妈狂喜！",
                    "warm": "给宝宝的第一口辅食，值得最好的准备。{feature1}，{feature2}，{feature3}。把时间留给陪伴，把辅食交给我们。",
                    "urgent": "{urgency}辅食添加黄金期别错过！{feature1}，{feature2}，{feature3}。现在入手送辅食食谱！",
                },
            },
        },
    }

    def __init__(self, seed: int = 42):
        random.seed(seed)

    def _select_template(
        self, persona: UserPersona, product: ProductProfile
    ) -> Tuple[List[str], Dict[str, str]]:
        """选择匹配的模板"""
        category_templates = self.TEMPLATE_LIBRARY.get(product.category, {})

        # 尝试精确匹配画像
        persona_templates = category_templates.get(persona.segment)

        # _fallback: 使用第一个可用画像
        if persona_templates is None:
            if category_templates:
                persona_templates = next(iter(category_templates.values()))
            else:
                # 通用 fallback
                return (
                    ["{emoji}高品质" + product.name + "，妈妈的放心之选"],
                    {
                        "professional": "{feature1}，{feature2}，{feature3}。品质之选，值得信赖。",
                        "friendly": "这个" + product.name + "真的太好用了！{feature1}，{feature2}，{feature3}。强烈推荐！",
                        "warm": "给宝宝最好的，{feature1}，{feature2}，{feature3}。爱与品质同行。",
                        "urgent": "{urgency}别犹豫了！{feature1}，{feature2}，{feature3}。限时优惠，手慢无！",
                    },
                )

        return persona_templates["headlines"], persona_templates["bodies"]

    def _fill_features(self, template: str, features: List[ProductFeature]) -> str:
        """用产品特性填充模板变量"""
        result = template
        for i, f in enumerate(features[:3], 1):
            result = result.replace(f"{{feature{i}}}", f.benefit)
        # 未填充的用备用特性
        for i in range(len(features) + 1, 4):
            fallback = features[0].benefit if features else "品质保证"
            result = result.replace(f"{{feature{i}}}", fallback)
        return result

    def _apply_emoji(self, text: str, level: str, lang: str) -> str:
        """应用表情符号"""
        if level == "none":
            return text.replace("{emoji} ", "").replace("{emoji}", "")

        emojis = PromptBuilder.EMOJI_MAP.get(lang, PromptBuilder.EMOJI_MAP["zh"])
        emoji_str = emojis["baby"] if "baby" in emojis else ""
        if level == "heavy":
            emoji_str += emojis.get("heart", "") + emojis.get("star", "")

        return text.replace("{emoji}", emoji_str)

    def _apply_urgency(self, text: str, level: str, lang: str) -> str:
        """应用紧迫短语"""
        phrases = PromptBuilder.URGENCY_PHRASES.get(lang, PromptBuilder.URGENCY_PHRASES["zh"])
        level_phrases = phrases.get(level, [])

        if not level_phrases:
            return text.replace("{urgency}", "")

        urgency_str = random.choice(level_phrases) + "！"
        return text.replace("{urgency}", urgency_str)

    def _adjust_length(self, text: str, target_length: str) -> str:
        """调整文案长度"""
        if target_length == "short":
            # 取前1-2句，保留实质性内容
            sentences = [s.strip() for s in re.split(r"[。！？]", text) if s.strip()]
            if not sentences:
                return text
            # 优先保留包含数字或特性描述的句子
            result = sentences[0]
            if len(result) < 20 and len(sentences) > 1:
                result += "。" + sentences[1]
            return result + "。" if not result.endswith("。") else result
        elif target_length == "long":
            # 添加补充说明
            extensions = [
                "无论是自用还是送礼，都是不二之选。",
                " thousands of 妈妈验证的好品质，值得信赖。",
                "现在下单享受专属优惠，让好品质触手可及。",
            ]
            return text + random.choice(extensions)
        return text

    def generate(
        self,
        persona: UserPersona,
        product: ProductProfile,
        attributes: CopyAttributes,
        n_candidates: int = 3,
    ) -> List[GeneratedCopy]:
        """
        生成多个文案候选

        Args:
            persona: 目标用户画像
            product: 产品档案
            attributes: 文案可控属性
            n_candidates: 生成候选数量

        Returns:
            List[GeneratedCopy]: 文案候选列表
        """
        headlines, bodies = self._select_template(persona, product)

        candidates = []
        for i in range(n_candidates):
            # 随机选择模板变体
            headline_tpl = random.choice(headlines)
            body_tpl = bodies.get(attributes.tone, bodies.get("friendly", ""))

            # 填充特性
            headline = self._fill_features(headline_tpl, product.features)
            body = self._fill_features(body_tpl, product.features)

            # 应用属性控制（先调整长度，再加紧迫感，避免短文案只剩 urgency）
            headline = self._apply_emoji(headline, attributes.emoji_level, attributes.language)
            body = self._apply_emoji(body, attributes.emoji_level, attributes.language)
            body = self._adjust_length(body, attributes.length)
            body = self._apply_urgency(body, attributes.urgency_level, attributes.language)

            # 生成 CTA
            cta = PromptBuilder.CTA_TEMPLATES.get(
                attributes.language, PromptBuilder.CTA_TEMPLATES["zh"]
            ).get(attributes.cta_type, "立即购买")

            candidates.append(
                GeneratedCopy(
                    copy_id=f"copy_{persona.persona_id}_{i+1}",
                    headline=headline,
                    body=body,
                    cta=cta,
                    attributes=attributes,
                )
            )

        return candidates


# =============================================================================
# 多目标评估器
# =============================================================================

class CopyEvaluator:
    """
    文案多目标评估器

    评估维度:
    1. Persona Relevance: 文案与用户画像需求的匹配度
    2. Feature Coverage: 产品特性覆盖度
    3. CTA Effectiveness: 行动号召有效性
    4. Diversity: 文案多样性（用于候选间对比）
    """

    def __init__(self):
        self.stopwords = {"的", "了", "是", "在", "和", "有", "为", "与", "及"}

    def _tokenize(self, text: str) -> List[str]:
        """简单分词（基于2-3字符滑动窗口）"""
        text = re.sub(r"[^\u4e00-\u9fff\w\s]", " ", text)
        tokens = []
        # 中文: 2-gram
        for i in range(len(text) - 1):
            if "\u4e00" <= text[i] <= "\u9fff":
                tokens.append(text[i : i + 2])
        # 英文/数字: 按词分割
        tokens.extend(re.findall(r"[a-zA-Z]{2,}", text))
        return [t for t in tokens if t not in self.stopwords]

    def _keyword_match_score(self, keyword: str, text: str) -> float:
        """关键词匹配度: 支持中文逐字匹配和英文整词匹配"""
        text_lower = text.lower()
        kw_lower = keyword.lower()

        # 英文: 整词匹配
        if all(ord(c) < 128 for c in keyword):
            return 1.0 if kw_lower in text_lower else 0.0

        # 中文: 逐字匹配，要求至少一半的字出现
        chars = [c for c in kw_lower if "\u4e00" <= c <= "\u9fff"]
        if not chars:
            return 1.0 if kw_lower in text_lower else 0.0

        matched = sum(1 for c in chars if c in text_lower)
        return 1.0 if matched >= max(1, len(chars) // 2) else 0.0

    def persona_relevance(
        self, copy: GeneratedCopy, persona: UserPersona
    ) -> float:
        """
        画像相关性: 文案覆盖用户核心需求和痛点的程度

        Score = (匹配的需求数 + 匹配的痛点数 + 匹配的决策因素) / 总关键词数
        """
        copy_text = (copy.headline + " " + copy.body).lower()
        all_keywords = persona.core_needs + persona.pain_points + persona.decision_factors

        if not all_keywords:
            return 0.5

        matches = 0
        for kw in all_keywords:
            if self._keyword_match_score(kw, copy_text) >= 0.5:
                matches += 1

        return matches / len(all_keywords)

    def feature_coverage(
        self, copy: GeneratedCopy, product: ProductProfile
    ) -> float:
        """
        特性覆盖度: 文案提及的产品特性比例

        Score = 被提及特性的 keyword 匹配数 / 总特性数
        """
        copy_text = (copy.headline + " " + copy.body).lower()

        if not product.features:
            return 0.5

        matched = 0
        for feat in product.features:
            for kw in feat.keywords:
                if kw.lower() in copy_text:
                    matched += 1
                    break

        return matched / len(product.features)

    def cta_effectiveness(self, copy: GeneratedCopy) -> float:
        """
        CTA 有效性评分

        检查:
        - CTA 是否非空
        - 是否包含行动动词
        - 紧迫感表达
        """
        if not copy.cta:
            return 0.0

        score = 0.3  # 基础分

        # 行动动词
        action_verbs = {"购买", "了解", "加入", "shop", "buy", "learn", "join", "comprar", "購入"}
        if any(v in copy.cta.lower() for v in action_verbs):
            score += 0.3

        # 紧迫感
        urgency_words = {"限时", "特惠", "limited", "oferta", "限定"}
        if any(w in copy.cta.lower() for w in urgency_words):
            score += 0.2

        # CTA 与文案连贯性
        if copy.cta.lower() in (copy.headline + copy.body).lower():
            score += 0.2

        return min(score, 1.0)

    def diversity_score(self, candidates: List[GeneratedCopy]) -> List[float]:
        """
        多样性评分: 每个候选与其他候选的差异度

        基于 n-gram Jaccard 距离计算 pairwise diversity
        """
        token_sets = []
        for c in candidates:
            tokens = self._tokenize(c.headline + " " + c.body)
            token_sets.append(set(tokens))

        scores = []
        for i, tokens_i in enumerate(token_sets):
            diversities = []
            for j, tokens_j in enumerate(token_sets):
                if i == j:
                    continue
                union = tokens_i | tokens_j
                if union:
                    jaccard_dist = 1 - len(tokens_i & tokens_j) / len(union)
                    diversities.append(jaccard_dist)

            avg_div = sum(diversities) / len(diversities) if diversities else 0.5
            scores.append(min(avg_div * 2, 1.0))  # 放大到 0-1

        return scores

    def evaluate(
        self,
        candidates: List[GeneratedCopy],
        persona: UserPersona,
        product: ProductProfile,
        weights: Optional[Dict[str, float]] = None,
    ) -> List[GeneratedCopy]:
        """
        多目标综合评估

        Args:
            candidates: 文案候选列表
            persona: 目标用户画像
            product: 产品档案
            weights: 各维度权重 {"relevance": 0.35, "coverage": 0.3, "cta": 0.2, "diversity": 0.15}

        Returns:
            评分后的候选列表（按 overall_score 排序）
        """
        if weights is None:
            weights = {"relevance": 0.35, "coverage": 0.3, "cta": 0.2, "diversity": 0.15}

        div_scores = self.diversity_score(candidates)

        for i, copy in enumerate(candidates):
            rel = self.persona_relevance(copy, persona)
            cov = self.feature_coverage(copy, product)
            cta = self.cta_effectiveness(copy)
            div = div_scores[i]

            overall = (
                weights["relevance"] * rel
                + weights["coverage"] * cov
                + weights["cta"] * cta
                + weights["diversity"] * div
            )

            copy.scores = {
                "relevance": round(rel, 3),
                "coverage": round(cov, 3),
                "cta": round(cta, 3),
                "diversity": round(div, 3),
            }
            copy.overall_score = round(overall, 3)

        return sorted(candidates, key=lambda x: x.overall_score, reverse=True)


# =============================================================================
# 文案生成 pipeline 主类
# =============================================================================

class MarketingCopyPipeline:
    """
    LLM 驱动个性化营销文案生成 Pipeline

    完整流程: 画像输入 → 属性配置 → Prompt 构建 → 多候选生成 → 多目标评估 → 最优选择
    """

    def __init__(self):
        self.prompt_builder = PromptBuilder()
        self.generator = MockLLMGenerator()
        self.evaluator = CopyEvaluator()

    def generate(
        self,
        persona: UserPersona,
        product: ProductProfile,
        attributes: CopyAttributes,
        n_candidates: int = 3,
        weights: Optional[Dict[str, float]] = None,
    ) -> Tuple[GeneratedCopy, List[GeneratedCopy], str]:
        """
        执行完整文案生成流程

        Args:
            persona: 目标用户画像
            product: 产品档案
            attributes: 文案可控属性
            n_candidates: 候选数量
            weights: 评估权重

        Returns:
            (最佳文案, 所有候选含评分, 构建的 Prompt)
        """
        # 1. 属性校验
        attributes.validate()

        # 2. 构建 Prompt
        prompt = self.prompt_builder.build(persona, product, attributes)

        # 3. 生成候选
        candidates = self.generator.generate(persona, product, attributes, n_candidates)

        # 4. 多目标评估
        scored = self.evaluator.evaluate(candidates, persona, product, weights)

        # 5. 返回最优
        return scored[0], scored, prompt


# =============================================================================
# 母婴出海业务数据
# =============================================================================

def create_mombaby_personas() -> Dict[str, UserPersona]:
    """创建母婴电商典型用户画像"""
    return {
        "working_mom": UserPersona(
            persona_id="P001",
            name="职场背奶妈妈",
            segment="职场背奶妈妈",
            core_needs=["高效吸奶", "静音体验", "便携性"],
            pain_points=["噪音困扰", "时间不够", "配件管理"],
            decision_factors=["吸力强劲", "静音", "便携", "续航"],
            preferred_tone="professional",
            language="zh",
        ),
        "new_mom": UserPersona(
            persona_id="P002",
            name="新手妈妈",
            segment="新手妈妈",
            core_needs=["操作简单", "安全可靠", "售后指导"],
            pain_points=["不懂选择", "担心伤害宝宝", "使用困难"],
            decision_factors=["安全认证", "操作简单", "医生推荐", "口碑"],
            preferred_tone="warm",
            language="zh",
        ),
        "price_sensitive": UserPersona(
            persona_id="P003",
            name="价格敏感妈妈",
            segment="价格敏感妈妈",
            core_needs=["性价比高", "耐用", "大包装实惠"],
            pain_points=["养娃成本高", "消耗太快", "品质参差不齐"],
            decision_factors=["价格", "单片成本", "促销活动", "评价数量"],
            preferred_tone="friendly",
            language="zh",
        ),
        "experienced_mom": UserPersona(
            persona_id="P004",
            name="经验妈妈",
            segment="经验妈妈",
            core_needs=["多功能", "全阶段适用", "品质稳定"],
            pain_points=["重复购买", "功能单一", "产品质量不稳定"],
            decision_factors=["品牌信誉", "多功能", "材质安全", "使用寿命"],
            preferred_tone="professional",
            language="zh",
        ),
    }


def create_mombaby_products() -> Dict[str, ProductProfile]:
    """创建母婴电商典型产品档案"""
    return {
        "breast_pump": ProductProfile(
            product_id="PROD001",
            name="Momcozy S12 Pro 智能吸奶器",
            category="吸奶器",
            price_tier="中端",
            target_market="北美/欧洲",
            features=[
                ProductFeature(
                    feature_id="F001",
                    name="3档智能吸力",
                    description="模拟婴儿自然吮吸节奏",
                    benefit="10分钟高效排空",
                    keywords=["吸力", "高效", "快", "10分钟"],
                ),
                ProductFeature(
                    feature_id="F002",
                    name="静音马达",
                    description="低于40分贝运行噪音",
                    benefit="办公室使用不尴尬",
                    keywords=["静音", "噪音", "安静", "办公室"],
                ),
                ProductFeature(
                    feature_id="F003",
                    name="超长续航",
                    description="2000mAh大容量电池",
                    benefit="一周只需充一次电",
                    keywords=["续航", "电池", "充电", "持久"],
                ),
                ProductFeature(
                    feature_id="F004",
                    name="便携设计",
                    description="整机仅280g，手掌大小",
                    benefit="随身携带无负担",
                    keywords=["便携", "轻便", "携带", "小巧"],
                ),
            ],
        ),
        "diaper": ProductProfile(
            product_id="PROD002",
            name="柔护天使超薄纸尿裤",
            category="纸尿裤",
            price_tier="中端",
            target_market="东南亚/拉美",
            features=[
                ProductFeature(
                    feature_id="F005",
                    name="0.2cm超薄芯体",
                    description="专利吸水高分子材料",
                    benefit="整夜干爽不漏尿",
                    keywords=["超薄", "干爽", "不漏", "吸水"],
                ),
                ProductFeature(
                    feature_id="F006",
                    name="透气底膜",
                    description="10万+透气微孔设计",
                    benefit="有效预防红屁屁",
                    keywords=["透气", "红屁屁", "舒适", "微孔"],
                ),
                ProductFeature(
                    feature_id="F007",
                    name="医护级标准",
                    description="通过SGS皮肤刺激测试",
                    benefit="新生儿也能安心使用",
                    keywords=["医护", "安全", "认证", "测试"],
                ),
            ],
        ),
        "food_processor": ProductProfile(
            product_id="PROD003",
            name="BabyChef全阶段辅食机",
            category="辅食机",
            price_tier="高端",
            target_market="北美/欧洲",
            features=[
                ProductFeature(
                    feature_id="F008",
                    name="6合1多功能",
                    description="蒸/煮/搅拌/加热/消毒/预约",
                    benefit="一台机器搞定所有辅食需求",
                    keywords=["多功能", "蒸煮", "搅拌", "全阶段"],
                ),
                ProductFeature(
                    feature_id="F009",
                    name="智能营养保留",
                    description="低温慢蒸技术",
                    benefit="保留90%以上维生素",
                    keywords=["营养", "维生素", "保留", "健康"],
                ),
                ProductFeature(
                    feature_id="F010",
                    name="FDA认证材质",
                    description="食品级不锈钢+玻璃",
                    benefit="无BPA，宝宝入口更安心",
                    keywords=["FDA", "BPA", "材质", "认证"],
                ),
            ],
        ),
    }


# =============================================================================
# 演示
# =============================================================================

def demo():
    print("=" * 70)
    print("LLM 驱动个性化营销文案生成 - 母婴出海跨境电商")
    print("基于: LLM-Driven E-Commerce Marketing Content Optimization")
    print("      arXiv:2505.23809 + MarketingFM arXiv:2506.17863")
    print("=" * 70)

    pipeline = MarketingCopyPipeline()
    personas = create_mombaby_personas()
    products = create_mombaby_products()

    # 场景1: 职场妈妈 × 吸奶器
    print("\n" + "=" * 70)
    print("【场景1】职场背奶妈妈 × Momcozy S12 Pro 吸奶器")
    print("=" * 70)

    best, candidates, prompt = pipeline.generate(
        persona=personas["working_mom"],
        product=products["breast_pump"],
        attributes=CopyAttributes(
            tone="professional",
            length="medium",
            language="zh",
            cta_type="buy_now",
            emoji_level="moderate",
            urgency_level="low",
        ),
        n_candidates=3,
    )

    print(f"\n构建的 Prompt 片段:")
    print(f"  {prompt[:120]}...")

    print(f"\n生成 {len(candidates)} 个候选文案:")
    for i, c in enumerate(candidates, 1):
        print(f"\n  候选 {i} (总分: {c.overall_score}):")
        print(f"    标题: {c.headline}")
        print(f"    正文: {c.body}")
        print(f"    CTA:  {c.cta}")
        print(f"    评分: {c.scores}")

    print(f"\n  最优文案:")
    print(f"    标题: {best.headline}")
    print(f"    正文: {best.body}")
    print(f"    CTA:  {best.cta}")

    # 场景2: 新手妈妈 × 纸尿裤 (温暖语气)
    print("\n" + "=" * 70)
    print("【场景2】新手妈妈 × 柔护天使纸尿裤 (温暖语气)")
    print("=" * 70)

    best2, candidates2, _ = pipeline.generate(
        persona=personas["new_mom"],
        product=products["diaper"],
        attributes=CopyAttributes(
            tone="warm",
            length="medium",
            language="zh",
            cta_type="learn_more",
            emoji_level="heavy",
            urgency_level="low",
        ),
        n_candidates=3,
    )

    print(f"\n  最优文案:")
    print(f"    标题: {best2.headline}")
    print(f"    正文: {best2.body}")
    print(f"    CTA:  {best2.cta}")
    print(f"    评分: {best2.scores} (总分: {best2.overall_score})")

    # 场景3: 价格敏感妈妈 × 纸尿裤 (紧迫语气)
    print("\n" + "=" * 70)
    print("【场景3】价格敏感妈妈 × 纸尿裤 (促销紧迫语气)")
    print("=" * 70)

    best3, candidates3, _ = pipeline.generate(
        persona=personas["price_sensitive"],
        product=products["diaper"],
        attributes=CopyAttributes(
            tone="urgent",
            length="short",
            language="zh",
            cta_type="limited_offer",
            emoji_level="moderate",
            urgency_level="high",
        ),
        n_candidates=3,
    )

    print(f"\n  最优文案:")
    print(f"    标题: {best3.headline}")
    print(f"    正文: {best3.body}")
    print(f"    CTA:  {best3.cta}")
    print(f"    评分: {best3.scores} (总分: {best3.overall_score})")

    # 场景4: 同用户不同语气对比 (AB测试场景)
    print("\n" + "=" * 70)
    print("【场景4】同画像不同语气 AB 测试对比 - 经验妈妈 × 辅食机")
    print("=" * 70)

    tones = ["professional", "friendly", "warm", "urgent"]
    for tone in tones:
        b, _, _ = pipeline.generate(
            persona=personas["experienced_mom"],
            product=products["food_processor"],
            attributes=CopyAttributes(
                tone=tone,
                length="medium",
                language="zh",
                cta_type="buy_now",
                emoji_level="none",
                urgency_level="low",
            ),
            n_candidates=1,
        )
        print(f"\n  [{tone}] 标题: {b.headline}")
        print(f"           正文: {b.body}")
        print(f"           总分: {b.overall_score}")

    # 场景5: 多语言出海 (英文版)
    print("\n" + "=" * 70)
    print("【场景5】北美市场英文版 - 职场妈妈 × 吸奶器")
    print("=" * 70)

    # 创建英文版画像
    en_persona = UserPersona(
        persona_id="P001_EN",
        name="Working Pumping Mom",
        segment="working_mom",
        core_needs=["efficient pumping", "quiet operation", "portability"],
        pain_points=["noise at office", "not enough time", "battery dies fast"],
        decision_factors=["strong suction", "quiet", "portable", "long battery"],
        preferred_tone="professional",
        language="en",
    )

    # 创建英文版产品
    en_product = ProductProfile(
        product_id="PROD001_EN",
        name="Momcozy S12 Pro Wearable Pump",
        category="breast pump",
        price_tier="mid-range",
        target_market="North America",
        features=[
            ProductFeature(
                feature_id="F001_EN",
                name="3-Mode Smart Suction",
                description="Mimics natural nursing rhythm",
                benefit="10-minute efficient emptying",
                keywords=["suction", "efficient", "fast", "10-minute"],
            ),
            ProductFeature(
                feature_id="F002_EN",
                name="Ultra-Quiet Motor",
                description="Under 40dB operation",
                benefit="Discreet office pumping",
                keywords=["quiet", "silent", "office", "discreet"],
            ),
            ProductFeature(
                feature_id="F003_EN",
                name="All-Day Battery",
                description="2000mAh battery",
                benefit="One charge lasts a full week",
                keywords=["battery", "charge", "week", "lasting"],
            ),
        ],
    )

    best_en, _, _ = pipeline.generate(
        persona=en_persona,
        product=en_product,
        attributes=CopyAttributes(
            tone="professional",
            length="medium",
            language="en",
            cta_type="buy_now",
            emoji_level="none",
            urgency_level="low",
        ),
        n_candidates=1,
    )

    print(f"\n  英文版文案:")
    print(f"    Headline: {best_en.headline}")
    print(f"    Body:     {best_en.body}")
    print(f"    CTA:      {best_en.cta}")

    # 总结
    print("\n" + "=" * 70)
    print("Pipeline 总结")
    print("=" * 70)
    print("可控属性:")
    print("  - tone: professional / friendly / warm / urgent")
    print("  - length: short / medium / long")
    print("  - language: zh / en / es / ja")
    print("  - cta_type: buy_now / learn_more / limited_offer / join_community")
    print("  - emoji_level: none / moderate / heavy")
    print("  - urgency_level: low / medium / high")
    print("\n评估维度:")
    print("  - relevance: 文案与用户画像需求匹配度")
    print("  - coverage: 产品特性覆盖度")
    print("  - cta: 行动号召有效性")
    print("  - diversity: 候选间多样性")
    print("\n业务价值:")
    print("  - 同一产品 × 不同画像 = 差异化文案")
    print("  - 同画像 × 不同语气 = AB 测试素材")
    print("  - 多语言 × 多市场 = 出海本地化")
    print(f"\n{'=' * 70}")
    print("LLM 个性化文案生成测试完成")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    demo()
