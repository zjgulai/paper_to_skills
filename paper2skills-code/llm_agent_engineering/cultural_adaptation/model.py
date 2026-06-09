"""
Cultural Adaptation Agent — 跨文化适应：母婴跨境的本地化 AI 策略
paper2skills-code: 16-智能体工程 | 母婴出海跨境电商

核心设计：
  - CultureProfile：市场文化档案（Hofstede 六维度 + 关键价值主张）
  - CulturalSignalDetector：从文本识别文化背景信号
  - ContentAdaptor：内容风格 + 认证优先级 + 价值主张调整
  - CulturalAdaptationAgent：感知→适配→输出完整流程
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Any


# ─────────────────────────────────────────────
# 文化档案数据结构
# ─────────────────────────────────────────────

@dataclass
class CultureProfile:
    """
    市场文化档案（基于 Hofstede 六维度）。

    Hofstede 维度范围：0-120（标准化评分）
    数据来源：hofstede-insights.com（公开数据）
    """
    market: str                     # 市场代码（ISO 3166-1: US/DE/JP/UK/...）
    individualism_score: int        # IDV：个人主义 vs 集体主义（高=个人主义）
    uncertainty_avoidance: int      # UAI：不确定性规避（高=规避，偏好认证/稳定）
    long_term_orientation: int      # LTO：长期导向（高=长期健康投资）
    power_distance: int             # PDI：权力距离（高=信任权威专家）
    key_values: list[str] = field(default_factory=list)         # 核心价值关键词
    preferred_certifications: list[str] = field(default_factory=list)   # 优选认证
    preferred_tone: str = "balanced"                            # direct / formal / indirect
    currency_format: str = "USD"
    validation_score: float = 1.0   # 适配策略历史验证得分（A/B 测试结果）


CULTURE_DB: dict[str, CultureProfile] = {
    "US": CultureProfile(
        market="US",
        individualism_score=91,
        uncertainty_avoidance=46,
        long_term_orientation=26,
        power_distance=40,
        key_values=["science-backed", "AAP-recommended", "your choice", "trusted by parents"],
        preferred_certifications=["AAP", "FDA", "USDA Organic", "NSF"],
        preferred_tone="direct",
        currency_format="USD",
    ),
    "DE": CultureProfile(
        market="DE",
        individualism_score=67,
        uncertainty_avoidance=65,
        long_term_orientation=83,
        power_distance=35,
        key_values=["bio-zertifiziert", "EU-Standard", "1000-Tage-Gesundheit", "Qualitätsversprechen"],
        preferred_certifications=["EU Organic", "DIN", "TÜV", "Demeter"],
        preferred_tone="formal",
        currency_format="EUR",
    ),
    "JP": CultureProfile(
        market="JP",
        individualism_score=46,
        uncertainty_avoidance=92,
        long_term_orientation=88,
        power_distance=54,
        key_values=["安心", "無添加", "厳選素材", "品質保証", "消費者庁"],
        preferred_certifications=["消費者庁", "JAS", "農林水産省"],
        preferred_tone="indirect",
        currency_format="JPY",
    ),
    "UK": CultureProfile(
        market="UK",
        individualism_score=89,
        uncertainty_avoidance=35,
        long_term_orientation=51,
        power_distance=35,
        key_values=["NHS-aligned", "trusted", "quality assured", "sustainable"],
        preferred_certifications=["NHS", "Soil Association", "BSI"],
        preferred_tone="direct",
        currency_format="GBP",
    ),
}


# ─────────────────────────────────────────────
# 文化信号检测器
# ─────────────────────────────────────────────

@dataclass
class CulturalSignal:
    """从文本检测到的文化信号"""
    detected_market: str
    directness_score: float    # 0.0（极委婉）~ 1.0（极直接）
    dominant_values: list[str]
    detected_certifications: list[str]
    confidence: float          # 检测置信度


class CulturalSignalDetector:
    """从文本（客服消息/评论/搜索词）识别文化背景信号"""

    _DIRECT_PATTERNS = [
        r'\bi want\b', r'\brefund now\b', r'\bimmediately\b', r'\bI demand\b',
        r'\bunacceptable\b', r'\bthis is wrong\b',
    ]
    _INDIRECT_PATTERNS = [
        r'\bnot sure\b', r'\bI wonder\b', r'\bperhaps\b', r'\bI was thinking\b',
        r'\bmay I ask\b', r'\bI\'m not entirely\b', r'でしょうか', r'かもしれません',
    ]

    _CERT_SIGNALS: dict[str, str] = {
        "aap": "US", "fda": "US", "usda organic": "US",
        "eu organic": "DE", "bio": "DE", "demeter": "DE", "öko": "DE",
        "消費者庁": "JP", "jas": "JP", "無添加": "JP", "安心": "JP",
        "nhs": "UK", "soil association": "UK",
    }

    _LANG_MARKET: dict[str, str] = {
        "en": "US",      # 英语默认美国（需结合其他信号校正为UK）
        "de": "DE",
        "ja": "JP",
    }

    def detect(self, text: str, prior_market: str | None = None) -> CulturalSignal:
        """从文本检测文化信号，prior_market 为已知国家（如来自 IP/账号）"""
        lower = text.lower()

        direct_count = sum(1 for p in self._DIRECT_PATTERNS if re.search(p, lower))
        indirect_count = sum(1 for p in self._INDIRECT_PATTERNS if re.search(p, lower))
        total = direct_count + indirect_count
        directness = direct_count / max(total, 1) if total > 0 else 0.5

        detected_certs: list[str] = []
        cert_markets: list[str] = []
        for cert_kw, market in self._CERT_SIGNALS.items():
            if cert_kw in lower:
                detected_certs.append(cert_kw)
                cert_markets.append(market)

        # 市场判断优先级：prior_market > cert_market > language
        if prior_market and prior_market in CULTURE_DB:
            final_market = prior_market
            confidence = 0.95
        elif cert_markets:
            final_market = max(set(cert_markets), key=cert_markets.count)
            confidence = 0.75
        else:
            # 简单语言检测（日文字符）
            if re.search(r'[\u3040-\u309f\u30a0-\u30ff\u4e00-\u9fff]', text):
                final_market = "JP"
                confidence = 0.70
            elif re.search(r'[äöüÄÖÜß]', text):
                final_market = "DE"
                confidence = 0.65
            else:
                final_market = "US"
                confidence = 0.50

        profile = CULTURE_DB.get(final_market)
        dominant_values = profile.key_values[:3] if profile else []

        return CulturalSignal(
            detected_market=final_market,
            directness_score=directness,
            dominant_values=dominant_values,
            detected_certifications=detected_certs,
            confidence=confidence,
        )


# ─────────────────────────────────────────────
# 内容适配器
# ─────────────────────────────────────────────

@dataclass
class AdaptedContent:
    """适配后的内容"""
    market: str
    headline: str
    value_proposition: str
    certification_emphasis: list[str]
    tone: str
    cta: str                        # Call-to-Action 文案
    price_display_hint: str         # 价格展示建议（总价/单价/价值比）


class ContentAdaptor:
    """根据文化档案调整内容风格、认证优先级、价值主张"""

    _HEADLINE_TEMPLATES: dict[str, dict[str, str]] = {
        "US": {
            "infant_formula": "Science-Backed Nutrition {certifications} — Trusted by {n}+ Parents",
            "default": "Premium Quality — Your Best Choice for Baby",
        },
        "DE": {
            "infant_formula": "{certifications} Zertifiziert — Investition in die Gesundheit der ersten 1.000 Tage",
            "default": "Geprüfte Qualität für Ihr Kind",
        },
        "JP": {
            "infant_formula": "厳選素材・{certifications}認定 — 赤ちゃんへの安心をお届けします",
            "default": "品質保証・安心の育児サポート",
        },
        "UK": {
            "infant_formula": "{certifications} Approved — Natural Nutrition for Your Little One",
            "default": "Quality You Can Trust for Baby",
        },
    }

    _CTA_TEMPLATES: dict[str, str] = {
        "US": "Add to Cart — Try It Risk-Free",
        "DE": "Jetzt bestellen — Kostenloser Rückversand",
        "JP": "今すぐご注文 — 安心の返品保証付き",
        "UK": "Shop Now — Free UK Delivery",
    }

    def adapt(
        self,
        product_info: dict[str, Any],
        market: str,
        product_type: str = "default",
    ) -> AdaptedContent:
        """根据市场文化档案生成适配内容"""
        profile = CULTURE_DB.get(market, CULTURE_DB["US"])

        matching_certs = [
            c for c in product_info.get("certifications", [])
            if any(p.lower() in c.lower() for p in profile.preferred_certifications)
        ]
        if not matching_certs:
            matching_certs = product_info.get("certifications", [])[:2]

        cert_str = " · ".join(matching_certs[:2]) if matching_certs else ""
        templates = self._HEADLINE_TEMPLATES.get(market, self._HEADLINE_TEMPLATES["US"])
        template = templates.get(product_type, templates["default"])
        headline = template.format(
            certifications=cert_str,
            n=product_info.get("parent_count", "10,000"),
        ).strip(" — ").strip()

        value_prop = self._build_value_prop(profile, product_info)
        price_hint = self._price_display_hint(profile)

        return AdaptedContent(
            market=market,
            headline=headline,
            value_proposition=value_prop,
            certification_emphasis=matching_certs,
            tone=profile.preferred_tone,
            cta=self._CTA_TEMPLATES.get(market, "Order Now"),
            price_display_hint=price_hint,
        )

    def _build_value_prop(self, profile: CultureProfile, product_info: dict[str, Any]) -> str:
        """基于文化维度构建价值主张"""
        props: list[str] = []

        if profile.uncertainty_avoidance >= 65:
            props.append(f"Certified: {', '.join(profile.preferred_certifications[:2])}")
        if profile.long_term_orientation >= 70:
            props.append("Supports healthy development for the first 1,000 days")
        if profile.individualism_score >= 80:
            props.append("Your choice, your baby's best start")
        elif profile.individualism_score <= 50:
            props.append("Recommended by pediatricians and trusted by families")

        return " | ".join(props) if props else product_info.get("description", "")

    def _price_display_hint(self, profile: CultureProfile) -> str:
        """价格展示策略"""
        if profile.long_term_orientation >= 70:
            return "per_day_cost"     # 德/日：展示每天成本（长期投资视角）
        elif profile.individualism_score >= 85:
            return "value_per_oz"     # 美：展示每盎司价值（个人ROI视角）
        return "total_price"           # 其他：直接展示总价


# ─────────────────────────────────────────────
# 跨文化适应 Agent
# ─────────────────────────────────────────────

@dataclass
class CustomerServiceResponse:
    """客服回复适配结果"""
    market: str
    tone: str
    opening: str
    solution: str
    closing: str
    full_response: str


class CulturalAdaptationAgent:
    """
    跨文化适应 Agent：感知文化信号 → 适配内容/沟通 → 输出本地化结果。

    两个核心能力：
      1. adapt_content()：产品内容适配（Listing/广告文案）
      2. adapt_response()：客服沟通适配（退款/投诉/咨询）
    """

    def __init__(self) -> None:
        self._detector = CulturalSignalDetector()
        self._adaptor = ContentAdaptor()

    def adapt_content(
        self,
        product_info: dict[str, Any],
        market: str,
        product_type: str = "infant_formula",
    ) -> AdaptedContent:
        """产品内容文化适配"""
        profile = CULTURE_DB.get(market)
        if profile and profile.validation_score < 0.6:
            market = "US"  # 适配质量不足时降级为默认市场
        return self._adaptor.adapt(product_info, market, product_type)

    def adapt_response(
        self,
        customer_message: str,
        detected_market: str | None = None,
        intent: str = "general",
    ) -> CustomerServiceResponse:
        """客服回复文化适配"""
        signal = self._detector.detect(customer_message, prior_market=detected_market)
        market = signal.detected_market
        profile = CULTURE_DB.get(market, CULTURE_DB["US"])

        opening, solution, closing = self._build_response_parts(
            profile, intent, signal.directness_score
        )
        full = f"{opening}\n\n{solution}\n\n{closing}".strip()

        return CustomerServiceResponse(
            market=market,
            tone=profile.preferred_tone,
            opening=opening,
            solution=solution,
            closing=closing,
            full_response=full,
        )

    def _build_response_parts(
        self,
        profile: CultureProfile,
        intent: str,
        customer_directness: float,
    ) -> tuple[str, str, str]:
        """根据文化档案构建回复三段式结构"""
        if profile.preferred_tone == "indirect" or profile.uncertainty_avoidance >= 80:
            opening = "ご連絡いただきありがとうございます。ご不便をおかけして大変申し訳ございません。"
            closing = "ご不明な点がございましたら、いつでもお気軽にご連絡ください。"
        elif profile.preferred_tone == "formal":
            opening = "Vielen Dank für Ihre Kontaktaufnahme. Es tut uns leid, dass Sie Unannehmlichkeiten hatten."
            closing = "Bei weiteren Fragen stehen wir Ihnen gerne zur Verfügung."
        else:
            opening = "Thank you for reaching out. I'm sorry to hear about your experience."
            closing = "Please don't hesitate to contact us if you need anything else."

        if intent == "refund_request":
            if profile.preferred_tone == "direct":
                solution = "We'll process your full refund within 2 business days. No return needed."
            elif profile.preferred_tone == "indirect":
                solution = "お子様のご状況について詳しくお聞かせいただけますでしょうか。最善のサポートをご提供できるよう努めます。返金についても柔軟に対応いたします。"
            else:
                solution = "Wir werden Ihre Rückerstattung umgehend bearbeiten. Die Gutschrift erfolgt innerhalb von 3-5 Werktagen."
        else:
            solution = "We're here to help and will resolve this as quickly as possible."

        return opening, solution, closing


# ─────────────────────────────────────────────
# 测试
# ─────────────────────────────────────────────

def _test_cultural_adaptation() -> None:
    """验证 US/DE/JP 三市场的适配差异"""

    agent = CulturalAdaptationAgent()

    product = {
        "name": "Infant Formula Stage 2",
        "certifications": ["AAP", "EU Organic", "消費者庁", "FDA", "JAS"],
        "parent_count": "15,000",
        "description": "Premium infant nutrition formula for 6-12 months",
    }

    # ── 场景一：三市场产品内容适配 ──────────
    results: dict[str, AdaptedContent] = {}
    for market in ["US", "DE", "JP"]:
        adapted = agent.adapt_content(product, market=market, product_type="infant_formula")
        results[market] = adapted
        print(f"\n[{market}] 标题: {adapted.headline}")
        print(f"  认证: {adapted.certification_emphasis}")
        print(f"  语调: {adapted.tone}")
        print(f"  价格展示: {adapted.price_display_hint}")

    us, de, jp = results["US"], results["DE"], results["JP"]

    # US 应优先展示 AAP 认证
    assert any("AAP" in c or "FDA" in c for c in us.certification_emphasis), \
        f"US 应优先 AAP/FDA，实际: {us.certification_emphasis}"
    # DE 应优先展示 EU Organic 认证
    assert any("EU Organic" in c or "Organic" in c.lower() for c in de.certification_emphasis), \
        f"DE 应优先 EU Organic，实际: {de.certification_emphasis}"
    # JP 应优先展示日本认证
    assert any("消費者庁" in c or "JAS" in c for c in jp.certification_emphasis), \
        f"JP 应优先日本认证，实际: {jp.certification_emphasis}"
    # 三个市场的标题应该不同
    assert us.headline != de.headline, "US 和 DE 标题不应相同"
    assert de.headline != jp.headline, "DE 和 JP 标题不应相同"
    print("\n✓ 三市场认证优先级差异验证通过")

    # ── 场景二：客服文化适配 ──────────────
    us_msg = "I want a full refund immediately, this product didn't work as advertised."
    jp_msg = "このたびは商品についてご連絡させていただきます。少し気になる点がございまして..."

    us_resp = agent.adapt_response(us_msg, detected_market="US", intent="refund_request")
    jp_resp = agent.adapt_response(jp_msg, intent="refund_request")

    assert us_resp.tone == "direct", f"美国客服应 direct，实际 {us_resp.tone}"
    assert jp_resp.tone == "indirect", f"日本客服应 indirect，实际 {jp_resp.tone}"
    assert "2 business days" in us_resp.solution or "refund" in us_resp.solution.lower()
    print(f"\n[US 客服] 语调: {us_resp.tone}")
    print(f"  解决方案: {us_resp.solution[:80]}...")
    print(f"\n[JP 客服] 语调: {jp_resp.tone}")
    print(f"  解决方案: {jp_resp.solution[:80]}...")

    # ── 场景三：文化信号检测 ──────────────
    detector = CulturalSignalDetector()

    en_direct = detector.detect("I demand a refund immediately!")
    assert en_direct.directness_score > 0.7, f"直接表达分应 >0.7，实际 {en_direct.directness_score}"

    jp_indirect = detector.detect("かもしれませんが、少し気になることがございまして...")
    assert jp_indirect.directness_score < 0.4, f"日语委婉分应 <0.4，实际 {jp_indirect.directness_score}"
    assert jp_indirect.detected_market == "JP"

    cert_signal = detector.detect("Is this product EU Organic certified?")
    assert cert_signal.detected_market == "DE", f"EU Organic 应识别为 DE，实际 {cert_signal.detected_market}"
    print(f"\n[信号检测] 直接英文 directness={en_direct.directness_score:.2f}")
    print(f"[信号检测] 日语委婉 directness={jp_indirect.directness_score:.2f}, market={jp_indirect.detected_market}")
    print(f"[信号检测] EU认证词 market={cert_signal.detected_market}")

    print("\n✅ 所有测试通过：US/DE/JP 三市场适配差异验证完成")


if __name__ == "__main__":
    _test_cultural_adaptation()
