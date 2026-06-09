"""
AI Consumer Wellbeing Ethics — 消费者福祉与 AI 伦理：母婴场景
paper2skills-code: 11-AI人文 | 母婴出海跨境电商

纯 Python 标准库实现（无外部依赖）
Python 3.14 兼容
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional


# ──────────────────────────────────────────────
# 枚举：伦理违规类型
# ──────────────────────────────────────────────

class EthicsViolationType(Enum):
    """AI 伦理违规分类"""
    DARK_PATTERN = "dark_pattern"          # 暗模式：操纵性 UI/文案
    UNDISCLOSED_AI = "undisclosed_ai"      # 未披露 AI 身份
    CHILD_DATA = "child_data"              # 儿童数据违规
    MEDICAL_ADVICE = "medical_advice"     # 未授权医疗建议


# ──────────────────────────────────────────────
# 数据类：检测结果
# ──────────────────────────────────────────────

@dataclass
class DetectionResult:
    """通用检测结果"""
    compliant: bool
    violations: list[EthicsViolationType] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    details: dict = field(default_factory=dict)

    def __bool__(self) -> bool:
        return self.compliant


@dataclass
class TransparencyCheckResult:
    """AI 透明度检测结果"""
    compliant: bool
    has_ai_disclosure: bool
    has_medical_disclaimer: bool
    triggered_medical_keywords: list[str] = field(default_factory=list)
    recommendation: str = ""


@dataclass
class ChildProtectionResult:
    """儿童保护检测结果"""
    requires_parental_consent: bool
    detected_child_signals: list[str] = field(default_factory=list)
    coppa_applicable: bool = False
    gdpr_k_applicable: bool = False


# ──────────────────────────────────────────────
# 检测器 1：暗模式检测
# ──────────────────────────────────────────────

class DarkPatternDetector:
    """
    检测 UI 文本/广告文案中的操纵性暗模式。

    覆盖的暗模式类型：
    - 虚假紧迫感（Fake Urgency）
    - 情绪劫持（Emotional Exploitation）
    - 隐藏成本/条款（Hidden Subscription）
    - 混淆型文案（Confirm-shaming）
    """

    # 虚假紧迫感关键词（含 CN/EN）
    URGENCY_PATTERNS = [
        r"仅剩\s*\d+\s*件",
        r"限时\s*\d+\s*分钟",
        r"最后一批",
        r"即将涨价",
        r"今天不买",
        r"only\s+\d+\s+left",
        r"limited time",
        r"price goes up",
        r"selling fast",
    ]

    # 情绪劫持：利用父母焦虑
    EMOTIONAL_EXPLOIT_PATTERNS = [
        r"你的宝宝值得最好",
        r"不买就是不爱",
        r"别让孩子输在起跑线",
        r"新手妈妈必备",
        r"专业妈妈都在用",
        r"don't let your baby miss",
        r"every good mother",
    ]

    # 隐藏订阅/自动续费
    HIDDEN_SUBSCRIPTION_PATTERNS = [
        r"默认勾选.*续费",
        r"自动续费.*小字",
        r"auto.?renew",
        r"pre-checked.*subscription",
    ]

    def __init__(self) -> None:
        self._urgency_re = [re.compile(p, re.IGNORECASE) for p in self.URGENCY_PATTERNS]
        self._emotional_re = [re.compile(p, re.IGNORECASE) for p in self.EMOTIONAL_EXPLOIT_PATTERNS]
        self._subscription_re = [re.compile(p, re.IGNORECASE) for p in self.HIDDEN_SUBSCRIPTION_PATTERNS]

    def check(self, text: str) -> DetectionResult:
        """扫描文本，返回暗模式违规情况"""
        violations: list[EthicsViolationType] = []
        warnings: list[str] = []
        details: dict = {
            "urgency_matches": [],
            "emotional_matches": [],
            "subscription_matches": [],
        }

        for pattern in self._urgency_re:
            m = pattern.search(text)
            if m:
                details["urgency_matches"].append(m.group())

        for pattern in self._emotional_re:
            m = pattern.search(text)
            if m:
                details["emotional_matches"].append(m.group())

        for pattern in self._subscription_re:
            m = pattern.search(text)
            if m:
                details["subscription_matches"].append(m.group())

        if details["urgency_matches"] or details["emotional_matches"] or details["subscription_matches"]:
            violations.append(EthicsViolationType.DARK_PATTERN)
            if details["urgency_matches"]:
                warnings.append(f"虚假紧迫感: {details['urgency_matches']}")
            if details["emotional_matches"]:
                warnings.append(f"情绪劫持: {details['emotional_matches']}")
            if details["subscription_matches"]:
                warnings.append(f"隐藏订阅: {details['subscription_matches']}")

        return DetectionResult(
            compliant=len(violations) == 0,
            violations=violations,
            warnings=warnings,
            details=details,
        )


# ──────────────────────────────────────────────
# 检测器 2：儿童数据保护检查
# ──────────────────────────────────────────────

class ChildProtectionChecker:
    """
    检测内容/功能是否涉及儿童数据，并判断 COPPA/GDPR-K 适用性。

    COPPA：美国，13 岁以下；GDPR-K：欧盟，16 岁以下
    """

    # 面向儿童的信号词
    CHILD_SIGNALS = [
        r"baby",
        r"infant",
        r"toddler",
        r"child",
        r"kids?",
        r"under\s+13",
        r"13岁以下",
        r"婴儿",
        r"幼儿",
        r"宝宝",
        r"儿童",
        r"出生日期",
        r"date.of.birth",
        r"baby.+data",
        r"child.+profile",
        r"baby.+growth",
    ]

    # 数据收集信号
    DATA_COLLECTION_SIGNALS = [
        r"collect",
        r"record",
        r"store",
        r"track",
        r"profile",
        r"收集",
        r"记录",
        r"存储",
        r"追踪",
    ]

    def __init__(self) -> None:
        self._child_re = [re.compile(p, re.IGNORECASE) for p in self.CHILD_SIGNALS]
        self._data_re = [re.compile(p, re.IGNORECASE) for p in self.DATA_COLLECTION_SIGNALS]

    def check_content(self, content: str, market: str = "US") -> ChildProtectionResult:
        """检测内容是否触发儿童数据保护规则"""
        child_signals = []
        for pattern in self._child_re:
            m = pattern.search(content)
            if m:
                child_signals.append(m.group())

        has_data_collection = any(p.search(content) for p in self._data_re)
        requires_consent = len(child_signals) > 0 and has_data_collection

        coppa = requires_consent and market in ("US", "GLOBAL")
        gdpr_k = requires_consent and market in ("EU", "UK", "GLOBAL")

        return ChildProtectionResult(
            requires_parental_consent=requires_consent,
            detected_child_signals=list(set(child_signals)),
            coppa_applicable=coppa,
            gdpr_k_applicable=gdpr_k,
        )


# ──────────────────────────────────────────────
# 检测器 3：AI 透明度合规检查
# ──────────────────────────────────────────────

class AITransparencyChecker:
    """
    确保 AI 身份披露合规，以及医疗/营养建议的免责声明。

    规则：
    1. AI 客服首条消息必须包含 AI 身份声明
    2. 涉及医疗/营养/过敏问题时，必须推荐专业人士
    3. 用户直接询问身份时，不可否认
    """

    AI_DISCLOSURE_PATTERNS = [
        r"AI\s*助手",
        r"人工智能",
        r"机器人",
        r"AI\s*assistant",
        r"chatbot",
        r"automated",
        r"非人工",
        r"not a human",
        r"I am an AI",
    ]

    MEDICAL_KEYWORDS = [
        r"过敏",
        r"allerg",
        r"剂量",
        r"dosage",
        r"生病",
        r"sick",
        r"发烧",
        r"fever",
        r"早产",
        r"premature",
        r"医疗",
        r"medical",
        r"营养建议",
        r"nutrition advice",
        r"混合喂养",
        r"混合配方",
    ]

    MEDICAL_DISCLAIMER = (
        "请在儿科医生或专业医疗人员指导下决定，本 AI 不提供医疗建议。"
    )

    def __init__(self) -> None:
        self._disclosure_re = [re.compile(p, re.IGNORECASE) for p in self.AI_DISCLOSURE_PATTERNS]
        self._medical_re = [re.compile(p, re.IGNORECASE) for p in self.MEDICAL_KEYWORDS]

    def check_disclosure(self, first_message: str) -> TransparencyCheckResult:
        """检查 AI 客服首条消息是否包含身份披露"""
        has_disclosure = any(p.search(first_message) for p in self._disclosure_re)
        triggered = [p.pattern for p in self._medical_re if p.search(first_message)]

        needs_medical = len(triggered) > 0
        has_medical_disclaimer = self.MEDICAL_DISCLAIMER in first_message or \
            "儿科医生" in first_message or "medical" in first_message.lower()

        compliant = has_disclosure and (not needs_medical or has_medical_disclaimer)

        rec_parts = []
        if not has_disclosure:
            rec_parts.append("在首条消息中添加 AI 身份声明")
        if needs_medical and not has_medical_disclaimer:
            rec_parts.append("在医疗/营养相关回复中添加专业人士推荐声明")

        return TransparencyCheckResult(
            compliant=compliant,
            has_ai_disclosure=has_disclosure,
            has_medical_disclaimer=has_medical_disclaimer,
            triggered_medical_keywords=triggered,
            recommendation="；".join(rec_parts) if rec_parts else "合规",
        )

    def check_message_for_medical(self, message: str) -> tuple[bool, list[str]]:
        """检测消息中是否包含医疗/营养关键词"""
        triggered = [p.pattern for p in self._medical_re if p.search(message)]
        return len(triggered) > 0, triggered


# ──────────────────────────────────────────────
# 组合入口：全量伦理检查
# ──────────────────────────────────────────────

def run_ethics_check(
    text: str,
    check_type: str = "ad",
    market: str = "US",
) -> dict:
    """
    一站式 AI 伦理合规检查。

    Args:
        text: 待检测文本（广告文案 / AI 对话首条消息 / 功能描述）
        check_type: "ad"（广告）/ "chatbot"（AI 客服）/ "feature"（功能描述）
        market: "US" / "EU" / "UK" / "GLOBAL"

    Returns:
        dict with keys: overall_compliant, violations, results
    """
    results = {}
    all_violations: list[EthicsViolationType] = []

    # 暗模式检测（广告 & 功能描述）
    if check_type in ("ad", "feature"):
        dp = DarkPatternDetector().check(text)
        results["dark_pattern"] = dp
        all_violations.extend(dp.violations)

    # AI 透明度检测（AI 客服）
    if check_type == "chatbot":
        at = AITransparencyChecker().check_disclosure(text)
        results["transparency"] = at
        if not at.compliant:
            all_violations.append(EthicsViolationType.UNDISCLOSED_AI)

    # 儿童保护检测（全场景）
    cp = ChildProtectionChecker().check_content(text, market=market)
    results["child_protection"] = cp
    if cp.requires_parental_consent:
        all_violations.append(EthicsViolationType.CHILD_DATA)

    return {
        "overall_compliant": len(all_violations) == 0,
        "violations": [v.value for v in all_violations],
        "results": results,
    }


# ──────────────────────────────────────────────
# 测试：3 个场景
# ──────────────────────────────────────────────

def _run_tests() -> None:
    print("=" * 60)
    print("AI Consumer Wellbeing Ethics — 伦理合规检测测试")
    print("=" * 60)

    # 场景 1：正常广告文案
    print("\n[场景 1] 正常广告文案")
    result = run_ethics_check(
        text="WF-B 有机婴儿奶粉，通过 FDA 认证，适合 6-12 月龄宝宝。",
        check_type="ad",
        market="US",
    )
    assert result["overall_compliant"], f"场景1应合规，实际: {result['violations']}"
    print(f"  ✓ 合规: {result['overall_compliant']}, 违规: {result['violations']}")

    # 场景 2：暗模式广告文案
    print("\n[场景 2] 暗模式广告文案（虚假紧迫感 + 情绪劫持）")
    result = run_ethics_check(
        text="仅剩3件！今天不买明天涨价！专为新手妈妈设计！你的宝宝值得最好的！",
        check_type="ad",
        market="US",
    )
    assert not result["overall_compliant"], "场景2应检测到暗模式"
    assert "dark_pattern" in result["violations"], f"应有 dark_pattern 违规，实际: {result['violations']}"
    print(f"  ✓ 检测到违规: {result['violations']}")
    dp_result = result["results"]["dark_pattern"]
    print(f"  ✓ 紧迫感匹配: {dp_result.details['urgency_matches']}")
    print(f"  ✓ 情绪劫持匹配: {dp_result.details['emotional_matches']}")

    # 场景 3：AI 客服未披露身份
    print("\n[场景 3] AI 客服首条消息未披露身份")
    result = run_ethics_check(
        text="您好，我可以帮您了解我们的婴儿奶粉产品，请问有什么可以帮您？",
        check_type="chatbot",
        market="US",
    )
    assert not result["overall_compliant"], "场景3应检测到未披露 AI 身份"
    assert "undisclosed_ai" in result["violations"], f"应有 undisclosed_ai 违规，实际: {result['violations']}"
    print(f"  ✓ 检测到违规: {result['violations']}")
    at_result = result["results"]["transparency"]
    print(f"  ✓ 建议: {at_result.recommendation}")

    # 额外：儿童数据保护
    print("\n[场景 4] 儿童数据收集检测")
    cp_checker = ChildProtectionChecker()
    cp_result = cp_checker.check_content(
        "Baby's first formula. Collect your baby's growth data to personalize recommendations.",
        market="US",
    )
    assert cp_result.requires_parental_consent, "应触发家长同意要求"
    assert cp_result.coppa_applicable, "COPPA 应适用"
    print(f"  ✓ 需要家长同意: {cp_result.requires_parental_consent}")
    print(f"  ✓ COPPA 适用: {cp_result.coppa_applicable}")
    print(f"  ✓ 检测到儿童信号: {cp_result.detected_child_signals}")

    print("\n" + "=" * 60)
    print("[✓] 所有场景验证通过 — AI Consumer Wellbeing Ethics")
    print("=" * 60)


if __name__ == "__main__":
    _run_tests()
