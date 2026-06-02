"""
生成式受众定向 & LLM 原生广告拍卖框架 (arXiv: 2512.10551 & 2509.18874)

核心逻辑：
1. ZeroShotProfiler   - 仅凭当前文本上下文推断受众画像（无需 Cookie / 历史 ID）
2. LLMAuctionEngine   - 广告与回答内容同步生成，通过迭代奖励优化找到纳什均衡点
3. NativeAdGenerator  - 将商品自然植入 LLM 回答，生成原生广告内容
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple


# ---------------------------------------------------------------------------
# 数据结构
# ---------------------------------------------------------------------------

@dataclass
class UserContext:
    """用户当前交互上下文（不依赖历史追踪，仅使用当下文本）"""
    query_text: str        # 用户的问题或浏览内容
    session_id: str = ""   # 会话 ID（不关联跨会话身份）


@dataclass
class AudienceProfile:
    """零方数据语义逆向推断出的受众画像"""
    intent: str                         # 购买意图标签（如 "holiday_dress"）
    income_level: str                   # 收入水平估算（"low" / "mid" / "high"）
    urgency: float                      # 紧迫度 0.0~1.0
    tags: List[str] = field(default_factory=list)  # 隐性标签列表
    confidence: float = 0.8             # 推断置信度


@dataclass
class AdCandidate:
    """广告候选商品"""
    ad_id: str
    sku: str
    category: str
    bid_price: float         # 广告主竞价（元/展示）
    relevance_score: float   # 与受众画像的相关度 0.0~1.0
    native_snippet: str = "" # 原生植入文本片段


@dataclass
class AuctionResult:
    """拍卖结果：选中的广告 + 生成的原生回答"""
    winner_ad: AdCandidate
    native_response: str
    reward_score: float      # 综合奖励分（广告主出价 × 用户体验加权）
    clearing_price: float    # 实际结算价（第二价格机制）


# ---------------------------------------------------------------------------
# 核心模块 1：零样本受众画像（Zero-shot Profiling）
# ---------------------------------------------------------------------------

# 关键词到意图的简单映射表（真实场景中由 LLM 推断）
_INTENT_RULES: Dict[str, str] = {
    "婚礼": "wedding_attire",
    "海边": "beach_vacation",
    "宝宝": "infant_product",
    "奶粉": "infant_formula",
    "防晒": "suncare",
    "裙子": "fashion_dress",
    "连衣裙": "fashion_dress",
    "穿搭": "fashion_styling",
}

_INCOME_RULES: List[Tuple[List[str], str]] = [
    (["限时", "打折", "优惠券", "便宜"], "low"),
    (["高端", "奢华", "大牌", "品质"], "high"),
]


class ZeroShotProfiler:
    """
    无 Cookie 实时受众画像推断器。
    模拟 LLM 的社会常识推断能力，从当前文本中抽取结构化画像。
    """

    def profile(self, context: UserContext) -> AudienceProfile:
        text = context.query_text.lower()

        # 意图推断（命中第一个关键词）
        intent = "general_shopping"
        for kw, intent_label in _INTENT_RULES.items():
            if kw in text:
                intent = intent_label
                break

        # 收入水平推断
        income = "mid"
        for signals, level in _INCOME_RULES:
            if any(s in text for s in signals):
                income = level
                break

        # 紧迫度推断（含时间限定词则提高紧迫度）
        urgency_signals = ["下周", "明天", "今天", "马上", "急"]
        urgency = 0.85 if any(s in text for s in urgency_signals) else 0.4

        # 隐性标签（规则提取，实际场景由 LLM 生成）
        tags = []
        if "梨形" in text:
            tags.append("body_type:pear")
        if "海边" in text:
            tags.append("scenario:beach")
        if "婚礼" in text:
            tags.append("occasion:wedding")
        if "宝宝" in text or "奶粉" in text:
            tags.append("lifecycle:new_parent")

        return AudienceProfile(
            intent=intent,
            income_level=income,
            urgency=urgency,
            tags=tags,
            confidence=0.82,
        )


# ---------------------------------------------------------------------------
# 核心模块 2：LLM 拍卖引擎（Iterative Reward-Preference Optimization）
# ---------------------------------------------------------------------------

def _compute_reward(ad: AdCandidate, profile: AudienceProfile, alpha: float = 0.6) -> float:
    """
    综合奖励函数：balance 广告主出价与用户体验。
    reward = alpha * normalized_bid + (1 - alpha) * relevance_score

    alpha 控制商业收益 vs 用户体验的权衡（可通过迭代训练学习到纳什均衡值）。
    """
    max_bid = 10.0  # 归一化基准（真实场景从历史拍卖记录中估计）
    normalized_bid = min(ad.bid_price / max_bid, 1.0)
    # 紧迫度调节：高紧迫度用户对相关广告更敏感
    urgency_boost = 1.0 + 0.2 * profile.urgency
    relevance = ad.relevance_score * urgency_boost
    return alpha * normalized_bid + (1 - alpha) * min(relevance, 1.0)


def _second_price(sorted_rewards: List[Tuple[AdCandidate, float]]) -> float:
    """第二价格（Vickrey）拍卖机制：赢家按第二高出价结算"""
    if len(sorted_rewards) < 2:
        return sorted_rewards[0][0].bid_price * 0.5
    return sorted_rewards[1][0].bid_price


class LLMAuctionEngine:
    """
    LLM 原生广告拍卖引擎。
    在 LLM 生成回答时同步完成广告竞拍，实现"广告即内容"的原生植入。
    """

    def __init__(self, alpha: float = 0.6):
        self.alpha = alpha     # 商业权重（通过 RL 迭代收敛到纳什均衡）
        self.round_history: List[Dict] = []

    def run_auction(
        self, candidates: List[AdCandidate], profile: AudienceProfile
    ) -> AuctionResult:
        if not candidates:
            raise ValueError("广告候选集为空，无法进行拍卖")

        # 计算每个广告的综合奖励分
        scored: List[Tuple[AdCandidate, float]] = [
            (ad, _compute_reward(ad, profile, self.alpha)) for ad in candidates
        ]
        scored.sort(key=lambda x: x[1], reverse=True)

        winner, reward = scored[0]
        clearing = _second_price(scored)

        # 记录拍卖历史（用于后续 alpha 的强化学习更新）
        self.round_history.append({
            "winner_id": winner.ad_id,
            "reward": reward,
            "clearing_price": clearing,
        })

        native_response = _build_native_response(winner, profile)

        return AuctionResult(
            winner_ad=winner,
            native_response=native_response,
            reward_score=reward,
            clearing_price=clearing,
        )

    def update_alpha(self, user_clicked: bool, learning_rate: float = 0.05) -> None:
        """
        基于用户点击反馈迭代更新 alpha（模拟奖励偏好对齐）。
        点击 → 说明当前 relevance 权重合适，稍微提高 alpha 以多变现。
        未点击 → 说明体验权重需要加大。
        """
        if user_clicked:
            self.alpha = min(self.alpha + learning_rate, 0.9)
        else:
            self.alpha = max(self.alpha - learning_rate, 0.1)


# ---------------------------------------------------------------------------
# 核心模块 3：原生广告内容生成
# ---------------------------------------------------------------------------

def _build_native_response(ad: AdCandidate, profile: AudienceProfile) -> str:
    """
    将广告商品自然融入 LLM 回答文本（模拟 LLM 生成能力）。
    真实场景中此处调用 LLM API 生成流畅自然语言。
    """
    tag_str = "、".join(profile.tags) if profile.tags else "一般用户"
    urgency_note = "（限时特惠！）" if profile.urgency > 0.7 else ""
    return (
        f"根据您的需求（{tag_str}），我特别推荐以下商品：\n"
        f"✨ {ad.sku}（类目：{ad.category}）{urgency_note}\n"
        f"   {ad.native_snippet}\n"
        f"   当前出价参考：{ad.bid_price:.1f} 元 → 结算价 ≤ 竞争对手出价"
    )


# ---------------------------------------------------------------------------
# 完整业务流水线
# ---------------------------------------------------------------------------

class GenerativeAudienceAdSystem:
    """
    GenAI Advertising 完整流水线：
    用户查询 → 零样本画像 → 广告候选 → LLM 拍卖 → 原生内容生成
    """

    def __init__(self, alpha: float = 0.6):
        self.profiler = ZeroShotProfiler()
        self.auction_engine = LLMAuctionEngine(alpha=alpha)

    def serve(
        self, context: UserContext, ad_candidates: List[AdCandidate]
    ) -> Tuple[AudienceProfile, AuctionResult]:
        profile = self.profiler.profile(context)
        result = self.auction_engine.run_auction(ad_candidates, profile)
        return profile, result


# ---------------------------------------------------------------------------
# 自测入口
# ---------------------------------------------------------------------------

def _run_tests() -> None:
    """内嵌自测用例，运行 python model.py 直接验证"""
    print("=" * 60)
    print("Self-Test: GenAI Advertising / LLM-Auction")
    print("=" * 60)

    # ------------------------------------------------------------------
    # 用例 1：出海女装品牌 AI 购物助手（海边婚礼穿搭）
    # ------------------------------------------------------------------
    print("\n[Test 1] 场景：匿名用户问海边婚礼穿搭")
    ctx1 = UserContext(
        query_text="我下周要去海边参加朋友的婚礼，但我是梨形身材，有什么穿搭建议吗？",
        session_id="anon-001",
    )
    ads1 = [
        AdCandidate("ad-001", "法式高腰长裙 A款", "fashion_dress", bid_price=5.0,
                    relevance_score=0.92, native_snippet="高腰设计完美修饰梨形身材，海风也吹不乱"),
        AdCandidate("ad-002", "防水防晒霜 SPF50", "suncare", bid_price=3.5,
                    relevance_score=0.75, native_snippet="海边外拍专用，持妆8小时不脱妆"),
        AdCandidate("ad-003", "沙滩平底凉鞋", "shoes", bid_price=2.0,
                    relevance_score=0.55, native_snippet="软底防滑，海边步行零疲劳"),
    ]

    system = GenerativeAudienceAdSystem(alpha=0.6)
    profile1, result1 = system.serve(ctx1, ads1)

    assert profile1.intent == "wedding_attire", f"意图推断错误: {profile1.intent}"
    assert profile1.urgency > 0.7, f"紧迫度应高: {profile1.urgency}"
    assert "body_type:pear" in profile1.tags, f"缺少身型标签: {profile1.tags}"
    assert "occasion:wedding" in profile1.tags, f"缺少场合标签: {profile1.tags}"
    assert result1.winner_ad.ad_id == "ad-001", f"拍卖赢家应为 ad-001: {result1.winner_ad.ad_id}"
    assert result1.clearing_price <= result1.winner_ad.bid_price, "结算价不能高于赢家出价"
    print(f"  画像: intent={profile1.intent}, urgency={profile1.urgency:.2f}, tags={profile1.tags}")
    print(f"  拍卖赢家: {result1.winner_ad.sku}, reward={result1.reward_score:.3f}, 结算价={result1.clearing_price:.2f}")
    print(f"  原生回答:\n{result1.native_response}")
    print("  ✓ Test 1 PASSED")

    # ------------------------------------------------------------------
    # 用例 2：母婴品牌场景（新手父母）
    # ------------------------------------------------------------------
    print("\n[Test 2] 场景：匿名用户询问奶粉推荐")
    ctx2 = UserContext(
        query_text="我家宝宝两个月了，最近换奶粉有点不适应，有没有推荐的进口奶粉？",
        session_id="anon-002",
    )
    ads2 = [
        AdCandidate("ad-101", "有机奶粉旗舰款", "infant_formula", bid_price=8.0,
                    relevance_score=0.95, native_snippet="0-6月龄专属配方，GMP认证工厂直供"),
        AdCandidate("ad-102", "益生菌滴剂", "baby_health", bid_price=4.0,
                    relevance_score=0.78, native_snippet="调节肠道菌群，改善换奶期不适"),
        AdCandidate("ad-103", "奶瓶清洗液", "baby_care", bid_price=2.5,
                    relevance_score=0.42, native_snippet="无香无荧光剂，宝宝安全使用"),
    ]

    profile2, result2 = system.serve(ctx2, ads2)
    assert "lifecycle:new_parent" in profile2.tags, f"缺少新手父母标签: {profile2.tags}"
    assert result2.winner_ad.ad_id == "ad-101", f"奶粉广告应获胜: {result2.winner_ad.ad_id}"
    print(f"  画像: intent={profile2.intent}, tags={profile2.tags}")
    print(f"  拍卖赢家: {result2.winner_ad.sku}, reward={result2.reward_score:.3f}")
    print("  ✓ Test 2 PASSED")

    # ------------------------------------------------------------------
    # 用例 3：alpha 迭代更新（奖励偏好对齐）
    # ------------------------------------------------------------------
    print("\n[Test 3] alpha 迭代更新（模拟 3 轮用户反馈）")
    engine = LLMAuctionEngine(alpha=0.6)
    alpha_trace = [engine.alpha]
    # 前两轮未点击 → alpha 下降（增加用户体验权重）
    engine.update_alpha(user_clicked=False)
    engine.update_alpha(user_clicked=False)
    # 第三轮点击 → alpha 上升
    engine.update_alpha(user_clicked=True)
    alpha_trace.append(engine.alpha)

    assert engine.alpha < 0.6, f"期望 alpha 下降: {engine.alpha}"
    print(f"  初始 alpha=0.6 → 两次未点击 → 一次点击 → 最终 alpha={engine.alpha:.3f}")
    print("  ✓ Test 3 PASSED")

    # ------------------------------------------------------------------
    # 用例 4：单候选拍卖（边界情况）
    # ------------------------------------------------------------------
    print("\n[Test 4] 边界：单一广告候选")
    ctx4 = UserContext(query_text="推荐防晒霜", session_id="anon-003")
    ads4 = [AdCandidate("ad-x", "矿物防晒棒", "suncare", bid_price=6.0,
                         relevance_score=0.9, native_snippet="纯物理防晒，孕妇可用")]
    profile4, result4 = system.serve(ctx4, ads4)
    assert result4.winner_ad.ad_id == "ad-x"
    print(f"  单候选结果: {result4.winner_ad.sku}, clearing_price={result4.clearing_price:.2f}")
    print("  ✓ Test 4 PASSED")

    print("\n" + "=" * 60)
    print("All 4 tests PASSED ✓")
    print("=" * 60)


def main() -> None:
    _run_tests()


if __name__ == "__main__":
    main()
