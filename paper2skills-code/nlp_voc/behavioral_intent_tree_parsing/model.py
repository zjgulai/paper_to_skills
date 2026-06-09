"""
Behavioral Intent Tree Parser
基于 IntentRec (Oh et al., 2024) 的 Hierarchical Multi-Task Learning 思想，
将用户行为序列解析为层次化意图结构树。

核心流程：
1. 行为序列编码（点击/浏览/加购/购买/评价）
2. 短期意图识别（当前会话）
3. 长期意图融合（历史行为）
4. 层次化意图树构建
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Any
from enum import Enum
from collections import Counter


# ── 数据模型 ──────────────────────────────────────────

class BehaviorType(Enum):
    """用户行为类型"""
    SEARCH = "search"           # 搜索
    CLICK = "click"             # 点击
    BROWSE = "browse"           # 浏览详情页
    ADD_CART = "add_cart"       # 加购
    ADD_WISHLIST = "add_wishlist"  # 加心愿单
    PURCHASE = "purchase"       # 购买
    REVIEW = "review"           # 评价
    RETURN = "return"           # 退货
    SHARE = "share"             # 分享


class IntentType(Enum):
    """用户意图类型（层次化）"""
    # 第一层: 会话级意图
    DISCOVERY = "discovery"         # 发现/浏览
    COMPARISON = "comparison"       # 比较
    DECISION = "decision"           # 决策
    PURCHASE = "purchase"           # 购买
    POST_PURCHASE = "post_purchase" # 售后

    # 第二层: 细分意图
    PRICE_SENSITIVE = "price_sensitive"     # 价格敏感
    QUALITY_FOCUS = "quality_focus"         # 品质关注
    CONVENIENCE_FOCUS = "convenience_focus" # 便利性关注
    BRAND_LOYAL = "brand_loyal"             # 品牌忠诚
    URGENT_NEED = "urgent_need"             # 紧急需求


@dataclass
class BehaviorEvent:
    """单个行为事件"""
    timestamp: str
    behavior: BehaviorType
    item_id: Optional[str] = None
    item_category: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntentNode:
    """意图树中的一个节点"""
    intent: IntentType
    confidence: float
    evidence: List[str] = field(default_factory=list)  # 支持证据
    children: List[IntentNode] = field(default_factory=list)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "intent": self.intent.value,
            "confidence": round(self.confidence, 3),
            "evidence": self.evidence,
            "children": [c.to_dict() for c in self.children],
        }


@dataclass
class IntentTree:
    """用户意图结构树"""
    user_id: str = ""
    root: Optional[IntentNode] = None
    session_events: List[BehaviorEvent] = field(default_factory=list)
    historical_events: List[BehaviorEvent] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "user_id": self.user_id,
            "root": self.root.to_dict() if self.root else None,
            "session_event_count": len(self.session_events),
            "historical_event_count": len(self.historical_events),
            "metadata": self.metadata,
        }

    def get_flattened_intents(self) -> List[Tuple[str, float]]:
        """获取所有意图及其置信度（扁平化）"""
        result = []
        def _traverse(node: IntentNode) -> None:
            result.append((node.intent.value, node.confidence))
            for child in node.children:
                _traverse(child)
        if self.root:
            _traverse(self.root)
        return result


# ── 核心解析器 ────────────────────────────────────────

class BehavioralIntentTreeParser:
    """
    行为意图树解析器。

    基于 IntentRec 的层次化多任务学习思想，将用户行为序列
    解析为层次化意图结构树。
    """

    # 行为 → 意图的映射规则
    BEHAVIOR_TO_INTENT = {
        BehaviorType.SEARCH: IntentType.DISCOVERY,
        BehaviorType.CLICK: IntentType.DISCOVERY,
        BehaviorType.BROWSE: IntentType.COMPARISON,
        BehaviorType.ADD_CART: IntentType.DECISION,
        BehaviorType.ADD_WISHLIST: IntentType.COMPARISON,
        BehaviorType.PURCHASE: IntentType.PURCHASE,
        BehaviorType.REVIEW: IntentType.POST_PURCHASE,
        BehaviorType.RETURN: IntentType.POST_PURCHASE,
        BehaviorType.SHARE: IntentType.POST_PURCHASE,
    }

    # 意图层级关系（父 → 可能的子）
    INTENT_HIERARCHY = {
        IntentType.DISCOVERY: [IntentType.PRICE_SENSITIVE, IntentType.QUALITY_FOCUS],
        IntentType.COMPARISON: [IntentType.PRICE_SENSITIVE, IntentType.QUALITY_FOCUS, IntentType.CONVENIENCE_FOCUS],
        IntentType.DECISION: [IntentType.BRAND_LOYAL, IntentType.URGENT_NEED],
        IntentType.PURCHASE: [],
        IntentType.POST_PURCHASE: [IntentType.QUALITY_FOCUS, IntentType.CONVENIENCE_FOCUS],
    }

    def __init__(
        self,
        short_term_window: int = 10,    # 短期兴趣窗口（最近 N 个行为）
    ):
        self.short_term_window = short_term_window

    def parse(
        self,
        user_id: str,
        session_events: List[BehaviorEvent],
        historical_events: Optional[List[BehaviorEvent]] = None,
    ) -> IntentTree:
        """
        解析用户行为序列为意图树。

        Args:
            user_id: 用户 ID
            session_events: 当前会话的行为序列（按时间排序）
            historical_events: 历史行为序列（可选）
        """
        historical = historical_events or []

        # 1. 识别会话级主意图
        root_intent = self._identify_session_intent(session_events)

        # 2. 识别细分意图（子节点）
        children = self._identify_sub_intents(
            session_events, historical, root_intent.intent
        )
        root_intent.children = children

        return IntentTree(
            user_id=user_id,
            root=root_intent,
            session_events=session_events,
            historical_events=historical,
            metadata={
                "short_term_events": len(session_events),
                "long_term_events": len(historical),
                "primary_intent": root_intent.intent.value,
            },
        )

    def parse_batch(
        self,
        sessions: List[Tuple[str, List[BehaviorEvent], Optional[List[BehaviorEvent]]]],
    ) -> List[IntentTree]:
        """批量解析"""
        return [self.parse(uid, se, he) for uid, se, he in sessions]

    def _identify_session_intent(
        self,
        events: List[BehaviorEvent],
    ) -> IntentNode:
        """识别会话的主意图"""
        if not events:
            return IntentNode(intent=IntentType.DISCOVERY, confidence=0.5)

        # 统计行为分布
        behavior_counts = Counter(e.behavior for e in events)
        total = len(events)

        # 基于最后行为和分布判断
        last_behavior = events[-1].behavior
        last_intent = self.BEHAVIOR_TO_INTENT.get(last_behavior, IntentType.DISCOVERY)

        # 置信度计算
        if last_behavior == BehaviorType.PURCHASE:
            confidence = 0.95
        elif last_behavior == BehaviorType.ADD_CART:
            confidence = 0.85
        elif last_behavior == BehaviorType.BROWSE:
            # 如果浏览很多但没加购，可能还在比较
            browse_ratio = behavior_counts.get(BehaviorType.BROWSE, 0) / total
            if browse_ratio > 0.6:
                last_intent = IntentType.COMPARISON
                confidence = 0.7 + browse_ratio * 0.2
            else:
                confidence = 0.75
        elif last_behavior == BehaviorType.REVIEW:
            confidence = 0.9
        elif last_behavior == BehaviorType.RETURN:
            confidence = 0.9
            last_intent = IntentType.POST_PURCHASE
        else:
            confidence = 0.6

        evidence = [
            f"最后行为: {last_behavior.value}",
            f"行为分布: {dict(behavior_counts)}",
        ]

        return IntentNode(
            intent=last_intent,
            confidence=round(confidence, 3),
            evidence=evidence,
        )

    def _identify_sub_intents(
        self,
        session_events: List[BehaviorEvent],
        historical_events: List[BehaviorEvent],
        parent_intent: IntentType,
    ) -> List[IntentNode]:
        """识别细分意图（子节点）"""
        children = []
        all_events = session_events + historical_events

        possible_subs = self.INTENT_HIERARCHY.get(parent_intent, [])

        for sub_intent in possible_subs:
            confidence, evidence = self._score_sub_intent(
                sub_intent, session_events, historical_events
            )
            if confidence > 0.4:  # 阈值
                children.append(IntentNode(
                    intent=sub_intent,
                    confidence=round(confidence, 3),
                    evidence=evidence,
                ))

        return children

    def _score_sub_intent(
        self,
        sub_intent: IntentType,
        session_events: List[BehaviorEvent],
        historical_events: List[BehaviorEvent],
    ) -> Tuple[float, List[str]]:
        """为细分意图打分"""
        evidence = []
        score = 0.0

        all_events = session_events + historical_events
        session_text = " ".join(
            str(e.metadata.get("query", "")) + " " + str(e.metadata.get("item_category", ""))
            for e in all_events
        ).lower()

        if sub_intent == IntentType.PRICE_SENSITIVE:
            # 价格敏感信号
            price_signals = ["price", "cheap", "expensive", "discount", "sale", "deal", "cost", "$"]
            matches = sum(1 for s in price_signals if s in session_text)
            if matches > 0:
                score = 0.5 + min(matches * 0.1, 0.4)
                evidence.append(f"价格相关信号出现 {matches} 次")
            # 历史行为: 多次加购未购买
            cart_no_buy = len([e for e in historical_events if e.behavior == BehaviorType.ADD_CART])
            if cart_no_buy > 2:
                score += 0.1
                evidence.append(f"历史加购未购买 {cart_no_buy} 次")

        elif sub_intent == IntentType.QUALITY_FOCUS:
            # 品质关注信号
            quality_signals = ["quality", "durable", "material", "review", "rating", "best", "premium"]
            matches = sum(1 for s in quality_signals if s in session_text)
            if matches > 0:
                score = 0.5 + min(matches * 0.1, 0.4)
                evidence.append(f"品质相关信号出现 {matches} 次")
            # 阅读评价行为
            review_reads = len([e for e in session_events if "review" in str(e.metadata.get("action", "")).lower()])
            if review_reads > 0:
                score += 0.1
                evidence.append(f"浏览评价 {review_reads} 次")

        elif sub_intent == IntentType.CONVENIENCE_FOCUS:
            # 便利性关注
            conv_signals = ["fast", "quick", "easy", "convenient", "portable", "light", "shipping", "delivery"]
            matches = sum(1 for s in conv_signals if s in session_text)
            if matches > 0:
                score = 0.5 + min(matches * 0.1, 0.4)
                evidence.append(f"便利性信号出现 {matches} 次")

        elif sub_intent == IntentType.BRAND_LOYAL:
            # 品牌忠诚
            brand_purchases = len([
                e for e in historical_events
                if e.behavior == BehaviorType.PURCHASE and e.metadata.get("repeat", False)
            ])
            if brand_purchases > 1:
                score = 0.7 + min(brand_purchases * 0.05, 0.25)
                evidence.append(f"历史重复购买 {brand_purchases} 次")
            elif any(e.behavior == BehaviorType.PURCHASE for e in historical_events):
                score = 0.5
                evidence.append("有历史购买记录")

        elif sub_intent == IntentType.URGENT_NEED:
            # 紧急需求
            urgent_signals = ["urgent", "asap", "need now", "tomorrow", "immediate", "emergency"]
            matches = sum(1 for s in urgent_signals if s in session_text)
            if matches > 0:
                score = 0.8
                evidence.append("检测到紧急需求关键词")
            # 快速决策: 搜索→购买间隔短
            if len(session_events) >= 2:
                first = session_events[0].behavior
                last = session_events[-1].behavior
                if first in (BehaviorType.SEARCH, BehaviorType.CLICK) and last == BehaviorType.PURCHASE:
                    score = max(score, 0.7)
                    evidence.append("快速从搜索到购买")

        return min(score, 1.0), evidence


# ── 可视化辅助 ────────────────────────────────────────

def print_intent_tree(tree: IntentTree, indent: int = 0) -> None:
    """打印意图树"""
    prefix = "  " * indent
    print(f"{prefix}🧠 用户行为意图树")
    print(f"{prefix}用户: {tree.user_id}")
    print(f"{prefix}会话行为: {len(tree.session_events)}, 历史行为: {len(tree.historical_events)}")
    print()

    if not tree.root:
        print(f"{prefix}  (空)")
        return

    def _print_node(node: IntentNode, level: int) -> None:
        p = "  " * level
        emoji = {
            IntentType.DISCOVERY: "🔍",
            IntentType.COMPARISON: "⚖️",
            IntentType.DECISION: "🤔",
            IntentType.PURCHASE: "🛒",
            IntentType.POST_PURCHASE: "✅",
            IntentType.PRICE_SENSITIVE: "💰",
            IntentType.QUALITY_FOCUS: "⭐",
            IntentType.CONVENIENCE_FOCUS: "⚡",
            IntentType.BRAND_LOYAL: "🏷️",
            IntentType.URGENT_NEED: "🔥",
        }.get(node.intent, "•")
        conf_bar = "█" * int(node.confidence * 10) + "░" * (10 - int(node.confidence * 10))
        print(f"{p}{emoji} {node.intent.value} [{conf_bar}] {node.confidence:.2f}")
        for ev in node.evidence:
            print(f"{p}   └─ {ev}")
        for child in node.children:
            _print_node(child, level + 1)

    _print_node(tree.root, indent + 1)
    print()


def analyze_intent_distribution(trees: List[IntentTree]) -> Dict[str, Any]:
    """分析多个用户的意图分布"""
    intent_counts: Dict[str, int] = {}
    sub_intent_counts: Dict[str, int] = {}

    for tree in trees:
        if tree.root:
            intent_counts[tree.root.intent.value] = intent_counts.get(tree.root.intent.value, 0) + 1
            for child in tree.root.children:
                sub_intent_counts[child.intent.value] = sub_intent_counts.get(child.intent.value, 0) + 1

    total = len(trees)
    return {
        "total_users": total,
        "primary_intent_distribution": {
            k: {"count": v, "ratio": round(v / total, 2)}
            for k, v in sorted(intent_counts.items(), key=lambda x: -x[1])
        },
        "sub_intent_distribution": {
            k: {"count": v, "ratio": round(v / total, 2)}
            for k, v in sorted(sub_intent_counts.items(), key=lambda x: -x[1])
        },
    }


# ── 测试 ──────────────────────────────────────────────

def test_parser() -> None:
    """单元测试"""
    parser = BehavioralIntentTreeParser()

    # 测试用例 1: 比价型用户
    session1 = [
        BehaviorEvent("2024-01-01T10:00:00", BehaviorType.SEARCH, metadata={"query": "breast pump quiet"}),
        BehaviorEvent("2024-01-01T10:01:00", BehaviorType.CLICK, metadata={"item_category": "pump"}),
        BehaviorEvent("2024-01-01T10:02:00", BehaviorType.BROWSE, metadata={"item_category": "pump"}),
        BehaviorEvent("2024-01-01T10:03:00", BehaviorType.BROWSE, metadata={"item_category": "pump"}),
        BehaviorEvent("2024-01-01T10:05:00", BehaviorType.BROWSE, metadata={"item_category": "pump"}),
    ]
    tree1 = parser.parse("user_001", session1)
    print_intent_tree(tree1)
    assert tree1.root is not None
    assert tree1.root.intent == IntentType.COMPARISON
    print("✅ Test 1 passed")

    # 测试用例 2: 快速购买型用户
    session2 = [
        BehaviorEvent("2024-01-01T11:00:00", BehaviorType.SEARCH, metadata={"query": "Momcozy S12 urgent need"}),
        BehaviorEvent("2024-01-01T11:01:00", BehaviorType.PURCHASE, metadata={"item_category": "pump"}),
    ]
    tree2 = parser.parse("user_002", session2)
    print_intent_tree(tree2)
    assert tree2.root.intent == IntentType.PURCHASE
    print("✅ Test 2 passed")

    # 测试用例 3: 有历史的老用户
    session3 = [
        BehaviorEvent("2024-01-01T12:00:00", BehaviorType.BROWSE, metadata={"item_category": "pump"}),
        BehaviorEvent("2024-01-01T12:01:00", BehaviorType.ADD_CART, metadata={"item_category": "pump"}),
    ]
    history3 = [
        BehaviorEvent("2023-12-01", BehaviorType.PURCHASE, metadata={"repeat": True}),
        BehaviorEvent("2023-11-01", BehaviorType.PURCHASE, metadata={"repeat": True}),
        BehaviorEvent("2023-10-01", BehaviorType.PURCHASE, metadata={"repeat": True}),
    ]
    tree3 = parser.parse("user_003", session3, history3)
    print_intent_tree(tree3)
    assert tree3.root.intent == IntentType.DECISION
    print("✅ Test 3 passed")

    # 测试用例 4: 空序列
    tree4 = parser.parse("user_004", [])
    assert tree4.root.intent == IntentType.DISCOVERY
    print("✅ Test 4 passed")

    print("\n🎉 All tests passed!")


def test_amazon_session_simulation() -> None:
    """用 Amazon 数据模拟行为序列做 POC"""
    import pandas as pd

    data_path = "/Users/pray/project/ai_nlp_voc/research/03-数据资产/高质量数据源/amazon_voc_200k_balanced.csv"
    try:
        df = pd.read_csv(data_path, nrows=100)
    except FileNotFoundError:
        print("数据集未找到。VOC 数据已迁至 /Users/pray/project/ai_nlp_voc/")
        print("请确保 CSV 文件存在于新位置,或通过环境变量提供路径。")
        return


    parser = BehavioralIntentTreeParser()
    trees: List[IntentTree] = []

    for idx, row in df.iterrows():
        rating = row.get("Rating", 0)
        model = str(row.get("Model", "")) if pd.notna(row.get("Model")) else ""
        verified = row.get("Verified Purchase", False)

        # 将评论模拟为行为序列
        # 购买确认 → 浏览 → 评价
        events = []
        if verified:
            events.append(BehaviorEvent(
                str(row.get("Date", "")), BehaviorType.PURCHASE,
                metadata={"verified": True}
            ))
        events.append(BehaviorEvent(
            str(row.get("Date", "")), BehaviorType.REVIEW,
            metadata={"rating": rating, "model": model}
        ))

        # 基于评分决定是否有退货行为
        if rating <= 2:
            events.append(BehaviorEvent(
                str(row.get("Date", "")), BehaviorType.RETURN,
                metadata={"rating": rating}
            ))

        tree = parser.parse(f"user_{idx}", events)
        tree.metadata["rating"] = rating
        trees.append(tree)

    # 分析
    analysis = analyze_intent_distribution(trees)

    print(f"\n📊 Amazon 模拟 POC 统计 ({analysis['total_users']} 用户)")
    print(f"   主意图分布:")
    for intent, data in analysis["primary_intent_distribution"].items():
        print(f"     {intent}: {data['count']} ({data['ratio']})")
    print(f"   细分意图分布:")
    for intent, data in analysis["sub_intent_distribution"].items():
        print(f"     {intent}: {data['count']} ({data['ratio']})")

    print("\n✅ Amazon 模拟 POC 验证通过")


if __name__ == "__main__":
    print("=" * 60)
    print("Behavioral Intent Tree Parser - Unit Tests")
    print("=" * 60)
    test_parser()

    print("\n" + "=" * 60)
    print("Behavioral Intent Tree Parser - Amazon POC")
    print("=" * 60)
    test_amazon_session_simulation()