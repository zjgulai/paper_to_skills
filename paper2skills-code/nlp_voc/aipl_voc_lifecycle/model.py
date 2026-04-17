"""
AIPL 生命周期 × VOC 动态标签体系

核心思路：基于用户 VOC 文本（评论、咨询、反馈）判断其 AIPL 生命周期阶段，
检测阶段迁移信号，构建动态演化的用户标签。

AIPL 模型：
  A (Awareness/认知) → I (Interest/兴趣) → P (Purchase/购买) → L (Loyalty/忠诚)

母婴出海场景：将消费者对母婴产品的言论映射到生命周期阶段，
支撑差异化运营触达策略。
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any

import numpy as np
from numpy.typing import NDArray


# ---------------------------------------------------------------------------
# 1. 数据模型
# ---------------------------------------------------------------------------

class AIPLStage(Enum):
    """AIPL 生命周期阶段"""
    AWARENESS = "A_认知"
    INTEREST = "I_兴趣"
    PURCHASE = "P_购买"
    LOYALTY = "L_忠诚"


class VOCSignalType(Enum):
    """VOC 信号类型"""
    SEARCH = "搜索"       # 搜索关键词
    BROWSE = "浏览"       # 浏览行为描述
    INQUIRY = "咨询"      # 客服/社群咨询
    REVIEW = "评价"       # 购买后评价
    FEEDBACK = "反馈"     # 主动反馈/投诉
    SHARE = "分享"        # 社交分享/推荐


@dataclass
class VOCSignal:
    """单个 VOC 信号"""
    user_id: str
    text: str
    signal_type: VOCSignalType
    timestamp: datetime
    sentiment: float = 0.0  # -1.0 ~ +1.0
    aspects: dict[str, float] = field(default_factory=dict)


@dataclass
class UserLifecycleProfile:
    """用户生命周期画像"""
    user_id: str
    current_stage: AIPLStage
    stage_history: list[tuple[datetime, AIPLStage]] = field(default_factory=list)
    signals: list[VOCSignal] = field(default_factory=list)
    tags: dict[str, Any] = field(default_factory=dict)
    transition_confidence: float = 0.0
    next_stage_predicted: AIPLStage | None = None


# ---------------------------------------------------------------------------
# 2. 阶段特征词典（基于母婴出海业务）
# ---------------------------------------------------------------------------

STAGE_KEYWORDS: dict[AIPLStage, dict[VOCSignalType, list[str]]] = {
    AIPLStage.AWARENESS: {
        VOCSignalType.SEARCH: [
            "婴儿纸尿裤推荐", "新生儿衣服品牌", "宝宝辅食哪个好",
            " maternity", "newborn essentials", "best baby formula",
            "奶粉排行榜", "婴儿车怎么选", "母婴用品推荐",
        ],
        VOCSignalType.BROWSE: [
            "看到广告", "朋友推荐", "了解一下", "看看评价",
            "随便看看", "第一次听说", "新品上市", "品牌介绍",
        ],
        VOCSignalType.INQUIRY: [
            "适合几个月宝宝", "和XX有什么区别", "成分安全吗",
            "什么时候开始用", "尺码怎么选", "有试用装吗",
            "what size for", "is it safe for", "difference between",
        ],
    },
    AIPLStage.INTEREST: {
        VOCSignalType.BROWSE: [
            "加购物车", "收藏了", "对比了几家", "加入心愿单",
            "看测评视频", "问朋友意见", "等活动降价", "关注店铺",
            "added to cart", "saved for later", "watching reviews",
        ],
        VOCSignalType.INQUIRY: [
            "什么时候有活动", "满减规则", "能不能包邮",
            "赠品是什么", "保质期多久", "发货地哪里",
            "any discount", "free shipping", "bundle deal",
        ],
        VOCSignalType.REVIEW: [
            "看了很多评价", "好评挺多", "有点犹豫",
            "大部分人说好", "再考虑考虑", "等更多人买了再说",
        ],
    },
    AIPLStage.PURCHASE: {
        VOCSignalType.REVIEW: [
            "刚收到货", "第一次买", "第一次用", "开箱",
            "包装不错", "物流很快", "下单顺利", "支付方便",
            "first purchase", "just arrived", "unboxing",
        ],
        VOCSignalType.FEEDBACK: [
            "订单状态", "物流查询", "什么时候到货",
            "换尺码", "要发票", "售后咨询",
            "order status", "tracking number", "exchange size",
        ],
    },
    AIPLStage.LOYALTY: {
        VOCSignalType.REVIEW: [
            "第二次买了", "一直用这个", "回购很多次",
            "推荐给闺蜜", "全家都用", "忠实粉丝",
            "second purchase", "repeated buyer", "always buy",
        ],
        VOCSignalType.SHARE: [
            "发朋友圈", "推荐给朋友", "宝妈群分享",
            "写种草", "做测评", "安利给同事",
            "recommended to", "shared with", "worth buying",
        ],
        VOCSignalType.FEEDBACK: [
            "希望出新品", "建议改进", "期待更多颜色",
            "VIP会员", "老客福利", "积分兑换",
            "new colors please", "loyalty member", "suggestion",
        ],
    },
}

# 阶段迁移触发词
TRANSITION_TRIGGERS: dict[tuple[AIPLStage, AIPLStage], list[str]] = {
    (AIPLStage.AWARENESS, AIPLStage.INTEREST): [
        "加购", "收藏", "对比", "等活动", "关注店铺",
        "added to cart", "saved", "compare", "watching price",
    ],
    (AIPLStage.INTEREST, AIPLStage.PURCHASE): [
        "下单", "付款", "买了", "支付成功", "确认订单",
        "ordered", "paid", "purchased", "checkout",
    ],
    (AIPLStage.PURCHASE, AIPLStage.LOYALTY): [
        "回购", "再买", "推荐", "分享", "第二次",
        "repurchase", "recommend", "share", "buy again",
    ],
}


# ---------------------------------------------------------------------------
# 3. VOC 信号合成数据生成
# ---------------------------------------------------------------------------

def generate_voc_signals(
    n_users: int = 100,
    signals_per_user: int = 8,
    seed: int = 42,
) -> list[VOCSignal]:
    """生成母婴出海场景合成 VOC 信号数据。"""
    rng = random.Random(seed)
    np_rng = np.random.default_rng(seed)

    all_signals: list[VOCSignal] = []
    base_time = datetime(2026, 1, 1)

    for u in range(n_users):
        user_id = f"user_{u:04d}"

        # 每个用户有随机的生命周期轨迹
        stages = [AIPLStage.AWARENESS]
        # 60% 用户停留在 A，25% 到 I，10% 到 P，5% 到 L
        p = rng.random()
        if p > 0.4:
            stages.append(AIPLStage.INTEREST)
        if p > 0.65:
            stages.append(AIPLStage.PURCHASE)
        if p > 0.95:
            stages.append(AIPLStage.LOYALTY)

        # 为该用户生成信号序列
        for i, stage in enumerate(stages):
            n_sigs = rng.randint(1, 3)
            for _ in range(n_sigs):
                sig_types = list(STAGE_KEYWORDS[stage].keys())
                sig_type = rng.choice(sig_types)
                keywords = STAGE_KEYWORDS[stage][sig_type]
                text = rng.choice(keywords)

                # 随机情感
                if stage in (AIPLStage.PURCHASE, AIPLStage.LOYALTY):
                    sentiment = float(np_rng.normal(0.4, 0.3))
                else:
                    sentiment = float(np_rng.normal(0.0, 0.3))
                sentiment = float(np.clip(sentiment, -1.0, 1.0))

                # 时间推进
                time_offset = timedelta(
                    days=i * 7 + rng.randint(0, 6),
                    hours=rng.randint(0, 23),
                )

                # 方面情感
                aspects = {}
                if stage == AIPLStage.PURCHASE:
                    aspects = {"产品质量": sentiment, "物流速度": sentiment * 0.8}
                elif stage == AIPLStage.LOYALTY:
                    aspects = {"产品质量": sentiment, "使用体验": sentiment * 0.9}

                all_signals.append(
                    VOCSignal(
                        user_id=user_id,
                        text=text,
                        signal_type=sig_type,
                        timestamp=base_time + time_offset,
                        sentiment=round(sentiment, 3),
                        aspects=aspects,
                    )
                )

    return all_signals


# ---------------------------------------------------------------------------
# 4. AIPL 阶段分类器
# ---------------------------------------------------------------------------

class AIPLClassifier:
    """基于 VOC 信号判断用户 AIPL 生命周期阶段。"""

    def __init__(self) -> None:
        self.stage_keywords = STAGE_KEYWORDS
        self.transition_triggers = TRANSITION_TRIGGERS

    def classify(
        self, signals: list[VOCSignal]
    ) -> tuple[AIPLStage, dict[AIPLStage, float]]:
        """
        基于用户 VOC 信号序列判断当前阶段。

        返回: (当前阶段, 各阶段得分)
        """
        stage_scores: dict[AIPLStage, float] = {
            AIPLStage.AWARENESS: 0.0,
            AIPLStage.INTEREST: 0.0,
            AIPLStage.PURCHASE: 0.0,
            AIPLStage.LOYALTY: 0.0,
        }

        for signal in signals:
            for stage, type_keywords in self.stage_keywords.items():
                if signal.signal_type not in type_keywords:
                    continue
                keywords = type_keywords[signal.signal_type]
                for kw in keywords:
                    if kw in signal.text:
                        # 关键词匹配得分 + 情感加权
                        stage_scores[stage] += 1.0 + abs(signal.sentiment)
                        break

            # 迁移触发词检测（偏向后阶段）
            for (from_stage, to_stage), triggers in self.transition_triggers.items():
                for trigger in triggers:
                    if trigger in signal.text:
                        stage_scores[to_stage] += 0.5

        # 时间权重：最近的信号权重更高
        if len(signals) > 1:
            signals_sorted = sorted(signals, key=lambda s: s.timestamp)
            latest = signals_sorted[-1]
            for stage, score in stage_scores.items():
                # 如果最新信号匹配该阶段，加分
                for kw in self._get_all_keywords(stage):
                    if kw in latest.text:
                        stage_scores[stage] = score * 1.5
                        break

        # 归一化
        total = sum(stage_scores.values())
        if total > 0:
            stage_scores = {
                k: round(v / total, 3) for k, v in stage_scores.items()
            }

        # 选择最高分的阶段
        current_stage = max(stage_scores, key=lambda k: stage_scores[k])
        return current_stage, stage_scores

    def _get_all_keywords(self, stage: AIPLStage) -> list[str]:
        """获取某阶段所有关键词。"""
        all_kw: list[str] = []
        for keywords in self.stage_keywords[stage].values():
            all_kw.extend(keywords)
        return all_kw


# ---------------------------------------------------------------------------
# 5. 阶段迁移检测器
# ---------------------------------------------------------------------------

class StageTransitionDetector:
    """检测用户生命周期阶段迁移信号。"""

    def __init__(self) -> None:
        self.transitions = TRANSITION_TRIGGERS

    def detect(
        self,
        signals: list[VOCSignal],
        current_stage: AIPLStage,
    ) -> dict[str, Any]:
        """
        检测迁移信号。

        返回包含迁移概率、目标阶段、置信度等信息。
        """
        if not signals:
            return {"can_transition": False, "confidence": 0.0}

        # 按时间排序
        signals_sorted = sorted(signals, key=lambda s: s.timestamp)
        recent = signals_sorted[-3:]  # 最近3条信号

        # 定义可能的迁移路径
        transition_map: dict[AIPLStage, list[AIPLStage]] = {
            AIPLStage.AWARENESS: [AIPLStage.INTEREST],
            AIPLStage.INTEREST: [AIPLStage.PURCHASE],
            AIPLStage.PURCHASE: [AIPLStage.LOYALTY],
            AIPLStage.LOYALTY: [],
        }

        possible_targets = transition_map.get(current_stage, [])
        if not possible_targets:
            return {"can_transition": False, "confidence": 0.0}

        best_target = possible_targets[0]
        best_score = 0.0

        for target in possible_targets:
            trigger_key = (current_stage, target)
            if trigger_key not in self.transitions:
                continue

            triggers = self.transitions[trigger_key]
            score = 0.0

            for sig in recent:
                for trigger in triggers:
                    if trigger in sig.text:
                        score += 1.0
                        # 正向情感加速迁移
                        if sig.sentiment > 0.3:
                            score += 0.5

            # 信号密度：短时间内多条相关信号 = 高迁移意愿
            if len(recent) >= 2:
                time_span = (recent[-1].timestamp - recent[0].timestamp).total_seconds()
                if time_span < 86400:  # 24小时内
                    score *= 1.3

            if score > best_score:
                best_score = score
                best_target = target

        confidence = min(best_score / 2.0, 1.0)

        return {
            "can_transition": confidence > 0.3,
            "current_stage": current_stage.value,
            "target_stage": best_target.value,
            "confidence": round(confidence, 3),
            "trigger_signals": len(recent),
        }


# ---------------------------------------------------------------------------
# 6. 动态标签系统
# ---------------------------------------------------------------------------

class DynamicTagSystem:
    """基于 AIPL 阶段的动态用户标签管理。"""

    def __init__(self) -> None:
        self.classifier = AIPLClassifier()
        self.transition_detector = StageTransitionDetector()
        self.profiles: dict[str, UserLifecycleProfile] = {}

    def ingest(self, signals: list[VOCSignal]) -> None:
        """摄入 VOC 信号，更新用户画像。"""
        # 按用户分组
        by_user: dict[str, list[VOCSignal]] = {}
        for sig in signals:
            by_user.setdefault(sig.user_id, []).append(sig)

        for user_id, user_signals in by_user.items():
            current_stage, scores = self.classifier.classify(user_signals)
            transition = self.transition_detector.detect(user_signals, current_stage)

            # 构建标签
            tags = self._build_tags(user_id, current_stage, scores, transition, user_signals)

            # 构建历史
            stage_history = [
                (s.timestamp, self.classifier.classify([s])[0]) for s in sorted(user_signals, key=lambda x: x.timestamp)
            ]

            self.profiles[user_id] = UserLifecycleProfile(
                user_id=user_id,
                current_stage=current_stage,
                stage_history=stage_history,
                signals=user_signals,
                tags=tags,
                transition_confidence=transition.get("confidence", 0.0),
                next_stage_predicted=(
                    next((s for s in AIPLStage if s.value == transition["target_stage"]), None)
                    if transition.get("can_transition") and "target_stage" in transition
                    else None
                ),
            )

    def _build_tags(
        self,
        user_id: str,
        stage: AIPLStage,
        scores: dict[AIPLStage, float],
        transition: dict[str, Any],
        signals: list[VOCSignal],
    ) -> dict[str, Any]:
        """构建动态标签。"""
        tags: dict[str, Any] = {
            "生命周期阶段": stage.value,
            "阶段得分": {k.value: v for k, v in scores.items()},
            "活跃信号数": len(signals),
            "最新活动时间": max(s.timestamp for s in signals).isoformat(),
        }

        # 情感标签
        avg_sentiment = np.mean([s.sentiment for s in signals])
        if avg_sentiment > 0.5:
            tags["情感倾向"] = "高满意"
        elif avg_sentiment > 0.0:
            tags["情感倾向"] = "中等满意"
        elif avg_sentiment > -0.5:
            tags["情感倾向"] = "一般"
        else:
            tags["情感倾向"] = "不满意"

        # 阶段特定标签
        if stage == AIPLStage.AWARENESS:
            tags["运营策略"] = "内容种草 + 品牌教育"
            tags["触达渠道"] = "社交媒体/搜索广告"
        elif stage == AIPLStage.INTEREST:
            tags["运营策略"] = "优惠促单 + 评价引导"
            tags["触达渠道"] = "站内信/短信/推送"
            if transition.get("can_transition"):
                tags["运营策略"] += " | [高优先级] 发放首单优惠券"
        elif stage == AIPLStage.PURCHASE:
            tags["运营策略"] = "复购引导 + 会员招募"
            tags["触达渠道"] = "包裹卡/售后回访"
        elif stage == AIPLStage.LOYALTY:
            tags["运营策略"] = "老客福利 + UGC激励"
            tags["触达渠道"] = "会员专属社群"

        # 迁移预测标签
        if transition.get("can_transition"):
            tags["迁移预警"] = f"{transition['current_stage']} → {transition['target_stage']}"
            tags["迁移置信度"] = transition["confidence"]

        # 方面标签
        all_aspects: dict[str, list[float]] = {}
        for s in signals:
            for aspect, score in s.aspects.items():
                all_aspects.setdefault(aspect, []).append(score)

        if all_aspects:
            tags["关注方面"] = {
                aspect: round(float(np.mean(scores)), 2)
                for aspect, scores in all_aspects.items()
            }

        return tags

    def get_profile(self, user_id: str) -> UserLifecycleProfile | None:
        return self.profiles.get(user_id)

    def get_segment_report(self) -> dict[str, Any]:
        """生成用户分群报告。"""
        stage_counts: dict[str, int] = {
            AIPLStage.AWARENESS.value: 0,
            AIPLStage.INTEREST.value: 0,
            AIPLStage.PURCHASE.value: 0,
            AIPLStage.LOYALTY.value: 0,
        }
        transition_ready = 0

        for profile in self.profiles.values():
            stage_counts[profile.current_stage.value] += 1
            if profile.transition_confidence > 0.3:
                transition_ready += 1

        total = len(self.profiles)

        return {
            "总用户数": total,
            "阶段分布": {
                stage: {"数量": count, "占比": f"{count/total:.1%}" if total else "0%"}
                for stage, count in stage_counts.items()
            },
            "迁移就绪用户": transition_ready,
            "迁移就绪率": f"{transition_ready/total:.1%}" if total else "0%",
        }


# ---------------------------------------------------------------------------
# 7. 核心入口函数
# ---------------------------------------------------------------------------

def run_aipl_voc_analysis(
    signals: list[VOCSignal] | None = None,
    n_users: int = 100,
) -> dict[str, Any]:
    """
    主入口：运行 AIPL × VOC 生命周期标签分析。

    返回包含分群报告和示例用户画像的字典。
    """
    if signals is None:
        signals = generate_voc_signals(n_users=n_users)

    system = DynamicTagSystem()
    system.ingest(signals)

    report = system.get_segment_report()

    # 展示几个示例用户
    examples = []
    for user_id in list(system.profiles.keys())[:5]:
        profile = system.profiles[user_id]
        examples.append({
            "用户ID": user_id,
            "当前阶段": profile.current_stage.value,
            "标签": profile.tags,
            "迁移置信度": profile.transition_confidence,
        })

    return {
        "segment_report": report,
        "example_profiles": examples,
        "total_signals": len(signals),
    }


# ---------------------------------------------------------------------------
# 8. 自测
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 70)
    print("AIPL 生命周期 × VOC 动态标签体系 - 母婴出海场景")
    print("=" * 70)

    result = run_aipl_voc_analysis(n_users=100)

    print(f"\n📊 用户分群报告")
    report = result["segment_report"]
    print(f"   总用户数: {report['总用户数']}")
    print(f"   迁移就绪用户: {report['迁移就绪用户']} ({report['迁移就绪率']})")
    print(f"\n   阶段分布:")
    for stage, info in report["阶段分布"].items():
        print(f"     {stage}: {info['数量']} ({info['占比']})")

    print(f"\n👤 示例用户画像")
    print("-" * 70)
    for ex in result["example_profiles"][:3]:
        print(f"\n  用户: {ex['用户ID']}")
        print(f"   阶段: {ex['当前阶段']}")
        print(f"   迁移置信度: {ex['迁移置信度']}")
        print(f"   标签:")
        for k, v in ex["标签"].items():
            print(f"     {k}: {v}")

    print("\n" + "=" * 70)
    print("✅ 分析完成")
    print("=" * 70)
