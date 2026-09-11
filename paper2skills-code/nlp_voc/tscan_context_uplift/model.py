#!/usr/bin/env python3
"""
TSCAN: Context-Aware Uplift Modeling for Churn Prevention
上下文感知流失挽回策略 - Momcozy场景

论文来源: TSCAN, arXiv:2504.18881
"""

import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from enum import Enum


class ChurnReason(Enum):
    """流失原因类型"""
    PRODUCT_FAULT = "产品故障"
    ACCESSORY_COST = "配件成本"
    NATURAL_WEANING = "自然断奶"
    COMPETITOR_SWITCH = "竞品转移"
    SERVICE_DISSATISFACTION = "服务不满"
    UNKNOWN = "未知"


@dataclass
class UserContext:
    """用户上下文"""
    user_id: str
    churn_probability: float
    last_activity_days: int
    feedback_keywords: List[str]
    baby_age_months: int
    lifetime_value: str  # '高', '中', '低'


class TSCANChurnIntervention:
    """
    TSCAN上下文感知挽回策略选择器
    """

    def __init__(self):
        self.strategies = [
            "发送优惠券",
            "免费配件包",
            "人工电话关怀",
            "产品以旧换新",
            "推送断奶指南",
            "竞品对比强调",
            "专属客服分配",
            "不干预"
        ]

    def identify_churn_reason(self, context: UserContext) -> ChurnReason:
        """识别流失原因"""
        keywords = ' '.join(context.feedback_keywords)

        if any(kw in keywords for kw in ["故障", "坏了", "质量问题"]):
            return ChurnReason.PRODUCT_FAULT
        if any(kw in keywords for kw in ["配件贵", "耗材"]):
            return ChurnReason.ACCESSORY_COST
        if context.baby_age_months >= 6 and context.last_activity_days > 14:
            return ChurnReason.NATURAL_WEANING
        if any(kw in keywords for kw in ["竞品", "别家"]):
            return ChurnReason.COMPETITOR_SWITCH
        if any(kw in keywords for kw in ["客服", "服务差"]):
            return ChurnReason.SERVICE_DISSATISFACTION

        return ChurnReason.UNKNOWN

    def estimate_uplift(
        self,
        context: UserContext,
        strategy: str
    ) -> float:
        """
        估计策略的Uplift值

        实际应使用两阶段神经网络(TSCAN)
        这里使用规则模拟
        """
        reason = self.identify_churn_reason(context)
        ltv = context.lifetime_value

        # 原因-策略匹配表
        uplift_matrix = {
            ChurnReason.PRODUCT_FAULT: {
                "产品以旧换新": 0.32,
                "免费配件包": 0.15,
                "不干预": -0.05
            },
            ChurnReason.ACCESSORY_COST: {
                "免费配件包": 0.28,
                "发送优惠券": 0.12,
                "不干预": 0.05
            },
            ChurnReason.NATURAL_WEANING: {
                "推送断奶指南": 0.35,
                "产品以旧换新": -0.10,  # 适得其反
                "不干预": 0.15
            },
            ChurnReason.COMPETITOR_SWITCH: {
                "竞品对比强调": 0.18,
                "发送优惠券": 0.10,
                "不干预": 0.05
            },
            ChurnReason.SERVICE_DISSATISFACTION: {
                "人工电话关怀": 0.22,
                "专属客服分配": 0.20,
                "不干预": -0.08
            }
        }

        # LTV调整：中价值用户响应更好
        ltv_multiplier = {"高": 0.8, "中": 1.2, "低": 1.0}.get(ltv, 1.0)

        base_uplift = uplift_matrix.get(reason, {}).get(strategy, 0.05)
        return base_uplift * ltv_multiplier

    def recommend_strategy(
        self,
        context: UserContext
    ) -> Tuple[str, float, Dict[str, float]]:
        """
        推荐最优挽回策略

        Returns:
            (最优策略, uplift值, 所有策略uplift字典)
        """
        uplifts = {}
        for strategy in self.strategies:
            uplift = self.estimate_uplift(context, strategy)
            uplifts[strategy] = uplift

        # 选择最高uplift
        best_strategy = max(uplifts, key=uplifts.get)
        best_uplift = uplifts[best_strategy]

        return best_strategy, best_uplift, uplifts


def demo():
    """TSCAN演示"""
    print("=" * 70)
    print("TSCAN上下文感知挽回策略 - Momcozy场景")
    print("=" * 70)

    test_cases = [
        UserContext(
            user_id="U001",
            churn_probability=0.85,
            last_activity_days=30,
            feedback_keywords=["配件贵", "鸭嘴阀坏了"],
            baby_age_months=4,
            lifetime_value="中"
        ),
        UserContext(
            user_id="U002",
            churn_probability=0.75,
            last_activity_days=21,
            feedback_keywords=["宝宝6个月了", "用得少了"],
            baby_age_months=7,
            lifetime_value="高"
        ),
        UserContext(
            user_id="U003",
            churn_probability=0.90,
            last_activity_days=14,
            feedback_keywords=["客服态度差", "退货被拒"],
            baby_age_months=3,
            lifetime_value="高"
        )
    ]

    tscan = TSCANChurnIntervention()

    for case in test_cases:
        print(f"\n【用户 {case.user_id}】")
        print(f"  流失概率: {case.churn_probability:.0%}")
        print(f"  流失原因: {tscan.identify_churn_reason(case).value}")
        print(f"  用户价值: {case.lifetime_value}")

        best_strategy, best_uplift, all_uplifts = tscan.recommend_strategy(case)

        print(f"\n  各策略Uplift:")
        for strategy, uplift in sorted(all_uplifts.items(), key=lambda x: -x[1]):
            marker = "★" if strategy == best_strategy else " "
            print(f"    {marker} {strategy}: {uplift:+.2f}")

    print("\n" + "=" * 70)
    print("业务洞察")
    print("=" * 70)
    print("✓ 不同流失原因匹配不同策略")
    print("✓ 识别'不干预'场景（自然断奶）")
    print("✓ 中价值用户比高价值用户响应更好")


if __name__ == '__main__':
    demo()
