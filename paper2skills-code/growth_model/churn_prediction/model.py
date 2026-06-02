"""
Customer Churn Prediction — 客户流失预测
paper2skills-code: 06-增长模型 | 母婴出海跨境电商
"""
from __future__ import annotations
import math, random
from dataclasses import dataclass


@dataclass
class CustomerFeatures:
    customer_id: str
    days_since_last_order: int
    total_orders: int
    avg_order_value: float
    baby_age_months: int     # 婴儿月龄（母婴关键特征）
    review_given: bool
    coupon_used_pct: float   # 优惠券使用率
    has_subscription: bool


@dataclass
class ChurnPrediction:
    customer_id: str
    churn_probability: float
    risk_level: str     # LOW / MEDIUM / HIGH / CRITICAL
    top_risk_factors: list[str]
    recommended_action: str
    estimated_ltv_at_risk: float


class LogisticChurnModel:
    """逻辑回归流失预测（简化版，生产替换为 XGBoost/LightGBM）"""

    WEIGHTS = {
        "days_since_last_order": 0.04,
        "total_orders": -0.08,
        "avg_order_value": -0.002,
        "baby_age_months": 0.015,
        "review_given": -0.3,
        "coupon_used_pct": 0.2,
        "has_subscription": -0.5,
    }
    INTERCEPT = -1.5

    def predict_proba(self, features: CustomerFeatures) -> float:
        logit = self.INTERCEPT
        logit += features.days_since_last_order * self.WEIGHTS["days_since_last_order"]
        logit += features.total_orders * self.WEIGHTS["total_orders"]
        logit += features.avg_order_value * self.WEIGHTS["avg_order_value"]
        logit += features.baby_age_months * self.WEIGHTS["baby_age_months"]
        logit += (1.0 if features.review_given else 0.0) * self.WEIGHTS["review_given"]
        logit += features.coupon_used_pct * self.WEIGHTS["coupon_used_pct"]
        logit += (1.0 if features.has_subscription else 0.0) * self.WEIGHTS["has_subscription"]
        return 1 / (1 + math.exp(-logit))

    def explain(self, features: CustomerFeatures) -> list[str]:
        factors = []
        if features.days_since_last_order > 60:
            factors.append(f"距上次购买 {features.days_since_last_order} 天（过长）")
        if features.baby_age_months > 36:
            factors.append(f"宝宝月龄 {features.baby_age_months} 月（需求可能减少）")
        if not features.has_subscription:
            factors.append("未开通订阅（复购粘性低）")
        if features.total_orders < 3:
            factors.append(f"累计仅 {features.total_orders} 单（客户生命周期短）")
        return factors


class ChurnInterventionEngine:
    """干预策略引擎"""
    def recommend(self, prob: float, features: CustomerFeatures) -> str:
        if prob < 0.3:
            return "维持现状：正常运营"
        elif prob < 0.5:
            return "轻度干预：发送个性化邮件 + 小额优惠券 (¥10)"
        elif prob < 0.7:
            return "中度干预：专属客服回访 + 推荐适龄新品 + ¥30 优惠"
        else:
            return "重度干预：高价值客服专员跟进 + ¥50 留存券 + 免费样品"


def run_churn_demo():
    random.seed(42)
    customers = [
        CustomerFeatures("C001", 10, 15, 350, 8, True, 0.2, True),
        CustomerFeatures("C002", 75, 2, 120, 42, False, 0.8, False),
        CustomerFeatures("C003", 45, 5, 280, 18, True, 0.5, False),
        CustomerFeatures("C004", 120, 1, 89, 60, False, 0.9, False),
    ]

    model = LogisticChurnModel()
    intervention = ChurnInterventionEngine()

    print("=== 客户流失预测（母婴跨境店铺）===")
    for c in customers:
        prob = model.predict_proba(c)
        factors = model.explain(c)
        action = intervention.recommend(prob, c)
        risk = "CRITICAL" if prob > 0.7 else "HIGH" if prob > 0.5 else "MEDIUM" if prob > 0.3 else "LOW"
        ltv_risk = prob * c.avg_order_value * 12

        print(f"客户: {c.customer_id} | 流失概率: {prob:.1%} [{risk}]")
        if factors:
            print(f"  风险因素: {'; '.join(factors[:2])}")
        print(f"  LTV 风险: ¥{ltv_risk:,.0f}/年")
        print(f"  建议动作: {action}")

    print("✅ 流失预测演示完成")


if __name__ == "__main__":
    run_churn_demo()
