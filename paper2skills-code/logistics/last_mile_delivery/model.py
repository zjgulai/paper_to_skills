"""
Last-Mile Delivery — 末公里配送时效预测
paper2skills-code: 18-物流履约 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
from dataclasses import dataclass


@dataclass
class DeliveryOrder:
    order_id: str
    destination_zip: str
    weight_kg: float
    carrier: str
    ship_date: str
    distance_km: float
    is_rural: bool = False
    requires_signature: bool = False


@dataclass
class DeliveryPrediction:
    order_id: str
    predicted_days: float
    confidence: float
    on_time_probability: float  # 在承诺时效内到达的概率
    risk_factors: list[str]


class LastMilePredictor:
    """末公里配送时效预测（特征工程 + 回归简化版）"""

    BASE_DAYS = {"fedex": 2.0, "ups": 2.5, "usps": 3.5, "dhl": 3.0}

    def predict(self, order: DeliveryOrder,
                promised_days: int = 3) -> DeliveryPrediction:
        base = self.BASE_DAYS.get(order.carrier.lower(), 3.0)
        # 特征调整
        if order.is_rural:
            base += 1.2
        if order.distance_km > 2000:
            base += 0.5
        if order.requires_signature:
            base += 0.3
        if order.weight_kg > 20:
            base += 0.4

        predicted = base
        # 简化的不确定性：远距 + 农村 增加方差
        std = 0.5 + (0.3 if order.is_rural else 0) + (0.2 if order.distance_km > 2000 else 0)
        on_time_prob = max(0.1, min(0.99,
            0.5 + (promised_days - predicted) / (2 * std)
        ))
        confidence = 1.0 - std / predicted if predicted > 0 else 0.5

        risks = []
        if order.is_rural:
            risks.append("农村地址：额外 1-2 天")
        if order.weight_kg > 20:
            risks.append("超重件：可能需要预约配送")
        if order.requires_signature:
            risks.append("签收件：如未到需重新安排")

        return DeliveryPrediction(
            order_id=order.order_id,
            predicted_days=round(predicted, 1),
            confidence=round(confidence, 2),
            on_time_probability=round(on_time_prob, 2),
            risk_factors=risks,
        )


def run_last_mile_demo():
    orders = [
        DeliveryOrder("ORD-001", "10001", 5.0, "fedex", "2026-06-01", 500, False, False),
        DeliveryOrder("ORD-002", "98765", 25.0, "ups", "2026-06-01", 3500, True, True),
    ]

    predictor = LastMilePredictor()
    print("=== 末公里配送时效预测 ===")
    for o in orders:
        pred = predictor.predict(o, promised_days=3)
        print(f"订单: {o.order_id} | 载体: {o.carrier}")
        print(f"  预测时效: {pred.predicted_days} 天 | 准时概率: {pred.on_time_probability:.0%}")
        if pred.risk_factors:
            print(f"  风险: {'; '.join(pred.risk_factors)}")
        print()

    print("✅ 末公里演示完成")


if __name__ == "__main__":
    run_last_mile_demo()
