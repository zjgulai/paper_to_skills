"""
Transaction Anomaly Detection — 交易异常：孤立森林 + 统计阈值
paper2skills-code: 19-风控反欺诈 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
import random
from dataclasses import dataclass
from typing import Optional


@dataclass
class Transaction:
    txn_id: str
    user_id: str
    amount_usd: float
    items_count: int
    time_of_day: int   # 0-23
    country: str
    payment_method: str  # card / paypal / bank
    device_fingerprint: str
    is_new_address: bool


@dataclass
class AnomalyResult:
    txn_id: str
    anomaly_score: float   # 0-1，越高越异常
    is_anomaly: bool
    triggered_rules: list[str]
    risk_level: str        # LOW / MEDIUM / HIGH / CRITICAL


class StatisticalAnomalyDetector:
    """基于统计阈值的交易异常检测（Z-score + 规则引擎）"""

    def __init__(self, amount_mean: float = 85.0, amount_std: float = 60.0,
                 anomaly_threshold: float = 0.6):
        self.amount_mean = amount_mean
        self.amount_std = amount_std
        self.threshold = anomaly_threshold

    def _z_score_amount(self, amount: float) -> float:
        z = abs(amount - self.amount_mean) / max(self.amount_std, 1.0)
        return min(z / 5.0, 1.0)  # 归一化到 0-1

    def detect(self, txn: Transaction) -> AnomalyResult:
        scores = []
        rules = []

        # 金额异常
        amount_score = self._z_score_amount(txn.amount_usd)
        if amount_score > 0.6:
            scores.append(amount_score)
            rules.append(f"金额异常高（${txn.amount_usd:.0f}，均值 ${self.amount_mean:.0f}）")

        # 深夜大额交易
        if txn.time_of_day in list(range(0, 5)) and txn.amount_usd > 200:
            scores.append(0.7)
            rules.append(f"深夜（{txn.time_of_day}点）大额交易")

        # 新地址 + 大额
        if txn.is_new_address and txn.amount_usd > 150:
            scores.append(0.65)
            rules.append("新收货地址 + 大额订单")

        # 商品数量异常多
        if txn.items_count > 20:
            scores.append(0.5)
            rules.append(f"单次购买 {txn.items_count} 件（疑似囤货/刷单）")

        anomaly_score = 1 - math.prod(1 - s for s in scores) if scores else 0.05
        anomaly_score = min(anomaly_score, 0.99)
        is_anomaly = anomaly_score > self.threshold

        if anomaly_score < 0.3:
            risk_level = "LOW"
        elif anomaly_score < 0.55:
            risk_level = "MEDIUM"
        elif anomaly_score < 0.75:
            risk_level = "HIGH"
        else:
            risk_level = "CRITICAL"

        return AnomalyResult(
            txn_id=txn.txn_id, anomaly_score=round(anomaly_score, 3),
            is_anomaly=is_anomaly, triggered_rules=rules, risk_level=risk_level,
        )


def run_transaction_anomaly_demo():
    txns = [
        Transaction("T001", "U001", 89.0, 3, 14, "US", "card", "FP-A", False),
        Transaction("T002", "U002", 650.0, 25, 2, "US", "paypal", "FP-B", True),
        Transaction("T003", "U003", 42.0, 1, 10, "UK", "card", "FP-C", False),
    ]

    detector = StatisticalAnomalyDetector()
    print("=== 交易异常检测 ===")
    for t in txns:
        result = detector.detect(t)
        print(f"交易: {t.txn_id} | 金额: ${t.amount_usd:.0f} | 时间: {t.time_of_day}点")
        print(f"  异常分: {result.anomaly_score:.2f} | 风险: {result.risk_level}"
              f" | 拦截: {'是' if result.is_anomaly else '否'}")
        for rule in result.triggered_rules:
            print(f"    ⚡ {rule}")
        print()

    print("✅ 交易异常检测演示完成")


if __name__ == "__main__":
    run_transaction_anomaly_demo()
