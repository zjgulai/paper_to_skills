"""
Identity Fraud Detection — 多维账号欺诈检测：设备+行为+网络三重验证
三维信号融合：设备指纹相似度 + 行为序列异常 + 账号关联图社区

依赖：纯 Python 标准库（无外部依赖）
Python 版本：3.8+
"""

import math
import statistics
from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, FrozenSet, List, Optional, Set, Tuple


@dataclass
class AccountProfile:
    account_id: str
    device_fingerprint: Dict[str, str]
    behavior_sequence: List[Dict]
    registration_time: datetime
    is_known_fraud: bool = False
    payment_hash: Optional[str] = None
    shipping_address_hash: Optional[str] = None
    ip_prefix: Optional[str] = None


@dataclass
class FraudSignal:
    dimension: str
    score: float
    detail: str


@dataclass
class FraudReport:
    account_id: str
    fraud_score: float
    risk_level: str
    signals: List[FraudSignal]
    recommendation: str


class DeviceFingerprintMatcher:
    """
    设备指纹相似度计算
    使用 Jaccard 相似度识别同一物理设备注册的多个账号
    """

    def jaccard_similarity(self, fp_a: Dict[str, str], fp_b: Dict[str, str]) -> float:
        keys = set(fp_a) | set(fp_b)
        if not keys:
            return 0.0
        matches = sum(1 for k in keys if fp_a.get(k) == fp_b.get(k) and k in fp_a and k in fp_b)
        intersection = sum(1 for k in keys if fp_a.get(k) == fp_b.get(k))
        return intersection / len(keys)

    def compute_device_fraud_score(
        self,
        target: AccountProfile,
        all_accounts: List[AccountProfile],
        similarity_threshold: float = 0.85,
    ) -> Tuple[float, str]:
        known_fraud_similar = 0
        total_similar = 0

        for acc in all_accounts:
            if acc.account_id == target.account_id:
                continue
            sim = self.jaccard_similarity(target.device_fingerprint, acc.device_fingerprint)
            if sim >= similarity_threshold:
                total_similar += 1
                if acc.is_known_fraud:
                    known_fraud_similar += 1

        if total_similar == 0:
            return 0.0, "设备指纹唯一，无同设备关联账号"

        fraud_ratio = known_fraud_similar / total_similar
        device_score = min(1.0, 0.3 * total_similar / 5 + 0.7 * fraud_ratio)
        detail = (
            f"发现 {total_similar} 个相似设备账号，"
            f"其中已知欺诈账号 {known_fraud_similar} 个（欺诈比例 {fraud_ratio:.1%}）"
        )
        return device_score, detail


class BehaviorAnomalyDetector:
    """
    行为序列统计异常检测
    基于 Z-score 检测购买时间、评论速度等行为维度的偏差
    """

    def compute_behavior_score(
        self,
        target: AccountProfile,
        all_accounts: List[AccountProfile],
    ) -> Tuple[float, str]:
        target_features = self._extract_features(target)
        if not target_features:
            return 0.0, "行为数据不足"

        all_features_list = [
            self._extract_features(a)
            for a in all_accounts
            if a.account_id != target.account_id
        ]
        all_features_list = [f for f in all_features_list if f]

        if len(all_features_list) < 3:
            return 0.0, "基线样本不足（需至少 3 个账号建立基线）"

        anomaly_scores = []
        anomaly_details = []

        for feat_name, target_val in target_features.items():
            population = [f[feat_name] for f in all_features_list if feat_name in f]
            if len(population) < 2:
                continue
            try:
                mu = statistics.mean(population)
                sigma = statistics.stdev(population)
            except statistics.StatisticsError:
                continue
            if sigma < 1e-9:
                continue
            z = abs((target_val - mu) / sigma)
            anomaly_scores.append(min(1.0, z / 5.0))
            if z > 2.5:
                anomaly_details.append(f"{feat_name} Z-score={z:.2f}")

        if not anomaly_scores:
            return 0.0, "无可用行为特征"

        behavior_score = min(1.0, statistics.mean(anomaly_scores) * 1.5)
        detail = (
            f"行为异常特征: {', '.join(anomaly_details)}" if anomaly_details
            else "行为模式在正常范围内"
        )
        return behavior_score, detail

    @staticmethod
    def _extract_features(account: AccountProfile) -> Dict[str, float]:
        if not account.behavior_sequence:
            return {}

        features: Dict[str, float] = {}

        purchase_events = [e for e in account.behavior_sequence if e.get("type") == "purchase"]
        review_events = [e for e in account.behavior_sequence if e.get("type") == "review"]

        features["purchase_count_per_day"] = len(purchase_events)

        if purchase_events and review_events:
            features["review_rate"] = len(review_events) / max(1, len(purchase_events))

        if purchase_events:
            hours = [e.get("hour", 12) for e in purchase_events]
            features["avg_purchase_hour"] = statistics.mean(hours)
            categories = [e.get("category", "") for e in purchase_events]
            cat_counter: Dict[str, int] = {}
            for c in categories:
                cat_counter[c] = cat_counter.get(c, 0) + 1
            top_cat_ratio = max(cat_counter.values()) / len(categories) if categories else 0
            features["category_concentration"] = top_cat_ratio

        account_age_days = max(
            1,
            (datetime.now() - account.registration_time).days
        )
        features["purchase_per_age_day"] = len(purchase_events) / account_age_days

        return features


class AccountGraphAnalyzer:
    """
    账号关联图社区发现
    基于 Label Propagation 算法（O(n)复杂度）识别欺诈社区
    """

    EDGE_WEIGHTS = {
        "device": 0.9,
        "payment": 0.8,
        "address": 0.7,
        "ip_prefix": 0.5,
    }

    def build_edges(
        self,
        accounts: List[AccountProfile],
        device_threshold: float = 0.85,
    ) -> Dict[str, List[Tuple[str, float]]]:
        matcher = DeviceFingerprintMatcher()
        adjacency: Dict[str, List[Tuple[str, float]]] = {a.account_id: [] for a in accounts}

        for i, a in enumerate(accounts):
            for j, b in enumerate(accounts):
                if j <= i:
                    continue
                edge_weight = 0.0

                sim = matcher.jaccard_similarity(a.device_fingerprint, b.device_fingerprint)
                if sim >= device_threshold:
                    edge_weight = max(edge_weight, sim * self.EDGE_WEIGHTS["device"])

                if a.payment_hash and b.payment_hash and a.payment_hash == b.payment_hash:
                    edge_weight = max(edge_weight, self.EDGE_WEIGHTS["payment"])

                if (a.shipping_address_hash and b.shipping_address_hash
                        and a.shipping_address_hash == b.shipping_address_hash):
                    edge_weight = max(edge_weight, self.EDGE_WEIGHTS["address"])

                if a.ip_prefix and b.ip_prefix and a.ip_prefix == b.ip_prefix:
                    edge_weight = max(edge_weight, self.EDGE_WEIGHTS["ip_prefix"])

                if edge_weight > 0:
                    adjacency[a.account_id].append((b.account_id, edge_weight))
                    adjacency[b.account_id].append((a.account_id, edge_weight))

        return adjacency

    def label_propagation(
        self,
        accounts: List[AccountProfile],
        adjacency: Dict[str, List[Tuple[str, float]]],
        n_iterations: int = 20,
    ) -> Dict[str, int]:
        labels = {a.account_id: i for i, a in enumerate(accounts)}

        for _ in range(n_iterations):
            changed = False
            for account in accounts:
                neighbors = adjacency.get(account.account_id, [])
                if not neighbors:
                    continue
                label_weights: Dict[int, float] = {}
                for neighbor_id, weight in neighbors:
                    lbl = labels[neighbor_id]
                    label_weights[lbl] = label_weights.get(lbl, 0.0) + weight

                if label_weights:
                    best_label = max(label_weights, key=lambda l: label_weights[l])
                    if best_label != labels[account.account_id]:
                        labels[account.account_id] = best_label
                        changed = True

            if not changed:
                break

        return labels

    def compute_network_score(
        self,
        target: AccountProfile,
        accounts: List[AccountProfile],
        fraud_community_threshold: float = 0.30,
    ) -> Tuple[float, str]:
        adjacency = self.build_edges(accounts)
        labels = self.label_propagation(accounts, adjacency)

        target_label = labels[target.account_id]
        community_members = [
            a for a in accounts if labels[a.account_id] == target_label
        ]

        if len(community_members) <= 1:
            return 0.0, "账号无关联社区（图孤立节点）"

        fraud_count = sum(1 for a in community_members if a.is_known_fraud)
        fraud_ratio = fraud_count / len(community_members)

        network_score = min(1.0, fraud_ratio * 1.5 + 0.1 * (len(community_members) - 1) / 10)
        detail = (
            f"所在社区 {len(community_members)} 个账号，"
            f"已知欺诈比例 {fraud_ratio:.1%}（阈值 {fraud_community_threshold:.0%}）"
        )
        return network_score, detail


class IdentityFraudDetector:
    WEIGHTS = {"device": 0.35, "behavior": 0.35, "network": 0.30}

    def __init__(self):
        self._device_matcher = DeviceFingerprintMatcher()
        self._behavior_detector = BehaviorAnomalyDetector()
        self._graph_analyzer = AccountGraphAnalyzer()

    def detect(
        self,
        target: AccountProfile,
        all_accounts: List[AccountProfile],
    ) -> FraudReport:
        device_score, device_detail = self._device_matcher.compute_device_fraud_score(
            target, all_accounts
        )
        behavior_score, behavior_detail = self._behavior_detector.compute_behavior_score(
            target, all_accounts
        )
        network_score, network_detail = self._graph_analyzer.compute_network_score(
            target, all_accounts
        )

        fraud_score = (
            self.WEIGHTS["device"] * device_score
            + self.WEIGHTS["behavior"] * behavior_score
            + self.WEIGHTS["network"] * network_score
        )
        fraud_score = round(min(1.0, max(0.0, fraud_score)), 4)

        if fraud_score >= 0.65:
            risk_level = "HIGH"
            recommendation = "⚠️ 高欺诈风险：建议立即限流 + 人工审核，暂停评论权限"
        elif fraud_score >= 0.40:
            risk_level = "MEDIUM"
            recommendation = "⚡ 中等欺诈风险：触发二次验证（手机短信/邮件），加强购买行为监控"
        else:
            risk_level = "LOW"
            recommendation = "✅ 低欺诈风险：正常放行，保持常规监控"

        signals = [
            FraudSignal("device_fingerprint", round(device_score, 4), device_detail),
            FraudSignal("behavior_anomaly", round(behavior_score, 4), behavior_detail),
            FraudSignal("account_network", round(network_score, 4), network_detail),
        ]

        return FraudReport(
            account_id=target.account_id,
            fraud_score=fraud_score,
            risk_level=risk_level,
            signals=signals,
            recommendation=recommendation,
        )


def _make_test_accounts() -> List[AccountProfile]:
    base_time = datetime(2024, 1, 1)
    normal_fp = {
        "user_agent": "Mozilla/5.0 Chrome/120",
        "screen": "1920x1080",
        "timezone": "America/New_York",
        "language": "en-US",
        "platform": "Win32",
    }
    fraud_fp = {
        "user_agent": "Mozilla/5.0 Chrome/119",
        "screen": "1366x768",
        "timezone": "UTC",
        "language": "en-US",
        "platform": "Win32",
    }

    accounts = []

    for i in range(7):
        from datetime import timedelta
        accounts.append(AccountProfile(
            account_id=f"USER-N{i+1:02d}",
            device_fingerprint={**normal_fp, "platform": f"Win{i%3+10}"},
            behavior_sequence=[
                {"type": "purchase", "hour": 14 + i % 6, "category": "stroller"},
                {"type": "purchase", "hour": 10 + i % 4, "category": "formula"},
                {"type": "review", "hour": 20},
            ],
            registration_time=base_time - timedelta(days=180 + i * 30),
            is_known_fraud=False,
            ip_prefix="192.168.1",
        ))

    fraud_device = {
        "user_agent": "Mozilla/5.0 Chrome/118",
        "screen": "1024x768",
        "timezone": "UTC+8",
        "language": "zh-CN",
        "platform": "Linux",
    }
    shared_payment = "xxxx-4321"
    shared_address = "hash-addr-fraud-warehouse"
    fraud_ip = "10.0.0"

    for i in range(3):
        from datetime import timedelta
        accounts.append(AccountProfile(
            account_id=f"FRAUD-{i+1:02d}",
            device_fingerprint={
                **fraud_device,
                "screen": "1024x768",
            },
            behavior_sequence=[
                {"type": "purchase", "hour": 3, "category": "stroller"},
                {"type": "purchase", "hour": 3, "category": "stroller"},
                {"type": "purchase", "hour": 4, "category": "stroller"},
                {"type": "review", "hour": 4},
                {"type": "review", "hour": 4},
                {"type": "review", "hour": 5},
            ],
            registration_time=base_time - timedelta(days=2),
            is_known_fraud=(i == 0),
            payment_hash=shared_payment,
            shipping_address_hash=shared_address,
            ip_prefix=fraud_ip,
        ))

    return accounts


def run_demo():
    print("=" * 60)
    print("Identity Fraud Detection — 演示")
    print("=" * 60)

    accounts = _make_test_accounts()
    detector = IdentityFraudDetector()

    targets = [
        ("FRAUD-02", "已知欺诈关联账号（欺诈组）"),
        ("USER-N01", "正常用户"),
        ("FRAUD-03", "新欺诈账号（与 FRAUD-01 共享设备/支付）"),
    ]

    reports = []
    for account_id, label in targets:
        target = next(a for a in accounts if a.account_id == account_id)
        report = detector.detect(target, accounts)
        reports.append(report)

        print(f"\n👤 {account_id} [{label}]")
        print(f"  欺诈概率: {report.fraud_score:.3f} | 风险等级: {report.risk_level}")
        for sig in report.signals:
            print(f"  [{sig.dimension}] 分值={sig.score:.3f}: {sig.detail}")
        print(f"  💡 {report.recommendation}")

    fraud_reports = [r for r in reports if r.account_id.startswith("FRAUD")]
    normal_reports = [r for r in reports if r.account_id.startswith("USER")]

    for r in fraud_reports:
        assert r.risk_level in {"MEDIUM", "HIGH"}, \
            f"欺诈账号 {r.account_id} 应被识别为中/高风险，实际: {r.risk_level}"
    for r in normal_reports:
        assert r.risk_level == "LOW", \
            f"正常账号 {r.account_id} 应为低风险，实际: {r.risk_level}"

    print(f"\n✅ 所有断言通过 — 欺诈账号均被正确识别为中/高风险")

    detected = sum(1 for r in fraud_reports if r.risk_level in {"MEDIUM", "HIGH"})
    print(f"✅ 欺诈检测率: {detected}/{len(fraud_reports)} = {detected/len(fraud_reports):.0%}")
    return reports


if __name__ == "__main__":
    run_demo()
