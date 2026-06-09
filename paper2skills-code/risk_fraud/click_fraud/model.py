"""
Click Fraud Detection — 刷量检测：行为序列分析
paper2skills-code: 19-风控反欺诈 | 母婴出海跨境电商
"""
from __future__ import annotations
import math
from dataclasses import dataclass, field
from collections import Counter


@dataclass
class ClickEvent:
    event_id: str
    ip_address: str
    user_agent: str
    campaign_id: str
    ad_id: str
    timestamp_ms: int
    click_to_action_ms: int  # 点击到行动的时间（毫秒）
    converted: bool


@dataclass
class IPClickPattern:
    ip: str
    click_count: int
    unique_ads: int
    conversion_rate: float
    avg_click_to_action_ms: float
    time_span_ms: int


@dataclass
class ClickFraudResult:
    campaign_id: str
    total_clicks: int
    suspected_fraud_clicks: int
    fraud_rate: float
    fraud_ips: list[str]
    estimated_wasted_spend_usd: float
    recommendations: list[str]


class ClickFraudDetector:
    """刷量检测：IP 行为模式 + 时间序列分析"""

    def __init__(self, clicks_per_ip_threshold: int = 10,
                 conversion_rate_floor: float = 0.01,
                 bot_action_time_ms: float = 500.0):
        self.ip_click_threshold = clicks_per_ip_threshold
        self.cvr_floor = conversion_rate_floor
        self.bot_action_time = bot_action_time_ms

    def _analyze_ip(self, ip: str, events: list[ClickEvent]) -> IPClickPattern:
        unique_ads = len(set(e.ad_id for e in events))
        conversions = sum(1 for e in events if e.converted)
        cvr = conversions / max(len(events), 1)
        avg_action_time = sum(e.click_to_action_ms for e in events) / max(len(events), 1)
        time_span = max(e.timestamp_ms for e in events) - min(e.timestamp_ms for e in events)
        return IPClickPattern(
            ip=ip, click_count=len(events), unique_ads=unique_ads,
            conversion_rate=round(cvr, 4),
            avg_click_to_action_ms=round(avg_action_time, 1),
            time_span_ms=time_span,
        )

    def _is_bot(self, pattern: IPClickPattern) -> bool:
        if pattern.click_count > self.ip_click_threshold:
            return True
        if pattern.avg_click_to_action_ms < self.bot_action_time:
            return True
        if pattern.click_count > 5 and pattern.conversion_rate < self.cvr_floor:
            return True
        return False

    def detect(self, events: list[ClickEvent],
               cost_per_click_usd: float = 0.50) -> ClickFraudResult:
        from collections import defaultdict
        by_ip: dict[str, list[ClickEvent]] = defaultdict(list)
        for e in events:
            by_ip[e.ip_address].append(e)

        fraud_ips = []
        fraud_click_count = 0
        for ip, ip_events in by_ip.items():
            pattern = self._analyze_ip(ip, ip_events)
            if self._is_bot(pattern):
                fraud_ips.append(ip)
                fraud_click_count += len(ip_events)

        fraud_rate = fraud_click_count / max(len(events), 1)
        wasted = fraud_click_count * cost_per_click_usd

        recs = []
        if fraud_rate > 0.3:
            recs.append("⚠️ 欺诈率超过 30%，建议暂停该广告系列并向平台申诉")
        if len(fraud_ips) > 0:
            recs.append(f"封禁 {len(fraud_ips)} 个高风险 IP")
        recs.append("建议启用 Google Click Fraud Protection 或 ClickCease")

        return ClickFraudResult(
            campaign_id=events[0].campaign_id if events else "unknown",
            total_clicks=len(events), suspected_fraud_clicks=fraud_click_count,
            fraud_rate=round(fraud_rate, 3), fraud_ips=fraud_ips,
            estimated_wasted_spend_usd=round(wasted, 2),
            recommendations=recs,
        )


def run_click_fraud_demo():
    events = []
    # 正常用户
    for i in range(15):
        events.append(ClickEvent(f"E{i}", f"IP-{i%8}.real", "Mozilla/Chrome",
                                 "CAMP-001", f"AD-{i%3}", 1000*i, 3000+i*100, i%3==0))
    # 机器人 IP
    for i in range(30):
        events.append(ClickEvent(f"B{i}", "IP-BOT-99", "python-requests/2.28",
                                 "CAMP-001", f"AD-{i%5}", 100*i, 120, False))

    detector = ClickFraudDetector()
    result = detector.detect(events)

    print(f"=== 刷量检测报告：{result.campaign_id} ===")
    print(f"总点击: {result.total_clicks} | 疑似欺诈: {result.suspected_fraud_clicks}"
          f" | 欺诈率: {result.fraud_rate:.1%}")
    print(f"预估损失: ${result.estimated_wasted_spend_usd:.0f}")
    print(f"高风险 IP: {result.fraud_ips[:3]}{'...' if len(result.fraud_ips)>3 else ''}")
    for r in result.recommendations:
        print(f"  {r}")
    print(f"=== 刷量检测报告：{result.campaign_id} ===")


if __name__ == "__main__":
    run_click_fraud_demo()
