---
title: 多 Agent 竞品情报系统 — 7×24 小时全天候竞品监控与预警
doc_type: knowledge
module: 10-MAS
topic: mas-competitive-intelligence-agent
status: stable
created: 2026-06-19
updated: 2026-06-19
owner: self
source: human+ai
roadmap_phase: phase3
---

# Skill Card: 多 Agent 竞品情报系统

> **论文**：Autonomous Competitive Intelligence via Multi-Agent Web Crawling and Analysis
> **arXiv**：2404.18291 | 2024 | **桥梁**: 多智能体系统 ↔ 广告分析 | **类型**: 商业化落地

## ① 算法原理

解决「每天早上花 1-2 小时手工刷竞品价格/Review/关键词，还是会漏掉竞品的关键变化，被竞品的促销策略打得措手不及」的业务问题。

**核心架构：4 个专业 Agent + 1 个情报汇聚 Agent**

- **价格监控 Agent**：每 4 小时抓取目标竞品 ASIN 价格，检测价格变化 > 5% 时触发预警，自动推送「竞品降价了，建议跟进/保持」决策
- **Review 分析 Agent**：每 24 小时分析新增竞品 Review，提取「客户吐槽点」（痛点 = 你的机会）和「客户赞美点」（优势 = 你需要超越的）
- **关键词追踪 Agent**：监控竞品自然排名变化，发现竞品在哪些词上排名上升（意味着他在投广告/做 SEO），帮你识别值得跟投的词
- **Listing 变化 Agent**：检测竞品标题/五点/主图变化，分析他们的 A/B 测试方向，学习优化思路

**情报汇聚 Agent（Intelligence Hub）**：收集 4 个 Agent 的原始信号，去重合并，按「威胁等级」排序（价格战 > 排名压制 > 产品升级 > 文案优化），输出每日情报简报。

**异常检测算法**：对价格/排名时序数据做 Z-score 检验（|z| > 2.5 = 异常），避免正常波动误报。

## ② 母婴出海应用案例

**场景A：竞品大促前 48 小时预警**
- 业务问题：竞品 Prime Day 前 48 小时开始囤货/调价，等你发现已经输了先机
- 数据要求：5-10 个主要竞品 ASIN 列表
- 部署方案：价格监控 Agent 发现竞品库存在 48 小时内快速减少（售罄 = 大促预备），触发预警
- 预期产出：提前 48h 知道竞品动向，Prime Day 广告预算提前到位，GMV 比上年同期 +$28,000

**场景B：发现竞品弱点，抢占 Review 差距**
- 业务问题：不知道竞品的真实缺陷，无法针对性地优化自己的产品卖点
- 数据要求：竞品 ASIN Review 数据（新增 3 星以下 Review）
- 部署方案：Review 分析 Agent 提取竞品一星/二星 Review 的高频痛点词，转化为己方 Listing 的强调卖点
- 预期产出：针对竞品痛点优化 Listing 后，转化率从 14% → 19%，年化增收 **$34,000**

## ③ 代码模板

```python
"""
多 Agent 竞品情报系统
价格/Review/关键词/Listing 四维监控 + 情报汇聚
"""
import math
import random
from dataclasses import dataclass, field
from enum import Enum
from typing import List, Dict, Optional, Tuple
from datetime import datetime, timedelta


class AlertLevel(Enum):
    INFO = 1        # 信息（一般变化）
    WARNING = 2     # 警告（值得关注）
    CRITICAL = 3    # 紧急（需要立即应对）


@dataclass
class CompetitorSignal:
    """竞品监控信号"""
    signal_id: str
    asin: str
    agent_type: str     # price/review/keyword/listing
    timestamp: str
    alert_level: AlertLevel
    description: str
    data: Dict
    action_suggestion: str


class ZScoreAnomalyDetector:
    """Z-score 异常检测（防止误报）"""
    
    def __init__(self, threshold: float = 2.5):
        self.threshold = threshold
        self.history: Dict[str, List[float]] = {}
    
    def update(self, series_id: str, value: float) -> Tuple[bool, float]:
        """
        Returns: (is_anomaly, z_score)
        """
        if series_id not in self.history:
            self.history[series_id] = []
        
        self.history[series_id].append(value)
        
        if len(self.history[series_id]) < 5:
            return False, 0.0
        
        data = self.history[series_id][-20:]  # 最近 20 个数据点
        mean = sum(data) / len(data)
        std = math.sqrt(sum((x - mean) ** 2 for x in data) / len(data))
        
        if std < 0.001:
            return False, 0.0
        
        z = abs(value - mean) / std
        return z > self.threshold, round(z, 2)


class PriceMonitorAgent:
    """价格监控 Agent"""
    
    def __init__(self, anomaly_detector: ZScoreAnomalyDetector):
        self.detector = anomaly_detector
        self.name = "价格监控Agent"
        self.signals_generated = 0
    
    def check_price(self, asin: str, current_price: float, our_price: float) -> Optional[CompetitorSignal]:
        is_anomaly, z_score = self.detector.update(f"price_{asin}", current_price)
        
        if not is_anomaly:
            return None
        
        price_change_pct = 0.0
        history = self.detector.history.get(f"price_{asin}", [current_price])
        if len(history) >= 2:
            prev_price = history[-2]
            price_change_pct = (current_price - prev_price) / prev_price * 100
        
        price_gap_pct = (our_price - current_price) / our_price * 100
        
        if abs(price_change_pct) >= 15:
            level = AlertLevel.CRITICAL
        elif abs(price_change_pct) >= 8:
            level = AlertLevel.WARNING
        else:
            level = AlertLevel.INFO
        
        direction = "⬇️ 降价" if price_change_pct < 0 else "⬆️ 涨价"
        
        if price_change_pct < -8:
            action = f"竞品大幅降价，考虑跟价 ${our_price * 0.95:.2f} 保住转化率"
        elif price_change_pct > 10:
            action = "竞品涨价，可小幅跟涨 3-5% 提升毛利"
        else:
            action = "观察 24h，量变化后再决策"
        
        self.signals_generated += 1
        return CompetitorSignal(
            signal_id=f"PRICE_{asin}_{self.signals_generated:04d}",
            asin=asin, agent_type="price",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            alert_level=level,
            description=f"{direction} {abs(price_change_pct):.1f}% (z={z_score:.1f}) | 价差 {price_gap_pct:.1f}%",
            data={"current": current_price, "change_pct": price_change_pct, "z_score": z_score},
            action_suggestion=action
        )


class ReviewAnalysisAgent:
    """Review 分析 Agent"""
    
    name = "Review分析Agent"
    
    # 痛点关键词库（生产环境用 NLP 提取）
    PAIN_KEYWORDS = {
        "quality": ["leak", "broke", "defect", "poor quality", "stopped working"],
        "usability": ["hard to use", "difficult", "complicated", "confusing"],
        "value": ["overpriced", "expensive", "not worth", "waste of money"],
        "shipping": ["late", "damaged", "wrong item", "missing"],
    }
    
    def analyze_reviews(self, asin: str, new_reviews: List[Dict]) -> Optional[CompetitorSignal]:
        if not new_reviews:
            return None
        
        # 分析低分 Review
        low_reviews = [r for r in new_reviews if r.get("stars", 5) <= 2]
        if len(low_reviews) < 2:
            return None
        
        # 统计痛点
        pain_counts: Dict[str, int] = {}
        for review in low_reviews:
            text = review.get("text", "").lower()
            for category, keywords in self.PAIN_KEYWORDS.items():
                if any(kw in text for kw in keywords):
                    pain_counts[category] = pain_counts.get(category, 0) + 1
        
        if not pain_counts:
            return None
        
        top_pain = max(pain_counts, key=pain_counts.get)
        pain_count = pain_counts[top_pain]
        
        return CompetitorSignal(
            signal_id=f"REVIEW_{asin}_{random.randint(1000,9999)}",
            asin=asin, agent_type="review",
            timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
            alert_level=AlertLevel.WARNING if pain_count >= 3 else AlertLevel.INFO,
            description=f"发现 {len(low_reviews)} 条差评，主要痛点：{top_pain}（{pain_count}次提及）",
            data={"low_review_count": len(low_reviews), "pain_points": pain_counts},
            action_suggestion=f"将「解决{top_pain}问题」加入己方 Listing 五点卖点，主动强调"
        )


class KeywordTrackingAgent:
    """关键词排名追踪 Agent"""
    
    def __init__(self, anomaly_detector: ZScoreAnomalyDetector):
        self.detector = anomaly_detector
        self.name = "关键词追踪Agent"
    
    def check_ranking(self, asin: str, keyword: str, current_rank: int) -> Optional[CompetitorSignal]:
        is_anomaly, z_score = self.detector.update(f"rank_{asin}_{keyword}", current_rank)
        
        if not is_anomaly:
            return None
        
        history = self.detector.history.get(f"rank_{asin}_{keyword}", [current_rank])
        if len(history) >= 2:
            prev_rank = history[-2]
            rank_change = prev_rank - current_rank  # 正数=排名上升（数字变小）
        else:
            return None
        
        if rank_change > 20:  # 排名快速上升
            return CompetitorSignal(
                signal_id=f"RANK_{asin}_{keyword[:10]}",
                asin=asin, agent_type="keyword",
                timestamp=datetime.now().strftime("%Y-%m-%d %H:%M"),
                alert_level=AlertLevel.WARNING,
                description=f"关键词「{keyword}」排名快升 {rank_change} 位（{prev_rank}→{current_rank}）",
                data={"keyword": keyword, "prev_rank": prev_rank, "current_rank": current_rank},
                action_suggestion=f"竞品加大「{keyword}」投放，建议跟投或调高该词竞价 15%"
            )
        return None


class IntelligenceHub:
    """情报汇聚 Agent：收集、去重、排优先级，输出日报"""
    
    def __init__(self):
        self.signals: List[CompetitorSignal] = []
        self.price_agent = None
        self.review_agent = None
        self.keyword_agent = None
    
    def ingest(self, signal: Optional[CompetitorSignal]):
        if signal:
            self.signals.append(signal)
    
    def generate_daily_brief(self) -> str:
        if not self.signals:
            return "今日无重要竞品变化"
        
        # 按威胁等级排序
        sorted_signals = sorted(self.signals, key=lambda s: s.alert_level.value, reverse=True)
        
        critical = [s for s in sorted_signals if s.alert_level == AlertLevel.CRITICAL]
        warnings = [s for s in sorted_signals if s.alert_level == AlertLevel.WARNING]
        
        lines = [
            "=" * 55,
            f"📊 竞品情报日报 {datetime.now().strftime('%Y-%m-%d')}",
            "=" * 55,
            f"🔴 紧急预警 {len(critical)} 条 | 🟡 关注 {len(warnings)} 条",
            ""
        ]
        
        for s in sorted_signals[:8]:  # 显示前 8 条
            icon = "🔴" if s.alert_level == AlertLevel.CRITICAL else "🟡"
            lines.append(f"{icon} [{s.agent_type.upper()}] {s.asin}")
            lines.append(f"   {s.description}")
            lines.append(f"   💡 建议：{s.action_suggestion}")
            lines.append("")
        
        return "\n".join(lines)


# 运行验证
if __name__ == "__main__":
    random.seed(42)
    
    detector = ZScoreAnomalyDetector(threshold=2.0)
    hub = IntelligenceHub()
    price_agent = PriceMonitorAgent(detector)
    review_agent = ReviewAnalysisAgent()
    keyword_agent = KeywordTrackingAgent(detector)
    
    # 模拟 2 周价格历史（稳定期）
    competitor_asin = "B08COMPETITOR"
    base_price = 28.99
    
    for i in range(14):
        price = base_price + random.gauss(0, 0.5)
        detector.update(f"price_{competitor_asin}", price)
    
    # 第 15 天：大幅降价（模拟大促备战）
    hub.ingest(price_agent.check_price(competitor_asin, 18.99, 29.99))  # 竞品降价 34%
    
    # Review 分析
    mock_reviews = [
        {"stars": 1, "text": "This product leaks badly, poor quality materials"},
        {"stars": 2, "text": "Hard to use, confusing instructions, leak after 2 days"},
        {"stars": 1, "text": "Stopped working after one week, not worth the money"},
        {"stars": 5, "text": "Great product, works perfectly"},
        {"stars": 2, "text": "Difficult to clean, hard to use for new moms"},
    ]
    hub.ingest(review_agent.analyze_reviews(competitor_asin, mock_reviews))
    
    # 关键词排名变化
    keyword = "breast pump portable"
    for rank in [45, 43, 44, 42, 43, 44, 43, 41, 40, 39]:
        detector.update(f"rank_{competitor_asin}_{keyword}", rank)
    hub.ingest(keyword_agent.check_ranking(competitor_asin, keyword, 18))  # 突然进入前 20
    
    # 输出情报日报
    print(hub.generate_daily_brief())
    
    # 验证
    assert len(hub.signals) >= 2, "应生成至少 2 条情报信号"
    critical_signals = [s for s in hub.signals if s.alert_level == AlertLevel.CRITICAL]
    assert len(critical_signals) >= 1, "应有至少 1 条紧急预警"
    assert all(s.action_suggestion for s in hub.signals), "每条信号都应有行动建议"
    
    print("[✓] 多 Agent 竞品情报系统 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Competitive-Price-Monitoring]]（价格监控的单维度基础实现）
- **前置（prerequisite）**：[[Skill-VOC-Aspect-Sentiment-Extraction]]（Review 分析的 NLP 基础能力）
- **延伸（extends）**：[[Skill-Amazon-A10-Algorithm-Ranking]]（关键词排名变化解读需要理解 A10 算法）
- **可组合（combinable）**：[[Skill-MAS-Ecommerce-Ops-Automation]]（竞品情报 + 运营自动化 → 「发现竞品降价」自动触发「我们的应对决策」完整闭环）

## ⑤ 商业价值评估

- **ROI 预估**：母婴跨境 50+ ASIN 卖家（GMV $500 万/年）：
  - 反应速度：竞品大促响应从 6-12h → 2h，Prime Day 额外 GMV **$28,000**
  - Review 洞察：针对竞品痛点优化 Listing，转化率 +5%，年化增收 **$34,000**
  - 省人力：竞品监控从 1.5h/天 → 0.1h/天，年化节省 $18,000 人力
  - **合计年化价值：约 $80,000**
- **实施难度**：⭐⭐⭐⭐☆（爬虫部分需要合规处理，使用官方 API 或合规数据服务）
- **优先级**：⭐⭐⭐⭐☆（竞品情报是进攻性工具，适合主动扩张阶段；防守阶段优先置信度门控和运营自动化）
- **合规提醒**：使用 Amazon SP-API 官方数据或合法数据服务商，不直接爬取 Amazon 页面
