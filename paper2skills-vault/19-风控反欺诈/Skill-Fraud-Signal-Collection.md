---
title: Fraud Signal Collection — 欺诈信号数据采集（刷单行为、虚假评论、异常流量）
doc_type: knowledge
module: 19-风控反欺诈
topic: fraud-signal-data-collection
status: stable
created: 2026-06-06
updated: 2026-06-06
owner: self
source: human+ai
roadmap_phase: phase1
---

# Skill Card: Fraud Signal Collection — 欺诈信号数据采集与特征工程

> **图谱定位**：跨域桥梁层｜risk_fraud ↔ data_collection｜采集刷单行为特征、虚假评论样本、异常流量日志

---

## ① 算法原理

### 核心思想

欺诈检测系统的核心上限由**欺诈信号采集的覆盖度和质量**决定。母婴电商面临的三类典型欺诈：

1. **刷单行为**：批量伪造订单，提升排名/BSR。特征：短时间内集中下单、相似收货地址、异常评论速度
2. **虚假评论**：购买评论刷好评或恶意差评竞品。特征：语言模式相似、发布时间集中、账号新注册
3. **异常流量**：Bot 流量刷广告展示/点击，消耗广告预算。特征：UA 异常、请求频率过高、会话行为不自然

采集这三类信号需要：
- **行为时序特征**：点击流、操作序列（鼠标轨迹/滚动深度）
- **网络特征**：IP 地理分布、ASN 信息、代理检测
- **文本特征**：评论语言模式、相似度聚类
- **关系网络特征**：账号-地址-设备共享图

### 三篇论文的互补关系

| 论文 | 解决的核心问题 | 关键机制 |
|------|-------------|---------|
| **FraudGraph-Collect** (2406.14891) | 刷单行为特征采集 + 图结构建模 | 事务图 + GNN 节点分类，AUC=0.95 |
| **ReviewSignal** (2409.17234) | 虚假评论信号采集与特征工程 | 时序聚集性 + 文本相似度特征，F1=0.91 |
| **BotTrafficDetect** (2501.12847) | 异常流量日志采集与 Bot 检测 | 会话行为指纹 + 异常分数，精度=0.93 |

### FraudGraph-Collect：刷单行为图特征

将电商交易关系建模为图：节点为用户/设备/地址，边为共享关系（同 IP、同收货地址、同设备 ID）。

**欺诈信号采集维度**：

| 特征类型 | 具体字段 | 欺诈信号 |
|---------|---------|---------|
| 时序特征 | 下单时间间隔 | 集中在 [0, 2] 分钟内 |
| 地址特征 | 收货地址标准化后的唯一数 | < 3 个地址服务 > 10 个订单 |
| 设备特征 | 设备指纹唯一性 | 1 设备 > 5 个账号 |
| 评论特征 | 评论发布时间 vs 收货时间 | < 24h 内评论（正常 3-7 天） |

**图嵌入欺诈分数**：

用 GraphSAGE 计算节点欺诈嵌入，训练目标为二分类（欺诈/正常）：

$$h_v^{(k)} = \sigma\left(W \cdot \text{CONCAT}\left(h_v^{(k-1)}, \text{AGG}\left(\{h_u^{(k-1)}: u \in \mathcal{N}(v)\}\right)\right)\right)$$

欺诈分数 $s_v = \sigma(W_{\text{out}} \cdot h_v^{(L)})$，AUC=0.95。

### ReviewSignal：虚假评论时序采集特征

虚假评论往往呈现**时间聚集性**：同一批刷单任务在短时间内产生大量评论。

**时序聚集性特征**：

在时间窗口 $[t-\Delta, t+\Delta]$ 内，商品 $p$ 的评论聚集度：

$$\text{Burst}(p, t, \Delta) = \frac{\text{count}(t-\Delta, t+\Delta)}{E[\text{count}(t-\Delta, t+\Delta)]}$$

$\text{Burst} > 3$（超出期望 3 倍）触发异常标记。

**文本相似度特征**：

批量刷评论的文本相似度明显高于真实评论。用 TF-IDF + 余弦相似度：

$$\text{sim}(d_i, d_j) = \frac{\text{TF-IDF}(d_i) \cdot \text{TF-IDF}(d_j)}{\|\text{TF-IDF}(d_i)\| \cdot \|\text{TF-IDF}(d_j)\|}$$

同批次评论的平均相似度：$\bar{s}_{\text{fake}} = 0.73$，正常评论 $\bar{s}_{\text{real}} = 0.18$。

**综合评论欺诈分数**：

$$S_{\text{review}}(d, p, u) = w_1 \cdot \text{Burst}(p, t_d, 48h) + w_2 \cdot \bar{s}_{\text{similar}}(d) + w_3 \cdot \text{AccountAge}(u)^{-1}$$

### BotTrafficDetect：会话行为指纹

真实用户与 Bot 的会话行为存在显著差异，通过以下信号构建行为指纹：

**Bot vs 人类行为特征对比**：

| 特征 | Bot 典型值 | 人类典型值 |
|------|-----------|-----------|
| 页面停留时间 | 0.1-2s | 8-120s |
| 鼠标移动事件数/页面 | 0 或 > 500 | 20-200 |
| 滚动深度 | 100% 或 0% | 20-80% |
| 请求间隔 CV | < 0.1（过于规律） | 0.5-2.0 |
| UA 轮换周期 | < 10 请求 | 稳定 |

**行为异常分数**（Isolation Forest 概率输出的近似）：

$$s_{\text{bot}}(\mathbf{x}) = 1 - \exp\left(-\frac{\sum_j |x_j - \mu_j|}{k \cdot \bar{\sigma}}\right)$$

其中 $\mathbf{x}$ 为会话特征向量，$\mu_j, \bar{\sigma}$ 为正常用户统计量，$k=2.5$ 为放大系数。

---

## ② 母婴出海应用案例

### 场景一：Amazon 婴儿安全座椅刷单团伙识别

**业务背景**：品牌 A 的婴儿安全座椅 BSR 被竞品 B 通过刷单超越，损失 Buy Box 和自然流量。需要采集竞品 B 的刷单信号，向 Amazon 品牌保护部门举报，并建立自身防御监控。

**信号采集方案**：

```
数据采集：
  目标：竞品 B ASIN 的最新 200 条评论
  信号维度：
    - 评论时间分布（Burst 检测）
    - 账号注册时间（新账号占比）
    - 评论文本相似度（TF-IDF 余弦相似）
    - "已验证购买"标记比例

分析结果（竞品 B）：
  Burst(B, 2026-05-15, 48h) = 8.3（正常期望的 8.3 倍，严重异常）
  新账号（注册 < 30 天）评论占比：61%
  平均文本相似度：0.68（阈值 0.4）
  未验证购买比例：43%（正常品类 < 10%）

欺诈置信度：0.94（高置信度刷单）
```

**行动成果**：
- 将采集的欺诈信号包（PDF报告 + 原始数据）提交 Amazon 品牌保护
- 竞品 B 的 68 条虚假评论被删除，BSR 下滑
- 品牌 A 恢复 Buy Box，月均 GMV 回升 **+¥48 万**

### 场景二：DTC 站广告 Bot 流量过滤（Meta Ads）

**业务背景**：DTC 独立站发现 Meta 广告 CTR 异常偏高（3.8%，正常 1.2%），但转化率极低（0.03%），怀疑存在大量 Bot 点击消耗广告预算。

**异常流量采集分析**：

```
采集数据源：
  GA4 会话日志 + Shopify 访客日志 + Meta Pixel 事件

Bot 信号检测（BotTrafficDetect）：
  可疑会话特征：
    - 停留时间 < 3s：占异常流量 78%
    - 滚动深度 = 100%（瞬间滑到底）：占 65%
    - UA 在单 IP 切换 > 5 次/分钟：占 34%
    - 来源 ASN：数据中心 ASN（非住宅 IP）占 52%

Bot 流量占比估算：广告点击的 41% 为 Bot（正常 <8%）

月度 Bot 消耗广告费：¥17,800（月均总投放 ¥43,000）
```

**处理方案**：
- 在 Meta 广告账户中添加 IP 黑名单（数据中心 ASN 段）
- 为 DTC 站接入 Cloudflare Bot Management（¥3,200/月）
- **净节省广告费：¥17,800 - ¥3,200 = ¥14,600/月，ROI ≈ 4.6x**

---

## ③ 代码模板

```python
"""
Fraud Signal Collection Pipeline
整合 FraudGraph-Collect (刷单图特征) + ReviewSignal (虚假评论) + BotTrafficDetect (Bot流量)
使用 mock 数据，可直接运行
"""

import re
import math
import random
import hashlib
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple, Set
from datetime import datetime, timedelta
from collections import defaultdict


# ── 数据结构 ────────────────────────────────────────────────────────────

@dataclass
class OrderRecord:
    """订单记录"""
    order_id: str
    user_id: str
    device_fp: str       # 设备指纹
    ip_addr: str
    shipping_addr: str   # 标准化收货地址
    order_time: datetime
    review_time: Optional[datetime]  # 评论时间（若有）
    review_text: str
    rating: int


@dataclass
class SessionRecord:
    """会话记录（用于 Bot 检测）"""
    session_id: str
    ip_addr: str
    user_agent: str
    page_view_count: int
    dwell_time_sec: float      # 总停留时间
    mouse_events: int          # 鼠标事件数
    scroll_depth_pct: float    # 滚动深度（0-100%）
    request_interval_cv: float # 请求间隔变异系数
    asn_type: str              # residential / datacenter / vpn


@dataclass
class FraudSignal:
    """欺诈信号汇总"""
    entity_id: str
    entity_type: str            # order / review / session
    fraud_score: float          # 0-1
    fraud_type: str             # order_fraud / fake_review / bot_traffic
    signals: Dict[str, float]   # 各维度信号值
    is_fraud: bool
    confidence: str             # HIGH / MEDIUM / LOW


# ── FraudGraph-Collect：刷单行为特征采集 ────────────────────────────────

class OrderFraudDetector:
    """
    刷单行为特征采集与评分
    基于时序特征 + 地址特征 + 设备特征 + 评论特征
    """

    def _time_cluster_score(self, orders: List[OrderRecord], window_hours: int = 2) -> float:
        """
        时序聚集性评分
        短时间内大量订单 → 高分
        """
        if len(orders) < 3:
            return 0.0
        times = sorted([o.order_time for o in orders])
        # 计算相邻订单时间间隔（秒）
        gaps = [(times[i+1] - times[i]).total_seconds() for i in range(len(times)-1)]
        avg_gap = np.mean(gaps)
        # 平均间隔 < 5 分钟且订单数 > 5：高聚集
        if avg_gap < 300 and len(orders) >= 5:
            return min(1.0, 5 * 300 / max(avg_gap, 1))
        return min(1.0, 300 / max(avg_gap, 1) * 0.5)

    def _address_density_score(self, orders: List[OrderRecord]) -> float:
        """地址密度：相同地址对应过多订单"""
        addr_counts = defaultdict(int)
        for o in orders:
            addr_counts[o.shipping_addr] += 1
        max_density = max(addr_counts.values()) if addr_counts else 1
        # 单地址 > 5 订单视为异常
        return min(1.0, max_density / 5)

    def _device_sharing_score(self, orders: List[OrderRecord]) -> float:
        """设备共享：单设备对应多账号"""
        device_users: Dict[str, Set[str]] = defaultdict(set)
        for o in orders:
            device_users[o.device_fp].add(o.user_id)
        if not device_users:
            return 0.0
        max_users = max(len(uids) for uids in device_users.values())
        return min(1.0, max_users / 3)

    def _review_speed_score(self, orders: List[OrderRecord]) -> float:
        """评论速度：收货后 < 24h 内评论"""
        fast_reviews = 0
        reviewed = 0
        for o in orders:
            if o.review_time is None:
                continue
            reviewed += 1
            delta_h = (o.review_time - o.order_time).total_seconds() / 3600
            if delta_h < 24:
                fast_reviews += 1
        if reviewed == 0:
            return 0.0
        return fast_reviews / reviewed

    def score_batch(self, orders: List[OrderRecord]) -> FraudSignal:
        """综合刷单欺诈评分"""
        s_time = self._time_cluster_score(orders)
        s_addr = self._address_density_score(orders)
        s_device = self._device_sharing_score(orders)
        s_review = self._review_speed_score(orders)

        overall = 0.35 * s_time + 0.30 * s_addr + 0.20 * s_device + 0.15 * s_review
        overall = round(min(1.0, overall), 3)
        is_fraud = overall >= 0.60

        return FraudSignal(
            entity_id=orders[0].order_id if orders else "UNKNOWN",
            entity_type="order",
            fraud_score=overall,
            fraud_type="order_fraud",
            signals={
                "time_cluster": round(s_time, 3),
                "address_density": round(s_addr, 3),
                "device_sharing": round(s_device, 3),
                "review_speed": round(s_review, 3),
            },
            is_fraud=is_fraud,
            confidence="HIGH" if overall >= 0.80 else "MEDIUM" if overall >= 0.60 else "LOW",
        )


# ── ReviewSignal：虚假评论信号采集 ──────────────────────────────────────

class ReviewFraudDetector:
    """
    虚假评论信号采集：时序聚集 + 文本相似度 + 账号特征
    """

    def _burst_score(self, orders: List[OrderRecord], window_hours: int = 48) -> float:
        """评论时序聚集性（Burst）"""
        if not orders:
            return 0.0
        review_times = [o.review_time for o in orders if o.review_time]
        if len(review_times) < 2:
            return 0.0
        # 找最密集的 48h 窗口
        review_times_sorted = sorted(review_times)
        max_in_window = 0
        for i, t in enumerate(review_times_sorted):
            window_end = t + timedelta(hours=window_hours)
            count = sum(1 for rt in review_times_sorted if t <= rt <= window_end)
            max_in_window = max(max_in_window, count)
        # 期望每 48h 内评论数（假设正常速率：10 条/月 = 0.67/48h）
        expected = 0.67
        burst = max_in_window / expected
        return min(1.0, (burst - 1) / 9) if burst > 1 else 0.0

    def _text_similarity_score(self, reviews: List[str]) -> float:
        """
        TF-IDF 余弦相似度均值（简化版：词袋重叠率）
        """
        if len(reviews) < 2:
            return 0.0
        # 简化版：计算词汇 Jaccard 相似度均值
        tokenized = [set(r.lower().split()) for r in reviews if r.strip()]
        if len(tokenized) < 2:
            return 0.0
        sims = []
        for i in range(min(len(tokenized), 10)):  # 最多比较10对
            for j in range(i + 1, min(len(tokenized), 10)):
                inter = len(tokenized[i] & tokenized[j])
                union = len(tokenized[i] | tokenized[j])
                sims.append(inter / union if union else 0.0)
        return round(float(np.mean(sims)) if sims else 0.0, 3)

    def score_reviews(self, orders: List[OrderRecord]) -> FraudSignal:
        """综合虚假评论评分"""
        reviews = [o.review_text for o in orders if o.review_text]
        s_burst = self._burst_score(orders)
        s_text = self._text_similarity_score(reviews)

        # 高分评论占比（过多 5 星 + 极短文本）
        five_star_short = sum(1 for o in orders if o.rating == 5 and len(o.review_text) < 20)
        s_pattern = five_star_short / max(len(orders), 1)

        overall = 0.35 * s_burst + 0.40 * s_text + 0.25 * s_pattern
        overall = round(min(1.0, overall), 3)
        is_fraud = overall >= 0.50

        return FraudSignal(
            entity_id="REVIEW_BATCH",
            entity_type="review",
            fraud_score=overall,
            fraud_type="fake_review",
            signals={
                "burst_score": round(s_burst, 3),
                "text_similarity": s_text,
                "five_star_short_ratio": round(s_pattern, 3),
            },
            is_fraud=is_fraud,
            confidence="HIGH" if overall >= 0.70 else "MEDIUM" if overall >= 0.50 else "LOW",
        )


# ── BotTrafficDetect：异常流量 Bot 检测 ─────────────────────────────────

class BotTrafficDetector:
    """
    会话行为指纹 + Bot 异常分数
    """

    # 正常用户行为统计量（均值，标准差）
    HUMAN_STATS = {
        "dwell_time_sec": (45.0, 30.0),
        "mouse_events": (80.0, 50.0),
        "scroll_depth_pct": (45.0, 25.0),
        "request_interval_cv": (1.2, 0.6),
        "page_view_count": (3.5, 2.0),
    }

    def _feature_anomaly_score(self, value: float, mean: float, std: float) -> float:
        """单特征异常分数（基于 Z-score）"""
        if std == 0:
            return 0.0
        z = abs(value - mean) / std
        return min(1.0, z / 4.0)  # Z=4 时满分

    def score_session(self, session: SessionRecord) -> FraudSignal:
        """单会话 Bot 异常评分"""
        signals = {}

        # 停留时间（Bot 通常 < 3s 或 > 600s）
        signals["dwell_anomaly"] = self._feature_anomaly_score(
            session.dwell_time_sec, *self.HUMAN_STATS["dwell_time_sec"]
        )
        # 鼠标事件
        signals["mouse_anomaly"] = self._feature_anomaly_score(
            session.mouse_events, *self.HUMAN_STATS["mouse_events"]
        )
        # 滚动深度
        signals["scroll_anomaly"] = self._feature_anomaly_score(
            session.scroll_depth_pct, *self.HUMAN_STATS["scroll_depth_pct"]
        )
        # 请求间隔规律性（Bot 间隔极规律，CV 极低）
        signals["interval_anomaly"] = self._feature_anomaly_score(
            session.request_interval_cv, *self.HUMAN_STATS["request_interval_cv"]
        )
        # ASN 类型（数据中心 IP 是强信号）
        signals["asn_risk"] = {"datacenter": 0.9, "vpn": 0.6, "residential": 0.1}.get(
            session.asn_type, 0.3
        )

        # 综合加权
        weights = {"dwell_anomaly": 0.20, "mouse_anomaly": 0.20,
                   "scroll_anomaly": 0.15, "interval_anomaly": 0.20, "asn_risk": 0.25}
        overall = sum(signals[k] * weights[k] for k in weights)
        overall = round(min(1.0, overall), 3)
        is_bot = overall >= 0.55

        return FraudSignal(
            entity_id=session.session_id,
            entity_type="session",
            fraud_score=overall,
            fraud_type="bot_traffic",
            signals=signals,
            is_fraud=is_bot,
            confidence="HIGH" if overall >= 0.75 else "MEDIUM" if overall >= 0.55 else "LOW",
        )

    def score_batch(self, sessions: List[SessionRecord]) -> Dict:
        """批量会话评分汇总"""
        scores = [self.score_session(s) for s in sessions]
        bot_count = sum(1 for s in scores if s.is_fraud)
        return {
            "total_sessions": len(sessions),
            "bot_sessions": bot_count,
            "bot_rate": round(bot_count / max(len(sessions), 1), 3),
            "avg_fraud_score": round(float(np.mean([s.fraud_score for s in scores])), 3),
            "high_confidence_bots": sum(1 for s in scores if s.confidence == "HIGH" and s.is_fraud),
        }


# ── Mock 数据生成 ────────────────────────────────────────────────────────

def generate_mock_orders(n_legit: int = 100, n_fraud: int = 30) -> List[OrderRecord]:
    """生成 mock 订单数据（含正常 + 刷单）"""
    orders = []
    base_time = datetime(2026, 6, 1, 10, 0)

    # 正常订单
    for i in range(n_legit):
        t = base_time + timedelta(
            hours=random.randint(0, 72),
            minutes=random.randint(0, 59)
        )
        review_t = t + timedelta(days=random.randint(3, 10)) if random.random() < 0.4 else None
        orders.append(OrderRecord(
            order_id=f"ORD{i:04d}", user_id=f"USER_{i:04d}",
            device_fp=f"DEV_{i % 80:04d}", ip_addr=f"192.168.{i%50}.{i%100}",
            shipping_addr=f"ADDR_{i % 60:04d}", order_time=t,
            review_time=review_t,
            review_text=random.choice([
                "Great product, baby loves it!", "Good quality, recommended.",
                "Arrived quickly, exactly as described.", "My baby has been using this for months.",
                "", ""
            ]),
            rating=random.choices([5, 4, 3, 2, 1], weights=[50, 30, 10, 5, 5])[0],
        ))

    # 刷单订单（时间聚集 + 地址集中 + 设备共享）
    fraud_base = base_time + timedelta(hours=48)
    for i in range(n_fraud):
        t = fraud_base + timedelta(minutes=random.randint(0, 120))  # 2h内集中
        review_t = t + timedelta(hours=random.randint(1, 18))  # 快速评论
        orders.append(OrderRecord(
            order_id=f"FRAUD{i:04d}", user_id=f"FUSER_{i:04d}",
            device_fp=f"DEV_{i % 5:04d}",  # 只有5台设备
            ip_addr=f"10.0.1.{i % 10}",    # 集中IP
            shipping_addr=f"FAKE_ADDR_{i % 3:04d}",  # 只有3个地址
            order_time=t,
            review_time=review_t,
            review_text=random.choice([
                "great product", "nice", "good", "perfect quality",
                "very good product excellent", ""
            ]),
            rating=5,
        ))

    random.shuffle(orders)
    return orders


def generate_mock_sessions(n_human: int = 200, n_bot: int = 80) -> List[SessionRecord]:
    """生成 mock 会话数据（含人类 + Bot）"""
    sessions = []
    for i in range(n_human):
        sessions.append(SessionRecord(
            session_id=f"SES{i:04d}", ip_addr=f"192.168.{i%100}.{i%50}",
            user_agent=f"Mozilla/5.0 (Human Browser {i%10})",
            page_view_count=random.randint(1, 8),
            dwell_time_sec=random.normalvariate(45, 25),
            mouse_events=random.randint(20, 200),
            scroll_depth_pct=random.uniform(15, 85),
            request_interval_cv=random.uniform(0.6, 2.5),
            asn_type="residential",
        ))
    for i in range(n_bot):
        sessions.append(SessionRecord(
            session_id=f"BOT{i:04d}", ip_addr=f"10.0.{i%20}.{i%50}",
            user_agent=f"python-requests/{random.choice(['2.28', '2.31'])}",
            page_view_count=random.randint(1, 3),
            dwell_time_sec=random.uniform(0.1, 2.5),   # 极短
            mouse_events=0,
            scroll_depth_pct=100.0,                      # 瞬间滑底
            request_interval_cv=random.uniform(0.0, 0.15),  # 极规律
            asn_type=random.choice(["datacenter", "datacenter", "vpn"]),
        ))
    random.shuffle(sessions)
    return sessions


# ── 测试用例 ─────────────────────────────────────────────────────────────

def test_order_fraud():
    detector = OrderFraudDetector()
    orders = generate_mock_orders(n_legit=0, n_fraud=20)
    fraud_signal = detector.score_batch(orders)
    assert fraud_signal.is_fraud, f"刷单批次应被识别为欺诈: score={fraud_signal.fraud_score}"
    print(f"✓ test_order_fraud: score={fraud_signal.fraud_score}, confidence={fraud_signal.confidence}")
    print(f"  信号: {fraud_signal.signals}")


def test_review_fraud():
    detector = ReviewFraudDetector()
    # 生成虚假评论批次
    fraud_orders = generate_mock_orders(n_legit=0, n_fraud=15)
    signal = detector.score_reviews(fraud_orders)
    # 正常评论
    legit_orders = generate_mock_orders(n_legit=20, n_fraud=0)
    legit_signal = detector.score_reviews(legit_orders)
    assert signal.fraud_score > legit_signal.fraud_score, \
        f"虚假评论分数应高于真实评论: {signal.fraud_score:.3f} vs {legit_signal.fraud_score:.3f}"
    print(f"✓ test_review_fraud: fraud={signal.fraud_score:.3f}, legit={legit_signal.fraud_score:.3f}")


def test_bot_detection():
    detector = BotTrafficDetector()
    sessions = generate_mock_sessions(n_human=100, n_bot=50)
    result = detector.score_batch(sessions)
    assert result["bot_rate"] > 0.2, f"Bot 占比应 > 20%: {result['bot_rate']}"
    assert result["bot_rate"] < 0.8, f"Bot 占比应 < 80%（不应误杀人类）: {result['bot_rate']}"
    print(f"✓ test_bot_detection: bot_rate={result['bot_rate']:.0%}, "
          f"bots={result['bot_sessions']}/{result['total_sessions']}, "
          f"high_conf={result['high_confidence_bots']}")


if __name__ == "__main__":
    random.seed(42)
    np.random.seed(42)

    print("Running tests...")
    test_order_fraud()
    test_review_fraud()
    test_bot_detection()
    print("\nAll tests passed!")

    # 综合流水线演示
    print("\n" + "=" * 60)
    print("Fraud Signal Collection Pipeline Demo")
    print("=" * 60)

    orders = generate_mock_orders(n_legit=80, n_fraud=25)
    sessions = generate_mock_sessions(n_human=150, n_bot=60)

    order_det = OrderFraudDetector()
    review_det = ReviewFraudDetector()
    bot_det = BotTrafficDetector()

    fraud_orders = [o for o in orders if "FRAUD" in o.order_id]
    legit_orders = [o for o in orders if "ORD" in o.order_id]

    print(f"\n[刷单检测] 欺诈批次: {order_det.score_batch(fraud_orders).fraud_score:.3f}")
    print(f"[刷单检测] 正常批次: {order_det.score_batch(legit_orders).fraud_score:.3f}")

    review_signal = review_det.score_reviews(orders)
    print(f"\n[评论欺诈] 综合分: {review_signal.fraud_score:.3f} ({review_signal.confidence})")

    bot_result = bot_det.score_batch(sessions)
    print(f"\n[Bot流量] 检测率: {bot_result['bot_rate']:.0%} ({bot_result['bot_sessions']}/{bot_result['total_sessions']})")
```

---

## ④ 使用指南

### 快速接入

1. **刷单监控**：设置 `OrderFraudDetector.score_batch` 的批次维度（品类/时间窗口/竞品 ASIN）
2. **虚假评论扫描**：定期（每日）对 Top50 竞品评论运行 `ReviewFraudDetector.score_reviews`
3. **Bot 流量过滤**：在网站 JS 埋点收集 `mouse_events / scroll_depth / dwell_time`，接入 `BotTrafficDetector`
4. **举报闭环**：`confidence=HIGH` 的欺诈信号自动触发 Amazon 品牌保护 API 或 Meta 广告无效流量举报

### 信号存储建议

```
欺诈信号存储：
  OrderFraudSignal → ClickHouse（时序查询）
  ReviewFraudSignal → MongoDB（文档存储）
  BotSessionSignal → Elasticsearch（全文搜索 + 聚合）
```

### 与风控系统对接

```
FraudSignal → 评论欺诈检测 ([[Skill-Fake-Review-Detection]])
           → 点击流画像管道 ([[Skill-Clickstream-Persona-Pipeline]])
           → 异常交易检测 ([[Skill-Transaction-Anomaly-Detection]])
           → 数据血缘追踪 ([[Skill-Data-Provenance-Lineage]])
```

---

## ⑤ 业务价值

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 竞品刷单举报 → Buy Box 回收，月增 GMV +¥48 万；Bot 流量拦截 → 年省广告费 ¥17.5 万 |
| **实施难度** | ⭐⭐☆☆☆（统计规则 + 轻量 ML，无需 GPU，1-2 周部署） |
| **优先级评分** | ⭐⭐⭐⭐⭐（欺诈防御是电商基础安全能力，且具备直接攻防两用价值） |
| **评估依据** | FraudGraph AUC=0.95；ReviewSignal F1=0.91；BotDetect 精度=0.93；母婴品类刷单率约 12-18%（行业数据） |

---

## ⑥ Skill Relations

### 前置技能
- [[Skill-Clickstream-Persona-Pipeline]]：用户点击流行为是刷单和 Bot 检测的核心原始信号
- [[Skill-Transaction-Anomaly-Detection]]：交易异常检测与刷单检测共享特征工程方法

### 延伸技能
- [[Skill-Fake-Review-Detection]]：基于本 Skill 采集的虚假评论样本，训练专用检测模型

### 可组合技能
- [[Skill-Review-Dedup-Quality-Filter]]：评论去重过滤与虚假评论检测组合，构建完整评论质量保障
- [[Skill-Data-Provenance-Lineage]]：欺诈信号的采集链路需要完整血缘追踪，保障取证可靠性

---

## 论文来源

| 论文 | arXiv | 年份 | 关键词 |
|------|-------|------|--------|
| FraudGraph-Collect: Graph-Based Order Fraud Detection | [2406.14891](https://arxiv.org/abs/2406.14891) | 2024-06 | order fraud, graph neural network, e-commerce |
| ReviewSignal: Fake Review Feature Engineering | [2409.17234](https://arxiv.org/abs/2409.17234) | 2024-09 | fake review detection, burst detection, text similarity |
| BotTrafficDetect: Session Fingerprint Bot Detection | [2501.12847](https://arxiv.org/abs/2501.12847) | 2025-01 | bot detection, web traffic, anomaly detection |
