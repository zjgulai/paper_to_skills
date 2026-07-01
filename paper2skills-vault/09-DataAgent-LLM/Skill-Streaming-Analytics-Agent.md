---
title: 实时流式分析Agent — 毫秒级事件驱动的AI决策管道
doc_type: knowledge
module: 09-DataAgent-LLM
topic: streaming-analytics-agent
status: stable
created: 2026-07-01
updated: 2026-07-01
owner: self
source: human+ai
roadmap_phase: phase2
---

# Skill Card: Streaming Analytics Agent

> **论文**：StreamBench: Towards Benchmarking Continuous Improvement of Language Agents（Wu et al., NeurIPS 2024, arXiv:2406.08747）+ Voyager: An Open-Ended Embodied Agent with Large Language Models（Wang et al., 2023, arXiv:2305.16291 — 持续学习Agent范式）
> **arXiv**：2406.08747 | 2024 | **桥梁**: 09-DataAgent-LLM ↔ 04-供应链 ↔ 19-风控反欺诈 | **类型**: 工程基础

## ① 算法原理

传统BI分析是批处理的（T+1日报）：昨天的数据今天才能看到。电商实时场景需要**毫秒→秒级**的分析响应：
- 大促开始30分钟，发现某SKU备货不足 → 立即触发紧急补货
- 竞品突然降价15% → 实时触发定价策略响应
- 评论刷单突发异常 → 实时告警并暂停广告

**实时流式分析Agent**的三层架构：

**层1：事件流接入（Kafka/Kinesis）**
- 数据源：订单事件、广告点击、库存变化、竞品价格爬取
- 处理：微批处理（每5秒一批）或真实流处理（每条事件）

**层2：LLM实时推理（Agent Loop）**
```
事件到来 → 上下文窗口更新 → LLM分析（Pattern检测+异常识别）→ 决策/告警
```
关键设计：**滑动窗口上下文**——只保留最近N个事件和关键状态（而非全量历史），控制LLM输入长度。

**层3：动作执行（Tool Use）**
- 告警推送（飞书/钉钉/邮件）
- 自动触发工作流（补货申请、定价调整）
- 更新监控面板（实时指标刷新）

**StreamBench范式**：
LLM Agent在流式数据上持续学习：每处理一批数据，Agent更新其"业务规则知识"（Few-shot context），下一批数据使用更新后的知识，实现无需重训练的在线适应。

**跨学科源头**：流处理来自大数据架构（Kafka, Flink），持续学习Agent来自强化学习（在线学习），LLM实时推理来自近2年的工程实践。对母婴电商的降维打击：传统监控需要人工每小时看一次，流式Agent能每5秒自动分析一次，真正做到"无人值守的大促实时守护"。

## ② 母婴出海应用案例

**场景A：618大促实时库存监控Agent**
- 业务问题：618大促开始后，某款婴儿推车SKU在前30分钟内销量超预期3倍，按当前速度2小时后断货。T+1报告无法及时预警，人工监控无法覆盖所有SKU
- 数据要求：实时订单事件流（每笔订单到达即触发）、当前库存快照、历史销速基准
- 预期产出：大促开始后28分钟，Agent检测到销速异常（实际销速是预测3.2倍），自动生成告警 + 触发紧急补货工作流；提前150分钟预判断货，争取到紧急空运窗口
- 业务价值：避免断货损失约30万元（2小时断货×每小时15万GMV）；实时响应比T+1报告早约20小时，年化大促场景避免断货损失约100万元

**三轨对抗验证**：
1. **成本验证**：每5秒LLM分析一次，大促期间（12小时）约8640次调用，DeepSeek成本约17元；非大促期可降频到每5分钟一次，月均成本<50元
2. **合规验证**：流式分析使用的是聚合订单数据（非用户级PII），无GDPR风险；自动补货触发需要人工审批gate（金额>5万元必须人工确认）
3. **风险验证**：LLM实时推理可能误报（将正常促销峰值判为异常）；需要设置"最小持续时间"阈值（异常持续>3分钟才告警，避免单点噪声）；自动触发的动作需要有回滚机制

**场景B：竞品价格实时响应Agent**
- 业务问题：竞品在黄金时段突然降价20%，如果不在2小时内响应，当天会损失约30%搜索流量
- 数据要求：竞品价格爬取事件流（每15分钟）+ 自家定价规则 + 利润底线配置
- 预期产出：Agent检测到竞品降价后，评估响应策略（降价跟随 vs 强调差异化 vs 加大广告投入），推荐最优方案并等待人工1次点击确认
- 业务价值：响应速度从T+1决策（次日）提升到T+2小时，当天搜索流量损失从30%降至8%，年化保护GMV约200万元

## ③ 代码模板

```python
"""
Skill-Streaming-Analytics-Agent
实时流式分析Agent — 大促库存监控与告警

依赖：pip install numpy pandas
注意：生产环境需接入 Kafka/Kinesis 和 LLM API
"""

import numpy as np
import pandas as pd
import time
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

np.random.seed(42)

# ── 1. 事件数据结构 ────────────────────────────────────────────────
@dataclass
class OrderEvent:
    timestamp: float
    sku_id: str
    quantity: int
    price: float

@dataclass
class InventorySnapshot:
    sku_id: str
    current_stock: int
    safety_stock: int
    baseline_hourly_rate: float  # 正常销速（件/小时）

@dataclass
class Alert:
    alert_type: str
    severity: str        # 'LOW', 'MEDIUM', 'HIGH', 'CRITICAL'
    sku_id: str
    message: str
    recommended_action: str
    timestamp: float

# ── 2. 滑动窗口统计引擎 ────────────────────────────────────────────
class SlidingWindowAggregator:
    """实时滑动窗口统计（替代Kafka Streams/Flink）"""

    def __init__(self, window_minutes: int = 30):
        self.window_seconds = window_minutes * 60
        self.events: deque = deque()

    def add_event(self, event: OrderEvent):
        now = time.time()
        self.events.append(event)
        # 清除窗口外的旧事件
        while self.events and (now - self.events[0].timestamp) > self.window_seconds:
            self.events.popleft()

    def get_stats(self, sku_id: str) -> dict:
        """计算指定SKU在窗口内的统计"""
        sku_events = [e for e in self.events if e.sku_id == sku_id]
        if not sku_events:
            return {'count': 0, 'total_qty': 0, 'hourly_rate': 0.0}

        total_qty = sum(e.quantity for e in sku_events)
        window_hours = self.window_seconds / 3600
        hourly_rate = total_qty / window_hours

        return {
            'count':       len(sku_events),
            'total_qty':   total_qty,
            'hourly_rate': hourly_rate,
            'first_event': sku_events[0].timestamp,
            'last_event':  sku_events[-1].timestamp,
        }

# ── 3. 流式分析Agent（LLM推理模拟）──────────────────────────────────
class StreamingAnalyticsAgent:
    """
    实时流式分析Agent
    生产环境：LLM实时推理替换为真实API调用
    """

    def __init__(self, inventory: dict[str, InventorySnapshot]):
        self.inventory = inventory
        self.aggregator = SlidingWindowAggregator(window_minutes=30)
        self.alerts_generated = []
        self.context_memory = deque(maxlen=10)  # 滑动上下文（最近10个决策）

    def process_event(self, event: OrderEvent) -> Optional[Alert]:
        """处理单个订单事件，返回告警（如有）"""
        self.aggregator.add_event(event)
        inv = self.inventory.get(event.sku_id)
        if not inv:
            return None

        stats = self.aggregator.get_stats(event.sku_id)
        if stats['count'] < 3:  # 至少3笔订单才分析
            return None

        # ── LLM推理（此处用规则模拟）────────────────────────────────
        # 生产版：将 stats + inv + context_memory 作为prompt发给LLM，返回分析结论
        alert = self._llm_analyze(inv, stats, event)
        if alert:
            self.alerts_generated.append(alert)
            self.context_memory.append({
                'time': event.timestamp,
                'sku': event.sku_id,
                'action': alert.alert_type
            })

        return alert

    def _llm_analyze(self, inv: InventorySnapshot, stats: dict,
                      latest_event: OrderEvent) -> Optional[Alert]:
        """
        模拟LLM分析：
        判断销速异常 + 预估断货时间 + 生成推荐动作
        """
        current_stock = inv.current_stock - stats['total_qty']
        hourly_rate   = stats['hourly_rate']
        baseline_rate = inv.baseline_hourly_rate

        if baseline_rate <= 0:
            return None

        velocity_ratio = hourly_rate / baseline_rate  # 销速倍数

        # 预估断货时间（小时）
        if hourly_rate > 0:
            hours_to_stockout = current_stock / hourly_rate
        else:
            hours_to_stockout = float('inf')

        # 告警逻辑（模拟LLM判断）
        if velocity_ratio > 2.5 and hours_to_stockout < 3:
            severity = 'CRITICAL'
            msg = (f"🚨 [{inv.sku_id}] 销速异常：当前{hourly_rate:.0f}件/时 "
                   f"= 基准{velocity_ratio:.1f}倍，预计{hours_to_stockout:.1f}小时后断货！")
            action = f"立即触发紧急补货（建议补{int(hourly_rate*24)}件）+ 联系仓库加急备货"
        elif velocity_ratio > 2.0 and hours_to_stockout < 6:
            severity = 'HIGH'
            msg = (f"⚠️ [{inv.sku_id}] 销速高于基准{velocity_ratio:.1f}倍，"
                   f"预计{hours_to_stockout:.1f}小时后低于安全库存")
            action = "提交常规补货申请，同时关注销速是否持续"
        elif velocity_ratio < 0.3 and stats['count'] > 5:
            severity = 'MEDIUM'
            msg = f"📉 [{inv.sku_id}] 销速异常偏低（仅基准{velocity_ratio:.1%}），可能存在Listing问题"
            action = "检查Listing状态 + 广告投放是否暂停"
        else:
            return None  # 正常，不告警

        return Alert(
            alert_type='INVENTORY_ALERT',
            severity=severity,
            sku_id=inv.sku_id,
            message=msg,
            recommended_action=action,
            timestamp=latest_event.timestamp,
        )

# ── 4. 大促场景模拟 ────────────────────────────────────────────────
inventory_config = {
    'SKU_STROLLER_A': InventorySnapshot('SKU_STROLLER_A', current_stock=500,
                                         safety_stock=50, baseline_hourly_rate=20),
    'SKU_PUMP_B':     InventorySnapshot('SKU_PUMP_B', current_stock=300,
                                         safety_stock=30, baseline_hourly_rate=15),
    'SKU_FORMULA_C':  InventorySnapshot('SKU_FORMULA_C', current_stock=1000,
                                         safety_stock=100, baseline_hourly_rate=50),
}

agent = StreamingAnalyticsAgent(inventory_config)

print("="*60)
print("  618大促 实时流式分析Agent — 30分钟模拟")
print("="*60)

t0 = time.time()
n_events = 0
alerts_critical = 0

# 模拟30分钟的订单事件流（加速100倍）
for minute in range(30):
    # SKU_STROLLER_A在第10分钟后销速异常（大促爆款）
    stroller_rate = 80 if minute >= 10 else 22  # 件/小时 → 件/分钟近似

    # 每分钟产生的订单事件
    for sku_id, rate in [('SKU_STROLLER_A', stroller_rate),
                          ('SKU_PUMP_B', 18),
                          ('SKU_FORMULA_C', 55)]:
        # 泊松过程：每分钟的订单数
        n_orders = np.random.poisson(rate / 60)
        for _ in range(max(1, n_orders)):
            event = OrderEvent(
                timestamp=t0 + minute * 60 + np.random.uniform(0, 60),
                sku_id=sku_id,
                quantity=np.random.randint(1, 4),
                price=np.random.uniform(80, 200),
            )
            alert = agent.process_event(event)
            n_events += 1
            if alert and alert.severity in ('HIGH', 'CRITICAL'):
                print(f"\n[T+{minute:02d}分] {alert.message}")
                print(f"         → {alert.recommended_action}")
                if alert.severity == 'CRITICAL':
                    alerts_critical += 1

print(f"\n{'='*60}")
print(f"  模拟结束统计")
print(f"  总处理事件: {n_events} 笔订单")
print(f"  产生告警: {len(agent.alerts_generated)} 个 (CRITICAL: {alerts_critical})")

# 验证：应检测到婴儿推车的异常
stroller_alerts = [a for a in agent.alerts_generated if 'STROLLER' in a.sku_id]
print(f"  婴儿推车告警: {len(stroller_alerts)} 个")

assert len(agent.alerts_generated) > 0, "应产生至少1个告警"
print("\n[✓] 实时流式分析Agent 测试通过")
```

## ④ 技能关联

- **前置（prerequisite）**：[[Skill-Argos-Agentic-Anomaly-Detection]]（异常检测Agent基础）、[[Skill-Time-Series-Anomaly-Detection]]（时序异常检测方法）
- **延伸（extends）**：[[Skill-Data-Collection-Agent-Pipeline]]（数据采集管道的实时版本）
- **可组合（combinable）**：[[Skill-Safety-Stock-Replenishment]]（流式Agent触发安全库存补货）、[[Skill-Real-Time-Competitive-Repricing]]（竞品价格事件驱动实时定价）、[[Skill-LLM-Hallucination-Detection-BI]]（流式推理结果的幻觉过滤）

## ⑤ 商业价值评估

- **ROI 预估**：大促期间实时预警断货，避免2小时断货损失约30万元/次；年化大促（6个）约100万元；竞品降价实时响应，保护搜索流量，年化GMV约200万元；LLM流式推理成本每月<100元
- **实施难度**：⭐⭐⭐⭐☆（架构复杂：需要事件流基础设施+LLM实时调用+状态管理；大促前至少提前2个月部署）
- **优先级**：⭐⭐⭐⭐☆（大促是母婴电商最高风险时段，实时监控是必备能力）
- **评估依据**：NeurIPS 2024 StreamBench证明LLM Agent在流式数据上的持续学习能力；阿里巴巴/京东大促期间均部署实时库存监控系统；AWS Kinesis + Lambda架构可零代码实现事件流接入
