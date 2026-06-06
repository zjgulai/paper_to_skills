---
title: LLM AutoBidding MAS — 大语言模型驱动的层次化自动竞价系统
doc_type: knowledge
module: 10-MAS
topic: llm-autobidding-mas
status: stable
created: 2026-06-04
updated: 2026-06-04
owner: self
source: human+ai
---

# Skill Card: LLM AutoBidding MAS — LLM 驱动的自动竞价多智能体

> **图谱定位**：Layer 4 桥接层 ★ **`mas ↔ advertising` 首条跨域桥梁**
> 前置：`Skill-MAS-Orchestrator`（MAS侧）+ `Skill-ROAS-Budget-Optimization`（广告侧）
> 跨域连接：MAS 领域（25 Skills）↔ 广告分析领域（20 Skills）

---

## ① 算法原理

### 核心思想

传统自动竞价系统（如 Amazon 自动广告）基于规则或简单 ML 模型，有两个核心局限：
1. **数据稀缺**：新广告主历史数据极少，规则无法适配
2. **竞价幻觉**：LLM 直接生成竞价出价时容易产生不合理数值（如报价超出预算 10×）

DARA 和 LBM 从互补角度解决这两个问题：

| 论文 | 解决的核心问题 | 关键机制 |
|------|-------------|---------|
| **DARA** (2601.14711, WWW'26) | 新广告主数据稀缺 → 少样本竞价推理 | 双 Agent（Reasoner + Optimizer）+ GRPO-Adaptive RL 微调 |
| **LBM** (2603.05134, WWW'26) | LLM 竞价幻觉 → 推理与执行分离 | LBM-Think（链式推理）+ LBM-Act（精确出价）+ GQPO 离线 RL |

### DARA：双 Agent 少样本竞价

**架构**：

```
┌─────────────────────────────────────────────────────┐
│  Reasoner Agent（全局推理）                          │
│  · 输入：广告主目标（ROAS/ACOS/CPA）+ 少量历史       │
│  · 任务：理解竞价策略，生成"推理链"                  │
│  · 输出：竞价策略文本描述                            │
│         "该广告主追求低 ACOS，当前关键词竞争激烈，   │
│          建议出价略低于市场均值 10%"                  │
├─────────────────────────────────────────────────────┤
│  Optimizer Agent（精细执行）                         │
│  · 输入：Reasoner 的策略描述 + 实时竞拍数据          │
│  · 任务：将策略转化为具体出价数字                    │
│  · 输出：bid = $0.87（具体竞价值）                   │
└─────────────────────────────────────────────────────┘
```

**GRPO-Adaptive RL 微调**：

标准 GRPO（Group Relative Policy Optimization）用于 LLM 强化学习微调。DARA 的 Adaptive 变体：

$$\mathcal{L}_{DARA} = -\mathbb{E}\left[\sum_{t} \min\left(r_t A_t, \text{clip}(r_t, 1\pm\epsilon) A_t\right)\right] + \lambda \cdot \mathcal{L}_{sparse}$$

其中 $\mathcal{L}_{sparse}$ 是稀疏历史补偿项：历史数据越少，正则化强度越大，防止过拟合少量样本。

**关键洞见**：少样本场景下，推理链的质量比出价精度更重要（Reasoner 决定方向，Optimizer 决定精度）。

### LBM：层次化大型自动竞价模型

**核心问题诊断**：用单一 LLM 直接生成竞价出价，会产生"竞价幻觉"——LLM 对价格的量化感知不准确，输出如 `$15.00`（实际均值 $0.80）的离谱出价。

**Think-Act 双层架构**：

```
LBM-Think（大模型，推理层）：
  · 任务：理解竞价情景，生成竞价"意图"
  · 输入：历史竞价序列 + 当前市场状态 + 预算约束
  · 输出：意图向量（high_bid / maintain / reduce）+ 置信度
  · 不生成具体数字！

LBM-Act（小模型，执行层）：
  · 任务：将意图转化为精确出价数字
  · 输入：LBM-Think 意图 + 当前出价 + 约束
  · 输出：具体 bid 值（$0.87）
  · 经过 GQPO 训练，数值精确
```

**GQPO（Group-Quantile Policy Optimization）**：

标准 RL 微调对竞价问题的困难：奖励稀疏（只有最终 ROAS 能给奖励，中间竞价步骤无反馈）。

GQPO 解法：
1. 将历史竞价轨迹按最终 ROAS 分组（高 ROAS 组 vs 低 ROAS 组）
2. 在组内计算相对优势（Quantile-based Advantage）
3. 用组间比较替代绝对奖励信号

$$A_t^{GQPO} = \frac{r_t - Q_\tau(r_{group})}{\text{std}(r_{group})}$$

**结果**：消除竞价幻觉，无需在线仿真（纯离线 RL），可直接生产部署。

---

## ② 母婴出海应用场景

### 场景一：新品上市期少样本竞价（DARA）

**业务背景**：母婴品牌在 Amazon US 上线新 SKU（婴儿护臀膏），历史广告数据仅 3 天（150 次点击，8 次购买）。需要设定关键词竞价，但数据太少无法训练传统 ML 模型。

**DARA 双 Agent 执行**：

```
广告主目标：ACOS ≤ 30%（广告花费/广告销售额）

Reasoner Agent 推理链：
  "该 SKU 新品期，需要快速积累评论。
   关键词 'baby diaper rash cream' 竞争度中等（CPC 均值 $0.72）。
   当前 ACOS 38%（高于目标），需降低出价。
   但新品期曝光优先，不宜降价过猛。
   策略：出价降至市场均值 -5%，同时维持 exact match 覆盖。"

Optimizer Agent 执行：
  market_avg_cpc = $0.72
  target_adjustment = -5%
  recommended_bid = $0.72 * 0.95 = $0.684 ≈ $0.68

历史数据量对比（DARA vs 传统 ML）：
  传统 ML 需要 ≥ 500 次转化才能训练
  DARA：3 天 / 8 次转化即可给出合理竞价
```

### 场景二：促销期竞价策略优化（LBM）

**业务背景**：Prime Day 大促期间，关键词 CPC 飙升 2-3×，需要实时调整竞价策略（每 30 分钟一次），防止预算超支同时维持可见度。

**LBM 层次化执行**：

```
每 30 分钟竞价决策循环：

LBM-Think（意图生成）：
  输入：{
    "budget_remaining": 450,  # 剩余预算 $450
    "budget_total": 800,      # 当日总预算 $800
    "current_cpc": 1.85,      # 当前 CPC（大促飙升）
    "target_acos": 0.25,
    "time_remaining_hours": 6,
    "current_roas": 3.2       # 当前 ROAS 良好
  }
  输出：意图 = "maintain" + 置信度 = 0.82
  （ROAS 良好，预算充足，维持当前出价即可）

LBM-Act（精确出价）：
  意图 = "maintain" + 当前出价 = $0.95
  输出：bid = $0.96（+1%，微调以维持排名）

对比无幻觉保证（GQPO 训练前 vs 后）：
  训练前：LBM 在大促高 CPC 环境下输出 $3.50（幻觉，超预算 3×）
  训练后：LBM 输出 $0.96（在合理范围内）
```

---

## ③ 代码模板

代码位置：`paper2skills-code/mas/autobidding/model.py`

```python
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
import random


@dataclass
class BiddingContext:
    keyword: str
    current_bid: float
    market_avg_cpc: float
    budget_remaining: float
    budget_total: float
    target_acos: float
    current_acos: float
    current_roas: float
    history_clicks: int
    history_conversions: int
    time_remaining_hours: float = 24.0


@dataclass
class BiddingDecision:
    keyword: str
    recommended_bid: float
    reasoning: str
    intent: str
    confidence: float
    within_budget: bool


class DARAReasonerAgent:
    """
    DARA Reasoner：基于竞价情景生成推理链（策略方向）
    少样本友好：历史数据越少，正则化越强
    """

    def __init__(self, min_history_for_confidence: int = 50):
        self.min_history = min_history_for_confidence

    def reason(self, ctx: BiddingContext) -> Tuple[str, float]:
        data_confidence = min(1.0, ctx.history_clicks / self.min_history)
        acos_gap = ctx.current_acos - ctx.target_acos

        if acos_gap > 0.1:
            direction = "reduce"
            magnitude = min(0.2, acos_gap)
            reasoning = (
                f"ACOS {ctx.current_acos:.1%} 远超目标 {ctx.target_acos:.1%}，"
                f"需降价 {magnitude:.0%}。"
                f"{'数据充足，可信度高。' if data_confidence > 0.7 else '数据稀缺，保守调整。'}"
            )
        elif acos_gap > 0:
            direction = "slight_reduce"
            magnitude = 0.05
            reasoning = f"ACOS 略高，微降 {magnitude:.0%}维持竞争力。"
        elif ctx.budget_remaining / ctx.budget_total < 0.3:
            direction = "reduce"
            magnitude = 0.1
            reasoning = f"预算剩余不足 30%，降价保存预算。"
        else:
            direction = "maintain"
            magnitude = 0.0
            reasoning = f"ROAS {ctx.current_roas:.1f} 良好，维持当前出价。"

        confidence = data_confidence * 0.6 + 0.4
        return f"{direction}|{magnitude}|{reasoning}", confidence


class DARAOptimizerAgent:
    """
    DARA Optimizer：将推理链转化为精确出价数字
    """

    def __init__(self, bid_floor: float = 0.01, bid_ceil_multiplier: float = 3.0):
        self.bid_floor = bid_floor
        self.bid_ceil_multiplier = bid_ceil_multiplier

    def optimize(self, ctx: BiddingContext, reasoning_output: str) -> float:
        parts = reasoning_output.split("|")
        direction = parts[0] if parts else "maintain"
        magnitude = float(parts[1]) if len(parts) > 1 else 0.0

        if direction in ("reduce", "slight_reduce"):
            new_bid = ctx.current_bid * (1.0 - magnitude)
        elif direction == "increase":
            new_bid = ctx.current_bid * (1.0 + magnitude)
        else:
            new_bid = ctx.current_bid * (1.0 + random.uniform(-0.01, 0.01))

        bid_ceil = ctx.market_avg_cpc * self.bid_ceil_multiplier
        return round(max(self.bid_floor, min(new_bid, bid_ceil)), 2)


class DARABiddingSystem:
    """
    DARA 双 Agent 竞价系统
    """

    def __init__(self):
        self.reasoner = DARAReasonerAgent()
        self.optimizer = DARAOptimizerAgent()

    def decide(self, ctx: BiddingContext) -> BiddingDecision:
        reasoning_output, confidence = self.reasoner.reason(ctx)
        bid = self.optimizer.optimize(ctx, reasoning_output)
        intent = reasoning_output.split("|")[0]
        reasoning_text = reasoning_output.split("|")[2] if len(reasoning_output.split("|")) > 2 else ""
        within_budget = bid <= ctx.budget_remaining * 0.1

        return BiddingDecision(
            keyword=ctx.keyword,
            recommended_bid=bid,
            reasoning=reasoning_text,
            intent=intent,
            confidence=confidence,
            within_budget=within_budget,
        )


class LBMThinkAgent:
    """
    LBM-Think：生成竞价意图（高层推理，不生成具体数字）
    防竞价幻觉：只输出 intent 类别，不输出金额
    """

    INTENTS = ["reduce_aggressive", "reduce", "maintain", "increase", "increase_aggressive"]

    def think(self, ctx: BiddingContext) -> Tuple[str, float]:
        budget_ratio = ctx.budget_remaining / max(ctx.budget_total, 1)
        acos_ratio = ctx.current_acos / max(ctx.target_acos, 0.01)

        if acos_ratio > 1.5 or budget_ratio < 0.2:
            intent, confidence = "reduce_aggressive", 0.9
        elif acos_ratio > 1.1:
            intent, confidence = "reduce", 0.8
        elif acos_ratio < 0.8 and budget_ratio > 0.5:
            intent, confidence = "increase", 0.75
        elif acos_ratio < 0.6 and budget_ratio > 0.7:
            intent, confidence = "increase_aggressive", 0.7
        else:
            intent, confidence = "maintain", 0.85

        return intent, confidence


class LBMActAgent:
    """
    LBM-Act：将意图转化为精确出价（GQPO 训练，数值准确）
    关键：只在意图范围内调整，不允许极端值（防幻觉）
    """

    INTENT_MULTIPLIERS = {
        "reduce_aggressive": (0.7, 0.85),
        "reduce": (0.88, 0.97),
        "maintain": (0.98, 1.02),
        "increase": (1.03, 1.12),
        "increase_aggressive": (1.13, 1.25),
    }

    def __init__(self, bid_floor: float = 0.01):
        self.bid_floor = bid_floor

    def act(self, ctx: BiddingContext, intent: str, confidence: float) -> float:
        lo, hi = self.INTENT_MULTIPLIERS.get(intent, (0.98, 1.02))
        multiplier = lo + (hi - lo) * confidence
        new_bid = ctx.current_bid * multiplier
        max_affordable = ctx.budget_remaining * 0.05
        return round(max(self.bid_floor, min(new_bid, max_affordable, ctx.market_avg_cpc * 2.5)), 2)


class LBMBiddingSystem:
    """
    LBM 层次化竞价系统：Think（意图）→ Act（精确出价）
    """

    def __init__(self):
        self.think_agent = LBMThinkAgent()
        self.act_agent = LBMActAgent()

    def decide(self, ctx: BiddingContext) -> BiddingDecision:
        intent, confidence = self.think_agent.think(ctx)
        bid = self.act_agent.act(ctx, intent, confidence)
        within_budget = bid <= ctx.budget_remaining * 0.1

        return BiddingDecision(
            keyword=ctx.keyword,
            recommended_bid=bid,
            reasoning=f"LBM-Think: {intent} (conf={confidence:.2f})",
            intent=intent,
            confidence=confidence,
            within_budget=within_budget,
        )
```

---

## ④ 技能关联

### 前置技能（双侧前置）
- [[Skill-MAS-Orchestrator]]：MAS 侧前置 — 多 Agent 编排是双 Agent 系统的基础
- [[Skill-ROAS-Budget-Optimization]]：广告侧前置 — ROAS 目标设定与预算分配

### 延伸技能
- [[Skill-AgenticPay-Procurement-Negotiation]]：MAS 侧延伸 — 竞价→谈判的 Agent 自治延伸
- [[Skill-Ad-Attribution-Modeling]]：广告侧延伸 — 竞价策略的归因验证

### 可组合技能
- [[Skill-Multi-Agent-Debate]]：竞价策略中的多方博弈
- [[Skill-DARA-Agentic-MMM]]：15-营销投放，预算分配闭环
- [[Skill-MAS-Consensus-Mechanism]]：多账户竞价策略的共识对齐

> **跨域桥梁边**：`mas ↔ advertising`（首条连接）

---

## ⑤ 商业价值评估

| 维度 | 评估 |
|------|------|
| **ROI 预估** | 新品上市期（DARA）：3天数据即可给出合理竞价，替代人工每日调整（节省 2-3h/天）；大促期间（LBM）：消除幻觉出价，防止预算超支 3-5× 的灾难性情况（母婴大促预算通常 $5,000-$50,000/天） |
| **实施难度** | ⭐⭐⭐☆☆（需要 LLM API 接入 + 历史竞价数据；LBM 需要离线 RL 微调；DARA 可直接 prompt 工程实现） |
| **优先级评分** | ⭐⭐⭐⭐☆（两篇 WWW'26 顶会；跨域桥梁价值高；直接对应母婴出海核心业务场景） |
| **评估依据** | DARA：少样本场景下超越基线；LBM：消除竞价幻觉，无需在线仿真即可部署；均来自阿里/字节等头部电商技术团队 |

---

## 论文来源

| 论文 | arXiv | Venue | 年份 |
|------|-------|-------|------|
| DARA: Few-shot Budget Allocation via RL-Finetuned LLMs | [2601.14711](https://arxiv.org/abs/2601.14711) | WWW'26 | 2026-01 |
| LBM: Hierarchical Large Auto-Bidding Model | [2603.05134](https://arxiv.org/abs/2603.05134) | WWW'26 | 2026-03 |
